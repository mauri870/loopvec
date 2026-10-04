# loopvec

`loopvec` analyzes Go source packages and rewrites element-wise loops to use Go's
experimental [simd](https://pkg.go.dev/simd) package for **automatic SIMD
vectorization**. Rewritten code requires `GOEXPERIMENT=simd` (Go 1.27+).

Go blog post: https://go.dev/blog/simd-experiment

> NOTE:  SIMD support in Go is experimental, so there are several bugs lurking around, both in the Go compiler/runtime and this tool.
> The bugs that affect loopvec and/or Go are in [BUGS.md](BUGS.md).
> For now, prioritize testing with [`gotip`](https://pkg.go.dev/golang.org/dl/gotip).

## What it detects

The tool recognizes these loop shapes and rewrites them to use portable SIMD
operations that lower to **AVX-512/AVX2/NEON** depending on the target CPU:

<details>
<summary>Patterns</summary>

| Pattern | Emitted operation |
|---|---|
| `for i := range dst { dst[i] = a[i] + b[i] }` | element-wise binary op |
| `for i := range dst { dst[i] += src[i] }` | in-place binary op |
| `for i := range dst { dst[i] *= scalar }` | scalar broadcast op |
| `for i := range dst { dst[i] = 7 }` | fill (broadcast non-zero literal) |
| `for i := range dst { dst[i] = src[i] }` | `copy(dst, src)` (no simd needed) |
| `for i, v := range src { dst[i] = v * f }` | two-variable range scalar op |
| `for i := range dst { dst[i] = a[i]*alpha + b[i] }` | `MulAdd` (FMA) |
| `for i := range dst { dst[i] = a[i]*alpha + b[i]*beta }` | `MulAdd` + `Mul` (two-scalar axpy) |
| `for i := range dst { dst[i] = (a[i] - mean) * inv }` | shift and scale (z-score normalization) |
| `for i := range dst { dst[i] += a[i]*alpha }` | in-place `MulAdd` (DAXPY) |
| `for i := range dst { dst[i] = -src[i] }` | `Neg` (unary negation) |
| `for i := range dst { dst[i] = ^src[i] }` | `Not` (unary bitwise NOT) |
| `for i := range a { a[i] /= norm; d[i] -= a[i] }` | several assignments in one vector loop |
| `for i := range a { t := b[i] + c[i]*d[i]; a[i] = t * t }` | temporary held as a vector |
| `for i := range dst { dst[i] = a[i]*b[i] + c[i]*k + (k*m - a[i]) }` | expressions of any depth |
| `for i := range dst { dst[i] = k }` | fill from a variable or constant |
| `for i := range a { sum += a[i] * b[i] }` | integer reduction (`+ * & \| ^ min max`); float sum and product with `-fp-reassoc` |
| `for i := 0; i < len(s); i++ { ... }` | three-clause for (all body shapes above) |
| `for i := len(s) - 1; i >= 0; i-- { ... }` | reverse three-clause for (rewritten to run forward) |
| `for i := 0; i < n; i++ { ... }`, `for i := range n { ... }` | explicit int limit; slices are length-checked first |
| `for i := 0; i < 16; i++ { ... }` | constant limit; a constant trip count under 8 stays scalar (under 4 full vectors for a reduction) |
| `for i := 1; i < len(s)-1; i++ { ... }`, `for i := lo; i < hi; i++ { ... }` | any start and limit that do not change in the loop |
| `for i := range x { if x[i] < 0 { x[i] = 0 } }`, `if c { dst[i] = u } else if d { dst[i] = v } else { dst[i] = w }` | compare, then select (`IfElse`): ReLU, leaky ReLU, ReLU6, hard tanh, clamp, sign, step |
| `for i := range v.Values { v.Values[i] *= a }`, `dst.f[i] += src.f[i]` | a slice reached through a field (`x.f`, `x.f.g`); the root must not be assigned in the loop, and the field is copied into a local once |
| `for i := range dst { dst[i] = a[i+1] - a[i] }`, `work[j] += a[row*lda+j]` | read at an offset from the index (stencil, row of a 2-D slice); a slice the loop also writes is read only at a constant positive offset (`a[i] = a[i+1] + b[i]`) |

Supported element types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
`uint32`, `uint64`, `float32`, `float64`.
Slices of named types (`type Values []float64`) work; named element types (`[]Celsius`) do not.

Supported binary operators: `+`, `-`, `*`, `/`, `&`, `|`, `^`, `&^` (and their `op=` forms). `/` requires `float32` or `float64` (no integer division in simd).
Supported unary operators: `-` (negation, all except unsigned integers), `^` (bitwise NOT, all integer types).
Also `math.Abs` and `math.Sqrt` (a `float32` loop writes `float32(math.Sqrt(float64(x)))`), `min` and `max` of integers, and `<<` and `>>` by a constant or an unsigned variable.
`MulAdd`/FMA patterns require `float32` or `float64`.
Note: `*` is not supported for `int64` and `uint64` (no SIMD multiply for 64-bit integers).
Zero fills (`dst[i] = 0`) are left alone: the compiler already turns them into `memclr`,
which is faster than a vector loop.

**Copy loops** become `copy(dst, src)`. The compiler does this for zero fills but
not for copies, and `copy` is 4-10x faster from 64 elements up. Overlapping
operands keep the original loop, and a source shorter than the destination still
panics before anything is written.

**Overlapping slices.** `dst[i] = a[i] + b[i]` is safe to vectorize when `dst`
and `a` are the exact same slice, but not when they partially overlap (e.g.
`dst = s[1:]`, `a = s[:len(s)-1]`): the scalar loop then carries values
forward one element at a time, while a vector loop reads a whole chunk before
writing any of it. This also applies to the reverse-loop pattern above —
`for i := len(s)-1; i >= 0; i--` is often written specifically because the
slices overlap, and rewriting it to run forward would change the result in
exactly that case. loopvec guards against this: every rewritten loop with
more than one distinct slice operand is wrapped in a runtime check
(`_loopvecOverlap`, comparing address ranges) that falls back to
the original scalar loop whenever the operands' memory overlaps, regardless
of loop direction. It does not tell an exact alias from an offset one, so
`Add(x, x, y)` also takes the scalar loop. For a loop over part of its slices it
compares only the elements the loop touches. A slice that is read ahead of the
index and written (`a[i] = a[i+1] + b[i]`) needs no check: the vector loop loads
the chunk before storing it, as the scalar loop reads the later element first.
Reading what an earlier iteration wrote (`a[i] = a[i-1] + b[i]`) is never
rewritten.

**FMA.** `dst[i] = a[i]*alpha + b[i]` (and the DAXPY/two-scalar-axpy
patterns) lower to a single fused multiply-add instruction, not a separate
multiply and add. The Go spec permits this, but the result can differ from
the scalar loop in the last bit of the float, so scalar and simd builds are
not guaranteed to be bitwise identical for these patterns. This isn't
currently opt-in. An explicit conversion (`float64(a[i])*alpha + b[i]`, which
the spec says forbids fusion) isn't recognized as this pattern in the first
place, so it's never fused — it just doesn't get vectorized.

</details>

## Performance

Synthetic benchmarks from [bench/](bench/) on an AMD Ryzen 9 9950X3D (AVX-512) show **~4.6x faster**,
ranging from 2.3x to 31x depending on the kernel

<details>
<summary>bench.txt</summary>

```
goos: linux
goarch: amd64
pkg: github.com/mauri870/loopvec/bench
cpu: AMD Ryzen 9 9950X3D 16-Core Processor          
                                │ /tmp/bench_scalar.txt │         /tmp/bench_simd.txt          │
                                │        sec/op         │    sec/op     vs base                │
AddFloat32s/64-32                         12.515n ±  2%   7.062n ±  2%  -43.57% (p=0.000 n=10)
AddFloat32s/4096-32                        815.0n ±  2%   191.4n ±  3%  -76.52% (p=0.000 n=10)
AddFloat32s/1048576-32                    213.90µ ±  1%   87.46µ ±  5%  -59.11% (p=0.000 n=10)
MulFloat32s/64-32                         12.780n ±  1%   6.520n ±  2%  -48.98% (p=0.000 n=10)
MulFloat32s/4096-32                        822.0n ±  2%   187.7n ±  1%  -77.17% (p=0.000 n=10)
MulFloat32s/1048576-32                    218.74µ ±  1%   84.26µ ±  5%  -61.48% (p=0.000 n=10)
ScalFloat32s/64-32                        10.770n ±  8%   3.585n ±  1%  -66.71% (p=0.000 n=10)
ScalFloat32s/4096-32                       745.0n ±  1%   145.1n ±  2%  -80.52% (p=0.000 n=10)
ScalFloat32s/1048576-32                   187.92µ ±  1%   37.15µ ±  3%  -80.23% (p=0.000 n=10)
MixFloat32s/64-32                         23.510n ±  1%   7.741n ±  2%  -67.07% (p=0.000 n=10)
MixFloat32s/4096-32                       1469.0n ±  1%   219.8n ±  5%  -85.04% (p=0.000 n=10)
MixFloat32s/1048576-32                    388.50µ ±  1%   87.18µ ±  5%  -77.56% (p=0.000 n=10)
AxpyFloat32s/64-32                        18.805n ±  4%   6.574n ±  1%  -65.04% (p=0.000 n=10)
AxpyFloat32s/4096-32                       799.1n ±  1%   196.4n ±  2%  -75.43% (p=0.000 n=10)
AxpyFloat32s/1048576-32                   209.79µ ±  3%   83.21µ ±  8%  -60.34% (p=0.000 n=10)
NegFloat32s/64-32                         14.380n ± 15%   5.058n ±  2%  -64.82% (p=0.000 n=10)
NegFloat32s/4096-32                        879.1n ± 15%   172.8n ±  4%  -80.35% (p=0.000 n=10)
NegFloat32s/1048576-32                    217.79µ ± 14%   59.55µ ±  3%  -72.66% (p=0.000 n=10)
DivFloat32s/64-32                         28.045n ±  3%   7.175n ±  3%  -74.42% (p=0.000 n=10)
DivFloat32s/4096-32                       1804.0n ±  2%   191.2n ±  4%  -89.40% (p=0.000 n=10)
DivFloat32s/1048576-32                    473.42µ ±  2%   85.20µ ±  8%  -82.00% (p=0.000 n=10)
DaxpyFloat32s/64-32                       14.035n ±  4%   5.596n ±  2%  -60.13% (p=0.000 n=10)
DaxpyFloat32s/4096-32                      910.6n ±  6%   176.9n ±  2%  -80.58% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                  225.55µ ±  5%   63.97µ ±  2%  -71.64% (p=0.000 n=10)
FillFloat32s/64-32                         8.000n ± 27%   3.014n ±  9%  -62.32% (p=0.000 n=10)
FillFloat32s/4096-32                       722.6n ±  1%   104.5n ± 10%  -85.55% (p=0.000 n=10)
FillFloat32s/1048576-32                   188.78µ ±  2%   29.96µ ±  1%  -84.13% (p=0.000 n=10)
FillUint8s/64-32                           8.042n ± 48%   2.113n ±  5%  -73.73% (p=0.000 n=10)
FillUint8s/4096-32                        732.60n ±  3%   25.45n ±  4%  -96.53% (p=0.000 n=10)
FillUint8s/1048576-32                    186.545µ ±  2%   6.011µ ±  1%  -96.78% (p=0.000 n=10)
ReverseIncFloat32s/64-32                  12.720n ±  8%   5.171n ±  2%  -59.34% (p=0.000 n=10)
ReverseIncFloat32s/4096-32                 743.5n ±  2%   159.3n ±  9%  -78.57% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32             199.44µ ±  2%   59.78µ ±  3%  -70.03% (p=0.000 n=10)
CopyFloat32s/64-32                        11.775n ±  2%   4.069n ±  1%  -65.44% (p=0.000 n=10)
CopyFloat32s/4096-32                      755.10n ±  3%   73.41n ±  3%  -90.28% (p=0.000 n=10)
CopyFloat32s/1048576-32                   190.45µ ±  2%   56.30µ ±  5%  -70.44% (p=0.000 n=10)
AndNotUint64s/64-32                       18.400n ±  8%   9.796n ±  2%  -46.76% (p=0.000 n=10)
AndNotUint64s/4096-32                     1220.5n ±  2%   374.3n ±  5%  -69.34% (p=0.000 n=10)
AndNotUint64s/1048576-32                   308.5µ ±  6%   167.3µ ±  7%  -45.76% (p=0.000 n=10)
NormalizeFloat32s/64-32                   28.260n ±  2%   6.424n ±  2%  -77.27% (p=0.000 n=10)
NormalizeFloat32s/4096-32                 1823.0n ±  1%   226.2n ±  2%  -87.59% (p=0.000 n=10)
NormalizeFloat32s/1048576-32              471.48µ ±  2%   67.76µ ±  3%  -85.63% (p=0.000 n=10)
ChainInt32s/64-32                          30.05n ± 16%   12.72n ±  2%  -57.69% (p=0.000 n=10)
ChainInt32s/4096-32                       2111.5n ± 12%   328.4n ±  5%  -84.44% (p=0.000 n=10)
ChainInt32s/1048576-32                     480.8µ ±  3%   131.3µ ±  7%  -72.69% (p=0.000 n=10)
WideInt32s/64-32                          46.380n ±  3%   9.552n ±  3%  -79.40% (p=0.000 n=10)
WideInt32s/4096-32                        2979.5n ±  3%   271.3n ±  3%  -90.89% (p=0.000 n=10)
WideInt32s/1048576-32                      755.3µ ±  3%   122.8µ ±  9%  -83.74% (p=0.000 n=10)
SumInt32s/64-32                           12.210n ±  3%   7.572n ±  2%  -37.99% (p=0.000 n=10)
SumInt32s/4096-32                          776.9n ±  3%   195.2n ±  2%  -74.87% (p=0.000 n=10)
SumInt32s/1048576-32                      189.66µ ±  3%   47.73µ ±  2%  -74.84% (p=0.000 n=10)
DotInt32s/64-32                           14.230n ±  2%   9.237n ±  4%  -35.08% (p=0.000 n=10)
DotInt32s/4096-32                          938.7n ±  1%   204.3n ±  2%  -78.24% (p=0.000 n=10)
DotInt32s/1048576-32                      245.57µ ±  3%   63.41µ ±  3%  -74.18% (p=0.000 n=10)
SumFloat32s/64-32                          10.87n ±  3%   12.96n ±  1%  +19.33% (p=0.000 n=10)
SumFloat32s/4096-32                       1434.0n ±  2%   112.3n ±  3%  -92.17% (p=0.000 n=10)
SumFloat32s/1048576-32                    372.32µ ±  2%   31.80µ ±  4%  -91.46% (p=0.000 n=10)
DotFloat32s/64-32                          16.14n ±  2%   13.44n ±  3%  -16.75% (p=0.000 n=10)
DotFloat32s/4096-32                       1465.0n ±  3%   183.3n ±  2%  -87.49% (p=0.000 n=10)
DotFloat32s/1048576-32                    380.65µ ±  3%   62.77µ ±  4%  -83.51% (p=0.000 n=10)
DotFloat64s/64-32                          11.99n ±  3%   12.19n ±  2%   +1.71% (p=0.003 n=10)
DotFloat64s/4096-32                       1415.0n ±  3%   362.3n ±  5%  -74.40% (p=0.000 n=10)
DotFloat64s/1048576-32                     389.5µ ±  3%   120.3µ ± 10%  -69.12% (p=0.000 n=10)
AbsFloat32s/64-32                         24.270n ±  4%   5.244n ±  3%  -78.40% (p=0.000 n=10)
AbsFloat32s/4096-32                       1488.0n ±  2%   165.9n ±  7%  -88.85% (p=0.000 n=10)
AbsFloat32s/1048576-32                    380.94µ ±  2%   58.89µ ±  2%  -84.54% (p=0.000 n=10)
SqrtFloat32s/64-32                        51.925n ±  0%   5.421n ±  3%  -89.56% (p=0.000 n=10)
SqrtFloat32s/4096-32                      3320.0n ±  3%   225.2n ±  2%  -93.22% (p=0.000 n=10)
SqrtFloat32s/1048576-32                   839.65µ ±  2%   61.38µ ±  3%  -92.69% (p=0.000 n=10)
ClampInt32s/64-32                         28.770n ±  2%   4.744n ±  1%  -83.51% (p=0.000 n=10)
ClampInt32s/4096-32                       1863.0n ±  2%   147.7n ±  2%  -92.07% (p=0.000 n=10)
ClampInt32s/1048576-32                    478.94µ ±  3%   56.62µ ±  3%  -88.18% (p=0.000 n=10)
ShrInt32s/64-32                           12.970n ±  2%   4.771n ±  2%  -63.22% (p=0.000 n=10)
ShrInt32s/4096-32                          846.6n ±  6%   159.6n ±  2%  -81.16% (p=0.000 n=10)
ShrInt32s/1048576-32                      214.58µ ±  2%   59.75µ ±  5%  -72.16% (p=0.000 n=10)
ShlUint32s/64-32                          12.895n ±  2%   5.060n ±  2%  -60.76% (p=0.000 n=10)
ShlUint32s/4096-32                         833.7n ±  2%   148.9n ±  3%  -82.14% (p=0.000 n=10)
ShlUint32s/1048576-32                     216.24µ ±  1%   57.42µ ±  4%  -73.44% (p=0.000 n=10)
DiffInt32s/64-32                          14.385n ±  6%   8.574n ±  3%  -40.40% (p=0.000 n=10)
DiffInt32s/4096-32                         879.1n ±  3%   426.3n ±  3%  -51.50% (p=0.000 n=10)
DiffInt32s/1048576-32                      228.0µ ±  2%   106.2µ ±  2%  -53.41% (p=0.000 n=10)
SmoothInt32s/64-32                        22.070n ±  2%   9.388n ±  2%  -57.46% (p=0.000 n=10)
SmoothInt32s/4096-32                      1782.0n ±  2%   499.9n ±  7%  -71.95% (p=0.000 n=10)
SmoothInt32s/1048576-32                    487.1µ ±  3%   118.7µ ±  2%  -75.62% (p=0.000 n=10)
ShiftInt32s/64-32                         16.070n ±  4%   5.221n ±  1%  -67.51% (p=0.000 n=10)
ShiftInt32s/4096-32                       1097.0n ±  2%   194.7n ±  3%  -82.26% (p=0.000 n=10)
ShiftInt32s/1048576-32                    277.69µ ±  2%   62.95µ ±  6%  -77.33% (p=0.000 n=10)
AccumRowInt32s/64-32                      16.500n ±  3%   6.244n ±  2%  -62.15% (p=0.000 n=10)
AccumRowInt32s/4096-32                    1043.5n ±  4%   243.0n ±  2%  -76.72% (p=0.000 n=10)
AccumRowInt32s/1048576-32                 268.47µ ±  3%   64.43µ ±  4%  -76.00% (p=0.000 n=10)
MinUint16s/64-32                           24.74n ±  2%   29.80n ±  1%  +20.43% (p=0.000 n=10)
MinUint16s/4096-32                       2180.00n ±  2%   67.06n ±  5%  -96.92% (p=0.000 n=10)
MinUint16s/1048576-32                     565.25µ ±  2%   15.50µ ±  3%  -97.26% (p=0.000 n=10)
ProductFloat32s/64-32                     21.210n ±  3%   7.885n ±  3%  -62.82% (p=0.000 n=10)
ProductFloat32s/4096-32                   2186.0n ±  1%   105.9n ±  3%  -95.16% (p=0.000 n=10)
ProductFloat32s/1048576-32                554.57µ ±  2%   31.78µ ±  2%  -94.27% (p=0.000 n=10)
ShiftAddInt32s/64-32                      14.025n ± 14%   7.560n ±  2%  -46.09% (p=0.000 n=10)
ShiftAddInt32s/4096-32                     880.1n ±  5%   268.8n ±  2%  -69.46% (p=0.000 n=10)
ShiftAddInt32s/1048576-32                 220.90µ ±  3%   74.11µ ±  3%  -66.45% (p=0.000 n=10)
ReadAfterStoreInt32s/64-32                 31.75n ±  2%   18.81n ±  1%  -40.75% (p=0.000 n=10)
ReadAfterStoreInt32s/4096-32              2367.0n ±  2%   586.3n ±  6%  -75.23% (p=0.000 n=10)
ReadAfterStoreInt32s/1048576-32            658.4µ ±  3%   157.8µ ±  3%  -76.04% (p=0.000 n=10)
ReLUFloat32s/64-32                        12.580n ±  6%   3.890n ± 17%  -69.08% (p=0.000 n=10)
ReLUFloat32s/4096-32                       795.6n ±  7%   147.7n ±  2%  -81.44% (p=0.000 n=10)
ReLUFloat32s/1048576-32                   190.65µ ±  8%   38.07µ ±  4%  -80.03% (p=0.000 n=10)
LeakyReLUFloat32s/64-32                   23.315n ±  3%   5.659n ±  2%  -75.73% (p=0.000 n=10)
LeakyReLUFloat32s/4096-32                 1428.0n ±  7%   169.5n ±  9%  -88.13% (p=0.000 n=10)
LeakyReLUFloat32s/1048576-32              375.36µ ±  2%   58.41µ ±  2%  -84.44% (p=0.000 n=10)
ReLU6Float32s/64-32                       22.765n ±  2%   3.869n ±  3%  -83.01% (p=0.000 n=10)
ReLU6Float32s/4096-32                     1467.0n ±  2%   164.4n ±  2%  -88.79% (p=0.000 n=10)
ReLU6Float32s/1048576-32                  380.37µ ±  2%   41.36µ ±  5%  -89.13% (p=0.000 n=10)
SignFloat32s/64-32                        23.235n ±  1%   5.335n ±  1%  -77.04% (p=0.000 n=10)
SignFloat32s/4096-32                      1490.5n ±  3%   184.3n ±  4%  -87.63% (p=0.000 n=10)
SignFloat32s/1048576-32                   379.07µ ±  4%   58.65µ ±  4%  -84.53% (p=0.000 n=10)
geomean                                    1.892µ         422.1n        -77.69%

                                │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                                │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                         57.15Gi ±  2%    101.28Gi ±  2%    +77.21% (p=0.000 n=10)
AddFloat32s/4096-32                       56.17Gi ±  2%    239.20Gi ±  3%   +325.89% (p=0.000 n=10)
AddFloat32s/1048576-32                    54.79Gi ±  1%    133.99Gi ±  5%   +144.57% (p=0.000 n=10)
MulFloat32s/64-32                         55.95Gi ±  1%    109.70Gi ±  2%    +96.05% (p=0.000 n=10)
MulFloat32s/4096-32                       55.69Gi ±  2%    243.83Gi ±  1%   +337.85% (p=0.000 n=10)
MulFloat32s/1048576-32                    53.57Gi ±  1%    139.09Gi ±  5%   +159.62% (p=0.000 n=10)
ScalFloat32s/64-32                        44.27Gi ±  8%    133.01Gi ±  1%   +200.43% (p=0.000 n=10)
ScalFloat32s/4096-32                      40.96Gi ±  1%    210.22Gi ±  2%   +413.22% (p=0.000 n=10)
ScalFloat32s/1048576-32                   41.57Gi ±  1%    210.28Gi ±  3%   +405.80% (p=0.000 n=10)
MixFloat32s/64-32                         30.42Gi ±  1%     92.40Gi ±  2%   +203.72% (p=0.000 n=10)
MixFloat32s/4096-32                       31.16Gi ±  1%    208.25Gi ±  5%   +568.35% (p=0.000 n=10)
MixFloat32s/1048576-32                    30.16Gi ±  1%    134.44Gi ±  5%   +345.68% (p=0.000 n=10)
AxpyFloat32s/64-32                        38.04Gi ±  4%    108.80Gi ±  1%   +186.01% (p=0.000 n=10)
AxpyFloat32s/4096-32                      57.29Gi ±  1%    233.12Gi ±  2%   +306.93% (p=0.000 n=10)
AxpyFloat32s/1048576-32                   55.86Gi ±  3%    140.84Gi ±  7%   +152.12% (p=0.000 n=10)
NegFloat32s/64-32                         33.17Gi ± 15%     94.26Gi ±  2%   +184.18% (p=0.000 n=10)
NegFloat32s/4096-32                       34.72Gi ± 16%    176.67Gi ±  4%   +408.90% (p=0.000 n=10)
NegFloat32s/1048576-32                    35.87Gi ± 12%    131.19Gi ±  3%   +265.71% (p=0.000 n=10)
DivFloat32s/64-32                         25.50Gi ±  3%     99.69Gi ±  3%   +290.88% (p=0.000 n=10)
DivFloat32s/4096-32                       25.37Gi ±  2%    239.53Gi ±  4%   +843.98% (p=0.000 n=10)
DivFloat32s/1048576-32                    24.75Gi ±  2%    137.70Gi ±  8%   +456.28% (p=0.000 n=10)
DaxpyFloat32s/64-32                       50.96Gi ±  4%    127.83Gi ±  2%   +150.83% (p=0.000 n=10)
DaxpyFloat32s/4096-32                     50.27Gi ±  6%    258.83Gi ±  2%   +414.88% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                  51.96Gi ±  5%    183.20Gi ±  2%   +252.58% (p=0.000 n=10)
FillFloat32s/64-32                        29.80Gi ± 21%     79.10Gi ±  8%   +165.41% (p=0.000 n=10)
FillFloat32s/4096-32                      21.12Gi ±  1%    146.32Gi ±  9%   +592.91% (p=0.000 n=10)
FillFloat32s/1048576-32                   20.69Gi ±  2%    130.37Gi ±  1%   +530.07% (p=0.000 n=10)
FillUint8s/64-32                          7.412Gi ± 32%    28.210Gi ±  6%   +280.62% (p=0.000 n=10)
FillUint8s/4096-32                        5.208Gi ±  3%   149.892Gi ±  4%  +2777.92% (p=0.000 n=10)
FillUint8s/1048576-32                     5.236Gi ±  2%   162.450Gi ±  1%  +3002.78% (p=0.000 n=10)
ReverseIncFloat32s/64-32                  37.48Gi ±  7%     92.20Gi ±  2%   +145.98% (p=0.000 n=10)
ReverseIncFloat32s/4096-32                41.05Gi ±  2%    191.51Gi ±  8%   +366.53% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32             39.17Gi ±  2%    130.69Gi ±  3%   +233.64% (p=0.000 n=10)
CopyFloat32s/64-32                        40.51Gi ±  2%    117.19Gi ±  1%   +189.30% (p=0.000 n=10)
CopyFloat32s/4096-32                      40.42Gi ±  3%    415.74Gi ±  3%   +928.68% (p=0.000 n=10)
CopyFloat32s/1048576-32                   41.02Gi ±  2%    138.78Gi ±  5%   +238.31% (p=0.000 n=10)
AndNotUint64s/64-32                       77.75Gi ±  7%    146.02Gi ±  2%    +87.82% (p=0.000 n=10)
AndNotUint64s/4096-32                     75.02Gi ±  2%    244.67Gi ±  5%   +226.12% (p=0.000 n=10)
AndNotUint64s/1048576-32                  75.98Gi ±  5%    140.07Gi ±  6%    +84.36% (p=0.000 n=10)
NormalizeFloat32s/64-32                   33.75Gi ±  2%    148.47Gi ±  2%   +339.89% (p=0.000 n=10)
NormalizeFloat32s/4096-32                 33.48Gi ±  1%    269.83Gi ±  2%   +705.84% (p=0.000 n=10)
NormalizeFloat32s/1048576-32              33.14Gi ±  2%    230.60Gi ±  3%   +595.83% (p=0.000 n=10)
ChainInt32s/64-32                         47.61Gi ± 14%    112.50Gi ±  2%   +136.30% (p=0.000 n=10)
ChainInt32s/4096-32                       43.59Gi ± 13%    278.72Gi ±  5%   +539.47% (p=0.000 n=10)
ChainInt32s/1048576-32                    48.75Gi ±  3%    178.51Gi ±  7%   +266.20% (p=0.000 n=10)
WideInt32s/64-32                          20.56Gi ±  3%     99.84Gi ±  3%   +385.53% (p=0.000 n=10)
WideInt32s/4096-32                        20.48Gi ±  3%    224.92Gi ±  3%   +997.99% (p=0.000 n=10)
WideInt32s/1048576-32                     20.69Gi ±  3%    127.23Gi ± 10%   +515.03% (p=0.000 n=10)
SumInt32s/64-32                           19.53Gi ±  3%     31.49Gi ±  2%    +61.24% (p=0.000 n=10)
SumInt32s/4096-32                         19.64Gi ±  3%     78.14Gi ±  2%   +297.87% (p=0.000 n=10)
SumInt32s/1048576-32                      20.60Gi ±  3%     81.85Gi ±  2%   +297.40% (p=0.000 n=10)
DotInt32s/64-32                           33.51Gi ±  2%     51.62Gi ±  4%    +54.07% (p=0.000 n=10)
DotInt32s/4096-32                         32.51Gi ±  1%    149.37Gi ±  2%   +359.43% (p=0.000 n=10)
DotInt32s/1048576-32                      31.81Gi ±  3%    123.20Gi ±  4%   +287.26% (p=0.000 n=10)
SumFloat32s/64-32                         21.95Gi ±  3%     18.38Gi ±  1%    -16.23% (p=0.000 n=10)
SumFloat32s/4096-32                       10.64Gi ±  2%    135.90Gi ±  3%  +1177.39% (p=0.000 n=10)
SumFloat32s/1048576-32                    10.49Gi ±  2%    122.83Gi ±  4%  +1070.54% (p=0.000 n=10)
DotFloat32s/64-32                         29.53Gi ±  2%     35.48Gi ±  3%    +20.14% (p=0.000 n=10)
DotFloat32s/4096-32                       20.83Gi ±  3%    166.51Gi ±  2%   +699.34% (p=0.000 n=10)
DotFloat32s/1048576-32                    20.52Gi ±  3%    124.47Gi ±  4%   +506.48% (p=0.000 n=10)
DotFloat64s/64-32                         79.58Gi ±  3%     78.25Gi ±  2%     -1.67% (p=0.003 n=10)
DotFloat64s/4096-32                       43.14Gi ±  3%    168.47Gi ±  4%   +290.57% (p=0.000 n=10)
DotFloat64s/1048576-32                    40.11Gi ±  3%    129.89Gi ±  9%   +223.82% (p=0.000 n=10)
AbsFloat32s/64-32                         19.65Gi ±  5%     90.94Gi ±  3%   +362.90% (p=0.000 n=10)
AbsFloat32s/4096-32                       20.51Gi ±  2%    183.97Gi ±  7%   +796.94% (p=0.000 n=10)
AbsFloat32s/1048576-32                    20.51Gi ±  2%    132.66Gi ±  2%   +546.87% (p=0.000 n=10)
SqrtFloat32s/64-32                        9.184Gi ±  0%    87.961Gi ±  3%   +857.79% (p=0.000 n=10)
SqrtFloat32s/4096-32                      9.192Gi ±  3%   135.526Gi ±  2%  +1374.41% (p=0.000 n=10)
SqrtFloat32s/1048576-32                   9.305Gi ±  2%   127.277Gi ±  3%  +1267.79% (p=0.000 n=10)
ClampInt32s/64-32                         16.57Gi ±  2%    100.50Gi ±  1%   +506.41% (p=0.000 n=10)
ClampInt32s/4096-32                       16.38Gi ±  2%    206.67Gi ±  2%  +1161.78% (p=0.000 n=10)
ClampInt32s/1048576-32                    16.31Gi ±  3%    137.98Gi ±  3%   +745.88% (p=0.000 n=10)
ShrInt32s/64-32                           36.78Gi ±  2%     99.95Gi ±  2%   +171.79% (p=0.000 n=10)
ShrInt32s/4096-32                         36.05Gi ±  5%    191.25Gi ±  2%   +430.58% (p=0.000 n=10)
ShrInt32s/1048576-32                      36.41Gi ±  2%    130.76Gi ±  5%   +259.15% (p=0.000 n=10)
ShlUint32s/64-32                          36.97Gi ±  2%     94.24Gi ±  2%   +154.88% (p=0.000 n=10)
ShlUint32s/4096-32                        36.61Gi ±  2%    204.91Gi ±  3%   +459.73% (p=0.000 n=10)
ShlUint32s/1048576-32                     36.13Gi ±  1%    136.06Gi ±  4%   +276.58% (p=0.000 n=10)
DiffInt32s/64-32                          33.15Gi ±  6%     55.62Gi ±  3%    +67.77% (p=0.000 n=10)
DiffInt32s/4096-32                        34.72Gi ±  3%     71.59Gi ±  3%   +106.22% (p=0.000 n=10)
DiffInt32s/1048576-32                     34.26Gi ±  2%     73.54Gi ±  2%   +114.67% (p=0.000 n=10)
SmoothInt32s/64-32                        21.61Gi ±  2%     50.79Gi ±  2%   +135.05% (p=0.000 n=10)
SmoothInt32s/4096-32                      17.12Gi ±  2%     61.20Gi ±  8%   +257.36% (p=0.000 n=10)
SmoothInt32s/1048576-32                   16.04Gi ±  3%     65.79Gi ±  2%   +310.22% (p=0.000 n=10)
ShiftInt32s/64-32                         29.67Gi ±  4%     91.32Gi ±  1%   +207.83% (p=0.000 n=10)
ShiftInt32s/4096-32                       27.82Gi ±  2%    156.78Gi ±  3%   +463.49% (p=0.000 n=10)
ShiftInt32s/1048576-32                    28.13Gi ±  2%    124.11Gi ±  6%   +341.13% (p=0.000 n=10)
AccumRowInt32s/64-32                      43.35Gi ±  3%    114.54Gi ±  2%   +164.23% (p=0.000 n=10)
AccumRowInt32s/4096-32                    43.88Gi ±  4%    188.45Gi ±  2%   +329.47% (p=0.000 n=10)
AccumRowInt32s/1048576-32                 43.65Gi ±  3%    181.90Gi ±  3%   +316.73% (p=0.000 n=10)
MinUint16s/64-32                          4.819Gi ±  2%     4.001Gi ±  1%    -16.99% (p=0.000 n=10)
MinUint16s/4096-32                        3.499Gi ±  2%   113.771Gi ±  5%  +3151.42% (p=0.000 n=10)
MinUint16s/1048576-32                     3.455Gi ±  2%   126.039Gi ±  4%  +3547.70% (p=0.000 n=10)
ProductFloat32s/64-32                     11.24Gi ±  2%     30.24Gi ±  3%   +168.99% (p=0.000 n=10)
ProductFloat32s/4096-32                   6.980Gi ±  1%   144.059Gi ±  2%  +1963.92% (p=0.000 n=10)
ProductFloat32s/1048576-32                7.044Gi ±  2%   122.924Gi ±  2%  +1645.14% (p=0.000 n=10)
ShiftAddInt32s/64-32                      51.00Gi ± 12%     94.60Gi ±  2%    +85.51% (p=0.000 n=10)
ShiftAddInt32s/4096-32                    52.01Gi ±  5%    170.31Gi ±  2%   +227.45% (p=0.000 n=10)
ShiftAddInt32s/1048576-32                 53.05Gi ±  3%    158.12Gi ±  3%   +198.05% (p=0.000 n=10)
ReadAfterStoreInt32s/64-32                22.52Gi ±  3%     38.02Gi ±  1%    +68.80% (p=0.000 n=10)
ReadAfterStoreInt32s/4096-32              19.34Gi ±  2%     78.07Gi ±  6%   +303.65% (p=0.000 n=10)
ReadAfterStoreInt32s/1048576-32           17.80Gi ±  3%     74.28Gi ±  3%   +317.36% (p=0.000 n=10)
ReLUFloat32s/64-32                        37.94Gi ±  6%    122.58Gi ± 14%   +223.11% (p=0.000 n=10)
ReLUFloat32s/4096-32                      38.37Gi ±  7%    206.60Gi ±  2%   +438.40% (p=0.000 n=10)
ReLUFloat32s/1048576-32                   40.98Gi ±  8%    205.22Gi ±  4%   +400.81% (p=0.000 n=10)
LeakyReLUFloat32s/64-32                   20.45Gi ±  3%     84.27Gi ±  3%   +312.02% (p=0.000 n=10)
LeakyReLUFloat32s/4096-32                 21.37Gi ±  7%    180.01Gi ±  8%   +742.21% (p=0.000 n=10)
LeakyReLUFloat32s/1048576-32              20.81Gi ±  2%    133.75Gi ±  2%   +542.62% (p=0.000 n=10)
ReLU6Float32s/64-32                       20.95Gi ±  2%    123.26Gi ±  3%   +488.48% (p=0.000 n=10)
ReLU6Float32s/4096-32                     20.81Gi ±  2%    185.62Gi ±  2%   +792.13% (p=0.000 n=10)
ReLU6Float32s/1048576-32                  20.54Gi ±  2%    188.91Gi ±  5%   +819.73% (p=0.000 n=10)
SignFloat32s/64-32                        20.52Gi ±  1%     89.38Gi ±  1%   +335.49% (p=0.000 n=10)
SignFloat32s/4096-32                      20.48Gi ±  3%    165.57Gi ±  4%   +708.64% (p=0.000 n=10)
SignFloat32s/1048576-32                   20.61Gi ±  4%    133.23Gi ±  4%   +546.43% (p=0.000 n=10)
geomean                                   26.77Gi           120.0Gi         +348.21%
```
</details>

Filling a byte slice with a non-zero value (`FillUint8s`, 1M elements) is over 30x
faster, +3000% throughput.

Activation functions written as plain `if`/`else` loops (ReLU, leaky ReLU, ReLU6, sign;
`float32`, 4K to 1M elements) run 5 to 9x faster.

Running `loopvec -methods` on [gorgonia/tensor](https://github.com/gorgonia/tensor)
detects 477 vectorizable loops. The result is **~2.7x faster**
on an AMD Ryzen 9 9950X3D (AVX-512), geomean over all rows. The rewritten kernels are
2.5 to 15x faster at 4K and 1M elements and 1.1 to 6.8x at 64.

<details>
<summary>bench.txt</summary>

```
                       │    scalar    │               simd               │
                       │    sec/op     │   sec/op     vs base                │
AddVSF32/64-32           12.750n ±  2%   4.629n ± 4%  -63.70% (p=0.000 n=10)
AddVSF32/4096-32          735.6n ±  2%   151.3n ± 2%  -79.43% (p=0.000 n=10)
AddVSF32/1048576-32      194.40µ ±  2%   36.28µ ± 2%  -81.34% (p=0.000 n=10)
MulVSF32/64-32            7.607n ±  3%   4.508n ± 1%  -40.75% (p=0.000 n=10)
MulVSF32/4096-32          744.6n ±  1%   149.1n ± 2%  -79.98% (p=0.000 n=10)
MulVSF32/1048576-32      186.15µ ±  1%   36.19µ ± 2%  -80.56% (p=0.000 n=10)
AddVSF64/64-32           13.325n ±  3%   6.848n ± 4%  -48.61% (p=0.000 n=10)
AddVSF64/4096-32          763.9n ±  1%   287.0n ± 2%  -62.43% (p=0.000 n=10)
AddVSF64/1048576-32      204.07µ ±  3%   72.13µ ± 2%  -64.65% (p=0.000 n=10)
MulVSF64/64-32            7.620n ±  4%   6.793n ± 2%  -10.85% (p=0.000 n=10)
MulVSF64/4096-32          738.9n ±  2%   286.0n ± 2%  -61.29% (p=0.000 n=10)
MulVSF64/1048576-32      185.61µ ±  1%   73.94µ ± 3%  -60.17% (p=0.000 n=10)
AddSVF32/64-32           12.765n ±  1%   4.555n ± 1%  -64.32% (p=0.000 n=10)
AddSVF32/4096-32          733.4n ±  1%   149.6n ± 2%  -79.61% (p=0.000 n=10)
AddSVF32/1048576-32      195.48µ ±  2%   36.94µ ± 3%  -81.10% (p=0.000 n=10)
MinVSF32/64-32           12.770n ±  1%   4.276n ± 3%  -66.52% (p=0.000 n=10)
MinVSF32/4096-32          768.4n ±  2%   151.5n ± 2%  -80.28% (p=0.000 n=10)
MinVSF32/1048576-32      199.91µ ±  2%   39.03µ ± 3%  -80.48% (p=0.000 n=10)
VecAddI32/64-32          12.645n ±  1%   7.038n ± 2%  -44.34% (p=0.000 n=10)
VecAddI32/4096-32         761.8n ±  2%   197.4n ± 1%  -74.09% (p=0.000 n=10)
VecAddI32/1048576-32     201.39µ ±  2%   64.13µ ± 3%  -68.16% (p=0.000 n=10)
VecMulI32/64-32          14.870n ±  1%   7.359n ± 2%  -50.51% (p=0.000 n=10)
VecMulI32/4096-32         953.8n ±  1%   195.8n ± 2%  -79.47% (p=0.000 n=10)
VecMulI32/1048576-32     249.86µ ±  2%   63.78µ ± 3%  -74.48% (p=0.000 n=10)
VecAddF32/64-32           13.47n ±  2%   13.61n ± 3%        ~ (p=0.093 n=10)
VecAddF32/4096-32         780.0n ±  5%   762.8n ± 3%   -2.21% (p=0.008 n=10)
VecAddF32/1048576-32      201.6µ ±  3%   203.0µ ± 1%        ~ (p=0.089 n=10)
VecDivF32/64-32           28.93n ±  3%   30.71n ± 2%   +6.12% (p=0.000 n=10)
VecDivF32/4096-32         1.852µ ±  1%   1.844µ ± 2%        ~ (p=0.492 n=10)
VecDivF32/1048576-32     1010.1µ ± 10%   974.0µ ± 6%        ~ (p=0.971 n=10)
VecMaxF32/64-32          13.890n ±  5%   6.993n ± 3%  -49.65% (p=0.000 n=10)
VecMaxF32/4096-32         749.5n ±  2%   199.4n ± 3%  -73.39% (p=0.000 n=10)
VecMaxF32/1048576-32     205.29µ ±  2%   64.29µ ± 2%  -68.68% (p=0.000 n=10)
NegF32/64-32             12.990n ±  1%   4.014n ± 2%  -69.10% (p=0.000 n=10)
NegF32/4096-32            737.4n ±  1%   147.8n ± 1%  -79.95% (p=0.000 n=10)
NegF32/1048576-32        198.26µ ±  1%   38.19µ ± 3%  -80.74% (p=0.000 n=10)
AbsF32/64-32              13.92n ±  1%   13.99n ± 2%        ~ (p=0.247 n=10)
AbsF32/4096-32            893.0n ±  2%   892.0n ± 1%        ~ (p=0.987 n=10)
AbsF32/1048576-32         230.1µ ±  1%   225.5µ ± 3%        ~ (p=0.089 n=10)
SqrtF32/64-32             58.46n ±  2%   59.14n ± 2%   +1.16% (p=0.011 n=10)
SqrtF32/4096-32           3.724µ ±  1%   3.662µ ± 4%        ~ (p=0.210 n=10)
SqrtF32/1048576-32        881.9µ ±  5%   885.3µ ± 2%        ~ (p=0.393 n=10)
InvF32/64-32             28.830n ±  1%   4.257n ± 2%  -85.23% (p=0.000 n=10)
InvF32/4096-32           1843.5n ±  2%   123.6n ± 1%  -93.30% (p=0.000 n=10)
InvF32/1048576-32        469.11µ ±  2%   33.03µ ± 3%  -92.96% (p=0.000 n=10)
SquareF32/64-32          13.035n ±  6%   4.216n ± 4%  -67.65% (p=0.000 n=10)
SquareF32/4096-32         773.1n ±  1%   129.3n ± 3%  -83.28% (p=0.000 n=10)
SquareF32/1048576-32     199.58µ ±  2%   32.58µ ± 5%  -83.68% (p=0.000 n=10)
ClampF32/64-32            35.89n ±  3%   35.57n ± 3%        ~ (p=0.239 n=10)
ClampF32/4096-32          2.304µ ±  4%   2.215µ ± 3%   -3.86% (p=0.002 n=10)
ClampF32/1048576-32       575.0µ ±  2%   570.6µ ± 2%        ~ (p=0.052 n=10)
GtSameF32/64-32          25.825n ±  3%   6.121n ± 2%  -76.30% (p=0.000 n=10)
GtSameF32/4096-32        1521.5n ±  2%   175.4n ± 2%  -88.47% (p=0.000 n=10)
GtSameF32/1048576-32     406.89µ ±  5%   60.96µ ± 4%  -85.02% (p=0.000 n=10)
EqSameF32/64-32          19.125n ±  8%   6.479n ± 7%  -66.12% (p=0.000 n=10)
EqSameF32/4096-32        1143.5n ±  4%   175.4n ± 1%  -84.66% (p=0.000 n=10)
EqSameF32/1048576-32     297.90µ ±  3%   63.64µ ± 2%  -78.64% (p=0.000 n=10)
SumI32/64-32             10.425n ±  3%   7.758n ± 3%  -25.58% (p=0.000 n=10)
SumI32/4096-32            738.7n ±  2%   118.5n ± 2%  -83.96% (p=0.000 n=10)
SumI32/1048576-32        191.63µ ±  3%   31.15µ ± 2%  -83.75% (p=0.000 n=10)
SliceMaxI32/64-32         26.44n ±  2%   26.38n ± 1%        ~ (p=0.930 n=10)
SliceMaxI32/4096-32       2.178µ ±  1%   2.169µ ± 2%        ~ (p=0.517 n=10)
SliceMaxI32/1048576-32    557.5µ ±  2%   555.8µ ± 2%        ~ (p=0.529 n=10)
SumF32/64-32              11.57n ±  9%   12.97n ± 2%  +12.05% (p=0.002 n=10)
SumF32/4096-32           1432.0n ±  1%   112.2n ± 2%  -92.16% (p=0.000 n=10)
SumF32/1048576-32        375.53µ ±  2%   30.79µ ± 5%  -91.80% (p=0.000 n=10)
geomean                   1.704µ         621.1n       -63.56%
```
</details>

Running `loopvec -methods -split` on [gonum](https://github.com/gonum/gonum)
rewrites 121 loops, or 180 with `-fp-reassoc`.

## Installation

```sh
go install github.com/mauri870/loopvec@latest
```

## Usage

### Standalone

```sh
loopvec ./...             # print rewritten source to stdout
loopvec -d ./...          # show unified diff (like gofmt -d)
loopvec -split ./...      # recommended: write file_simd.go + guard original
loopvec -w ./...          # overwrite files in place (requires version control)
loopvec -json ./...       # print one JSON line per candidate loop; see below
```

### Coverage reporting: `-json`

`-json` prints one JSON line per loop detected with context about vectorizer decisions

```sh
$ loopvec -json ./mypkg/
{"file":"/path/to/mypkg/ops.go","line":4,"func":"AddFloat32s","vectorized":true}
{"file":"/path/to/mypkg/ops.go","line":10,"func":"Stride","vectorized":false,"reason":"loop start is not 0 (or len(s)-1 when counting down)","stage":"lower"}
```

Loops with an `if` in the body or several statements are listed too. `stage` is
`lower` (the shape of the loop), `plan` (no simd method for that type), or
`analysis` (a method, skipped without `-methods`).

### As a `go tool` (Go 1.24+)

Add `loopvec` as a tool dependency in your module:

```sh
go get -tool github.com/mauri870/loopvec@latest
```

Then invoke it without installing globally:

```sh
go tool loopvec -d ./...
go tool loopvec -split ./mypkg/...
```

## Workflow

`-split` is the recommended mode. It writes the vectorized code to
`ops_simd.go` (with `//go:build goexperiment.simd`) and adds
`//go:build !goexperiment.simd` to the original `ops.go`. Both files stay in
your repository: the scalar version builds by default, and the simd version
builds when `GOEXPERIMENT=simd` is set.

```sh
# 1. Preview changes
go tool loopvec -d ./mypkg/...

# 2. Apply: creates ops_simd.go and guards ops.go
go tool loopvec -split ./mypkg/...

# 3. Verify correctness and benchmark
go test ./mypkg/...
GOEXPERIMENT=simd go test -bench=. ./mypkg/
```

## Whole-program mode: `-toolexec`

`loopvec-toolexec` is an experimental `go build -toolexec` wrapper that vectorizes
loops while the go command compiles a build, dependencies and standard library
included, without touching any source file. See [toolexec.md](toolexec.md).

## Requirements

- Go 1.27 or later
- `GOEXPERIMENT=simd` at build time for the rewritten code

## Known Limitations


## Testing

The testing suite is built on top of [rsc.io/script](https://pkg.go.dev/rsc.io/script) with test scripts in `testdata/*.txt`,
the same approach used by [`go` command tests](https://github.com/golang/go/tree/f2d76b7/src/cmd/go/testdata/script).

```sh
make test
```

[tsvc/](tsvc/) is a Go port of the [TSVC_2](https://github.com/UoB-HPC/TSVC_2)
vectorizer kernel suite. It checks the scalar and SIMD builds and records which
kernels loopvec vectorizes. See [tsvc/README.md](tsvc/README.md).
