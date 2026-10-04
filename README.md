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
AddFloat32s/64-32                        12.610n ±   3%   7.157n ±  2%  -43.24% (p=0.000 n=10)
AddFloat32s/4096-32                       831.5n ±   2%   193.6n ±  3%  -76.72% (p=0.000 n=10)
AddFloat32s/1048576-32                   213.55µ ±   2%   86.15µ ±  2%  -59.66% (p=0.000 n=10)
MulFloat32s/64-32                        12.715n ±   1%   6.609n ±  1%  -48.02% (p=0.000 n=10)
MulFloat32s/4096-32                       840.2n ±   4%   188.0n ±  2%  -77.63% (p=0.000 n=10)
MulFloat32s/1048576-32                   216.67µ ±   5%   89.25µ ±  6%  -58.81% (p=0.000 n=10)
ScalFloat32s/64-32                       10.025n ±   6%   3.695n ±  3%  -63.14% (p=0.000 n=10)
ScalFloat32s/4096-32                      741.5n ±   2%   147.2n ±  1%  -80.14% (p=0.000 n=10)
ScalFloat32s/1048576-32                  190.32µ ±   2%   37.22µ ±  2%  -80.44% (p=0.000 n=10)
MixFloat32s/64-32                        23.505n ±   1%   7.825n ±  3%  -66.71% (p=0.000 n=10)
MixFloat32s/4096-32                      1490.0n ±   3%   225.4n ±  7%  -84.87% (p=0.000 n=10)
MixFloat32s/1048576-32                   381.78µ ±   2%   89.60µ ±  6%  -76.53% (p=0.000 n=10)
AxpyFloat32s/64-32                       19.035n ±   2%   6.695n ±  1%  -64.83% (p=0.000 n=10)
AxpyFloat32s/4096-32                      802.8n ±   3%   197.8n ±  1%  -75.36% (p=0.000 n=10)
AxpyFloat32s/1048576-32                  219.40µ ±   2%   85.16µ ±  8%  -61.19% (p=0.000 n=10)
NegFloat32s/64-32                        14.690n ±   6%   5.149n ±  4%  -64.95% (p=0.000 n=10)
NegFloat32s/4096-32                       936.0n ±   6%   173.6n ±  3%  -81.45% (p=0.000 n=10)
NegFloat32s/1048576-32                   219.55µ ±   6%   59.78µ ±  1%  -72.77% (p=0.000 n=10)
DivFloat32s/64-32                        28.850n ±   1%   7.264n ±  1%  -74.82% (p=0.000 n=10)
DivFloat32s/4096-32                      1828.5n ±   2%   196.3n ±  2%  -89.27% (p=0.000 n=10)
DivFloat32s/1048576-32                   470.52µ ±   2%   92.77µ ±  3%  -80.28% (p=0.000 n=10)
DaxpyFloat32s/64-32                      14.085n ±   7%   5.590n ±  4%  -60.32% (p=0.000 n=10)
DaxpyFloat32s/4096-32                     887.3n ±   7%   177.2n ±  2%  -80.03% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                 221.06µ ±   2%   64.31µ ±  1%  -70.91% (p=0.000 n=10)
FillFloat32s/64-32                        8.162n ±   4%   3.070n ±  9%  -62.39% (p=0.000 n=10)
FillFloat32s/4096-32                      727.6n ±   2%   108.4n ±  5%  -85.10% (p=0.000 n=10)
FillFloat32s/1048576-32                  187.83µ ±   3%   30.68µ ±  3%  -83.67% (p=0.000 n=10)
FillUint8s/64-32                          7.929n ±   6%   1.996n ±  4%  -74.83% (p=0.000 n=10)
FillUint8s/4096-32                       738.70n ±   2%   28.06n ±  5%  -96.20% (p=0.000 n=10)
FillUint8s/1048576-32                   189.782µ ±   1%   6.203µ ±  4%  -96.73% (p=0.000 n=10)
ReverseIncFloat32s/64-32                 13.395n ±   5%   5.188n ±  9%  -61.27% (p=0.000 n=10)
ReverseIncFloat32s/4096-32                762.6n ±   2%   166.4n ±  3%  -78.17% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32            200.83µ ±  12%   59.20µ ±  4%  -70.52% (p=0.000 n=10)
CopyFloat32s/64-32                       11.760n ±   2%   4.159n ±  1%  -64.63% (p=0.000 n=10)
CopyFloat32s/4096-32                     758.60n ±   1%   73.19n ±  2%  -90.35% (p=0.000 n=10)
CopyFloat32s/1048576-32                  190.08µ ±   2%   58.57µ ±  5%  -69.18% (p=0.000 n=10)
AndNotUint64s/64-32                      18.705n ±   2%   9.806n ±  1%  -47.57% (p=0.000 n=10)
AndNotUint64s/4096-32                    1223.0n ±   3%   391.4n ±  3%  -68.00% (p=0.000 n=10)
AndNotUint64s/1048576-32                  316.6µ ±  19%   170.4µ ±  6%  -46.18% (p=0.000 n=10)
NormalizeFloat32s/64-32                  28.990n ±   2%   6.603n ±  1%  -77.22% (p=0.000 n=10)
NormalizeFloat32s/4096-32                1851.0n ±   4%   232.5n ±  3%  -87.44% (p=0.000 n=10)
NormalizeFloat32s/1048576-32             475.71µ ±   5%   67.50µ ±  2%  -85.81% (p=0.000 n=10)
ChainInt32s/64-32                         32.44n ±  10%   13.13n ±  6%  -59.52% (p=0.000 n=10)
ChainInt32s/4096-32                      2080.0n ±   8%   328.8n ±  2%  -84.19% (p=0.000 n=10)
ChainInt32s/1048576-32                    489.8µ ±   4%   114.9µ ± 14%  -76.54% (p=0.000 n=10)
WideInt32s/64-32                        112.385n ±  51%   9.595n ±  1%  -91.46% (p=0.000 n=10)
WideInt32s/4096-32                       3052.0n ±   4%   275.2n ±  3%  -90.98% (p=0.000 n=10)
WideInt32s/1048576-32                     771.4µ ±   4%   119.6µ ± 11%  -84.49% (p=0.000 n=10)
SumInt32s/64-32                         115.950n ±  88%   7.621n ±  1%  -93.43% (p=0.000 n=10)
SumInt32s/4096-32                        8374.0n ±  20%   196.9n ±  2%  -97.65% (p=0.000 n=10)
SumInt32s/1048576-32                    1591.65µ ± 174%   46.94µ ±  2%  -97.05% (p=0.000 n=10)
DotInt32s/64-32                          51.990n ± 172%   9.222n ±  4%  -82.26% (p=0.000 n=10)
DotInt32s/4096-32                        1082.5n ±  14%   205.8n ±  1%  -80.99% (p=0.000 n=10)
DotInt32s/1048576-32                     248.85µ ±  12%   64.55µ ±  1%  -74.06% (p=0.000 n=10)
SumFloat32s/64-32                         11.03n ±  11%   13.03n ±  2%  +18.19% (p=0.001 n=10)
SumFloat32s/4096-32                      1446.0n ±   2%   115.1n ±  4%  -92.04% (p=0.000 n=10)
SumFloat32s/1048576-32                   374.95µ ±   2%   31.54µ ±  3%  -91.59% (p=0.000 n=10)
DotFloat32s/64-32                         16.29n ±   3%   13.38n ±  2%  -17.83% (p=0.000 n=10)
DotFloat32s/4096-32                      1469.0n ±   2%   185.1n ±  4%  -87.40% (p=0.000 n=10)
DotFloat32s/1048576-32                   385.04µ ±   1%   64.32µ ±  2%  -83.29% (p=0.000 n=10)
DotFloat64s/64-32                         12.23n ±   8%   12.27n ±  1%        ~ (p=0.853 n=10)
DotFloat64s/4096-32                      1437.5n ±   2%   366.7n ±  2%  -74.49% (p=0.000 n=10)
DotFloat64s/1048576-32                    381.7µ ±   2%   122.3µ ±  9%  -67.95% (p=0.000 n=10)
AbsFloat32s/64-32                        24.750n ±   2%   5.059n ±  2%  -79.56% (p=0.000 n=10)
AbsFloat32s/4096-32                      1513.5n ±   4%   165.3n ±  3%  -89.07% (p=0.000 n=10)
AbsFloat32s/1048576-32                   386.12µ ±   2%   60.46µ ±  4%  -84.34% (p=0.000 n=10)
SqrtFloat32s/64-32                       52.060n ±   1%   5.340n ±  2%  -89.74% (p=0.000 n=10)
SqrtFloat32s/4096-32                     3316.0n ±   1%   220.5n ±  1%  -93.35% (p=0.000 n=10)
SqrtFloat32s/1048576-32                  851.27µ ±   3%   59.16µ ±  6%  -93.05% (p=0.000 n=10)
ClampInt32s/64-32                        29.670n ±   3%   4.771n ±  1%  -83.92% (p=0.000 n=10)
ClampInt32s/4096-32                      1921.0n ±   4%   149.8n ±  3%  -92.20% (p=0.000 n=10)
ClampInt32s/1048576-32                   485.84µ ±   1%   59.73µ ±  4%  -87.71% (p=0.000 n=10)
ShrInt32s/64-32                          13.700n ±   2%   4.830n ±  1%  -64.74% (p=0.000 n=10)
ShrInt32s/4096-32                         854.2n ±   3%   160.6n ±  2%  -81.19% (p=0.000 n=10)
ShrInt32s/1048576-32                     213.33µ ±   5%   61.11µ ±  5%  -71.35% (p=0.000 n=10)
ShlUint32s/64-32                         13.045n ±   5%   5.028n ±  5%  -61.46% (p=0.000 n=10)
ShlUint32s/4096-32                        847.6n ±   4%   150.2n ±  2%  -82.29% (p=0.000 n=10)
ShlUint32s/1048576-32                    218.54µ ±   1%   58.24µ ±  6%  -73.35% (p=0.000 n=10)
DiffInt32s/64-32                         14.940n ±   5%   8.758n ±  2%  -41.38% (p=0.000 n=10)
DiffInt32s/4096-32                        922.9n ±   2%   427.9n ±  1%  -53.64% (p=0.000 n=10)
DiffInt32s/1048576-32                     234.1µ ±   2%   106.9µ ±  2%  -54.33% (p=0.000 n=10)
SmoothInt32s/64-32                       22.075n ±   1%   9.456n ±  2%  -57.16% (p=0.000 n=10)
SmoothInt32s/4096-32                     1784.5n ±   2%   499.6n ±  8%  -72.01% (p=0.000 n=10)
SmoothInt32s/1048576-32                   497.2µ ±   2%   120.5µ ±  2%  -75.77% (p=0.000 n=10)
ShiftInt32s/64-32                        16.160n ±   2%   5.224n ±  3%  -67.67% (p=0.000 n=10)
ShiftInt32s/4096-32                      1084.5n ±   3%   196.8n ±  3%  -81.85% (p=0.000 n=10)
ShiftInt32s/1048576-32                   273.53µ ±   4%   65.11µ ±  6%  -76.20% (p=0.000 n=10)
AccumRowInt32s/64-32                     16.580n ±   2%   6.359n ±  1%  -61.65% (p=0.000 n=10)
AccumRowInt32s/4096-32                   1070.5n ±   3%   241.3n ±  2%  -77.46% (p=0.000 n=10)
AccumRowInt32s/1048576-32                279.55µ ±   7%   64.59µ ±  3%  -76.89% (p=0.000 n=10)
MinUint16s/64-32                          25.01n ±   2%   30.47n ±  1%  +21.85% (p=0.000 n=10)
MinUint16s/4096-32                      2205.50n ±   0%   67.57n ± 10%  -96.94% (p=0.000 n=10)
MinUint16s/1048576-32                    563.17µ ±   1%   15.44µ ±  3%  -97.26% (p=0.000 n=10)
ProductFloat32s/64-32                    21.430n ±   2%   7.755n ±  3%  -63.81% (p=0.000 n=10)
ProductFloat32s/4096-32                  2155.5n ±   2%   108.5n ±  3%  -94.96% (p=0.000 n=10)
ProductFloat32s/1048576-32               564.44µ ±   2%   31.33µ ±  1%  -94.45% (p=0.000 n=10)
ShiftAddInt32s/64-32                     15.150n ±  10%   7.713n ±  1%  -49.09% (p=0.000 n=10)
ShiftAddInt32s/4096-32                    861.1n ±   4%   274.9n ±  2%  -68.07% (p=0.000 n=10)
ShiftAddInt32s/1048576-32                222.47µ ±   1%   73.95µ ±  2%  -66.76% (p=0.000 n=10)
ReadAfterStoreInt32s/64-32                32.22n ±   3%   19.09n ±  1%  -40.78% (p=0.000 n=10)
ReadAfterStoreInt32s/4096-32             2410.5n ±   7%   584.1n ±  4%  -75.77% (p=0.000 n=10)
ReadAfterStoreInt32s/1048576-32           656.7µ ±   2%   153.0µ ±  3%  -76.70% (p=0.000 n=10)
ReLUFloat32s/64-32                       12.920n ±   5%   3.777n ±  1%  -70.77% (p=0.000 n=10)
ReLUFloat32s/4096-32                      824.2n ±   7%   148.7n ±  1%  -81.96% (p=0.000 n=10)
ReLUFloat32s/1048576-32                  202.77µ ±   7%   37.81µ ±  4%  -81.35% (p=0.000 n=10)
LeakyReLUFloat32s/64-32                  23.445n ±   3%   5.640n ±  2%  -75.94% (p=0.000 n=10)
LeakyReLUFloat32s/4096-32                1473.0n ±   5%   173.6n ±  2%  -88.21% (p=0.000 n=10)
LeakyReLUFloat32s/1048576-32             379.33µ ±   3%   59.79µ ±  3%  -84.24% (p=0.000 n=10)
ReLU6Float32s/64-32                      23.315n ±   2%   3.947n ±  1%  -83.07% (p=0.000 n=10)
ReLU6Float32s/4096-32                    1466.5n ±   1%   164.2n ±  2%  -88.80% (p=0.000 n=10)
ReLU6Float32s/1048576-32                 393.61µ ±  13%   42.48µ ±  6%  -89.21% (p=0.000 n=10)
SignFloat32s/64-32                       24.350n ±   2%   5.383n ±  3%  -77.90% (p=0.000 n=10)
SignFloat32s/4096-32                     1574.0n ±   3%   183.5n ±  3%  -88.34% (p=0.000 n=10)
SignFloat32s/1048576-32                  389.94µ ±   2%   59.53µ ±  4%  -84.73% (p=0.000 n=10)
geomean                                   2.075µ          426.0n        -79.47%

                                │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                                │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                        56.70Gi ±   3%     99.94Gi ±  2%    +76.25% (p=0.000 n=10)
AddFloat32s/4096-32                      55.05Gi ±   2%    236.42Gi ±  3%   +329.43% (p=0.000 n=10)
AddFloat32s/1048576-32                   54.88Gi ±   2%    136.02Gi ±  2%   +147.88% (p=0.000 n=10)
MulFloat32s/64-32                        56.25Gi ±   1%    108.22Gi ±  1%    +92.38% (p=0.000 n=10)
MulFloat32s/4096-32                      54.48Gi ±   5%    243.45Gi ±  2%   +346.88% (p=0.000 n=10)
MulFloat32s/1048576-32                   54.09Gi ±   5%    131.31Gi ±  6%   +142.77% (p=0.000 n=10)
ScalFloat32s/64-32                       47.57Gi ±   5%    129.04Gi ±  4%   +171.26% (p=0.000 n=10)
ScalFloat32s/4096-32                     41.15Gi ±   2%    207.22Gi ±  1%   +403.55% (p=0.000 n=10)
ScalFloat32s/1048576-32                  41.05Gi ±   2%    209.89Gi ±  2%   +411.31% (p=0.000 n=10)
MixFloat32s/64-32                        30.43Gi ±   1%     91.41Gi ±  3%   +200.42% (p=0.000 n=10)
MixFloat32s/4096-32                      30.72Gi ±   3%    203.04Gi ±  7%   +560.88% (p=0.000 n=10)
MixFloat32s/1048576-32                   30.70Gi ±   2%    130.80Gi ±  7%   +326.11% (p=0.000 n=10)
AxpyFloat32s/64-32                       37.58Gi ±   2%    106.84Gi ±  1%   +184.34% (p=0.000 n=10)
AxpyFloat32s/4096-32                     57.02Gi ±   3%    231.40Gi ±  1%   +305.80% (p=0.000 n=10)
AxpyFloat32s/1048576-32                  53.41Gi ±   2%    137.77Gi ±  7%   +157.92% (p=0.000 n=10)
NegFloat32s/64-32                        32.46Gi ±   7%     92.61Gi ±  4%   +185.33% (p=0.000 n=10)
NegFloat32s/4096-32                      32.62Gi ±   7%    175.77Gi ±  4%   +438.79% (p=0.000 n=10)
NegFloat32s/1048576-32                   35.59Gi ±   6%    130.69Gi ±  1%   +267.26% (p=0.000 n=10)
DivFloat32s/64-32                        24.79Gi ±   1%     98.46Gi ±  1%   +297.15% (p=0.000 n=10)
DivFloat32s/4096-32                      25.04Gi ±   2%    233.27Gi ±  2%   +831.61% (p=0.000 n=10)
DivFloat32s/1048576-32                   24.91Gi ±   2%    126.32Gi ±  3%   +407.16% (p=0.000 n=10)
DaxpyFloat32s/64-32                      50.79Gi ±   6%    127.96Gi ±  4%   +151.92% (p=0.000 n=10)
DaxpyFloat32s/4096-32                    51.61Gi ±   8%    258.44Gi ±  2%   +400.79% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                 53.01Gi ±   2%    182.22Gi ±  1%   +243.73% (p=0.000 n=10)
FillFloat32s/64-32                       29.21Gi ±   4%     77.69Gi ±  8%   +165.91% (p=0.000 n=10)
FillFloat32s/4096-32                     20.97Gi ±   2%    140.79Gi ±  5%   +571.39% (p=0.000 n=10)
FillFloat32s/1048576-32                  20.80Gi ±   3%    127.32Gi ±  3%   +512.21% (p=0.000 n=10)
FillUint8s/64-32                         7.517Gi ±   5%    29.862Gi ±  4%   +297.25% (p=0.000 n=10)
FillUint8s/4096-32                       5.164Gi ±   2%   135.940Gi ±  5%  +2532.21% (p=0.000 n=10)
FillUint8s/1048576-32                    5.146Gi ±   1%   157.423Gi ±  4%  +2959.31% (p=0.000 n=10)
ReverseIncFloat32s/64-32                 35.60Gi ±   5%     91.92Gi ±  8%   +158.20% (p=0.000 n=10)
ReverseIncFloat32s/4096-32               40.02Gi ±   2%    183.41Gi ±  3%   +358.31% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32            38.90Gi ±  11%    131.97Gi ±  4%   +239.25% (p=0.000 n=10)
CopyFloat32s/64-32                       40.55Gi ±   2%    114.66Gi ±  1%   +182.77% (p=0.000 n=10)
CopyFloat32s/4096-32                     40.23Gi ±   1%    416.99Gi ±  2%   +936.53% (p=0.000 n=10)
CopyFloat32s/1048576-32                  41.11Gi ±   2%    133.38Gi ±  5%   +224.46% (p=0.000 n=10)
AndNotUint64s/64-32                      76.47Gi ±   2%    145.88Gi ±  1%    +90.76% (p=0.000 n=10)
AndNotUint64s/4096-32                    74.85Gi ±   3%    233.91Gi ±  3%   +212.51% (p=0.000 n=10)
AndNotUint64s/1048576-32                 74.04Gi ±  16%    137.56Gi ±  6%    +85.80% (p=0.000 n=10)
NormalizeFloat32s/64-32                  32.90Gi ±   2%    144.43Gi ±  1%   +338.96% (p=0.000 n=10)
NormalizeFloat32s/4096-32                32.98Gi ±   4%    262.59Gi ±  3%   +696.27% (p=0.000 n=10)
NormalizeFloat32s/1048576-32             32.85Gi ±   5%    231.50Gi ±  2%   +604.81% (p=0.000 n=10)
ChainInt32s/64-32                        44.09Gi ±   9%    108.92Gi ±  5%   +147.01% (p=0.000 n=10)
ChainInt32s/4096-32                      44.02Gi ±   8%    278.47Gi ±  2%   +532.56% (p=0.000 n=10)
ChainInt32s/1048576-32                   47.86Gi ±   4%    205.10Gi ± 12%   +328.58% (p=0.000 n=10)
WideInt32s/64-32                         8.975Gi ±  94%    99.389Gi ±  1%  +1007.41% (p=0.000 n=10)
WideInt32s/4096-32                       20.00Gi ±   4%    221.80Gi ±  3%  +1009.11% (p=0.000 n=10)
WideInt32s/1048576-32                    20.26Gi ±   3%    130.62Gi ± 13%   +544.88% (p=0.000 n=10)
SumInt32s/64-32                          2.068Gi ± 739%    31.281Gi ±  1%  +1412.48% (p=0.000 n=10)
SumInt32s/4096-32                        1.822Gi ±  25%    77.500Gi ±  3%  +4153.03% (p=0.000 n=10)
SumInt32s/1048576-32                     2.762Gi ±  68%    83.219Gi ±  2%  +2913.03% (p=0.000 n=10)
DotInt32s/64-32                          9.172Gi ±  63%    51.707Gi ±  4%   +463.73% (p=0.000 n=10)
DotInt32s/4096-32                        28.19Gi ±  15%    148.32Gi ±  2%   +426.07% (p=0.000 n=10)
DotInt32s/1048576-32                     31.40Gi ±  11%    121.04Gi ±  1%   +285.52% (p=0.000 n=10)
SumFloat32s/64-32                        21.62Gi ±  10%     18.30Gi ±  2%    -15.37% (p=0.002 n=10)
SumFloat32s/4096-32                      10.55Gi ±   2%    132.61Gi ±  5%  +1156.58% (p=0.000 n=10)
SumFloat32s/1048576-32                   10.42Gi ±   2%    123.84Gi ±  3%  +1088.72% (p=0.000 n=10)
DotFloat32s/64-32                        29.27Gi ±   3%     35.62Gi ±  2%    +21.68% (p=0.000 n=10)
DotFloat32s/4096-32                      20.77Gi ±   2%    164.88Gi ±  4%   +693.70% (p=0.000 n=10)
DotFloat32s/1048576-32                   20.29Gi ±   1%    121.46Gi ±  2%   +498.59% (p=0.000 n=10)
DotFloat64s/64-32                        77.96Gi ±   8%     77.68Gi ±  1%          ~ (p=0.853 n=10)
DotFloat64s/4096-32                      42.46Gi ±   2%    166.44Gi ±  2%   +292.02% (p=0.000 n=10)
DotFloat64s/1048576-32                   40.94Gi ±   2%    127.79Gi ±  8%   +212.14% (p=0.000 n=10)
AbsFloat32s/64-32                        19.26Gi ±   2%     94.25Gi ±  2%   +389.27% (p=0.000 n=10)
AbsFloat32s/4096-32                      20.17Gi ±   4%    184.58Gi ±  3%   +815.21% (p=0.000 n=10)
AbsFloat32s/1048576-32                   20.23Gi ±   2%    129.22Gi ±  4%   +538.62% (p=0.000 n=10)
SqrtFloat32s/64-32                       9.159Gi ±   1%    89.299Gi ±  2%   +874.95% (p=0.000 n=10)
SqrtFloat32s/4096-32                     9.204Gi ±   1%   138.415Gi ±  1%  +1403.91% (p=0.000 n=10)
SqrtFloat32s/1048576-32                  9.178Gi ±   3%   132.056Gi ±  6%  +1338.91% (p=0.000 n=10)
ClampInt32s/64-32                        16.07Gi ±   3%     99.94Gi ±  1%   +521.79% (p=0.000 n=10)
ClampInt32s/4096-32                      15.89Gi ±   4%    203.79Gi ±  3%  +1182.85% (p=0.000 n=10)
ClampInt32s/1048576-32                   16.08Gi ±   1%    130.81Gi ±  4%   +713.46% (p=0.000 n=10)
ShrInt32s/64-32                          34.80Gi ±   2%     98.71Gi ±  1%   +183.65% (p=0.000 n=10)
ShrInt32s/4096-32                        35.73Gi ±   3%    189.97Gi ±  2%   +431.66% (p=0.000 n=10)
ShrInt32s/1048576-32                     36.63Gi ±   5%    127.84Gi ±  5%   +249.06% (p=0.000 n=10)
ShlUint32s/64-32                         36.54Gi ±   5%     94.83Gi ±  5%   +159.51% (p=0.000 n=10)
ShlUint32s/4096-32                       36.01Gi ±   4%    203.28Gi ±  2%   +464.58% (p=0.000 n=10)
ShlUint32s/1048576-32                    35.75Gi ±   1%    134.14Gi ±  6%   +275.23% (p=0.000 n=10)
DiffInt32s/64-32                         31.92Gi ±   5%     54.45Gi ±  2%    +70.58% (p=0.000 n=10)
DiffInt32s/4096-32                       33.07Gi ±   2%     71.32Gi ±  1%   +115.70% (p=0.000 n=10)
DiffInt32s/1048576-32                    33.37Gi ±   2%     73.06Gi ±  2%   +118.96% (p=0.000 n=10)
SmoothInt32s/64-32                       21.60Gi ±   1%     50.43Gi ±  2%   +133.48% (p=0.000 n=10)
SmoothInt32s/4096-32                     17.10Gi ±   2%     61.11Gi ±  7%   +257.35% (p=0.000 n=10)
SmoothInt32s/1048576-32                  15.71Gi ±   2%     64.84Gi ±  2%   +312.69% (p=0.000 n=10)
ShiftInt32s/64-32                        29.51Gi ±   2%     91.27Gi ±  3%   +209.28% (p=0.000 n=10)
ShiftInt32s/4096-32                      28.14Gi ±   3%    155.09Gi ±  3%   +451.15% (p=0.000 n=10)
ShiftInt32s/1048576-32                   28.56Gi ±   4%    120.00Gi ±  6%   +320.13% (p=0.000 n=10)
AccumRowInt32s/64-32                     43.13Gi ±   3%    112.47Gi ±  1%   +160.75% (p=0.000 n=10)
AccumRowInt32s/4096-32                   42.76Gi ±   3%    189.72Gi ±  2%   +343.66% (p=0.000 n=10)
AccumRowInt32s/1048576-32                41.92Gi ±   7%    181.42Gi ±  3%   +332.76% (p=0.000 n=10)
MinUint16s/64-32                         4.767Gi ±   2%     3.912Gi ±  1%    -17.94% (p=0.000 n=10)
MinUint16s/4096-32                       3.459Gi ±   0%   112.924Gi ±  9%  +3164.80% (p=0.000 n=10)
MinUint16s/1048576-32                    3.468Gi ±   1%   126.475Gi ±  2%  +3546.84% (p=0.000 n=10)
ProductFloat32s/64-32                    11.13Gi ±   2%     30.74Gi ±  3%   +176.29% (p=0.000 n=10)
ProductFloat32s/4096-32                  7.078Gi ±   2%   140.611Gi ±  3%  +1886.61% (p=0.000 n=10)
ProductFloat32s/1048576-32               6.921Gi ±   2%   124.682Gi ±  1%  +1701.60% (p=0.000 n=10)
ShiftAddInt32s/64-32                     47.26Gi ±  10%     92.74Gi ±  1%    +96.23% (p=0.000 n=10)
ShiftAddInt32s/4096-32                   53.16Gi ±   4%    166.51Gi ±  2%   +213.19% (p=0.000 n=10)
ShiftAddInt32s/1048576-32                52.68Gi ±   1%    158.47Gi ±  2%   +200.84% (p=0.000 n=10)
ReadAfterStoreInt32s/64-32               22.20Gi ±   3%     37.48Gi ±  1%    +68.85% (p=0.000 n=10)
ReadAfterStoreInt32s/4096-32             18.99Gi ±   6%     78.36Gi ±  4%   +312.57% (p=0.000 n=10)
ReadAfterStoreInt32s/1048576-32          17.84Gi ±   2%     76.57Gi ±  3%   +329.09% (p=0.000 n=10)
ReLUFloat32s/64-32                       36.90Gi ±   5%    126.27Gi ±  1%   +242.22% (p=0.000 n=10)
ReLUFloat32s/4096-32                     37.03Gi ±   7%    205.21Gi ±  1%   +454.15% (p=0.000 n=10)
ReLUFloat32s/1048576-32                  38.53Gi ±   7%    206.62Gi ±  4%   +436.26% (p=0.000 n=10)
LeakyReLUFloat32s/64-32                  20.34Gi ±   3%     84.54Gi ±  2%   +315.62% (p=0.000 n=10)
LeakyReLUFloat32s/4096-32                20.72Gi ±   5%    175.78Gi ±  2%   +748.44% (p=0.000 n=10)
LeakyReLUFloat32s/1048576-32             20.60Gi ±   3%    130.66Gi ±  2%   +534.43% (p=0.000 n=10)
ReLU6Float32s/64-32                      20.45Gi ±   2%    120.81Gi ±  1%   +490.76% (p=0.000 n=10)
ReLU6Float32s/4096-32                    20.81Gi ±   1%    185.84Gi ±  2%   +792.93% (p=0.000 n=10)
ReLU6Float32s/1048576-32                 19.85Gi ±  12%    183.91Gi ±  5%   +826.50% (p=0.000 n=10)
SignFloat32s/64-32                       19.58Gi ±   2%     88.59Gi ±  3%   +352.42% (p=0.000 n=10)
SignFloat32s/4096-32                     19.39Gi ±   3%    166.31Gi ±  3%   +757.80% (p=0.000 n=10)
SignFloat32s/1048576-32                  20.04Gi ±   2%    131.27Gi ±  4%   +555.19% (p=0.000 n=10)
geomean                                  24.45Gi            118.9Gi         +386.35%
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
