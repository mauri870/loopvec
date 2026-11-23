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
| `for i := range dst { dst[i] = 7 }`, `dst[i] = math.NaN()`, `math.Inf(1)` | fill (broadcast non-zero literal, `math.NaN`, `math.Inf`) |
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
| `for i := range x { if x[i] < 0 { x[i] = 0 } }`, `if c { dst[i] = u } else if d { dst[i] = v } else { dst[i] = w }` | compare, then select (`IfElse`): ReLU, leaky ReLU, ReLU6, hard tanh, clamp, sign, step; also when the branches assign a temporary (`t := x[i]+3; if t < 0 { t = 0 } else if t > 6 { t = 6 }`: hard sigmoid, hard swish, cubic SiLU) |
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
ranging from 1.5x to 37x depending on the kernel

<details>
<summary>bench.txt</summary>

```
goos: linux
goarch: amd64
pkg: github.com/mauri870/loopvec/bench
cpu: AMD Ryzen 9 9950X3D 16-Core Processor          
                                │ /tmp/bench_scalar.txt │         /tmp/bench_simd.txt         │
                                │        sec/op         │    sec/op     vs base               │
AddFloat32s/64-32                         12.765n ±  0%   6.437n ±  5%  -49.57% (p=0.002 n=6)
AddFloat32s/4096-32                        821.3n ±  1%   200.1n ±  9%  -75.64% (p=0.002 n=6)
AddFloat32s/1048576-32                    209.84µ ±  7%   79.79µ ±  9%  -61.98% (p=0.002 n=6)
MulFloat32s/64-32                         12.870n ± 12%   6.903n ±  5%  -46.36% (p=0.002 n=6)
MulFloat32s/4096-32                        809.4n ±  4%   196.3n ±  5%  -75.75% (p=0.002 n=6)
MulFloat32s/1048576-32                    215.37µ ±  6%   80.96µ ± 11%  -62.41% (p=0.002 n=6)
ScalFloat32s/64-32                         9.692n ± 11%   3.476n ±  6%  -64.14% (p=0.002 n=6)
ScalFloat32s/4096-32                       728.2n ±  1%   125.0n ±  6%  -82.84% (p=0.002 n=6)
ScalFloat32s/1048576-32                   189.65µ ±  4%   33.03µ ±  5%  -82.58% (p=0.002 n=6)
MixFloat32s/64-32                         23.205n ±  3%   7.815n ±  3%  -66.32% (p=0.002 n=6)
MixFloat32s/4096-32                       1463.5n ±  1%   200.5n ± 11%  -86.30% (p=0.002 n=6)
MixFloat32s/1048576-32                    375.85µ ±  5%   85.43µ ±  7%  -77.27% (p=0.002 n=6)
AxpyFloat32s/64-32                        12.840n ±  2%   7.502n ±  3%  -41.57% (p=0.002 n=6)
AxpyFloat32s/4096-32                       785.3n ±  1%   205.4n ±  6%  -73.84% (p=0.002 n=6)
AxpyFloat32s/1048576-32                   209.76µ ±  2%   80.99µ ± 11%  -61.39% (p=0.002 n=6)
NegFloat32s/64-32                         14.345n ± 30%   4.899n ±  3%  -65.85% (p=0.002 n=6)
NegFloat32s/4096-32                        809.1n ± 48%   168.6n ± 12%  -79.16% (p=0.002 n=6)
NegFloat32s/1048576-32                    229.51µ ± 67%   58.02µ ±  3%  -74.72% (p=0.002 n=6)
DivFloat32s/64-32                         28.050n ±  0%   6.508n ±  1%  -76.80% (p=0.002 n=6)
DivFloat32s/4096-32                       1796.5n ±  0%   191.4n ±  5%  -89.35% (p=0.002 n=6)
DivFloat32s/1048576-32                    463.08µ ±  4%   81.93µ ± 10%  -82.31% (p=0.002 n=6)
DaxpyFloat32s/64-32                       13.770n ± 12%   6.383n ±  5%  -53.65% (p=0.002 n=6)
DaxpyFloat32s/4096-32                      901.9n ± 10%   198.6n ±  4%  -77.98% (p=0.002 n=6)
DaxpyFloat32s/1048576-32                  217.30µ ± 13%   62.29µ ±  6%  -71.33% (p=0.002 n=6)
FillFloat32s/64-32                         9.427n ± 23%   2.678n ±  3%  -71.60% (p=0.002 n=6)
FillFloat32s/4096-32                      733.90n ±  2%   85.26n ±  5%  -88.38% (p=0.002 n=6)
FillFloat32s/1048576-32                   184.21µ ±  2%   29.94µ ±  4%  -83.75% (p=0.002 n=6)
FillUint8s/64-32                           7.825n ±  6%   1.969n ± 10%  -74.84% (p=0.002 n=6)
FillUint8s/4096-32                        733.20n ±  2%   20.34n ± 23%  -97.23% (p=0.002 n=6)
FillUint8s/1048576-32                    184.659µ ±  1%   5.735µ ±  2%  -96.89% (p=0.002 n=6)
ReverseIncFloat32s/64-32                  12.785n ±  5%   4.842n ±  5%  -62.13% (p=0.002 n=6)
ReverseIncFloat32s/4096-32                 736.7n ±  0%   148.7n ±  2%  -79.82% (p=0.002 n=6)
ReverseIncFloat32s/1048576-32             194.95µ ±  5%   56.95µ ±  5%  -70.79% (p=0.002 n=6)
CopyFloat32s/64-32                        11.825n ±  1%   4.096n ±  2%  -65.37% (p=0.002 n=6)
CopyFloat32s/4096-32                      737.30n ±  1%   71.45n ±  4%  -90.31% (p=0.002 n=6)
CopyFloat32s/1048576-32                   187.56µ ±  2%   60.36µ ±  7%  -67.82% (p=0.002 n=6)
AndNotUint64s/64-32                       17.955n ±  1%   9.078n ±  3%  -49.44% (p=0.002 n=6)
AndNotUint64s/4096-32                     1193.0n ± 29%   362.0n ±  3%  -69.66% (p=0.002 n=6)
AndNotUint64s/1048576-32                   314.2µ ± 38%   206.5µ ± 13%  -34.29% (p=0.002 n=6)
NormalizeFloat32s/64-32                   28.180n ±  1%   6.301n ±  3%  -77.64% (p=0.002 n=6)
NormalizeFloat32s/4096-32                 1811.0n ±  1%   227.8n ±  2%  -87.42% (p=0.002 n=6)
NormalizeFloat32s/1048576-32              465.70µ ±  8%   65.57µ ± 23%  -85.92% (p=0.002 n=6)
ChainInt32s/64-32                          30.39n ±  5%   13.61n ±  2%  -55.22% (p=0.002 n=6)
ChainInt32s/4096-32                       2063.5n ± 10%   343.2n ±  8%  -83.37% (p=0.002 n=6)
ChainInt32s/1048576-32                     472.3µ ± 33%   112.4µ ± 23%  -76.20% (p=0.002 n=6)
WideInt32s/64-32                           45.11n ±  1%   10.30n ±  2%  -77.17% (p=0.002 n=6)
WideInt32s/4096-32                        2905.5n ±  1%   312.0n ±  7%  -89.26% (p=0.002 n=6)
WideInt32s/1048576-32                      736.9µ ±  5%   123.7µ ± 15%  -83.21% (p=0.002 n=6)
SumInt32s/64-32                           12.175n ±  3%   7.580n ±  3%  -37.74% (p=0.002 n=6)
SumInt32s/4096-32                          767.9n ±  4%   115.3n ±  1%  -84.99% (p=0.002 n=6)
SumInt32s/1048576-32                      189.10µ ±  3%   30.45µ ± 11%  -83.90% (p=0.002 n=6)
DotInt32s/64-32                           13.980n ±  1%   8.675n ±  2%  -37.95% (p=0.002 n=6)
DotInt32s/4096-32                          925.2n ± 26%   227.5n ±  8%  -75.41% (p=0.002 n=6)
DotInt32s/1048576-32                      237.66µ ± 28%   60.75µ ±  5%  -74.44% (p=0.002 n=6)
SumFloat32s/64-32                          10.68n ± 28%   12.74n ±  3%        ~ (p=0.065 n=6)
SumFloat32s/4096-32                       1413.0n ±  1%   106.6n ±  3%  -92.46% (p=0.002 n=6)
SumFloat32s/1048576-32                    368.95µ ±  1%   30.52µ ± 16%  -91.73% (p=0.002 n=6)
DotFloat32s/64-32                          15.86n ±  1%   13.48n ±  1%  -15.01% (p=0.002 n=6)
DotFloat32s/4096-32                       1438.5n ±  1%   181.8n ±  2%  -87.36% (p=0.002 n=6)
DotFloat32s/1048576-32                    374.60µ ±  1%   62.47µ ± 28%  -83.32% (p=0.002 n=6)
DotFloat64s/64-32                          13.89n ± 15%   12.47n ±  3%        ~ (p=0.240 n=6)
DotFloat64s/4096-32                       1414.0n ±  2%   362.9n ±  2%  -74.34% (p=0.002 n=6)
DotFloat64s/1048576-32                     369.6µ ±  3%   120.5µ ± 18%  -67.41% (p=0.002 n=6)
AbsFloat32s/64-32                         24.620n ±  6%   4.875n ±  2%  -80.20% (p=0.002 n=6)
AbsFloat32s/4096-32                       1461.0n ± 12%   179.2n ±  6%  -87.74% (p=0.002 n=6)
AbsFloat32s/1048576-32                    381.91µ ±  2%   58.01µ ±  5%  -84.81% (p=0.002 n=6)
SqrtFloat32s/64-32                        50.465n ±  1%   4.595n ±  4%  -90.89% (p=0.002 n=6)
SqrtFloat32s/4096-32                      3231.5n ±  2%   219.1n ±  4%  -93.22% (p=0.002 n=6)
SqrtFloat32s/1048576-32                   845.01µ ±  1%   60.24µ ±  2%  -92.87% (p=0.002 n=6)
ClampInt32s/64-32                         28.450n ±  1%   5.528n ±  4%  -80.57% (p=0.002 n=6)
ClampInt32s/4096-32                       1820.0n ±  1%   159.3n ± 11%  -91.24% (p=0.002 n=6)
ClampInt32s/1048576-32                    465.04µ ±  5%   57.57µ ±  6%  -87.62% (p=0.002 n=6)
ShrInt32s/64-32                           13.320n ±  5%   4.649n ±  3%  -65.09% (p=0.002 n=6)
ShrInt32s/4096-32                          847.4n ±  2%   150.6n ±  1%  -82.23% (p=0.002 n=6)
ShrInt32s/1048576-32                      211.64µ ±  2%   60.56µ ±  3%  -71.39% (p=0.002 n=6)
ShlUint32s/64-32                          12.760n ±  1%   4.915n ±  3%  -61.48% (p=0.002 n=6)
ShlUint32s/4096-32                         832.6n ±  2%   152.2n ±  3%  -81.71% (p=0.002 n=6)
ShlUint32s/1048576-32                     209.74µ ±  4%   58.84µ ±  5%  -71.95% (p=0.002 n=6)
DiffInt32s/64-32                          16.090n ± 18%   7.588n ±  2%  -52.84% (p=0.002 n=6)
DiffInt32s/4096-32                         928.8n ± 12%   286.0n ±  2%  -69.21% (p=0.002 n=6)
DiffInt32s/1048576-32                     234.48µ ± 10%   74.82µ ±  5%  -68.09% (p=0.002 n=6)
SmoothInt32s/64-32                        21.525n ±  2%   9.298n ±  3%  -56.80% (p=0.002 n=6)
SmoothInt32s/4096-32                      1742.5n ±  3%   427.0n ± 23%  -75.49% (p=0.002 n=6)
SmoothInt32s/1048576-32                    462.0µ ±  3%   111.2µ ±  2%  -75.94% (p=0.002 n=6)
ShiftInt32s/64-32                         15.575n ± 18%   5.753n ±  5%  -63.06% (p=0.002 n=6)
ShiftInt32s/4096-32                       1056.0n ± 11%   238.0n ±  2%  -77.46% (p=0.002 n=6)
ShiftInt32s/1048576-32                    267.26µ ± 15%   61.91µ ±  3%  -76.84% (p=0.002 n=6)
AccumRowInt32s/64-32                      16.040n ±  4%   6.025n ±  2%  -62.44% (p=0.002 n=6)
AccumRowInt32s/4096-32                    1036.0n ±  6%   209.2n ±  3%  -79.81% (p=0.002 n=6)
AccumRowInt32s/1048576-32                 274.83µ ±  7%   61.81µ ±  5%  -77.51% (p=0.002 n=6)
MinUint16s/64-32                           24.36n ±  0%   29.87n ±  1%  +22.62% (p=0.002 n=6)
MinUint16s/4096-32                       2144.50n ±  0%   65.87n ±  8%  -96.93% (p=0.002 n=6)
MinUint16s/1048576-32                     551.25µ ±  3%   14.77µ ± 12%  -97.32% (p=0.002 n=6)
ProductFloat32s/64-32                     21.140n ±  1%   6.855n ±  2%  -67.57% (p=0.002 n=6)
ProductFloat32s/4096-32                   2153.5n ±  0%   110.8n ± 15%  -94.86% (p=0.002 n=6)
ProductFloat32s/1048576-32                551.46µ ±  1%   31.19µ ±  8%  -94.34% (p=0.002 n=6)
ShiftAddInt32s/64-32                      13.010n ± 16%   7.635n ±  1%  -41.31% (p=0.002 n=6)
ShiftAddInt32s/4096-32                     860.2n ±  6%   266.0n ±  2%  -69.08% (p=0.002 n=6)
ShiftAddInt32s/1048576-32                 217.54µ ±  3%   69.59µ ± 13%  -68.01% (p=0.002 n=6)
ReadAfterStoreInt32s/64-32                 31.59n ±  2%   19.02n ±  3%  -39.78% (p=0.002 n=6)
ReadAfterStoreInt32s/4096-32              2354.0n ±  8%   571.0n ±  4%  -75.74% (p=0.002 n=6)
ReadAfterStoreInt32s/1048576-32            648.8µ ±  7%   155.4µ ±  7%  -76.05% (p=0.002 n=6)
ReLUFloat32s/64-32                        13.590n ± 17%   3.646n ±  3%  -73.17% (p=0.002 n=6)
ReLUFloat32s/4096-32                       799.3n ±  6%   143.9n ±  2%  -82.00% (p=0.002 n=6)
ReLUFloat32s/1048576-32                   202.67µ ±  9%   36.19µ ± 14%  -82.14% (p=0.002 n=6)
LeakyReLUFloat32s/64-32                   21.835n ±  5%   5.217n ±  4%  -76.11% (p=0.002 n=6)
LeakyReLUFloat32s/4096-32                 1470.0n ±  8%   156.8n ±  2%  -89.33% (p=0.002 n=6)
LeakyReLUFloat32s/1048576-32              386.79µ ± 28%   58.20µ ± 10%  -84.95% (p=0.002 n=6)
ReLU6Float32s/64-32                       22.865n ±  2%   4.940n ±  3%  -78.39% (p=0.002 n=6)
ReLU6Float32s/4096-32                     1449.0n ±  0%   232.6n ±  0%  -83.95% (p=0.002 n=6)
ReLU6Float32s/1048576-32                  369.71µ ± 10%   57.91µ ±  3%  -84.34% (p=0.002 n=6)
SignFloat32s/64-32                        23.095n ± 12%   5.125n ±  2%  -77.81% (p=0.002 n=6)
SignFloat32s/4096-32                      1572.5n ± 10%   169.3n ±  3%  -89.23% (p=0.002 n=6)
SignFloat32s/1048576-32                   398.02µ ±  4%   58.09µ ±  6%  -85.40% (p=0.002 n=6)
FillNaNFloat64s/64-32                     11.735n ±  1%   3.757n ±  3%  -67.98% (p=0.002 n=6)
FillNaNFloat64s/4096-32                    737.6n ±  3%   158.8n ±  8%  -78.47% (p=0.002 n=6)
FillNaNFloat64s/1048576-32                187.53µ ±  3%   58.75µ ± 11%  -68.67% (p=0.002 n=6)
ScaleFieldFloat32s/64-32                  12.645n ±  3%   3.579n ±  3%  -71.69% (p=0.002 n=6)
ScaleFieldFloat32s/4096-32                 788.9n ± 12%   160.6n ± 18%  -79.64% (p=0.002 n=6)
ScaleFieldFloat32s/1048576-32             187.84µ ±  3%   40.40µ ±  4%  -78.49% (p=0.002 n=6)
AddFieldFloat32s/64-32                    25.905n ±  9%   6.632n ±  1%  -74.40% (p=0.002 n=6)
AddFieldFloat32s/4096-32                  1446.5n ±  3%   327.5n ±  1%  -77.36% (p=0.002 n=6)
AddFieldFloat32s/1048576-32               383.91µ ±  6%   84.91µ ±  2%  -77.88% (p=0.002 n=6)
HardSwishFloat32s/64-32                   31.590n ±  7%   6.021n ±  3%  -80.94% (p=0.002 n=6)
HardSwishFloat32s/4096-32                 2019.0n ±  2%   215.8n ± 10%  -89.31% (p=0.002 n=6)
HardSwishFloat32s/1048576-32              511.01µ ±  2%   61.40µ ±  4%  -87.98% (p=0.002 n=6)
HardSigmoidFloat32s/64-32                 30.045n ±  1%   5.623n ±  2%  -81.28% (p=0.002 n=6)
HardSigmoidFloat32s/4096-32               1919.5n ±  1%   202.5n ±  1%  -89.45% (p=0.002 n=6)
HardSigmoidFloat32s/1048576-32            495.99µ ±  2%   61.21µ ±  1%  -87.66% (p=0.002 n=6)
geomean                                    1.892µ         409.6n        -78.35%

                                │ /tmp/bench_scalar.txt │           /tmp/bench_simd.txt            │
                                │          B/s          │       B/s        vs base                 │
AddFloat32s/64-32                         56.04Gi ±  0%    111.13Gi ±  5%    +98.29% (p=0.002 n=6)
AddFloat32s/4096-32                       55.73Gi ±  1%    228.86Gi ± 10%   +310.64% (p=0.002 n=6)
AddFloat32s/1048576-32                    55.86Gi ±  6%    146.88Gi ±  8%   +162.93% (p=0.002 n=6)
MulFloat32s/64-32                         55.56Gi ± 11%    103.61Gi ±  6%    +86.47% (p=0.002 n=6)
MulFloat32s/4096-32                       56.56Gi ±  4%    233.29Gi ±  4%   +312.47% (p=0.002 n=6)
MulFloat32s/1048576-32                    54.42Gi ±  6%    144.77Gi ± 10%   +166.04% (p=0.002 n=6)
ScalFloat32s/64-32                        49.20Gi ± 10%    137.22Gi ±  5%   +178.90% (p=0.002 n=6)
ScalFloat32s/4096-32                      41.91Gi ±  1%    244.28Gi ±  6%   +482.91% (p=0.002 n=6)
ScalFloat32s/1048576-32                   41.19Gi ±  4%    236.59Gi ±  5%   +474.33% (p=0.002 n=6)
MixFloat32s/64-32                         30.82Gi ±  3%     91.52Gi ±  3%   +196.96% (p=0.002 n=6)
MixFloat32s/4096-32                       31.27Gi ±  1%    228.37Gi ± 10%   +630.21% (p=0.002 n=6)
MixFloat32s/1048576-32                    31.18Gi ±  5%    137.31Gi ±  7%   +340.40% (p=0.002 n=6)
AxpyFloat32s/64-32                        55.69Gi ±  2%     95.35Gi ±  3%    +71.20% (p=0.002 n=6)
AxpyFloat32s/4096-32                      58.29Gi ±  1%    222.91Gi ±  6%   +282.41% (p=0.002 n=6)
AxpyFloat32s/1048576-32                   55.87Gi ±  2%    144.69Gi ± 10%   +158.99% (p=0.002 n=6)
NegFloat32s/64-32                         33.25Gi ± 23%     97.32Gi ±  3%   +192.75% (p=0.002 n=6)
NegFloat32s/4096-32                       37.72Gi ± 33%    181.02Gi ± 11%   +379.88% (p=0.002 n=6)
NegFloat32s/1048576-32                    34.13Gi ± 40%    134.65Gi ±  4%   +294.46% (p=0.002 n=6)
DivFloat32s/64-32                         25.50Gi ±  0%    109.90Gi ±  1%   +331.00% (p=0.002 n=6)
DivFloat32s/4096-32                       25.48Gi ±  0%    239.24Gi ±  5%   +838.85% (p=0.002 n=6)
DivFloat32s/1048576-32                    25.31Gi ±  4%    143.03Gi ±  9%   +465.21% (p=0.002 n=6)
DaxpyFloat32s/64-32                       52.01Gi ± 12%    112.06Gi ±  4%   +115.44% (p=0.002 n=6)
DaxpyFloat32s/4096-32                     50.76Gi ±  9%    230.57Gi ±  4%   +354.21% (p=0.002 n=6)
DaxpyFloat32s/1048576-32                  53.94Gi ± 12%    188.12Gi ±  5%   +248.74% (p=0.002 n=6)
FillFloat32s/64-32                        26.04Gi ± 21%     89.04Gi ±  4%   +241.97% (p=0.002 n=6)
FillFloat32s/4096-32                      20.79Gi ±  2%    179.03Gi ±  5%   +761.08% (p=0.002 n=6)
FillFloat32s/1048576-32                   21.21Gi ±  2%    130.47Gi ±  4%   +515.26% (p=0.002 n=6)
FillUint8s/64-32                          7.618Gi ±  6%    30.279Gi ±  9%   +297.48% (p=0.002 n=6)
FillUint8s/4096-32                        5.203Gi ±  2%   187.610Gi ± 18%  +3505.90% (p=0.002 n=6)
FillUint8s/1048576-32                     5.288Gi ±  1%   170.291Gi ±  2%  +3120.05% (p=0.002 n=6)
ReverseIncFloat32s/64-32                  37.29Gi ±  5%     98.49Gi ±  4%   +164.10% (p=0.002 n=6)
ReverseIncFloat32s/4096-32                41.42Gi ±  0%    205.24Gi ±  2%   +395.49% (p=0.002 n=6)
ReverseIncFloat32s/1048576-32             40.08Gi ±  5%    137.20Gi ±  5%   +242.37% (p=0.002 n=6)
CopyFloat32s/64-32                        40.32Gi ±  1%    116.44Gi ±  2%   +188.78% (p=0.002 n=6)
CopyFloat32s/4096-32                      41.39Gi ±  1%    427.12Gi ±  4%   +931.91% (p=0.002 n=6)
CopyFloat32s/1048576-32                   41.65Gi ±  2%    129.43Gi ±  8%   +210.73% (p=0.002 n=6)
AndNotUint64s/64-32                       79.66Gi ±  1%    157.59Gi ±  3%    +97.83% (p=0.002 n=6)
AndNotUint64s/4096-32                     76.73Gi ± 22%    252.93Gi ±  3%   +229.62% (p=0.002 n=6)
AndNotUint64s/1048576-32                  74.59Gi ± 27%    113.51Gi ± 15%    +52.19% (p=0.002 n=6)
NormalizeFloat32s/64-32                   33.84Gi ±  1%    151.34Gi ±  3%   +347.27% (p=0.002 n=6)
NormalizeFloat32s/4096-32                 33.70Gi ±  1%    268.00Gi ±  2%   +695.33% (p=0.002 n=6)
NormalizeFloat32s/1048576-32              33.55Gi ±  7%    238.28Gi ± 19%   +610.19% (p=0.002 n=6)
ChainInt32s/64-32                         47.07Gi ±  5%    105.13Gi ±  2%   +123.33% (p=0.002 n=6)
ChainInt32s/4096-32                       44.57Gi ± 10%    266.80Gi ±  7%   +498.62% (p=0.002 n=6)
ChainInt32s/1048576-32                    49.64Gi ± 25%    209.09Gi ± 19%   +321.21% (p=0.002 n=6)
WideInt32s/64-32                          21.14Gi ±  1%     92.59Gi ±  2%   +337.93% (p=0.002 n=6)
WideInt32s/4096-32                        21.01Gi ±  1%    195.60Gi ±  6%   +831.12% (p=0.002 n=6)
WideInt32s/1048576-32                     21.20Gi ±  5%    126.28Gi ± 18%   +495.60% (p=0.002 n=6)
SumInt32s/64-32                           19.58Gi ±  3%     31.45Gi ±  3%    +60.62% (p=0.002 n=6)
SumInt32s/4096-32                         19.87Gi ±  4%    132.36Gi ±  1%   +566.06% (p=0.002 n=6)
SumInt32s/1048576-32                      20.66Gi ±  3%    128.29Gi ± 13%   +520.95% (p=0.002 n=6)
DotInt32s/64-32                           34.11Gi ±  1%     54.97Gi ±  2%    +61.17% (p=0.002 n=6)
DotInt32s/4096-32                         32.99Gi ± 21%    134.21Gi ±  9%   +306.83% (p=0.002 n=6)
DotInt32s/1048576-32                      32.87Gi ± 22%    128.61Gi ±  5%   +291.22% (p=0.002 n=6)
SumFloat32s/64-32                         22.32Gi ± 22%     18.71Gi ±  3%          ~ (p=0.065 n=6)
SumFloat32s/4096-32                       10.80Gi ±  1%    143.08Gi ±  3%  +1224.56% (p=0.002 n=6)
SumFloat32s/1048576-32                    10.59Gi ±  1%    128.00Gi ± 19%  +1108.94% (p=0.002 n=6)
DotFloat32s/64-32                         30.06Gi ±  1%     35.38Gi ±  1%    +17.68% (p=0.002 n=6)
DotFloat32s/4096-32                       21.21Gi ±  1%    167.86Gi ±  2%   +691.40% (p=0.002 n=6)
DotFloat32s/1048576-32                    20.86Gi ±  1%    125.06Gi ± 38%   +499.65% (p=0.002 n=6)
DotFloat64s/64-32                         69.22Gi ± 17%     76.50Gi ±  3%          ~ (p=0.240 n=6)
DotFloat64s/4096-32                       43.16Gi ±  2%    168.24Gi ±  2%   +289.81% (p=0.002 n=6)
DotFloat64s/1048576-32                    42.28Gi ±  2%    129.71Gi ± 22%   +206.81% (p=0.002 n=6)
AbsFloat32s/64-32                         19.37Gi ±  6%     97.80Gi ±  2%   +404.94% (p=0.002 n=6)
AbsFloat32s/4096-32                       20.89Gi ± 11%    170.35Gi ±  6%   +715.41% (p=0.002 n=6)
AbsFloat32s/1048576-32                    20.46Gi ±  2%    134.69Gi ±  5%   +558.37% (p=0.002 n=6)
SqrtFloat32s/64-32                        9.449Gi ±  1%   103.770Gi ±  4%   +998.21% (p=0.002 n=6)
SqrtFloat32s/4096-32                      9.443Gi ±  2%   139.262Gi ±  4%  +1374.75% (p=0.002 n=6)
SqrtFloat32s/1048576-32                   9.246Gi ±  1%   129.693Gi ±  2%  +1302.76% (p=0.002 n=6)
ClampInt32s/64-32                         16.76Gi ±  1%     86.26Gi ±  4%   +414.65% (p=0.002 n=6)
ClampInt32s/4096-32                       16.77Gi ±  1%    191.46Gi ± 10%  +1041.90% (p=0.002 n=6)
ClampInt32s/1048576-32                    16.80Gi ±  5%    135.77Gi ±  6%   +708.17% (p=0.002 n=6)
ShrInt32s/64-32                           35.80Gi ±  5%    102.56Gi ±  3%   +186.50% (p=0.002 n=6)
ShrInt32s/4096-32                         36.01Gi ±  3%    202.72Gi ±  1%   +462.88% (p=0.002 n=6)
ShrInt32s/1048576-32                      36.91Gi ±  2%    129.01Gi ±  3%   +249.49% (p=0.002 n=6)
ShlUint32s/64-32                          37.37Gi ±  1%     97.01Gi ±  3%   +159.62% (p=0.002 n=6)
ShlUint32s/4096-32                        36.65Gi ±  2%    200.47Gi ±  2%   +446.92% (p=0.002 n=6)
ShlUint32s/1048576-32                     37.25Gi ±  4%    132.80Gi ±  6%   +256.52% (p=0.002 n=6)
DiffInt32s/64-32                          29.69Gi ± 16%     62.84Gi ±  2%   +111.62% (p=0.002 n=6)
DiffInt32s/4096-32                        32.87Gi ± 11%    106.71Gi ±  2%   +224.61% (p=0.002 n=6)
DiffInt32s/1048576-32                     33.32Gi ±  9%    104.49Gi ±  5%   +213.61% (p=0.002 n=6)
SmoothInt32s/64-32                        22.15Gi ±  2%     51.28Gi ±  3%   +131.51% (p=0.002 n=6)
SmoothInt32s/4096-32                      17.51Gi ±  3%     71.47Gi ± 19%   +308.16% (p=0.002 n=6)
SmoothInt32s/1048576-32                   16.91Gi ±  3%     70.28Gi ±  2%   +315.62% (p=0.002 n=6)
ShiftInt32s/64-32                         30.62Gi ± 15%     82.92Gi ±  5%   +170.84% (p=0.002 n=6)
ShiftInt32s/4096-32                       28.92Gi ± 10%    128.25Gi ±  2%   +343.51% (p=0.002 n=6)
ShiftInt32s/1048576-32                    29.25Gi ± 13%    126.20Gi ±  2%   +331.43% (p=0.002 n=6)
AccumRowInt32s/64-32                      44.59Gi ±  3%    118.71Gi ±  2%   +166.22% (p=0.002 n=6)
AccumRowInt32s/4096-32                    44.18Gi ±  5%    218.78Gi ±  3%   +395.16% (p=0.002 n=6)
AccumRowInt32s/1048576-32                 42.64Gi ±  7%    189.60Gi ±  5%   +344.65% (p=0.002 n=6)
MinUint16s/64-32                          4.895Gi ±  0%     3.992Gi ±  1%    -18.45% (p=0.002 n=6)
MinUint16s/4096-32                        3.558Gi ±  0%   115.836Gi ±  7%  +3155.64% (p=0.002 n=6)
MinUint16s/1048576-32                     3.543Gi ±  3%   132.239Gi ± 14%  +3632.31% (p=0.002 n=6)
ProductFloat32s/64-32                     11.28Gi ±  1%     34.78Gi ±  2%   +208.39% (p=0.002 n=6)
ProductFloat32s/4096-32                   7.086Gi ±  0%   137.827Gi ± 13%  +1845.11% (p=0.002 n=6)
ProductFloat32s/1048576-32                7.083Gi ±  1%   125.265Gi ±  7%  +1668.41% (p=0.002 n=6)
ShiftAddInt32s/64-32                      54.98Gi ± 14%     93.69Gi ±  1%    +70.40% (p=0.002 n=6)
ShiftAddInt32s/4096-32                    53.21Gi ±  6%    172.10Gi ±  2%   +223.40% (p=0.002 n=6)
ShiftAddInt32s/1048576-32                 53.87Gi ±  3%    168.43Gi ± 11%   +212.67% (p=0.002 n=6)
ReadAfterStoreInt32s/64-32                22.64Gi ±  2%     37.60Gi ±  3%    +66.06% (p=0.002 n=6)
ReadAfterStoreInt32s/4096-32              19.45Gi ±  8%     80.17Gi ±  4%   +312.21% (p=0.002 n=6)
ReadAfterStoreInt32s/1048576-32           18.06Gi ±  7%     75.43Gi ±  6%   +317.60% (p=0.002 n=6)
ReLUFloat32s/64-32                        35.09Gi ± 15%    130.78Gi ±  3%   +272.71% (p=0.002 n=6)
ReLUFloat32s/4096-32                      38.18Gi ±  6%    212.11Gi ±  2%   +455.55% (p=0.002 n=6)
ReLUFloat32s/1048576-32                   38.55Gi ±  9%    215.86Gi ± 12%   +459.91% (p=0.002 n=6)
LeakyReLUFloat32s/64-32                   21.84Gi ±  5%     91.41Gi ±  4%   +318.58% (p=0.002 n=6)
LeakyReLUFloat32s/4096-32                 20.77Gi ±  8%    194.60Gi ±  2%   +837.11% (p=0.002 n=6)
LeakyReLUFloat32s/1048576-32              20.20Gi ± 22%    134.32Gi ±  9%   +564.83% (p=0.002 n=6)
ReLU6Float32s/64-32                       20.85Gi ±  2%     96.53Gi ±  3%   +362.90% (p=0.002 n=6)
ReLU6Float32s/4096-32                     21.06Gi ±  0%    131.21Gi ±  0%   +523.15% (p=0.002 n=6)
ReLU6Float32s/1048576-32                  21.13Gi ±  9%    134.92Gi ±  2%   +538.46% (p=0.002 n=6)
SignFloat32s/64-32                        20.67Gi ± 13%     93.04Gi ±  2%   +350.14% (p=0.002 n=6)
SignFloat32s/4096-32                      19.41Gi ± 11%    180.27Gi ±  3%   +828.73% (p=0.002 n=6)
SignFloat32s/1048576-32                   19.64Gi ±  4%    134.59Gi ±  6%   +585.39% (p=0.002 n=6)
FillNaNFloat64s/64-32                     40.64Gi ±  1%    126.94Gi ±  3%   +212.34% (p=0.002 n=6)
FillNaNFloat64s/4096-32                   41.37Gi ±  3%    192.24Gi ±  8%   +364.67% (p=0.002 n=6)
FillNaNFloat64s/1048576-32                41.66Gi ±  3%    133.01Gi ± 12%   +219.27% (p=0.002 n=6)
ScaleFieldFloat32s/64-32                  37.70Gi ±  3%    133.23Gi ±  3%   +253.41% (p=0.002 n=6)
ScaleFieldFloat32s/4096-32                38.69Gi ± 10%    190.12Gi ± 15%   +391.45% (p=0.002 n=6)
ScaleFieldFloat32s/1048576-32             41.59Gi ±  3%    193.38Gi ±  4%   +364.92% (p=0.002 n=6)
AddFieldFloat32s/64-32                    27.61Gi ±  9%    107.85Gi ±  1%   +290.56% (p=0.002 n=6)
AddFieldFloat32s/4096-32                  31.65Gi ±  3%    139.78Gi ±  1%   +341.67% (p=0.002 n=6)
AddFieldFloat32s/1048576-32               30.53Gi ±  6%    138.02Gi ±  2%   +352.15% (p=0.002 n=6)
HardSwishFloat32s/64-32                   15.09Gi ±  6%     79.19Gi ±  3%   +424.64% (p=0.002 n=6)
HardSwishFloat32s/4096-32                 15.11Gi ±  2%    141.47Gi ±  9%   +836.03% (p=0.002 n=6)
HardSwishFloat32s/1048576-32              15.29Gi ±  2%    127.24Gi ±  4%   +732.26% (p=0.002 n=6)
HardSigmoidFloat32s/64-32                 15.87Gi ±  1%     84.80Gi ±  1%   +434.29% (p=0.002 n=6)
HardSigmoidFloat32s/4096-32               15.90Gi ±  1%    150.73Gi ±  1%   +848.13% (p=0.002 n=6)
HardSigmoidFloat32s/1048576-32            15.75Gi ±  2%    127.64Gi ±  1%   +710.35% (p=0.002 n=6)
geomean                                   26.90Gi           124.2Gi         +361.66%
```
</details>

Filling a byte slice with a non-zero value (`FillUint8s`, 1M elements) is over 30x
faster, +3000% throughput.

Activation functions written as plain `if`/`else` loops (ReLU, leaky ReLU, ReLU6, sign;
`float32`, 4K to 1M elements) run 5 to 9x faster.

Running `loopvec -methods` on [gorgonia/tensor](https://github.com/gorgonia/tensor)
detects 464 vectorizable loops. The result is **~2.8x faster**
on an AMD Ryzen 9 9950X3D (AVX-512), geomean over all rows. The rewritten kernels are
2.8 to 14x faster at 4K and 1M elements and 1.3 to 6.6x at 64.

<details>
<summary>bench.txt</summary>

```
                       │    scalar     │            simd             │
                       │     sec/op     │    sec/op      vs base                │
AddVSF32/64-32            13.315n ±  4%    3.803n ±  5%   -71.44% (p=0.002 n=6)
AddVSF32/4096-32           760.2n ±  2%    126.7n ±  5%   -83.33% (p=0.002 n=6)
AddVSF32/1048576-32       192.82µ ±  4%    31.85µ ±  3%   -83.48% (p=0.002 n=6)
MulVSF32/64-32             9.081n ± 35%    3.909n ±  4%   -56.95% (p=0.002 n=6)
MulVSF32/4096-32           741.6n ±  2%    126.6n ±  2%   -82.93% (p=0.002 n=6)
MulVSF32/1048576-32       184.60µ ±  2%    32.80µ ±  4%   -82.23% (p=0.002 n=6)
AddVSF64/64-32            12.770n ±  1%    5.373n ±  4%   -57.93% (p=0.002 n=6)
AddVSF64/4096-32           735.0n ±  2%    237.1n ± 18%   -67.75% (p=0.002 n=6)
AddVSF64/1048576-32       201.06µ ±  1%    65.64µ ±  2%   -67.35% (p=0.002 n=6)
MulVSF64/64-32             7.357n ± 23%    5.480n ±  6%   -25.51% (p=0.002 n=6)
MulVSF64/4096-32           732.5n ±  2%    239.3n ±  3%   -67.33% (p=0.002 n=6)
MulVSF64/1048576-32       183.47µ ±  0%    64.68µ ±  5%   -64.75% (p=0.002 n=6)
AddSVF32/64-32            12.980n ±  2%    3.942n ±  4%   -69.63% (p=0.002 n=6)
AddSVF32/4096-32           747.3n ±  1%    124.4n ±  2%   -83.35% (p=0.002 n=6)
AddSVF32/1048576-32       192.05µ ±  2%    32.20µ ±  1%   -83.24% (p=0.002 n=6)
MinVSF32/64-32            23.085n ±  2%    3.993n ±  4%   -82.70% (p=0.002 n=6)
MinVSF32/4096-32          1452.5n ±  1%    143.5n ±  1%   -90.12% (p=0.002 n=6)
MinVSF32/1048576-32       368.04µ ±  1%    36.73µ ±  1%   -90.02% (p=0.002 n=6)
VecAddI32/64-32           12.715n ±  2%    5.679n ±  5%   -55.34% (p=0.002 n=6)
VecAddI32/4096-32          759.0n ±  1%    149.7n ±  2%   -80.28% (p=0.002 n=6)
VecAddI32/1048576-32      200.76µ ±  3%    62.12µ ±  4%   -69.06% (p=0.002 n=6)
VecMulI32/64-32           14.690n ±  1%    5.545n ±  7%   -62.25% (p=0.002 n=6)
VecMulI32/4096-32          924.9n ±  1%    149.2n ±  1%   -83.86% (p=0.002 n=6)
VecMulI32/1048576-32      241.14µ ±  3%    63.60µ ± 12%   -73.62% (p=0.002 n=6)
VecAddF32/64-32            13.06n ±  3%    23.55n ± 58%   +80.36% (p=0.002 n=6)
VecAddF32/4096-32          750.8n ±  3%   1554.0n ± 46%  +106.99% (p=0.002 n=6)
VecAddF32/1048576-32       196.5µ ±  3%    294.5µ ± 77%   +49.89% (p=0.002 n=6)
VecDivF32/64-32            28.20n ±  4%    52.94n ± 36%   +87.76% (p=0.002 n=6)
VecDivF32/4096-32          1.837µ ±  3%    4.073µ ± 33%  +121.78% (p=0.002 n=6)
VecDivF32/1048576-32       736.9µ ±  2%   1603.2µ ± 24%  +117.57% (p=0.002 n=6)
VecMaxF32/64-32           13.170n ±  4%    5.468n ±  7%   -58.48% (p=0.002 n=6)
VecMaxF32/4096-32          794.2n ±  1%    170.8n ±  1%   -78.50% (p=0.002 n=6)
VecMaxF32/1048576-32      201.07µ ±  3%    61.91µ ±  6%   -69.21% (p=0.002 n=6)
NegF32/64-32              12.890n ±  3%    4.012n ±  2%   -68.88% (p=0.002 n=6)
NegF32/4096-32             746.1n ±  2%    168.2n ± 10%   -77.45% (p=0.002 n=6)
NegF32/1048576-32         194.19µ ±  2%    40.45µ ± 10%   -79.17% (p=0.002 n=6)
AbsF32/64-32               13.66n ±  3%    13.71n ±  3%         ~ (p=0.669 n=6)
AbsF32/4096-32             883.6n ±  2%    884.1n ±  1%         ~ (p=0.784 n=6)
AbsF32/1048576-32          230.5µ ±  4%    227.6µ ±  1%         ~ (p=0.394 n=6)
SqrtF32/64-32              57.89n ±  1%    57.40n ±  1%    -0.85% (p=0.009 n=6)
SqrtF32/4096-32            3.505µ ±  1%    3.677µ ± 70%    +4.91% (p=0.041 n=6)
SqrtF32/1048576-32         866.8µ ±  2%   1202.4µ ± 75%         ~ (p=0.240 n=6)
InvF32/64-32              28.455n ±  2%    4.339n ±  8%   -84.75% (p=0.002 n=6)
InvF32/4096-32            1796.0n ±  1%    144.6n ±  1%   -91.95% (p=0.002 n=6)
InvF32/1048576-32         461.91µ ±  2%    36.05µ ± 11%   -92.19% (p=0.002 n=6)
SquareF32/64-32            9.954n ± 10%    3.887n ±  5%   -60.95% (p=0.002 n=6)
SquareF32/4096-32          728.1n ±  1%    150.6n ± 12%   -79.32% (p=0.002 n=6)
SquareF32/1048576-32      185.50µ ±  0%    37.50µ ± 25%   -79.78% (p=0.002 n=6)
ClampF32/64-32             36.01n ± 14%    34.43n ±  8%    -4.39% (p=0.041 n=6)
ClampF32/4096-32           2.198µ ±  4%    2.208µ ±  5%         ~ (p=0.818 n=6)
ClampF32/1048576-32        584.9µ ±  5%    560.4µ ±  2%    -4.20% (p=0.004 n=6)
GtSameF32/64-32           23.625n ± 14%    6.785n ±  1%   -71.28% (p=0.002 n=6)
GtSameF32/4096-32         1464.0n ±  1%    192.0n ±  2%   -86.89% (p=0.002 n=6)
GtSameF32/1048576-32      414.47µ ±  4%    61.54µ ±  4%   -85.15% (p=0.002 n=6)
EqSameF32/64-32           23.895n ±  3%    6.489n ±  5%   -72.84% (p=0.002 n=6)
EqSameF32/4096-32         1440.5n ± 13%    191.3n ±  2%   -86.72% (p=0.002 n=6)
EqSameF32/1048576-32      373.51µ ±  1%    62.34µ ±  2%   -83.31% (p=0.002 n=6)
SumI32/64-32              10.135n ± 12%    6.427n ±  5%   -36.59% (p=0.002 n=6)
SumI32/4096-32             724.2n ±  2%    193.8n ±  2%   -73.24% (p=0.002 n=6)
SumI32/1048576-32         184.66µ ±  0%    46.38µ ±  2%   -74.89% (p=0.002 n=6)
SliceMaxI32/64-32          25.73n ±  0%    25.79n ±  1%         ~ (p=0.288 n=6)
SliceMaxI32/4096-32        2.151µ ±  1%    2.152µ ±  1%         ~ (p=0.920 n=6)
SliceMaxI32/1048576-32     550.4µ ±  2%    550.4µ ±  1%         ~ (p=1.000 n=6)
SumF32/64-32               11.27n ± 18%    12.76n ±  2%         ~ (p=0.221 n=6)
SumF32/4096-32            1414.5n ±  3%    102.4n ±  2%   -92.76% (p=0.002 n=6)
SumF32/1048576-32         367.68µ ±  2%    30.07µ ±  4%   -91.82% (p=0.002 n=6)
geomean                    1.735µ          630.6n         -63.64%
```
</details>

Running `loopvec -methods -split` on [gonum](https://github.com/gonum/gonum)
rewrites 161 loops, or 222 with `-fp-reassoc`.

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
