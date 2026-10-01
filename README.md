# loopvec

`loopvec` analyzes Go source packages and rewrites element-wise loops to use Go's
experimental [simd](https://pkg.go.dev/simd) package for **automatic SIMD
vectorization**. Rewritten code requires `GOEXPERIMENT=simd` (Go 1.27+).

Go blog post: https://go.dev/blog/simd-experiment

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
| `for i := 0; i < 4; i++ { ... }` | constant limit |
| `for i := 1; i < len(s)-1; i++ { ... }`, `for i := lo; i < hi; i++ { ... }` | any start and limit that do not change in the loop |
| `for i := range dst { dst[i] = a[i+1] - a[i] }`, `work[j] += a[row*lda+j]` | read at an offset from the index (stencil, row of a 2-D slice); the slice read is not written in the loop |

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
(`_loopvecOverlap`, comparing `unsafe.SliceData` ranges) that falls back to
the original scalar loop whenever the operands' memory overlaps, regardless
of loop direction. It does not tell an exact alias from an offset one, so
`Add(x, x, y)` also takes the scalar loop.

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
AddFloat32s/64-32                       12.705n ± 15%   4.885n ±  3%  -61.55% (p=0.000 n=10)
AddFloat32s/4096-32                      816.4n ±  4%   166.0n ±  5%  -79.67% (p=0.000 n=10)
AddFloat32s/1048576-32                  216.16µ ±  1%   86.99µ ±  9%  -59.76% (p=0.000 n=10)
MulFloat32s/64-32                       12.725n ±  1%   5.633n ±  1%  -55.73% (p=0.000 n=10)
MulFloat32s/4096-32                      821.6n ±  4%   196.7n ±  5%  -76.06% (p=0.000 n=10)
MulFloat32s/1048576-32                  213.49µ ±  3%   86.40µ ±  8%  -59.53% (p=0.000 n=10)
ScalFloat32s/64-32                      10.635n ±  8%   3.547n ±  3%  -66.65% (p=0.000 n=10)
ScalFloat32s/4096-32                     741.1n ±  2%   144.9n ±  2%  -80.45% (p=0.000 n=10)
ScalFloat32s/1048576-32                 190.22µ ±  2%   36.65µ ±  2%  -80.73% (p=0.000 n=10)
MixFloat32s/64-32                       23.320n ±  2%   6.037n ±  4%  -74.11% (p=0.000 n=10)
MixFloat32s/4096-32                     1477.0n ±  1%   202.8n ±  3%  -86.27% (p=0.000 n=10)
MixFloat32s/1048576-32                  390.03µ ±  2%   83.60µ ± 10%  -78.57% (p=0.000 n=10)
AxpyFloat32s/64-32                      19.040n ±  1%   4.460n ±  3%  -76.58% (p=0.000 n=10)
AxpyFloat32s/4096-32                     803.0n ±  2%   158.8n ±  2%  -80.22% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 211.28µ ±  1%   91.60µ ± 13%  -56.65% (p=0.000 n=10)
NegFloat32s/64-32                       13.640n ± 14%   4.143n ±  2%  -69.63% (p=0.000 n=10)
NegFloat32s/4096-32                      813.6n ±  7%   158.4n ±  3%  -80.53% (p=0.000 n=10)
NegFloat32s/1048576-32                  204.91µ ±  3%   60.01µ ±  4%  -70.71% (p=0.000 n=10)
DivFloat32s/64-32                       28.250n ±  2%   5.079n ±  3%  -82.02% (p=0.000 n=10)
DivFloat32s/4096-32                     1795.0n ±  3%   168.1n ±  2%  -90.64% (p=0.000 n=10)
DivFloat32s/1048576-32                  464.81µ ±  2%   82.68µ ± 11%  -82.21% (p=0.000 n=10)
DaxpyFloat32s/64-32                     12.720n ±  4%   5.412n ±  6%  -57.46% (p=0.000 n=10)
DaxpyFloat32s/4096-32                    827.6n ±  3%   195.8n ±  1%  -76.35% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                207.52µ ±  6%   65.02µ ±  1%  -68.67% (p=0.000 n=10)
FillFloat32s/64-32                       8.292n ± 37%   3.240n ±  7%  -60.93% (p=0.000 n=10)
FillFloat32s/4096-32                     731.6n ±  3%   103.3n ±  2%  -85.88% (p=0.000 n=10)
FillFloat32s/1048576-32                 188.77µ ±  1%   29.87µ ±  1%  -84.18% (p=0.000 n=10)
FillUint8s/64-32                         7.966n ± 43%   2.153n ±  8%  -72.97% (p=0.000 n=10)
FillUint8s/4096-32                      739.65n ±  2%   25.71n ± 19%  -96.52% (p=0.000 n=10)
FillUint8s/1048576-32                  183.478µ ±  2%   6.073µ ±  2%  -96.69% (p=0.000 n=10)
ReverseIncFloat32s/64-32                13.020n ±  4%   3.851n ±  2%  -70.43% (p=0.000 n=10)
ReverseIncFloat32s/4096-32               750.0n ±  1%   123.3n ±  3%  -83.55% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           199.75µ ±  3%   59.16µ ±  5%  -70.38% (p=0.000 n=10)
CopyFloat32s/64-32                      11.620n ±  1%   3.255n ±  2%  -71.99% (p=0.000 n=10)
CopyFloat32s/4096-32                    754.90n ±  1%   73.17n ±  2%  -90.31% (p=0.000 n=10)
CopyFloat32s/1048576-32                 189.07µ ±  2%   56.96µ ±  3%  -69.87% (p=0.000 n=10)
AndNotUint64s/64-32                     18.000n ±  2%   8.805n ±  1%  -51.08% (p=0.000 n=10)
AndNotUint64s/4096-32                   1228.5n ± 25%   395.7n ±  3%  -67.79% (p=0.000 n=10)
AndNotUint64s/1048576-32                 308.9µ ±  1%   169.1µ ± 10%  -45.26% (p=0.000 n=10)
NormalizeFloat32s/64-32                 28.870n ±  2%   5.840n ±  2%  -79.77% (p=0.000 n=10)
NormalizeFloat32s/4096-32               1824.5n ±  1%   237.9n ±  1%  -86.96% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            466.49µ ±  3%   67.24µ ±  1%  -85.59% (p=0.000 n=10)
ChainInt32s/64-32                       29.425n ±  2%   7.296n ±  2%  -75.20% (p=0.000 n=10)
ChainInt32s/4096-32                     1910.5n ±  2%   268.5n ±  1%  -85.95% (p=0.000 n=10)
ChainInt32s/1048576-32                   479.6µ ±  3%   126.5µ ± 12%  -73.63% (p=0.000 n=10)
WideInt32s/64-32                        46.215n ±  2%   7.316n ±  2%  -84.17% (p=0.000 n=10)
WideInt32s/4096-32                      2956.0n ±  3%   270.2n ±  5%  -90.86% (p=0.000 n=10)
WideInt32s/1048576-32                    744.9µ ±  2%   125.1µ ±  6%  -83.20% (p=0.000 n=10)
SumInt32s/64-32                         12.455n ±  8%   7.546n ±  2%  -39.41% (p=0.000 n=10)
SumInt32s/4096-32                        755.6n ±  6%   197.8n ±  2%  -73.83% (p=0.000 n=10)
SumInt32s/1048576-32                    185.79µ ±  2%   47.42µ ±  1%  -74.48% (p=0.000 n=10)
DotInt32s/64-32                         14.145n ±  2%   9.403n ±  1%  -33.52% (p=0.000 n=10)
DotInt32s/4096-32                        939.6n ± 24%   207.1n ±  7%  -77.96% (p=0.000 n=10)
DotInt32s/1048576-32                    244.61µ ±  2%   62.90µ ±  5%  -74.29% (p=0.000 n=10)
SumFloat32s/64-32                        12.04n ± 30%   12.81n ±  2%        ~ (p=0.403 n=10)
SumFloat32s/4096-32                     1415.0n ±  3%   112.0n ±  4%  -92.08% (p=0.000 n=10)
SumFloat32s/1048576-32                  377.27µ ±  0%   31.40µ ±  2%  -91.68% (p=0.000 n=10)
DotFloat32s/64-32                        16.17n ±  2%   13.27n ±  2%  -17.93% (p=0.000 n=10)
DotFloat32s/4096-32                     1447.5n ±  2%   183.8n ±  2%  -87.31% (p=0.000 n=10)
DotFloat32s/1048576-32                  370.84µ ±  3%   63.95µ ±  4%  -82.76% (p=0.000 n=10)
DotFloat64s/64-32                        12.00n ± 13%   12.13n ±  2%        ~ (p=0.239 n=10)
DotFloat64s/4096-32                     1421.0n ±  2%   367.7n ±  3%  -74.12% (p=0.000 n=10)
DotFloat64s/1048576-32                   378.9µ ±  3%   118.6µ ± 13%  -68.69% (p=0.000 n=10)
AbsFloat32s/64-32                       25.305n ±  9%   4.142n ±  5%  -83.63% (p=0.000 n=10)
AbsFloat32s/4096-32                     1541.5n ±  6%   146.3n ±  2%  -90.51% (p=0.000 n=10)
AbsFloat32s/1048576-32                  381.27µ ±  2%   59.10µ ±  4%  -84.50% (p=0.000 n=10)
SqrtFloat32s/64-32                      51.710n ±  2%   3.621n ±  3%  -93.00% (p=0.000 n=10)
SqrtFloat32s/4096-32                    3273.0n ±  1%   211.4n ±  2%  -93.54% (p=0.000 n=10)
SqrtFloat32s/1048576-32                 849.90µ ±  3%   59.45µ ±  5%  -93.01% (p=0.000 n=10)
ClampInt32s/64-32                       29.175n ±  3%   4.553n ±  2%  -84.40% (p=0.000 n=10)
ClampInt32s/4096-32                     1862.0n ±  0%   128.8n ±  3%  -93.09% (p=0.000 n=10)
ClampInt32s/1048576-32                  468.52µ ±  2%   59.42µ ±  6%  -87.32% (p=0.000 n=10)
ShrInt32s/64-32                         12.965n ±  2%   4.252n ±  2%  -67.21% (p=0.000 n=10)
ShrInt32s/4096-32                        841.1n ± 11%   134.9n ±  2%  -83.97% (p=0.000 n=10)
ShrInt32s/1048576-32                    215.51µ ±  3%   58.01µ ±  7%  -73.08% (p=0.000 n=10)
ShlUint32s/64-32                        13.145n ±  3%   3.958n ±  4%  -69.89% (p=0.000 n=10)
ShlUint32s/4096-32                       848.9n ±  1%   134.9n ±  1%  -84.11% (p=0.000 n=10)
ShlUint32s/1048576-32                   214.90µ ±  2%   58.30µ ±  8%  -72.87% (p=0.000 n=10)
DiffInt32s/64-32                        15.425n ±  5%   7.470n ±  3%  -51.58% (p=0.000 n=10)
DiffInt32s/4096-32                       947.9n ±  3%   416.2n ±  2%  -56.08% (p=0.000 n=10)
DiffInt32s/1048576-32                    249.5µ ±  3%   108.6µ ±  2%  -56.46% (p=0.000 n=10)
SmoothInt32s/64-32                      22.045n ±  0%   8.759n ±  1%  -60.27% (p=0.000 n=10)
SmoothInt32s/4096-32                    1739.5n ±  2%   426.4n ± 13%  -75.49% (p=0.000 n=10)
SmoothInt32s/1048576-32                  470.7µ ±  0%   110.2µ ±  1%  -76.58% (p=0.000 n=10)
ShiftInt32s/64-32                       15.670n ±  5%   4.402n ±  4%  -71.91% (p=0.000 n=10)
ShiftInt32s/4096-32                     1042.0n ±  2%   190.1n ±  2%  -81.76% (p=0.000 n=10)
ShiftInt32s/1048576-32                  267.28µ ±  6%   63.78µ ±  2%  -76.14% (p=0.000 n=10)
AccumRowInt32s/64-32                    16.945n ±  5%   5.421n ±  2%  -68.01% (p=0.000 n=10)
AccumRowInt32s/4096-32                  1089.5n ±  8%   240.3n ±  1%  -77.94% (p=0.000 n=10)
AccumRowInt32s/1048576-32               276.33µ ±  3%   65.14µ ±  5%  -76.42% (p=0.000 n=10)
MinUint16s/64-32                         24.32n ±  2%   30.58n ±  0%  +25.77% (p=0.000 n=10)
MinUint16s/4096-32                     2145.00n ±  3%   67.24n ±  3%  -96.87% (p=0.000 n=10)
MinUint16s/1048576-32                   550.29µ ±  0%   15.46µ ±  3%  -97.19% (p=0.000 n=10)
ProductFloat32s/64-32                   21.080n ±  3%   8.177n ±  3%  -61.21% (p=0.000 n=10)
ProductFloat32s/4096-32                 2184.5n ±  3%   109.3n ±  1%  -95.00% (p=0.000 n=10)
ProductFloat32s/1048576-32              551.69µ ±  3%   31.92µ ±  2%  -94.21% (p=0.000 n=10)
geomean                                  1.851µ         391.3n        -78.86%

                              │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                              │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                       56.30Gi ± 13%    146.42Gi ±  3%   +160.10% (p=0.000 n=10)
AddFloat32s/4096-32                     56.08Gi ±  4%    275.84Gi ±  5%   +391.90% (p=0.000 n=10)
AddFloat32s/1048576-32                  54.21Gi ±  1%    135.04Gi ±  9%   +149.10% (p=0.000 n=10)
MulFloat32s/64-32                       56.21Gi ±  1%    126.98Gi ±  1%   +125.92% (p=0.000 n=10)
MulFloat32s/4096-32                     55.72Gi ±  4%    232.82Gi ±  5%   +317.84% (p=0.000 n=10)
MulFloat32s/1048576-32                  54.89Gi ±  3%    135.69Gi ±  8%   +147.20% (p=0.000 n=10)
ScalFloat32s/64-32                      44.84Gi ±  8%    134.44Gi ±  3%   +199.82% (p=0.000 n=10)
ScalFloat32s/4096-32                    41.18Gi ±  2%    210.60Gi ±  2%   +411.46% (p=0.000 n=10)
ScalFloat32s/1048576-32                 41.07Gi ±  2%    213.15Gi ±  2%   +418.98% (p=0.000 n=10)
MixFloat32s/64-32                       30.67Gi ±  2%    118.49Gi ±  4%   +286.29% (p=0.000 n=10)
MixFloat32s/4096-32                     30.99Gi ±  2%    225.65Gi ±  3%   +628.12% (p=0.000 n=10)
MixFloat32s/1048576-32                  30.05Gi ±  2%    140.20Gi ±  9%   +366.61% (p=0.000 n=10)
AxpyFloat32s/64-32                      37.57Gi ±  1%    160.40Gi ±  3%   +326.88% (p=0.000 n=10)
AxpyFloat32s/4096-32                    57.01Gi ±  2%    288.18Gi ±  2%   +405.52% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 55.47Gi ±  1%    127.94Gi ± 15%   +130.66% (p=0.000 n=10)
NegFloat32s/64-32                       34.95Gi ± 12%    115.12Gi ±  2%   +229.34% (p=0.000 n=10)
NegFloat32s/4096-32                     37.51Gi ±  7%    192.64Gi ±  3%   +413.56% (p=0.000 n=10)
NegFloat32s/1048576-32                  38.13Gi ±  3%    130.19Gi ±  4%   +241.46% (p=0.000 n=10)
DivFloat32s/64-32                       25.32Gi ±  2%    140.82Gi ±  3%   +456.17% (p=0.000 n=10)
DivFloat32s/4096-32                     25.50Gi ±  3%    272.37Gi ±  2%   +967.92% (p=0.000 n=10)
DivFloat32s/1048576-32                  25.21Gi ±  2%    141.76Gi ± 10%   +462.27% (p=0.000 n=10)
DaxpyFloat32s/64-32                     56.23Gi ±  3%    132.20Gi ±  6%   +135.10% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   55.31Gi ±  3%    233.82Gi ±  1%   +322.75% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                56.48Gi ±  5%    180.22Gi ±  1%   +219.09% (p=0.000 n=10)
FillFloat32s/64-32                      28.79Gi ± 27%     73.59Gi ±  8%   +155.60% (p=0.000 n=10)
FillFloat32s/4096-32                    20.86Gi ±  3%    147.77Gi ±  2%   +608.47% (p=0.000 n=10)
FillFloat32s/1048576-32                 20.69Gi ±  1%    130.78Gi ±  1%   +531.98% (p=0.000 n=10)
FillUint8s/64-32                        7.482Gi ± 30%    27.683Gi ±  8%   +269.99% (p=0.000 n=10)
FillUint8s/4096-32                      5.157Gi ±  2%   148.361Gi ± 16%  +2776.76% (p=0.000 n=10)
FillUint8s/1048576-32                   5.323Gi ±  2%   160.802Gi ±  2%  +2921.16% (p=0.000 n=10)
ReverseIncFloat32s/64-32                36.64Gi ±  4%    123.84Gi ±  2%   +238.03% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              40.69Gi ±  1%    247.39Gi ±  3%   +508.01% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           39.11Gi ±  3%    132.06Gi ±  5%   +237.66% (p=0.000 n=10)
CopyFloat32s/64-32                      41.03Gi ±  1%    146.53Gi ±  2%   +257.10% (p=0.000 n=10)
CopyFloat32s/4096-32                    40.43Gi ±  1%    417.05Gi ±  2%   +931.62% (p=0.000 n=10)
CopyFloat32s/1048576-32                 41.32Gi ±  2%    137.16Gi ±  3%   +231.92% (p=0.000 n=10)
AndNotUint64s/64-32                     79.47Gi ±  2%    162.45Gi ±  1%   +104.41% (p=0.000 n=10)
AndNotUint64s/4096-32                   74.52Gi ± 20%    231.34Gi ±  3%   +210.45% (p=0.000 n=10)
AndNotUint64s/1048576-32                75.86Gi ±  1%    138.58Gi ± 11%    +82.67% (p=0.000 n=10)
NormalizeFloat32s/64-32                 33.04Gi ±  2%    163.29Gi ±  2%   +394.27% (p=0.000 n=10)
NormalizeFloat32s/4096-32               33.45Gi ±  1%    256.61Gi ±  1%   +667.13% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            33.50Gi ±  3%    232.37Gi ±  1%   +593.75% (p=0.000 n=10)
ChainInt32s/64-32                       48.62Gi ±  2%    196.06Gi ±  2%   +303.26% (p=0.000 n=10)
ChainInt32s/4096-32                     47.92Gi ±  2%    341.00Gi ±  1%   +611.67% (p=0.000 n=10)
ChainInt32s/1048576-32                  48.87Gi ±  3%    185.37Gi ± 13%   +279.33% (p=0.000 n=10)
WideInt32s/64-32                        20.63Gi ±  2%    130.36Gi ±  2%   +531.72% (p=0.000 n=10)
WideInt32s/4096-32                      20.65Gi ±  3%    225.84Gi ±  5%   +993.78% (p=0.000 n=10)
WideInt32s/1048576-32                   20.98Gi ±  2%    124.90Gi ±  7%   +495.47% (p=0.000 n=10)
SumInt32s/64-32                         19.14Gi ±  7%     31.60Gi ±  2%    +65.04% (p=0.000 n=10)
SumInt32s/4096-32                       20.19Gi ±  5%     77.15Gi ±  2%   +282.08% (p=0.000 n=10)
SumInt32s/1048576-32                    21.03Gi ±  2%     82.37Gi ±  1%   +291.77% (p=0.000 n=10)
DotInt32s/64-32                         33.72Gi ±  2%     50.71Gi ±  1%    +50.39% (p=0.000 n=10)
DotInt32s/4096-32                       32.48Gi ± 19%    147.38Gi ±  6%   +353.74% (p=0.000 n=10)
DotInt32s/1048576-32                    31.94Gi ±  2%    124.21Gi ±  5%   +288.90% (p=0.000 n=10)
SumFloat32s/64-32                       19.81Gi ± 23%     18.61Gi ±  2%          ~ (p=0.436 n=10)
SumFloat32s/4096-32                     10.78Gi ±  3%    136.22Gi ±  4%  +1163.27% (p=0.000 n=10)
SumFloat32s/1048576-32                  10.35Gi ±  0%    124.39Gi ±  2%  +1101.41% (p=0.000 n=10)
DotFloat32s/64-32                       29.49Gi ±  2%     35.94Gi ±  2%    +21.89% (p=0.000 n=10)
DotFloat32s/4096-32                     21.08Gi ±  2%    166.07Gi ±  2%   +687.69% (p=0.000 n=10)
DotFloat32s/1048576-32                  21.07Gi ±  3%    122.17Gi ±  4%   +479.92% (p=0.000 n=10)
DotFloat64s/64-32                       79.52Gi ± 11%     78.64Gi ±  2%          ~ (p=0.247 n=10)
DotFloat64s/4096-32                     42.96Gi ±  2%    165.98Gi ±  3%   +286.34% (p=0.000 n=10)
DotFloat64s/1048576-32                  41.24Gi ±  3%    131.72Gi ± 11%   +219.38% (p=0.000 n=10)
AbsFloat32s/64-32                       18.85Gi ± 10%    115.12Gi ±  5%   +510.85% (p=0.000 n=10)
AbsFloat32s/4096-32                     19.80Gi ±  6%    208.49Gi ±  2%   +953.05% (p=0.000 n=10)
AbsFloat32s/1048576-32                  20.49Gi ±  1%    132.21Gi ±  4%   +545.22% (p=0.000 n=10)
SqrtFloat32s/64-32                      9.221Gi ±  2%   131.680Gi ±  3%  +1328.01% (p=0.000 n=10)
SqrtFloat32s/4096-32                    9.325Gi ±  1%   144.343Gi ±  2%  +1447.92% (p=0.000 n=10)
SqrtFloat32s/1048576-32                 9.192Gi ±  3%   131.424Gi ±  4%  +1329.73% (p=0.000 n=10)
ClampInt32s/64-32                       16.34Gi ±  3%    104.73Gi ±  2%   +540.89% (p=0.000 n=10)
ClampInt32s/4096-32                     16.39Gi ±  0%    237.03Gi ±  3%  +1346.33% (p=0.000 n=10)
ClampInt32s/1048576-32                  16.67Gi ±  2%    131.49Gi ±  6%   +688.55% (p=0.000 n=10)
ShrInt32s/64-32                         36.78Gi ±  2%    112.15Gi ±  2%   +204.94% (p=0.000 n=10)
ShrInt32s/4096-32                       36.28Gi ± 10%    226.29Gi ±  2%   +523.67% (p=0.000 n=10)
ShrInt32s/1048576-32                    36.25Gi ±  3%    134.67Gi ±  6%   +271.49% (p=0.000 n=10)
ShlUint32s/64-32                        36.27Gi ±  4%    120.47Gi ±  3%   +232.12% (p=0.000 n=10)
ShlUint32s/4096-32                      35.95Gi ±  1%    226.33Gi ±  1%   +529.58% (p=0.000 n=10)
ShlUint32s/1048576-32                   36.35Gi ±  2%    134.01Gi ±  7%   +268.62% (p=0.000 n=10)
DiffInt32s/64-32                        30.91Gi ±  5%     63.84Gi ±  3%   +106.54% (p=0.000 n=10)
DiffInt32s/4096-32                      32.20Gi ±  3%     73.31Gi ±  2%   +127.71% (p=0.000 n=10)
DiffInt32s/1048576-32                   31.31Gi ±  3%     71.92Gi ±  2%   +129.68% (p=0.000 n=10)
SmoothInt32s/64-32                      21.63Gi ±  0%     54.44Gi ±  1%   +151.72% (p=0.000 n=10)
SmoothInt32s/4096-32                    17.54Gi ±  2%     71.57Gi ± 11%   +307.95% (p=0.000 n=10)
SmoothInt32s/1048576-32                 16.60Gi ±  0%     70.88Gi ±  1%   +327.07% (p=0.000 n=10)
ShiftInt32s/64-32                       30.44Gi ±  5%    108.33Gi ±  4%   +255.91% (p=0.000 n=10)
ShiftInt32s/4096-32                     29.29Gi ±  2%    160.61Gi ±  2%   +448.29% (p=0.000 n=10)
ShiftInt32s/1048576-32                  29.23Gi ±  7%    122.48Gi ±  2%   +318.99% (p=0.000 n=10)
AccumRowInt32s/64-32                    42.20Gi ±  5%    131.95Gi ±  2%   +212.66% (p=0.000 n=10)
AccumRowInt32s/4096-32                  42.02Gi ±  7%    190.46Gi ±  1%   +353.28% (p=0.000 n=10)
AccumRowInt32s/1048576-32               42.41Gi ±  3%    179.90Gi ±  6%   +324.19% (p=0.000 n=10)
MinUint16s/64-32                        4.902Gi ±  2%     3.899Gi ±  0%    -20.48% (p=0.000 n=10)
MinUint16s/4096-32                      3.557Gi ±  3%   113.465Gi ±  3%  +3090.17% (p=0.000 n=10)
MinUint16s/1048576-32                   3.549Gi ±  0%   126.361Gi ±  3%  +3460.16% (p=0.000 n=10)
ProductFloat32s/64-32                   11.31Gi ±  3%     29.16Gi ±  3%   +157.79% (p=0.000 n=10)
ProductFloat32s/4096-32                 6.985Gi ±  3%   139.595Gi ±  1%  +1898.48% (p=0.000 n=10)
ProductFloat32s/1048576-32              7.081Gi ±  2%   122.383Gi ±  2%  +1628.45% (p=0.000 n=10)
geomean                                 26.90Gi           127.3Gi         +373.11%
```
</details>


Running `loopvec -methods -split` on [gorgonia/tensor](https://github.com/gorgonia/tensor)
detects 176 vectorizable loops, yielding **~3.5x faster** on an AMD Ryzen 9 9950X3D (AVX-512):

<details>
<summary>bench.txt</summary>

```
                          │    scalar     │              simd               │
                          │    sec/op     │   sec/op     vs base            │
AddVSF32/64-32               12.235n ± 0%   3.558n ± 1%  -70.92% (p=0.002 n=6)
AddVSF32/4096-32              762.5n ± 1%   146.9n ± 1%  -80.74% (p=0.002 n=6)
AddVSF32/1048576-32          201.21µ ± 1%   37.64µ ± 1%  -81.30% (p=0.002 n=6)
MulVSF32/64-32               14.115n ± 1%   3.736n ± 1%  -73.53% (p=0.002 n=6)
MulVSF32/4096-32              843.8n ± 2%   147.2n ± 1%  -82.56% (p=0.002 n=6)
MulVSF32/1048576-32          220.16µ ± 3%   37.43µ ± 0%  -83.00% (p=0.002 n=6)
AddVSF64/64-32               12.405n ± 1%   5.668n ± 5%  -54.31% (p=0.002 n=6)
AddVSF64/4096-32              757.7n ± 3%   286.8n ± 1%  -62.16% (p=0.002 n=6)
AddVSF64/1048576-32          204.33µ ± 1%   75.24µ ± 1%  -63.18% (p=0.002 n=6)
MulVSF64/64-32               12.445n ± 1%   5.907n ± 1%  -52.53% (p=0.002 n=6)
MulVSF64/4096-32              754.3n ± 1%   286.9n ± 1%  -61.97% (p=0.002 n=6)
MulVSF64/1048576-32          204.05µ ± 1%   75.39µ ± 2%  -63.05% (p=0.002 n=6)
VecAddI32/64-32              12.035n ± 1%   4.423n ± 1%  -63.24% (p=0.002 n=6)
VecAddI32/4096-32             761.0n ± 1%   194.9n ± 1%  -74.39% (p=0.002 n=6)
VecAddI32/1048576-32         204.74µ ± 1%   65.12µ ± 2%  -68.19% (p=0.002 n=6)
VecMulI32/64-32              14.440n ± 0%   4.464n ± 3%  -69.09% (p=0.002 n=6)
VecMulI32/4096-32             948.4n ± 0%   193.9n ± 1%  -79.55% (p=0.002 n=6)
VecMulI32/1048576-32         250.38µ ± 1%   64.80µ ± 1%  -74.12% (p=0.002 n=6)
geomean                        1.303µ        373.4n       -71.33%
```
</details>

Running `loopvec -methods -split` on [gonum](https://github.com/gonum/gonum)
detects loops in multiple packages (floats, blas, lapack, stat, and others).
With `-fp-reassoc`, gonum's own `BenchmarkMeanLargeWeighted` (a weighted
`stat.Mean` over 100,000 elements) runs **3.2x faster** (35.7 µs to 11.2 µs), and
the `stat` and `floats` tests pass on the rewritten code.

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

SIMD support in Go is experimental, so there are likely several bugs lurking around, both in the Go compiler/runtime and this tool.

<details>
<summary>Details</summary>

**Blank identifier parameters are renamed in the simd file.** The gotip
`GOEXPERIMENT=simd` compiler rejects blank identifier (`_`) parameters in
functions that contain simd code, emitting `cannot use _ as value or type`.
loopvec automatically renames them to `_p0`, `_p1`, … in the generated simd
file so the compiler does not see them. The original file is unchanged.

**Methods are not rewritten by default.** The experimental SIMD compiler crashes
with an internal error when a `//go:build goexperiment.simd` file contains a
method (function with a receiver). Only top-level functions are vectorized.
Tracked at [golang/go#80657](https://github.com/golang/go/issues/80657).

I have a preliminary fix at https://go.dev/cl/839405. Once you have a toolchain
that includes it, pass `-methods` to enable method rewriting:

```bash
go install golang.org/dl/gotip@latest
gotip download 839405
# rewrite both functions and methods
GOEXPERIMENT=simd gotip tool loopvec -methods -split ./...
```

**Loop limits.** When a loop is limited by something other than the length of
the slice it writes (`i < n`, `range n`, or `range src` while writing `dst`), the
rewritten loop stops at that limit and first checks every slice with
`_ = s[limit-1]`. An out-of-range access therefore still panics, with an index
error, but before any element is written, whereas the scalar loop would have
written the elements ahead of the failing index first. Only an `int` variable or
a positive integer constant is accepted as a limit.

**Files with build constraints are skipped.** Any file that carries a
`//go:build` (or legacy `// +build`) line is left untouched. This includes files
already guarded by `-split`, so re-running `-split` on a package you have
already split is a no-op.

**No reductions.** `sum += a[i]`, `dot += a[i]*b[i]`, `min`/`max` accumulation,
and similar are not rewritten. The `simd` package's `ReduceSum` and friends
are only in `gotip`, not the current `go1.27` release.

**Float `min` and `max`** are not rewritten: on amd64 the hardware instruction
differs from Go's for NaN and signed zero.

**No SIMD hardware.** With `GODEBUG=simd=0`, or on a CPU `simd` does not support,
go1.27's emulated `float64` vectors return wrong results, so rewritten `float64`
loops are unreliable there.

**Reductions.** A fold into an integer local (`sum += a[i]`, `m = min(m, a[i])`)
runs whole vectors into an accumulator that starts at the operation's identity,
combines the lanes at the end, and runs the leftover elements with the original
body. Integer operations wrap the same way in any order, so the result is
identical. A float sum or product is regrouped, and the result can differ in the
last bits, so it is rewritten only with `-fp-reassoc` (`LOOPVEC_TOOLEXEC_FP_REASSOC=1`
for `loopvec-toolexec`); the main loop then folds four vectors per iteration so
that the dependent adds overlap. The error stays within the usual summation bound,
`n*u*sum|term|`. Float `min` and `max` are never rewritten. Loops under 64
elements (for `int32` or `float32`) stay scalar.

**Loop shapes.** Range loops must declare their index (`for i := range x`). The
body must be assignments to `dst[i]` and to temporaries that nothing reads after
the loop, with no `if`; `dst[i+1]`, `a[i+1]` and `a[0]` are not rewritten.

</details>

## Testing

The testing suite is built on top of [rsc.io/script](https://pkg.go.dev/rsc.io/script) with test scripts in `testdata/*.txt`,
the same approach used by [`go` command tests](https://github.com/golang/go/tree/f2d76b7/src/cmd/go/testdata/script).

```sh
make test
```

[tsvc/](tsvc/) is a Go port of the [TSVC_2](https://github.com/UoB-HPC/TSVC_2)
vectorizer kernel suite. It checks the scalar and SIMD builds and records which
kernels loopvec vectorizes. See [tsvc/README.md](tsvc/README.md).
