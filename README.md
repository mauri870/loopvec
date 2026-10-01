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
AddFloat32s/64-32                       12.885n ±  2%   5.739n ±  1%  -55.46% (p=0.000 n=10)
AddFloat32s/4096-32                      819.6n ±  2%   203.3n ±  1%  -75.20% (p=0.000 n=10)
AddFloat32s/1048576-32                  219.94µ ±  9%   82.89µ ±  7%  -62.31% (p=0.000 n=10)
MulFloat32s/64-32                       12.830n ±  1%   4.819n ±  2%  -62.44% (p=0.000 n=10)
MulFloat32s/4096-32                      832.9n ±  3%   168.0n ±  3%  -79.83% (p=0.000 n=10)
MulFloat32s/1048576-32                  217.28µ ±  1%   81.87µ ±  6%  -62.32% (p=0.000 n=10)
ScalFloat32s/64-32                      10.615n ± 12%   3.593n ±  3%  -66.16% (p=0.000 n=10)
ScalFloat32s/4096-32                     741.4n ±  2%   125.7n ±  2%  -83.04% (p=0.000 n=10)
ScalFloat32s/1048576-32                 190.44µ ±  2%   33.27µ ±  2%  -82.53% (p=0.000 n=10)
MixFloat32s/64-32                       23.790n ±  1%   5.346n ±  2%  -77.53% (p=0.000 n=10)
MixFloat32s/4096-32                     1500.5n ±  2%   179.5n ±  5%  -88.03% (p=0.000 n=10)
MixFloat32s/1048576-32                  385.56µ ±  2%   82.42µ ±  6%  -78.62% (p=0.000 n=10)
AxpyFloat32s/64-32                      18.980n ±  2%   5.618n ±  2%  -70.40% (p=0.000 n=10)
AxpyFloat32s/4096-32                     813.9n ±  7%   169.8n ±  3%  -79.14% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 215.38µ ±  3%   79.62µ ±  8%  -63.03% (p=0.000 n=10)
NegFloat32s/64-32                       14.170n ± 10%   4.095n ±  3%  -71.10% (p=0.000 n=10)
NegFloat32s/4096-32                      811.6n ±  7%   147.9n ±  1%  -81.77% (p=0.000 n=10)
NegFloat32s/1048576-32                  200.87µ ±  3%   58.77µ ±  2%  -70.74% (p=0.000 n=10)
DivFloat32s/64-32                       28.365n ±  2%   5.933n ±  2%  -79.08% (p=0.000 n=10)
DivFloat32s/4096-32                     1811.0n ±  2%   202.2n ±  9%  -88.83% (p=0.000 n=10)
DivFloat32s/1048576-32                  475.60µ ±  3%   83.37µ ±  7%  -82.47% (p=0.000 n=10)
DaxpyFloat32s/64-32                     14.025n ± 12%   4.762n ±  1%  -66.05% (p=0.000 n=10)
DaxpyFloat32s/4096-32                    870.1n ± 15%   166.6n ±  1%  -80.85% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                223.17µ ±  6%   61.53µ ±  3%  -72.43% (p=0.000 n=10)
FillFloat32s/64-32                       8.038n ±  8%   2.734n ±  6%  -65.99% (p=0.000 n=10)
FillFloat32s/4096-32                    743.35n ±  1%   84.56n ± 13%  -88.63% (p=0.000 n=10)
FillFloat32s/1048576-32                 189.41µ ±  3%   29.81µ ±  1%  -84.26% (p=0.000 n=10)
FillUint8s/64-32                         8.232n ± 12%   1.917n ±  4%  -76.71% (p=0.000 n=10)
FillUint8s/4096-32                      743.10n ±  3%   20.20n ± 14%  -97.28% (p=0.000 n=10)
FillUint8s/1048576-32                  189.186µ ±  3%   5.836µ ±  3%  -96.92% (p=0.000 n=10)
ReverseIncFloat32s/64-32                13.235n ±  4%   4.314n ±  3%  -67.40% (p=0.000 n=10)
ReverseIncFloat32s/4096-32               766.1n ±  2%   147.4n ±  2%  -80.76% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           201.07µ ±  2%   58.37µ ±  5%  -70.97% (p=0.000 n=10)
CopyFloat32s/64-32                      12.055n ±  2%   3.279n ±  1%  -72.80% (p=0.000 n=10)
CopyFloat32s/4096-32                    754.30n ±  2%   71.77n ±  3%  -90.49% (p=0.000 n=10)
CopyFloat32s/1048576-32                 192.33µ ±  1%   58.77µ ±  3%  -69.45% (p=0.000 n=10)
AndNotUint64s/64-32                     18.510n ±  2%   7.530n ±  5%  -59.32% (p=0.000 n=10)
AndNotUint64s/4096-32                   1225.5n ±  2%   347.0n ±  3%  -71.69% (p=0.000 n=10)
AndNotUint64s/1048576-32                 313.7µ ±  3%   172.2µ ± 11%  -45.11% (p=0.000 n=10)
NormalizeFloat32s/64-32                 28.915n ±  3%   7.075n ±  4%  -75.53% (p=0.000 n=10)
NormalizeFloat32s/4096-32               1850.0n ±  2%   305.3n ±  7%  -83.50% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            479.11µ ±  1%   79.80µ ±  3%  -83.34% (p=0.000 n=10)
ChainInt32s/64-32                       31.770n ±  8%   9.584n ±  2%  -69.83% (p=0.000 n=10)
ChainInt32s/4096-32                     1939.5n ± 17%   355.9n ±  1%  -81.65% (p=0.000 n=10)
ChainInt32s/1048576-32                   487.0µ ± 22%   127.8µ ± 15%  -73.76% (p=0.000 n=10)
WideInt32s/64-32                        46.390n ±  2%   7.965n ±  1%  -82.83% (p=0.000 n=10)
WideInt32s/4096-32                      2952.0n ±  1%   260.6n ±  3%  -91.17% (p=0.000 n=10)
WideInt32s/1048576-32                    756.3µ ±  2%   119.7µ ±  5%  -84.18% (p=0.000 n=10)
SumInt32s/64-32                         12.455n ±  4%   7.645n ±  1%  -38.62% (p=0.000 n=10)
SumInt32s/4096-32                        807.3n ±  6%   119.5n ±  1%  -85.20% (p=0.000 n=10)
SumInt32s/1048576-32                    189.85µ ±  7%   31.70µ ±  2%  -83.30% (p=0.000 n=10)
DotInt32s/64-32                         14.320n ±  3%   8.556n ±  2%  -40.25% (p=0.000 n=10)
DotInt32s/4096-32                        957.1n ± 12%   219.0n ±  7%  -77.12% (p=0.000 n=10)
DotInt32s/1048576-32                    243.43µ ±  2%   64.34µ ±  2%  -73.57% (p=0.000 n=10)
SumFloat32s/64-32                        10.75n ±  2%   13.11n ±  2%  +22.06% (p=0.000 n=10)
SumFloat32s/4096-32                     1452.5n ±  3%   105.9n ±  2%  -92.71% (p=0.000 n=10)
SumFloat32s/1048576-32                  378.28µ ±  3%   31.31µ ±  3%  -91.72% (p=0.000 n=10)
DotFloat32s/64-32                        16.20n ±  2%   13.30n ±  2%  -17.87% (p=0.000 n=10)
DotFloat32s/4096-32                     1449.0n ±  2%   179.8n ±  1%  -87.59% (p=0.000 n=10)
DotFloat32s/1048576-32                  383.59µ ±  1%   64.00µ ±  3%  -83.31% (p=0.000 n=10)
DotFloat64s/64-32                        12.38n ±  8%   12.46n ±  1%        ~ (p=0.630 n=10)
DotFloat64s/4096-32                     1451.0n ±  2%   364.9n ±  2%  -74.85% (p=0.000 n=10)
DotFloat64s/1048576-32                   379.4µ ±  1%   120.0µ ± 11%  -68.38% (p=0.000 n=10)
geomean                                  1.701µ         385.7n        -77.33%

                              │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                              │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                       55.52Gi ±  2%    124.64Gi ±  1%   +124.50% (p=0.000 n=10)
AddFloat32s/4096-32                     55.86Gi ±  2%    225.17Gi ±  1%   +303.13% (p=0.000 n=10)
AddFloat32s/1048576-32                  53.28Gi ±  8%    141.38Gi ±  6%   +165.33% (p=0.000 n=10)
MulFloat32s/64-32                       55.74Gi ±  1%    148.42Gi ±  2%   +166.28% (p=0.000 n=10)
MulFloat32s/4096-32                     54.96Gi ±  3%    272.41Gi ±  3%   +395.67% (p=0.000 n=10)
MulFloat32s/1048576-32                  53.93Gi ±  1%    143.13Gi ±  6%   +165.38% (p=0.000 n=10)
ScalFloat32s/64-32                      45.02Gi ± 11%    132.74Gi ±  3%   +194.86% (p=0.000 n=10)
ScalFloat32s/4096-32                    41.16Gi ±  2%    242.73Gi ±  2%   +489.74% (p=0.000 n=10)
ScalFloat32s/1048576-32                 41.02Gi ±  2%    234.85Gi ±  2%   +472.48% (p=0.000 n=10)
MixFloat32s/64-32                       30.07Gi ±  1%    133.80Gi ±  2%   +345.03% (p=0.000 n=10)
MixFloat32s/4096-32                     30.51Gi ±  3%    255.00Gi ±  6%   +735.92% (p=0.000 n=10)
MixFloat32s/1048576-32                  30.39Gi ±  2%    142.18Gi ±  6%   +367.78% (p=0.000 n=10)
AxpyFloat32s/64-32                      37.69Gi ±  2%    127.32Gi ±  2%   +237.84% (p=0.000 n=10)
AxpyFloat32s/4096-32                    56.25Gi ±  7%    269.59Gi ±  3%   +379.27% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 54.41Gi ±  3%    147.20Gi ±  7%   +170.54% (p=0.000 n=10)
NegFloat32s/64-32                       33.65Gi ±  9%    116.45Gi ±  3%   +246.09% (p=0.000 n=10)
NegFloat32s/4096-32                     37.60Gi ±  6%    206.27Gi ±  1%   +448.59% (p=0.000 n=10)
NegFloat32s/1048576-32                  38.89Gi ±  3%    132.94Gi ±  3%   +241.80% (p=0.000 n=10)
DivFloat32s/64-32                       25.22Gi ±  2%    120.55Gi ±  2%   +378.00% (p=0.000 n=10)
DivFloat32s/4096-32                     25.27Gi ±  2%    226.46Gi ±  8%   +796.08% (p=0.000 n=10)
DivFloat32s/1048576-32                  24.64Gi ±  3%    140.56Gi ±  8%   +470.46% (p=0.000 n=10)
DaxpyFloat32s/64-32                     50.99Gi ± 11%    150.20Gi ±  1%   +194.55% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   52.61Gi ± 13%    274.70Gi ±  1%   +422.13% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                52.52Gi ±  6%    190.46Gi ±  3%   +262.62% (p=0.000 n=10)
FillFloat32s/64-32                      29.66Gi ±  8%     87.21Gi ±  6%   +194.04% (p=0.000 n=10)
FillFloat32s/4096-32                    20.53Gi ±  1%    180.57Gi ± 12%   +779.63% (p=0.000 n=10)
FillFloat32s/1048576-32                 20.62Gi ±  3%    131.05Gi ±  1%   +535.43% (p=0.000 n=10)
FillUint8s/64-32                        7.242Gi ± 11%    31.098Gi ±  4%   +329.44% (p=0.000 n=10)
FillUint8s/4096-32                      5.134Gi ±  3%   188.858Gi ± 12%  +3578.75% (p=0.000 n=10)
FillUint8s/1048576-32                   5.162Gi ±  3%   167.329Gi ±  3%  +3141.62% (p=0.000 n=10)
ReverseIncFloat32s/64-32                36.03Gi ±  4%    110.52Gi ±  3%   +206.76% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              39.84Gi ±  2%    207.01Gi ±  2%   +419.64% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           38.85Gi ±  2%    133.84Gi ±  5%   +244.46% (p=0.000 n=10)
CopyFloat32s/64-32                      39.54Gi ±  3%    145.42Gi ±  1%   +267.75% (p=0.000 n=10)
CopyFloat32s/4096-32                    40.46Gi ±  3%    425.24Gi ±  3%   +951.07% (p=0.000 n=10)
CopyFloat32s/1048576-32                 40.62Gi ±  1%    132.94Gi ±  3%   +227.28% (p=0.000 n=10)
AndNotUint64s/64-32                     77.28Gi ±  2%    189.99Gi ±  4%   +145.84% (p=0.000 n=10)
AndNotUint64s/4096-32                   74.69Gi ±  2%    263.84Gi ±  3%   +253.25% (p=0.000 n=10)
AndNotUint64s/1048576-32                74.72Gi ±  3%    136.14Gi ± 12%    +82.20% (p=0.000 n=10)
NormalizeFloat32s/64-32                 32.98Gi ±  3%    134.78Gi ±  3%   +308.65% (p=0.000 n=10)
NormalizeFloat32s/4096-32               32.99Gi ±  2%    199.92Gi ±  6%   +506.06% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            32.61Gi ±  1%    195.81Gi ±  3%   +500.39% (p=0.000 n=10)
ChainInt32s/64-32                       45.03Gi ±  9%    149.26Gi ±  2%   +231.49% (p=0.000 n=10)
ChainInt32s/4096-32                     47.20Gi ± 15%    257.24Gi ±  2%   +445.02% (p=0.000 n=10)
ChainInt32s/1048576-32                  48.13Gi ± 18%    183.47Gi ± 18%   +281.23% (p=0.000 n=10)
WideInt32s/64-32                        20.56Gi ±  2%    119.74Gi ±  1%   +482.44% (p=0.000 n=10)
WideInt32s/4096-32                      20.67Gi ±  1%    234.18Gi ±  3%  +1032.76% (p=0.000 n=10)
WideInt32s/1048576-32                   20.66Gi ±  2%    130.59Gi ±  4%   +532.10% (p=0.000 n=10)
SumInt32s/64-32                         19.14Gi ±  4%     31.19Gi ±  1%    +62.97% (p=0.000 n=10)
SumInt32s/4096-32                       18.91Gi ±  7%    127.71Gi ±  1%   +575.36% (p=0.000 n=10)
SumInt32s/1048576-32                    20.58Gi ±  6%    123.22Gi ±  2%   +498.86% (p=0.000 n=10)
DotInt32s/64-32                         33.31Gi ±  3%     55.73Gi ±  2%    +67.31% (p=0.000 n=10)
DotInt32s/4096-32                       31.89Gi ± 11%    139.33Gi ±  7%   +336.94% (p=0.000 n=10)
DotInt32s/1048576-32                    32.10Gi ±  2%    121.43Gi ±  2%   +278.34% (p=0.000 n=10)
SumFloat32s/64-32                       22.18Gi ±  2%     18.18Gi ±  2%    -18.05% (p=0.000 n=10)
SumFloat32s/4096-32                     10.51Gi ±  3%    144.10Gi ±  2%  +1271.58% (p=0.000 n=10)
SumFloat32s/1048576-32                  10.33Gi ±  3%    124.75Gi ±  3%  +1108.07% (p=0.000 n=10)
DotFloat32s/64-32                       29.44Gi ±  2%     35.83Gi ±  2%    +21.73% (p=0.000 n=10)
DotFloat32s/4096-32                     21.06Gi ±  2%    169.73Gi ±  1%   +705.94% (p=0.000 n=10)
DotFloat32s/1048576-32                  20.37Gi ±  1%    122.07Gi ±  3%   +499.35% (p=0.000 n=10)
DotFloat64s/64-32                       77.07Gi ±  7%     76.58Gi ±  0%          ~ (p=0.631 n=10)
DotFloat64s/4096-32                     42.07Gi ±  2%    167.25Gi ±  2%   +297.60% (p=0.000 n=10)
DotFloat64s/1048576-32                  41.18Gi ±  1%    130.36Gi ± 10%   +216.56% (p=0.000 n=10)
geomean                                 32.15Gi           141.8Gi         +341.14%
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
