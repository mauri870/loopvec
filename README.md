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
AddFloat32s/64-32                       12.665n ±  2%   4.844n ±  3%  -61.75% (p=0.000 n=10)
AddFloat32s/4096-32                      820.8n ±  2%   168.0n ±  4%  -79.53% (p=0.000 n=10)
AddFloat32s/1048576-32                  217.41µ ±  2%   89.96µ ±  3%  -58.62% (p=0.000 n=10)
MulFloat32s/64-32                       12.625n ±  1%   5.666n ±  3%  -55.12% (p=0.000 n=10)
MulFloat32s/4096-32                      806.5n ±  3%   199.0n ±  6%  -75.33% (p=0.000 n=10)
MulFloat32s/1048576-32                  217.94µ ±  2%   91.02µ ±  3%  -58.24% (p=0.000 n=10)
ScalFloat32s/64-32                       9.969n ± 17%   3.523n ±  1%  -64.65% (p=0.000 n=10)
ScalFloat32s/4096-32                     736.0n ±  1%   144.7n ±  2%  -80.35% (p=0.000 n=10)
ScalFloat32s/1048576-32                 186.09µ ±  2%   36.01µ ±  3%  -80.65% (p=0.000 n=10)
MixFloat32s/64-32                       23.540n ±  1%   5.953n ±  2%  -74.71% (p=0.000 n=10)
MixFloat32s/4096-32                     1455.0n ±  0%   201.8n ±  1%  -86.13% (p=0.000 n=10)
MixFloat32s/1048576-32                  379.42µ ±  2%   91.60µ ±  6%  -75.86% (p=0.000 n=10)
AxpyFloat32s/64-32                      12.985n ±  1%   4.465n ±  1%  -65.61% (p=0.000 n=10)
AxpyFloat32s/4096-32                     799.9n ±  0%   158.0n ±  4%  -80.25% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 210.33µ ±  2%   90.77µ ±  5%  -56.84% (p=0.000 n=10)
NegFloat32s/64-32                       14.400n ± 11%   4.138n ±  2%  -71.26% (p=0.000 n=10)
NegFloat32s/4096-32                      769.1n ±  4%   156.6n ±  4%  -79.65% (p=0.000 n=10)
NegFloat32s/1048576-32                  201.67µ ±  3%   59.10µ ±  6%  -70.70% (p=0.000 n=10)
DivFloat32s/64-32                       28.785n ±  3%   5.106n ±  2%  -82.26% (p=0.000 n=10)
DivFloat32s/4096-32                     1843.5n ±  2%   167.3n ±  3%  -90.92% (p=0.000 n=10)
DivFloat32s/1048576-32                  471.22µ ±  2%   86.48µ ± 10%  -81.65% (p=0.000 n=10)
DaxpyFloat32s/64-32                     13.815n ±  6%   5.370n ±  5%  -61.13% (p=0.000 n=10)
DaxpyFloat32s/4096-32                    875.3n ±  8%   194.0n ±  1%  -77.84% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                217.58µ ±  4%   63.38µ ±  3%  -70.87% (p=0.000 n=10)
FillFloat32s/64-32                       8.098n ± 39%   3.029n ± 11%  -62.59% (p=0.000 n=10)
FillFloat32s/4096-32                     737.0n ±  2%   102.5n ±  2%  -86.10% (p=0.000 n=10)
FillFloat32s/1048576-32                 186.30µ ±  2%   29.86µ ±  1%  -83.97% (p=0.000 n=10)
FillUint8s/64-32                         7.973n ± 42%   2.147n ±  6%  -73.07% (p=0.000 n=10)
FillUint8s/4096-32                      730.10n ±  2%   23.71n ± 14%  -96.75% (p=0.000 n=10)
FillUint8s/1048576-32                  183.660µ ±  3%   6.068µ ±  2%  -96.70% (p=0.000 n=10)
ReverseIncFloat32s/64-32                12.930n ±  5%   3.787n ±  3%  -70.72% (p=0.000 n=10)
ReverseIncFloat32s/4096-32               760.0n ±  5%   122.4n ±  2%  -83.90% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           200.76µ ±  6%   56.57µ ± 10%  -71.82% (p=0.000 n=10)
CopyFloat32s/64-32                      11.750n ±  1%   3.263n ±  0%  -72.23% (p=0.000 n=10)
CopyFloat32s/4096-32                    736.10n ±  3%   71.00n ±  2%  -90.35% (p=0.000 n=10)
CopyFloat32s/1048576-32                 190.27µ ±  2%   59.05µ ±  4%  -68.97% (p=0.000 n=10)
AndNotUint64s/64-32                     18.535n ±  4%   8.796n ±  4%  -52.54% (p=0.000 n=10)
AndNotUint64s/4096-32                   1232.0n ±  2%   398.7n ±  2%  -67.64% (p=0.000 n=10)
AndNotUint64s/1048576-32                 311.6µ ±  9%   159.2µ ±  8%  -48.92% (p=0.000 n=10)
NormalizeFloat32s/64-32                 28.460n ±  1%   5.786n ±  1%  -79.67% (p=0.000 n=10)
NormalizeFloat32s/4096-32               1816.5n ±  2%   238.3n ±  1%  -86.88% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            473.95µ ±  1%   68.81µ ±  3%  -85.48% (p=0.000 n=10)
ChainInt32s/64-32                       29.820n ±  2%   7.405n ±  1%  -75.17% (p=0.000 n=10)
ChainInt32s/4096-32                     1900.5n ±  2%   270.1n ±  4%  -85.79% (p=0.000 n=10)
ChainInt32s/1048576-32                   479.1µ ±  2%   130.5µ ± 18%  -72.75% (p=0.000 n=10)
WideInt32s/64-32                        46.120n ±  2%   7.261n ±  2%  -84.26% (p=0.000 n=10)
WideInt32s/4096-32                      2872.5n ±  3%   263.4n ±  7%  -90.83% (p=0.000 n=10)
WideInt32s/1048576-32                    735.3µ ±  3%   126.9µ ±  8%  -82.74% (p=0.000 n=10)
SumInt32s/64-32                         12.430n ±  6%   7.397n ±  1%  -40.49% (p=0.000 n=10)
SumInt32s/4096-32                        753.2n ±  7%   194.2n ±  1%  -74.21% (p=0.000 n=10)
SumInt32s/1048576-32                    189.31µ ±  3%   46.56µ ±  2%  -75.40% (p=0.000 n=10)
DotInt32s/64-32                         14.140n ±  2%   9.377n ±  1%  -33.68% (p=0.000 n=10)
DotInt32s/4096-32                        939.4n ±  3%   205.1n ±  3%  -78.17% (p=0.000 n=10)
DotInt32s/1048576-32                    244.44µ ±  0%   61.86µ ±  6%  -74.69% (p=0.000 n=10)
SumFloat32s/64-32                        11.03n ± 23%   12.85n ±  2%        ~ (p=0.138 n=10)
SumFloat32s/4096-32                     1443.0n ±  2%   112.1n ±  3%  -92.23% (p=0.000 n=10)
SumFloat32s/1048576-32                  367.71µ ±  3%   31.90µ ±  3%  -91.33% (p=0.000 n=10)
DotFloat32s/64-32                        15.78n ±  2%   13.26n ±  2%  -15.97% (p=0.000 n=10)
DotFloat32s/4096-32                     1423.0n ±  3%   183.3n ±  4%  -87.12% (p=0.000 n=10)
DotFloat32s/1048576-32                  380.46µ ±  1%   63.99µ ±  2%  -83.18% (p=0.000 n=10)
DotFloat64s/64-32                        12.93n ± 21%   12.38n ±  2%        ~ (p=0.853 n=10)
DotFloat64s/4096-32                     1418.0n ±  3%   380.1n ± 13%  -73.20% (p=0.000 n=10)
DotFloat64s/1048576-32                   379.0µ ±  2%   123.8µ ±  8%  -67.34% (p=0.000 n=10)
geomean                                  1.669µ         391.1n        -76.56%

                              │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                              │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                       56.48Gi ±  2%    147.66Gi ±  3%   +161.44% (p=0.000 n=10)
AddFloat32s/4096-32                     55.77Gi ±  2%    272.48Gi ±  4%   +388.56% (p=0.000 n=10)
AddFloat32s/1048576-32                  53.90Gi ±  2%    130.28Gi ±  3%   +141.70% (p=0.000 n=10)
MulFloat32s/64-32                       56.63Gi ±  1%    126.23Gi ±  3%   +122.89% (p=0.000 n=10)
MulFloat32s/4096-32                     56.75Gi ±  2%    230.02Gi ±  5%   +305.30% (p=0.000 n=10)
MulFloat32s/1048576-32                  53.77Gi ±  2%    128.75Gi ±  3%   +139.44% (p=0.000 n=10)
ScalFloat32s/64-32                      47.84Gi ± 15%    135.34Gi ±  1%   +182.92% (p=0.000 n=10)
ScalFloat32s/4096-32                    41.47Gi ±  1%    211.02Gi ±  2%   +408.90% (p=0.000 n=10)
ScalFloat32s/1048576-32                 41.98Gi ±  2%    216.95Gi ±  3%   +416.77% (p=0.000 n=10)
MixFloat32s/64-32                       30.38Gi ±  1%    120.15Gi ±  2%   +295.48% (p=0.000 n=10)
MixFloat32s/4096-32                     31.46Gi ±  0%    226.92Gi ±  1%   +621.33% (p=0.000 n=10)
MixFloat32s/1048576-32                  30.89Gi ±  2%    127.94Gi ±  6%   +314.23% (p=0.000 n=10)
AxpyFloat32s/64-32                      55.07Gi ±  1%    160.17Gi ±  1%   +190.85% (p=0.000 n=10)
AxpyFloat32s/4096-32                    57.23Gi ±  0%    289.72Gi ±  4%   +406.24% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 55.72Gi ±  2%    129.11Gi ±  5%   +131.72% (p=0.000 n=10)
NegFloat32s/64-32                       33.12Gi ± 13%    115.24Gi ±  2%   +247.99% (p=0.000 n=10)
NegFloat32s/4096-32                     39.68Gi ±  4%    194.92Gi ±  4%   +391.22% (p=0.000 n=10)
NegFloat32s/1048576-32                  38.74Gi ±  3%    132.20Gi ±  5%   +241.26% (p=0.000 n=10)
DivFloat32s/64-32                       24.85Gi ±  3%    140.11Gi ±  2%   +463.90% (p=0.000 n=10)
DivFloat32s/4096-32                     24.84Gi ±  2%    273.60Gi ±  3%  +1001.56% (p=0.000 n=10)
DivFloat32s/1048576-32                  24.87Gi ±  2%    135.52Gi ± 11%   +444.85% (p=0.000 n=10)
DaxpyFloat32s/64-32                     51.76Gi ±  7%    133.19Gi ±  4%   +157.32% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   52.31Gi ±  8%    235.98Gi ±  1%   +351.15% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                53.88Gi ±  4%    184.91Gi ±  3%   +243.20% (p=0.000 n=10)
FillFloat32s/64-32                      29.44Gi ± 28%     78.72Gi ± 10%   +167.39% (p=0.000 n=10)
FillFloat32s/4096-32                    20.70Gi ±  2%    148.96Gi ±  1%   +619.46% (p=0.000 n=10)
FillFloat32s/1048576-32                 20.97Gi ±  2%    130.84Gi ±  1%   +524.03% (p=0.000 n=10)
FillUint8s/64-32                        7.476Gi ± 29%    27.764Gi ±  6%   +271.36% (p=0.000 n=10)
FillUint8s/4096-32                      5.226Gi ±  2%   160.855Gi ± 12%  +2978.15% (p=0.000 n=10)
FillUint8s/1048576-32                   5.317Gi ±  3%   160.947Gi ±  2%  +2926.90% (p=0.000 n=10)
ReverseIncFloat32s/64-32                36.88Gi ±  5%    125.93Gi ±  3%   +241.46% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              40.15Gi ±  4%    249.28Gi ±  2%   +520.86% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           38.92Gi ±  6%    138.13Gi ±  9%   +254.93% (p=0.000 n=10)
CopyFloat32s/64-32                      40.58Gi ±  1%    146.11Gi ±  0%   +260.05% (p=0.000 n=10)
CopyFloat32s/4096-32                    41.46Gi ±  3%    429.84Gi ±  2%   +936.81% (p=0.000 n=10)
CopyFloat32s/1048576-32                 41.06Gi ±  2%    132.32Gi ±  4%   +222.24% (p=0.000 n=10)
AndNotUint64s/64-32                     77.18Gi ±  4%    162.63Gi ±  4%   +110.73% (p=0.000 n=10)
AndNotUint64s/4096-32                   74.31Gi ±  2%    229.63Gi ±  2%   +209.03% (p=0.000 n=10)
AndNotUint64s/1048576-32                75.22Gi ±  8%    147.30Gi ±  7%    +95.83% (p=0.000 n=10)
NormalizeFloat32s/64-32                 33.51Gi ±  1%    164.85Gi ±  1%   +391.95% (p=0.000 n=10)
NormalizeFloat32s/4096-32               33.60Gi ±  2%    256.10Gi ±  1%   +662.19% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            32.97Gi ±  1%    227.07Gi ±  3%   +588.75% (p=0.000 n=10)
ChainInt32s/64-32                       47.97Gi ±  2%    193.18Gi ±  1%   +302.70% (p=0.000 n=10)
ChainInt32s/4096-32                     48.17Gi ±  2%    338.92Gi ±  4%   +603.65% (p=0.000 n=10)
ChainInt32s/1048576-32                  48.92Gi ±  2%    179.66Gi ± 22%   +267.23% (p=0.000 n=10)
WideInt32s/64-32                        20.68Gi ±  2%    131.34Gi ±  2%   +535.19% (p=0.000 n=10)
WideInt32s/4096-32                      21.25Gi ±  3%    231.77Gi ±  6%   +990.65% (p=0.000 n=10)
WideInt32s/1048576-32                   21.25Gi ±  3%    123.18Gi ±  9%   +479.64% (p=0.000 n=10)
SumInt32s/64-32                         19.19Gi ±  7%     32.23Gi ±  1%    +68.00% (p=0.000 n=10)
SumInt32s/4096-32                       20.26Gi ±  7%     78.54Gi ±  1%   +287.71% (p=0.000 n=10)
SumInt32s/1048576-32                    20.63Gi ±  3%     83.89Gi ±  2%   +306.56% (p=0.000 n=10)
DotInt32s/64-32                         33.73Gi ±  2%     50.85Gi ±  1%    +50.75% (p=0.000 n=10)
DotInt32s/4096-32                       32.49Gi ±  3%    148.81Gi ±  3%   +358.03% (p=0.000 n=10)
DotInt32s/1048576-32                    31.96Gi ±  0%    126.29Gi ±  6%   +295.14% (p=0.000 n=10)
SumFloat32s/64-32                       21.63Gi ± 19%     18.55Gi ±  2%          ~ (p=0.143 n=10)
SumFloat32s/4096-32                     10.57Gi ±  2%    136.10Gi ±  3%  +1187.27% (p=0.000 n=10)
SumFloat32s/1048576-32                  10.62Gi ±  3%    122.46Gi ±  3%  +1052.76% (p=0.000 n=10)
DotFloat32s/64-32                       30.21Gi ±  2%     35.96Gi ±  2%    +19.03% (p=0.000 n=10)
DotFloat32s/4096-32                     21.45Gi ±  3%    166.51Gi ±  4%   +676.39% (p=0.000 n=10)
DotFloat32s/1048576-32                  20.53Gi ±  1%    122.09Gi ±  2%   +494.55% (p=0.000 n=10)
DotFloat64s/64-32                       73.95Gi ± 18%     77.07Gi ±  2%          ~ (p=0.853 n=10)
DotFloat64s/4096-32                     43.04Gi ±  2%    160.64Gi ± 12%   +273.20% (p=0.000 n=10)
DotFloat64s/1048576-32                  41.22Gi ±  2%    126.24Gi ±  7%   +206.22% (p=0.000 n=10)
geomean                                 32.78Gi           139.9Gi         +326.59%
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
