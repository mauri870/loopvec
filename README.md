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
(`_loopvecOverlap`, comparing `unsafe.SliceData` ranges) that falls back to
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
AddFloat32s/64-32                         17.375n ± 35%   6.806n ±  1%  -60.83% (p=0.000 n=10)
AddFloat32s/4096-32                        882.8n ± 28%   192.9n ±  2%  -78.14% (p=0.000 n=10)
AddFloat32s/1048576-32                    214.28µ ± 35%   79.93µ ±  6%  -62.70% (p=0.000 n=10)
MulFloat32s/64-32                         16.255n ± 12%   6.359n ±  2%  -60.88% (p=0.000 n=10)
MulFloat32s/4096-32                        964.7n ± 10%   186.7n ±  2%  -80.65% (p=0.000 n=10)
MulFloat32s/1048576-32                    229.98µ ±  7%   81.10µ ±  7%  -64.74% (p=0.000 n=10)
ScalFloat32s/64-32                         9.899n ± 14%   3.712n ±  3%  -62.51% (p=0.000 n=10)
ScalFloat32s/4096-32                       744.0n ±  2%   145.8n ±  2%  -80.40% (p=0.000 n=10)
ScalFloat32s/1048576-32                   187.40µ ±  2%   36.72µ ±  3%  -80.41% (p=0.000 n=10)
MixFloat32s/64-32                         23.160n ±  2%   7.541n ±  3%  -67.44% (p=0.000 n=10)
MixFloat32s/4096-32                       1483.5n ±  2%   214.8n ±  3%  -85.52% (p=0.000 n=10)
MixFloat32s/1048576-32                    380.02µ ±  1%   93.33µ ±  9%  -75.44% (p=0.000 n=10)
AxpyFloat32s/64-32                        13.175n ±  1%   6.651n ±  1%  -49.52% (p=0.000 n=10)
AxpyFloat32s/4096-32                       822.6n ±  4%   196.2n ±  2%  -76.15% (p=0.000 n=10)
AxpyFloat32s/1048576-32                   214.66µ ±  4%   83.99µ ±  9%  -60.87% (p=0.000 n=10)
NegFloat32s/64-32                         13.090n ±  9%   5.061n ±  3%  -61.34% (p=0.000 n=10)
NegFloat32s/4096-32                        756.4n ±  1%   176.6n ±  2%  -76.65% (p=0.000 n=10)
NegFloat32s/1048576-32                    199.05µ ±  3%   58.57µ ±  5%  -70.57% (p=0.000 n=10)
DivFloat32s/64-32                         27.995n ±  3%   7.049n ±  1%  -74.82% (p=0.000 n=10)
DivFloat32s/4096-32                       1842.5n ±  3%   192.1n ±  2%  -89.58% (p=0.000 n=10)
DivFloat32s/1048576-32                    472.46µ ±  2%   87.00µ ±  8%  -81.59% (p=0.000 n=10)
DaxpyFloat32s/64-32                       12.600n ±  3%   5.418n ±  2%  -57.00% (p=0.000 n=10)
DaxpyFloat32s/4096-32                      756.5n ±  1%   175.7n ±  1%  -76.78% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                  200.36µ ±  1%   64.23µ ±  2%  -67.94% (p=0.000 n=10)
FillFloat32s/64-32                        11.900n ±  4%   3.479n ±  9%  -70.76% (p=0.000 n=10)
FillFloat32s/4096-32                       756.0n ±  7%   105.0n ±  3%  -86.10% (p=0.000 n=10)
FillFloat32s/1048576-32                   188.84µ ±  3%   29.39µ ±  1%  -84.44% (p=0.000 n=10)
FillUint8s/64-32                          12.085n ±  3%   2.098n ±  5%  -82.64% (p=0.000 n=10)
FillUint8s/4096-32                        758.30n ±  4%   25.36n ± 20%  -96.66% (p=0.000 n=10)
FillUint8s/1048576-32                    188.864µ ±  3%   6.020µ ±  1%  -96.81% (p=0.000 n=10)
ReverseIncFloat32s/64-32                  12.650n ±  6%   5.087n ±  2%  -59.78% (p=0.000 n=10)
ReverseIncFloat32s/4096-32                 773.2n ±  8%   169.8n ± 12%  -78.03% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32             199.47µ ± 91%   61.17µ ±  6%  -69.34% (p=0.000 n=10)
CopyFloat32s/64-32                        12.475n ±  9%   4.081n ±  2%  -67.28% (p=0.000 n=10)
CopyFloat32s/4096-32                      750.40n ±  1%   72.36n ±  0%  -90.36% (p=0.000 n=10)
CopyFloat32s/1048576-32                   192.64µ ±  2%   56.73µ ±  4%  -70.55% (p=0.000 n=10)
AndNotUint64s/64-32                       23.545n ±  1%   9.657n ±  2%  -58.98% (p=0.000 n=10)
AndNotUint64s/4096-32                     1474.5n ±  1%   387.5n ±  4%  -73.72% (p=0.000 n=10)
AndNotUint64s/1048576-32                   382.0µ ±  2%   168.1µ ±  9%  -55.99% (p=0.000 n=10)
NormalizeFloat32s/64-32                   28.380n ±  2%   6.473n ±  2%  -77.19% (p=0.000 n=10)
NormalizeFloat32s/4096-32                 1814.5n ±  2%   229.1n ±  2%  -87.37% (p=0.000 n=10)
NormalizeFloat32s/1048576-32              469.18µ ±  1%   68.98µ ±  3%  -85.30% (p=0.000 n=10)
ChainInt32s/64-32                          29.77n ±  2%   12.89n ±  1%  -56.68% (p=0.000 n=10)
ChainInt32s/4096-32                       1938.0n ±  4%   313.4n ±  7%  -83.83% (p=0.000 n=10)
ChainInt32s/1048576-32                     474.5µ ±  2%   126.5µ ± 12%  -73.35% (p=0.000 n=10)
WideInt32s/64-32                          30.340n ±  3%   9.613n ±  2%  -68.32% (p=0.000 n=10)
WideInt32s/4096-32                        1928.5n ± 43%   277.7n ±  3%  -85.60% (p=0.000 n=10)
WideInt32s/1048576-32                      484.9µ ±  2%   120.4µ ± 10%  -75.17% (p=0.000 n=10)
SumInt32s/64-32                           11.195n ±  9%   7.601n ±  3%  -32.10% (p=0.000 n=10)
SumInt32s/4096-32                          732.0n ±  2%   199.6n ±  1%  -72.74% (p=0.000 n=10)
SumInt32s/1048576-32                      189.46µ ±  2%   47.72µ ±  1%  -74.81% (p=0.000 n=10)
DotInt32s/64-32                           13.910n ±  3%   9.242n ±  1%  -33.56% (p=0.000 n=10)
DotInt32s/4096-32                          947.2n ±  3%   205.7n ±  3%  -78.29% (p=0.000 n=10)
DotInt32s/1048576-32                      244.98µ ±  3%   63.72µ ±  2%  -73.99% (p=0.000 n=10)
SumFloat32s/64-32                          10.77n ±  2%   13.37n ±  2%  +24.14% (p=0.000 n=10)
SumFloat32s/4096-32                       1411.5n ±  3%   111.9n ±  1%  -92.07% (p=0.000 n=10)
SumFloat32s/1048576-32                    370.97µ ±  2%   31.86µ ±  4%  -91.41% (p=0.000 n=10)
DotFloat32s/64-32                          15.31n ± 22%   13.38n ±  2%        ~ (p=0.469 n=10)
DotFloat32s/4096-32                       1419.0n ±  3%   181.1n ±  2%  -87.24% (p=0.000 n=10)
DotFloat32s/1048576-32                    371.29µ ±  3%   64.00µ ±  3%  -82.76% (p=0.000 n=10)
DotFloat64s/64-32                          15.91n ±  2%   12.38n ±  2%  -22.24% (p=0.000 n=10)
DotFloat64s/4096-32                       1469.0n ±  3%   367.0n ±  3%  -75.02% (p=0.000 n=10)
DotFloat64s/1048576-32                     381.9µ ±  2%   129.3µ ± 12%  -66.14% (p=0.000 n=10)
AbsFloat32s/64-32                         23.000n ±  1%   4.883n ±  3%  -78.77% (p=0.000 n=10)
AbsFloat32s/4096-32                       1461.5n ±  1%   167.3n ±  2%  -88.55% (p=0.000 n=10)
AbsFloat32s/1048576-32                    378.12µ ±  2%   57.20µ ±  4%  -84.87% (p=0.000 n=10)
SqrtFloat32s/64-32                        52.465n ±  2%   5.208n ±  2%  -90.07% (p=0.000 n=10)
SqrtFloat32s/4096-32                      3403.0n ±  1%   227.2n ±  5%  -93.33% (p=0.000 n=10)
SqrtFloat32s/1048576-32                   851.85µ ±  1%   61.20µ ±  4%  -92.82% (p=0.000 n=10)
ClampInt32s/64-32                         29.210n ±  0%   4.770n ±  1%  -83.67% (p=0.000 n=10)
ClampInt32s/4096-32                       1863.0n ±  3%   149.0n ±  1%  -92.00% (p=0.000 n=10)
ClampInt32s/1048576-32                    478.97µ ±  3%   59.01µ ±  4%  -87.68% (p=0.000 n=10)
ShrInt32s/64-32                           12.920n ±  2%   4.831n ±  2%  -62.61% (p=0.000 n=10)
ShrInt32s/4096-32                          839.5n ±  3%   156.1n ±  2%  -81.41% (p=0.000 n=10)
ShrInt32s/1048576-32                      211.74µ ±  2%   61.96µ ±  6%  -70.74% (p=0.000 n=10)
ShlUint32s/64-32                          12.920n ±  3%   4.970n ±  3%  -61.53% (p=0.000 n=10)
ShlUint32s/4096-32                         844.1n ±  2%   149.5n ±  1%  -82.29% (p=0.000 n=10)
ShlUint32s/1048576-32                     215.12µ ±  1%   57.94µ ±  6%  -73.07% (p=0.000 n=10)
DiffInt32s/64-32                          13.095n ±  0%   8.565n ±  1%  -34.60% (p=0.000 n=10)
DiffInt32s/4096-32                         855.8n ±  2%   426.1n ± 11%  -50.21% (p=0.000 n=10)
DiffInt32s/1048576-32                      222.0µ ±  3%   107.0µ ±  2%  -51.77% (p=0.000 n=10)
SmoothInt32s/64-32                        22.645n ± 27%   9.501n ±  1%  -58.04% (p=0.000 n=10)
SmoothInt32s/4096-32                      1863.5n ±  9%   528.9n ± 10%  -71.62% (p=0.000 n=10)
SmoothInt32s/1048576-32                    488.2µ ±  2%   120.9µ ±  1%  -75.23% (p=0.000 n=10)
ShiftInt32s/64-32                         15.535n ±  2%   5.234n ±  2%  -66.31% (p=0.000 n=10)
ShiftInt32s/4096-32                       1049.0n ±  2%   193.8n ±  2%  -81.52% (p=0.000 n=10)
ShiftInt32s/1048576-32                    263.23µ ±  4%   60.42µ ±  5%  -77.05% (p=0.000 n=10)
AccumRowInt32s/64-32                      15.795n ±  2%   6.472n ±  1%  -59.03% (p=0.000 n=10)
AccumRowInt32s/4096-32                    1026.0n ±  3%   241.2n ±  2%  -76.50% (p=0.000 n=10)
AccumRowInt32s/1048576-32                 262.34µ ±  2%   65.45µ ±  3%  -75.05% (p=0.000 n=10)
MinUint16s/64-32                           24.34n ±  3%   30.53n ±  3%  +25.43% (p=0.000 n=10)
MinUint16s/4096-32                       2206.50n ±  2%   70.02n ±  3%  -96.83% (p=0.000 n=10)
MinUint16s/1048576-32                     552.28µ ±  2%   15.49µ ±  3%  -97.20% (p=0.000 n=10)
ProductFloat32s/64-32                     19.145n ± 15%   7.659n ±  2%  -60.00% (p=0.000 n=10)
ProductFloat32s/4096-32                   2132.0n ±  2%   106.4n ±  2%  -95.01% (p=0.000 n=10)
ProductFloat32s/1048576-32                550.14µ ±  3%   32.27µ ±  2%  -94.13% (p=0.000 n=10)
ShiftAddInt32s/64-32                      12.930n ±  2%   7.700n ±  2%  -40.45% (p=0.000 n=10)
ShiftAddInt32s/4096-32                     827.7n ±  2%   269.8n ±  3%  -67.41% (p=0.000 n=10)
ShiftAddInt32s/1048576-32                 219.46µ ±  2%   73.07µ ±  3%  -66.70% (p=0.000 n=10)
ReadAfterStoreInt32s/64-32                 37.49n ±  5%   18.80n ±  1%  -49.86% (p=0.000 n=10)
ReadAfterStoreInt32s/4096-32              2449.5n ±  5%   575.8n ±  3%  -76.49% (p=0.000 n=10)
ReadAfterStoreInt32s/1048576-32            655.5µ ±  2%   156.8µ ±  3%  -76.08% (p=0.000 n=10)
geomean                                    1.871µ         435.4n        -76.73%

                                │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                                │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                         42.02Gi ± 33%    105.10Gi ±  1%   +150.10% (p=0.000 n=10)
AddFloat32s/4096-32                       51.92Gi ± 22%    237.27Gi ±  2%   +356.99% (p=0.000 n=10)
AddFloat32s/1048576-32                    54.69Gi ± 26%    146.61Gi ±  5%   +168.09% (p=0.000 n=10)
MulFloat32s/64-32                         44.00Gi ± 11%    112.47Gi ±  2%   +155.61% (p=0.000 n=10)
MulFloat32s/4096-32                       47.45Gi ± 11%    245.27Gi ±  2%   +416.89% (p=0.000 n=10)
MulFloat32s/1048576-32                    50.96Gi ±  8%    144.50Gi ±  7%   +183.57% (p=0.000 n=10)
ScalFloat32s/64-32                        48.17Gi ± 12%    128.47Gi ±  3%   +166.70% (p=0.000 n=10)
ScalFloat32s/4096-32                      41.02Gi ±  2%    209.24Gi ±  2%   +410.08% (p=0.000 n=10)
ScalFloat32s/1048576-32                   41.69Gi ±  2%    212.76Gi ±  3%   +410.35% (p=0.000 n=10)
MixFloat32s/64-32                         30.89Gi ±  2%     94.85Gi ±  3%   +207.09% (p=0.000 n=10)
MixFloat32s/4096-32                       30.86Gi ±  2%    213.17Gi ±  3%   +590.87% (p=0.000 n=10)
MixFloat32s/1048576-32                    30.84Gi ±  1%    125.56Gi ± 10%   +307.18% (p=0.000 n=10)
AxpyFloat32s/64-32                        54.29Gi ±  1%    107.56Gi ±  1%    +98.11% (p=0.000 n=10)
AxpyFloat32s/4096-32                      55.65Gi ±  4%    233.41Gi ±  2%   +319.44% (p=0.000 n=10)
AxpyFloat32s/1048576-32                   54.59Gi ±  4%    139.53Gi ±  9%   +155.58% (p=0.000 n=10)
NegFloat32s/64-32                         36.44Gi ±  8%     94.22Gi ±  3%   +158.58% (p=0.000 n=10)
NegFloat32s/4096-32                       40.34Gi ±  1%    172.80Gi ±  2%   +328.31% (p=0.000 n=10)
NegFloat32s/1048576-32                    39.25Gi ±  3%    133.38Gi ±  5%   +239.84% (p=0.000 n=10)
DivFloat32s/64-32                         25.55Gi ±  3%    101.47Gi ±  1%   +297.18% (p=0.000 n=10)
DivFloat32s/4096-32                       24.84Gi ±  3%    238.36Gi ±  2%   +859.43% (p=0.000 n=10)
DivFloat32s/1048576-32                    24.80Gi ±  2%    134.71Gi ±  7%   +443.11% (p=0.000 n=10)
DaxpyFloat32s/64-32                       56.76Gi ±  3%    132.00Gi ±  2%   +132.55% (p=0.000 n=10)
DaxpyFloat32s/4096-32                     60.51Gi ±  1%    260.63Gi ±  1%   +330.72% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                  58.49Gi ±  1%    182.45Gi ±  2%   +211.95% (p=0.000 n=10)
FillFloat32s/64-32                        20.03Gi ±  4%     68.53Gi ± 10%   +242.14% (p=0.000 n=10)
FillFloat32s/4096-32                      20.18Gi ±  6%    145.28Gi ±  3%   +619.76% (p=0.000 n=10)
FillFloat32s/1048576-32                   20.69Gi ±  3%    132.92Gi ±  1%   +542.55% (p=0.000 n=10)
FillUint8s/64-32                          4.933Gi ±  3%    28.414Gi ±  5%   +475.98% (p=0.000 n=10)
FillUint8s/4096-32                        5.031Gi ±  4%   150.487Gi ± 16%  +2891.40% (p=0.000 n=10)
FillUint8s/1048576-32                     5.171Gi ±  3%   162.242Gi ±  1%  +3037.70% (p=0.000 n=10)
ReverseIncFloat32s/64-32                  37.69Gi ±  6%     93.73Gi ±  2%   +148.68% (p=0.000 n=10)
ReverseIncFloat32s/4096-32                39.47Gi ±  7%    179.71Gi ± 11%   +355.31% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32             39.17Gi ± 48%    127.72Gi ±  6%   +226.11% (p=0.000 n=10)
CopyFloat32s/64-32                        38.23Gi ±  8%    116.82Gi ±  2%   +205.60% (p=0.000 n=10)
CopyFloat32s/4096-32                      40.67Gi ±  1%    421.77Gi ±  0%   +937.11% (p=0.000 n=10)
CopyFloat32s/1048576-32                   40.55Gi ±  2%    137.72Gi ±  4%   +239.58% (p=0.000 n=10)
AndNotUint64s/64-32                       60.76Gi ±  1%    148.12Gi ±  2%   +143.79% (p=0.000 n=10)
AndNotUint64s/4096-32                     62.10Gi ±  1%    236.23Gi ±  4%   +280.43% (p=0.000 n=10)
AndNotUint64s/1048576-32                  61.36Gi ±  2%    139.42Gi ±  8%   +127.22% (p=0.000 n=10)
NormalizeFloat32s/64-32                   33.60Gi ±  2%    147.32Gi ±  2%   +338.42% (p=0.000 n=10)
NormalizeFloat32s/4096-32                 33.64Gi ±  2%    266.44Gi ±  2%   +692.07% (p=0.000 n=10)
NormalizeFloat32s/1048576-32              33.30Gi ±  1%    226.52Gi ±  3%   +580.19% (p=0.000 n=10)
ChainInt32s/64-32                         48.05Gi ±  2%    110.94Gi ±  1%   +130.89% (p=0.000 n=10)
ChainInt32s/4096-32                       47.24Gi ±  4%    292.20Gi ±  6%   +518.55% (p=0.000 n=10)
ChainInt32s/1048576-32                    49.39Gi ±  2%    185.37Gi ± 14%   +275.30% (p=0.000 n=10)
WideInt32s/64-32                          31.43Gi ±  3%     99.21Gi ±  2%   +215.60% (p=0.000 n=10)
WideInt32s/4096-32                        31.65Gi ± 30%    219.78Gi ±  3%   +594.34% (p=0.000 n=10)
WideInt32s/1048576-32                     32.23Gi ±  2%    129.78Gi ± 11%   +302.72% (p=0.000 n=10)
SumInt32s/64-32                           21.29Gi ± 10%     31.37Gi ±  3%    +47.30% (p=0.000 n=10)
SumInt32s/4096-32                         20.85Gi ±  2%     76.48Gi ±  1%   +266.86% (p=0.000 n=10)
SumInt32s/1048576-32                      20.62Gi ±  2%     81.86Gi ±  1%   +297.01% (p=0.000 n=10)
DotInt32s/64-32                           34.28Gi ±  3%     51.60Gi ±  1%    +50.52% (p=0.000 n=10)
DotInt32s/4096-32                         32.22Gi ±  3%    148.38Gi ±  3%   +360.53% (p=0.000 n=10)
DotInt32s/1048576-32                      31.89Gi ±  3%    122.61Gi ±  2%   +284.47% (p=0.000 n=10)
SumFloat32s/64-32                         22.14Gi ±  2%     17.84Gi ±  2%    -19.45% (p=0.000 n=10)
SumFloat32s/4096-32                       10.81Gi ±  3%    136.35Gi ±  1%  +1161.32% (p=0.000 n=10)
SumFloat32s/1048576-32                    10.53Gi ±  2%    122.62Gi ±  5%  +1064.52% (p=0.000 n=10)
DotFloat32s/64-32                         31.14Gi ± 28%     35.65Gi ±  2%          ~ (p=0.481 n=10)
DotFloat32s/4096-32                       21.51Gi ±  3%    168.53Gi ±  2%   +683.48% (p=0.000 n=10)
DotFloat32s/1048576-32                    21.04Gi ±  3%    122.08Gi ±  4%   +480.18% (p=0.000 n=10)
DotFloat64s/64-32                         59.92Gi ±  2%     77.09Gi ±  2%    +28.64% (p=0.000 n=10)
DotFloat64s/4096-32                       41.55Gi ±  3%    166.32Gi ±  3%   +300.29% (p=0.000 n=10)
DotFloat64s/1048576-32                    40.91Gi ±  2%    120.88Gi ± 14%   +195.44% (p=0.000 n=10)
AbsFloat32s/64-32                         20.74Gi ±  1%     97.66Gi ±  2%   +370.99% (p=0.000 n=10)
AbsFloat32s/4096-32                       20.88Gi ±  1%    182.38Gi ±  1%   +773.28% (p=0.000 n=10)
AbsFloat32s/1048576-32                    20.66Gi ±  2%    136.57Gi ±  3%   +561.01% (p=0.000 n=10)
SqrtFloat32s/64-32                        9.090Gi ±  2%    91.563Gi ±  2%   +907.34% (p=0.000 n=10)
SqrtFloat32s/4096-32                      8.968Gi ±  1%   134.364Gi ±  5%  +1398.30% (p=0.000 n=10)
SqrtFloat32s/1048576-32                   9.171Gi ±  1%   127.651Gi ±  4%  +1291.86% (p=0.000 n=10)
ClampInt32s/64-32                         16.32Gi ±  0%     99.96Gi ±  1%   +512.37% (p=0.000 n=10)
ClampInt32s/4096-32                       16.38Gi ±  3%    204.85Gi ±  1%  +1150.55% (p=0.000 n=10)
ClampInt32s/1048576-32                    16.31Gi ±  3%    132.42Gi ±  4%   +711.82% (p=0.000 n=10)
ShrInt32s/64-32                           36.90Gi ±  2%     98.70Gi ±  2%   +167.44% (p=0.000 n=10)
ShrInt32s/4096-32                         36.35Gi ±  3%    195.54Gi ±  2%   +437.91% (p=0.000 n=10)
ShrInt32s/1048576-32                      36.90Gi ±  2%    126.09Gi ±  7%   +241.72% (p=0.000 n=10)
ShlUint32s/64-32                          36.91Gi ±  3%     95.94Gi ±  3%   +159.91% (p=0.000 n=10)
ShlUint32s/4096-32                        36.15Gi ±  2%    204.16Gi ±  1%   +464.73% (p=0.000 n=10)
ShlUint32s/1048576-32                     36.32Gi ±  1%    134.84Gi ±  5%   +271.30% (p=0.000 n=10)
DiffInt32s/64-32                          36.41Gi ±  1%     55.68Gi ±  1%    +52.93% (p=0.000 n=10)
DiffInt32s/4096-32                        35.66Gi ±  2%     71.62Gi ± 10%   +100.84% (p=0.000 n=10)
DiffInt32s/1048576-32                     35.20Gi ±  3%     72.98Gi ±  2%   +107.34% (p=0.000 n=10)
SmoothInt32s/64-32                        21.07Gi ± 21%     50.19Gi ±  1%   +138.26% (p=0.000 n=10)
SmoothInt32s/4096-32                      16.38Gi ±  8%     57.70Gi ± 12%   +252.35% (p=0.000 n=10)
SmoothInt32s/1048576-32                   16.00Gi ±  2%     64.60Gi ±  1%   +303.71% (p=0.000 n=10)
ShiftInt32s/64-32                         30.69Gi ±  2%     91.10Gi ±  2%   +196.80% (p=0.000 n=10)
ShiftInt32s/4096-32                       29.09Gi ±  2%    157.44Gi ±  2%   +441.24% (p=0.000 n=10)
ShiftInt32s/1048576-32                    29.68Gi ±  4%    129.31Gi ±  5%   +335.68% (p=0.000 n=10)
AccumRowInt32s/64-32                      45.29Gi ±  2%    110.53Gi ±  1%   +144.03% (p=0.000 n=10)
AccumRowInt32s/4096-32                    44.63Gi ±  3%    189.85Gi ±  2%   +325.42% (p=0.000 n=10)
AccumRowInt32s/1048576-32                 44.67Gi ±  2%    179.05Gi ±  4%   +300.80% (p=0.000 n=10)
MinUint16s/64-32                          4.898Gi ±  3%     3.905Gi ±  3%    -20.28% (p=0.000 n=10)
MinUint16s/4096-32                        3.457Gi ±  3%   108.970Gi ±  3%  +3052.16% (p=0.000 n=10)
MinUint16s/1048576-32                     3.536Gi ±  2%   126.086Gi ±  3%  +3465.34% (p=0.000 n=10)
ProductFloat32s/64-32                     12.45Gi ± 18%     31.13Gi ±  2%   +149.96% (p=0.000 n=10)
ProductFloat32s/4096-32                   7.157Gi ±  2%   143.530Gi ±  2%  +1905.52% (p=0.000 n=10)
ProductFloat32s/1048576-32                7.100Gi ±  3%   121.046Gi ±  2%  +1604.76% (p=0.000 n=10)
ShiftAddInt32s/64-32                      55.31Gi ±  2%     92.90Gi ±  2%    +67.95% (p=0.000 n=10)
ShiftAddInt32s/4096-32                    55.31Gi ±  2%    169.71Gi ±  3%   +206.84% (p=0.000 n=10)
ShiftAddInt32s/1048576-32                 53.40Gi ±  3%    160.38Gi ±  3%   +200.34% (p=0.000 n=10)
ReadAfterStoreInt32s/64-32                19.08Gi ±  5%     38.06Gi ±  1%    +99.46% (p=0.000 n=10)
ReadAfterStoreInt32s/4096-32              18.69Gi ±  5%     79.51Gi ±  3%   +325.46% (p=0.000 n=10)
ReadAfterStoreInt32s/1048576-32           17.88Gi ±  2%     74.73Gi ±  3%   +318.02% (p=0.000 n=10)
geomean                                   27.21Gi           116.9Gi         +329.71%
```
</details>

Filling a byte slice with a non-zero value (`FillUint8s`, 1M elements) is over 30x
faster, +3000% throughput.

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
the loop, with no `if`; `dst[i+1]` and `a[0]` are not rewritten.

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
