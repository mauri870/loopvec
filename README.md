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
ranging from 2.3x to 31x depending on the kernel

<details>
<summary>bench.txt</summary>

```
goos: linux
goarch: amd64
pkg: github.com/mauri870/loopvec/bench
cpu: AMD Ryzen 9 9950X3D 16-Core Processor          
                                │ /tmp/bench_scalar.txt │         /tmp/bench_simd.txt         │
                                │        sec/op         │    sec/op     vs base               │
AddFloat32s/64-32                        12.980n ±   5%   6.823n ± 29%  -47.43% (p=0.002 n=6)
AddFloat32s/4096-32                       827.9n ±   6%   199.2n ±  2%  -75.95% (p=0.002 n=6)
AddFloat32s/1048576-32                   216.57µ ±  68%   84.71µ ± 11%  -60.88% (p=0.002 n=6)
MulFloat32s/64-32                        12.705n ±   2%   7.710n ± 18%  -39.32% (p=0.002 n=6)
MulFloat32s/4096-32                       820.1n ±   6%   228.8n ± 22%  -72.11% (p=0.002 n=6)
MulFloat32s/1048576-32                   213.35µ ±   3%   85.21µ ±  8%  -60.06% (p=0.002 n=6)
ScalFloat32s/64-32                       10.046n ±  17%   3.771n ±  3%  -62.47% (p=0.002 n=6)
ScalFloat32s/4096-32                      742.6n ±   2%   148.0n ±  4%  -80.06% (p=0.002 n=6)
ScalFloat32s/1048576-32                  187.93µ ±   3%   40.03µ ± 11%  -78.70% (p=0.002 n=6)
MixFloat32s/64-32                        23.330n ±  58%   7.381n ± 13%  -68.36% (p=0.002 n=6)
MixFloat32s/4096-32                      1470.5n ±   1%   208.2n ± 18%  -85.84% (p=0.002 n=6)
MixFloat32s/1048576-32                   387.75µ ±   4%   87.09µ ±  8%  -77.54% (p=0.002 n=6)
AxpyFloat32s/64-32                       13.185n ±  30%   7.594n ±  2%  -42.41% (p=0.002 n=6)
AxpyFloat32s/4096-32                      801.8n ±  16%   190.7n ±  3%  -76.22% (p=0.002 n=6)
AxpyFloat32s/1048576-32                  201.80µ ±   9%   88.44µ ± 12%  -56.17% (p=0.002 n=6)
NegFloat32s/64-32                        14.035n ±  12%   5.178n ±  1%  -63.11% (p=0.002 n=6)
NegFloat32s/4096-32                       778.9n ±   9%   170.5n ±  3%  -78.11% (p=0.002 n=6)
NegFloat32s/1048576-32                   207.34µ ±   4%   60.25µ ±  5%  -70.94% (p=0.002 n=6)
DivFloat32s/64-32                        28.155n ±   3%   7.718n ±  4%  -72.59% (p=0.002 n=6)
DivFloat32s/4096-32                      1801.5n ±   2%   224.1n ±  8%  -87.56% (p=0.002 n=6)
DivFloat32s/1048576-32                   468.58µ ±   2%   88.85µ ± 25%  -81.04% (p=0.002 n=6)
DaxpyFloat32s/64-32                      13.890n ±  16%   6.425n ± 18%  -53.75% (p=0.002 n=6)
DaxpyFloat32s/4096-32                     841.4n ±  13%   250.6n ±  5%  -70.22% (p=0.002 n=6)
DaxpyFloat32s/1048576-32                 221.99µ ±  11%   69.30µ ± 10%  -68.78% (p=0.002 n=6)
FillFloat32s/64-32                       12.335n ±   2%   2.938n ± 14%  -76.18% (p=0.002 n=6)
FillFloat32s/4096-32                      795.9n ±   8%   105.7n ± 42%  -86.72% (p=0.002 n=6)
FillFloat32s/1048576-32                  190.53µ ±  68%   30.29µ ±  6%  -84.10% (p=0.002 n=6)
FillUint8s/64-32                         12.335n ±   1%   2.101n ± 10%  -82.97% (p=0.002 n=6)
FillUint8s/4096-32                       759.05n ±   1%   24.64n ± 48%  -96.75% (p=0.002 n=6)
FillUint8s/1048576-32                   189.414µ ±   4%   6.132µ ±  3%  -96.76% (p=0.002 n=6)
ReverseIncFloat32s/64-32                 13.300n ±  26%   4.630n ±  4%  -65.18% (p=0.002 n=6)
ReverseIncFloat32s/4096-32                866.0n ±  11%   140.4n ±  7%  -83.79% (p=0.002 n=6)
ReverseIncFloat32s/1048576-32            229.16µ ±  10%   57.96µ ±  5%  -74.71% (p=0.002 n=6)
CopyFloat32s/64-32                       13.705n ±  17%   3.895n ±  2%  -71.58% (p=0.002 n=6)
CopyFloat32s/4096-32                     813.90n ±  88%   73.19n ±  2%  -91.01% (p=0.002 n=6)
CopyFloat32s/1048576-32                  191.10µ ±   2%   53.07µ ±  8%  -72.23% (p=0.002 n=6)
AndNotUint64s/64-32                       23.62n ±   1%   10.21n ±  9%  -56.80% (p=0.002 n=6)
AndNotUint64s/4096-32                    1510.0n ±   2%   419.5n ±  7%  -72.22% (p=0.002 n=6)
AndNotUint64s/1048576-32                  392.8µ ±  13%   178.6µ ± 16%  -54.53% (p=0.002 n=6)
NormalizeFloat32s/64-32                  28.895n ±   1%   6.613n ±  2%  -77.11% (p=0.002 n=6)
NormalizeFloat32s/4096-32                1837.5n ±   2%   242.1n ±  2%  -86.82% (p=0.002 n=6)
NormalizeFloat32s/1048576-32             475.35µ ±   7%   70.04µ ± 15%  -85.27% (p=0.002 n=6)
ChainInt32s/64-32                         30.13n ±   5%   14.78n ±  7%  -50.95% (p=0.002 n=6)
ChainInt32s/4096-32                      1994.5n ±  47%   375.4n ± 10%  -81.18% (p=0.002 n=6)
ChainInt32s/1048576-32                    482.9µ ±  30%   131.6µ ± 16%  -72.76% (p=0.002 n=6)
WideInt32s/64-32                          46.78n ±   1%   10.75n ±  6%  -77.03% (p=0.002 n=6)
WideInt32s/4096-32                       2966.5n ±   1%   301.1n ± 16%  -89.85% (p=0.002 n=6)
WideInt32s/1048576-32                     766.1µ ±   9%   136.0µ ±  4%  -82.25% (p=0.002 n=6)
SumInt32s/64-32                          12.685n ±   8%   7.848n ± 43%  -38.13% (p=0.002 n=6)
SumInt32s/4096-32                         756.1n ±   3%   114.4n ±  3%  -84.88% (p=0.002 n=6)
SumInt32s/1048576-32                     187.84µ ±   8%   31.32µ ± 19%  -83.33% (p=0.002 n=6)
DotInt32s/64-32                          15.275n ±   6%   8.684n ±  5%  -43.15% (p=0.002 n=6)
DotInt32s/4096-32                         945.8n ±  27%   195.2n ±  8%  -79.36% (p=0.002 n=6)
DotInt32s/1048576-32                     242.93µ ±  29%   64.34µ ± 33%  -73.51% (p=0.002 n=6)
SumFloat32s/64-32                         11.42n ±  32%   13.37n ±  1%        ~ (p=0.065 n=6)
SumFloat32s/4096-32                      1474.5n ±   1%   108.5n ±  3%  -92.64% (p=0.002 n=6)
SumFloat32s/1048576-32                   383.91µ ±   2%   28.14µ ±  9%  -92.67% (p=0.002 n=6)
DotFloat32s/64-32                         16.47n ±   1%   13.57n ±  4%  -17.66% (p=0.002 n=6)
DotFloat32s/4096-32                      1493.0n ±   1%   227.6n ± 16%  -84.75% (p=0.002 n=6)
DotFloat32s/1048576-32                   383.42µ ±   3%   62.53µ ± 21%  -83.69% (p=0.002 n=6)
DotFloat64s/64-32                         12.73n ±  23%   12.24n ±  1%        ~ (p=0.372 n=6)
DotFloat64s/4096-32                      1449.5n ±   2%   372.5n ±  2%  -74.30% (p=0.002 n=6)
DotFloat64s/1048576-32                    380.6µ ±   3%   120.6µ ± 32%  -68.31% (p=0.002 n=6)
AbsFloat32s/64-32                        25.595n ±   6%   5.266n ±  4%  -79.43% (p=0.002 n=6)
AbsFloat32s/4096-32                      1486.0n ±   6%   171.8n ±  3%  -88.44% (p=0.002 n=6)
AbsFloat32s/1048576-32                   385.40µ ±   4%   58.36µ ±  3%  -84.86% (p=0.002 n=6)
SqrtFloat32s/64-32                       52.450n ±   1%   5.207n ±  4%  -90.07% (p=0.002 n=6)
SqrtFloat32s/4096-32                     3301.5n ±   3%   211.9n ±  2%  -93.58% (p=0.002 n=6)
SqrtFloat32s/1048576-32                  852.41µ ±   3%   61.64µ ± 20%  -92.77% (p=0.002 n=6)
ClampInt32s/64-32                        29.165n ±   2%   4.780n ±  1%  -83.61% (p=0.002 n=6)
ClampInt32s/4096-32                      1873.5n ±   3%   150.8n ±  2%  -91.95% (p=0.002 n=6)
ClampInt32s/1048576-32                   477.27µ ±   6%   59.67µ ±  4%  -87.50% (p=0.002 n=6)
ShrInt32s/64-32                          13.135n ±  17%   4.553n ±  1%  -65.34% (p=0.002 n=6)
ShrInt32s/4096-32                         842.7n ±   2%   149.5n ±  2%  -82.26% (p=0.002 n=6)
ShrInt32s/1048576-32                     214.71µ ±   3%   57.31µ ±  5%  -73.31% (p=0.002 n=6)
ShlUint32s/64-32                         12.940n ±   2%   4.541n ±  1%  -64.91% (p=0.002 n=6)
ShlUint32s/4096-32                        844.7n ±  22%   146.1n ±  1%  -82.70% (p=0.002 n=6)
ShlUint32s/1048576-32                    218.23µ ±   4%   58.13µ ±  4%  -73.36% (p=0.002 n=6)
DiffInt32s/64-32                         14.055n ±  33%   7.537n ±  2%  -46.37% (p=0.002 n=6)
DiffInt32s/4096-32                        917.5n ±  27%   294.0n ± 23%  -67.96% (p=0.002 n=6)
DiffInt32s/1048576-32                    248.58µ ±  50%   74.91µ ±  4%  -69.87% (p=0.002 n=6)
SmoothInt32s/64-32                       21.990n ±   3%   9.513n ±  4%  -56.74% (p=0.002 n=6)
SmoothInt32s/4096-32                     1782.5n ±   3%   424.2n ± 40%  -76.20% (p=0.002 n=6)
SmoothInt32s/1048576-32                   494.0µ ±   2%   111.9µ ±  5%  -77.36% (p=0.002 n=6)
ShiftInt32s/64-32                        16.810n ±   8%   5.628n ±  2%  -66.52% (p=0.002 n=6)
ShiftInt32s/4096-32                      1095.0n ±   3%   184.3n ±  5%  -83.16% (p=0.002 n=6)
ShiftInt32s/1048576-32                   264.11µ ±  10%   62.23µ ±  7%  -76.44% (p=0.002 n=6)
AccumRowInt32s/64-32                     18.545n ±  16%   6.483n ± 16%  -65.04% (p=0.002 n=6)
AccumRowInt32s/4096-32                   1037.0n ±  14%   244.8n ±  9%  -76.40% (p=0.002 n=6)
AccumRowInt32s/1048576-32                271.66µ ±  17%   64.72µ ± 18%  -76.18% (p=0.002 n=6)
MinUint16s/64-32                          25.05n ±   2%   31.41n ±  1%  +25.41% (p=0.002 n=6)
MinUint16s/4096-32                      2224.00n ±   1%   66.00n ± 23%  -97.03% (p=0.002 n=6)
MinUint16s/1048576-32                    572.67µ ±   2%   14.84µ ± 14%  -97.41% (p=0.002 n=6)
ProductFloat32s/64-32                    21.585n ±   3%   6.511n ±  2%  -69.83% (p=0.002 n=6)
ProductFloat32s/4096-32                  2136.0n ±   3%   110.6n ±  2%  -94.82% (p=0.002 n=6)
ProductFloat32s/1048576-32               562.27µ ±   2%   31.52µ ±  6%  -94.39% (p=0.002 n=6)
ShiftAddInt32s/64-32                     13.605n ±  20%   7.557n ±  1%  -44.45% (p=0.002 n=6)
ShiftAddInt32s/4096-32                    834.6n ±   6%   266.2n ±  4%  -68.10% (p=0.002 n=6)
ShiftAddInt32s/1048576-32                221.59µ ±   3%   69.88µ ±  7%  -68.46% (p=0.002 n=6)
ReadAfterStoreInt32s/64-32                32.54n ±   7%   19.19n ±  7%  -41.03% (p=0.002 n=6)
ReadAfterStoreInt32s/4096-32             2558.5n ±   8%   532.8n ±  6%  -79.18% (p=0.002 n=6)
ReadAfterStoreInt32s/1048576-32           646.4µ ±   8%   147.6µ ±  8%  -77.16% (p=0.002 n=6)
ReLUFloat32s/64-32                       12.240n ±   8%   3.850n ±  2%  -68.55% (p=0.002 n=6)
ReLUFloat32s/4096-32                      738.8n ±   2%   149.1n ±  4%  -79.81% (p=0.002 n=6)
ReLUFloat32s/1048576-32                  203.99µ ±  25%   38.41µ ±  3%  -81.17% (p=0.002 n=6)
LeakyReLUFloat32s/64-32                  23.265n ±  12%   5.077n ±  2%  -78.18% (p=0.002 n=6)
LeakyReLUFloat32s/4096-32                1448.0n ±   8%   149.1n ±  2%  -89.70% (p=0.002 n=6)
LeakyReLUFloat32s/1048576-32             405.97µ ±   9%   57.96µ ±  5%  -85.72% (p=0.002 n=6)
ReLU6Float32s/64-32                      22.810n ±   2%   4.834n ±  1%  -78.81% (p=0.002 n=6)
ReLU6Float32s/4096-32                    1469.0n ±   2%   233.8n ±  3%  -84.08% (p=0.002 n=6)
ReLU6Float32s/1048576-32                 460.63µ ±  39%   59.64µ ±  3%  -87.05% (p=0.002 n=6)
SignFloat32s/64-32                       25.655n ±  66%   5.180n ±  2%  -79.81% (p=0.002 n=6)
SignFloat32s/4096-32                     1565.0n ±  10%   177.6n ±  6%  -88.65% (p=0.002 n=6)
SignFloat32s/1048576-32                  387.07µ ±   8%   59.82µ ±  6%  -84.55% (p=0.002 n=6)
FillNaNFloat64s/64-32                    12.070n ±   2%   3.541n ±  8%  -70.67% (p=0.002 n=6)
FillNaNFloat64s/4096-32                   758.9n ±   4%   150.7n ± 34%  -80.14% (p=0.002 n=6)
FillNaNFloat64s/1048576-32               193.59µ ± 157%   59.90µ ±  8%  -69.06% (p=0.002 n=6)
ScaleFieldFloat32s/64-32                 12.765n ±   4%   4.174n ±  4%  -67.30% (p=0.002 n=6)
ScaleFieldFloat32s/4096-32                774.8n ±  15%   195.0n ± 15%  -74.83% (p=0.002 n=6)
ScaleFieldFloat32s/1048576-32            191.79µ ±   8%   42.60µ ±  6%  -77.79% (p=0.002 n=6)
AddFieldFloat32s/64-32                   26.235n ±   6%   6.431n ±  4%  -75.49% (p=0.002 n=6)
AddFieldFloat32s/4096-32                 1561.0n ±   5%   256.2n ±  1%  -83.59% (p=0.002 n=6)
AddFieldFloat32s/1048576-32              394.35µ ±   5%   70.31µ ±  5%  -82.17% (p=0.002 n=6)
HardSwishFloat32s/64-32                  30.430n ±  30%   5.984n ±  5%  -80.34% (p=0.002 n=6)
HardSwishFloat32s/4096-32                1891.5n ±   4%   216.4n ±  4%  -88.56% (p=0.002 n=6)
HardSwishFloat32s/1048576-32             480.45µ ±   2%   64.03µ ±  4%  -86.67% (p=0.002 n=6)
HardSigmoidFloat32s/64-32                29.100n ±   1%   6.234n ±  2%  -78.58% (p=0.002 n=6)
HardSigmoidFloat32s/4096-32              1865.5n ±   2%   249.9n ± 17%  -86.61% (p=0.002 n=6)
HardSigmoidFloat32s/1048576-32           467.35µ ±   5%   65.82µ ±  7%  -85.92% (p=0.002 n=6)
geomean                                   1.947µ          420.0n        -78.42%

                                │ /tmp/bench_scalar.txt │           /tmp/bench_simd.txt            │
                                │          B/s          │       B/s        vs base                 │
AddFloat32s/64-32                         55.12Gi ±  5%    104.83Gi ± 22%    +90.20% (p=0.002 n=6)
AddFloat32s/4096-32                       55.29Gi ±  6%    229.89Gi ±  2%   +315.77% (p=0.002 n=6)
AddFloat32s/1048576-32                    54.12Gi ± 40%    138.54Gi ± 10%   +155.98% (p=0.002 n=6)
MulFloat32s/64-32                         56.31Gi ±  2%     92.77Gi ± 16%    +64.75% (p=0.002 n=6)
MulFloat32s/4096-32                       55.82Gi ±  5%    200.47Gi ± 18%   +259.13% (p=0.002 n=6)
MulFloat32s/1048576-32                    54.93Gi ±  3%    137.74Gi ±  8%   +150.75% (p=0.002 n=6)
ScalFloat32s/64-32                        47.46Gi ± 15%    126.46Gi ±  3%   +166.46% (p=0.002 n=6)
ScalFloat32s/4096-32                      41.09Gi ±  2%    206.17Gi ±  4%   +401.71% (p=0.002 n=6)
ScalFloat32s/1048576-32                   41.57Gi ±  3%    195.54Gi ± 10%   +370.37% (p=0.002 n=6)
MixFloat32s/64-32                         30.66Gi ± 37%     96.94Gi ± 11%   +216.19% (p=0.002 n=6)
MixFloat32s/4096-32                       31.14Gi ±  1%    219.94Gi ± 15%   +606.37% (p=0.002 n=6)
MixFloat32s/1048576-32                    30.22Gi ±  3%    134.56Gi ±  7%   +345.21% (p=0.002 n=6)
AxpyFloat32s/64-32                        54.25Gi ± 23%     94.20Gi ±  2%    +73.63% (p=0.002 n=6)
AxpyFloat32s/4096-32                      57.09Gi ± 14%    240.07Gi ±  3%   +320.51% (p=0.002 n=6)
AxpyFloat32s/1048576-32                   58.07Gi ±  9%    132.51Gi ± 14%   +128.18% (p=0.002 n=6)
NegFloat32s/64-32                         34.02Gi ± 14%     92.09Gi ±  1%   +170.68% (p=0.002 n=6)
NegFloat32s/4096-32                       39.18Gi ±  8%    179.02Gi ±  3%   +356.90% (p=0.002 n=6)
NegFloat32s/1048576-32                    37.68Gi ±  5%    129.70Gi ±  5%   +244.19% (p=0.002 n=6)
DivFloat32s/64-32                         25.41Gi ±  3%     92.67Gi ±  4%   +264.76% (p=0.002 n=6)
DivFloat32s/4096-32                       25.41Gi ±  2%    204.52Gi ±  9%   +704.91% (p=0.002 n=6)
DivFloat32s/1048576-32                    25.01Gi ±  2%    131.89Gi ± 20%   +427.32% (p=0.002 n=6)
DaxpyFloat32s/64-32                       51.50Gi ± 14%    111.33Gi ± 16%   +116.17% (p=0.002 n=6)
DaxpyFloat32s/4096-32                     54.41Gi ± 14%    182.65Gi ±  5%   +235.67% (p=0.002 n=6)
DaxpyFloat32s/1048576-32                  52.81Gi ± 13%    169.26Gi ±  9%   +220.52% (p=0.002 n=6)
FillFloat32s/64-32                        19.33Gi ±  2%     81.19Gi ± 13%   +320.08% (p=0.002 n=6)
FillFloat32s/4096-32                      19.18Gi ±  8%    145.03Gi ± 30%   +656.30% (p=0.002 n=6)
FillFloat32s/1048576-32                   20.52Gi ± 41%    128.98Gi ±  5%   +528.72% (p=0.002 n=6)
FillUint8s/64-32                          4.832Gi ±  1%    28.374Gi ±  9%   +487.22% (p=0.002 n=6)
FillUint8s/4096-32                        5.026Gi ±  1%   154.848Gi ± 32%  +2981.06% (p=0.002 n=6)
FillUint8s/1048576-32                     5.156Gi ±  3%   159.251Gi ±  3%  +2988.82% (p=0.002 n=6)
ReverseIncFloat32s/64-32                  35.85Gi ± 20%    102.97Gi ±  4%   +187.25% (p=0.002 n=6)
ReverseIncFloat32s/4096-32                35.24Gi ± 12%    217.48Gi ±  7%   +517.11% (p=0.002 n=6)
ReverseIncFloat32s/1048576-32             34.09Gi ± 11%    134.87Gi ±  6%   +295.58% (p=0.002 n=6)
CopyFloat32s/64-32                        34.81Gi ± 15%    122.41Gi ±  2%   +251.70% (p=0.002 n=6)
CopyFloat32s/4096-32                      37.67Gi ± 47%    417.00Gi ±  2%  +1006.94% (p=0.002 n=6)
CopyFloat32s/1048576-32                   40.88Gi ±  2%    147.23Gi ±  7%   +260.13% (p=0.002 n=6)
AndNotUint64s/64-32                       60.55Gi ±  1%    140.20Gi ±  8%   +131.55% (p=0.002 n=6)
AndNotUint64s/4096-32                     60.63Gi ±  2%    218.23Gi ±  8%   +259.91% (p=0.002 n=6)
AndNotUint64s/1048576-32                  59.67Gi ± 11%    131.43Gi ± 17%   +120.26% (p=0.002 n=6)
NormalizeFloat32s/64-32                   33.00Gi ±  1%    144.20Gi ±  2%   +336.95% (p=0.002 n=6)
NormalizeFloat32s/4096-32                 33.22Gi ±  2%    252.15Gi ±  2%   +659.08% (p=0.002 n=6)
NormalizeFloat32s/1048576-32              32.87Gi ±  6%    223.10Gi ± 13%   +578.72% (p=0.002 n=6)
ChainInt32s/64-32                         47.47Gi ±  4%     96.79Gi ±  6%   +103.89% (p=0.002 n=6)
ChainInt32s/4096-32                       45.95Gi ± 32%    243.92Gi ± 11%   +430.82% (p=0.002 n=6)
ChainInt32s/1048576-32                    48.53Gi ± 23%    178.67Gi ± 14%   +268.14% (p=0.002 n=6)
WideInt32s/64-32                          20.39Gi ±  1%     88.74Gi ±  6%   +335.26% (p=0.002 n=6)
WideInt32s/4096-32                        20.58Gi ±  1%    202.73Gi ± 14%   +885.20% (p=0.002 n=6)
WideInt32s/1048576-32                     20.40Gi ±  8%    114.91Gi ±  5%   +463.44% (p=0.002 n=6)
SumInt32s/64-32                           18.81Gi ±  8%     30.38Gi ± 30%    +61.54% (p=0.002 n=6)
SumInt32s/4096-32                         20.18Gi ±  3%    133.45Gi ±  3%   +561.25% (p=0.002 n=6)
SumInt32s/1048576-32                      20.80Gi ±  7%    124.73Gi ± 23%   +499.79% (p=0.002 n=6)
DotInt32s/64-32                           31.22Gi ±  6%     54.91Gi ±  4%    +75.87% (p=0.002 n=6)
DotInt32s/4096-32                         32.27Gi ± 21%    156.29Gi ±  7%   +384.35% (p=0.002 n=6)
DotInt32s/1048576-32                      32.16Gi ± 22%    121.53Gi ± 48%   +277.87% (p=0.002 n=6)
SumFloat32s/64-32                         20.89Gi ± 24%     17.83Gi ±  1%          ~ (p=0.065 n=6)
SumFloat32s/4096-32                       10.35Gi ±  1%    140.73Gi ±  3%  +1259.54% (p=0.002 n=6)
SumFloat32s/1048576-32                    10.17Gi ±  2%    138.82Gi ±  8%  +1264.33% (p=0.002 n=6)
DotFloat32s/64-32                         28.95Gi ±  1%     35.15Gi ±  3%    +21.44% (p=0.002 n=6)
DotFloat32s/4096-32                       20.44Gi ±  1%    136.64Gi ± 17%   +568.44% (p=0.002 n=6)
DotFloat32s/1048576-32                    20.38Gi ±  3%    124.95Gi ± 26%   +513.21% (p=0.002 n=6)
DotFloat64s/64-32                         74.91Gi ± 19%     77.91Gi ±  1%          ~ (p=0.394 n=6)
DotFloat64s/4096-32                       42.10Gi ±  2%    163.85Gi ±  2%   +289.19% (p=0.002 n=6)
DotFloat64s/1048576-32                    41.06Gi ±  3%    129.54Gi ± 24%   +215.51% (p=0.002 n=6)
AbsFloat32s/64-32                         18.63Gi ±  6%     90.56Gi ±  4%   +386.07% (p=0.002 n=6)
AbsFloat32s/4096-32                       20.54Gi ±  6%    177.70Gi ±  3%   +765.16% (p=0.002 n=6)
AbsFloat32s/1048576-32                    20.27Gi ±  4%    133.86Gi ±  3%   +560.37% (p=0.002 n=6)
SqrtFloat32s/64-32                        9.091Gi ±  1%    91.584Gi ±  5%   +907.40% (p=0.002 n=6)
SqrtFloat32s/4096-32                      9.244Gi ±  3%   143.988Gi ±  2%  +1457.57% (p=0.002 n=6)
SqrtFloat32s/1048576-32                   9.165Gi ±  3%   126.738Gi ± 17%  +1282.82% (p=0.002 n=6)
ClampInt32s/64-32                         16.35Gi ±  2%     99.76Gi ±  1%   +510.09% (p=0.002 n=6)
ClampInt32s/4096-32                       16.29Gi ±  3%    202.36Gi ±  2%  +1142.25% (p=0.002 n=6)
ClampInt32s/1048576-32                    16.37Gi ±  6%    130.93Gi ±  4%   +699.86% (p=0.002 n=6)
ShrInt32s/64-32                           36.30Gi ± 14%    104.75Gi ±  1%   +188.53% (p=0.002 n=6)
ShrInt32s/4096-32                         36.22Gi ±  2%    204.19Gi ±  2%   +463.80% (p=0.002 n=6)
ShrInt32s/1048576-32                      36.39Gi ±  3%    136.33Gi ±  5%   +274.68% (p=0.002 n=6)
ShlUint32s/64-32                          36.86Gi ±  2%    105.01Gi ±  1%   +184.93% (p=0.002 n=6)
ShlUint32s/4096-32                        36.13Gi ± 18%    208.85Gi ±  1%   +478.07% (p=0.002 n=6)
ShlUint32s/1048576-32                     35.80Gi ±  4%    134.39Gi ±  4%   +275.40% (p=0.002 n=6)
DiffInt32s/64-32                          33.97Gi ± 25%     63.27Gi ±  2%    +86.26% (p=0.002 n=6)
DiffInt32s/4096-32                        33.34Gi ± 22%    103.82Gi ± 19%   +211.42% (p=0.002 n=6)
DiffInt32s/1048576-32                     31.45Gi ± 33%    104.29Gi ±  4%   +231.67% (p=0.002 n=6)
SmoothInt32s/64-32                        21.69Gi ±  2%     50.13Gi ±  4%   +131.15% (p=0.002 n=6)
SmoothInt32s/4096-32                      17.12Gi ±  2%     71.94Gi ± 28%   +320.14% (p=0.002 n=6)
SmoothInt32s/1048576-32                   15.82Gi ±  2%     69.84Gi ±  5%   +341.61% (p=0.002 n=6)
ShiftInt32s/64-32                         28.36Gi ±  9%     84.73Gi ±  2%   +198.71% (p=0.002 n=6)
ShiftInt32s/4096-32                       27.86Gi ±  3%    165.52Gi ±  5%   +494.12% (p=0.002 n=6)
ShiftInt32s/1048576-32                    29.58Gi ±  9%    125.66Gi ±  7%   +324.81% (p=0.002 n=6)
AccumRowInt32s/64-32                      38.58Gi ± 19%    110.33Gi ± 14%   +185.99% (p=0.002 n=6)
AccumRowInt32s/4096-32                    44.13Gi ± 12%    187.16Gi ±  8%   +324.07% (p=0.002 n=6)
AccumRowInt32s/1048576-32                 43.15Gi ± 14%    181.07Gi ± 16%   +319.61% (p=0.002 n=6)
MinUint16s/64-32                          4.759Gi ±  2%     3.795Gi ±  1%    -20.26% (p=0.002 n=6)
MinUint16s/4096-32                        3.430Gi ±  1%   115.608Gi ± 19%  +3270.08% (p=0.002 n=6)
MinUint16s/1048576-32                     3.411Gi ±  2%   131.605Gi ± 17%  +3758.76% (p=0.002 n=6)
ProductFloat32s/64-32                     11.05Gi ±  2%     36.62Gi ±  2%   +231.50% (p=0.002 n=6)
ProductFloat32s/4096-32                   7.144Gi ±  3%   137.962Gi ±  2%  +1831.22% (p=0.002 n=6)
ProductFloat32s/1048576-32                6.947Gi ±  2%   123.938Gi ±  6%  +1683.93% (p=0.002 n=6)
ShiftAddInt32s/64-32                      52.58Gi ± 17%     94.65Gi ±  1%    +80.00% (p=0.002 n=6)
ShiftAddInt32s/4096-32                    54.86Gi ±  5%    171.99Gi ±  4%   +213.53% (p=0.002 n=6)
ShiftAddInt32s/1048576-32                 52.89Gi ±  3%    167.73Gi ±  7%   +217.16% (p=0.002 n=6)
ReadAfterStoreInt32s/64-32                21.98Gi ±  6%     37.28Gi ±  7%    +69.59% (p=0.002 n=6)
ReadAfterStoreInt32s/4096-32              17.90Gi ±  8%     85.92Gi ±  5%   +380.14% (p=0.002 n=6)
ReadAfterStoreInt32s/1048576-32           18.13Gi ±  7%     79.39Gi ±  7%   +337.86% (p=0.002 n=6)
ReLUFloat32s/64-32                        38.96Gi ±  8%    123.87Gi ±  2%   +217.95% (p=0.002 n=6)
ReLUFloat32s/4096-32                      41.31Gi ±  2%    204.60Gi ±  4%   +395.27% (p=0.002 n=6)
ReLUFloat32s/1048576-32                   38.47Gi ± 20%    203.38Gi ±  3%   +428.63% (p=0.002 n=6)
LeakyReLUFloat32s/64-32                   20.54Gi ± 14%     93.92Gi ±  2%   +357.18% (p=0.002 n=6)
LeakyReLUFloat32s/4096-32                 21.08Gi ±  8%    204.67Gi ±  2%   +871.07% (p=0.002 n=6)
LeakyReLUFloat32s/1048576-32              19.26Gi ±  9%    134.81Gi ±  5%   +599.84% (p=0.002 n=6)
ReLU6Float32s/64-32                       20.90Gi ±  2%     98.64Gi ±  1%   +371.88% (p=0.002 n=6)
ReLU6Float32s/4096-32                     20.78Gi ±  2%    130.51Gi ±  2%   +528.10% (p=0.002 n=6)
ReLU6Float32s/1048576-32                  16.96Gi ± 28%    131.00Gi ±  3%   +672.22% (p=0.002 n=6)
SignFloat32s/64-32                        18.59Gi ± 40%     92.05Gi ±  2%   +395.18% (p=0.002 n=6)
SignFloat32s/4096-32                      19.50Gi ±  9%    171.90Gi ±  5%   +781.38% (p=0.002 n=6)
SignFloat32s/1048576-32                   20.18Gi ±  7%    130.60Gi ±  6%   +547.08% (p=0.002 n=6)
FillNaNFloat64s/64-32                     39.50Gi ±  2%    134.73Gi ±  7%   +241.09% (p=0.002 n=6)
FillNaNFloat64s/4096-32                   40.22Gi ±  4%    202.48Gi ± 25%   +403.47% (p=0.002 n=6)
FillNaNFloat64s/1048576-32                40.36Gi ± 61%    130.43Gi ±  9%   +223.17% (p=0.002 n=6)
ScaleFieldFloat32s/64-32                  37.35Gi ±  4%    114.25Gi ±  4%   +205.88% (p=0.002 n=6)
ScaleFieldFloat32s/4096-32                39.39Gi ± 13%    156.48Gi ± 18%   +297.30% (p=0.002 n=6)
ScaleFieldFloat32s/1048576-32             40.74Gi ±  8%    183.42Gi ±  7%   +350.28% (p=0.002 n=6)
AddFieldFloat32s/64-32                    27.26Gi ±  7%    111.24Gi ±  4%   +308.00% (p=0.002 n=6)
AddFieldFloat32s/4096-32                  29.33Gi ±  4%    178.71Gi ±  1%   +509.32% (p=0.002 n=6)
AddFieldFloat32s/1048576-32               29.72Gi ±  5%    166.69Gi ±  6%   +460.92% (p=0.002 n=6)
HardSwishFloat32s/64-32                   15.67Gi ± 23%     79.69Gi ±  6%   +408.42% (p=0.002 n=6)
HardSwishFloat32s/4096-32                 16.13Gi ±  4%    141.00Gi ±  4%   +773.94% (p=0.002 n=6)
HardSwishFloat32s/1048576-32              16.26Gi ±  2%    122.01Gi ±  4%   +650.33% (p=0.002 n=6)
HardSigmoidFloat32s/64-32                 16.39Gi ±  1%     76.49Gi ±  2%   +366.81% (p=0.002 n=6)
HardSigmoidFloat32s/4096-32               16.36Gi ±  2%    122.14Gi ± 14%   +646.49% (p=0.002 n=6)
HardSigmoidFloat32s/1048576-32            16.72Gi ±  5%    118.72Gi ±  6%   +610.20% (p=0.002 n=6)
geomean                                   26.13Gi           121.1Gi         +363.51%
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
