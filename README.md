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
AddFloat32s/64-32                         14.270n ± 18%   6.730n ±  1%  -52.84% (p=0.000 n=10)
AddFloat32s/4096-32                        876.4n ±  5%   191.6n ±  3%  -78.14% (p=0.000 n=10)
AddFloat32s/1048576-32                    218.36µ ±  4%   85.72µ ±  4%  -60.75% (p=0.000 n=10)
MulFloat32s/64-32                         14.795n ± 13%   6.311n ±  1%  -57.34% (p=0.000 n=10)
MulFloat32s/4096-32                        903.0n ± 18%   187.0n ±  3%  -79.29% (p=0.000 n=10)
MulFloat32s/1048576-32                    234.26µ ± 25%   87.02µ ±  6%  -62.85% (p=0.000 n=10)
ScalFloat32s/64-32                        10.930n ± 10%   3.684n ±  1%  -66.29% (p=0.000 n=10)
ScalFloat32s/4096-32                       743.7n ±  1%   145.4n ±  2%  -80.45% (p=0.000 n=10)
ScalFloat32s/1048576-32                   187.48µ ±  2%   37.22µ ±  2%  -80.15% (p=0.000 n=10)
MixFloat32s/64-32                         23.210n ±  1%   7.655n ±  3%  -67.02% (p=0.000 n=10)
MixFloat32s/4096-32                       1483.0n ±  1%   221.3n ±  4%  -85.08% (p=0.000 n=10)
MixFloat32s/1048576-32                    393.18µ ±  3%   89.59µ ±  9%  -77.21% (p=0.000 n=10)
AxpyFloat32s/64-32                        13.730n ±  3%   6.637n ±  3%  -51.66% (p=0.000 n=10)
AxpyFloat32s/4096-32                       871.9n ±  2%   197.7n ±  2%  -77.33% (p=0.000 n=10)
AxpyFloat32s/1048576-32                   218.95µ ±  4%   89.46µ ±  5%  -59.14% (p=0.000 n=10)
NegFloat32s/64-32                         14.265n ±  2%   5.059n ±  2%  -64.53% (p=0.000 n=10)
NegFloat32s/4096-32                        756.5n ±  1%   172.8n ±  3%  -77.16% (p=0.000 n=10)
NegFloat32s/1048576-32                    197.01µ ±  1%   59.25µ ±  3%  -69.92% (p=0.000 n=10)
DivFloat32s/64-32                         28.635n ±  2%   7.069n ±  1%  -75.31% (p=0.000 n=10)
DivFloat32s/4096-32                       1839.5n ±  1%   195.3n ±  3%  -89.38% (p=0.000 n=10)
DivFloat32s/1048576-32                    467.44µ ±  2%   87.55µ ±  6%  -81.27% (p=0.000 n=10)
DaxpyFloat32s/64-32                       12.500n ±  1%   5.468n ±  3%  -56.25% (p=0.000 n=10)
DaxpyFloat32s/4096-32                      755.9n ±  2%   176.0n ±  1%  -76.71% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                  198.29µ ±  1%   64.72µ ±  1%  -67.36% (p=0.000 n=10)
FillFloat32s/64-32                        11.905n ±  6%   2.969n ±  4%  -75.07% (p=0.000 n=10)
FillFloat32s/4096-32                       760.2n ±  2%   104.0n ±  2%  -86.32% (p=0.000 n=10)
FillFloat32s/1048576-32                   188.36µ ±  3%   30.26µ ±  1%  -83.94% (p=0.000 n=10)
FillUint8s/64-32                          12.570n ±  3%   1.994n ±  6%  -84.13% (p=0.000 n=10)
FillUint8s/4096-32                        773.75n ±  5%   27.72n ±  1%  -96.42% (p=0.000 n=10)
FillUint8s/1048576-32                    185.832µ ±  2%   6.118µ ±  2%  -96.71% (p=0.000 n=10)
ReverseIncFloat32s/64-32                  12.895n ± 13%   5.115n ±  2%  -60.33% (p=0.000 n=10)
ReverseIncFloat32s/4096-32                 860.0n ±  5%   165.9n ± 25%  -80.71% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32             215.27µ ± 25%   59.56µ ±  3%  -72.33% (p=0.000 n=10)
CopyFloat32s/64-32                        13.410n ± 12%   4.086n ±  3%  -69.53% (p=0.000 n=10)
CopyFloat32s/4096-32                      757.25n ±  3%   72.18n ±  2%  -90.47% (p=0.000 n=10)
CopyFloat32s/1048576-32                   189.00µ ±  3%   59.30µ ±  3%  -68.62% (p=0.000 n=10)
AndNotUint64s/64-32                       23.505n ±  3%   9.927n ±  3%  -57.77% (p=0.000 n=10)
AndNotUint64s/4096-32                     1477.5n ±  1%   380.0n ±  2%  -74.28% (p=0.000 n=10)
AndNotUint64s/1048576-32                   411.7µ ±  2%   179.3µ ±  2%  -56.43% (p=0.000 n=10)
NormalizeFloat32s/64-32                   28.935n ±  2%   6.636n ±  4%  -77.06% (p=0.000 n=10)
NormalizeFloat32s/4096-32                 1842.0n ±  1%   231.5n ±  2%  -87.43% (p=0.000 n=10)
NormalizeFloat32s/1048576-32              470.08µ ±  1%   69.01µ ±  5%  -85.32% (p=0.000 n=10)
ChainInt32s/64-32                          29.39n ±  2%   12.74n ±  1%  -56.68% (p=0.000 n=10)
ChainInt32s/4096-32                       1894.5n ±  4%   329.4n ±  6%  -82.62% (p=0.000 n=10)
ChainInt32s/1048576-32                     477.5µ ±  2%   123.5µ ± 12%  -74.14% (p=0.000 n=10)
WideInt32s/64-32                          30.945n ±  4%   9.871n ±  2%  -68.10% (p=0.000 n=10)
WideInt32s/4096-32                        1941.5n ±  2%   279.4n ±  5%  -85.61% (p=0.000 n=10)
WideInt32s/1048576-32                      484.4µ ± 20%   123.2µ ±  7%  -74.57% (p=0.000 n=10)
SumInt32s/64-32                           10.260n ±  7%   7.608n ±  2%  -25.84% (p=0.000 n=10)
SumInt32s/4096-32                          740.7n ±  2%   194.2n ±  1%  -73.78% (p=0.000 n=10)
SumInt32s/1048576-32                      187.26µ ±  2%   47.50µ ±  1%  -74.63% (p=0.000 n=10)
DotInt32s/64-32                           14.300n ±  6%   9.360n ±  2%  -34.55% (p=0.000 n=10)
DotInt32s/4096-32                          941.6n ±  3%   206.4n ±  2%  -78.07% (p=0.000 n=10)
DotInt32s/1048576-32                      245.44µ ±  1%   63.89µ ±  3%  -73.97% (p=0.000 n=10)
SumFloat32s/64-32                          14.51n ± 11%   13.11n ±  1%        ~ (p=0.065 n=10)
SumFloat32s/4096-32                       1422.0n ±  1%   111.0n ±  1%  -92.19% (p=0.000 n=10)
SumFloat32s/1048576-32                    370.19µ ±  2%   31.99µ ±  4%  -91.36% (p=0.000 n=10)
DotFloat32s/64-32                          12.76n ±  8%   13.31n ±  1%        ~ (p=0.469 n=10)
DotFloat32s/4096-32                       1431.0n ±  2%   185.3n ±  6%  -87.05% (p=0.000 n=10)
DotFloat32s/1048576-32                    374.83µ ±  1%   64.40µ ±  3%  -82.82% (p=0.000 n=10)
DotFloat64s/64-32                          16.15n ±  1%   12.45n ±  5%  -22.91% (p=0.000 n=10)
DotFloat64s/4096-32                       1440.0n ±  1%   365.7n ±  2%  -74.60% (p=0.000 n=10)
DotFloat64s/1048576-32                     384.8µ ±  3%   132.1µ ±  3%  -65.67% (p=0.000 n=10)
AbsFloat32s/64-32                         23.030n ±  2%   4.941n ±  1%  -78.55% (p=0.000 n=10)
AbsFloat32s/4096-32                       1478.0n ±  2%   167.5n ±  3%  -88.67% (p=0.000 n=10)
AbsFloat32s/1048576-32                    376.53µ ±  2%   59.00µ ±  3%  -84.33% (p=0.000 n=10)
SqrtFloat32s/64-32                        51.985n ±  1%   5.196n ±  1%  -90.00% (p=0.000 n=10)
SqrtFloat32s/4096-32                      3315.5n ±  4%   220.1n ±  2%  -93.36% (p=0.000 n=10)
SqrtFloat32s/1048576-32                   843.69µ ±  2%   61.35µ ±  3%  -92.73% (p=0.000 n=10)
ClampInt32s/64-32                         28.910n ±  2%   4.756n ±  2%  -83.55% (p=0.000 n=10)
ClampInt32s/4096-32                       1873.5n ±  1%   147.8n ±  1%  -92.11% (p=0.000 n=10)
ClampInt32s/1048576-32                    476.29µ ±  2%   60.05µ ±  2%  -87.39% (p=0.000 n=10)
ShrInt32s/64-32                           13.105n ±  4%   4.914n ±  3%  -62.50% (p=0.000 n=10)
ShrInt32s/4096-32                          841.3n ±  2%   159.6n ±  2%  -81.03% (p=0.000 n=10)
ShrInt32s/1048576-32                      216.86µ ±  2%   59.67µ ±  1%  -72.49% (p=0.000 n=10)
ShlUint32s/64-32                          12.940n ±  3%   4.818n ±  3%  -62.77% (p=0.000 n=10)
ShlUint32s/4096-32                         846.9n ±  2%   150.4n ±  1%  -82.24% (p=0.000 n=10)
ShlUint32s/1048576-32                     217.39µ ±  3%   59.23µ ±  4%  -72.75% (p=0.000 n=10)
DiffInt32s/64-32                          13.450n ±  4%   8.650n ±  4%  -35.69% (p=0.000 n=10)
DiffInt32s/4096-32                         893.8n ±  9%   418.8n ±  4%  -53.14% (p=0.000 n=10)
DiffInt32s/1048576-32                      229.8µ ±  4%   108.2µ ±  2%  -52.90% (p=0.000 n=10)
SmoothInt32s/64-32                        24.225n ±  4%   9.662n ± 14%  -60.12% (p=0.000 n=10)
SmoothInt32s/4096-32                      1864.0n ±  4%   486.3n ± 13%  -73.91% (p=0.000 n=10)
SmoothInt32s/1048576-32                    500.8µ ±  2%   121.3µ ±  2%  -75.79% (p=0.000 n=10)
ShiftInt32s/64-32                         16.775n ±  4%   5.289n ±  3%  -68.47% (p=0.000 n=10)
ShiftInt32s/4096-32                       1062.0n ±  5%   191.2n ±  2%  -82.00% (p=0.000 n=10)
ShiftInt32s/1048576-32                    272.97µ ±  2%   63.55µ ±  3%  -76.72% (p=0.000 n=10)
AccumRowInt32s/64-32                      16.245n ±  2%   6.435n ±  2%  -60.38% (p=0.000 n=10)
AccumRowInt32s/4096-32                    1065.0n ±  3%   240.4n ±  2%  -77.42% (p=0.000 n=10)
AccumRowInt32s/1048576-32                 271.98µ ±  3%   65.98µ ±  4%  -75.74% (p=0.000 n=10)
MinUint16s/64-32                           25.07n ±  1%   30.52n ±  1%  +21.69% (p=0.000 n=10)
MinUint16s/4096-32                       2186.00n ±  2%   67.88n ±  4%  -96.89% (p=0.000 n=10)
MinUint16s/1048576-32                     564.34µ ±  1%   15.58µ ±  5%  -97.24% (p=0.000 n=10)
ProductFloat32s/64-32                     17.020n ± 16%   8.034n ±  2%  -52.80% (p=0.000 n=10)
ProductFloat32s/4096-32                   2154.0n ±  1%   108.8n ±  2%  -94.95% (p=0.000 n=10)
ProductFloat32s/1048576-32                560.32µ ±  1%   31.89µ ±  2%  -94.31% (p=0.000 n=10)
ShiftAddInt32s/64-32                      13.630n ±  4%   7.705n ±  5%  -43.47% (p=0.000 n=10)
ShiftAddInt32s/4096-32                     841.0n ±  3%   271.5n ±  1%  -67.71% (p=0.000 n=10)
ShiftAddInt32s/1048576-32                 221.97µ ±  3%   72.10µ ±  5%  -67.52% (p=0.000 n=10)
ReadAfterStoreInt32s/64-32                 35.41n ±  3%   19.07n ±  3%  -46.15% (p=0.000 n=10)
ReadAfterStoreInt32s/4096-32              2537.0n ±  5%   562.5n ±  2%  -77.83% (p=0.000 n=10)
ReadAfterStoreInt32s/1048576-32            677.0µ ±  5%   160.2µ ±  4%  -76.34% (p=0.000 n=10)
ReLUFloat32s/64-32                        12.490n ±  6%   3.869n ±  2%  -69.02% (p=0.000 n=10)
ReLUFloat32s/4096-32                       777.7n ±  5%   146.1n ±  2%  -81.21% (p=0.000 n=10)
ReLUFloat32s/1048576-32                   196.59µ ± 23%   38.33µ ±  2%  -80.50% (p=0.000 n=10)
LeakyReLUFloat32s/64-32                   24.145n ±  5%   5.584n ±  2%  -76.87% (p=0.000 n=10)
LeakyReLUFloat32s/4096-32                 1398.5n ±  9%   171.8n ±  3%  -87.72% (p=0.000 n=10)
LeakyReLUFloat32s/1048576-32              404.13µ ±  4%   60.96µ ±  5%  -84.91% (p=0.000 n=10)
ReLU6Float32s/64-32                       24.310n ±  8%   3.952n ±  3%  -83.74% (p=0.000 n=10)
ReLU6Float32s/4096-32                     1483.0n ±  8%   167.4n ±  2%  -88.72% (p=0.000 n=10)
ReLU6Float32s/1048576-32                  383.56µ ±  5%   43.98µ ±  2%  -88.53% (p=0.000 n=10)
SignFloat32s/64-32                        28.525n ±  6%   5.471n ±  6%  -80.82% (p=0.000 n=10)
SignFloat32s/4096-32                      1552.5n ±  8%   184.6n ±  5%  -88.11% (p=0.000 n=10)
SignFloat32s/1048576-32                   379.37µ ±  8%   60.61µ ±  4%  -84.02% (p=0.000 n=10)
FillNaNFloat64s/64-32                     11.295n ±  2%   4.008n ±  2%  -64.51% (p=0.000 n=10)
FillNaNFloat64s/4096-32                    742.8n ±  2%   159.1n ±  2%  -78.58% (p=0.000 n=10)
FillNaNFloat64s/1048576-32                188.74µ ±  1%   60.12µ ±  1%  -68.15% (p=0.000 n=10)
ScaleFieldFloat32s/64-32                  11.830n ±  4%   4.054n ±  2%  -65.74% (p=0.000 n=10)
ScaleFieldFloat32s/4096-32                 749.9n ±  8%   161.6n ±  3%  -78.46% (p=0.000 n=10)
ScaleFieldFloat32s/1048576-32             187.45µ ±  2%   42.89µ ±  8%  -77.12% (p=0.000 n=10)
AddFieldFloat32s/64-32                    23.390n ±  1%   6.603n ±  1%  -71.77% (p=0.000 n=10)
AddFieldFloat32s/4096-32                  1496.5n ±  1%   265.0n ±  3%  -82.29% (p=0.000 n=10)
AddFieldFloat32s/1048576-32               393.94µ ±  2%   73.53µ ±  8%  -81.34% (p=0.000 n=10)
geomean                                    1.873µ         421.3n        -77.51%

                                │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                                │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                         50.14Gi ± 15%    106.28Gi ±  1%   +111.99% (p=0.000 n=10)
AddFloat32s/4096-32                       52.23Gi ±  5%    238.94Gi ±  3%   +357.44% (p=0.000 n=10)
AddFloat32s/1048576-32                    53.67Gi ±  4%    136.72Gi ±  4%   +154.75% (p=0.000 n=10)
MulFloat32s/64-32                         48.34Gi ± 12%    113.34Gi ±  1%   +134.44% (p=0.000 n=10)
MulFloat32s/4096-32                       50.72Gi ± 15%    244.80Gi ±  3%   +382.61% (p=0.000 n=10)
MulFloat32s/1048576-32                    50.03Gi ± 20%    134.67Gi ±  5%   +169.18% (p=0.000 n=10)
ScalFloat32s/64-32                        43.63Gi ± 11%    129.43Gi ±  1%   +196.66% (p=0.000 n=10)
ScalFloat32s/4096-32                      41.04Gi ±  2%    209.85Gi ±  2%   +411.38% (p=0.000 n=10)
ScalFloat32s/1048576-32                   41.67Gi ±  1%    209.90Gi ±  2%   +403.72% (p=0.000 n=10)
MixFloat32s/64-32                         30.82Gi ±  1%     93.44Gi ±  2%   +203.24% (p=0.000 n=10)
MixFloat32s/4096-32                       30.87Gi ±  1%    206.87Gi ±  4%   +570.14% (p=0.000 n=10)
MixFloat32s/1048576-32                    29.80Gi ±  3%    130.81Gi ±  9%   +338.88% (p=0.000 n=10)
AxpyFloat32s/64-32                        52.10Gi ±  3%    107.78Gi ±  3%   +106.88% (p=0.000 n=10)
AxpyFloat32s/4096-32                      52.51Gi ±  2%    231.60Gi ±  2%   +341.09% (p=0.000 n=10)
AxpyFloat32s/1048576-32                   53.52Gi ±  4%    131.01Gi ±  5%   +144.78% (p=0.000 n=10)
NegFloat32s/64-32                         33.42Gi ±  2%     94.25Gi ±  2%   +181.98% (p=0.000 n=10)
NegFloat32s/4096-32                       40.34Gi ±  1%    176.63Gi ±  3%   +337.85% (p=0.000 n=10)
NegFloat32s/1048576-32                    39.66Gi ±  1%    131.85Gi ±  3%   +232.49% (p=0.000 n=10)
DivFloat32s/64-32                         24.98Gi ±  2%    101.19Gi ±  1%   +305.10% (p=0.000 n=10)
DivFloat32s/4096-32                       24.88Gi ±  1%    234.38Gi ±  3%   +842.04% (p=0.000 n=10)
DivFloat32s/1048576-32                    25.07Gi ±  2%    133.85Gi ±  6%   +433.91% (p=0.000 n=10)
DaxpyFloat32s/64-32                       57.23Gi ±  1%    130.80Gi ±  3%   +128.54% (p=0.000 n=10)
DaxpyFloat32s/4096-32                     60.56Gi ±  3%    260.07Gi ±  1%   +329.42% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                  59.10Gi ±  1%    181.07Gi ±  1%   +206.37% (p=0.000 n=10)
FillFloat32s/64-32                        20.02Gi ±  5%     80.31Gi ±  4%   +301.08% (p=0.000 n=10)
FillFloat32s/4096-32                      20.07Gi ±  2%    146.73Gi ±  2%   +630.96% (p=0.000 n=10)
FillFloat32s/1048576-32                   20.74Gi ±  3%    129.10Gi ±  1%   +522.50% (p=0.000 n=10)
FillUint8s/64-32                          4.741Gi ±  3%    29.885Gi ±  6%   +530.35% (p=0.000 n=10)
FillUint8s/4096-32                        4.930Gi ±  4%   137.596Gi ±  1%  +2690.92% (p=0.000 n=10)
FillUint8s/1048576-32                     5.255Gi ±  2%   159.628Gi ±  2%  +2937.55% (p=0.000 n=10)
ReverseIncFloat32s/64-32                  36.99Gi ± 11%     93.22Gi ±  2%   +152.03% (p=0.000 n=10)
ReverseIncFloat32s/4096-32                35.49Gi ±  6%    183.94Gi ± 20%   +418.31% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32             36.30Gi ± 20%    131.17Gi ±  4%   +261.34% (p=0.000 n=10)
CopyFloat32s/64-32                        35.56Gi ± 11%    116.71Gi ±  3%   +228.21% (p=0.000 n=10)
CopyFloat32s/4096-32                      40.30Gi ±  3%    422.80Gi ±  2%   +949.14% (p=0.000 n=10)
CopyFloat32s/1048576-32                   41.34Gi ±  3%    131.75Gi ±  3%   +218.72% (p=0.000 n=10)
AndNotUint64s/64-32                       60.86Gi ±  3%    144.11Gi ±  3%   +136.77% (p=0.000 n=10)
AndNotUint64s/4096-32                     61.96Gi ±  1%    240.94Gi ±  2%   +288.86% (p=0.000 n=10)
AndNotUint64s/1048576-32                  56.93Gi ±  2%    130.68Gi ±  2%   +129.53% (p=0.000 n=10)
NormalizeFloat32s/64-32                   32.96Gi ±  2%    143.71Gi ±  4%   +335.95% (p=0.000 n=10)
NormalizeFloat32s/4096-32                 33.14Gi ±  1%    263.60Gi ±  2%   +695.52% (p=0.000 n=10)
NormalizeFloat32s/1048576-32              33.24Gi ±  1%    226.41Gi ±  4%   +581.18% (p=0.000 n=10)
ChainInt32s/64-32                         48.67Gi ±  2%    112.32Gi ±  1%   +130.79% (p=0.000 n=10)
ChainInt32s/4096-32                       48.32Gi ±  4%    278.00Gi ±  6%   +475.35% (p=0.000 n=10)
ChainInt32s/1048576-32                    49.09Gi ±  2%    189.81Gi ± 14%   +286.68% (p=0.000 n=10)
WideInt32s/64-32                          30.82Gi ±  4%     96.61Gi ±  2%   +213.45% (p=0.000 n=10)
WideInt32s/4096-32                        31.44Gi ±  2%    218.47Gi ±  5%   +594.83% (p=0.000 n=10)
WideInt32s/1048576-32                     32.26Gi ± 16%    126.86Gi ±  8%   +293.25% (p=0.000 n=10)
SumInt32s/64-32                           23.24Gi ±  7%     31.33Gi ±  2%    +34.83% (p=0.000 n=10)
SumInt32s/4096-32                         20.60Gi ±  2%     78.57Gi ±  2%   +281.41% (p=0.000 n=10)
SumInt32s/1048576-32                      20.86Gi ±  2%     82.24Gi ±  1%   +294.24% (p=0.000 n=10)
DotInt32s/64-32                           33.34Gi ±  6%     50.95Gi ±  2%    +52.82% (p=0.000 n=10)
DotInt32s/4096-32                         32.41Gi ±  3%    147.79Gi ±  2%   +355.98% (p=0.000 n=10)
DotInt32s/1048576-32                      31.83Gi ±  1%    122.28Gi ±  3%   +284.15% (p=0.000 n=10)
SumFloat32s/64-32                         16.44Gi ± 12%     18.19Gi ±  2%          ~ (p=0.075 n=10)
SumFloat32s/4096-32                       10.73Gi ±  1%    137.40Gi ±  1%  +1180.49% (p=0.000 n=10)
SumFloat32s/1048576-32                    10.55Gi ±  1%    122.10Gi ±  4%  +1057.16% (p=0.000 n=10)
DotFloat32s/64-32                         37.42Gi ±  7%     35.82Gi ±  1%          ~ (p=0.481 n=10)
DotFloat32s/4096-32                       21.32Gi ±  2%    164.64Gi ±  6%   +672.12% (p=0.000 n=10)
DotFloat32s/1048576-32                    20.84Gi ±  1%    121.31Gi ±  3%   +482.02% (p=0.000 n=10)
DotFloat64s/64-32                         59.05Gi ±  1%     76.58Gi ±  5%    +29.67% (p=0.000 n=10)
DotFloat64s/4096-32                       42.38Gi ±  1%    166.91Gi ±  2%   +293.81% (p=0.000 n=10)
DotFloat64s/1048576-32                    40.61Gi ±  3%    118.30Gi ±  3%   +191.32% (p=0.000 n=10)
AbsFloat32s/64-32                         20.70Gi ±  2%     96.51Gi ±  1%   +366.15% (p=0.000 n=10)
AbsFloat32s/4096-32                       20.65Gi ±  3%    182.21Gi ±  3%   +782.46% (p=0.000 n=10)
AbsFloat32s/1048576-32                    20.75Gi ±  2%    132.42Gi ±  3%   +538.20% (p=0.000 n=10)
SqrtFloat32s/64-32                        9.173Gi ±  1%    91.759Gi ±  1%   +900.31% (p=0.000 n=10)
SqrtFloat32s/4096-32                      9.204Gi ±  4%   138.675Gi ±  2%  +1406.69% (p=0.000 n=10)
SqrtFloat32s/1048576-32                   9.260Gi ±  2%   127.347Gi ±  3%  +1275.26% (p=0.000 n=10)
ClampInt32s/64-32                         16.49Gi ±  2%    100.26Gi ±  2%   +507.83% (p=0.000 n=10)
ClampInt32s/4096-32                       16.29Gi ±  1%    206.50Gi ±  1%  +1167.51% (p=0.000 n=10)
ClampInt32s/1048576-32                    16.40Gi ±  2%    130.10Gi ±  2%   +693.17% (p=0.000 n=10)
ShrInt32s/64-32                           36.38Gi ±  3%     97.04Gi ±  3%   +166.72% (p=0.000 n=10)
ShrInt32s/4096-32                         36.27Gi ±  2%    191.23Gi ±  2%   +427.19% (p=0.000 n=10)
ShrInt32s/1048576-32                      36.03Gi ±  2%    130.94Gi ±  1%   +263.44% (p=0.000 n=10)
ShlUint32s/64-32                          36.85Gi ±  3%     98.97Gi ±  3%   +168.60% (p=0.000 n=10)
ShlUint32s/4096-32                        36.03Gi ±  2%    202.82Gi ±  1%   +462.90% (p=0.000 n=10)
ShlUint32s/1048576-32                     35.94Gi ±  3%    131.90Gi ±  3%   +267.03% (p=0.000 n=10)
DiffInt32s/64-32                          35.45Gi ±  4%     55.13Gi ±  4%    +55.49% (p=0.000 n=10)
DiffInt32s/4096-32                        34.15Gi ±  8%     72.87Gi ±  5%   +113.40% (p=0.000 n=10)
DiffInt32s/1048576-32                     34.01Gi ±  4%     72.19Gi ±  2%   +112.29% (p=0.000 n=10)
SmoothInt32s/64-32                        19.69Gi ±  4%     49.35Gi ± 12%   +150.72% (p=0.000 n=10)
SmoothInt32s/4096-32                      16.37Gi ±  4%     62.76Gi ± 12%   +283.33% (p=0.000 n=10)
SmoothInt32s/1048576-32                   15.60Gi ±  2%     64.43Gi ±  2%   +313.04% (p=0.000 n=10)
ShiftInt32s/64-32                         28.43Gi ±  4%     90.16Gi ±  3%   +217.16% (p=0.000 n=10)
ShiftInt32s/4096-32                       28.73Gi ±  4%    159.65Gi ±  2%   +455.69% (p=0.000 n=10)
ShiftInt32s/1048576-32                    28.62Gi ±  2%    122.94Gi ±  3%   +329.53% (p=0.000 n=10)
AccumRowInt32s/64-32                      44.03Gi ±  2%    111.14Gi ±  2%   +152.42% (p=0.000 n=10)
AccumRowInt32s/4096-32                    42.98Gi ±  3%    190.38Gi ±  2%   +342.95% (p=0.000 n=10)
AccumRowInt32s/1048576-32                 43.09Gi ±  3%    177.61Gi ±  4%   +312.20% (p=0.000 n=10)
MinUint16s/64-32                          4.754Gi ±  1%     3.906Gi ±  0%    -17.82% (p=0.000 n=10)
MinUint16s/4096-32                        3.490Gi ±  2%   112.394Gi ±  4%  +3120.40% (p=0.000 n=10)
MinUint16s/1048576-32                     3.461Gi ±  1%   125.386Gi ±  5%  +3522.86% (p=0.000 n=10)
ProductFloat32s/64-32                     14.01Gi ± 19%     29.68Gi ±  3%   +111.86% (p=0.000 n=10)
ProductFloat32s/4096-32                   7.085Gi ±  1%   140.291Gi ±  2%  +1880.16% (p=0.000 n=10)
ProductFloat32s/1048576-32                6.972Gi ±  1%   122.507Gi ±  2%  +1657.24% (p=0.000 n=10)
ShiftAddInt32s/64-32                      52.48Gi ±  5%     92.82Gi ±  5%    +76.88% (p=0.000 n=10)
ShiftAddInt32s/4096-32                    54.43Gi ±  3%    168.58Gi ±  1%   +209.70% (p=0.000 n=10)
ShiftAddInt32s/1048576-32                 52.80Gi ±  3%    162.54Gi ±  4%   +207.85% (p=0.000 n=10)
ReadAfterStoreInt32s/64-32                20.20Gi ±  3%     37.51Gi ±  3%    +85.68% (p=0.000 n=10)
ReadAfterStoreInt32s/4096-32              18.07Gi ±  5%     81.39Gi ±  2%   +350.47% (p=0.000 n=10)
ReadAfterStoreInt32s/1048576-32           17.31Gi ±  5%     73.15Gi ±  4%   +322.61% (p=0.000 n=10)
ReLUFloat32s/64-32                        38.18Gi ±  6%    123.23Gi ±  2%   +222.73% (p=0.000 n=10)
ReLUFloat32s/4096-32                      39.24Gi ±  5%    208.81Gi ±  2%   +432.10% (p=0.000 n=10)
ReLUFloat32s/1048576-32                   39.75Gi ± 19%    203.81Gi ±  2%   +412.79% (p=0.000 n=10)
LeakyReLUFloat32s/64-32                   19.75Gi ±  5%     85.40Gi ±  2%   +332.39% (p=0.000 n=10)
LeakyReLUFloat32s/4096-32                 21.83Gi ±  8%    177.70Gi ±  3%   +714.19% (p=0.000 n=10)
LeakyReLUFloat32s/1048576-32              19.33Gi ±  4%    128.15Gi ±  5%   +562.90% (p=0.000 n=10)
ReLU6Float32s/64-32                       19.62Gi ±  8%    120.66Gi ±  3%   +515.02% (p=0.000 n=10)
ReLU6Float32s/4096-32                     20.58Gi ±  7%    182.35Gi ±  2%   +786.24% (p=0.000 n=10)
ReLU6Float32s/1048576-32                  20.37Gi ±  4%    177.65Gi ±  3%   +772.18% (p=0.000 n=10)
SignFloat32s/64-32                        16.72Gi ±  6%     87.15Gi ±  5%   +421.38% (p=0.000 n=10)
SignFloat32s/4096-32                      19.66Gi ±  7%    165.31Gi ±  5%   +740.74% (p=0.000 n=10)
SignFloat32s/1048576-32                   20.59Gi ±  9%    128.92Gi ±  4%   +526.02% (p=0.000 n=10)
FillNaNFloat64s/64-32                     42.21Gi ±  3%    118.96Gi ±  2%   +181.83% (p=0.000 n=10)
FillNaNFloat64s/4096-32                   41.09Gi ±  1%    191.80Gi ±  2%   +366.80% (p=0.000 n=10)
FillNaNFloat64s/1048576-32                41.39Gi ±  1%    129.96Gi ±  1%   +213.96% (p=0.000 n=10)
ScaleFieldFloat32s/64-32                  40.30Gi ±  3%    117.64Gi ±  2%   +191.93% (p=0.000 n=10)
ScaleFieldFloat32s/4096-32                40.70Gi ±  7%    188.91Gi ±  3%   +364.16% (p=0.000 n=10)
ScaleFieldFloat32s/1048576-32             41.68Gi ±  2%    182.17Gi ±  7%   +337.08% (p=0.000 n=10)
AddFieldFloat32s/64-32                    30.58Gi ±  1%    108.32Gi ±  1%   +254.22% (p=0.000 n=10)
AddFieldFloat32s/4096-32                  30.59Gi ±  1%    172.72Gi ±  3%   +464.55% (p=0.000 n=10)
AddFieldFloat32s/1048576-32               29.75Gi ±  2%    159.38Gi ±  7%   +435.78% (p=0.000 n=10)
geomean                                   27.22Gi           121.0Gi         +344.56%
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
