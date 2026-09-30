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
| `for i := range a { sum += a[i] * b[i] }` | integer reduction (`+ * & \| ^ min max`) |
| `for i := 0; i < len(s); i++ { ... }` | three-clause for (all body shapes above) |
| `for i := len(s) - 1; i >= 0; i-- { ... }` | reverse three-clause for (rewritten to run forward) |
| `for i := 0; i < n; i++ { ... }`, `for i := range n { ... }` | explicit int limit; slices are length-checked first |
| `for i := 0; i < 4; i++ { ... }` | constant limit |

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
AddFloat32s/64-32                       12.955n ± 44%   5.820n ±  3%  -55.08% (p=0.000 n=10)
AddFloat32s/4096-32                      835.0n ± 18%   196.1n ±  2%  -76.52% (p=0.000 n=10)
AddFloat32s/1048576-32                  211.40µ ±  6%   84.84µ ±  9%  -59.87% (p=0.000 n=10)
MulFloat32s/64-32                       14.735n ±  3%   5.217n ±  3%  -64.59% (p=0.000 n=10)
MulFloat32s/4096-32                      931.5n ±  7%   168.3n ±  3%  -81.93% (p=0.000 n=10)
MulFloat32s/1048576-32                  238.08µ ±  4%   86.81µ ±  9%  -63.54% (p=0.000 n=10)
ScalFloat32s/64-32                      10.525n ± 13%   3.720n ±  2%  -64.65% (p=0.000 n=10)
ScalFloat32s/4096-32                     728.8n ±  2%   127.0n ±  2%  -82.58% (p=0.000 n=10)
ScalFloat32s/1048576-32                 189.60µ ±  2%   32.61µ ±  2%  -82.80% (p=0.000 n=10)
MixFloat32s/64-32                       22.810n ±  2%   5.034n ±  3%  -77.93% (p=0.000 n=10)
MixFloat32s/4096-32                     1463.5n ±  1%   175.2n ±  3%  -88.03% (p=0.000 n=10)
MixFloat32s/1048576-32                  383.78µ ±  1%   80.77µ ± 17%  -78.95% (p=0.000 n=10)
AxpyFloat32s/64-32                      12.830n ±  3%   5.832n ±  3%  -54.54% (p=0.000 n=10)
AxpyFloat32s/4096-32                     807.3n ±  2%   176.4n ±  3%  -78.15% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 210.68µ ±  4%   89.35µ ±  7%  -57.59% (p=0.000 n=10)
NegFloat32s/64-32                       13.120n ±  9%   4.333n ±  2%  -66.97% (p=0.000 n=10)
NegFloat32s/4096-32                      748.8n ±  1%   147.7n ±  2%  -80.28% (p=0.000 n=10)
NegFloat32s/1048576-32                  198.56µ ±  7%   61.39µ ±  1%  -69.08% (p=0.000 n=10)
DivFloat32s/64-32                       28.400n ±  2%   5.783n ±  2%  -79.64% (p=0.000 n=10)
DivFloat32s/4096-32                     1825.5n ±  2%   199.1n ±  2%  -89.09% (p=0.000 n=10)
DivFloat32s/1048576-32                  464.19µ ±  3%   82.21µ ± 12%  -82.29% (p=0.000 n=10)
DaxpyFloat32s/64-32                     12.805n ±  3%   4.557n ±  4%  -64.41% (p=0.000 n=10)
DaxpyFloat32s/4096-32                    755.4n ±  8%   166.1n ±  3%  -78.02% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                196.72µ ±  2%   64.76µ ±  3%  -67.08% (p=0.000 n=10)
FillFloat32s/64-32                      11.975n ±  3%   2.930n ±  5%  -75.53% (p=0.000 n=10)
FillFloat32s/4096-32                    762.40n ±  4%   85.86n ±  3%  -88.74% (p=0.000 n=10)
FillFloat32s/1048576-32                 186.84µ ±  5%   29.63µ ±  2%  -84.14% (p=0.000 n=10)
FillUint8s/64-32                        12.345n ±  3%   2.071n ±  8%  -83.22% (p=0.000 n=10)
FillUint8s/4096-32                      754.75n ±  3%   19.41n ±  7%  -97.43% (p=0.000 n=10)
FillUint8s/1048576-32                  188.489µ ±  3%   5.662µ ±  3%  -97.00% (p=0.000 n=10)
ReverseIncFloat32s/64-32                12.610n ±  9%   4.474n ±  1%  -64.52% (p=0.000 n=10)
ReverseIncFloat32s/4096-32               801.6n ±  6%   145.0n ±  5%  -81.91% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           198.53µ ±  2%   60.87µ ±  5%  -69.34% (p=0.000 n=10)
CopyFloat32s/64-32                      12.590n ± 75%   3.240n ±  1%  -74.27% (p=0.000 n=10)
CopyFloat32s/4096-32                    962.15n ± 53%   70.07n ±  2%  -92.72% (p=0.000 n=10)
CopyFloat32s/1048576-32                 191.01µ ± 87%   56.18µ ±  5%  -70.59% (p=0.000 n=10)
AndNotUint64s/64-32                     23.285n ±  6%   7.099n ±  6%  -69.51% (p=0.000 n=10)
AndNotUint64s/4096-32                   1473.0n ±  1%   350.5n ±  3%  -76.21% (p=0.000 n=10)
AndNotUint64s/1048576-32                 385.1µ ±  5%   167.4µ ± 15%  -56.53% (p=0.000 n=10)
NormalizeFloat32s/64-32                 28.885n ±  0%   6.880n ±  3%  -76.18% (p=0.000 n=10)
NormalizeFloat32s/4096-32               1848.0n ±  0%   294.0n ±  3%  -84.09% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            474.94µ ±  2%   78.77µ ±  3%  -83.42% (p=0.000 n=10)
ChainInt32s/64-32                       29.415n ±  1%   9.798n ±  2%  -66.69% (p=0.000 n=10)
ChainInt32s/4096-32                     1923.0n ± 18%   342.2n ±  3%  -82.21% (p=0.000 n=10)
ChainInt32s/1048576-32                   475.9µ ±  1%   135.2µ ±  5%  -71.59% (p=0.000 n=10)
WideInt32s/64-32                        30.305n ±  3%   8.219n ±  3%  -72.88% (p=0.000 n=10)
WideInt32s/4096-32                      1929.5n ± 46%   338.6n ±  2%  -82.45% (p=0.000 n=10)
WideInt32s/1048576-32                    478.4µ ±  2%   124.8µ ± 17%  -73.91% (p=0.000 n=10)
geomean                                  1.693µ         366.6n        -78.35%

                              │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                              │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                       55.23Gi ± 31%    122.90Gi ±  3%   +122.53% (p=0.000 n=10)
AddFloat32s/4096-32                     54.83Gi ± 15%    233.53Gi ±  2%   +325.92% (p=0.000 n=10)
AddFloat32s/1048576-32                  55.43Gi ±  6%    138.20Gi ±  9%   +149.30% (p=0.000 n=10)
MulFloat32s/64-32                       48.53Gi ±  3%    137.11Gi ±  3%   +182.50% (p=0.000 n=10)
MulFloat32s/4096-32                     49.14Gi ±  6%    272.01Gi ±  3%   +453.52% (p=0.000 n=10)
MulFloat32s/1048576-32                  49.23Gi ±  4%    135.04Gi ±  8%   +174.31% (p=0.000 n=10)
ScalFloat32s/64-32                      45.37Gi ± 12%    128.16Gi ±  2%   +182.47% (p=0.000 n=10)
ScalFloat32s/4096-32                    41.88Gi ±  2%    240.46Gi ±  2%   +474.20% (p=0.000 n=10)
ScalFloat32s/1048576-32                 41.21Gi ±  2%    239.61Gi ±  2%   +481.49% (p=0.000 n=10)
MixFloat32s/64-32                       31.35Gi ±  2%    142.11Gi ±  3%   +353.24% (p=0.000 n=10)
MixFloat32s/4096-32                     31.27Gi ±  1%    261.21Gi ±  3%   +735.22% (p=0.000 n=10)
MixFloat32s/1048576-32                  30.53Gi ±  1%    145.09Gi ± 14%   +375.17% (p=0.000 n=10)
AxpyFloat32s/64-32                      55.75Gi ±  3%    122.65Gi ±  3%   +119.99% (p=0.000 n=10)
AxpyFloat32s/4096-32                    56.70Gi ±  2%    259.50Gi ±  3%   +357.64% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 55.62Gi ±  4%    131.16Gi ±  8%   +135.81% (p=0.000 n=10)
NegFloat32s/64-32                       36.40Gi ±  9%    110.04Gi ±  2%   +202.29% (p=0.000 n=10)
NegFloat32s/4096-32                     40.76Gi ±  1%    206.69Gi ±  2%   +407.14% (p=0.000 n=10)
NegFloat32s/1048576-32                  39.35Gi ±  6%    127.26Gi ±  1%   +223.43% (p=0.000 n=10)
DivFloat32s/64-32                       25.19Gi ±  2%    123.68Gi ±  2%   +391.08% (p=0.000 n=10)
DivFloat32s/4096-32                     25.08Gi ±  2%    229.91Gi ±  2%   +816.72% (p=0.000 n=10)
DivFloat32s/1048576-32                  25.25Gi ±  3%    142.55Gi ± 10%   +464.64% (p=0.000 n=10)
DaxpyFloat32s/64-32                     55.87Gi ±  3%    156.95Gi ±  4%   +180.94% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   60.60Gi ±  8%    275.67Gi ±  3%   +354.92% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                59.57Gi ±  2%    180.96Gi ±  3%   +203.76% (p=0.000 n=10)
FillFloat32s/64-32                      19.91Gi ±  3%     81.36Gi ±  5%   +308.66% (p=0.000 n=10)
FillFloat32s/4096-32                    20.01Gi ±  4%    177.72Gi ±  3%   +787.95% (p=0.000 n=10)
FillFloat32s/1048576-32                 20.91Gi ±  4%    131.83Gi ±  2%   +530.56% (p=0.000 n=10)
FillUint8s/64-32                        4.829Gi ±  3%    28.784Gi ±  9%   +496.13% (p=0.000 n=10)
FillUint8s/4096-32                      5.054Gi ±  3%   196.493Gi ±  7%  +3787.64% (p=0.000 n=10)
FillUint8s/1048576-32                   5.181Gi ±  3%   172.500Gi ±  3%  +3229.46% (p=0.000 n=10)
ReverseIncFloat32s/64-32                37.80Gi ±  8%    106.56Gi ±  1%   +181.91% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              38.08Gi ±  6%    210.41Gi ±  5%   +452.60% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           39.35Gi ±  2%    128.35Gi ±  5%   +226.16% (p=0.000 n=10)
CopyFloat32s/64-32                      37.88Gi ± 43%    147.17Gi ±  1%   +288.49% (p=0.000 n=10)
CopyFloat32s/4096-32                    31.72Gi ± 35%    435.54Gi ±  2%  +1272.98% (p=0.000 n=10)
CopyFloat32s/1048576-32                 40.90Gi ± 47%    139.05Gi ±  5%   +239.95% (p=0.000 n=10)
AndNotUint64s/64-32                     61.44Gi ±  6%    201.52Gi ±  5%   +228.01% (p=0.000 n=10)
AndNotUint64s/4096-32                   62.15Gi ±  1%    261.21Gi ±  3%   +320.26% (p=0.000 n=10)
AndNotUint64s/1048576-32                60.86Gi ±  5%    140.02Gi ± 13%   +130.05% (p=0.000 n=10)
NormalizeFloat32s/64-32                 33.02Gi ±  0%    138.62Gi ±  3%   +319.84% (p=0.000 n=10)
NormalizeFloat32s/4096-32               33.02Gi ±  0%    207.62Gi ±  3%   +528.74% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            32.90Gi ±  2%    198.37Gi ±  3%   +502.96% (p=0.000 n=10)
ChainInt32s/64-32                       48.63Gi ±  1%    146.00Gi ±  2%   +200.21% (p=0.000 n=10)
ChainInt32s/4096-32                     47.60Gi ± 15%    267.59Gi ±  3%   +462.12% (p=0.000 n=10)
ChainInt32s/1048576-32                  49.25Gi ±  1%    173.36Gi ±  5%   +252.02% (p=0.000 n=10)
WideInt32s/64-32                        31.47Gi ±  3%    116.03Gi ±  3%   +268.70% (p=0.000 n=10)
WideInt32s/4096-32                      31.64Gi ± 31%    180.27Gi ±  3%   +469.83% (p=0.000 n=10)
WideInt32s/1048576-32                   32.66Gi ±  2%    125.19Gi ± 21%   +283.27% (p=0.000 n=10)
geomean                                 35.04Gi           161.8Gi         +361.82%
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
identical; float reductions are not rewritten because regrouping changes them.
Loops under 64 elements (for `int32`) stay scalar.

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
