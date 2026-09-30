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
AddFloat32s/64-32                       18.785n ± 14%   4.961n ±  1%  -73.59% (p=0.000 n=10)
AddFloat32s/4096-32                     1001.4n ± 31%   168.2n ±  3%  -83.20% (p=0.000 n=10)
AddFloat32s/1048576-32                  231.27µ ± 10%   82.22µ ± 10%  -64.45% (p=0.000 n=10)
MulFloat32s/64-32                       15.250n ± 25%   5.639n ±  1%  -63.02% (p=0.000 n=10)
MulFloat32s/4096-32                      899.1n ±  8%   203.4n ± 11%  -77.37% (p=0.000 n=10)
MulFloat32s/1048576-32                  228.10µ ±  6%   92.01µ ±  9%  -59.66% (p=0.000 n=10)
ScalFloat32s/64-32                      11.140n ±  9%   3.657n ±  2%  -67.18% (p=0.000 n=10)
ScalFloat32s/4096-32                     749.0n ±  9%   146.4n ±  3%  -80.45% (p=0.000 n=10)
ScalFloat32s/1048576-32                 187.51µ ±  1%   36.78µ ±  3%  -80.38% (p=0.000 n=10)
MixFloat32s/64-32                       23.300n ±  1%   6.149n ±  3%  -73.61% (p=0.000 n=10)
MixFloat32s/4096-32                     1483.0n ± 16%   203.6n ±  3%  -86.27% (p=0.000 n=10)
MixFloat32s/1048576-32                  405.65µ ±  5%   90.67µ ± 10%  -77.65% (p=0.000 n=10)
AxpyFloat32s/64-32                      20.840n ±  5%   4.449n ±  1%  -78.65% (p=0.000 n=10)
AxpyFloat32s/4096-32                     870.5n ±  9%   158.2n ±  3%  -81.83% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 230.22µ ± 94%   92.24µ ±  8%  -59.93% (p=0.000 n=10)
NegFloat32s/64-32                       14.000n ±  8%   4.115n ±  3%  -70.61% (p=0.000 n=10)
NegFloat32s/4096-32                      759.3n ±  6%   155.4n ±  2%  -79.53% (p=0.000 n=10)
NegFloat32s/1048576-32                  194.56µ ±  2%   60.61µ ±  2%  -68.85% (p=0.000 n=10)
DivFloat32s/64-32                       28.695n ±  2%   5.125n ±  3%  -82.14% (p=0.000 n=10)
DivFloat32s/4096-32                     1799.0n ±  3%   166.6n ±  2%  -90.74% (p=0.000 n=10)
DivFloat32s/1048576-32                  471.02µ ±  2%   91.76µ ±  7%  -80.52% (p=0.000 n=10)
DaxpyFloat32s/64-32                     12.545n ±  2%   5.506n ±  4%  -56.11% (p=0.000 n=10)
DaxpyFloat32s/4096-32                    738.1n ±  2%   200.1n ±  4%  -72.89% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                194.98µ ±  1%   64.66µ ±  3%  -66.84% (p=0.000 n=10)
FillFloat32s/64-32                      11.960n ±  3%   3.237n ± 10%  -72.94% (p=0.000 n=10)
FillFloat32s/4096-32                    743.40n ±  3%   99.58n ± 12%  -86.60% (p=0.000 n=10)
FillFloat32s/1048576-32                 185.23µ ±  3%   30.19µ ±  3%  -83.70% (p=0.000 n=10)
FillUint8s/64-32                        12.120n ±  3%   2.117n ±  5%  -82.54% (p=0.000 n=10)
FillUint8s/4096-32                      773.65n ±  5%   25.37n ±  7%  -96.72% (p=0.000 n=10)
FillUint8s/1048576-32                  186.382µ ±  2%   5.974µ ±  2%  -96.80% (p=0.000 n=10)
ReverseIncFloat32s/64-32                13.510n ±  7%   3.756n ±  3%  -72.20% (p=0.000 n=10)
ReverseIncFloat32s/4096-32               824.1n ±  5%   124.7n ±  6%  -84.87% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           203.77µ ± 10%   60.79µ ±  5%  -70.17% (p=0.000 n=10)
CopyFloat32s/64-32                      12.465n ±  6%   3.297n ±  1%  -73.55% (p=0.000 n=10)
CopyFloat32s/4096-32                    740.55n ±  4%   71.97n ±  2%  -90.28% (p=0.000 n=10)
CopyFloat32s/1048576-32                 186.56µ ±  2%   56.81µ ±  5%  -69.55% (p=0.000 n=10)
AndNotUint64s/64-32                     23.495n ±  1%   8.939n ±  4%  -61.96% (p=0.000 n=10)
AndNotUint64s/4096-32                   1498.0n ±  2%   391.5n ±  5%  -73.87% (p=0.000 n=10)
AndNotUint64s/1048576-32                 383.8µ ±  1%   169.1µ ± 10%  -55.93% (p=0.000 n=10)
NormalizeFloat32s/64-32                 28.365n ±  2%   5.819n ±  2%  -79.48% (p=0.000 n=10)
NormalizeFloat32s/4096-32               1803.0n ±  3%   240.2n ±  2%  -86.68% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            474.03µ ±  2%   67.83µ ±  4%  -85.69% (p=0.000 n=10)
geomean                                  1.604µ         337.4n        -78.96%

                              │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                              │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                       38.08Gi ± 12%    144.17Gi ±  1%   +278.57% (p=0.000 n=10)
AddFloat32s/4096-32                     45.73Gi ± 24%    272.16Gi ±  3%   +495.16% (p=0.000 n=10)
AddFloat32s/1048576-32                  50.67Gi ±  9%    142.52Gi ±  9%   +181.27% (p=0.000 n=10)
MulFloat32s/64-32                       46.90Gi ± 20%    126.83Gi ±  1%   +170.42% (p=0.000 n=10)
MulFloat32s/4096-32                     50.92Gi ±  7%    225.11Gi ± 10%   +342.11% (p=0.000 n=10)
MulFloat32s/1048576-32                  51.38Gi ±  6%    127.36Gi ± 10%   +147.90% (p=0.000 n=10)
ScalFloat32s/64-32                      42.81Gi ±  9%    130.39Gi ±  2%   +204.58% (p=0.000 n=10)
ScalFloat32s/4096-32                    40.75Gi ±  8%    208.42Gi ±  3%   +411.49% (p=0.000 n=10)
ScalFloat32s/1048576-32                 41.66Gi ±  1%    212.40Gi ±  3%   +409.79% (p=0.000 n=10)
MixFloat32s/64-32                       30.70Gi ±  1%    116.34Gi ±  4%   +278.98% (p=0.000 n=10)
MixFloat32s/4096-32                     30.87Gi ± 14%    224.80Gi ±  3%   +628.20% (p=0.000 n=10)
MixFloat32s/1048576-32                  28.89Gi ±  5%    129.24Gi ± 11%   +347.37% (p=0.000 n=10)
AxpyFloat32s/64-32                      34.33Gi ±  5%    160.74Gi ±  1%   +368.19% (p=0.000 n=10)
AxpyFloat32s/4096-32                    52.59Gi ±  8%    289.38Gi ±  3%   +450.24% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 50.90Gi ± 48%    127.05Gi ±  9%   +149.58% (p=0.000 n=10)
NegFloat32s/64-32                       34.06Gi ±  9%    115.88Gi ±  3%   +240.20% (p=0.000 n=10)
NegFloat32s/4096-32                     40.19Gi ±  6%    196.32Gi ±  3%   +388.49% (p=0.000 n=10)
NegFloat32s/1048576-32                  40.15Gi ±  2%    128.90Gi ±  2%   +221.00% (p=0.000 n=10)
DivFloat32s/64-32                       24.93Gi ±  2%    139.55Gi ±  3%   +459.77% (p=0.000 n=10)
DivFloat32s/4096-32                     25.45Gi ±  3%    274.74Gi ±  2%   +979.69% (p=0.000 n=10)
DivFloat32s/1048576-32                  24.88Gi ±  2%    127.72Gi ±  8%   +413.33% (p=0.000 n=10)
DaxpyFloat32s/64-32                     57.01Gi ±  2%    129.89Gi ±  4%   +127.85% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   62.02Gi ±  2%    228.81Gi ±  4%   +268.90% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                60.10Gi ±  1%    181.23Gi ±  3%   +201.53% (p=0.000 n=10)
FillFloat32s/64-32                      19.93Gi ±  3%     73.69Gi ± 11%   +269.70% (p=0.000 n=10)
FillFloat32s/4096-32                    20.53Gi ±  3%    153.23Gi ± 11%   +646.49% (p=0.000 n=10)
FillFloat32s/1048576-32                 21.09Gi ±  3%    129.38Gi ±  3%   +513.46% (p=0.000 n=10)
FillUint8s/64-32                        4.917Gi ±  3%    28.164Gi ±  5%   +472.77% (p=0.000 n=10)
FillUint8s/4096-32                      4.933Gi ±  5%   150.400Gi ±  7%  +2949.10% (p=0.000 n=10)
FillUint8s/1048576-32                   5.241Gi ±  2%   163.485Gi ±  2%  +3019.64% (p=0.000 n=10)
ReverseIncFloat32s/64-32                35.29Gi ±  7%    126.96Gi ±  3%   +259.77% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              37.03Gi ±  5%    244.80Gi ±  5%   +561.07% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           38.34Gi ±  9%    128.52Gi ±  5%   +235.21% (p=0.000 n=10)
CopyFloat32s/64-32                      38.25Gi ±  5%    144.60Gi ±  1%   +278.01% (p=0.000 n=10)
CopyFloat32s/4096-32                    41.21Gi ±  4%    424.04Gi ±  2%   +929.04% (p=0.000 n=10)
CopyFloat32s/1048576-32                 41.88Gi ±  2%    137.53Gi ±  5%   +228.40% (p=0.000 n=10)
AndNotUint64s/64-32                     60.88Gi ±  1%    160.04Gi ±  4%   +162.87% (p=0.000 n=10)
AndNotUint64s/4096-32                   61.13Gi ±  2%    233.84Gi ±  5%   +282.56% (p=0.000 n=10)
AndNotUint64s/1048576-32                61.07Gi ±  1%    138.58Gi ± 11%   +126.91% (p=0.000 n=10)
NormalizeFloat32s/64-32                 33.63Gi ±  2%    163.88Gi ±  2%   +387.34% (p=0.000 n=10)
NormalizeFloat32s/4096-32               33.85Gi ±  2%    254.07Gi ±  2%   +650.67% (p=0.000 n=10)
NormalizeFloat32s/1048576-32            32.96Gi ±  2%    230.37Gi ±  4%   +598.90% (p=0.000 n=10)
geomean                                 33.50Gi           159.3Gi         +375.36%
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

**Loop shapes.** Range loops must declare their index (`for i := range x`). The
body must be assignments to `dst[i]`, with no temporaries or `if`; `dst[i+1]`,
`a[i+1]` and `a[0]` are not rewritten.

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
