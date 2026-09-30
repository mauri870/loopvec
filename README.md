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
AddFloat32s/64-32                      15.830n ±  10%   5.149n ±  2%  -67.47% (p=0.000 n=10)
AddFloat32s/4096-32                     966.2n ±   6%   165.2n ±  2%  -82.90% (p=0.000 n=10)
AddFloat32s/1048576-32                 228.61µ ±   5%   88.47µ ± 10%  -61.30% (p=0.000 n=10)
MulFloat32s/64-32                      15.755n ±   6%   5.901n ±  1%  -62.54% (p=0.000 n=10)
MulFloat32s/4096-32                     890.0n ±  17%   192.7n ±  1%  -78.35% (p=0.000 n=10)
MulFloat32s/1048576-32                 229.77µ ±   6%   88.91µ ± 12%  -61.31% (p=0.000 n=10)
ScalFloat32s/64-32                      9.943n ±   4%   3.749n ±  3%  -62.29% (p=0.000 n=10)
ScalFloat32s/4096-32                    744.0n ±   3%   148.9n ±  4%  -79.99% (p=0.000 n=10)
ScalFloat32s/1048576-32                186.56µ ±   2%   36.67µ ±  2%  -80.35% (p=0.000 n=10)
MixFloat32s/64-32                      23.195n ±   1%   5.644n ±  4%  -75.67% (p=0.000 n=10)
MixFloat32s/4096-32                    1485.0n ±   1%   203.7n ±  3%  -86.28% (p=0.000 n=10)
MixFloat32s/1048576-32                 383.19µ ±   2%   89.44µ ± 10%  -76.66% (p=0.000 n=10)
AxpyFloat32s/64-32                     13.765n ±   5%   4.736n ±  2%  -65.59% (p=0.000 n=10)
AxpyFloat32s/4096-32                    851.8n ±   4%   158.3n ±  3%  -81.41% (p=0.000 n=10)
AxpyFloat32s/1048576-32                232.55µ ±   6%   82.37µ ± 11%  -64.58% (p=0.000 n=10)
NegFloat32s/64-32                      12.795n ±   4%   4.369n ±  1%  -65.86% (p=0.000 n=10)
NegFloat32s/4096-32                     749.0n ±   2%   158.2n ±  2%  -78.88% (p=0.000 n=10)
NegFloat32s/1048576-32                 200.04µ ±   3%   58.10µ ±  7%  -70.95% (p=0.000 n=10)
DivFloat32s/64-32                      28.785n ±   2%   4.792n ±  4%  -83.35% (p=0.000 n=10)
DivFloat32s/4096-32                    1839.0n ±   2%   171.7n ±  3%  -90.66% (p=0.000 n=10)
DivFloat32s/1048576-32                 476.48µ ±   3%   87.81µ ±  7%  -81.57% (p=0.000 n=10)
DaxpyFloat32s/64-32                    12.690n ±   2%   5.404n ±  4%  -57.42% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   757.5n ±   6%   201.6n ±  2%  -73.39% (p=0.000 n=10)
DaxpyFloat32s/1048576-32               197.72µ ±   4%   64.72µ ±  4%  -67.27% (p=0.000 n=10)
FillFloat32s/64-32                     11.915n ±   2%   3.447n ±  6%  -71.07% (p=0.000 n=10)
FillFloat32s/4096-32                    755.3n ±   3%   102.4n ±  2%  -86.45% (p=0.000 n=10)
FillFloat32s/1048576-32                184.00µ ±   3%   30.16µ ±  3%  -83.61% (p=0.000 n=10)
FillUint8s/64-32                       12.080n ±   2%   2.364n ±  7%  -80.43% (p=0.000 n=10)
FillUint8s/4096-32                     752.00n ±   3%   26.40n ±  4%  -96.49% (p=0.000 n=10)
FillUint8s/1048576-32                 189.384µ ±  11%   6.114µ ±  2%  -96.77% (p=0.000 n=10)
ReverseIncFloat32s/64-32               13.100n ±   5%   4.149n ±  2%  -68.32% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              882.1n ±   8%   122.4n ±  7%  -86.12% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32          220.87µ ±   3%   61.31µ ±  7%  -72.24% (p=0.000 n=10)
CopyFloat32s/64-32                     14.220n ±  24%   3.239n ±  2%  -77.22% (p=0.000 n=10)
CopyFloat32s/4096-32                   777.95n ± 832%   72.52n ±  2%  -90.68% (p=0.000 n=10)
CopyFloat32s/1048576-32                189.97µ ±   2%   58.20µ ±  8%  -69.36% (p=0.000 n=10)
AndNotUint64s/64-32                    23.520n ±   2%   8.527n ±  2%  -63.75% (p=0.000 n=10)
AndNotUint64s/4096-32                  1503.0n ±   3%   399.8n ±  4%  -73.40% (p=0.000 n=10)
AndNotUint64s/1048576-32                384.9µ ±   1%   166.6µ ±  9%  -56.71% (p=0.000 n=10)
geomean                                 1.514µ          331.5n        -78.11%

                              │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                              │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                       45.18Gi ± 12%    138.90Gi ±  2%   +207.45% (p=0.000 n=10)
AddFloat32s/4096-32                     47.38Gi ±  6%    277.02Gi ±  2%   +484.70% (p=0.000 n=10)
AddFloat32s/1048576-32                  51.26Gi ±  5%    132.58Gi ± 11%   +158.63% (p=0.000 n=10)
MulFloat32s/64-32                       45.39Gi ±  7%    121.19Gi ±  1%   +166.98% (p=0.000 n=10)
MulFloat32s/4096-32                     51.44Gi ± 14%    237.61Gi ±  1%   +361.90% (p=0.000 n=10)
MulFloat32s/1048576-32                  51.01Gi ±  6%    132.03Gi ± 13%   +158.85% (p=0.000 n=10)
ScalFloat32s/64-32                      47.96Gi ±  4%    127.18Gi ±  2%   +165.18% (p=0.000 n=10)
ScalFloat32s/4096-32                    41.02Gi ±  3%    204.95Gi ±  3%   +399.65% (p=0.000 n=10)
ScalFloat32s/1048576-32                 41.88Gi ±  2%    213.07Gi ±  2%   +408.80% (p=0.000 n=10)
MixFloat32s/64-32                       30.83Gi ±  1%    126.73Gi ±  4%   +311.00% (p=0.000 n=10)
MixFloat32s/4096-32                     30.82Gi ±  1%    224.77Gi ±  3%   +629.23% (p=0.000 n=10)
MixFloat32s/1048576-32                  30.58Gi ±  2%    131.17Gi ± 11%   +328.92% (p=0.000 n=10)
AxpyFloat32s/64-32                      51.98Gi ±  5%    151.02Gi ±  2%   +190.56% (p=0.000 n=10)
AxpyFloat32s/4096-32                    53.74Gi ±  4%    289.11Gi ±  3%   +437.94% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 50.39Gi ±  5%    142.67Gi ± 10%   +183.11% (p=0.000 n=10)
NegFloat32s/64-32                       37.26Gi ±  4%    109.15Gi ±  1%   +192.97% (p=0.000 n=10)
NegFloat32s/4096-32                     40.75Gi ±  2%    192.95Gi ±  2%   +373.55% (p=0.000 n=10)
NegFloat32s/1048576-32                  39.06Gi ±  3%    134.48Gi ±  7%   +244.31% (p=0.000 n=10)
DivFloat32s/64-32                       24.85Gi ±  3%    149.26Gi ±  3%   +500.69% (p=0.000 n=10)
DivFloat32s/4096-32                     24.89Gi ±  2%    266.57Gi ±  3%   +971.08% (p=0.000 n=10)
DivFloat32s/1048576-32                  24.59Gi ±  3%    133.53Gi ±  7%   +442.92% (p=0.000 n=10)
DaxpyFloat32s/64-32                     56.36Gi ±  2%    132.35Gi ±  5%   +134.84% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   60.43Gi ±  6%    227.07Gi ±  2%   +275.78% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                59.27Gi ±  4%    181.06Gi ±  4%   +205.49% (p=0.000 n=10)
FillFloat32s/64-32                      20.01Gi ±  2%     69.16Gi ±  6%   +245.62% (p=0.000 n=10)
FillFloat32s/4096-32                    20.20Gi ±  3%    149.07Gi ±  2%   +637.94% (p=0.000 n=10)
FillFloat32s/1048576-32                 21.23Gi ±  3%    129.50Gi ±  3%   +510.02% (p=0.000 n=10)
FillUint8s/64-32                        4.934Gi ±  2%    25.219Gi ±  7%   +411.10% (p=0.000 n=10)
FillUint8s/4096-32                      5.073Gi ±  3%   144.459Gi ±  4%  +2747.76% (p=0.000 n=10)
FillUint8s/1048576-32                   5.157Gi ± 10%   159.740Gi ±  2%  +2997.82% (p=0.000 n=10)
ReverseIncFloat32s/64-32                36.40Gi ±  5%    114.93Gi ±  2%   +215.72% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              34.60Gi ±  9%    249.20Gi ±  7%   +620.22% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           35.37Gi ±  3%    127.43Gi ±  7%   +260.25% (p=0.000 n=10)
CopyFloat32s/64-32                      33.82Gi ± 20%    147.21Gi ±  2%   +335.27% (p=0.000 n=10)
CopyFloat32s/4096-32                    39.23Gi ± 89%    420.80Gi ±  2%   +972.68% (p=0.000 n=10)
CopyFloat32s/1048576-32                 41.13Gi ±  2%    134.24Gi ±  9%   +226.41% (p=0.000 n=10)
AndNotUint64s/64-32                     60.81Gi ±  2%    167.76Gi ±  2%   +175.86% (p=0.000 n=10)
AndNotUint64s/4096-32                   60.91Gi ±  2%    228.99Gi ±  4%   +275.92% (p=0.000 n=10)
AndNotUint64s/1048576-32                60.90Gi ±  1%    140.73Gi ± 10%   +131.08% (p=0.000 n=10)
geomean                                 33.92Gi           155.0Gi         +356.83%
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
body must be one assignment to `dst[i]`; `dst[i+1]`, `a[i+1]` and `a[0]` are not
rewritten.

</details>

## Testing

The testing suite is built on top of [rsc.io/script](https://pkg.go.dev/rsc.io/script) with test scripts in `testdata/*.txt`,
the same approach used by [`go` command tests](https://github.com/golang/go/tree/f2d76b7/src/cmd/go/testdata/script).

```sh
make test
```

[tsvc/](tsvc/) is a Go port of the [TSVC_2](https://github.com/UoB-HPC/TSVC_2)
vectorizer kernel suite (93 of 151 kernels). It checks the scalar and SIMD builds
against the same golden on amd64 and arm64, and records which kernels loopvec
vectorizes (9 today). See [tsvc/README.md](tsvc/README.md).
