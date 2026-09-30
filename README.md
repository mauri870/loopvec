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

Supported binary operators: `+`, `-`, `*`, `/`, `&`, `|`, `^`, `&^` (and their `op=` forms). `/` requires `float32` or `float64` (no integer division in simd).
Supported unary operators: `-` (negation, all except unsigned integers), `^` (bitwise NOT, all integer types).
`MulAdd`/FMA patterns require `float32` or `float64`.
Note: `*` is not supported for `int64` and `uint64` (no SIMD multiply for 64-bit integers).
Zero fills (`dst[i] = 0`) are left alone: the compiler already turns them into `memclr`,
which is faster than a vector loop.

**Copy loops** become the `copy` builtin, which the runtime implements with wide
moves: the compiler turns a zero-fill loop into a `memclr` but leaves a copy loop
as one scalar load and store per element, and `copy` is 4-10x faster from 64
elements up (slightly slower below 4). `copy` moves elements as if through a
temporary, so it differs from the loop when `dst` starts after `src` inside the
same array (the loop repeats the first element); the overlap check above sends
that case to the original loop, and a source shorter than the destination still
panics before anything is written. The rewritten code does not import `simd`.

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
of loop direction. The check does not tell an exact alias from an offset
one, so a call like `Add(x, x, y)`, which is safe to vectorize, also takes
the scalar loop. Only an operand that is written as the destination slice
itself (`dst[i] += a[i]*alpha`) is exempt.

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

Synthetic benchmarks from [bench/](bench/) on an AMD Ryzen 9 9950X3D (AVX-512) show **~4.9x faster**,
ranging from 2.4x to 39x depending on the kernel

<details>
<summary>bench.txt</summary>

```
goos: linux
goarch: amd64
pkg: github.com/mauri870/loopvec/bench
cpu: AMD Ryzen 9 9950X3D 16-Core Processor          
                              │ /tmp/bench_scalar.txt │         /tmp/bench_simd.txt          │
                              │        sec/op         │    sec/op     vs base                │
AddFloat32s/64-32                       15.150n ±  8%   5.942n ±  2%  -60.78% (p=0.000 n=10)
AddFloat32s/4096-32                      881.9n ±  5%   200.6n ±  4%  -77.25% (p=0.000 n=10)
AddFloat32s/1048576-32                  224.43µ ±  2%   90.20µ ±  5%  -59.81% (p=0.000 n=10)
MulFloat32s/64-32                       15.100n ±  6%   5.028n ±  2%  -66.70% (p=0.000 n=10)
MulFloat32s/4096-32                      901.9n ± 17%   168.9n ±  2%  -81.27% (p=0.000 n=10)
MulFloat32s/1048576-32                  235.55µ ±  5%   88.14µ ±  7%  -62.58% (p=0.000 n=10)
ScalFloat32s/64-32                      11.245n ±  8%   3.671n ±  4%  -67.36% (p=0.000 n=10)
ScalFloat32s/4096-32                     746.6n ±  1%   129.1n ±  2%  -82.71% (p=0.000 n=10)
ScalFloat32s/1048576-32                 190.28µ ±  1%   34.21µ ±  4%  -82.02% (p=0.000 n=10)
MixFloat32s/64-32                       23.220n ±  0%   5.075n ±  2%  -78.14% (p=0.000 n=10)
MixFloat32s/4096-32                     1471.5n ±  0%   175.6n ±  3%  -88.07% (p=0.000 n=10)
MixFloat32s/1048576-32                  391.35µ ±  3%   88.66µ ±  8%  -77.34% (p=0.000 n=10)
AxpyFloat32s/64-32                      14.230n ± 17%   5.984n ±  1%  -57.95% (p=0.000 n=10)
AxpyFloat32s/4096-32                     907.2n ±  4%   174.8n ±  4%  -80.74% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 243.72µ ±  6%   91.36µ ± 13%  -62.52% (p=0.000 n=10)
NegFloat32s/64-32                       14.490n ±  3%   4.368n ±  3%  -69.86% (p=0.000 n=10)
NegFloat32s/4096-32                      771.1n ±  1%   148.2n ±  1%  -80.79% (p=0.000 n=10)
NegFloat32s/1048576-32                  204.84µ ±  4%   57.92µ ±  8%  -71.72% (p=0.000 n=10)
DivFloat32s/64-32                       29.010n ±  2%   5.764n ±  2%  -80.13% (p=0.000 n=10)
DivFloat32s/4096-32                     1833.5n ±  1%   199.6n ±  5%  -89.11% (p=0.000 n=10)
DivFloat32s/1048576-32                  471.51µ ±  1%   85.50µ ±  9%  -81.87% (p=0.000 n=10)
DaxpyFloat32s/64-32                     12.835n ±  2%   4.400n ±  2%  -65.72% (p=0.000 n=10)
DaxpyFloat32s/4096-32                    752.8n ±  3%   165.4n ±  2%  -78.03% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                199.42µ ±  1%   61.70µ ±  5%  -69.06% (p=0.000 n=10)
FillFloat32s/64-32                      11.855n ±  3%   2.852n ±  6%  -75.94% (p=0.000 n=10)
FillFloat32s/4096-32                    759.80n ±  6%   85.39n ±  2%  -88.76% (p=0.000 n=10)
FillFloat32s/1048576-32                 191.59µ ±  7%   29.76µ ±  2%  -84.47% (p=0.000 n=10)
FillUint8s/64-32                        12.390n ±  6%   2.212n ±  3%  -82.14% (p=0.000 n=10)
FillUint8s/4096-32                      804.50n ±  6%   20.41n ±  4%  -97.46% (p=0.000 n=10)
FillUint8s/1048576-32                  188.750µ ±  2%   5.756µ ±  2%  -96.95% (p=0.000 n=10)
ReverseIncFloat32s/64-32                12.710n ±  8%   4.629n ±  2%  -63.58% (p=0.000 n=10)
ReverseIncFloat32s/4096-32               826.9n ±  9%   149.1n ±  1%  -81.97% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           210.60µ ±  4%   60.00µ ±  7%  -71.51% (p=0.000 n=10)
CopyFloat32s/64-32                      12.955n ± 11%   3.313n ±  1%  -74.43% (p=0.000 n=10)
CopyFloat32s/4096-32                    773.85n ±  5%   73.06n ±  1%  -90.56% (p=0.000 n=10)
CopyFloat32s/1048576-32                 192.95µ ±  3%   58.54µ ±  3%  -69.66% (p=0.000 n=10)
geomean                                  1.468µ         301.5n        -79.45%

                              │ /tmp/bench_scalar.txt │            /tmp/bench_simd.txt            │
                              │          B/s          │       B/s        vs base                  │
AddFloat32s/64-32                       47.22Gi ±  8%    120.38Gi ±  2%   +154.91% (p=0.000 n=10)
AddFloat32s/4096-32                     51.92Gi ±  5%    228.22Gi ±  3%   +339.60% (p=0.000 n=10)
AddFloat32s/1048576-32                  52.22Gi ±  2%    129.95Gi ±  5%   +148.87% (p=0.000 n=10)
MulFloat32s/64-32                       47.36Gi ±  6%    142.26Gi ±  2%   +200.36% (p=0.000 n=10)
MulFloat32s/4096-32                     50.75Gi ± 14%    270.91Gi ±  2%   +433.76% (p=0.000 n=10)
MulFloat32s/1048576-32                  49.75Gi ±  5%    132.96Gi ±  8%   +167.26% (p=0.000 n=10)
ScalFloat32s/64-32                      42.41Gi ±  8%    129.90Gi ±  3%   +206.32% (p=0.000 n=10)
ScalFloat32s/4096-32                    40.87Gi ±  1%    236.55Gi ±  1%   +478.72% (p=0.000 n=10)
ScalFloat32s/1048576-32                 41.06Gi ±  1%    228.35Gi ±  4%   +456.16% (p=0.000 n=10)
MixFloat32s/64-32                       30.81Gi ±  0%    140.94Gi ±  2%   +357.48% (p=0.000 n=10)
MixFloat32s/4096-32                     31.11Gi ±  0%    260.67Gi ±  2%   +737.95% (p=0.000 n=10)
MixFloat32s/1048576-32                  29.94Gi ±  3%    132.22Gi ±  7%   +341.56% (p=0.000 n=10)
AxpyFloat32s/64-32                      50.26Gi ± 15%    119.53Gi ±  1%   +137.85% (p=0.000 n=10)
AxpyFloat32s/4096-32                    50.46Gi ±  4%    261.98Gi ±  4%   +419.23% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 48.08Gi ±  7%    128.35Gi ± 15%   +166.93% (p=0.000 n=10)
NegFloat32s/64-32                       32.91Gi ±  3%    109.16Gi ±  3%   +231.72% (p=0.000 n=10)
NegFloat32s/4096-32                     39.57Gi ±  1%    205.95Gi ±  1%   +420.45% (p=0.000 n=10)
NegFloat32s/1048576-32                  38.14Gi ±  4%    134.90Gi ±  8%   +253.65% (p=0.000 n=10)
DivFloat32s/64-32                       24.66Gi ±  2%    124.08Gi ±  2%   +403.25% (p=0.000 n=10)
DivFloat32s/4096-32                     24.97Gi ±  1%    229.41Gi ±  5%   +818.79% (p=0.000 n=10)
DivFloat32s/1048576-32                  24.85Gi ±  1%    137.10Gi ±  8%   +451.62% (p=0.000 n=10)
DaxpyFloat32s/64-32                     55.72Gi ±  2%    162.58Gi ±  2%   +191.80% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   60.81Gi ±  3%    276.72Gi ±  2%   +355.10% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                58.76Gi ±  1%    189.93Gi ±  5%   +223.20% (p=0.000 n=10)
FillFloat32s/64-32                      20.11Gi ±  3%     83.60Gi ±  5%   +315.68% (p=0.000 n=10)
FillFloat32s/4096-32                    20.08Gi ±  6%    178.69Gi ±  2%   +789.77% (p=0.000 n=10)
FillFloat32s/1048576-32                 20.39Gi ±  6%    131.26Gi ±  2%   +543.79% (p=0.000 n=10)
FillUint8s/64-32                        4.810Gi ±  6%    26.940Gi ±  3%   +460.11% (p=0.000 n=10)
FillUint8s/4096-32                      4.742Gi ±  6%   186.973Gi ±  4%  +3842.98% (p=0.000 n=10)
FillUint8s/1048576-32                   5.174Gi ±  2%   169.657Gi ±  2%  +3179.10% (p=0.000 n=10)
ReverseIncFloat32s/64-32                37.52Gi ±  7%    103.03Gi ±  2%   +174.63% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              36.91Gi ±  8%    204.72Gi ±  1%   +454.71% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           37.10Gi ±  5%    130.30Gi ±  7%   +251.25% (p=0.000 n=10)
CopyFloat32s/64-32                      36.80Gi ± 10%    143.92Gi ±  1%   +291.07% (p=0.000 n=10)
CopyFloat32s/4096-32                    39.45Gi ±  5%    417.73Gi ±  1%   +958.87% (p=0.000 n=10)
CopyFloat32s/1048576-32                 40.49Gi ±  3%    133.45Gi ±  3%   +229.57% (p=0.000 n=10)
geomean                                 32.09Gi           156.2Gi         +386.75%
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

</details>

## Testing

The testing suite is built on top of [rsc.io/script](https://pkg.go.dev/rsc.io/script) with test scripts in `testdata/*.txt`,
the same approach used by [`go` command tests](https://github.com/golang/go/tree/f2d76b7/src/cmd/go/testdata/script).

```sh
make test
```

[tsvc/](tsvc/) is a Go port of the [TSVC_2](https://github.com/UoB-HPC/TSVC_2)
vectorizer kernel suite. It exists to test rewrites, and as a guardrail for drifts in the scalar vs SIMD builds. See [tsvc/README.md](tsvc/README.md).
