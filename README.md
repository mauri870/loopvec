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

Supported binary operators: `+`, `-`, `*`, `/`, `&`, `|`, `^` (and their `op=` forms). `/` requires `float32` or `float64` (no integer division in simd).
Supported unary operators: `-` (negation, all except unsigned integers), `^` (bitwise NOT, all integer types).
`MulAdd`/FMA patterns require `float32` or `float64`.
Note: `*` is not supported for `int64` and `uint64` (no SIMD multiply for 64-bit integers).
Zero fills (`dst[i] = 0`) are left alone: the compiler already turns them into `memclr`,
which is faster than a vector loop.

</details>

## Performance

Synthetic benchmarks from [bench/](bench/) on AMD Ryzen 9 9950X3D (AVX-512) show a **78% speedup**, resulting in a **4.56x throughput increase**.

<details>
<summary>bench.txt</summary>

```
goos: linux
goarch: amd64
pkg: github.com/mauri870/loopvec/bench
cpu: AMD Ryzen 9 9950X3D 16-Core Processor          
                              │ /tmp/bench_scalar.txt │         /tmp/bench_simd.txt          │
                              │        sec/op         │    sec/op     vs base                │
AddFloat32s/64-32                       12.725n ±  1%   4.877n ±  4%  -61.67% (p=0.000 n=10)
AddFloat32s/4096-32                      816.5n ±  2%   170.0n ±  3%  -79.18% (p=0.000 n=10)
AddFloat32s/1048576-32                  213.93µ ±  1%   83.99µ ±  5%  -60.74% (p=0.000 n=10)
MulFloat32s/64-32                       12.665n ±  2%   5.566n ±  2%  -56.05% (p=0.000 n=10)
MulFloat32s/4096-32                      820.0n ±  2%   198.5n ±  1%  -75.79% (p=0.000 n=10)
MulFloat32s/1048576-32                  215.38µ ±  3%   90.03µ ±  4%  -58.20% (p=0.000 n=10)
ScalFloat32s/64-32                      10.185n ±  8%   3.668n ± 10%  -63.99% (p=0.000 n=10)
ScalFloat32s/4096-32                     741.5n ±  1%   147.8n ±  2%  -80.07% (p=0.000 n=10)
ScalFloat32s/1048576-32                 190.89µ ±  2%   36.53µ ±  1%  -80.86% (p=0.000 n=10)
MixFloat32s/64-32                       23.600n ±  1%   6.095n ±  2%  -74.17% (p=0.000 n=10)
MixFloat32s/4096-32                     1495.5n ±  1%   201.4n ±  2%  -86.53% (p=0.000 n=10)
MixFloat32s/1048576-32                  385.23µ ±  2%   84.31µ ±  5%  -78.11% (p=0.000 n=10)
AxpyFloat32s/64-32                      12.870n ±  1%   4.514n ±  2%  -64.93% (p=0.000 n=10)
AxpyFloat32s/4096-32                     793.8n ±  1%   160.5n ±  2%  -79.78% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 211.80µ ±  2%   84.45µ ±  5%  -60.13% (p=0.000 n=10)
NegFloat32s/64-32                       14.595n ±  4%   4.210n ±  4%  -71.15% (p=0.000 n=10)
NegFloat32s/4096-32                      844.8n ±  4%   156.1n ±  2%  -81.53% (p=0.000 n=10)
NegFloat32s/1048576-32                  208.69µ ±  3%   59.58µ ±  4%  -71.45% (p=0.000 n=10)
DivFloat32s/64-32                       28.440n ±  2%   5.128n ±  1%  -81.97% (p=0.000 n=10)
DivFloat32s/4096-32                     1816.0n ±  1%   167.8n ±  4%  -90.76% (p=0.000 n=10)
DivFloat32s/1048576-32                  469.98µ ±  1%   86.22µ ±  6%  -81.65% (p=0.000 n=10)
DaxpyFloat32s/64-32                     12.760n ±  3%   5.591n ±  1%  -56.18% (p=0.000 n=10)
DaxpyFloat32s/4096-32                    839.2n ±  4%   200.1n ±  2%  -76.15% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                203.93µ ±  2%   64.92µ ±  2%  -68.17% (p=0.000 n=10)
FillFloat32s/64-32                       8.430n ± 18%   2.993n ±  3%  -64.50% (p=0.000 n=10)
FillFloat32s/4096-32                     737.8n ±  1%   105.6n ±  1%  -85.69% (p=0.000 n=10)
FillFloat32s/1048576-32                 185.22µ ±  2%   30.27µ ±  1%  -83.66% (p=0.000 n=10)
FillUint8s/64-32                         8.613n ± 13%   1.967n ±  5%  -77.16% (p=0.000 n=10)
FillUint8s/4096-32                      740.80n ±  2%   27.62n ±  3%  -96.27% (p=0.000 n=10)
FillUint8s/1048576-32                  186.676µ ±  1%   6.266µ ±  2%  -96.64% (p=0.000 n=10)
ReverseIncFloat32s/64-32                14.000n ±  3%   3.641n ±  2%  -73.99% (p=0.000 n=10)
ReverseIncFloat32s/4096-32               755.9n ±  3%   122.6n ±  2%  -83.77% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           199.84µ ±  3%   57.22µ ±  5%  -71.37% (p=0.000 n=10)
geomean                                  1.410µ         309.3n        -78.06%

                              │ /tmp/bench_scalar.txt │           /tmp/bench_simd.txt            │
                              │          B/s          │      B/s        vs base                  │
AddFloat32s/64-32                       56.20Gi ±  1%    146.64Gi ± 4%   +160.94% (p=0.000 n=10)
AddFloat32s/4096-32                     56.07Gi ±  2%    269.24Gi ± 3%   +380.21% (p=0.000 n=10)
AddFloat32s/1048576-32                  54.78Gi ±  1%    139.56Gi ± 5%   +154.76% (p=0.000 n=10)
MulFloat32s/64-32                       56.48Gi ±  2%    128.50Gi ± 2%   +127.52% (p=0.000 n=10)
MulFloat32s/4096-32                     55.83Gi ±  2%    230.61Gi ± 1%   +313.07% (p=0.000 n=10)
MulFloat32s/1048576-32                  54.41Gi ±  3%    130.17Gi ± 4%   +139.24% (p=0.000 n=10)
ScalFloat32s/64-32                      46.83Gi ±  8%    130.00Gi ± 9%   +177.60% (p=0.000 n=10)
ScalFloat32s/4096-32                    41.16Gi ±  1%    206.51Gi ± 2%   +401.75% (p=0.000 n=10)
ScalFloat32s/1048576-32                 40.93Gi ±  2%    213.88Gi ± 1%   +422.59% (p=0.000 n=10)
MixFloat32s/64-32                       30.31Gi ±  1%    117.35Gi ± 2%   +287.18% (p=0.000 n=10)
MixFloat32s/4096-32                     30.60Gi ±  1%    227.31Gi ± 2%   +642.81% (p=0.000 n=10)
MixFloat32s/1048576-32                  30.42Gi ±  2%    138.99Gi ± 5%   +356.92% (p=0.000 n=10)
AxpyFloat32s/64-32                      55.58Gi ±  1%    158.46Gi ± 2%   +185.10% (p=0.000 n=10)
AxpyFloat32s/4096-32                    57.67Gi ±  1%    285.16Gi ± 2%   +394.48% (p=0.000 n=10)
AxpyFloat32s/1048576-32                 55.33Gi ±  2%    138.77Gi ± 5%   +150.81% (p=0.000 n=10)
NegFloat32s/64-32                       32.67Gi ±  4%    113.25Gi ± 4%   +246.63% (p=0.000 n=10)
NegFloat32s/4096-32                     36.13Gi ±  4%    195.58Gi ± 2%   +441.35% (p=0.000 n=10)
NegFloat32s/1048576-32                  37.44Gi ±  3%    131.14Gi ± 5%   +250.30% (p=0.000 n=10)
DivFloat32s/64-32                       25.15Gi ±  2%    139.49Gi ± 1%   +454.71% (p=0.000 n=10)
DivFloat32s/4096-32                     25.21Gi ±  1%    272.92Gi ± 4%   +982.66% (p=0.000 n=10)
DivFloat32s/1048576-32                  24.93Gi ±  1%    135.91Gi ± 7%   +445.08% (p=0.000 n=10)
DaxpyFloat32s/64-32                     56.07Gi ±  3%    127.92Gi ± 2%   +128.16% (p=0.000 n=10)
DaxpyFloat32s/4096-32                   54.54Gi ±  4%    228.76Gi ± 2%   +319.42% (p=0.000 n=10)
DaxpyFloat32s/1048576-32                57.46Gi ±  2%    180.52Gi ± 2%   +214.14% (p=0.000 n=10)
FillFloat32s/64-32                      28.29Gi ± 16%     79.68Gi ± 3%   +181.65% (p=0.000 n=10)
FillFloat32s/4096-32                    20.68Gi ±  1%    144.60Gi ± 1%   +599.10% (p=0.000 n=10)
FillFloat32s/1048576-32                 21.09Gi ±  2%    129.05Gi ± 1%   +511.93% (p=0.000 n=10)
FillUint8s/64-32                        6.920Gi ± 11%    30.303Gi ± 5%   +337.87% (p=0.000 n=10)
FillUint8s/4096-32                      5.149Gi ±  2%   138.113Gi ± 3%  +2582.18% (p=0.000 n=10)
FillUint8s/1048576-32                   5.231Gi ±  1%   155.858Gi ± 2%  +2879.31% (p=0.000 n=10)
ReverseIncFloat32s/64-32                34.06Gi ±  3%    130.96Gi ± 2%   +284.52% (p=0.000 n=10)
ReverseIncFloat32s/4096-32              40.37Gi ±  3%    248.84Gi ± 2%   +516.34% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32           39.09Gi ±  3%    136.53Gi ± 5%   +249.23% (p=0.000 n=10)
geomean                                 33.32Gi           151.9Gi        +355.77%
```
</details>


Running `loopvec -methods -split` on [gorgonia/tensor](https://github.com/gorgonia/tensor)
detects 176 vectorizable loops, yielding a **~71% speedup** on a AMD Ryzen 9 9950X3D (AVX-512):

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
{"file":"/path/to/mypkg/ops.go","line":10,"func":"Stride","vectorized":false,"reason":"loop clauses (start, step, direction, or bound) do not match a supported shape"}
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
your repository, while scalar version builds by default the simd version
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

SIMD support in Go is experimental, so there is likely several bugs lurking around, both in the Go compiler/runtime and this tool.

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

</details>

## Testing

The testing suite is built on top of [rsc.io/script](https://pkg.go.dev/rsc.io/script) with test scripts in `testdata/*.txt`,
the same approach used by [`go` command tests](https://github.com/golang/go/tree/f2d76b7/src/cmd/go/testdata/script).

```sh
make test
```

[tsvc/](tsvc/) is a Go port of the [TSVC_2](https://github.com/UoB-HPC/TSVC_2)
vectorizer kernel suite. It exists to test rewrites, and as a guardrail for drifts in the scalar vs SIMD builds. See [tsvc/README.md](tsvc/README.md).
