# loopvec

`loopvec` analyzes Go source packages and rewrites element-wise loops to use Go's
experimental [simd](https://pkg.go.dev/simd) package for automatic SIMD
vectorization. Rewritten code requires `GOEXPERIMENT=simd` (Go 1.27+).

Go blog post: https://go.dev/blog/simd-experiment

## What it detects

The tool recognizes these loop shapes and rewrites them to use portable SIMD
operations that lower to AVX-512/AVX2/NEON depending on the target CPU:

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

## Performance

Benchmarks from [bench/](bench/) on AMD Ryzen 9 9950X3D (AVX-512), `float32` and `uint8` operations:

```
                              │     scalar     │                simd                  │
                              │     sec/op     │    sec/op     vs base                │
AddFloat32s/64-32               13.065n ±  4%   5.405n ±  6%  -58.63% (p=0.000 n=10)
AddFloat32s/4096-32              866.0n ±  3%   261.5n ± 13%  -69.80% (p=0.000 n=10)
AddFloat32s/1048576-32          240.83µ ±  7%   94.74µ ±  8%  -60.66% (p=0.000 n=10)
MulFloat32s/64-32               13.345n ±  5%   5.343n ±  6%  -59.96% (p=0.000 n=10)
MulFloat32s/4096-32              859.4n ±  5%   267.2n ±  5%  -68.91% (p=0.000 n=10)
MulFloat32s/1048576-32          229.33µ ± 10%   93.53µ ±  7%  -59.22% (p=0.000 n=10)
ScalFloat32s/64-32              10.985n ±  5%   4.041n ±  9%  -63.21% (p=0.000 n=10)
ScalFloat32s/4096-32             768.6n ±  4%   149.5n ±  7%  -80.55% (p=0.000 n=10)
ScalFloat32s/1048576-32         193.46µ ±  5%   37.71µ ±  7%  -80.51% (p=0.000 n=10)
MixFloat32s/64-32               24.460n ±  9%   5.828n ±  7%  -76.17% (p=0.000 n=10)
MixFloat32s/4096-32             1531.5n ±  4%   253.2n ±  5%  -83.47% (p=0.000 n=10)
MixFloat32s/1048576-32          403.51µ ±  5%   97.81µ ±  9%  -75.76% (p=0.000 n=10)
AxpyFloat32s/64-32              13.285n ±  9%   6.151n ±  8%  -53.70% (p=0.000 n=10)
AxpyFloat32s/4096-32             831.5n ± 10%   308.6n ±  6%  -62.88% (p=0.000 n=10)
AxpyFloat32s/1048576-32         226.36µ ±  5%   96.97µ ±  7%  -57.16% (p=0.000 n=10)
NegFloat32s/64-32               15.895n ±  5%   4.370n ±  8%  -72.51% (p=0.000 n=10)
NegFloat32s/4096-32              914.4n ±  8%   192.4n ±  3%  -78.95% (p=0.000 n=10)
NegFloat32s/1048576-32          221.19µ ± 13%   67.94µ ±  9%  -69.28% (p=0.000 n=10)
DivFloat32s/64-32               29.125n ±  2%   5.514n ± 11%  -81.07% (p=0.000 n=10)
DivFloat32s/4096-32             1836.0n ±  1%   266.8n ±  5%  -85.47% (p=0.000 n=10)
DivFloat32s/1048576-32          475.04µ ±  2%   93.67µ ± 11%  -80.28% (p=0.000 n=10)
DaxpyFloat32s/64-32             13.975n ±  6%   5.089n ±  9%  -63.58% (p=0.000 n=10)
DaxpyFloat32s/4096-32            859.0n ±  7%   242.2n ±  5%  -71.80% (p=0.000 n=10)
DaxpyFloat32s/1048576-32        210.13µ ± 13%   67.68µ ±  7%  -67.79% (p=0.000 n=10)
FillFloat32s/64-32               8.452n ± 30%   2.873n ±  5%  -66.01% (p=0.000 n=10)
FillFloat32s/4096-32            755.15n ±  4%   89.34n ±  8%  -88.17% (p=0.000 n=10)
FillFloat32s/1048576-32         190.85µ ±  7%   31.18µ ±  3%  -83.67% (p=0.000 n=10)
FillUint8s/64-32                 8.735n ±  7%   1.937n ±  7%  -77.82% (p=0.000 n=10)
FillUint8s/4096-32              759.50n ±  1%   21.09n ± 11%  -97.22% (p=0.000 n=10)
FillUint8s/1048576-32          191.739µ ±  3%   6.123µ ±  7%  -96.81% (p=0.000 n=10)
ReverseIncFloat32s/64-32        14.340n ±  4%   3.783n ±  9%  -73.62% (p=0.000 n=10)
ReverseIncFloat32s/4096-32       774.7n ±  6%   171.6n ±  7%  -77.86% (p=0.000 n=10)
ReverseIncFloat32s/1048576-32   209.86µ ±  8%   64.83µ ±  8%  -69.11% (p=0.000 n=10)
geomean                           1.470µ        345.7n       -76.48%
```

Running `loopvec -methods -split` on [gorgonia/tensor](https://github.com/gorgonia/tensor)
detects 176 vectorizable loops, yielding the following speedups on AMD Ryzen 9 9950X3D (AVX-512):

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

Running `loopvec -methods -split` on [gonum](https://github.com/gonum/gonum)
detects loops in multiple packages (floats, blas, lapack, stat, and others).
Benchmarks for `gonum/floats.Mul` and `MulTo` on AMD Ryzen 9 9950X3D (AVX-512):

```
              │    old.txt    │               new.txt               │
              │    sec/op     │    sec/op     vs base               │
MulMed-32        211.9n ± ∞ ¹   122.6n ± ∞ ¹  -42.14% (p=0.008 n=5)
MulLarge-32      23.36µ ± ∞ ¹   16.80µ ± ∞ ¹  -28.09% (p=0.008 n=5)
MulHuge-32       3.227m ± ∞ ¹   2.977m ± ∞ ¹        ~ (p=0.690 n=5)
MulToMed-32     199.10n ± ∞ ¹   97.68n ± ∞ ¹  -50.94% (p=0.008 n=5)
MulToLarge-32    20.37µ ± ∞ ¹   16.37µ ± ∞ ¹  -19.62% (p=0.008 n=5)
MulToHuge-32     5.004m ± ∞ ¹   4.462m ± ∞ ¹        ~ (p=0.222 n=5)
geomean          26.21µ         18.77µ        -28.38%
```

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

`-json` prints one JSON line per loop that at least looks like an
element-wise indexed-store loop, whether or not loopvec actually rewrote it:

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
your repository — the scalar version builds by default, and the simd version
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

## Example

**Input (`ops.go`):**

```go
package ops

func AddFloat32s(dst, a, b []float32) {
    for i := range dst {
        dst[i] = a[i] + b[i]
    }
}
```

**After `loopvec -split ops.go`:**

`ops.go` (original, now guarded):
```go
//go:build !goexperiment.simd

package ops

func AddFloat32s(dst, a, b []float32) {
    for i := range dst {
        dst[i] = a[i] + b[i]
    }
}
```

`ops_simd.go` (new file):
```go
//go:build goexperiment.simd

package ops

import "simd"

func AddFloat32s(dst, a, b []float32) {
    for _i := 0; _i < len(dst); {
        _v1, _n := simd.LoadFloat32sPart(a[_i:])
        _v2, _ := simd.LoadFloat32sPart(b[_i:])
        _v1.Add(_v2).StorePart(dst[_i:])
        _i += _n
    }
}
```

## Whole-program mode: `-toolexec`

`loopvec-toolexec` is an experimental `go build -toolexec` wrapper that vectorizes
loops while the go command compiles a build, dependencies and standard library
included, without touching any source file. See [toolexec.md](toolexec.md).

## Requirements

- Go 1.27 or later
- `GOEXPERIMENT=simd` at build time for the rewritten code

## Known Limitations

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

## Testing

The testing suite is built on top of [rsc.io/script](https://pkg.go.dev/rsc.io/script) with test scripts in `testdata/*.txt`,
the same approach used by [`go` command tests](https://github.com/golang/go/tree/f2d76b7/src/cmd/go/testdata/script).

```sh
make test
```

[tsvc/](tsvc/) is a Go port of the [TSVC_2](https://github.com/UoB-HPC/TSVC_2)
vectorizer kernel suite. It exists to test rewrites, and as a guardrail for drifts in the scalar vs SIMD builds. See [tsvc/README.md](tsvc/README.md).
