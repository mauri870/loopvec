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
| `for i := range dst { dst[i] = 0 }` | fill (broadcast literal) |
| `for i, v := range src { dst[i] = v * f }` | two-variable range scalar op |
| `for i := range dst { dst[i] = a[i]*alpha + b[i] }` | `MulAdd` (FMA) |
| `for i := range dst { dst[i] = a[i]*alpha + b[i]*beta }` | `MulAdd` + `Mul` (two-scalar axpy) |
| `for i := range dst { dst[i] += a[i]*alpha }` | in-place `MulAdd` (DAXPY) |
| `for i := range dst { dst[i] = -src[i] }` | `Neg` (unary negation) |
| `for i := range dst { dst[i] = ^src[i] }` | `Not` (unary bitwise NOT) |
| `for i := 0; i < len(s); i++ { ... }` | three-clause for (all body shapes above) |

Supported element types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
`uint32`, `uint64`, `float32`, `float64`.

Supported binary operators: `+`, `-`, `*`, `/`, `&`, `|`, `^` (and their `op=` forms). `/` requires `float32` or `float64` (no integer division in simd).
Supported unary operators: `-` (negation, all except unsigned integers), `^` (bitwise NOT, all integer types).
`MulAdd`/FMA patterns require `float32` or `float64`.
Note: `*` is not supported for `int64` and `uint64` (no SIMD multiply for 64-bit integers).

## Performance

Benchmarks from [bench/](bench/) on AMD Ryzen 9 9950X3D (AVX-512), `float32` operations:

```
                         │    scalar     │              simd               │
                         │    sec/op     │   sec/op     vs base            │
AddFloat32s/64-32            12.780n ± 1%   5.340n ± 1%  -58.22% (p=0.000 n=10)
AddFloat32s/4096-32           820.4n ± 1%   256.7n ± 1%  -68.71% (p=0.000 n=10)
AddFloat32s/1048576-32       220.49µ ± 1%   92.98µ ± 4%  -57.83% (p=0.000 n=10)
MulFloat32s/64-32            13.025n ± 1%   5.278n ± 1%  -59.48% (p=0.000 n=10)
MulFloat32s/4096-32           841.2n ± 1%   256.4n ± 2%  -69.52% (p=0.000 n=10)
MulFloat32s/1048576-32       220.52µ ± 1%   94.63µ ± 4%  -57.09% (p=0.000 n=10)
ScalFloat32s/64-32           12.445n ± 1%   3.614n ± 1%  -70.96% (p=0.000 n=10)
ScalFloat32s/4096-32          819.1n ± 1%   128.7n ± 1%  -84.29% (p=0.000 n=10)
ScalFloat32s/1048576-32      193.09µ ± 1%   33.61µ ± 1%  -82.59% (p=0.000 n=10)
MixFloat32s/64-32            23.305n ± 1%   5.468n ± 1%  -76.54% (p=0.000 n=10)
MixFloat32s/4096-32          1489.0n ± 1%   250.2n ± 0%  -83.20% (p=0.000 n=10)
MixFloat32s/1048576-32       389.37µ ± 2%   94.02µ ± 3%  -75.85% (p=0.000 n=10)
AxpyFloat32s/64-32           16.580n ± 2%   5.602n ± 1%  -66.21% (p=0.000 n=10)
AxpyFloat32s/4096-32          961.4n ± 3%   264.2n ± 1%  -72.52% (p=0.000 n=10)
AxpyFloat32s/1048576-32      254.42µ ± 2%   91.11µ ± 4%  -64.19% (p=0.000 n=10)
NegFloat32s/64-32            14.065n ± 1%   4.851n ± 1%  -65.51% (p=0.000 n=10)
NegFloat32s/4096-32           811.2n ± 3%   200.0n ± 1%  -75.35% (p=0.000 n=10)
NegFloat32s/1048576-32       210.29µ ± 2%   63.92µ ± 1%  -69.60% (p=0.000 n=10)
DivFloat32s/64-32            28.620n ± 1%   5.291n ± 1%  -81.51% (p=0.000 n=10)
DivFloat32s/4096-32          1830.0n ± 0%   258.8n ± 1%  -85.86% (p=0.000 n=10)
DivFloat32s/1048576-32       472.00µ ± 1%   92.88µ ± 2%  -80.32% (p=0.000 n=10)
DaxpyFloat32s/64-32          13.375n ± 2%   4.567n ± 2%  -65.85% (p=0.000 n=10)
DaxpyFloat32s/4096-32         843.4n ± 3%   196.8n ± 1%  -76.67% (p=0.000 n=10)
DaxpyFloat32s/1048576-32     211.81µ ± 3%   65.48µ ± 1%  -69.09% (p=0.000 n=10)
geomean                        1.605µ        434.5n       -72.93%
```

Running `loopvec` on [gorgonia/tensor](https://github.com/gorgonia/tensor)
detects 94 vectorizable loops across the core arithmetic execution package,
yielding the following speedups on AMD Ryzen 9 9950X3D (AVX-512):

```
                          │    scalar     │              simd               │
                          │    sec/op     │   sec/op     vs base            │
AddVSF32/64-32               12.315n ± 1%   3.608n ± 2%  -70.70% (p=0.002 n=6)
AddVSF32/4096-32              759.8n ± 2%   146.3n ± 1%  -80.75% (p=0.002 n=6)
AddVSF32/1048576-32          201.41µ ± 3%   37.42µ ± 2%  -81.42% (p=0.002 n=6)
MulVSF32/64-32               14.065n ± 2%   3.771n ± 7%  -73.19% (p=0.002 n=6)
MulVSF32/4096-32              843.9n ± 2%   149.2n ± 1%  -82.33% (p=0.002 n=6)
MulVSF32/1048576-32          219.59µ ± 3%   37.53µ ± 3%  -82.91% (p=0.002 n=6)
AddVSF64/64-32               12.475n ± 1%   5.716n ± 6%  -54.18% (p=0.002 n=6)
AddVSF64/4096-32              756.8n ± 3%   289.3n ± 2%  -61.77% (p=0.002 n=6)
AddVSF64/1048576-32          204.91µ ± 2%   75.05µ ± 2%  -63.37% (p=0.002 n=6)
MulVSF64/64-32               12.235n ± 1%   6.027n ± 2%  -50.74% (p=0.002 n=6)
MulVSF64/4096-32              756.1n ± 1%   286.1n ± 1%  -62.16% (p=0.002 n=6)
MulVSF64/1048576-32          204.16µ ± 1%   74.65µ ± 1%  -63.44% (p=0.002 n=6)
VecAddI32/64-32              12.040n ± 2%   4.350n ± 2%  -63.87% (p=0.002 n=6)
VecAddI32/4096-32             764.0n ± 3%   195.5n ± 1%  -74.41% (p=0.002 n=6)
VecAddI32/1048576-32         204.32µ ± 2%   63.44µ ± 3%  -68.95% (p=0.002 n=6)
VecMulI32/64-32              14.405n ± 1%   4.409n ± 3%  -69.40% (p=0.002 n=6)
VecMulI32/4096-32             948.9n ± 1%   196.1n ± 3%  -79.33% (p=0.002 n=6)
VecMulI32/1048576-32         249.27µ ± 1%   65.54µ ± 2%  -73.71% (p=0.002 n=6)
geomean                        1.302µ        373.9n       -71.28%
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

## Requirements

- Go 1.27 or later
- `GOEXPERIMENT=simd` at build time for the rewritten code

## Known Limitations

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

## Testing

The test suite uses [txtar](https://pkg.go.dev/golang.org/x/tools/txtar) archives
in `testdata/`. Each archive defines the input, expected rewrite, and an optional
test program that is compiled and run both with and without `GOEXPERIMENT=simd` to
verify behavioral equivalence.

```sh
GOTOOLCHAIN=go1.27.1 go test -count=1 ./...
```
