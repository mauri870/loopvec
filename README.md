# loopvec

`loopvec` analyzes Go source packages and rewrites element-wise loops to use Go's
experimental [simd](https://pkg.go.dev/simd) package for automatic SIMD
vectorization. Rewritten code requires `GOEXPERIMENT=simd` (Go 1.27+).

## What it detects

The tool recognizes these loop shapes and rewrites them to use portable SIMD
operations that lower to AVX-512/AVX2/NEON depending on the target CPU:

| Pattern | Operation |
|---|---|
| `for i := range dst { dst[i] = a[i] + b[i] }` | element-wise binary op |
| `for i := range dst { dst[i] += src[i] }` | in-place binary op |
| `for i := range dst { dst[i] *= scalar }` | scalar broadcast op |
| `for i := range dst { dst[i] = 0 }` | fill (broadcast literal) |
| `for i, v := range src { dst[i] = v * f }` | two-variable range scalar op |
| `for i := 0; i < len(s); i++ { ... }` | three-clause for (same body shapes) |

Supported element types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
`uint32`, `uint64`, `float32`, `float64`.

Supported operators: `+`, `-`, `*`, `&`, `|`, `^` (and their `op=` forms).

## Performance

Benchmarks from [bench/](bench/) on AMD Ryzen 9 9950X3D (AVX-512), `float32` operations:

```
                        │    scalar     │              simd               │
                        │    sec/op     │   sec/op     vs base            │
AddFloat32s/64-32           22.750n ±  8%   5.401n ± 2%  -76.26% (p=0.000 n=10)
AddFloat32s/4096-32          899.3n ±  2%   260.3n ± 2%  -71.05% (p=0.000 n=10)
AddFloat32s/1048576-32      234.92µ ±  2%   95.15µ ± 5%  -59.50% (p=0.000 n=10)
MulFloat32s/64-32           24.825n ± 14%   5.347n ± 2%  -78.46% (p=0.000 n=10)
MulFloat32s/4096-32          858.6n ±  8%   257.0n ± 2%  -70.07% (p=0.000 n=10)
MulFloat32s/1048576-32      231.60µ ±  7%   92.05µ ± 4%  -60.25% (p=0.000 n=10)
ScalFloat32s/64-32          10.570n ±  5%   3.936n ± 3%  -62.77% (p=0.000 n=10)
ScalFloat32s/4096-32         740.8n ±  1%   148.0n ± 1%  -80.02% (p=0.000 n=10)
ScalFloat32s/1048576-32     191.06µ ±  1%   37.36µ ± 3%  -80.45% (p=0.000 n=10)
geomean                       1.487µ         415.4n       -72.06%
```

Running `loopvec` on [gonum](https://github.com/gonum/gonum) detects 38
vectorizable loops across 17 files, including LAPACK kernels, optimization
functions, and statistical routines. BLAS methods are excluded due to the
compiler limitation noted below.

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

**Methods are not rewritten.** The experimental SIMD compiler crashes with an
internal error when a `//go:build goexperiment.simd` file contains a method
(function with a receiver). Only top-level functions are vectorized.
Tracked at [golang/go#80657](https://github.com/golang/go/issues/80657).

## Testing

The test suite uses [txtar](https://pkg.go.dev/golang.org/x/tools/txtar) archives
in `testdata/`. Each archive defines the input, expected rewrite, and an optional
test program that is compiled and run both with and without `GOEXPERIMENT=simd` to
verify behavioral equivalence.

```sh
GOTOOLCHAIN=go1.27.1 go test -count=1 ./...
```
