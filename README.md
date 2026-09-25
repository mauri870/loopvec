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

Synthetic benchmarks on AMD Ryzen 9 9950X3D (AVX-512), 1M `float32` elements:

```
               │   scalar    │             simd              │
               │   sec/op    │   sec/op    vs base           │
ScalUnitary-32   189.38µ ± 2%   33.86µ ± 2%  -82.12% (p=0.000 n=10)
AddSlices-32     245.53µ ± 3%   92.57µ ± 4%  -62.30% (p=0.000 n=10)
geomean           215.6µ         55.99µ       -74.04%
```

Running `loopvec` on [gonum](https://github.com/gonum/gonum) detects 140
vectorizable loops across 46 files, including BLAS level-3 routines, LAPACK
kernels, and statistical functions.

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
