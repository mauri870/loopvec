# loopvec

`loopvec` analyzes Go source packages and rewrites element-wise loops to use Go's
experimental [simd](https://pkg.go.dev/simd) package for automatic SIMD
vectorization. Rewritten code requires `GOEXPERIMENT=simd` (Go 1.27+).

## What it detects

The tool recognizes three loop shapes and rewrites them to use portable SIMD
operations that lower to AVX-512/AVX2/NEON depending on the target CPU:

| Pattern | Operation |
|---|---|
| `for i := range dst { dst[i] = a[i] + b[i] }` | element-wise binary op |
| `for i := range dst { dst[i] += src[i] }` | in-place binary op |
| `for i := range dst { dst[i] *= scalar }` | scalar broadcast |

Supported element types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
`uint32`, `uint64`, `float32`, `float64`.

Supported operators: `+`, `-`, `*`, `&`, `|`, `^` (and their `op=` forms).

## Performance

On AMD Ryzen 9 9950X3D (AVX-512), 1M `float32` elements:

```
               │   scalar    │             simd              │
               │   sec/op    │   sec/op    vs base           │
ScalUnitary-32   189.38µ ± 2%   33.86µ ± 2%  -82.12% (p=0.000 n=10)
AddSlices-32     245.53µ ± 3%   92.57µ ± 4%  -62.30% (p=0.000 n=10)
geomean           215.6µ         55.99µ       -74.04%
```

## Installation

```sh
go install github.com/mauri870/loopvec@latest
```

## Usage

### Standalone

Print the rewritten source to stdout (dry run):

```sh
loopvec ./...
loopvec ./mypkg/...
```

Rewrite files in place:

```sh
loopvec -w ./...
```

### As a `go tool` (Go 1.24+)

Add `loopvec` as a tool dependency in your module:

```sh
go get -tool github.com/mauri870/loopvec@latest
```

Then invoke it without installing globally:

```sh
go tool loopvec ./...
go tool loopvec -w ./...
```

## Output

The rewritten file gets a `//go:build goexperiment.simd` build constraint added
automatically. The original (scalar) code will not be compiled when
`GOEXPERIMENT=simd` is set; you will typically want to keep the original file
as a `!goexperiment.simd` fallback.

A typical workflow:

```sh
# 1. Rewrite a file in place to create the simd variant
loopvec -w ./mypkg/ops.go

# 2. Manually split the file:
#    ops.go          → add //go:build !goexperiment.simd  (keep original)
#    ops_simd.go     → the rewritten file (already has //go:build goexperiment.simd)

# 3. Build and run benchmarks to verify the improvement
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

**After `loopvec -w ops.go`:**

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

## Testing

The test suite uses [txtar](https://pkg.go.dev/golang.org/x/tools/txtar) archives
in `testdata/`. Each archive defines the input, expected rewrite, and an optional
test program that is compiled and run both with and without `GOEXPERIMENT=simd` to
verify behavioral equivalence.

```sh
GOTOOLCHAIN=go1.27.1 go test -count=1 ./...
```
