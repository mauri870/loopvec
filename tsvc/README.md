# tsvc

`tsvc` ports the [TSVC_2](https://github.com/UoB-HPC/TSVC_2) vectorizer suite from C to Go.
It exists primarily to test [loopvec](..), it gives a coverage score and a
correctness gate (normal vs SIMD build) and a benchmark set that is standardized.

The loop shapes are kept exactly as TSVC_2.

## Kernel inventory

| Kernel | What it tests | Exactness |
| --- | --- | --- |
| s000 | no dependence | bits |
| s111 | no dependence, stride 2 | bits |
| s1111 | no dependence, jump in data access (strided store) | fused |
| s112 | loop reversal, anti-dependence | bits |
| s1112 | loop reversal, no dependence | bits |
| s113 | `a[0]` looks loop-carried but is invariant | bits |
| s1113 | one-iteration dependency on `a[len(a)/2]`, still vectorizable | bits |
| s114 | 2-D triangular transpose (column) access | bits |
| s115 | 2-D triangular saxpy loop | fused |
| s1115 | 2-D triangular saxpy loop with a transposed read | fused |

Today, loopvec rewrites 1 of the 10, s000:

```sh
go run . -d ./tsvc/
```

## Correctness

Each kernel has a deterministic `Setup`, runs `Reps` times, and its checksum
is checked against a golden produced by the scalar build:

```sh
make tsvc-test
```

This also checks the SIMD build against the same golden, via
`GOEXPERIMENT=simd` and `-toolexec=loopvec-toolexec`.

Goldens live in `testdata/golden_<GOARCH>.json`, one per architecture: Go's
compiler contracts a multiply feeding an add into a single FMA instruction
on some architectures (arm64) but not others (baseline amd64), so a "fused"
kernel's checksum can differ slightly between them. Check the arm64 golden
without arm64 hardware, via `qemu-aarch64-static`:

```sh
make tsvc-test-qemu-arm64
```

After a deliberate change to a kernel or its setup, regenerate both goldens:

```sh
make tsvc-update
```

## Generated code

`Kernels`, the table driving the tests, is generated from the
`//tsvc:kernel` directive on each kernel in `kernels.go`:

```sh
go generate ./tsvc/...
```

This regenerates `zz_registry.go` (from `internal/tsvcgen`) and
`category_string.go` (from `stringer`). Never edit either by hand.

## License

The kernels, their setup, and their checksums are ports of TSVC_2 source
under the University of Illinois/NCSA license; see [LICENSE-TSVC](LICENSE-TSVC).
