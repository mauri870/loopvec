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

## Benchmarks

```sh
make tsvc-bench
```

Baseline on AMD Ryzen 9 9950X3D (AVX-512), scalar vs. SIMD:

```
                 │ /tmp/tsvc_bench_scalar.txt │        /tmp/tsvc_bench_simd.txt        │
                 │           sec/op           │    sec/op      vs base                 │
Kernels/s000-32                   6.646µ ± 3%    1.259µ ±  2%   -81.05% (p=0.000 n=10)
Kernels/s111-32                   5.823µ ± 2%    8.490µ ± 38%   +45.81% (p=0.000 n=10)
Kernels/s1111-32                  8.818µ ± 4%   15.124µ ± 60%   +71.51% (p=0.000 n=10)
Kernels/s112-32                   6.227µ ± 4%   13.953µ ± 36%  +124.08% (p=0.000 n=10)
Kernels/s1112-32                  6.680µ ± 5%   12.678µ ± 26%   +89.80% (p=0.000 n=10)
Kernels/s113-32                   6.893µ ± 1%   12.649µ ± 39%   +83.51% (p=0.000 n=10)
Kernels/s1113-32                  6.480µ ± 5%    9.831µ ± 24%   +51.71% (p=0.000 n=10)
Kernels/s114-32                   20.15µ ± 1%    23.38µ ± 10%   +16.02% (p=0.000 n=10)
Kernels/s115-32                   12.18µ ± 2%    14.01µ ± 36%   +15.00% (p=0.000 n=10)
Kernels/s1115-32                  49.30µ ± 2%    62.05µ ± 16%   +25.88% (p=0.000 n=10)
geomean                           9.738µ         12.19µ         +25.15%
```

s000 is the only kernel loopvec rewrites, and it's the only one with a real
win. The rest are unchanged scalar source, yet they're both slower and far
noisier here (up to ±60%) than the same code compiled without `-toolexec`
(±1-5%) — that's not measurement noise. Isolating each factor: `GOEXPERIMENT=simd`
alone, with nothing ever rewritten, costs almost nothing; the slowdown and
jitter appear specifically once something in the binary is actually rewritten
into SIMD (`s000` here), and hit every kernel in the process, not just the
one touched. The likely cause is Go's runtime: once any code in a binary
uses the wider vector registers, goroutine preemption's signal-based
register save/restore has to account for that wider state everywhere, not
just in the function using it, raising cost and variance program-wide. This
is a real cost of the SIMD experiment as it stands today, not an artifact of
this benchmark.

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
