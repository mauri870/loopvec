# tsvc

`tsvc` ports the [TSVC_2](https://github.com/UoB-HPC/TSVC_2) vectorizer suite from C to Go.
It exists primarily to test [loopvec](..), it gives a coverage score and a
correctness gate (scalar vs SIMD build) and a benchmark set that is standardized.

The loop shapes are kept exactly as TSVC_2.

## Kernel inventory

| Kernel | What it tests | Exactness | Vectorized |
| --- | --- | --- | --- |
| s000 | no dependence | bits | yes |
| s111 | no dependence, stride 2 | bits | no |
| s1111 | no dependence, jump in data access (strided store) | fused | no |
| s112 | loop reversal, anti-dependence | bits | no |
| s1112 | loop reversal, no dependence | bits | yes |
| s113 | `a[0]` looks loop-carried but is invariant | bits | no |
| s1113 | one-iteration dependency on `a[len(a)/2]`, still vectorizable | bits | no |
| s114 | 2-D triangular transpose (column) access | bits | no |
| s115 | 2-D triangular saxpy loop | fused | no |
| s1115 | 2-D triangular saxpy loop with a transposed read | fused | no |

Today, loopvec rewrites 2 of the 10, s000 and s1112:

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

Goldens live in `testdata/golden_<GOARCH>.json`, one per architecture.

Check the arm64 golden without arm64 hardware, via `qemu-aarch64-static`:

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
                 │           sec/op           │     sec/op      vs base                │
Kernels/s000-32                   7.532µ ± 4%    1.343µ ±   5%  -82.18% (p=0.000 n=10)
Kernels/s111-32                   6.012µ ± 4%    6.582µ ±  20%        ~ (p=0.393 n=10)
Kernels/s1111-32                  9.025µ ± 3%   13.886µ ±  36%  +53.88% (p=0.000 n=10)
Kernels/s112-32                   7.002µ ± 9%    7.827µ ± 244%  +11.78% (p=0.009 n=10)
Kernels/s1112-32                  7.203µ ± 4%    1.272µ ±  28%  -82.35% (p=0.000 n=10)
Kernels/s113-32                   7.178µ ± 7%   11.520µ ±  57%  +60.48% (p=0.000 n=10)
Kernels/s1113-32                  6.708µ ± 6%   12.811µ ±  30%  +90.97% (p=0.000 n=10)
Kernels/s114-32                   20.84µ ± 7%    24.07µ ±  17%  +15.53% (p=0.000 n=10)
Kernels/s115-32                   12.31µ ± 5%    17.20µ ±  56%  +39.69% (p=0.000 n=10)
Kernels/s1115-32                  50.30µ ± 2%    64.19µ ±  27%  +27.62% (p=0.023 n=10)
geomean                           10.25µ         9.290µ          -9.37%
```

s000 and s1112 are the two kernels loopvec rewrites, and they're the only
ones with a real win (both around -82%). The rest are unchanged scalar
source, yet they're both slower and far noisier here (up to ±244%) than the
same code built without `GOEXPERIMENT=simd` (±1-9%). This seems to be an
effect of the runtime saving the full register file for every preemption of
every goroutine in the process, regardless of whether anything calls into
SIMD.

`GODEBUG=asyncpreemptoff=1` recovers most of the regression, which heavily indicates
that async preemption might be the culprint.

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
