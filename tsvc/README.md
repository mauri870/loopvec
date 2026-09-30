# tsvc

`tsvc` ports the [TSVC_2](https://github.com/UoB-HPC/TSVC_2) vectorizer suite from C to Go.
It exists primarily to test [loopvec](..), it gives a coverage score and a
correctness gate (scalar vs SIMD build) and a benchmark set that is standardized.

The loop shapes are kept exactly as TSVC_2.

## Kernel inventory

Each kernel has an expectation: `vectorize` (a vectorizer can do it, so leaving
it alone is a gap in loopvec), `decline` (it has a loop-carried dependence, or
needs a shape loopvec avoids on purpose, such as a gather), or `skip` (something
else does it better, such as `copy`).

| Kernel | What it tests | Exactness | Expect | Vectorized |
| --- | --- | --- | --- | --- |
| s000 | no dependence | bits | vectorize | yes |
| s111 | no dependence, stride 2 | bits | vectorize | no |
| s1111 | no dependence, jump in data access (strided store) | fused | vectorize | no |
| s112 | loop reversal, anti-dependence | bits | vectorize | no |
| s1112 | loop reversal, no dependence | bits | vectorize | yes |
| s113 | `a[0]` looks loop-carried but is invariant | bits | vectorize | no |
| s1113 | one-iteration dependency on `a[len(a)/2]`, still vectorizable | bits | vectorize | no |
| s114 | 2-D triangular transpose (column) access, a gather | bits | decline | no |
| s115 | 2-D triangular saxpy loop | fused | vectorize | no |
| s1115 | 2-D triangular saxpy loop with a transposed read, a gather | fused | decline | no |
| va | vector assignment (a copy) | bits | skip | no |
| vif | vector if, a conditional store | bits | vectorize | no |
| vpv | vector plus vector | bits | vectorize | yes |
| vtv | vector times vector | bits | vectorize | yes |
| vpvtv | vector plus vector times vector | fused | vectorize | yes |
| vpvts | vector plus vector times scalar | fused | vectorize | yes |
| vpvpv | vector plus vector plus vector | bits | vectorize | yes |
| vtvtv | vector times vector times vector | bits | vectorize | yes |
| vbor | basic operations rate, a multi-statement body | fused | vectorize | no |

Not ported yet: the gather and scatter kernels `vag` and `vas`, and the
reductions `vsumr` and `vdotr`, which return a scalar instead of filling an
array. `vpvts` differs from TSVC_2 in one way: the original passes its scalar
through a void pointer and reads it back as an int, which is 1065353216; the
port uses the intended 1.

Today loopvec rewrites 8 of the 19 kernels. To see what it does to them:

```sh
go run . -d ./tsvc/
```

## Coverage

A bare count would mislead, because some kernels should not be vectorized: a
loop with a real dependence between iterations is correct to leave alone.
`testdata/coverage.txt` records one line per kernel, built from `loopvec -json`
(see the root README's "Coverage reporting" section for the JSON format), with
the expectation copied from the kernel's directive so the file reads on its own:

```
# loopvec TSVC coverage: vectorized 8, gaps 8, declined 2, skipped 1 (of 19)
s000	vectorize	yes
s111	vectorize	no	loop start is not 0 (or len(s)-1 when counting down)
...
```

A gap is a kernel expected to be vectorized that is not; declined and skipped
kernels left alone are correct.

`TestCoverage` (part of `make tsvc-test` / `make test`) rebuilds `loopvec`
fresh, runs `-json` against this package, and compares the result against
`testdata/coverage.txt`. A kernel that stops vectorizing is a build failure (a
regression). A kernel that starts vectorizing is *also* a build failure, with a
message pointing at the command below, so an improvement gets recorded on
purpose rather than silently drifting:

```sh
make tsvc-coverage-update
```

A kernel expected to be declined or skipped that becomes vectorized fails the
test even under that command: it is a bug, not an improvement. If the new
behavior is intended, change the kernel's `expect` in `kernels.go` first.

Kept separate from `make tsvc-update` (which only regenerates the golden
checksums) on purpose: a coverage change should be reviewed and committed
deliberately, not folded into routine golden regeneration.

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
                 │ /tmp/tsvc_bench_scalar.txt │        /tmp/tsvc_bench_simd.txt         │
                 │           sec/op           │     sec/op      vs base                 │
Kernels/s000-32                  6143.0n ± 2%    927.6n ±   1%   -84.90% (p=0.000 n=10)
Kernels/s111-32                   3.612µ ± 2%    5.374µ ±  80%   +48.77% (p=0.000 n=10)
Kernels/s1111-32                  8.860µ ± 1%   18.428µ ±  33%  +107.99% (p=0.000 n=10)
Kernels/s112-32                   6.028µ ± 0%   13.042µ ±  31%  +116.35% (p=0.000 n=10)
Kernels/s1112-32                  6.061µ ± 2%    1.114µ ±   1%   -81.62% (p=0.000 n=10)
Kernels/s113-32                   6.359µ ± 1%    8.419µ ± 113%   +32.41% (p=0.000 n=10)
Kernels/s1113-32                  6.791µ ± 4%   10.080µ ±  63%   +48.44% (p=0.000 n=10)
Kernels/s114-32                   20.65µ ± 3%    24.95µ ±  11%   +20.82% (p=0.000 n=10)
Kernels/s115-32                   13.22µ ± 5%    18.91µ ±  48%   +43.03% (p=0.000 n=10)
Kernels/s1115-32                  43.82µ ± 1%    63.44µ ±  22%   +44.78% (p=0.000 n=10)
Kernels/va-32                     6.016µ ± 3%    7.865µ ±  27%   +30.73% (p=0.000 n=10)
Kernels/vif-32                    9.168µ ± 1%    9.587µ ±   6%    +4.58% (p=0.003 n=10)
Kernels/vpv-32                    6.028µ ± 1%    1.315µ ±   1%   -78.18% (p=0.000 n=10)
Kernels/vtv-32                    5.992µ ± 1%    1.304µ ±   1%   -78.23% (p=0.000 n=10)
Kernels/vpvtv-32                  9.859µ ± 1%    1.627µ ±   1%   -83.50% (p=0.000 n=10)
Kernels/vpvts-32                  7.116µ ± 1%    1.255µ ±   0%   -82.36% (p=0.000 n=10)
Kernels/vpvpv-32                  9.859µ ± 0%    1.928µ ±   2%   -80.44% (p=0.000 n=10)
Kernels/vtvtv-32                  9.084µ ± 0%    1.846µ ±   0%   -79.68% (p=0.000 n=10)
Kernels/vbor-32                   763.1n ± 0%    761.7n ±   1%         ~ (p=0.393 n=10)
geomean                           7.430µ         4.486µ          -39.63%

                 │ /tmp/tsvc_bench_scalar.txt │        /tmp/tsvc_bench_simd.txt         │
                 │            B/s             │      B/s        vs base                 │
Kernels/s000-32                  38.81Gi ± 2%   257.04Gi ±  1%  +562.28% (p=0.000 n=10)
Kernels/s111-32                  66.00Gi ± 2%    44.37Gi ± 44%   -32.77% (p=0.000 n=10)
Kernels/s1111-32                 53.82Gi ± 1%    25.88Gi ± 39%   -51.92% (p=0.000 n=10)
Kernels/s112-32                  39.55Gi ± 0%    18.33Gi ± 45%   -53.66% (p=0.000 n=10)
Kernels/s1112-32                 39.33Gi ± 2%   214.03Gi ±  1%  +444.13% (p=0.000 n=10)
Kernels/s113-32                  37.50Gi ± 1%    28.35Gi ± 53%   -24.41% (p=0.000 n=10)
Kernels/s1113-32                 35.11Gi ± 3%    23.69Gi ± 39%   -32.54% (p=0.000 n=10)
Kernels/s114-32                  23.65Gi ± 2%    19.59Gi ± 12%   -17.14% (p=0.000 n=10)
Kernels/s115-32                  27.48Gi ± 5%    19.27Gi ± 33%   -29.88% (p=0.000 n=10)
Kernels/s1115-32                 16.72Gi ± 1%    11.55Gi ± 29%   -30.93% (p=0.000 n=10)
Kernels/va-32                    39.63Gi ± 3%    30.34Gi ± 21%   -23.44% (p=0.000 n=10)
Kernels/vif-32                   26.01Gi ± 1%    24.87Gi ±  5%    -4.37% (p=0.003 n=10)
Kernels/vpv-32                   39.56Gi ± 1%   181.30Gi ±  1%  +358.35% (p=0.000 n=10)
Kernels/vtv-32                   39.79Gi ± 1%   182.72Gi ±  1%  +359.19% (p=0.000 n=10)
Kernels/vpvtv-32                 36.28Gi ± 1%   219.79Gi ±  1%  +505.89% (p=0.000 n=10)
Kernels/vpvts-32                 33.51Gi ± 1%   189.98Gi ±  0%  +467.02% (p=0.000 n=10)
Kernels/vpvpv-32                 36.27Gi ± 0%   185.47Gi ±  2%  +411.31% (p=0.000 n=10)
Kernels/vtvtv-32                 39.37Gi ± 0%   193.74Gi ±  0%  +392.12% (p=0.000 n=10)
Kernels/vbor-32                  1.228Ti ± 0%    1.230Ti ±  1%         ~ (p=0.393 n=10)
geomean                          43.00Gi         71.26Gi         +65.73%
```

The eight kernels loopvec rewrites (s000, s1112, vpv, vtv, vpvtv, vpvts, vpvpv,
vtvtv) are 78-85% faster. The rest are unchanged scalar source, yet they are
mostly slower and far noisier here (up to ±113%) than the same code built
without `GOEXPERIMENT=simd`. This seems to be an effect of the runtime saving
the full register file for every preemption of every goroutine in the process,
regardless of whether anything calls into SIMD. `vbor` touches only 256 of the
32000 elements of each array, so its bytes-per-second figure is not meaningful.

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
