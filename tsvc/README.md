# tsvc

`tsvc` ports the [TSVC_2](https://github.com/UoB-HPC/TSVC_2) vectorizer suite from C to Go.
It exists primarily to test [loopvec](..), it gives a coverage score and a
correctness gate (scalar vs SIMD build) and a benchmark set that is standardized.

The loop shapes are kept exactly as TSVC_2, and so are its names: a kernel is
named after its TSVC_2 function, its arguments after the arrays and scalars it
uses (`a`, `b`, `aa`, `x`, `s1`), and its comment after the TSVC_2 section and
description. The adaptations for Go are a `range` loop where TSVC_2 counts to
`LEN_1D`, a repetition count (`Reps`) in place of the `iterations` timing loop
and its `dummy` call, and a checksum accumulated in float64.

## Kernel inventory

Upstream `tsvc.c` has 151 kernels. This package has 93: every kernel a vectorizer
can plausibly vectorize, and `s114`, which is declined on purpose. Each has an
expectation:

- `vectorize`: a vectorizer can do it, so leaving it alone is a gap in loopvec.
- `decline`: a shape loopvec avoids on purpose, or a dependence no
  transformation removes. Vectorizing it fails the test.
- `skip`: something else does it better, such as `copy`.

### How an expectation is decided

A kernel is `vectorize` when a mainstream compiler vectorizes its loop, or
TSVC's own comment says it can be vectorized, and it needs no gather, scatter,
call, early exit, or `sinf`/`cosf`. The compiler evidence comes from compiling
upstream's `tsvc.c` with GCC 16.2 and clang at `-O3` and upstream's own flags
(`-fstrict-aliasing -fivopts -ftree-vectorize`, plus `-fno-inline` so a call stays
a call), for GCC at the SSE2 baseline and at AVX2, and again with `-ffast-math`,
which float reductions need and which is the trade-off loopvec plans behind
`-fp-reassoc`. GCC vectorizes 71 of the 151 kernels and clang 64; 83 by at least
one; `-ffast-math` adds 9 more that loopvec plans to handle. TSVC's comments add
five more (`s1113`, `s211`, `s241`, `s2102`, `s317`) and explicitly rule out
five (`s114`, `s123`, `s341`, `s342`, `s343`).

The rule is deliberately about evidence, not about what loopvec could do one day.
Anything else is `decline`, even where a smarter tool could vectorize it: change a
kernel's `expect` when a milestone covers it, and `TestCoverage` makes that change
deliberate. So `decline` here means "not planned", not "unsafe"; the recurrences
are declined for the second reason, and several if-convertible loops only because
no compiler does them.

Where a kernel differs from upstream it says so in its comment. The values that
matter for reading a checksum:

- TSVC_2's `init` gives every array a baseline value, and each kernel's
  initialisation overrides some of them; a kernel with no initialisation of its own
  (`setupBaseline`) runs on the baseline, where upstream runs it on whatever the
  previous kernel left behind.
- A kernel that returns a value, such as a reduction, is checked by that value.
  Upstream checks the unchanged array `a` for `s311`, which never looks at the sum.
- The integers TSVC_2's main passes (`n1`, `n3`) are 1, and so are the scalars
  read back through a void pointer, where upstream reads a float as an int.

| Kernel | Section | What it tests | Exactness | Expect | Vectorized |
| --- | --- | --- | --- | --- | --- |
| s000 | linear dependence testing | no dependence - vectorizable | bits | vectorize | yes |
| s111 | linear dependence testing | no dependence - vectorizable | bits | vectorize | no |
| s1111 | no dependence - vectorizable | jump in data access | fused | vectorize | no |
| s112 | linear dependence testing | loop reversal | bits | vectorize | no |
| s1112 | linear dependence testing | loop reversal | bits | vectorize | yes |
| s113 | linear dependence testing | a(i)=a(1) but no actual dependence cycle | bits | vectorize | no |
| s1113 | linear dependence testing | one iteration dependency on a(LEN_1D/2) but still vectorizable | bits | vectorize | no |
| s114 | linear dependence testing | transpose vectorization; Jump in data access - not vectorizable | bits | decline | no |
| s115 | linear dependence testing | triangular saxpy loop | fused | vectorize | no |
| s1115 | linear dependence testing | triangular saxpy loop | fused | vectorize | no |
| s118 | linear dependence testing | potential dot product recursion | fused | vectorize | no |
| s119 | linear dependence testing | no dependence - vectorizable | bits | vectorize | no |
| s1119 | linear dependence testing | no dependence - vectorizable | bits | vectorize | no |
| s121 | induction variable recognition | loop with possible ambiguity because of scalar store | bits | vectorize | no |
| s122 | induction variable recognition | variable lower and upper bound, and stride; reverse data access and jump in data access | bits | vectorize | no |
| s124 | induction variable recognition | induction variable under both sides of if (same value) | fused | vectorize | no |
| s125 | induction variable recognition | induction variable in two loops; collapsing possible | fused | vectorize | no |
| s127 | induction variable recognition | induction variable with multiple increments | fused | vectorize | no |
| s128 | induction variables | coupled induction variables; jump in data access | bits | vectorize | no |
| s131 | global data flow analysis | forward substitution | bits | vectorize | no |
| s132 | global data flow analysis | loop with multiple dimension ambiguous subscripts | fused | vectorize | no |
| s141 | nonlinear dependence testing | walk a row in a symmetric packed array; element a(i,j) for (int j>i) stored in location j*(j-1)/2+i | bits | vectorize | no |
| s162 | control flow | deriving assertions | fused | vectorize | no |
| s171 | symbolics | symbolic dependence tests | bits | vectorize | no |
| s172 | symbolics | vectorizable if n3 .ne. 0 | bits | vectorize | no |
| s173 | symbolics | expression in loop bounds and subscripts | bits | vectorize | no |
| s174 | symbolics | loop with subscript that may seem ambiguous | bits | vectorize | no |
| s175 | symbolics | symbolic dependence tests | bits | vectorize | no |
| s176 | symbolics | convolution | fused | vectorize | no |
| s211 | statement reordering | statement reordering allows vectorization | fused | vectorize | no |
| s1221 | run-time symbolic resolution |  | bits | vectorize | no |
| s222 | loop distribution | partial loop vectorizatio recurrence in middle | fused | vectorize | no |
| s231 | loop interchange | loop with data dependency | bits | vectorize | no |
| s2233 | loop interchange | interchanging with one of two inner loops | bits | vectorize | no |
| s235 | loop interchanging | imperfectly nested loops | fused | vectorize | no |
| s241 | node splitting | preloading necessary to allow vectorization | fused | vectorize | no |
| s243 | node splitting | false dependence cycle breaking | fused | vectorize | no |
| s2244 | node splitting | cycle with ture and anti dependency | bits | vectorize | no |
| s251 | scalar and array expansion | scalar expansion | fused | vectorize | no |
| s1251 | scalar and array expansion | scalar expansion | fused | vectorize | no |
| s3251 | scalar and array expansion | scalar expansion | fused | vectorize | no |
| s252 | scalar and array expansion | loop with ambiguous scalar temporary | bits | vectorize | no |
| s254 | scalar and array expansion | carry around variable | bits | vectorize | no |
| s255 | scalar and array expansion | carry around variables, 2 levels | bits | vectorize | no |
| s257 | scalar and array expansion | array expansion | bits | vectorize | no |
| s271 | control flow | loop with singularity handling | fused | vectorize | no |
| s273 | control flow | simple loop with dependent conditional | fused | vectorize | no |
| s2275 | loop distribution is needed to be able to interchange |  | fused | vectorize | no |
| s276 | control flow | if test using loop index | fused | vectorize | no |
| s1279 | control flow | vector if/gotos | fused | vectorize | no |
| s2711 | control flow | semantic if removal | fused | vectorize | no |
| s2712 | control flow | if to elemental min | fused | vectorize | no |
| s1281 | crossing thresholds | index set splitting; reverse data access | fused | vectorize | no |
| s291 | loop peeling | wrap around variable, 1 level | bits | vectorize | no |
| s293 | loop peeling | a(i)=a(0) with actual dependence cycle, loop is vectorizable | bits | vectorize | no |
| s2101 | diagonals | main diagonal calculation; jump in data access | fused | vectorize | no |
| s2102 | diagonals | identity matrix, best results vectorize both inner and outer loops | bits | vectorize | no |
| s311 | reductions | sum reduction | fused | vectorize | no |
| s312 | reductions | product reduction | fused | vectorize | no |
| s313 | reductions | dot product | fused | vectorize | no |
| s314 | reductions | if to max reduction | bits | vectorize | no |
| s315 | reductions | if to max with index reductio 1 dimension | bits | vectorize | no |
| s316 | reductions | if to min reduction | bits | vectorize | no |
| s317 | reductions | product reductio vectorize with; 1. scalar expansion of factor, and product reduction; 2. closed form solution: q = factor**n | fused | vectorize | no |
| s319 | reductions | coupled reductions | fused | vectorize | no |
| s3111 | reductions | conditional sum reduction | fused | vectorize | no |
| s3113 | reductions | maximum of absolute value | bits | vectorize | no |
| s331 | search loops | if to last-1 | bits | vectorize | no |
| s351 | loop rerolling | unrolled saxpy | fused | vectorize | no |
| s1351 | induction pointer recognition |  | bits | vectorize | no |
| s352 | loop rerolling | unrolled dot product | fused | vectorize | no |
| s421 | storage classes and equivalencing | equivalence- no overlap | bits | vectorize | no |
| s1421 | storage classes and equivalencing | equivalence- no overlap | bits | vectorize | no |
| s422 | storage classes and equivalencing | common and equivalence statement; anti-dependence, threshold of 4 | bits | vectorize | no |
| s423 | storage classes and equivalencing | common and equivalenced variables - with anti-dependence; do this again here | bits | vectorize | no |
| s424 | storage classes and equivalencing | common and equivalenced variables - overlap; vectorizeable in strips of 64 or less; do this again here | bits | vectorize | no |
| s431 | parameters | parameter statement | bits | vectorize | no |
| s441 | non-logical if's | arithmetic if | fused | vectorize | no |
| s443 | non-logical if's | arithmetic if | fused | vectorize | no |
| s452 | intrinsic functions | seq function | fused | vectorize | no |
| s453 | induction varibale recognition |  | fused | vectorize | no |
| s4117 | indirect addressing | seq function | fused | vectorize | no |
| va | control loops | vector assignment | bits | vectorize | yes |
| vif | control loops | vector if | bits | vectorize | no |
| vpv | control loops | vector plus vector | bits | vectorize | yes |
| vtv | control loops | vector times vector | bits | vectorize | yes |
| vpvtv | control loops | vector plus vector times vector | fused | vectorize | yes |
| vpvts | control loops | vector plus vector times scalar | fused | vectorize | yes |
| vpvpv | control loops | vector plus vector plus vector | bits | vectorize | yes |
| vtvtv | control loops | vector times vector times vector | bits | vectorize | yes |
| vsumr | control loops | vector sum reduction | fused | vectorize | no |
| vdotr | control loops | vector dot product reduction | fused | vectorize | no |
| vbor | control loops | basic operations rates, isolate arithmetic from memory traffic; all combinations of three, 59 flops for 6 loads and 1 store. | fused | vectorize | no |

### Not ported

The 58 kernels the rule declines are not ported yet:

- no compiler vectorizes it and TSVC does not claim it (31): `s116` `s126` `s1161` `s212` `s1213` `s221` `s232` `s1232` `s233` `s242` `s244` `s1244` `s2251` `s253` `s256` `s258` `s261` `s272` `s274` `s275` `s2710` `s281` `s292` `s2111` `s31111` `s3110` `s13110` `s3112` `s321` `s322` `s323`
- gather (6): `s353` `s4112` `s4114` `s4115` `s4116` `vag`
- goto (5): `s161` `s277` `s278` `s279` `s318`
- not vectorizable per TSVC (4): `s123` `s341` `s342` `s343`
- call (3): `s151` `s471` `s4121`
- early exit (3): `s332` `s481` `s482`
- scatter (2): `s491` `vas`
- call per element (1): `s152`
- computed goto (1): `s442`
- sinf and cosf (1): `s451`
- gather and scatter (1): `s4113`

Porting them would add tests that loopvec leaves them alone. The first group, where
no compiler vectorizes the loop and TSVC does not claim it, is the one to revisit as
the roadmap lands (if-conversion, loop distribution, interchange): port a kernel
with `expect=vectorize` when a milestone covers it.

To see what loopvec does to the kernels:

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
# loopvec TSVC coverage: vectorized 9, gaps 83, declined 1, skipped 0 (of 93)
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

Baseline on AMD Ryzen 9 9950X3D (AVX-512), scalar vs. SIMD, for the first 19 kernels (the
rest were ported after and run unchanged scalar code in both builds):

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

The kernels loopvec rewrites (s000, s1112, va, vpv, vtv, vpvtv, vpvts, vpvpv,
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
