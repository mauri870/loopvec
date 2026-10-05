# tsvc

`tsvc` ports the [TSVC_2](https://github.com/UoB-HPC/TSVC_2) vectorizer suite from C to Go.
It exists primarily to test [loopvec](..), it gives a coverage score and a
correctness gate (scalar vs SIMD build) and a benchmark set that is standardized.

Names follow TSVC_2: the kernels, their arguments (`a`, `aa`, `x`, `s1`), and their
comments. The Go adaptations are `range` loops, a repetition count (`Reps`) instead
of the timing loop, and a float64 checksum.

## Kernel inventory

Upstream `tsvc.c` has 151 kernels. 93 are ported: those a vectorizer can plausibly
vectorize, plus `s114`. Each has an expectation:

- `vectorize`: a vectorizer can do it, so leaving it alone is a gap in loopvec.
- `decline`: a dependence, or a shape loopvec avoids. Vectorizing it fails the test.
- `skip`: something else does it better, such as `copy`.

A kernel is `vectorize` if GCC 16.2 or clang vectorizes its loop at `-O3` with
upstream's flags (GCC also at AVX2, and both with `-ffast-math` for float
reductions), or TSVC's own comment says it can be, and it needs no gather, scatter,
call, early exit or `sinf`/`cosf`. GCC vectorizes 71 of the 151, clang 64, either 83.
Everything else is `decline`, which means "not planned", not "unsafe".

Each kernel's comment notes where it differs from upstream. Two differences apply
widely: a kernel with no `initialise_arrays` branch (`setupBaseline`) runs on the
values TSVC_2's `init` sets, where upstream runs it on the previous kernel's
leftovers; and a kernel that returns a value, such as a reduction, is checked by that
value. Exactness is `bits` (hash and sum identical), `fused` (the relative error a
fused multiply-add introduces), or `reassoc` (the error of regrouping a sum, bounded
by `n*2^-24`, for the kernels that vectorize only with `-fp-reassoc`).

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
| s241 | node splitting | preloading necessary to allow vectorization | fused | vectorize | yes |
| s243 | node splitting | false dependence cycle breaking | fused | vectorize | yes |
| s2244 | node splitting | cycle with ture and anti dependency | bits | vectorize | no |
| s251 | scalar and array expansion | scalar expansion | fused | vectorize | yes |
| s1251 | scalar and array expansion | scalar expansion | fused | vectorize | yes |
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
| s1281 | crossing thresholds | index set splitting; reverse data access | fused | vectorize | yes |
| s291 | loop peeling | wrap around variable, 1 level | bits | vectorize | no |
| s293 | loop peeling | a(i)=a(0) with actual dependence cycle, loop is vectorizable | bits | vectorize | no |
| s2101 | diagonals | main diagonal calculation; jump in data access | fused | vectorize | no |
| s2102 | diagonals | identity matrix, best results vectorize both inner and outer loops | bits | vectorize | no |
| s311 | reductions | sum reduction | reassoc | vectorize | yes |
| s312 | reductions | product reduction | reassoc | vectorize | yes |
| s313 | reductions | dot product | reassoc | vectorize | yes |
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
| s421 | storage classes and equivalencing | equivalence- no overlap | bits | vectorize | yes |
| s1421 | storage classes and equivalencing | equivalence- no overlap | bits | vectorize | yes |
| s422 | storage classes and equivalencing | common and equivalence statement; anti-dependence, threshold of 4 | bits | vectorize | yes |
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
| vsumr | control loops | vector sum reduction | reassoc | vectorize | yes |
| vdotr | control loops | vector dot product reduction | reassoc | vectorize | yes |
| vbor | control loops | basic operations rates, isolate arithmetic from memory traffic; all combinations of three, 59 flops for 6 loads and 1 store. | fused | vectorize | no |

### Not ported

The other 58 kernels are not ported:

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

Porting them would only test that loopvec leaves them alone.

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
# loopvec TSVC coverage: vectorized 22, gaps 70, declined 1, skipped 0 (of 93)
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
goos: linux
goarch: amd64
pkg: github.com/mauri870/loopvec/tsvc
cpu: AMD Ryzen 9 9950X3D 16-Core Processor          
                 │ /tmp/tsvc_bench_scalar.txt │        /tmp/tsvc_bench_simd.txt        │
                 │           sec/op           │     sec/op      vs base                │
Kernels/s000-32                 6.759µ ±  12%    1.261µ ±  10%   -81.35% (p=0.002 n=6)
Kernels/s111-32                 5.786µ ±   2%    8.491µ ±  27%   +46.76% (p=0.002 n=6)
Kernels/s1111-32                8.629µ ±   2%    9.378µ ± 122%    +8.68% (p=0.009 n=6)
Kernels/s112-32                 5.934µ ±   2%   11.983µ ±  46%  +101.96% (p=0.002 n=6)
Kernels/s1112-32                6.330µ ±   4%    1.288µ ±   8%   -79.65% (p=0.002 n=6)
Kernels/s113-32                 6.984µ ±   4%   11.077µ ±  36%   +58.62% (p=0.004 n=6)
Kernels/s1113-32                6.124µ ±   9%    7.770µ ±  68%   +26.88% (p=0.002 n=6)
Kernels/s114-32                 19.59µ ±   2%    21.90µ ±  12%   +11.80% (p=0.041 n=6)
Kernels/s115-32                 11.84µ ±   5%    16.53µ ±  29%   +39.65% (p=0.009 n=6)
Kernels/s1115-32                48.95µ ±   2%    57.23µ ±  31%   +16.91% (p=0.002 n=6)
Kernels/s118-32                 69.75µ ±   2%    73.34µ ±   5%    +5.15% (p=0.026 n=6)
Kernels/s119-32                 31.66µ ±   9%    38.40µ ±  17%   +21.28% (p=0.009 n=6)
Kernels/s1119-32                32.22µ ±   2%    41.61µ ±  55%   +29.14% (p=0.002 n=6)
Kernels/s121-32                 6.217µ ±   6%    7.753µ ±  33%   +24.71% (p=0.026 n=6)
Kernels/s122-32                 13.38µ ±  28%    18.31µ ±  32%   +36.84% (p=0.009 n=6)
Kernels/s124-32                 15.52µ ±   6%    21.43µ ±  27%   +38.03% (p=0.004 n=6)
Kernels/s125-32                 49.49µ ±  18%    50.83µ ±   5%         ~ (p=0.240 n=6)
Kernels/s127-32                 10.22µ ±   1%    16.21µ ±  37%   +58.60% (p=0.015 n=6)
Kernels/s128-32                 5.352µ ±   7%    6.165µ ±  57%         ~ (p=0.065 n=6)
Kernels/s131-32                 6.032µ ±   2%   11.885µ ±  47%   +97.04% (p=0.002 n=6)
Kernels/s132-32                 122.2n ±  14%    124.9n ±  37%         ~ (p=0.310 n=6)
Kernels/s141-32                 18.92µ ±   5%    20.85µ ±  16%         ~ (p=0.093 n=6)
Kernels/s162-32                 9.239µ ±   7%   11.196µ ±  62%         ~ (p=0.240 n=6)
Kernels/s171-32                 11.62µ ±   3%    14.12µ ±  37%   +21.55% (p=0.002 n=6)
Kernels/s172-32                 9.587µ ±   3%   16.838µ ±  30%   +75.63% (p=0.002 n=6)
Kernels/s173-32                 5.814µ ±   3%    7.530µ ±  25%   +29.53% (p=0.002 n=6)
Kernels/s174-32                 3.518µ ±   4%    6.402µ ±  52%   +81.99% (p=0.002 n=6)
Kernels/s175-32                 7.149µ ±  57%   13.724µ ±  47%   +91.98% (p=0.041 n=6)
Kernels/s176-32                 95.00m ±   3%   149.49m ±  35%   +57.36% (p=0.004 n=6)
Kernels/s211-32                 21.01µ ±   2%    23.52µ ±  10%   +11.98% (p=0.002 n=6)
Kernels/s1221-32                17.06µ ±   2%    21.60µ ±  19%   +26.60% (p=0.002 n=6)
Kernels/s222-32                 68.01µ ±   2%    80.31µ ±  15%   +18.08% (p=0.002 n=6)
Kernels/s231-32                 357.6µ ±   4%    388.9µ ±   5%    +8.73% (p=0.002 n=6)
Kernels/s2233-32                426.6µ ±   3%    446.0µ ±  10%         ~ (p=0.065 n=6)
Kernels/s235-32                 335.8µ ±   5%    355.4µ ±   3%    +5.84% (p=0.002 n=6)
Kernels/s241-32                15.201µ ±   9%    4.342µ ±   3%   -71.44% (p=0.002 n=6)
Kernels/s243-32                24.101µ ±   5%    5.905µ ±   6%   -75.50% (p=0.002 n=6)
Kernels/s2244-32                12.54µ ±   5%    24.98µ ±  50%   +99.17% (p=0.026 n=6)
Kernels/s251-32                10.604µ ±   5%    1.995µ ±   8%   -81.18% (p=0.002 n=6)
Kernels/s1251-32               15.420µ ±   8%    3.046µ ±   2%   -80.25% (p=0.002 n=6)
Kernels/s3251-32                22.81µ ±   4%    36.49µ ±  38%   +59.94% (p=0.002 n=6)
Kernels/s252-32                 11.61µ ±   2%    18.94µ ±  11%   +63.20% (p=0.002 n=6)
Kernels/s254-32                 9.133µ ±   4%   19.312µ ±  51%  +111.46% (p=0.002 n=6)
Kernels/s255-32                 9.642µ ±   5%   16.100µ ±  94%   +66.97% (p=0.002 n=6)
Kernels/s257-32                 192.0µ ±   3%    177.3µ ±   3%    -7.66% (p=0.002 n=6)
Kernels/s271-32                 16.56µ ±  17%    19.32µ ±  30%   +16.67% (p=0.026 n=6)
Kernels/s273-32                 20.91µ ±   9%    36.75µ ±  26%   +75.72% (p=0.002 n=6)
Kernels/s2275-32                250.2µ ±   3%    243.4µ ±   1%    -2.69% (p=0.009 n=6)
Kernels/s276-32                 11.01µ ±   4%    14.90µ ±  46%   +35.34% (p=0.009 n=6)
Kernels/s1279-32                5.820µ ±   6%    6.348µ ±  36%         ~ (p=0.065 n=6)
Kernels/s2711-32                17.33µ ±   3%    19.65µ ±  25%   +13.39% (p=0.002 n=6)
Kernels/s2712-32                14.79µ ±  22%    19.22µ ±  19%   +29.97% (p=0.004 n=6)
Kernels/s1281-32               19.752µ ±   7%    3.349µ ±   5%   -83.05% (p=0.002 n=6)
Kernels/s291-32                 10.17µ ±   4%    11.19µ ±  48%         ~ (p=0.310 n=6)
Kernels/s293-32                 5.754µ ±   2%    6.436µ ±  15%   +11.85% (p=0.015 n=6)
Kernels/s2101-32                357.0n ±   2%    378.5n ±   7%         ~ (p=0.240 n=6)
Kernels/s2102-32                107.6µ ±  31%    142.2µ ±   6%   +32.21% (p=0.004 n=6)
Kernels/s311-32               11499.0n ±   2%    829.3n ±   4%   -92.79% (p=0.002 n=6)
Kernels/s312-32               17194.0n ±   2%    867.2n ±   3%   -94.96% (p=0.002 n=6)
Kernels/s313-32                11.430µ ±   2%    1.409µ ±   4%   -87.67% (p=0.002 n=6)
Kernels/s314-32                 8.144µ ±   3%   10.941µ ±  62%   +34.34% (p=0.002 n=6)
Kernels/s315-32                 13.00µ ±  11%    15.28µ ±  23%   +17.54% (p=0.009 n=6)
Kernels/s316-32                 5.916µ ±  17%    7.235µ ±  21%   +22.30% (p=0.015 n=6)
Kernels/s317-32                 12.13µ ±  18%    11.36µ ±  12%         ~ (p=0.699 n=6)
Kernels/s319-32                 22.55µ ±   2%    26.19µ ±  37%   +16.17% (p=0.002 n=6)
Kernels/s3111-32                12.33µ ±   2%    14.85µ ±  13%   +20.48% (p=0.002 n=6)
Kernels/s3113-32                13.56µ ±   2%    35.06µ ± 105%  +158.54% (p=0.015 n=6)
Kernels/s331-32                 7.249µ ±   3%    8.050µ ±  16%   +11.05% (p=0.002 n=6)
Kernels/s351-32                 6.241µ ±   4%    7.042µ ± 131%         ~ (p=0.065 n=6)
Kernels/s1351-32                29.26µ ±   4%    28.39µ ±   3%    -3.00% (p=0.041 n=6)
Kernels/s352-32                 11.56µ ±   3%    13.19µ ±  14%         ~ (p=0.240 n=6)
Kernels/s421-32                 6.659µ ±   5%    8.554µ ±  67%         ~ (p=1.000 n=6)
Kernels/s1421-32               3184.5n ±  21%    758.0n ±   5%   -76.20% (p=0.002 n=6)
Kernels/s422-32                 9.524µ ±   2%   13.973µ ±  33%   +46.71% (p=0.004 n=6)
Kernels/s423-32                 8.771µ ±   2%   13.999µ ±  37%   +59.61% (p=0.002 n=6)
Kernels/s424-32                 9.054µ ±   6%   12.213µ ±  53%   +34.90% (p=0.026 n=6)
Kernels/s431-32                 6.033µ ±   3%   14.400µ ±  50%  +138.71% (p=0.002 n=6)
Kernels/s441-32                 18.75µ ±   7%    24.54µ ±  22%   +30.86% (p=0.009 n=6)
Kernels/s443-32                 13.49µ ±  16%    25.18µ ±  33%   +86.71% (p=0.002 n=6)
Kernels/s452-32                 11.65µ ±   5%    18.70µ ±  48%   +60.55% (p=0.026 n=6)
Kernels/s453-32                 11.38µ ±   2%    16.05µ ±  27%   +41.06% (p=0.002 n=6)
Kernels/s4117-32                11.76µ ±  20%    16.78µ ±  31%   +42.70% (p=0.015 n=6)
Kernels/va-32                   6.006µ ±  16%    1.004µ ±   4%   -83.28% (p=0.002 n=6)
Kernels/vif-32                  8.820µ ±   4%    9.201µ ±   7%    +4.32% (p=0.026 n=6)
Kernels/vpv-32                  5.989µ ±   2%    1.329µ ±   5%   -77.82% (p=0.002 n=6)
Kernels/vtv-32                  6.003µ ±   4%    1.447µ ±   5%   -75.90% (p=0.002 n=6)
Kernels/vpvtv-32               10.036µ ± 450%    1.865µ ±   5%   -81.42% (p=0.002 n=6)
Kernels/vpvts-32                7.072µ ±  11%    1.540µ ±   5%   -78.22% (p=0.002 n=6)
Kernels/vpvpv-32               10.070µ ±   9%    1.719µ ±   4%   -82.93% (p=0.002 n=6)
Kernels/vtvtv-32                9.058µ ±   4%    2.200µ ±  10%   -75.72% (p=0.002 n=6)
Kernels/vsumr-32              11213.5n ±   2%    852.0n ±   4%   -92.40% (p=0.002 n=6)
Kernels/vdotr-32               11.396µ ±   2%    1.431µ ±   5%   -87.45% (p=0.002 n=6)
Kernels/vbor-32                 773.8n ±   3%    770.9n ±   3%         ~ (p=0.848 n=6)
geomean                         13.65µ           11.54µ          -15.45%

                 │ /tmp/tsvc_bench_scalar.txt │         /tmp/tsvc_bench_simd.txt          │
                 │            B/s             │       B/s         vs base                 │
Kernels/s000-32                 35.32Gi ± 14%    189.15Gi ±  11%   +435.61% (p=0.002 n=6)
Kernels/s111-32                 41.21Gi ±  2%     28.22Gi ±  37%    -31.51% (p=0.002 n=6)
Kernels/s1111-32                55.26Gi ±  2%     50.94Gi ±  55%     -7.83% (p=0.009 n=6)
Kernels/s112-32                 40.18Gi ±  2%     20.13Gi ±  33%    -49.90% (p=0.002 n=6)
Kernels/s1112-32                37.67Gi ±  4%    185.15Gi ±   8%   +391.54% (p=0.002 n=6)
Kernels/s113-32                 34.14Gi ±  4%     22.68Gi ±  48%    -33.58% (p=0.004 n=6)
Kernels/s1113-32                38.93Gi ±  8%     30.69Gi ±  40%    -21.16% (p=0.002 n=6)
Kernels/s114-32                 24.92Gi ±  2%     22.30Gi ±  13%    -10.51% (p=0.041 n=6)
Kernels/s115-32                 30.69Gi ±  5%     22.15Gi ±  34%    -27.84% (p=0.009 n=6)
Kernels/s1115-32                14.96Gi ±  2%     12.85Gi ±  24%    -14.09% (p=0.002 n=6)
Kernels/s118-32                 5.209Gi ±  2%     4.957Gi ±   5%     -4.83% (p=0.026 n=6)
Kernels/s119-32                 15.42Gi ±  8%     12.72Gi ±  21%    -17.50% (p=0.009 n=6)
Kernels/s1119-32                15.16Gi ±  2%     11.75Gi ±  36%    -22.46% (p=0.002 n=6)
Kernels/s121-32                 38.39Gi ±  6%     31.38Gi ±  26%    -18.25% (p=0.026 n=6)
Kernels/s122-32                 18.61Gi ± 32%     13.02Gi ±  24%    -30.04% (p=0.009 n=6)
Kernels/s124-32                 38.40Gi ±  6%     27.85Gi ±  36%    -27.47% (p=0.004 n=6)
Kernels/s125-32                 19.73Gi ± 15%     19.21Gi ±   5%          ~ (p=0.240 n=6)
Kernels/s127-32                 58.33Gi ±  1%     38.10Gi ±  53%    -34.69% (p=0.015 n=6)
Kernels/s128-32                 89.10Gi ±  7%     77.50Gi ±  36%          ~ (p=0.065 n=6)
Kernels/s131-32                 39.53Gi ±  2%     20.19Gi ±  88%    -48.93% (p=0.002 n=6)
Kernels/s132-32                 3.859Ti ± 12%     3.774Ti ±  27%          ~ (p=0.310 n=6)
Kernels/s141-32                 25.82Gi ±  5%     23.43Gi ±  14%          ~ (p=0.093 n=6)
Kernels/s162-32                 38.72Gi ±  7%     31.95Gi ±  38%          ~ (p=0.240 n=6)
Kernels/s171-32                 20.53Gi ±  3%     16.91Gi ±  27%    -17.63% (p=0.002 n=6)
Kernels/s172-32                 24.87Gi ±  3%     14.17Gi ±  42%    -43.03% (p=0.002 n=6)
Kernels/s173-32                 41.01Gi ±  2%     31.68Gi ±  22%    -22.75% (p=0.002 n=6)
Kernels/s174-32                 67.78Gi ±  4%     37.33Gi ±  38%    -44.92% (p=0.002 n=6)
Kernels/s175-32                 33.40Gi ± 36%     17.48Gi ±  87%    -47.66% (p=0.041 n=6)
Kernels/s176-32                 3.853Mi ±  3%     2.475Mi ±  52%    -35.77% (p=0.006 n=6)
Kernels/s211-32                 28.38Gi ±  2%     25.34Gi ±  10%    -10.69% (p=0.002 n=6)
Kernels/s1221-32                13.97Gi ±  2%     11.07Gi ±  23%    -20.79% (p=0.002 n=6)
Kernels/s222-32                 7.011Gi ±  2%     5.942Gi ±  13%    -15.25% (p=0.002 n=6)
Kernels/s231-32                 1.365Gi ±  4%     1.256Gi ±   5%     -8.03% (p=0.002 n=6)
Kernels/s2233-32                1.717Gi ±  4%     1.642Gi ±  11%          ~ (p=0.065 n=6)
Kernels/s235-32                 2.519Gi ±  5%     2.380Gi ±   3%     -5.52% (p=0.002 n=6)
Kernels/s241-32                 31.37Gi ±  8%    109.82Gi ±   3%   +250.07% (p=0.002 n=6)
Kernels/s243-32                 24.73Gi ±  5%    100.95Gi ±   6%   +308.16% (p=0.002 n=6)
Kernels/s2244-32                38.02Gi ±  5%     19.62Gi ±  93%    -48.40% (p=0.026 n=6)
Kernels/s251-32                 44.99Gi ±  5%    239.04Gi ±   9%   +431.35% (p=0.002 n=6)
Kernels/s1251-32                38.66Gi ±  7%    195.73Gi ±   2%   +406.35% (p=0.002 n=6)
Kernels/s3251-32                26.13Gi ±  4%     16.54Gi ±  47%    -36.70% (p=0.002 n=6)
Kernels/s252-32                 30.81Gi ±  2%     18.89Gi ±  13%    -38.69% (p=0.002 n=6)
Kernels/s254-32                 26.12Gi ±  5%     12.35Gi ± 101%    -52.71% (p=0.002 n=6)
Kernels/s255-32                 24.73Gi ±  5%     15.10Gi ±  49%    -38.93% (p=0.002 n=6)
Kernels/s257-32                 3.164Gi ±  3%     3.427Gi ±   3%     +8.30% (p=0.002 n=6)
Kernels/s271-32                 21.61Gi ± 20%     18.51Gi ±  23%    -14.36% (p=0.026 n=6)
Kernels/s273-32                 28.50Gi ±  8%     16.23Gi ±  35%    -43.07% (p=0.002 n=6)
Kernels/s2275-32                4.834Gi ±  3%     4.968Gi ±   1%     +2.76% (p=0.009 n=6)
Kernels/s276-32                 43.31Gi ±  4%     32.38Gi ±  32%    -25.24% (p=0.009 n=6)
Kernels/s1279-32               102.42Gi ±  6%     94.15Gi ±  27%          ~ (p=0.065 n=6)
Kernels/s2711-32                20.64Gi ±  3%     18.21Gi ±  20%    -11.78% (p=0.002 n=6)
Kernels/s2712-32                24.28Gi ± 27%     18.70Gi ±  17%    -22.96% (p=0.004 n=6)
Kernels/s1281-32                30.18Gi ±  7%    178.01Gi ±   5%   +489.88% (p=0.002 n=6)
Kernels/s291-32                 23.44Gi ±  5%     21.35Gi ±  32%          ~ (p=0.310 n=6)
Kernels/s293-32                 20.72Gi ±  2%     18.52Gi ±  13%    -10.59% (p=0.015 n=6)
Kernels/s2101-32                2.004Ti ±  2%     1.890Ti ±   7%          ~ (p=0.240 n=6)
Kernels/s2102-32                2.271Gi ± 24%     1.717Gi ±   6%    -24.40% (p=0.004 n=6)
Kernels/s311-32                 10.37Gi ±  2%    143.76Gi ±   4%  +1286.77% (p=0.002 n=6)
Kernels/s312-32                 6.933Gi ±  3%   137.457Gi ±   4%  +1882.62% (p=0.002 n=6)
Kernels/s313-32                 20.86Gi ±  2%    169.18Gi ±   4%   +711.06% (p=0.002 n=6)
Kernels/s314-32                 14.64Gi ±  3%     11.02Gi ±  39%    -24.70% (p=0.002 n=6)
Kernels/s315-32                 9.173Gi ± 10%     7.821Gi ±  19%    -14.74% (p=0.009 n=6)
Kernels/s316-32                 20.15Gi ± 15%     16.63Gi ±  18%    -17.48% (p=0.015 n=6)
Kernels/s319-32                 26.44Gi ±  2%     22.83Gi ±  27%    -13.64% (p=0.002 n=6)
Kernels/s3111-32                9.670Gi ±  2%     8.027Gi ±  15%    -17.00% (p=0.002 n=6)
Kernels/s3113-32                8.790Gi ±  2%     5.218Gi ±  68%    -40.64% (p=0.015 n=6)
Kernels/s331-32                 16.45Gi ±  3%     14.81Gi ±  13%     -9.93% (p=0.002 n=6)
Kernels/s351-32                 57.30Gi ±  4%     50.81Gi ±  57%          ~ (p=0.065 n=6)
Kernels/s1351-32                12.22Gi ±  4%     12.60Gi ±   3%     +3.09% (p=0.041 n=6)
Kernels/s352-32                 20.62Gi ±  3%     18.12Gi ±  15%          ~ (p=0.240 n=6)
Kernels/s421-32                 54.57Gi ±  5%     45.28Gi ±  44%          ~ (p=1.000 n=6)
Kernels/s1421-32                74.87Gi ± 17%    314.56Gi ±   4%   +320.15% (p=0.002 n=6)
Kernels/s422-32                 38.15Gi ±  2%     26.44Gi ±  42%    -30.70% (p=0.004 n=6)
Kernels/s423-32                 41.43Gi ±  2%     25.96Gi ±  27%    -37.35% (p=0.002 n=6)
Kernels/s424-32                 40.14Gi ±  7%     29.87Gi ±  37%    -25.58% (p=0.026 n=6)
Kernels/s431-32                 39.52Gi ±  3%     16.58Gi ±  99%    -58.05% (p=0.002 n=6)
Kernels/s441-32                 25.43Gi ±  7%     19.44Gi ±  28%    -23.58% (p=0.009 n=6)
Kernels/s443-32                 35.36Gi ± 14%     18.94Gi ±  50%    -46.44% (p=0.002 n=6)
Kernels/s452-32                 30.70Gi ±  4%     19.21Gi ±  57%    -37.43% (p=0.026 n=6)
Kernels/s453-32                 20.96Gi ±  2%     14.87Gi ±  36%    -29.08% (p=0.002 n=6)
Kernels/s4117-32                40.56Gi ± 17%     28.48Gi ±  44%    -29.78% (p=0.015 n=6)
Kernels/va-32                   39.73Gi ± 14%    237.45Gi ±   4%   +497.65% (p=0.002 n=6)
Kernels/vif-32                  27.03Gi ±  4%     25.91Gi ±   7%     -4.14% (p=0.026 n=6)
Kernels/vpv-32                  39.81Gi ±  2%    179.44Gi ±   4%   +350.73% (p=0.002 n=6)
Kernels/vtv-32                  39.72Gi ±  4%    164.85Gi ±   6%   +315.05% (p=0.002 n=6)
Kernels/vpvtv-32                35.64Gi ± 82%    191.72Gi ±   5%   +437.99% (p=0.002 n=6)
Kernels/vpvts-32                33.72Gi ± 10%    154.85Gi ±   5%   +359.15% (p=0.002 n=6)
Kernels/vpvpv-32                35.52Gi ±  9%    208.06Gi ±   4%   +485.71% (p=0.002 n=6)
Kernels/vtvtv-32                39.48Gi ±  4%    162.60Gi ±   9%   +311.80% (p=0.002 n=6)
Kernels/vsumr-32                10.63Gi ±  2%    139.91Gi ±   4%  +1216.08% (p=0.002 n=6)
Kernels/vdotr-32                20.92Gi ±  2%    166.70Gi ±   5%   +696.75% (p=0.002 n=6)
Kernels/vbor-32                 1.211Ti ±  3%     1.215Ti ±   3%          ~ (p=0.937 n=6)
geomean                         24.41Gi           29.14Gi           +19.39%
```

The kernels loopvec rewrites (s000, s1112, va, vpv, vtv, vpvtv, vpvts, vpvpv,
vtvtv) are 78-85% faster. The rest are unchanged scalar source, yet they are
mostly slower and far noisier here (up to ±113%) than the same code built
without `GOEXPERIMENT=simd`. This seems to be an effect of the runtime saving
the full register file for every preemption of every goroutine in the process,
regardless of whether anything calls into SIMD. `vbor` touches only 256 of the
32000 elements of each array, so its bytes-per-second figure is not meaningful.

`GODEBUG=asyncpreemptoff=1` recovers most of the regression, which heavily indicates
that async preemption might be the culprint. See https://github.com/golang/go/issues/81847.

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
