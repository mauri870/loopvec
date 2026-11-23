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

Baseline on AMD Ryzen 9 9950X3D (AVX-512), scalar vs. SIMD, for all 93 kernels:

```
goos: linux
goarch: amd64
pkg: github.com/mauri870/loopvec/tsvc
cpu: AMD Ryzen 9 9950X3D 16-Core Processor          
                 │ /tmp/tsvc_bench_scalar.txt │        /tmp/tsvc_bench_simd.txt        │
                 │           sec/op           │     sec/op      vs base                │
Kernels/s000-32                  6.104µ ±  3%    1.259µ ±   5%   -79.37% (p=0.002 n=6)
Kernels/s111-32                  5.654µ ±  2%    8.356µ ±  31%   +47.78% (p=0.002 n=6)
Kernels/s1111-32                 8.599µ ±  3%   22.028µ ±  53%  +156.17% (p=0.002 n=6)
Kernels/s112-32                  5.922µ ±  2%   15.841µ ±  28%  +167.51% (p=0.002 n=6)
Kernels/s1112-32                 6.508µ ± 17%    1.256µ ±   7%   -80.70% (p=0.002 n=6)
Kernels/s113-32                  6.831µ ±  3%   11.911µ ±  53%   +74.38% (p=0.002 n=6)
Kernels/s1113-32                 6.019µ ±  2%   14.743µ ±  58%  +144.94% (p=0.002 n=6)
Kernels/s114-32                  19.20µ ±  3%    22.57µ ±  12%   +17.54% (p=0.002 n=6)
Kernels/s115-32                  11.64µ ±  1%    17.76µ ±  22%   +52.58% (p=0.002 n=6)
Kernels/s1115-32                 48.63µ ±  1%    61.19µ ±  13%   +25.83% (p=0.002 n=6)
Kernels/s118-32                  68.88µ ±  0%    73.73µ ±   2%    +7.04% (p=0.002 n=6)
Kernels/s119-32                  31.60µ ±  1%    40.74µ ±  15%   +28.93% (p=0.002 n=6)
Kernels/s1119-32                 31.78µ ±  1%    42.11µ ±  21%   +32.49% (p=0.002 n=6)
Kernels/s121-32                  5.880µ ±  5%   13.237µ ±  55%  +125.11% (p=0.004 n=6)
Kernels/s122-32                  15.91µ ± 37%    20.03µ ±  26%   +25.86% (p=0.026 n=6)
Kernels/s124-32                  15.15µ ±  1%    22.34µ ±  16%   +47.52% (p=0.002 n=6)
Kernels/s125-32                  47.58µ ±  2%    51.79µ ±   5%    +8.85% (p=0.002 n=6)
Kernels/s127-32                  10.04µ ±  4%    28.19µ ±  23%  +180.91% (p=0.002 n=6)
Kernels/s128-32                  5.190µ ±  5%   11.743µ ±  40%  +126.28% (p=0.002 n=6)
Kernels/s131-32                  5.917µ ±  2%   11.796µ ±  54%   +99.37% (p=0.002 n=6)
Kernels/s132-32                  119.5n ±  1%    159.5n ±  24%   +33.47% (p=0.002 n=6)
Kernels/s141-32                  18.10µ ±  1%    22.25µ ±   3%   +22.88% (p=0.002 n=6)
Kernels/s162-32                  9.224µ ±  6%   15.429µ ±  39%   +67.27% (p=0.002 n=6)
Kernels/s171-32                  11.29µ ±  1%    15.22µ ±  30%   +34.83% (p=0.002 n=6)
Kernels/s172-32                  9.389µ ±  1%   16.924µ ±  23%   +80.25% (p=0.002 n=6)
Kernels/s173-32                  5.650µ ±  1%    9.550µ ±  32%   +69.04% (p=0.002 n=6)
Kernels/s174-32                  3.600µ ±  4%    8.745µ ±  26%  +142.93% (p=0.002 n=6)
Kernels/s175-32                  7.261µ ± 13%   13.296µ ±  52%   +83.13% (p=0.002 n=6)
Kernels/s176-32                  91.00m ±  0%   198.29m ±  31%  +117.88% (p=0.002 n=6)
Kernels/s211-32                  20.27µ ±  1%    59.67µ ±  19%  +194.41% (p=0.002 n=6)
Kernels/s1221-32                 16.81µ ±  1%    21.15µ ±  10%   +25.81% (p=0.002 n=6)
Kernels/s222-32                  67.17µ ±  1%    82.09µ ±  11%   +22.20% (p=0.002 n=6)
Kernels/s231-32                  356.4µ ±  4%    385.1µ ±   5%    +8.06% (p=0.002 n=6)
Kernels/s2233-32                 407.5µ ±  3%    426.1µ ±   5%    +4.58% (p=0.002 n=6)
Kernels/s235-32                  346.7µ ±  2%    376.5µ ±   4%    +8.59% (p=0.002 n=6)
Kernels/s241-32                 14.561µ ±  1%    4.240µ ±   2%   -70.88% (p=0.002 n=6)
Kernels/s243-32                 21.730µ ±  5%    5.860µ ±   1%   -73.03% (p=0.002 n=6)
Kernels/s2244-32                 12.33µ ±  5%    27.48µ ±  51%  +122.95% (p=0.002 n=6)
Kernels/s251-32                 11.241µ ± 10%    1.920µ ±  13%   -82.92% (p=0.002 n=6)
Kernels/s1251-32                15.241µ ±  1%    3.036µ ±   2%   -80.08% (p=0.002 n=6)
Kernels/s3251-32                 21.89µ ±  1%    38.33µ ±  50%   +75.10% (p=0.002 n=6)
Kernels/s252-32                  11.28µ ±  0%    17.49µ ± 227%   +55.05% (p=0.002 n=6)
Kernels/s254-32                  8.663µ ±  1%   18.432µ ±  51%  +112.77% (p=0.002 n=6)
Kernels/s255-32                  9.098µ ± 23%   36.703µ ±  33%  +303.41% (p=0.002 n=6)
Kernels/s257-32                  182.2µ ±  1%    183.5µ ±   3%         ~ (p=0.132 n=6)
Kernels/s271-32                  15.93µ ± 29%    20.81µ ±  19%   +30.59% (p=0.002 n=6)
Kernels/s273-32                  20.96µ ±  1%    59.43µ ±  62%  +183.62% (p=0.002 n=6)
Kernels/s2275-32                 258.5µ ±  2%    258.1µ ±   4%         ~ (p=0.310 n=6)
Kernels/s276-32                  10.65µ ±  8%    22.77µ ±  38%  +113.80% (p=0.002 n=6)
Kernels/s1279-32                 6.028µ ±  8%    7.878µ ±  23%   +30.69% (p=0.015 n=6)
Kernels/s2711-32                 17.03µ ±  1%    22.48µ ±  12%   +32.02% (p=0.002 n=6)
Kernels/s2712-32                 15.25µ ± 26%    21.87µ ±  29%   +43.42% (p=0.009 n=6)
Kernels/s1281-32                19.283µ ±  2%    3.323µ ±   4%   -82.77% (p=0.002 n=6)
Kernels/s291-32                  9.764µ ±  2%   20.215µ ±  45%  +107.03% (p=0.002 n=6)
Kernels/s293-32                  5.657µ ±  0%    8.025µ ±  15%   +41.86% (p=0.002 n=6)
Kernels/s2101-32                 360.2n ±  5%    364.8n ±   8%         ~ (p=0.310 n=6)
Kernels/s2102-32                 115.7µ ± 19%    141.3µ ±  23%   +22.14% (p=0.041 n=6)
Kernels/s311-32                11261.0n ±  1%    815.6n ±   5%   -92.76% (p=0.002 n=6)
Kernels/s312-32                16856.5n ±  1%    813.5n ±   1%   -95.17% (p=0.002 n=6)
Kernels/s313-32                 11.195µ ±  1%    1.381µ ±   7%   -87.66% (p=0.002 n=6)
Kernels/s314-32                  8.056µ ± 16%   13.027µ ±  36%   +61.70% (p=0.015 n=6)
Kernels/s315-32                  12.46µ ±  3%    15.42µ ±  20%   +23.76% (p=0.026 n=6)
Kernels/s316-32                  5.756µ ±  1%    7.877µ ±  14%   +36.84% (p=0.002 n=6)
Kernels/s317-32                 10.914µ ± 12%    9.829µ ±   2%         ~ (p=0.699 n=6)
Kernels/s319-32                  22.42µ ±  0%    34.15µ ±  24%   +52.34% (p=0.002 n=6)
Kernels/s3111-32                 12.13µ ±  1%    14.63µ ±  13%   +20.56% (p=0.002 n=6)
Kernels/s3113-32                 12.92µ ±  2%    30.84µ ±  58%  +138.58% (p=0.002 n=6)
Kernels/s331-32                  7.083µ ±  1%    8.903µ ±  10%   +25.70% (p=0.002 n=6)
Kernels/s351-32                  6.133µ ±  1%   16.219µ ±  62%  +164.42% (p=0.015 n=6)
Kernels/s1351-32                 28.62µ ±  1%    28.42µ ±   2%         ~ (p=0.065 n=6)
Kernels/s352-32                  11.21µ ±  1%    15.90µ ±  29%   +41.75% (p=0.004 n=6)
Kernels/s421-32                  5.861µ ±  1%   12.457µ ±  60%  +112.54% (p=0.002 n=6)
Kernels/s1421-32                3685.0n ± 10%    756.4n ±   3%   -79.47% (p=0.002 n=6)
Kernels/s422-32                  9.300µ ± 12%   16.695µ ±  21%   +79.52% (p=0.002 n=6)
Kernels/s423-32                  8.542µ ±  0%   16.949µ ±  15%   +98.41% (p=0.002 n=6)
Kernels/s424-32                  8.615µ ±  2%   17.017µ ±  27%   +97.54% (p=0.002 n=6)
Kernels/s431-32                  5.878µ ±  1%   15.712µ ±  31%  +167.30% (p=0.002 n=6)
Kernels/s441-32                  17.25µ ±  6%    26.47µ ±  23%   +53.48% (p=0.002 n=6)
Kernels/s443-32                  12.63µ ±  9%    29.58µ ±  39%  +134.16% (p=0.002 n=6)
Kernels/s452-32                  11.29µ ±  1%    21.42µ ±  44%   +89.78% (p=0.002 n=6)
Kernels/s453-32                  11.30µ ±  2%    16.47µ ±  24%   +45.68% (p=0.002 n=6)
Kernels/s4117-32                 11.32µ ±  1%    20.31µ ±  38%   +79.34% (p=0.002 n=6)
Kernels/va-32                   5764.0n ±  2%    974.0n ±   1%   -83.10% (p=0.002 n=6)
Kernels/vif-32                   8.655µ ±  3%    9.258µ ±  32%    +6.96% (p=0.009 n=6)
Kernels/vpv-32                   5.853µ ±  0%    1.298µ ±   2%   -77.82% (p=0.002 n=6)
Kernels/vtv-32                   5.877µ ±  1%    1.374µ ±   3%   -76.63% (p=0.002 n=6)
Kernels/vpvtv-32                 9.577µ ±  3%    1.810µ ±   1%   -81.10% (p=0.002 n=6)
Kernels/vpvts-32                 7.146µ ± 12%    1.525µ ±   5%   -78.66% (p=0.002 n=6)
Kernels/vpvpv-32                 9.423µ ±  1%    1.650µ ±   2%   -82.49% (p=0.002 n=6)
Kernels/vtvtv-32                 8.896µ ±  1%    2.150µ ±   1%   -75.83% (p=0.002 n=6)
Kernels/vsumr-32               11175.5n ±  1%    817.6n ±   2%   -92.68% (p=0.002 n=6)
Kernels/vdotr-32                11.243µ ±  1%    1.363µ ±   1%   -87.88% (p=0.002 n=6)
Kernels/vbor-32                  752.1n ±  2%    752.1n ±   1%         ~ (p=1.000 n=6)
geomean                          13.39µ          13.15µ           -1.76%

                 │ /tmp/tsvc_bench_scalar.txt │         /tmp/tsvc_bench_simd.txt          │
                 │            B/s             │       B/s         vs base                 │
Kernels/s000-32                 39.07Gi ±  3%    189.39Gi ±   5%   +384.80% (p=0.002 n=6)
Kernels/s111-32                 42.17Gi ±  2%     28.57Gi ±  44%    -32.25% (p=0.002 n=6)
Kernels/s1111-32                55.45Gi ±  2%     21.82Gi ±  35%    -60.66% (p=0.002 n=6)
Kernels/s112-32                 40.26Gi ±  2%     15.08Gi ±  40%    -62.54% (p=0.002 n=6)
Kernels/s1112-32                36.65Gi ± 15%    189.83Gi ±   7%   +417.92% (p=0.002 n=6)
Kernels/s113-32                 34.90Gi ±  3%     20.02Gi ±  34%    -42.65% (p=0.002 n=6)
Kernels/s1113-32                39.61Gi ±  2%     16.19Gi ± 139%    -59.14% (p=0.002 n=6)
Kernels/s114-32                 25.43Gi ±  3%     21.66Gi ±  14%    -14.84% (p=0.002 n=6)
Kernels/s115-32                 31.22Gi ±  1%     20.47Gi ±  18%    -34.43% (p=0.002 n=6)
Kernels/s1115-32                15.06Gi ±  1%     11.97Gi ±  12%    -20.51% (p=0.002 n=6)
Kernels/s118-32                 5.275Gi ±  0%     4.928Gi ±   2%     -6.57% (p=0.002 n=6)
Kernels/s119-32                 15.45Gi ±  1%     11.99Gi ±  13%    -22.44% (p=0.002 n=6)
Kernels/s1119-32                15.36Gi ±  1%     11.60Gi ±  27%    -24.49% (p=0.002 n=6)
Kernels/s121-32                 40.55Gi ±  5%     18.04Gi ± 122%    -55.50% (p=0.004 n=6)
Kernels/s122-32                 14.98Gi ± 59%     12.06Gi ±  33%    -19.51% (p=0.026 n=6)
Kernels/s124-32                 39.35Gi ±  1%     26.68Gi ±  19%    -32.21% (p=0.002 n=6)
Kernels/s125-32                 20.53Gi ±  2%     18.86Gi ±   5%     -8.13% (p=0.002 n=6)
Kernels/s127-32                 59.39Gi ±  4%     21.21Gi ±  30%    -64.28% (p=0.002 n=6)
Kernels/s128-32                 91.89Gi ±  5%     40.61Gi ±  29%    -55.80% (p=0.002 n=6)
Kernels/s131-32                 40.30Gi ±  2%     20.21Gi ±  85%    -49.84% (p=0.002 n=6)
Kernels/s132-32                 3.943Ti ±  1%     2.962Ti ±  31%    -24.89% (p=0.002 n=6)
Kernels/s141-32                 26.97Gi ±  1%     21.95Gi ±   3%    -18.61% (p=0.002 n=6)
Kernels/s162-32                 38.77Gi ±  6%     23.40Gi ±  50%    -39.66% (p=0.002 n=6)
Kernels/s171-32                 21.12Gi ±  1%     15.93Gi ±  29%    -24.59% (p=0.002 n=6)
Kernels/s172-32                 25.39Gi ±  1%     14.09Gi ±  30%    -44.51% (p=0.002 n=6)
Kernels/s173-32                 42.20Gi ±  1%     24.96Gi ±  46%    -40.85% (p=0.002 n=6)
Kernels/s174-32                 66.23Gi ±  5%     27.40Gi ±  34%    -58.63% (p=0.002 n=6)
Kernels/s175-32                 32.85Gi ± 12%     17.95Gi ±  34%    -45.36% (p=0.002 n=6)
Kernels/s176-32                 4.025Mi ±  0%     1.850Mi ±  45%    -54.03% (p=0.002 n=6)
Kernels/s211-32                29.407Gi ±  1%     9.993Gi ±  22%    -66.02% (p=0.002 n=6)
Kernels/s1221-32                14.18Gi ±  1%     11.32Gi ±  10%    -20.21% (p=0.002 n=6)
Kernels/s222-32                 7.099Gi ±  1%     5.831Gi ±  10%    -17.86% (p=0.002 n=6)
Kernels/s231-32                 1.371Gi ±  4%     1.268Gi ±   5%     -7.52% (p=0.002 n=6)
Kernels/s2233-32                1.797Gi ±  3%     1.719Gi ±   5%     -4.38% (p=0.002 n=6)
Kernels/s235-32                 2.440Gi ±  2%     2.247Gi ±   4%     -7.91% (p=0.002 n=6)
Kernels/s241-32                 32.75Gi ±  1%    112.48Gi ±   2%   +243.47% (p=0.002 n=6)
Kernels/s243-32                 27.43Gi ±  4%    101.72Gi ±   1%   +270.85% (p=0.002 n=6)
Kernels/s2244-32                38.69Gi ±  5%     17.54Gi ± 100%    -54.66% (p=0.002 n=6)
Kernels/s251-32                 42.51Gi ±  9%    248.42Gi ±  12%   +484.37% (p=0.002 n=6)
Kernels/s1251-32                39.11Gi ±  1%    196.36Gi ±   2%   +402.10% (p=0.002 n=6)
Kernels/s3251-32                27.23Gi ±  1%     15.55Gi ±  33%    -42.88% (p=0.002 n=6)
Kernels/s252-32                 31.71Gi ±  0%     20.47Gi ±  69%    -35.44% (p=0.002 n=6)
Kernels/s254-32                 27.52Gi ±  1%     12.94Gi ± 103%    -52.99% (p=0.002 n=6)
Kernels/s255-32                26.225Gi ± 19%     6.498Gi ±  49%    -75.22% (p=0.002 n=6)
Kernels/s257-32                 3.333Gi ±  1%     3.310Gi ±   3%          ~ (p=0.132 n=6)
Kernels/s271-32                 22.44Gi ± 40%     17.19Gi ±  17%    -23.42% (p=0.002 n=6)
Kernels/s273-32                 28.44Gi ±  1%     10.03Gi ± 166%    -64.74% (p=0.002 n=6)
Kernels/s2275-32                4.678Gi ±  2%     4.685Gi ±   4%          ~ (p=0.310 n=6)
Kernels/s276-32                 44.78Gi ±  7%     20.95Gi ±  28%    -53.23% (p=0.002 n=6)
Kernels/s1279-32                98.90Gi ±  8%     75.70Gi ±  29%    -23.46% (p=0.015 n=6)
Kernels/s2711-32                21.00Gi ±  1%     15.91Gi ±  14%    -24.24% (p=0.002 n=6)
Kernels/s2712-32                23.45Gi ± 35%     16.51Gi ±  40%    -29.58% (p=0.009 n=6)
Kernels/s1281-32                30.91Gi ±  2%    179.39Gi ±   4%   +480.33% (p=0.002 n=6)
Kernels/s291-32                 24.42Gi ±  2%     11.80Gi ±  81%    -51.70% (p=0.002 n=6)
Kernels/s293-32                 21.07Gi ±  0%     14.86Gi ±  18%    -29.50% (p=0.002 n=6)
Kernels/s2101-32                1.986Ti ±  6%     1.961Ti ±   8%          ~ (p=0.310 n=6)
Kernels/s2102-32                2.115Gi ± 22%     1.728Gi ±  31%    -18.28% (p=0.041 n=6)
Kernels/s311-32                 10.59Gi ±  1%    146.16Gi ±   5%  +1280.69% (p=0.002 n=6)
Kernels/s312-32                 7.072Gi ±  1%   146.540Gi ±   1%  +1972.07% (p=0.002 n=6)
Kernels/s313-32                 21.30Gi ±  1%    172.63Gi ±   7%   +710.55% (p=0.002 n=6)
Kernels/s314-32                14.847Gi ± 14%     9.161Gi ±  56%    -38.30% (p=0.015 n=6)
Kernels/s315-32                 9.569Gi ±  3%     7.732Gi ±  24%    -19.20% (p=0.026 n=6)
Kernels/s316-32                 20.71Gi ±  1%     15.13Gi ±  16%    -26.92% (p=0.002 n=6)
Kernels/s319-32                 26.59Gi ±  0%     17.46Gi ±  28%    -34.35% (p=0.002 n=6)
Kernels/s3111-32                9.825Gi ±  1%     8.152Gi ±  11%    -17.03% (p=0.002 n=6)
Kernels/s3113-32                9.225Gi ±  2%     4.164Gi ±  99%    -54.86% (p=0.002 n=6)
Kernels/s331-32                 16.83Gi ±  1%     13.40Gi ±   9%    -20.38% (p=0.002 n=6)
Kernels/s351-32                 58.31Gi ±  1%     24.27Gi ± 141%    -58.38% (p=0.015 n=6)
Kernels/s1351-32                12.50Gi ±  1%     12.58Gi ±   2%          ~ (p=0.065 n=6)
Kernels/s352-32                 21.26Gi ±  1%     15.00Gi ±  40%    -29.44% (p=0.004 n=6)
Kernels/s421-32                 62.00Gi ±  1%     29.17Gi ± 100%    -52.95% (p=0.002 n=6)
Kernels/s1421-32                64.72Gi ± 11%    315.21Gi ±   3%   +387.08% (p=0.002 n=6)
Kernels/s422-32                 39.07Gi ± 11%     21.77Gi ±  26%    -44.27% (p=0.002 n=6)
Kernels/s423-32                 42.54Gi ±  0%     21.45Gi ±  14%    -49.57% (p=0.002 n=6)
Kernels/s424-32                 42.18Gi ±  2%     21.36Gi ±  36%    -49.37% (p=0.002 n=6)
Kernels/s431-32                 40.56Gi ±  1%     15.19Gi ±  44%    -62.57% (p=0.002 n=6)
Kernels/s441-32                 27.65Gi ±  7%     18.05Gi ±  30%    -34.72% (p=0.002 n=6)
Kernels/s443-32                 37.77Gi ±  9%     16.22Gi ±  63%    -57.04% (p=0.002 n=6)
Kernels/s452-32                 31.68Gi ±  1%     16.69Gi ±  79%    -47.31% (p=0.002 n=6)
Kernels/s453-32                 21.09Gi ±  2%     14.50Gi ±  31%    -31.24% (p=0.002 n=6)
Kernels/s4117-32                42.11Gi ±  1%     23.49Gi ±  61%    -44.22% (p=0.002 n=6)
Kernels/va-32                   41.36Gi ±  2%    244.78Gi ±   1%   +491.81% (p=0.002 n=6)
Kernels/vif-32                  27.55Gi ±  3%     25.76Gi ±  24%     -6.51% (p=0.009 n=6)
Kernels/vpv-32                  40.74Gi ±  0%    183.73Gi ±   2%   +351.01% (p=0.002 n=6)
Kernels/vtv-32                  40.57Gi ±  1%    173.55Gi ±   3%   +327.74% (p=0.002 n=6)
Kernels/vpvtv-32                37.34Gi ±  3%    197.54Gi ±   1%   +428.99% (p=0.002 n=6)
Kernels/vpvts-32                33.37Gi ± 11%    156.34Gi ±   5%   +368.57% (p=0.002 n=6)
Kernels/vpvpv-32                37.95Gi ±  1%    216.79Gi ±   2%   +471.19% (p=0.002 n=6)
Kernels/vtvtv-32                40.20Gi ±  1%    166.36Gi ±   1%   +313.82% (p=0.002 n=6)
Kernels/vsumr-32                10.67Gi ±  1%    145.80Gi ±   2%  +1266.75% (p=0.002 n=6)
Kernels/vdotr-32                21.21Gi ±  1%    174.94Gi ±   1%   +724.93% (p=0.002 n=6)
Kernels/vbor-32                 1.246Ti ±  2%     1.246Ti ±   1%          ~ (p=1.000 n=6)
geomean                         24.86Gi           25.35Gi            +2.00%
```

Of the 22 kernels loopvec rewrites, 20 (s000, s1112, s241, s243, s251, s1251, s1281, s311,
s312, s313, s1421, va, vpv, vtv, vpvtv, vpvts, vpvpv, vtvtv, vsumr, vdotr) are 71-95% faster.
s421 and s422 are rewritten too but measured 113% and 80% slower, with a spread of ±60% and
±21%; the cause is not known. The rest are unchanged scalar source, yet 67 of the 93 are
slower and far noisier here (up to ±227%) than the same code built
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
