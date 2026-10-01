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
| s241 | node splitting | preloading necessary to allow vectorization | fused | vectorize | no |
| s243 | node splitting | false dependence cycle breaking | fused | vectorize | no |
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
# loopvec TSVC coverage: vectorized 17, gaps 75, declined 1, skipped 0 (of 93)
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
                 │           sec/op           │    sec/op      vs base                 │
Kernels/s000-32                 6515.5n ±  5%    933.6n ±  4%   -85.67% (p=0.000 n=10)
Kernels/s111-32                  5.664µ ±  1%   11.447µ ± 16%  +102.11% (p=0.000 n=10)
Kernels/s1111-32                 8.555µ ±  2%   31.882µ ± 35%  +272.67% (p=0.000 n=10)
Kernels/s112-32                  5.970µ ±  2%   15.504µ ± 46%  +159.71% (p=0.000 n=10)
Kernels/s1112-32                 6.792µ ± 14%    1.108µ ±  2%   -83.69% (p=0.000 n=10)
Kernels/s113-32                  6.874µ ±  3%   19.713µ ± 26%  +186.80% (p=0.000 n=10)
Kernels/s1113-32                 6.139µ ±  7%   22.384µ ± 35%  +264.65% (p=0.000 n=10)
Kernels/s114-32                  19.73µ ±  3%    26.60µ ± 13%   +34.84% (p=0.000 n=10)
Kernels/s115-32                  12.03µ ±  1%    26.54µ ±  7%  +120.65% (p=0.000 n=10)
Kernels/s1115-32                 49.32µ ±  2%    81.10µ ±  9%   +64.45% (p=0.000 n=10)
Kernels/s118-32                  70.36µ ±  3%    80.02µ ±  8%   +13.73% (p=0.000 n=10)
Kernels/s119-32                  32.07µ ±  2%    49.96µ ± 12%   +55.78% (p=0.000 n=10)
Kernels/s1119-32                 31.97µ ±  3%    50.90µ ± 18%   +59.22% (p=0.000 n=10)
Kernels/s121-32                  5.992µ ±  2%   17.901µ ± 36%  +198.75% (p=0.000 n=10)
Kernels/s122-32                  14.58µ ± 35%    23.46µ ± 15%   +60.94% (p=0.000 n=10)
Kernels/s124-32                  15.48µ ±  2%    27.11µ ± 17%   +75.18% (p=0.000 n=10)
Kernels/s125-32                  48.67µ ±  2%    57.60µ ±  2%   +18.35% (p=0.000 n=10)
Kernels/s127-32                  10.16µ ±  1%    26.66µ ± 50%  +162.35% (p=0.000 n=10)
Kernels/s128-32                  5.359µ ±  1%   16.157µ ± 13%  +201.48% (p=0.000 n=10)
Kernels/s131-32                  5.967µ ±  2%   16.624µ ± 36%  +178.62% (p=0.000 n=10)
Kernels/s132-32                  118.8n ±  0%    207.0n ± 17%   +74.27% (p=0.000 n=10)
Kernels/s141-32                  18.58µ ±  3%    27.33µ ±  3%   +47.05% (p=0.000 n=10)
Kernels/s162-32                  9.170µ ±  2%   21.399µ ± 25%  +133.36% (p=0.000 n=10)
Kernels/s171-32                  11.39µ ±  2%    24.48µ ± 27%  +114.90% (p=0.000 n=10)
Kernels/s172-32                  9.498µ ±  1%   22.541µ ± 11%  +137.34% (p=0.000 n=10)
Kernels/s173-32                  5.806µ ±  1%   10.079µ ± 25%   +73.60% (p=0.000 n=10)
Kernels/s174-32                  3.726µ ±  5%    8.030µ ± 30%  +115.53% (p=0.000 n=10)
Kernels/s175-32                  7.952µ ± 14%   21.213µ ± 20%  +166.76% (p=0.000 n=10)
Kernels/s176-32                  93.01m ±  2%   137.50m ± 45%   +47.84% (p=0.000 n=10)
Kernels/s211-32                  20.80µ ±  2%    64.75µ ± 43%  +211.31% (p=0.000 n=10)
Kernels/s1221-32                 17.03µ ±  2%    25.20µ ± 14%   +47.95% (p=0.000 n=10)
Kernels/s222-32                  69.08µ ±  3%   103.14µ ±  7%   +49.32% (p=0.000 n=10)
Kernels/s231-32                  343.3µ ±  5%    391.6µ ±  5%   +14.06% (p=0.000 n=10)
Kernels/s2233-32                 411.9µ ±  2%    443.5µ ±  3%    +7.68% (p=0.000 n=10)
Kernels/s235-32                  347.3µ ±  2%    380.5µ ±  5%    +9.58% (p=0.000 n=10)
Kernels/s241-32                  14.85µ ±  1%    52.35µ ± 27%  +252.53% (p=0.000 n=10)
Kernels/s243-32                  21.80µ ±  2%    69.86µ ± 33%  +220.41% (p=0.000 n=10)
Kernels/s2244-32                 13.49µ ±  3%    42.05µ ± 37%  +211.74% (p=0.000 n=10)
Kernels/s251-32                  10.97µ ±  3%    29.60µ ± 29%  +169.98% (p=0.000 n=10)
Kernels/s1251-32                 15.65µ ±  2%    32.31µ ± 28%  +106.49% (p=0.000 n=10)
Kernels/s3251-32                 24.55µ ±  1%    46.55µ ± 40%   +89.64% (p=0.000 n=10)
Kernels/s252-32                  11.50µ ±  1%    24.61µ ± 13%  +114.10% (p=0.000 n=10)
Kernels/s254-32                  9.182µ ±  2%   25.166µ ± 46%  +174.07% (p=0.000 n=10)
Kernels/s255-32                  9.247µ ±  2%   41.227µ ± 50%  +345.85% (p=0.000 n=10)
Kernels/s257-32                  186.3µ ±  3%    193.7µ ±  3%    +3.96% (p=0.000 n=10)
Kernels/s271-32                  17.11µ ±  6%    25.70µ ± 12%   +50.25% (p=0.000 n=10)
Kernels/s273-32                  21.20µ ±  1%    87.94µ ± 30%  +314.76% (p=0.000 n=10)
Kernels/s2275-32                 249.9µ ±  2%    246.6µ ±  1%    -1.31% (p=0.005 n=10)
Kernels/s276-32                  11.01µ ±  7%    36.17µ ± 17%  +228.65% (p=0.000 n=10)
Kernels/s1279-32                 5.802µ ±  2%   10.049µ ± 14%   +73.20% (p=0.000 n=10)
Kernels/s2711-32                 17.33µ ±  2%    28.22µ ±  6%   +62.86% (p=0.000 n=10)
Kernels/s2712-32                 16.47µ ± 17%    27.03µ ± 11%   +64.12% (p=0.000 n=10)
Kernels/s1281-32                 19.52µ ±  1%    58.36µ ± 12%  +198.93% (p=0.000 n=10)
Kernels/s291-32                  10.21µ ±  2%    21.98µ ± 27%  +115.32% (p=0.000 n=10)
Kernels/s293-32                  5.784µ ±  3%    9.945µ ±  4%   +71.95% (p=0.000 n=10)
Kernels/s2101-32                 339.6n ±  5%    351.7n ±  2%    +3.55% (p=0.043 n=10)
Kernels/s2102-32                 110.6µ ± 25%    101.7µ ±  8%         ~ (p=0.089 n=10)
Kernels/s311-32                  11.48µ ±  2%    15.99µ ± 12%   +39.35% (p=0.000 n=10)
Kernels/s312-32                  17.09µ ±  2%    16.78µ ±  1%    -1.80% (p=0.028 n=10)
Kernels/s313-32                  11.50µ ±  2%    24.01µ ± 11%  +108.71% (p=0.000 n=10)
Kernels/s314-32                  8.072µ ±  2%   17.282µ ± 17%  +114.10% (p=0.000 n=10)
Kernels/s315-32                  12.96µ ±  3%    20.33µ ± 17%   +56.85% (p=0.000 n=10)
Kernels/s316-32                  5.908µ ±  3%   10.209µ ±  6%   +72.79% (p=0.000 n=10)
Kernels/s317-32                  10.93µ ± 10%    10.97µ ± 12%         ~ (p=0.631 n=10)
Kernels/s319-32                  22.96µ ±  2%    42.36µ ± 34%   +84.50% (p=0.000 n=10)
Kernels/s3111-32                 12.48µ ±  3%    18.22µ ± 12%   +45.96% (p=0.000 n=10)
Kernels/s3113-32                 12.81µ ±  1%    59.93µ ± 43%  +367.65% (p=0.000 n=10)
Kernels/s331-32                  7.141µ ±  2%    8.979µ ± 13%   +25.74% (p=0.000 n=10)
Kernels/s351-32                  6.188µ ±  1%   19.130µ ± 60%  +209.16% (p=0.000 n=10)
Kernels/s1351-32                 28.78µ ±  2%    28.39µ ±  1%         ~ (p=0.247 n=10)
Kernels/s352-32                  11.46µ ±  2%    19.61µ ± 14%   +71.12% (p=0.000 n=10)
Kernels/s421-32                  6.024µ ±  1%   15.180µ ± 39%  +152.00% (p=0.000 n=10)
Kernels/s1421-32                 3.256µ ±  3%    9.329µ ± 40%  +186.56% (p=0.000 n=10)
Kernels/s422-32                  9.481µ ±  2%   14.455µ ± 23%   +52.47% (p=0.000 n=10)
Kernels/s423-32                  8.621µ ±  2%   18.275µ ± 30%  +111.98% (p=0.000 n=10)
Kernels/s424-32                  8.992µ ±  4%   19.610µ ± 33%  +118.09% (p=0.000 n=10)
Kernels/s431-32                  5.868µ ±  2%   19.497µ ± 48%  +232.26% (p=0.000 n=10)
Kernels/s441-32                  17.32µ ±  3%    40.93µ ± 25%  +136.24% (p=0.000 n=10)
Kernels/s443-32                  13.02µ ±  7%    37.48µ ± 21%  +187.81% (p=0.000 n=10)
Kernels/s452-32                  11.56µ ±  2%    29.56µ ± 30%  +155.59% (p=0.000 n=10)
Kernels/s453-32                  11.55µ ±  3%    21.34µ ± 17%   +84.73% (p=0.001 n=10)
Kernels/s4117-32                 11.45µ ±  2%    24.34µ ± 34%  +112.65% (p=0.000 n=10)
Kernels/va-32                   5920.0n ±  5%    992.0n ±  2%   -83.24% (p=0.000 n=10)
Kernels/vif-32                   9.376µ ±  4%    9.207µ ±  2%         ~ (p=0.247 n=10)
Kernels/vpv-32                   5.932µ ±  2%    1.303µ ±  2%   -78.04% (p=0.000 n=10)
Kernels/vtv-32                   5.962µ ±  2%    1.320µ ±  3%   -77.86% (p=0.000 n=10)
Kernels/vpvtv-32                 9.748µ ±  7%    1.631µ ±  4%   -83.27% (p=0.000 n=10)
Kernels/vpvts-32                 7.668µ ±  5%    1.256µ ±  2%   -83.63% (p=0.000 n=10)
Kernels/vpvpv-32                 9.526µ ±  1%    1.973µ ±  2%   -79.29% (p=0.000 n=10)
Kernels/vtvtv-32                 8.962µ ±  2%    1.837µ ±  2%   -79.50% (p=0.000 n=10)
Kernels/vsumr-32                 11.37µ ±  1%    15.23µ ±  8%   +33.97% (p=0.000 n=10)
Kernels/vdotr-32                 11.26µ ±  1%    23.20µ ± 28%  +106.14% (p=0.000 n=10)
Kernels/vbor-32                  765.0n ±  2%    773.6n ± 24%         ~ (p=0.105 n=10)
geomean                          13.61µ          20.94µ         +53.83%

                 │ /tmp/tsvc_bench_scalar.txt │        /tmp/tsvc_bench_simd.txt         │
                 │            B/s             │      B/s        vs base                 │
Kernels/s000-32                 36.59Gi ±  5%   255.40Gi ±  4%  +597.96% (p=0.000 n=10)
Kernels/s111-32                 42.10Gi ±  1%    20.84Gi ± 19%   -50.50% (p=0.000 n=10)
Kernels/s1111-32                55.74Gi ±  2%    14.97Gi ± 52%   -73.14% (p=0.000 n=10)
Kernels/s112-32                 39.94Gi ±  2%    15.38Gi ± 45%   -61.50% (p=0.000 n=10)
Kernels/s1112-32                35.10Gi ± 12%   215.19Gi ±  2%  +513.01% (p=0.000 n=10)
Kernels/s113-32                 34.69Gi ±  3%    12.19Gi ± 35%   -64.87% (p=0.000 n=10)
Kernels/s1113-32                38.84Gi ±  6%    10.65Gi ± 54%   -72.57% (p=0.000 n=10)
Kernels/s114-32                 24.75Gi ±  3%    18.36Gi ± 15%   -25.83% (p=0.000 n=10)
Kernels/s115-32                 30.21Gi ±  1%    13.70Gi ±  7%   -54.66% (p=0.000 n=10)
Kernels/s1115-32               14.852Gi ±  2%    9.031Gi ±  8%   -39.19% (p=0.000 n=10)
Kernels/s118-32                 5.164Gi ±  3%    4.541Gi ±  9%   -12.06% (p=0.000 n=10)
Kernels/s119-32                15.225Gi ±  2%    9.818Gi ± 14%   -35.51% (p=0.000 n=10)
Kernels/s1119-32               15.273Gi ±  3%    9.593Gi ± 19%   -37.19% (p=0.000 n=10)
Kernels/s121-32                 39.79Gi ±  2%    13.32Gi ± 27%   -66.52% (p=0.000 n=10)
Kernels/s122-32                 16.53Gi ± 52%    10.17Gi ± 17%   -38.48% (p=0.000 n=10)
Kernels/s124-32                 38.51Gi ±  2%    21.99Gi ± 21%   -42.91% (p=0.000 n=10)
Kernels/s125-32                 20.07Gi ±  2%    16.96Gi ±  2%   -15.50% (p=0.000 n=10)
Kernels/s127-32                 58.65Gi ±  2%    22.36Gi ± 98%   -61.88% (p=0.000 n=10)
Kernels/s128-32                 88.98Gi ±  1%    29.51Gi ± 12%   -66.83% (p=0.000 n=10)
Kernels/s131-32                 39.96Gi ±  2%    14.38Gi ± 26%   -64.01% (p=0.000 n=10)
Kernels/s132-32                 3.968Ti ±  0%    2.280Ti ± 20%   -42.56% (p=0.000 n=10)
Kernels/s141-32                 26.27Gi ±  3%    17.87Gi ±  3%   -32.00% (p=0.000 n=10)
Kernels/s162-32                 39.00Gi ±  2%    16.73Gi ± 20%   -57.11% (p=0.000 n=10)
Kernels/s171-32                20.931Gi ±  2%    9.751Gi ± 37%   -53.41% (p=0.000 n=10)
Kernels/s172-32                 25.10Gi ±  1%    10.61Gi ± 10%   -57.72% (p=0.000 n=10)
Kernels/s173-32                 41.06Gi ±  1%    23.66Gi ± 33%   -42.39% (p=0.000 n=10)
Kernels/s174-32                 63.99Gi ±  4%    29.83Gi ± 37%   -53.39% (p=0.000 n=10)
Kernels/s175-32                 29.99Gi ± 16%    11.24Gi ± 25%   -62.51% (p=0.000 n=10)
Kernels/s176-32                 3.939Mi ±  2%    2.661Mi ± 31%   -32.45% (p=0.000 n=10)
Kernels/s211-32                28.657Gi ±  2%    9.251Gi ± 39%   -67.72% (p=0.000 n=10)
Kernels/s1221-32               14.000Gi ±  2%    9.462Gi ± 16%   -32.41% (p=0.000 n=10)
Kernels/s222-32                 6.903Gi ±  3%    4.623Gi ±  7%   -33.03% (p=0.000 n=10)
Kernels/s231-32                 1.422Gi ±  5%    1.247Gi ±  4%   -12.33% (p=0.000 n=10)
Kernels/s2233-32                1.778Gi ±  2%    1.652Gi ±  3%    -7.13% (p=0.000 n=10)
Kernels/s235-32                 2.436Gi ±  2%    2.223Gi ±  5%    -8.72% (p=0.000 n=10)
Kernels/s241-32                32.113Gi ±  1%    9.110Gi ± 37%   -71.63% (p=0.000 n=10)
Kernels/s243-32                27.339Gi ±  2%    8.543Gi ± 39%   -68.75% (p=0.000 n=10)
Kernels/s2244-32                35.35Gi ±  3%    11.34Gi ± 60%   -67.92% (p=0.000 n=10)
Kernels/s251-32                 43.48Gi ±  3%    16.11Gi ± 38%   -62.96% (p=0.000 n=10)
Kernels/s1251-32                38.09Gi ±  2%    18.51Gi ± 38%   -51.42% (p=0.000 n=10)
Kernels/s3251-32                24.28Gi ±  1%    12.97Gi ± 36%   -46.59% (p=0.000 n=10)
Kernels/s252-32                 31.11Gi ±  2%    14.56Gi ± 15%   -53.20% (p=0.000 n=10)
Kernels/s254-32                25.967Gi ±  2%    9.536Gi ± 49%   -63.28% (p=0.000 n=10)
Kernels/s255-32                25.784Gi ±  1%    5.800Gi ± 99%   -77.51% (p=0.000 n=10)
Kernels/s257-32                 3.261Gi ±  3%    3.137Gi ±  3%    -3.81% (p=0.000 n=10)
Kernels/s271-32                 20.91Gi ±  6%    13.91Gi ± 11%   -33.44% (p=0.000 n=10)
Kernels/s273-32                28.113Gi ±  1%    6.799Gi ± 23%   -75.81% (p=0.000 n=10)
Kernels/s2275-32                4.839Gi ±  2%    4.904Gi ±  1%    +1.33% (p=0.005 n=10)
Kernels/s276-32                 43.33Gi ±  6%    13.18Gi ± 21%   -69.57% (p=0.000 n=10)
Kernels/s1279-32               102.73Gi ±  2%    59.32Gi ± 16%   -42.26% (p=0.000 n=10)
Kernels/s2711-32                20.64Gi ±  2%    12.68Gi ±  6%   -38.59% (p=0.000 n=10)
Kernels/s2712-32                21.71Gi ± 21%    13.24Gi ± 12%   -39.04% (p=0.000 n=10)
Kernels/s1281-32                30.53Gi ±  1%    10.23Gi ± 13%   -66.51% (p=0.000 n=10)
Kernels/s291-32                 23.35Gi ±  2%    10.86Gi ± 22%   -53.48% (p=0.000 n=10)
Kernels/s293-32                 20.61Gi ±  3%    11.99Gi ±  4%   -41.85% (p=0.000 n=10)
Kernels/s2101-32                2.106Ti ±  5%    2.034Ti ±  2%    -3.44% (p=0.043 n=10)
Kernels/s2102-32                2.208Gi ± 20%    2.401Gi ±  7%         ~ (p=0.089 n=10)
Kernels/s311-32                10.387Gi ±  2%    7.455Gi ± 13%   -28.23% (p=0.000 n=10)
Kernels/s312-32                 6.975Gi ±  2%    7.103Gi ±  1%    +1.83% (p=0.029 n=10)
Kernels/s313-32                20.725Gi ±  2%    9.936Gi ± 12%   -52.06% (p=0.000 n=10)
Kernels/s314-32                14.768Gi ±  2%    6.900Gi ± 17%   -53.28% (p=0.000 n=10)
Kernels/s315-32                 9.198Gi ±  3%    5.866Gi ± 20%   -36.22% (p=0.000 n=10)
Kernels/s316-32                 20.18Gi ±  3%    11.68Gi ±  6%   -42.13% (p=0.000 n=10)
Kernels/s319-32                 25.96Gi ±  2%    14.07Gi ± 51%   -45.80% (p=0.000 n=10)
Kernels/s3111-32                9.552Gi ±  3%    6.544Gi ± 13%   -31.49% (p=0.000 n=10)
Kernels/s3113-32                9.303Gi ±  1%    1.990Gi ± 77%   -78.60% (p=0.000 n=10)
Kernels/s331-32                 16.69Gi ±  2%    13.28Gi ± 12%   -20.46% (p=0.000 n=10)
Kernels/s351-32                 57.80Gi ±  1%    18.74Gi ± 47%   -67.57% (p=0.000 n=10)
Kernels/s1351-32                12.43Gi ±  2%    12.60Gi ±  1%         ~ (p=0.247 n=10)
Kernels/s352-32                 20.81Gi ±  2%    12.18Gi ± 16%   -41.45% (p=0.000 n=10)
Kernels/s421-32                 60.32Gi ±  1%    24.05Gi ± 64%   -60.14% (p=0.000 n=10)
Kernels/s1421-32                73.23Gi ±  3%    25.72Gi ± 67%   -64.88% (p=0.000 n=10)
Kernels/s422-32                 38.33Gi ±  2%    25.14Gi ± 18%   -34.41% (p=0.000 n=10)
Kernels/s423-32                 42.15Gi ±  2%    19.94Gi ± 35%   -52.68% (p=0.000 n=10)
Kernels/s424-32                 40.41Gi ±  4%    18.54Gi ± 49%   -54.13% (p=0.000 n=10)
Kernels/s431-32                 40.63Gi ±  2%    12.24Gi ± 91%   -69.87% (p=0.000 n=10)
Kernels/s441-32                 27.53Gi ±  3%    11.66Gi ± 34%   -57.65% (p=0.000 n=10)
Kernels/s443-32                 36.62Gi ±  6%    12.75Gi ± 26%   -65.19% (p=0.000 n=10)
Kernels/s452-32                 30.93Gi ±  2%    12.10Gi ± 42%   -60.87% (p=0.000 n=10)
Kernels/s453-32                 20.64Gi ±  3%    11.17Gi ± 20%   -45.86% (p=0.001 n=10)
Kernels/s4117-32                41.66Gi ±  2%    19.66Gi ± 51%   -52.80% (p=0.000 n=10)
Kernels/va-32                   40.27Gi ±  5%   240.34Gi ±  2%  +496.79% (p=0.000 n=10)
Kernels/vif-32                  25.43Gi ±  4%    25.89Gi ±  2%         ~ (p=0.247 n=10)
Kernels/vpv-32                  40.19Gi ±  2%   183.08Gi ±  2%  +355.48% (p=0.000 n=10)
Kernels/vtv-32                  39.99Gi ±  2%   180.60Gi ±  3%  +351.64% (p=0.000 n=10)
Kernels/vpvtv-32                36.70Gi ±  7%   219.29Gi ±  4%  +497.58% (p=0.000 n=10)
Kernels/vpvts-32                31.09Gi ±  5%   189.94Gi ±  2%  +510.85% (p=0.000 n=10)
Kernels/vpvpv-32                37.54Gi ±  1%   181.26Gi ±  2%  +382.81% (p=0.000 n=10)
Kernels/vtvtv-32                39.91Gi ±  2%   194.68Gi ±  2%  +387.83% (p=0.000 n=10)
Kernels/vsumr-32               10.488Gi ±  1%    7.830Gi ±  9%   -25.34% (p=0.000 n=10)
Kernels/vdotr-32                21.18Gi ±  1%    10.28Gi ± 39%   -51.48% (p=0.000 n=10)
Kernels/vbor-32                 1.225Ti ±  2%    1.211Ti ± 19%         ~ (p=0.105 n=10)
geomean                         24.44Gi          15.83Gi         -35.23%
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
