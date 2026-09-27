# AGENTS.md

Working notes for agents in this repo. Read this before making non-trivial
changes to `internal/analysis`, `internal/rewrite`, or `tsvc/`.

## What this is

`loopvec` rewrites element-wise Go loops to use the experimental `simd`
package (`GOEXPERIMENT=simd`). Three parts:

- `internal/analysis` — finds candidate loops (`Analyze`).
- `internal/rewrite` — turns a matched `Loop` into SIMD source text.
- `main.go` / `cmd/loopvec-toolexec` — two ways to apply it: standalone CLI
  (`-split`/`-w`/`-d`) or a `go build -toolexec` wrapper that rewrites a
  whole build (including the standard library) in memory, no files touched.

## Build and test

```sh
make build   # ./bin/loopvec, ./bin/loopvec-toolexec
make test    # full suite, -race
make ci      # what CI runs, in one command (needs qemu-aarch64-static locally)
```

Prefer a `make` target over the equivalent raw `go` command whenever one
exists — they encode flags, env vars (`GOTOOLCHAIN`, `GOEXPERIMENT`,
`-toolexec` paths, `GOARCH`/`qemu-aarch64-static`) and dependencies
(`generate`, `build`) that are easy to get wrong or forget by hand. Reach
for a raw `go test -run ...`/`go build` only for tight iteration on a single
test or file where no target is scoped that narrowly.

Always set GOTOOLCHAIN for this repo (match the Makefile). Exception:
`gotip`, the Go language tip, never set `GOTOOLCHAIN` there.

`benchstat` and `qemu-aarch64-static` are assumed to be on `PATH`, not
installed by any target (same convention as assuming `go` itself is there).

## Testing convention — there is no unit-test layer

`internal/analysis` and `internal/rewrite` have **no `_test.go` files**.
Every behavior is tested through `testdata/*.txt` — txtar scripts driven by
`rsc.io/script` in `script_test.go`, which builds the real `loopvec` binary
and runs it. If you extend the analyzer or rewriter, add a new
`testdata/<name>.txt`, not a Go unit test. Naming convention: `<pattern>.txt`
for something that gets rewritten, `<pattern>_no_rewrite.txt` for something
that correctly doesn't. Look at `add_float32.txt` (accepted) and
`bound_unsupported_no_rewrite.txt` (rejected) for the exact script shape
(`-split`, check `stderr.want`/`ops_simd.go.want`/`ops.go.guarded`, then run
`testmain` under both builds to check runtime behavior too).

`tsvc/` and `bench/` do have ordinary Go tests/benchmarks alongside this.

## The analyzer is shape-matching, not dependence analysis

This is the single most important thing to know before trying to widen
coverage. `internal/analysis/analyzer.go` does **no general dependence or
invariance analysis**. It's a literal AST shape match:

- Loop clauses must match one of a small fixed set exactly (`for i := 0;
  i < len(s); i++`, and now `for i := len(s) - 1; i >= 0; i--`) — no other
  start, step, or direction.
- The destination must be `ident[i]` where `i` is *exactly* the loop
  variable (`asSliceIndex`) — `dst[i+1]`, `dst[2*i]` are rejected outright,
  regardless of whether they're actually dependent.
- A non-slice operand is only accepted as `*ast.Ident` or `*ast.BasicLit`
  (`buildExprTree`). An expression that's genuinely loop-invariant but isn't
  one of those two node kinds (`a[0]`, `a[len(a)/2]`, `t.n`) is rejected —
  there's a `refersTo(expr, indexVar, valueVar)` helper already used to
  *reject* scalars that vary per iteration; it isn't used to *accept*
  arbitrary invariant expressions.
- The loop body must be exactly one statement (`destinationIdent`). Nested
  loops (2-D access, `aa[i][j]`) are rejected on body shape alone, before
  any indexing is even looked at.

Consequence: several TSVC_2 kernels that a real dependence checker would
call trivially vectorizable (`s1112`, reverse with no dependence; `s113`,
`s1113`, invariant scalar reads) fail here on syntax, not on any actual
safety concern. `s1112` was added by extending the loop-clause whitelist
only — see the split of `analyzeFor` into `analyzeForForward`/
`analyzeForReverse`. The safety argument for accepting a new shape is always:
does every accepted body pattern already guarantee no two iterations touch
the same element? If yes (as with reversal), no new dependence checking is
needed, just a wider clause match.

## The rewriter has no idea about loop direction

`internal/rewrite/rewriter.go` never looks at the original `ForStmt`'s
clauses — it only uses `loop.ForStmt.Pos()/.End()` to know what text span to
replace, and generates a **fixed forward** `for _i := 0; _i < bound; { ... }`
template driven entirely by the `Loop` struct's semantic fields (`DstSlice`,
`Src1Slice`, `Op`, `Bound`, ...). `Loop` doesn't even have a direction field.
This means: extending the analyzer to accept a new *loop clause shape* that
still produces the same kind of `Loop` (same body semantics) needs **zero**
rewriter changes. Only a genuinely new body shape (strided access, 2-D,
gather) would need new codegen here.

## `tsvc/`

Ports dependence-testing kernels from
[TSVC_2](https://github.com/UoB-HPC/TSVC_2) to measure loopvec's actual
coverage and correctness, not to extend loopvec. See `tsvc/README.md` for
the day-to-day commands (`make tsvc-test`, `tsvc-test-qemu-arm64`,
`tsvc-update`, `tsvc-bench`). Two things worth knowing before touching it:

- It deliberately uses `-toolexec`, not `-split`, on the real package.
  `-split` would add a `//go:build !goexperiment.simd` tag to `kernels.go`,
  and loopvec skips build-tagged files on every future run — that would
  silently freeze coverage at whatever it was when split ran. If you need a
  `-split` build for some experiment, split a throwaway copy elsewhere
  (`cp` the package to a scratch dir with a standalone `go.mod`), never the
  real `tsvc/kernels.go`.
- `Kernels` (`zz_registry.go`) and `category_string.go` are generated from
  the `//tsvc:kernel` directives in `kernels.go` — never hand-edit either;
  run `go generate ./tsvc/...` (or `make generate`, which every `tsvc-*`
  target already depends on).

## Adding a new optimization (checklist)

Order that actually works, based on adding reverse-loop support (`s1112`):

1. **Establish the safety argument first, in words, before writing any
   code.** The question is always: does every body shape the analyzer would
   accept for this new loop-clause/access shape already guarantee no two
   iterations touch the same element? If yes, no dependence analysis is
   needed — just widen the match. If you can't answer this cleanly, don't
   add the shape yet.
2. **Extend `internal/analysis/analyzer.go`** to accept the new AST shape.
   Reuse `analyzeBody`/`analyzeBodyShape` unchanged if possible — most new
   *loop-clause* shapes (direction, alternate bounds) don't need to touch
   body-matching at all.
3. **Check whether `internal/rewrite/rewriter.go` needs any change.** If the
   new shape produces the same kind of `Loop` struct as an existing
   supported pattern, it doesn't — the rewriter is direction- and
   clause-agnostic (see above). Only a genuinely new body/access shape needs
   new codegen.
4. **Add txtar tests**, not unit tests: `testdata/<pattern>.txt` (positive:
   check the exact rewritten output *and* run the scalar/SIMD binary to
   confirm matching runtime behavior) and, if there's a shape that looks
   similar but is actually unsafe, `testdata/<pattern>_no_rewrite.txt` to
   pin that it stays rejected.
5. **Update the root `README.md` pattern table** with the new row, plus a
   short safety-rationale sentence if it's non-obvious why the shape is
   safe.
6. **Demonstrate it in `bench/`** if it's a general-purpose pattern (not
   `tsvc`-specific): add the function to `ops.go`, temporarily remove its
   `//go:build !goexperiment.simd` guard, run `make bench-regen` to
   regenerate `ops_simd.go` (it re-adds the guard), add the matching
   `Benchmark*` in `bench_test.go`. Manually diff scalar vs.
   `GOEXPERIMENT=simd` output once before trusting any benchmark number —
   benchmarks measure speed, not correctness.
7. **If a `tsvc/` kernel now gets rewritten**, that's the real correctness
   proof: run `make tsvc-test` and `make tsvc-test-qemu-arm64` and confirm
   the new rewrite is bit-exact against the existing golden on both amd64
   and arm64. Then run `make tsvc-coverage-update` — `TestCoverage` compares
   `loopvec -json`'s output against `tsvc/testdata/coverage.txt` and fails
   on *any* change, improvement or regression, so this is required, not
   optional, once coverage moves. Update `tsvc/README.md`'s coverage
   table/count too.
8. **Re-run benchmarks last** (`make bench`, `make tsvc-bench`) and update
   README numbers. Watch for stale output: these targets redirect into fixed
   `/tmp/*.txt` paths, so a previous run that got killed can leave
   overlapping writes that corrupt the next one — `benchstat` will warn
   (`parsing measurement: invalid syntax`) or show mismatched sample counts
   (`n=9+10`) instead of a clean `n=10`; if you see either, delete the
   `/tmp` files and rerun before trusting the numbers.
9. **Final pass**: `make ci` (build, test, `tsvc` on amd64 and arm64, fmt,
   lint, all in one command) before calling it done. Fall back to
   `make test && make fmt && make lint` only if `qemu-aarch64-static` isn't
   available locally.
