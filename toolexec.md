# loopvec-toolexec

`loopvec-toolexec` is a `go build -toolexec` wrapper that vectorizes loops while
the go command compiles the build, so it also covers your dependencies and the
standard library without touching any source file:

```sh
go install github.com/mauri870/loopvec/cmd/loopvec-toolexec@latest
GOEXPERIMENT=simd go build -toolexec=$(which loopvec-toolexec) ./...
GOEXPERIMENT=simd go test  -toolexec=$(which loopvec-toolexec) ./...
```

For every `compile` it type-checks the package from the compiler's export data,
rewrites the loops `loopvec` recognizes, type-checks the result again, and
compiles the rewritten source. If the rewritten package does not type-check, or
anything else is doubtful, the original source is compiled instead, so the
wrapper cannot make a working build fail. It also adds `simd` to the linker's
import configuration, since the go command does not know rewritten packages
import it. Build `loopvec-toolexec` with the same Go toolchain that runs the
build.

Packages are left alone when:

- `GOEXPERIMENT` does not enable `simd`.
- They are `simd` itself or anything `simd` or the runtime depends on (`fmt`,
  `os`, `strconv`, `sync`, `sync/atomic`, `math`, and more). Rewritten code
  imports `simd`, so rewriting these would create an import cycle.
- They use cgo, or the build uses `-race`, `-msan`, `-asan`, `-shared`,
  `-dynlink` or `-trimpath`.
- They are built against a test variant of a package `simd` depends on, which
  happens while testing those packages: `simd` would then be linked against a
  different build of the same package.

Loops in methods and in package-level initializers are never rewritten, as with
`-split`.

To see what happened, set `LOOPVEC_TOOLEXEC_LOG` to a file; one line is appended
for each rewritten package and for each rewrite that was rejected.
`LOOPVEC_TOOLEXEC_LOG_PKGS` restricts the log to a comma-separated list of import
paths, and `LOOPVEC_TOOLEXEC_DEBUG=1` also records why a package was skipped.

This mode rewrites code you have not reviewed. It applies the same
transformation as `-split` without the chance to benchmark it, and loops over
short slices can get slower. Treat it as an experiment and compare with the
scalar build.
