# loopvec-toolexec

`loopvec-toolexec` is a `go build -toolexec` wrapper that vectorizes loops while
the go command compiles the build, so it also covers your dependencies and the
standard library without touching any source file:

```sh
go install github.com/mauri870/loopvec/cmd/loopvec-toolexec@latest
GOEXPERIMENT=simd go build -toolexec=$(which loopvec-toolexec) ./...
GOEXPERIMENT=simd go test  -toolexec=$(which loopvec-toolexec) ./...
```

For every `compile` it type-checks the package from the compiler's export data
(on a toolchain whose archives are in the compiler's private format, from the
export data `go list -export` writes for tools; see [BUGS.md](BUGS.md)),
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
- They are cryptographic: `crypto/...` and `golang.org/x/crypto/...`, vendored
  copies included. Constant-time code is written so that its running time does
  not depend on secret data, and a rewrite adds a runtime overlap check, a length
  check, and vector loads that nobody audited. Set `LOOPVEC_TOOLEXEC_CRYPTO=1` to
  rewrite them anyway.
- The build tests a package `simd` depends on (`go test fmt`). The go command
  then compiles a test variant of that package, and a rewritten package would
  link `simd` against a different build of it. The go command compiles packages
  concurrently, so nothing can be rewritten safely once such a build starts; the
  wrapper reads the go command's command line to notice (Linux only) and rewrites
  nothing. Elsewhere it can only leave alone the packages it sees built against
  such a variant, so testing those packages may fail to link. Set
  `LOOPVEC_TOOLEXEC_NO_TARGET_CHECK=1` to use only that per-package protection.

Loops in methods and in package-level initializers are never rewritten, as with
`-split`.

To see what happened, set `LOOPVEC_TOOLEXEC_LOG` to a file; one line is appended
for each rewritten package and for each rewrite that was rejected.
`LOOPVEC_TOOLEXEC_LOG_PKGS` restricts the log to a comma-separated list of import
paths, and `LOOPVEC_TOOLEXEC_DEBUG=1` also records why a package was skipped.

Set `LOOPVEC_TOOLEXEC_FP_REASSOC=1` for the `-fp-reassoc` behavior: floating-point
sums and products are regrouped, so a program's results can change in the last
bits. It is part of the wrapper's build identity, so toggling it rebuilds.

This mode rewrites code you have not reviewed. It applies the same
transformation as `-split` without the chance to benchmark it, and loops over
short slices can get slower. Treat it as an experiment and compare with the
scalar build.
