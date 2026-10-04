# Upstream bugs

loopvec rewrites code to use `GOEXPERIMENT=simd`, which is new, so most of what it hits
is in the Go toolchain. This file lists the Go issues that affect loopvec. "tip" means
gotip, "1.27.1" is the release toolchain loopvec is tested against.

| Issue | Status | Affects | loopvec's answer |
|---|---|---|---|
| [#80657](https://github.com/golang/go/issues/80657) simd in a method: internal compiler error | fixed on tip (CL 839405), still crashes on 1.27.1 | `-methods` | methods are skipped unless `-methods` is passed |
| [#80835](https://github.com/golang/go/issues/80835) legacy SSE encodings in functions using simd cause AVX-SSE transition penalties | open, fix pending (CLs 825145, 825146, in tip) | speed of any scalar float code that runs after a rewritten loop, in the same function or another | none; the portable `simd` package has no `ClearAVXUpperBits` |
| [#81844](https://github.com/golang/go/issues/81844) blank identifier parameters rejected | fixed on tip (CL 841145) | every rewrite of a function with a `_` parameter | the generated file renames them `_p0`, `_p1`, ... |
| [#81845](https://github.com/golang/go/issues/81845) calls left unresolved when inlined into a package initializer | closed as a duplicate of [#80689](https://github.com/golang/go/issues/80689), which is open | loops in package-level `var x = func() {...}()` | not rewritten (the generated initializer fails to link), see `package_initializer_no_rewrite.txt` and `toolexec_package_initializer.txt` |
| [#81846](https://github.com/golang/go/issues/81846) `Float32s.Neg` rebuilds its sign constant every iteration | open | speed of negation loops | none; the loop is correct, only slower than it should be |
| [#81847](https://github.com/golang/go/issues/81847) `GOEXPERIMENT=simd` alone slows async preemption on AVX-512 | open | any benchmark comparing a scalar and a simd build | benchmark with `GODEBUG=asyncpreemptoff=1` |
| [#82001](https://github.com/golang/go/issues/82001) emulated `Load{Float64,Int64,Uint64}sPart` return `len(s)` | open | generated loops under `GODEBUG=simd=0` | none; the loops use the count `Load*Part` returns, as the package documents |
| [#82002](https://github.com/golang/go/issues/82002) `Min` and `Max` on floats differ from Go's for NaN and -0 | open | float `min`/`max` | not rewritten (`float min and max differ from Go's ...`) |

## #80657: simd in a method

A function with a receiver that contains simd code failed to compile with
`internal compiler error: missing Types entry`. loopvec does not rewrite loops in methods
by default; `-methods` turns it on. The fix is on tip. On 1.27.1 the crash is still there,
and `flag_methods_ice.txt` pins it, so that test starts failing when the pinned toolchain
gets the fix. Reported by someone else; the fix is [CL 839405](https://go.dev/cl/839405).

## #80835: legacy SSE after vector code

The compiler emits legacy (non-VEX) SSE encodings for scalar float code and never a
`VZEROUPPER`, so scalar code that runs while the upper halves of the vector registers are
dirty pays a penalty. The issue reports it for legacy instructions inside functions that use
the intrinsics, and the pending CLs use VEX encodings in functions where AVX is present.
loopvec meets a broader case that those CLs do not obviously cover: a rewritten loop leaves
the state dirty, and the scalar float code that follows, in the same function or in a
function that calls it, has no AVX in it to key on.

Measured on an AMD Ryzen 9 9950X3D with a vector loop followed by a scalar min/max scan,
pinned to one core with `GODEBUG=asyncpreemptoff=1`: the pair costs 1.5x (go1.27.1) to 2.4x
(tip, which already has the two VEX CLs) what its parts cost, at 256 and 512 bits, and
exactly the sum of the parts once `archsimd.ClearAVXUpperBits()` runs between them. The
vector loop itself was 7.6x faster than the scalar one, so a function that is mostly the
vector loop still wins and one that is mostly scalar float code around it can lose. Only
`simd/archsimd` (amd64) has `ClearAVXUpperBits`, not the portable `simd`, so loopvec has no
portable way to clear the state. When comparing a function before and after a rewrite,
benchmark the whole function, not the loop. This is separate from #81847: it reproduces with
async preemption off. One machine and no performance counters, so the size of the effect on
Intel, where the penalty works differently, is unknown.

## #81844: blank identifier parameters

`func f(_ int, v []float32)` with simd code in the body was rejected by the compiler with
`cannot use _ as value or type`. loopvec renames blank parameters in the generated simd
file. Once the minimum toolchain has [CL 841145](https://go.dev/cl/841145) the renaming
can be removed.

## #81845 and #80689: simd in a package initializer

A simd call that the compiler inlines into a package-level variable initializer leaves the
link unresolved. The Go team treats it as a known problem
([#80689](https://github.com/golang/go/issues/80689)). loopvec leaves loops in
initializers alone.

## #81846: `Neg` rebuilds a constant

The sign-bit mask for `Float32s.Neg` is materialized inside the loop on every iteration
instead of once before it. Negation loops are correct but slower than a hoisted constant
would make them.

## #81847: the preemption tax

`GOEXPERIMENT=simd` changes `runtime.asyncPreempt` to save the whole vector register file
on AVX-512 machines, even in a binary with no simd code. Loops loopvec does not rewrite
become several times slower and much noisier. Set `GODEBUG=asyncpreemptoff=1` when
comparing a scalar and a simd build; otherwise the rows that were not rewritten look like
regressions.

## #82001: emulated partial loads

With `GODEBUG=simd=0` the emulated `Load{Float64,Int64,Uint64}sPart` return `len(s)`
instead of the number of elements loaded. loopvec's loops advance by that count (the loop
shape the `simd` package documents), so in that mode a copy of ten `float64`s produces
`[1 2 0 0 0 0 0 0 0 0]` instead of the input. The hardware-backed mode gives the right
answer.

## #82002: float `Min` and `Max`

On amd64 `Float32s.Min` and `Max` are the hardware instruction, which returns the second
operand for a NaN and does not order -0 below +0. Go's `min` and `max` propagate the NaN
and order the zeros, and arm64 matches Go. loopvec does not rewrite float `min`/`max`
because the result would depend on the machine.
