# The invocations from the README usage section, over a multi-package module:
# "loopvec ./...", "-d ./...", "-split ./...", "-w ./...".

# print rewritten source to stdout
loopvec ./...
cmp stdout stdout.want
cmpenv stderr both.stderr
cmp a/ops.go a/ops.go.orig
cmp b/ops.go b/ops.go.orig

# unified diff, once per changed file
loopvec -d ./...
stdout -count=1 '^--- \S+/a/ops\.go\t'
stdout -count=1 '^\+\+\+ \S+/a/ops\.go\t'
stdout -count=1 '^--- \S+/b/ops\.go\t'
stdout -count=1 '^\+\+\+ \S+/b/ops\.go\t'
stdout -count=2 '^@@ -1,7 \+1,14 @@$'
cmpenv stderr both.stderr
cmp a/ops.go a/ops.go.orig
cmp b/ops.go b/ops.go.orig

# split only a sub-tree
loopvec -split ./a/...
! stdout .
cmpenv stderr a.stderr
cmp a/ops.go a/ops.go.guarded
cmp a/ops_simd.go ops_simd.go.want
cmp b/ops.go b/ops.go.orig
! exists b/ops_simd.go

# a is already guarded and is skipped; b is overwritten in place
loopvec -w ./...
! stdout .
cmpenv stderr b.stderr
cmp a/ops.go a/ops.go.guarded
cmp b/ops.go ops_simd.go.want
! exists b/ops_simd.go

-- go.mod --
module example.com/test

go 1.27.1
-- a/ops.go --
package test

func AddInt32s(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- a/ops.go.orig --
package test

func AddInt32s(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- b/ops.go --
package test

func AddInt32s(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- b/ops.go.orig --
package test

func AddInt32s(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- a/ops.go.guarded --
//go:build !goexperiment.simd

package test

func AddInt32s(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- ops_simd.go.want --
//go:build goexperiment.simd

package test

import "simd"

func AddInt32s(dst, a, b []int32) {
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadInt32sPart(a[_i:])
		_v2, _ := simd.LoadInt32sPart(b[_i:])
		_v1.Add(_v2).StorePart(dst[_i:])
		_i += _n
	}
}
-- stdout.want --
//go:build goexperiment.simd

package test

import "simd"

func AddInt32s(dst, a, b []int32) {
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadInt32sPart(a[_i:])
		_v2, _ := simd.LoadInt32sPart(b[_i:])
		_v1.Add(_v2).StorePart(dst[_i:])
		_i += _n
	}
}
//go:build goexperiment.simd

package test

import "simd"

func AddInt32s(dst, a, b []int32) {
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadInt32sPart(a[_i:])
		_v2, _ := simd.LoadInt32sPart(b[_i:])
		_v1.Add(_v2).StorePart(dst[_i:])
		_i += _n
	}
}
-- both.stderr --
loopvec: rewrote 1 loop(s) in $WORK/a/ops.go
loopvec: rewrote 1 loop(s) in $WORK/b/ops.go
-- a.stderr --
loopvec: rewrote 1 loop(s) in $WORK/a/ops.go
-- b.stderr --
loopvec: rewrote 1 loop(s) in $WORK/b/ops.go
