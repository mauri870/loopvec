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
stdout -count=2 '^@@ -1,7 \+1,38 @@$'
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

import (
	"simd"
	"unsafe"
)

func AddInt32s(dst, a, b []int32) {
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadInt32sPart(a[_i:])
				_v2, _ := simd.LoadInt32sPart(b[_i:])
				_v1.Add(_v2).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func _loopvecOverlap[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(a[0])
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(b[0])
	return aStart < bEnd && bStart < aEnd
}
-- stdout.want --
//go:build goexperiment.simd

package test

import (
	"simd"
	"unsafe"
)

func AddInt32s(dst, a, b []int32) {
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadInt32sPart(a[_i:])
				_v2, _ := simd.LoadInt32sPart(b[_i:])
				_v1.Add(_v2).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func _loopvecOverlap[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(a[0])
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(b[0])
	return aStart < bEnd && bStart < aEnd
}
//go:build goexperiment.simd

package test

import (
	"simd"
	"unsafe"
)

func AddInt32s(dst, a, b []int32) {
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadInt32sPart(a[_i:])
				_v2, _ := simd.LoadInt32sPart(b[_i:])
				_v1.Add(_v2).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func _loopvecOverlap[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(a[0])
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(b[0])
	return aStart < bEnd && bStart < aEnd
}
-- both.stderr --
loopvec: rewrote 1 loop(s) in $WORK/a/ops.go
loopvec: rewrote 1 loop(s) in $WORK/b/ops.go
-- a.stderr --
loopvec: rewrote 1 loop(s) in $WORK/a/ops.go
-- b.stderr --
loopvec: rewrote 1 loop(s) in $WORK/b/ops.go
