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
stdout -count=2 '^@@ -1,7 \+1,63 @@$'
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
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2, _w3 := a[:_count:_count], b[:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v2 := simd.LoadInt32s(_c3)
					_v1.Add(_v2).Store(_c1)
					_v3 := simd.LoadInt32s(_c2[_lanes:])
					_v4 := simd.LoadInt32s(_c3[_lanes:])
					_v3.Add(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadInt32s(_c2[2*_lanes:])
					_v6 := simd.LoadInt32s(_c3[2*_lanes:])
					_v5.Add(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadInt32s(_c2[3*_lanes:])
					_v8 := simd.LoadInt32s(_c3[3*_lanes:])
					_v7.Add(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadInt32s(_w2[_i:])
					_v10 := simd.LoadInt32s(_w3[_i:])
					_v9.Add(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadInt32sPart(_w2[_i:])
					_v12, _ := simd.LoadInt32sPart(_w3[_i:])
					_v11.Add(_v12).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func _loopvecOverlap[T any](a, b []T, aLo, aHi, bLo, bHi int) bool {
	if aLo >= aHi || bLo >= bHi {
		return false
	}
	size := unsafe.Sizeof(a[0])
	aBase := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bBase := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return aBase+uintptr(aLo)*size < bBase+uintptr(bHi)*size && bBase+uintptr(bLo)*size < aBase+uintptr(aHi)*size
}
-- stdout.want --
//go:build goexperiment.simd

package test

import (
	"simd"
	"unsafe"
)

func AddInt32s(dst, a, b []int32) {
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2, _w3 := a[:_count:_count], b[:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v2 := simd.LoadInt32s(_c3)
					_v1.Add(_v2).Store(_c1)
					_v3 := simd.LoadInt32s(_c2[_lanes:])
					_v4 := simd.LoadInt32s(_c3[_lanes:])
					_v3.Add(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadInt32s(_c2[2*_lanes:])
					_v6 := simd.LoadInt32s(_c3[2*_lanes:])
					_v5.Add(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadInt32s(_c2[3*_lanes:])
					_v8 := simd.LoadInt32s(_c3[3*_lanes:])
					_v7.Add(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadInt32s(_w2[_i:])
					_v10 := simd.LoadInt32s(_w3[_i:])
					_v9.Add(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadInt32sPart(_w2[_i:])
					_v12, _ := simd.LoadInt32sPart(_w3[_i:])
					_v11.Add(_v12).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func _loopvecOverlap[T any](a, b []T, aLo, aHi, bLo, bHi int) bool {
	if aLo >= aHi || bLo >= bHi {
		return false
	}
	size := unsafe.Sizeof(a[0])
	aBase := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bBase := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return aBase+uintptr(aLo)*size < bBase+uintptr(bHi)*size && bBase+uintptr(bLo)*size < aBase+uintptr(aHi)*size
}
//go:build goexperiment.simd

package test

import (
	"simd"
	"unsafe"
)

func AddInt32s(dst, a, b []int32) {
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2, _w3 := a[:_count:_count], b[:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v2 := simd.LoadInt32s(_c3)
					_v1.Add(_v2).Store(_c1)
					_v3 := simd.LoadInt32s(_c2[_lanes:])
					_v4 := simd.LoadInt32s(_c3[_lanes:])
					_v3.Add(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadInt32s(_c2[2*_lanes:])
					_v6 := simd.LoadInt32s(_c3[2*_lanes:])
					_v5.Add(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadInt32s(_c2[3*_lanes:])
					_v8 := simd.LoadInt32s(_c3[3*_lanes:])
					_v7.Add(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadInt32s(_w2[_i:])
					_v10 := simd.LoadInt32s(_w3[_i:])
					_v9.Add(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadInt32sPart(_w2[_i:])
					_v12, _ := simd.LoadInt32sPart(_w3[_i:])
					_v11.Add(_v12).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func _loopvecOverlap[T any](a, b []T, aLo, aHi, bLo, bHi int) bool {
	if aLo >= aHi || bLo >= bHi {
		return false
	}
	size := unsafe.Sizeof(a[0])
	aBase := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bBase := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return aBase+uintptr(aLo)*size < bBase+uintptr(bHi)*size && bBase+uintptr(bLo)*size < aBase+uintptr(aHi)*size
}
-- both.stderr --
loopvec: rewrote 1 loop(s) in $WORK/a/ops.go
loopvec: rewrote 1 loop(s) in $WORK/b/ops.go
-- a.stderr --
loopvec: rewrote 1 loop(s) in $WORK/a/ops.go
-- b.stderr --
loopvec: rewrote 1 loop(s) in $WORK/b/ops.go
