# The README example, verbatim: "loopvec -split ops.go" on a single file.

loopvec -split ops.go
! stdout .
cmpenv stderr stderr.want
cmp ops.go ops.go.want
cmp ops_simd.go ops_simd.go.want

-- go.mod --
module example.com/ops

go 1.27.1
-- ops.go --
package ops

func AddFloat32s(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- ops.go.want --
//go:build !goexperiment.simd

package ops

func AddFloat32s(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- ops_simd.go.want --
//go:build goexperiment.simd

package ops

import (
	"simd"
	"unsafe"
)

func AddFloat32s(dst, a, b []float32) {
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
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_v2 := simd.LoadFloat32s(_c3)
					_v1.Add(_v2).Store(_c1)
					_v3 := simd.LoadFloat32s(_c2[_lanes:])
					_v4 := simd.LoadFloat32s(_c3[_lanes:])
					_v3.Add(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v6 := simd.LoadFloat32s(_c3[2*_lanes:])
					_v5.Add(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v8 := simd.LoadFloat32s(_c3[3*_lanes:])
					_v7.Add(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadFloat32s(_w2[_i:])
					_v10 := simd.LoadFloat32s(_w3[_i:])
					_v9.Add(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v12, _ := simd.LoadFloat32sPart(_w3[_i:])
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
-- stderr.want --
loopvec: rewrote 1 loop(s) in $WORK/ops.go
