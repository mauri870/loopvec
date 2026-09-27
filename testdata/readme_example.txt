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
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(a[_i:])
				_v2, _ := simd.LoadFloat32sPart(b[_i:])
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
-- stderr.want --
loopvec: rewrote 1 loop(s) in $WORK/ops.go
