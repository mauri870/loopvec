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

import "simd"

func AddFloat32s(dst, a, b []float32) {
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadFloat32sPart(a[_i:])
		_v2, _ := simd.LoadFloat32sPart(b[_i:])
		_v1.Add(_v2).StorePart(dst[_i:])
		_i += _n
	}
}
-- stderr.want --
loopvec: rewrote 1 loop(s) in $WORK/ops.go
