//go:build goexperiment.simd

package bench

import "simd"

func AddFloat32s(dst, a, b []float32) {
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadFloat32sPart(a[_i:])
		_v2, _ := simd.LoadFloat32sPart(b[_i:])
		_v1.Add(_v2).StorePart(dst[_i:])
		_i += _n
	}
}

func MulFloat32s(dst, a, b []float32) {
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadFloat32sPart(a[_i:])
		_v2, _ := simd.LoadFloat32sPart(b[_i:])
		_v1.Mul(_v2).StorePart(dst[_i:])
		_i += _n
	}
}

func ScalFloat32s(x []float32, c float32) {
	_vcFloat32s := simd.BroadcastFloat32s(c)
	for _i := 0; _i < len(x); {
		_v1, _n := simd.LoadFloat32sPart(x[_i:])
		_v1.Mul(_vcFloat32s).StorePart(x[_i:])
		_i += _n
	}
}

func AxpyFloat32s(dst, a []float32, alpha float32, b []float32) {
	_vcAFloat32s := simd.BroadcastFloat32s(alpha)
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadFloat32sPart(a[_i:])
		_v2, _ := simd.LoadFloat32sPart(b[_i:])
		_v1.MulAdd(_vcAFloat32s, _v2).StorePart(dst[_i:])
		_i += _n
	}
}

func MixFloat32s(dst, a, b []float32, alpha, beta float32) {
	_vcAFloat32s := simd.BroadcastFloat32s(alpha)
	_vcBFloat32s := simd.BroadcastFloat32s(beta)
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadFloat32sPart(a[_i:])
		_v2, _ := simd.LoadFloat32sPart(b[_i:])
		_v1.MulAdd(_vcAFloat32s, _v2.Mul(_vcBFloat32s)).StorePart(dst[_i:])
		_i += _n
	}
}

func NegFloat32s(dst, src []float32) {
	for _i := 0; _i < len(dst); {
		_v1, _n := simd.LoadFloat32sPart(src[_i:])
		_v1.Neg().StorePart(dst[_i:])
		_i += _n
	}
}
