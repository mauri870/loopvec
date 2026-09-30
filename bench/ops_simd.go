//go:build goexperiment.simd

package bench

import (
	"math"
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

func MulFloat32s(dst, a, b []float32) {
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i] * b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(a[_i:])
				_v2, _ := simd.LoadFloat32sPart(b[_i:])
				_v1.Mul(_v2).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func ScalFloat32s(x []float32, c float32) {
	_vcFloat32s2 := simd.BroadcastFloat32s(c)
	for _i := 0; _i < len(x); {
		_v1, _n := simd.LoadFloat32sPart(x[_i:])
		_v1.Mul(_vcFloat32s2).StorePart(x[_i:])
		_i += _n
	}
}

func AxpyFloat32s(dst, a []float32, alpha float32, b []float32) {
	_vcAFloat32s3 := simd.BroadcastFloat32s(alpha)
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i]*alpha + b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(a[_i:])
				_v2, _ := simd.LoadFloat32sPart(b[_i:])
				_v1.MulAdd(_vcAFloat32s3, _v2).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func MixFloat32s(dst, a, b []float32, alpha, beta float32) {
	_vcAFloat32s4 := simd.BroadcastFloat32s(alpha)
	_vcBFloat32s4 := simd.BroadcastFloat32s(beta)
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i]*alpha + b[i]*beta
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(a[_i:])
				_v2, _ := simd.LoadFloat32sPart(b[_i:])
				_v1.MulAdd(_vcAFloat32s4, _v2.Mul(_vcBFloat32s4)).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func NegFloat32s(dst, src []float32) {
	if _loopvecOverlap(dst, src) {
		for i := range dst {
			dst[i] = -src[i]
		}
	} else {
		if len(dst) > 0 {
			_ = src[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(src[_i:])
				_v1.Neg().StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func DivFloat32s(dst, a, b []float32) {
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i] / b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(a[_i:])
				_v2, _ := simd.LoadFloat32sPart(b[_i:])
				_v1.Div(_v2).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func DaxpyFloat32s(dst, a []float32, alpha float32) {
	_vcAFloat32s7 := simd.BroadcastFloat32s(alpha)
	if _loopvecOverlap(dst, a) {
		for i := range dst {
			dst[i] += a[i] * alpha
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(a[_i:])
				_v2, _ := simd.LoadFloat32sPart(dst[_i:])
				_v1.MulAdd(_vcAFloat32s7, _v2).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func FillFloat32s(dst []float32) {
	_vcFloat32s8 := simd.BroadcastFloat32s(2.5)
	for _i := 0; _i < len(dst); {
		_n := _vcFloat32s8.StorePart(dst[_i:])
		_i += _n
	}
}

func FillUint8s(dst []byte) {
	_vcUint8s9 := simd.BroadcastUint8s(7)
	for _i := 0; _i < len(dst); {
		_n := _vcUint8s9.StorePart(dst[_i:])
		_i += _n
	}
}

func ReverseIncFloat32s(dst, src []float32) {
	_vcFloat32s10 := simd.BroadcastFloat32s(1)
	if _loopvecOverlap(dst, src) {
		for i := len(dst) - 1; i >= 0; i-- {
			dst[i] = src[i] + 1
		}
	} else {
		if len(dst) > 0 {
			_ = src[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(src[_i:])
				_v1.Add(_vcFloat32s10).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func CopyFloat32s(dst, src []float32) {
	if _loopvecOverlap(dst, src) {
		for i := range dst {
			dst[i] = src[i]
		}
	} else {
		if len(dst) > 0 {
			_ = src[len(dst)-1]
			copy(dst, src)
		}
	}
}

func AndNotUint64s(dst, a, b []uint64) {
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) {
		for i := range dst {
			dst[i] = a[i] &^ b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadUint64sPart(a[_i:])
				_v2, _ := simd.LoadUint64sPart(b[_i:])
				_v1.AndNot(_v2).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func AbsFloat32s(dst, a []float32) {
	if _loopvecOverlap(dst, a) {
		for i := range dst {
			dst[i] = float32(math.Abs(float64(a[i])))
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(a[_i:])
				_v1.Abs().StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func SqrtFloat32s(dst, a []float32) {
	if _loopvecOverlap(dst, a) {
		for i := range dst {
			dst[i] = float32(math.Sqrt(float64(a[i])))
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadFloat32sPart(a[_i:])
				_v1.Sqrt().StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func ClampInt32s(dst, a []int32, lo, hi int32) {
	_vcAInt32s15 := simd.BroadcastInt32s(lo)
	_vcBInt32s15 := simd.BroadcastInt32s(hi)
	if _loopvecOverlap(dst, a) {
		for i := range dst {
			dst[i] = min(max(a[i], lo), hi)
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadInt32sPart(a[_i:])
				_v1.Max(_vcAInt32s15).Min(_vcBInt32s15).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func ShrInt32s(dst, a []int32, n uint) {
	if _loopvecOverlap(dst, a) {
		for i := range dst {
			dst[i] = a[i] >> n
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadInt32sPart(a[_i:])
				_v1.ShiftAllRight(uint64(n)).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func ShlUint32s(dst, a []uint32, n uint) {
	if _loopvecOverlap(dst, a) {
		for i := range dst {
			dst[i] = a[i] << n
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadUint32sPart(a[_i:])
				_v1.ShiftAllLeft(uint64(n)).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func NormalizeFloat32s(auth, delta []float32, norm float32) {
	_vcFloat32s18_1 := simd.BroadcastFloat32s(norm)
	if _loopvecOverlap(auth, delta) {
		for i := range auth {
			auth[i] /= norm
			delta[i] -= auth[i]
		}
	} else {
		if len(auth) > 0 {
			_ = auth[len(auth)-1]
			_ = delta[len(auth)-1]
			for _i := 0; _i < len(auth); {
				_v1, _n := simd.LoadFloat32sPart(auth[_i:len(auth)])
				_v1.Div(_vcFloat32s18_1).StorePart(auth[_i:len(auth)])
				_v2, _ := simd.LoadFloat32sPart(delta[_i:len(auth)])
				_v3, _ := simd.LoadFloat32sPart(auth[_i:len(auth)])
				_v2.Sub(_v3).StorePart(delta[_i:len(auth)])
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
