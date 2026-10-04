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
	if _loopvecOverlap(auth, delta, 0, len(auth), 0, len(auth)) {
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

func ChainInt32s(a, b, c, e []int32) {
	if _loopvecOverlap(b, c, 0, len(a), 0, len(a)) || _loopvecOverlap(b, a, 0, len(a), 0, len(a)) || _loopvecOverlap(b, e, 0, len(a), 0, len(a)) || _loopvecOverlap(c, a, 0, len(a), 0, len(a)) || _loopvecOverlap(a, e, 0, len(a), 0, len(a)) {
		for i := range a {
			x := b[i] * c[i]
			a[i] = x + e[i]
			b[i] = x - a[i]
		}
	} else {
		if len(a) > 0 {
			_ = b[len(a)-1]
			_ = c[len(a)-1]
			_ = a[len(a)-1]
			_ = e[len(a)-1]
			for _i := 0; _i < len(a); {
				_v1, _n := simd.LoadInt32sPart(b[_i:len(a)])
				_v2, _ := simd.LoadInt32sPart(c[_i:len(a)])
				_t1 := _v1.Mul(_v2)
				_v3, _ := simd.LoadInt32sPart(e[_i:len(a)])
				_t1.Add(_v3).StorePart(a[_i:len(a)])
				_v4, _ := simd.LoadInt32sPart(a[_i:len(a)])
				_t1.Sub(_v4).StorePart(b[_i:len(a)])
				_i += _n
			}
		}
	}
}

func WideInt32s(dst, a, b, c []int32, k, m int32) {
	_vcInt32s20_1 := simd.BroadcastInt32s(k)
	_vcInt32s20_2 := simd.BroadcastInt32s(k * m)
	if _loopvecOverlap(dst, a) || _loopvecOverlap(dst, b) || _loopvecOverlap(dst, c) {
		for i := range dst {
			dst[i] = a[i]*b[i] + c[i]*k + (k*m - a[i])
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			_ = c[len(dst)-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadInt32sPart(a[_i:])
				_v2, _ := simd.LoadInt32sPart(b[_i:])
				_v3, _ := simd.LoadInt32sPart(c[_i:])
				_v1.Mul(_v2).Add(_v3.Mul(_vcInt32s20_1)).Add(_vcInt32s20_2.Sub(_v1)).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func SumInt32s(a []int32, init int32) int32 {
	sum := init
	{
		_i := 0
		if len(a) >= 64 {
			_a1 := simd.BroadcastInt32s(0)
			_lanes := _a1.Len()
			for ; _i+_lanes <= len(a); _i += _lanes {
				_v1 := simd.LoadInt32s(a[_i:])
				_a1 = _a1.Add(_v1)
			}
			var _buf1 [16]int32
			_a1.Store(_buf1[:])
			for _, _x := range _buf1[:_lanes] {
				sum = sum + _x
			}
		}
		for i := _i; i < len(a); i++ {
			sum += a[i]
		}
	}
	return sum
}

func DotInt32s(a, b []int32) int32 {
	var sum int32
	if len(a) > 0 {
		_ = b[len(a)-1]
		{
			_i := 0
			if len(a) >= 64 {
				_a1 := simd.BroadcastInt32s(0)
				_lanes := _a1.Len()
				for ; _i+_lanes <= len(a); _i += _lanes {
					_v1 := simd.LoadInt32s(a[_i:])
					_v2 := simd.LoadInt32s(b[_i:])
					_a1 = _a1.Add(_v1.Mul(_v2))
				}
				var _buf1 [16]int32
				_a1.Store(_buf1[:])
				for _, _x := range _buf1[:_lanes] {
					sum = sum + _x
				}
			}
			for i := _i; i < len(a); i++ {
				sum += a[i] * b[i]
			}
		}
	}
	return sum
}

func MinUint16s(a []uint16) uint16 {
	m := uint16(65535)
	{
		_i := 0
		if len(a) >= 128 {
			_a1 := simd.BroadcastUint16s(65535)
			_lanes := _a1.Len()
			for ; _i+_lanes <= len(a); _i += _lanes {
				_v1 := simd.LoadUint16s(a[_i:])
				_a1 = _a1.Min(_v1)
			}
			var _buf1 [32]uint16
			_a1.Store(_buf1[:])
			for _, _x := range _buf1[:_lanes] {
				m = min(m, _x)
			}
		}
		for i := _i; i < len(a); i++ {
			m = min(m, a[i])
		}
	}
	return m
}

func SumFloat32s(a []float32) float32 {
	var sum float32
	{
		_i := 0
		if len(a) >= 64 {
			_a1_0 := simd.BroadcastFloat32s(0)
			_a1_1 := simd.BroadcastFloat32s(0)
			_a1_2 := simd.BroadcastFloat32s(0)
			_a1_3 := simd.BroadcastFloat32s(0)
			_lanes := _a1_0.Len()
			for ; _i+4*_lanes <= len(a); _i += 4 * _lanes {
				_v1 := simd.LoadFloat32s(a[_i:])
				_a1_0 = _a1_0.Add(_v1)
				_v2 := simd.LoadFloat32s(a[_i+_lanes:])
				_a1_1 = _a1_1.Add(_v2)
				_v3 := simd.LoadFloat32s(a[_i+2*_lanes:])
				_a1_2 = _a1_2.Add(_v3)
				_v4 := simd.LoadFloat32s(a[_i+3*_lanes:])
				_a1_3 = _a1_3.Add(_v4)
			}
			for ; _i+_lanes <= len(a); _i += _lanes {
				_v5 := simd.LoadFloat32s(a[_i:])
				_a1_0 = _a1_0.Add(_v5)
			}
			_a1_0 = _a1_0.Add(_a1_1)
			_a1_0 = _a1_0.Add(_a1_2)
			_a1_0 = _a1_0.Add(_a1_3)
			var _buf1 [16]float32
			_a1_0.Store(_buf1[:])
			for _, _x := range _buf1[:_lanes] {
				sum = sum + _x
			}
		}
		for i := _i; i < len(a); i++ {
			sum += a[i]
		}
	}
	return sum
}

func DotFloat32s(a, b []float32) float32 {
	var sum float32
	if len(a) > 0 {
		_ = b[len(a)-1]
		{
			_i := 0
			if len(a) >= 64 {
				_a1_0 := simd.BroadcastFloat32s(0)
				_a1_1 := simd.BroadcastFloat32s(0)
				_a1_2 := simd.BroadcastFloat32s(0)
				_a1_3 := simd.BroadcastFloat32s(0)
				_lanes := _a1_0.Len()
				for ; _i+4*_lanes <= len(a); _i += 4 * _lanes {
					_v1 := simd.LoadFloat32s(a[_i:])
					_v2 := simd.LoadFloat32s(b[_i:])
					_a1_0 = _v1.MulAdd(_v2, _a1_0)
					_v3 := simd.LoadFloat32s(a[_i+_lanes:])
					_v4 := simd.LoadFloat32s(b[_i+_lanes:])
					_a1_1 = _v3.MulAdd(_v4, _a1_1)
					_v5 := simd.LoadFloat32s(a[_i+2*_lanes:])
					_v6 := simd.LoadFloat32s(b[_i+2*_lanes:])
					_a1_2 = _v5.MulAdd(_v6, _a1_2)
					_v7 := simd.LoadFloat32s(a[_i+3*_lanes:])
					_v8 := simd.LoadFloat32s(b[_i+3*_lanes:])
					_a1_3 = _v7.MulAdd(_v8, _a1_3)
				}
				for ; _i+_lanes <= len(a); _i += _lanes {
					_v9 := simd.LoadFloat32s(a[_i:])
					_v10 := simd.LoadFloat32s(b[_i:])
					_a1_0 = _v9.MulAdd(_v10, _a1_0)
				}
				_a1_0 = _a1_0.Add(_a1_1)
				_a1_0 = _a1_0.Add(_a1_2)
				_a1_0 = _a1_0.Add(_a1_3)
				var _buf1 [16]float32
				_a1_0.Store(_buf1[:])
				for _, _x := range _buf1[:_lanes] {
					sum = sum + _x
				}
			}
			for i := _i; i < len(a); i++ {
				sum += a[i] * b[i]
			}
		}
	}
	return sum
}

func ProductFloat32s(a []float32) float32 {
	prod := float32(1)
	{
		_i := 0
		if len(a) >= 64 {
			_a1_0 := simd.BroadcastFloat32s(1)
			_a1_1 := simd.BroadcastFloat32s(1)
			_a1_2 := simd.BroadcastFloat32s(1)
			_a1_3 := simd.BroadcastFloat32s(1)
			_lanes := _a1_0.Len()
			for ; _i+4*_lanes <= len(a); _i += 4 * _lanes {
				_v1 := simd.LoadFloat32s(a[_i:])
				_a1_0 = _a1_0.Mul(_v1)
				_v2 := simd.LoadFloat32s(a[_i+_lanes:])
				_a1_1 = _a1_1.Mul(_v2)
				_v3 := simd.LoadFloat32s(a[_i+2*_lanes:])
				_a1_2 = _a1_2.Mul(_v3)
				_v4 := simd.LoadFloat32s(a[_i+3*_lanes:])
				_a1_3 = _a1_3.Mul(_v4)
			}
			for ; _i+_lanes <= len(a); _i += _lanes {
				_v5 := simd.LoadFloat32s(a[_i:])
				_a1_0 = _a1_0.Mul(_v5)
			}
			_a1_0 = _a1_0.Mul(_a1_1)
			_a1_0 = _a1_0.Mul(_a1_2)
			_a1_0 = _a1_0.Mul(_a1_3)
			var _buf1 [16]float32
			_a1_0.Store(_buf1[:])
			for _, _x := range _buf1[:_lanes] {
				prod = prod * _x
			}
		}
		for i := _i; i < len(a); i++ {
			prod *= a[i]
		}
	}
	return prod
}

func DotFloat64s(a, b []float64) float64 {
	var sum float64
	if len(a) > 0 {
		_ = b[len(a)-1]
		{
			_i := 0
			if len(a) >= 32 {
				_a1_0 := simd.BroadcastFloat64s(0)
				_a1_1 := simd.BroadcastFloat64s(0)
				_a1_2 := simd.BroadcastFloat64s(0)
				_a1_3 := simd.BroadcastFloat64s(0)
				_lanes := _a1_0.Len()
				for ; _i+4*_lanes <= len(a); _i += 4 * _lanes {
					_v1 := simd.LoadFloat64s(a[_i:])
					_v2 := simd.LoadFloat64s(b[_i:])
					_a1_0 = _v1.MulAdd(_v2, _a1_0)
					_v3 := simd.LoadFloat64s(a[_i+_lanes:])
					_v4 := simd.LoadFloat64s(b[_i+_lanes:])
					_a1_1 = _v3.MulAdd(_v4, _a1_1)
					_v5 := simd.LoadFloat64s(a[_i+2*_lanes:])
					_v6 := simd.LoadFloat64s(b[_i+2*_lanes:])
					_a1_2 = _v5.MulAdd(_v6, _a1_2)
					_v7 := simd.LoadFloat64s(a[_i+3*_lanes:])
					_v8 := simd.LoadFloat64s(b[_i+3*_lanes:])
					_a1_3 = _v7.MulAdd(_v8, _a1_3)
				}
				for ; _i+_lanes <= len(a); _i += _lanes {
					_v9 := simd.LoadFloat64s(a[_i:])
					_v10 := simd.LoadFloat64s(b[_i:])
					_a1_0 = _v9.MulAdd(_v10, _a1_0)
				}
				_a1_0 = _a1_0.Add(_a1_1)
				_a1_0 = _a1_0.Add(_a1_2)
				_a1_0 = _a1_0.Add(_a1_3)
				var _buf1 [8]float64
				_a1_0.Store(_buf1[:])
				for _, _x := range _buf1[:_lanes] {
					sum = sum + _x
				}
			}
			for i := _i; i < len(a); i++ {
				sum += a[i] * b[i]
			}
		}
	}
	return sum
}

func AddWindowFloat32s(dst, a, b []float32, lo, hi int) {
	if _loopvecOverlap(dst, a, lo, hi, lo, hi) || _loopvecOverlap(dst, b, lo, hi, lo, hi) {
		for i := lo; i < hi; i++ {
			dst[i] = a[i] + b[i]
		}
	} else {
		if hi > lo {
			_ = dst[hi-1]
			_ = a[hi-1]
			_ = b[hi-1]
			for _i := lo; _i < hi; {
				_v1, _n := simd.LoadFloat32sPart(a[_i:hi])
				_v2, _ := simd.LoadFloat32sPart(b[_i:hi])
				_v1.Add(_v2).StorePart(dst[_i:hi])
				_i += _n
			}
		}
	}
}

func ScaleInnerFloat32s(dst, a []float32, k float32) {
	_vcFloat32s29 := simd.BroadcastFloat32s(k)
	if _loopvecOverlap(dst, a, 1, len(dst)-1, 1, len(dst)-1) {
		for i := 1; i < len(dst)-1; i++ {
			dst[i] = a[i] * k
		}
	} else {
		if len(dst)-1 > 1 {
			_ = dst[(len(dst)-1)-1]
			_ = a[(len(dst)-1)-1]
			for _i := 1; _i < len(dst)-1; {
				_v1, _n := simd.LoadFloat32sPart(a[_i : len(dst)-1])
				_v1.Mul(_vcFloat32s29).StorePart(dst[_i : len(dst)-1])
				_i += _n
			}
		}
	}
}

func IncDownFloat32s(dst, a []float32, hi, lo int) {
	if _loopvecOverlap(dst, a, lo, hi, lo, hi) {
		for i := hi - 1; i >= lo; i-- {
			dst[i] += a[i]
		}
	} else {
		if hi > lo {
			_ = dst[hi-1]
			_ = a[hi-1]
			for _i := lo; _i < hi; {
				_v1, _n := simd.LoadFloat32sPart(dst[_i:hi])
				_v2, _ := simd.LoadFloat32sPart(a[_i:hi])
				_v1.Add(_v2).StorePart(dst[_i:hi])
				_i += _n
			}
		}
	}
}

func DiffInt32s(dst, a []int32) {
	if _loopvecOverlap(dst, a, 0, len(dst)-1, 0, len(dst)-1+1) {
		for i := 0; i < len(dst)-1; i++ {
			dst[i] = a[i+1] - a[i]
		}
	} else {
		if len(dst)-1 > 0 {
			_ = dst[(len(dst)-1)-1]
			_ = a[len(dst)-1]
			_ = a[(len(dst)-1)-1]
			for _i := 0; _i < len(dst)-1; {
				_v1, _n := simd.LoadInt32sPart(a[_i+1:])
				_v2, _ := simd.LoadInt32sPart(a[_i : len(dst)-1])
				_v1.Sub(_v2).StorePart(dst[_i : len(dst)-1])
				_i += _n
			}
		}
	}
}

func SmoothInt32s(dst, a []int32) {
	if _loopvecOverlap(dst, a, 1, len(dst)-1, 0, len(dst)-1+1) {
		for i := 1; i < len(dst)-1; i++ {
			dst[i] = a[i-1] + a[i] + a[i+1]
		}
	} else {
		if len(dst)-1 > 1 {
			_ = dst[(len(dst)-1)-1]
			_ = a[(len(dst)-1)-2]
			_ = a[(len(dst)-1)-1]
			_ = a[len(dst)-1]
			for _i := 1; _i < len(dst)-1; {
				_v1, _n := simd.LoadInt32sPart(a[_i-1:])
				_v2, _ := simd.LoadInt32sPart(a[_i : len(dst)-1])
				_v3, _ := simd.LoadInt32sPart(a[_i+1:])
				_v1.Add(_v2).Add(_v3).StorePart(dst[_i : len(dst)-1])
				_i += _n
			}
		}
	}
}

func ShiftInt32s(dst, a []int32, off int) {
	_vcInt32s33 := simd.BroadcastInt32s(3)
	if _loopvecOverlap(dst, a) {
		for i := range dst {
			dst[i] = a[i+off] * 3
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)+off-1]
			for _i := 0; _i < len(dst); {
				_v1, _n := simd.LoadInt32sPart(a[_i+off:])
				_v1.Mul(_vcInt32s33).StorePart(dst[_i:])
				_i += _n
			}
		}
	}
}

func AccumRowInt32s(work, a []int32, row, lda int) {
	if _loopvecOverlap(work, a) {
		for j := range work {
			work[j] += a[row*lda+j]
		}
	} else {
		if len(work) > 0 {
			_ = a[len(work)+(row*lda)-1]
			for _i := 0; _i < len(work); {
				_v1, _n := simd.LoadInt32sPart(work[_i:])
				_v2, _ := simd.LoadInt32sPart(a[_i+(row*lda):])
				_v1.Add(_v2).StorePart(work[_i:])
				_i += _n
			}
		}
	}
}

func ShiftAddInt32s(a, b []int32) {
	if _loopvecOverlap(a, b, 0, len(a)-1+1, 0, len(a)-1) {
		for i := 0; i < len(a)-1; i++ {
			a[i] = a[i+1] + b[i]
		}
	} else {
		if len(a)-1 > 0 {
			_ = a[(len(a)-1)-1]
			_ = a[len(a)-1]
			_ = b[(len(a)-1)-1]
			for _i := 0; _i < len(a)-1; {
				_v1, _n := simd.LoadInt32sPart(a[_i+1:])
				_v2, _ := simd.LoadInt32sPart(b[_i : len(a)-1])
				_v1.Add(_v2).StorePart(a[_i : len(a)-1])
				_i += _n
			}
		}
	}
}

func ReadAfterStoreInt32s(a, b, d []int32) {
	_vcInt32s36_1 := simd.BroadcastInt32s(2)
	if _loopvecOverlap(a, b, 0, len(a)-1+1, 0, len(a)-1) || _loopvecOverlap(a, d, 0, len(a)-1+1, 0, len(a)-1) || _loopvecOverlap(b, d, 0, len(a)-1, 0, len(a)-1) {
		for i := 0; i < len(a)-1; i++ {
			a[i] = b[i] + d[i]
			b[i] = a[i] * 2
			a[i] = b[i] + a[i+1]*d[i]
		}
	} else {
		if len(a)-1 > 0 {
			_ = a[(len(a)-1)-1]
			_ = b[(len(a)-1)-1]
			_ = d[(len(a)-1)-1]
			_ = a[len(a)-1]
			for _i := 0; _i < len(a)-1; {
				_v1, _n := simd.LoadInt32sPart(a[_i+1:])
				_v2, _ := simd.LoadInt32sPart(b[_i : len(a)-1])
				_v3, _ := simd.LoadInt32sPart(d[_i : len(a)-1])
				_v2.Add(_v3).StorePart(a[_i : len(a)-1])
				_v4, _ := simd.LoadInt32sPart(a[_i : len(a)-1])
				_v4.Mul(_vcInt32s36_1).StorePart(b[_i : len(a)-1])
				_v5, _ := simd.LoadInt32sPart(b[_i : len(a)-1])
				_v1.Mul(_v3).Add(_v5).StorePart(a[_i : len(a)-1])
				_i += _n
			}
		}
	}
}

func _loopvecOverlap[T any](a, b []T, window ...int) bool {
	aLo, aHi, bLo, bHi := 0, len(a), 0, len(b)
	if len(window) == 4 {
		aLo, aHi, bLo, bHi = window[0], window[1], window[2], window[3]
	}
	if aLo >= aHi || bLo >= bHi {
		return false
	}
	size := unsafe.Sizeof(a[0])
	aBase := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bBase := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	aStart, aEnd := aBase+uintptr(aLo)*size, aBase+uintptr(aHi)*size
	bStart, bEnd := bBase+uintptr(bLo)*size, bBase+uintptr(bHi)*size
	return aStart < bEnd && bStart < aEnd
}
