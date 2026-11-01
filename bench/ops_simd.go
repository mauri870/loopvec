//go:build goexperiment.simd

package bench

import (
	"math"
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

func MulFloat32s(dst, a, b []float32) {
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) {
		for i := range dst {
			dst[i] = a[i] * b[i]
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
					_v1.Mul(_v2).Store(_c1)
					_v3 := simd.LoadFloat32s(_c2[_lanes:])
					_v4 := simd.LoadFloat32s(_c3[_lanes:])
					_v3.Mul(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v6 := simd.LoadFloat32s(_c3[2*_lanes:])
					_v5.Mul(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v8 := simd.LoadFloat32s(_c3[3*_lanes:])
					_v7.Mul(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadFloat32s(_w2[_i:])
					_v10 := simd.LoadFloat32s(_w3[_i:])
					_v9.Mul(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v12, _ := simd.LoadFloat32sPart(_w3[_i:])
					_v11.Mul(_v12).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func ScalFloat32s(x []float32, c float32) {
	_vcFloat32s2 := simd.BroadcastFloat32s(c)
	{
		_w1 := x[:len(x):len(x)]
		_count := len(_w1)
		_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
		for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
			_c1 := _w1[_i:][:4*_lanes]
			_v1 := simd.LoadFloat32s(_c1)
			_v1.Mul(_vcFloat32s2).Store(_c1)
			_v2 := simd.LoadFloat32s(_c1[_lanes:])
			_v2.Mul(_vcFloat32s2).Store(_c1[_lanes:])
			_v3 := simd.LoadFloat32s(_c1[2*_lanes:])
			_v3.Mul(_vcFloat32s2).Store(_c1[2*_lanes:])
			_v4 := simd.LoadFloat32s(_c1[3*_lanes:])
			_v4.Mul(_vcFloat32s2).Store(_c1[3*_lanes:])
		}
		for ; _i <= _count-_lanes; _i += _lanes {
			_v5 := simd.LoadFloat32s(_w1[_i:])
			_v5.Mul(_vcFloat32s2).Store(_w1[_i:])
		}
		for _i < _count {
			_v6, _n := simd.LoadFloat32sPart(_w1[_i:])
			_v6.Mul(_vcFloat32s2).StorePart(_w1[_i:])
			_i += _n
		}
	}
}

func AxpyFloat32s(dst, a []float32, alpha float32, b []float32) {
	_vcAFloat32s3 := simd.BroadcastFloat32s(alpha)
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) {
		for i := range dst {
			dst[i] = a[i]*alpha + b[i]
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
					_v1.MulAdd(_vcAFloat32s3, _v2).Store(_c1)
					_v3 := simd.LoadFloat32s(_c2[_lanes:])
					_v4 := simd.LoadFloat32s(_c3[_lanes:])
					_v3.MulAdd(_vcAFloat32s3, _v4).Store(_c1[_lanes:])
					_v5 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v6 := simd.LoadFloat32s(_c3[2*_lanes:])
					_v5.MulAdd(_vcAFloat32s3, _v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v8 := simd.LoadFloat32s(_c3[3*_lanes:])
					_v7.MulAdd(_vcAFloat32s3, _v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadFloat32s(_w2[_i:])
					_v10 := simd.LoadFloat32s(_w3[_i:])
					_v9.MulAdd(_vcAFloat32s3, _v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v12, _ := simd.LoadFloat32sPart(_w3[_i:])
					_v11.MulAdd(_vcAFloat32s3, _v12).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func MixFloat32s(dst, a, b []float32, alpha, beta float32) {
	_vcAFloat32s4 := simd.BroadcastFloat32s(alpha)
	_vcBFloat32s4 := simd.BroadcastFloat32s(beta)
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) {
		for i := range dst {
			dst[i] = a[i]*alpha + b[i]*beta
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
					_v1.MulAdd(_vcAFloat32s4, _v2.Mul(_vcBFloat32s4)).Store(_c1)
					_v3 := simd.LoadFloat32s(_c2[_lanes:])
					_v4 := simd.LoadFloat32s(_c3[_lanes:])
					_v3.MulAdd(_vcAFloat32s4, _v4.Mul(_vcBFloat32s4)).Store(_c1[_lanes:])
					_v5 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v6 := simd.LoadFloat32s(_c3[2*_lanes:])
					_v5.MulAdd(_vcAFloat32s4, _v6.Mul(_vcBFloat32s4)).Store(_c1[2*_lanes:])
					_v7 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v8 := simd.LoadFloat32s(_c3[3*_lanes:])
					_v7.MulAdd(_vcAFloat32s4, _v8.Mul(_vcBFloat32s4)).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadFloat32s(_w2[_i:])
					_v10 := simd.LoadFloat32s(_w3[_i:])
					_v9.MulAdd(_vcAFloat32s4, _v10.Mul(_vcBFloat32s4)).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v12, _ := simd.LoadFloat32sPart(_w3[_i:])
					_v11.MulAdd(_vcAFloat32s4, _v12.Mul(_vcBFloat32s4)).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func NegFloat32s(dst, src []float32) {
	if _loopvecOverlap(dst, src, 0, len(dst), 0, len(src)) {
		for i := range dst {
			dst[i] = -src[i]
		}
	} else {
		if len(dst) > 0 {
			_ = src[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := src[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_v1.Neg().Store(_c1)
					_v2 := simd.LoadFloat32s(_c2[_lanes:])
					_v2.Neg().Store(_c1[_lanes:])
					_v3 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v3.Neg().Store(_c1[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v4.Neg().Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w2[_i:])
					_v5.Neg().Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v6.Neg().StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func DivFloat32s(dst, a, b []float32) {
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) {
		for i := range dst {
			dst[i] = a[i] / b[i]
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
					_v1.Div(_v2).Store(_c1)
					_v3 := simd.LoadFloat32s(_c2[_lanes:])
					_v4 := simd.LoadFloat32s(_c3[_lanes:])
					_v3.Div(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v6 := simd.LoadFloat32s(_c3[2*_lanes:])
					_v5.Div(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v8 := simd.LoadFloat32s(_c3[3*_lanes:])
					_v7.Div(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadFloat32s(_w2[_i:])
					_v10 := simd.LoadFloat32s(_w3[_i:])
					_v9.Div(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v12, _ := simd.LoadFloat32sPart(_w3[_i:])
					_v11.Div(_v12).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func DaxpyFloat32s(dst, a []float32, alpha float32) {
	_vcAFloat32s7 := simd.BroadcastFloat32s(alpha)
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) {
		for i := range dst {
			dst[i] += a[i] * alpha
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := a[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_v2 := simd.LoadFloat32s(_c1)
					_v1.MulAdd(_vcAFloat32s7, _v2).Store(_c1)
					_v3 := simd.LoadFloat32s(_c2[_lanes:])
					_v4 := simd.LoadFloat32s(_c1[_lanes:])
					_v3.MulAdd(_vcAFloat32s7, _v4).Store(_c1[_lanes:])
					_v5 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v6 := simd.LoadFloat32s(_c1[2*_lanes:])
					_v5.MulAdd(_vcAFloat32s7, _v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v8 := simd.LoadFloat32s(_c1[3*_lanes:])
					_v7.MulAdd(_vcAFloat32s7, _v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadFloat32s(_w2[_i:])
					_v10 := simd.LoadFloat32s(_w1[_i:])
					_v9.MulAdd(_vcAFloat32s7, _v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v12, _ := simd.LoadFloat32sPart(_w1[_i:])
					_v11.MulAdd(_vcAFloat32s7, _v12).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func FillFloat32s(dst []float32) {
	_vcFloat32s8 := simd.BroadcastFloat32s(2.5)
	{
		_w1 := dst[:len(dst):len(dst)]
		_count := len(_w1)
		_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
		for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
			_c1 := _w1[_i:][:4*_lanes]
			_vcFloat32s8.Store(_c1)
			_vcFloat32s8.Store(_c1[_lanes:])
			_vcFloat32s8.Store(_c1[2*_lanes:])
			_vcFloat32s8.Store(_c1[3*_lanes:])
		}
		for ; _i <= _count-_lanes; _i += _lanes {
			_vcFloat32s8.Store(_w1[_i:])
		}
		for _i < _count {
			_n := _vcFloat32s8.StorePart(_w1[_i:])
			_i += _n
		}
	}
}

func FillUint8s(dst []byte) {
	_vcUint8s9 := simd.BroadcastUint8s(7)
	{
		_w1 := dst[:len(dst):len(dst)]
		_count := len(_w1)
		_i, _lanes := 0, simd.BroadcastUint8s(0).Len()
		for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
			_c1 := _w1[_i:][:4*_lanes]
			_vcUint8s9.Store(_c1)
			_vcUint8s9.Store(_c1[_lanes:])
			_vcUint8s9.Store(_c1[2*_lanes:])
			_vcUint8s9.Store(_c1[3*_lanes:])
		}
		for ; _i <= _count-_lanes; _i += _lanes {
			_vcUint8s9.Store(_w1[_i:])
		}
		for _i < _count {
			_n := _vcUint8s9.StorePart(_w1[_i:])
			_i += _n
		}
	}
}

func ReverseIncFloat32s(dst, src []float32) {
	_vcFloat32s10 := simd.BroadcastFloat32s(1)
	if _loopvecOverlap(dst, src, 0, len(dst), 0, len(src)) {
		for i := len(dst) - 1; i >= 0; i-- {
			dst[i] = src[i] + 1
		}
	} else {
		if len(dst) > 0 {
			_ = src[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := src[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_v1.Add(_vcFloat32s10).Store(_c1)
					_v2 := simd.LoadFloat32s(_c2[_lanes:])
					_v2.Add(_vcFloat32s10).Store(_c1[_lanes:])
					_v3 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v3.Add(_vcFloat32s10).Store(_c1[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v4.Add(_vcFloat32s10).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w2[_i:])
					_v5.Add(_vcFloat32s10).Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v6.Add(_vcFloat32s10).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func CopyFloat32s(dst, src []float32) {
	if _loopvecOverlap(dst, src, 0, len(dst), 0, len(src)) {
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
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) {
		for i := range dst {
			dst[i] = a[i] &^ b[i]
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2, _w3 := a[:_count:_count], b[:_count:_count]
				_i, _lanes := 0, simd.BroadcastUint64s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes]
					_v1 := simd.LoadUint64s(_c2)
					_v2 := simd.LoadUint64s(_c3)
					_v1.AndNot(_v2).Store(_c1)
					_v3 := simd.LoadUint64s(_c2[_lanes:])
					_v4 := simd.LoadUint64s(_c3[_lanes:])
					_v3.AndNot(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadUint64s(_c2[2*_lanes:])
					_v6 := simd.LoadUint64s(_c3[2*_lanes:])
					_v5.AndNot(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadUint64s(_c2[3*_lanes:])
					_v8 := simd.LoadUint64s(_c3[3*_lanes:])
					_v7.AndNot(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadUint64s(_w2[_i:])
					_v10 := simd.LoadUint64s(_w3[_i:])
					_v9.AndNot(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadUint64sPart(_w2[_i:])
					_v12, _ := simd.LoadUint64sPart(_w3[_i:])
					_v11.AndNot(_v12).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func AbsFloat32s(dst, a []float32) {
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) {
		for i := range dst {
			dst[i] = float32(math.Abs(float64(a[i])))
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := a[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_v1.Abs().Store(_c1)
					_v2 := simd.LoadFloat32s(_c2[_lanes:])
					_v2.Abs().Store(_c1[_lanes:])
					_v3 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v3.Abs().Store(_c1[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v4.Abs().Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w2[_i:])
					_v5.Abs().Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v6.Abs().StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func SqrtFloat32s(dst, a []float32) {
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) {
		for i := range dst {
			dst[i] = float32(math.Sqrt(float64(a[i])))
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := a[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_v1.Sqrt().Store(_c1)
					_v2 := simd.LoadFloat32s(_c2[_lanes:])
					_v2.Sqrt().Store(_c1[_lanes:])
					_v3 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v3.Sqrt().Store(_c1[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v4.Sqrt().Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w2[_i:])
					_v5.Sqrt().Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v6.Sqrt().StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func ClampInt32s(dst, a []int32, lo, hi int32) {
	_vcAInt32s15 := simd.BroadcastInt32s(lo)
	_vcBInt32s15 := simd.BroadcastInt32s(hi)
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) {
		for i := range dst {
			dst[i] = min(max(a[i], lo), hi)
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := a[:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v1.Max(_vcAInt32s15).Min(_vcBInt32s15).Store(_c1)
					_v2 := simd.LoadInt32s(_c2[_lanes:])
					_v2.Max(_vcAInt32s15).Min(_vcBInt32s15).Store(_c1[_lanes:])
					_v3 := simd.LoadInt32s(_c2[2*_lanes:])
					_v3.Max(_vcAInt32s15).Min(_vcBInt32s15).Store(_c1[2*_lanes:])
					_v4 := simd.LoadInt32s(_c2[3*_lanes:])
					_v4.Max(_vcAInt32s15).Min(_vcBInt32s15).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadInt32s(_w2[_i:])
					_v5.Max(_vcAInt32s15).Min(_vcBInt32s15).Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadInt32sPart(_w2[_i:])
					_v6.Max(_vcAInt32s15).Min(_vcBInt32s15).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func ShrInt32s(dst, a []int32, n uint) {
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) {
		for i := range dst {
			dst[i] = a[i] >> n
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := a[:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v1.ShiftAllRight(uint64(n)).Store(_c1)
					_v2 := simd.LoadInt32s(_c2[_lanes:])
					_v2.ShiftAllRight(uint64(n)).Store(_c1[_lanes:])
					_v3 := simd.LoadInt32s(_c2[2*_lanes:])
					_v3.ShiftAllRight(uint64(n)).Store(_c1[2*_lanes:])
					_v4 := simd.LoadInt32s(_c2[3*_lanes:])
					_v4.ShiftAllRight(uint64(n)).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadInt32s(_w2[_i:])
					_v5.ShiftAllRight(uint64(n)).Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadInt32sPart(_w2[_i:])
					_v6.ShiftAllRight(uint64(n)).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func ShlUint32s(dst, a []uint32, n uint) {
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) {
		for i := range dst {
			dst[i] = a[i] << n
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := a[:_count:_count]
				_i, _lanes := 0, simd.BroadcastUint32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadUint32s(_c2)
					_v1.ShiftAllLeft(uint64(n)).Store(_c1)
					_v2 := simd.LoadUint32s(_c2[_lanes:])
					_v2.ShiftAllLeft(uint64(n)).Store(_c1[_lanes:])
					_v3 := simd.LoadUint32s(_c2[2*_lanes:])
					_v3.ShiftAllLeft(uint64(n)).Store(_c1[2*_lanes:])
					_v4 := simd.LoadUint32s(_c2[3*_lanes:])
					_v4.ShiftAllLeft(uint64(n)).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadUint32s(_w2[_i:])
					_v5.ShiftAllLeft(uint64(n)).Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadUint32sPart(_w2[_i:])
					_v6.ShiftAllLeft(uint64(n)).StorePart(_w1[_i:])
					_i += _n
				}
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
			{
				_w1 := auth[:len(auth):len(auth)]
				_count := len(_w1)
				_w2 := delta[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c1)
					_v1.Div(_vcFloat32s18_1).Store(_c1)
					_v2 := simd.LoadFloat32s(_c2)
					_v3 := simd.LoadFloat32s(_c1)
					_v2.Sub(_v3).Store(_c2)
					_v4 := simd.LoadFloat32s(_c1[_lanes:])
					_v4.Div(_vcFloat32s18_1).Store(_c1[_lanes:])
					_v5 := simd.LoadFloat32s(_c2[_lanes:])
					_v6 := simd.LoadFloat32s(_c1[_lanes:])
					_v5.Sub(_v6).Store(_c2[_lanes:])
					_v7 := simd.LoadFloat32s(_c1[2*_lanes:])
					_v7.Div(_vcFloat32s18_1).Store(_c1[2*_lanes:])
					_v8 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v9 := simd.LoadFloat32s(_c1[2*_lanes:])
					_v8.Sub(_v9).Store(_c2[2*_lanes:])
					_v10 := simd.LoadFloat32s(_c1[3*_lanes:])
					_v10.Div(_vcFloat32s18_1).Store(_c1[3*_lanes:])
					_v11 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v12 := simd.LoadFloat32s(_c1[3*_lanes:])
					_v11.Sub(_v12).Store(_c2[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v13 := simd.LoadFloat32s(_w1[_i:])
					_v13.Div(_vcFloat32s18_1).Store(_w1[_i:])
					_v14 := simd.LoadFloat32s(_w2[_i:])
					_v15 := simd.LoadFloat32s(_w1[_i:])
					_v14.Sub(_v15).Store(_w2[_i:])
				}
				for _i < _count {
					_v16, _n := simd.LoadFloat32sPart(_w1[_i:])
					_v16.Div(_vcFloat32s18_1).StorePart(_w1[_i:])
					_v17, _ := simd.LoadFloat32sPart(_w2[_i:])
					_v18, _ := simd.LoadFloat32sPart(_w1[_i:])
					_v17.Sub(_v18).StorePart(_w2[_i:])
					_i += _n
				}
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
			{
				_w1 := b[:len(a):len(a)]
				_count := len(_w1)
				_w2, _w3, _w4 := c[:_count:_count], a[:_count:_count], e[:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3, _c4 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes], _w4[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c1)
					_v2 := simd.LoadInt32s(_c2)
					_t1 := _v1.Mul(_v2)
					_v3 := simd.LoadInt32s(_c4)
					_t1.Add(_v3).Store(_c3)
					_v4 := simd.LoadInt32s(_c3)
					_t1.Sub(_v4).Store(_c1)
					_v5 := simd.LoadInt32s(_c1[_lanes:])
					_v6 := simd.LoadInt32s(_c2[_lanes:])
					_t2 := _v5.Mul(_v6)
					_v7 := simd.LoadInt32s(_c4[_lanes:])
					_t2.Add(_v7).Store(_c3[_lanes:])
					_v8 := simd.LoadInt32s(_c3[_lanes:])
					_t2.Sub(_v8).Store(_c1[_lanes:])
					_v9 := simd.LoadInt32s(_c1[2*_lanes:])
					_v10 := simd.LoadInt32s(_c2[2*_lanes:])
					_t3 := _v9.Mul(_v10)
					_v11 := simd.LoadInt32s(_c4[2*_lanes:])
					_t3.Add(_v11).Store(_c3[2*_lanes:])
					_v12 := simd.LoadInt32s(_c3[2*_lanes:])
					_t3.Sub(_v12).Store(_c1[2*_lanes:])
					_v13 := simd.LoadInt32s(_c1[3*_lanes:])
					_v14 := simd.LoadInt32s(_c2[3*_lanes:])
					_t4 := _v13.Mul(_v14)
					_v15 := simd.LoadInt32s(_c4[3*_lanes:])
					_t4.Add(_v15).Store(_c3[3*_lanes:])
					_v16 := simd.LoadInt32s(_c3[3*_lanes:])
					_t4.Sub(_v16).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v17 := simd.LoadInt32s(_w1[_i:])
					_v18 := simd.LoadInt32s(_w2[_i:])
					_t5 := _v17.Mul(_v18)
					_v19 := simd.LoadInt32s(_w4[_i:])
					_t5.Add(_v19).Store(_w3[_i:])
					_v20 := simd.LoadInt32s(_w3[_i:])
					_t5.Sub(_v20).Store(_w1[_i:])
				}
				for _i < _count {
					_v21, _n := simd.LoadInt32sPart(_w1[_i:])
					_v22, _ := simd.LoadInt32sPart(_w2[_i:])
					_t6 := _v21.Mul(_v22)
					_v23, _ := simd.LoadInt32sPart(_w4[_i:])
					_t6.Add(_v23).StorePart(_w3[_i:])
					_v24, _ := simd.LoadInt32sPart(_w3[_i:])
					_t6.Sub(_v24).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func WideInt32s(dst, a, b, c []int32, k, m int32) {
	_vcInt32s20_1 := simd.BroadcastInt32s(k)
	_vcInt32s20_2 := simd.BroadcastInt32s(k * m)
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) || _loopvecOverlap(dst, b, 0, len(dst), 0, len(b)) || _loopvecOverlap(dst, c, 0, len(dst), 0, len(c)) {
		for i := range dst {
			dst[i] = a[i]*b[i] + c[i]*k + (k*m - a[i])
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)-1]
			_ = b[len(dst)-1]
			_ = c[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2, _w3, _w4 := a[:_count:_count], b[:_count:_count], c[:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3, _c4 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes], _w4[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v2 := simd.LoadInt32s(_c3)
					_v3 := simd.LoadInt32s(_c4)
					_v1.Mul(_v2).Add(_v3.Mul(_vcInt32s20_1)).Add(_vcInt32s20_2.Sub(_v1)).Store(_c1)
					_v4 := simd.LoadInt32s(_c2[_lanes:])
					_v5 := simd.LoadInt32s(_c3[_lanes:])
					_v6 := simd.LoadInt32s(_c4[_lanes:])
					_v4.Mul(_v5).Add(_v6.Mul(_vcInt32s20_1)).Add(_vcInt32s20_2.Sub(_v4)).Store(_c1[_lanes:])
					_v7 := simd.LoadInt32s(_c2[2*_lanes:])
					_v8 := simd.LoadInt32s(_c3[2*_lanes:])
					_v9 := simd.LoadInt32s(_c4[2*_lanes:])
					_v7.Mul(_v8).Add(_v9.Mul(_vcInt32s20_1)).Add(_vcInt32s20_2.Sub(_v7)).Store(_c1[2*_lanes:])
					_v10 := simd.LoadInt32s(_c2[3*_lanes:])
					_v11 := simd.LoadInt32s(_c3[3*_lanes:])
					_v12 := simd.LoadInt32s(_c4[3*_lanes:])
					_v10.Mul(_v11).Add(_v12.Mul(_vcInt32s20_1)).Add(_vcInt32s20_2.Sub(_v10)).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v13 := simd.LoadInt32s(_w2[_i:])
					_v14 := simd.LoadInt32s(_w3[_i:])
					_v15 := simd.LoadInt32s(_w4[_i:])
					_v13.Mul(_v14).Add(_v15.Mul(_vcInt32s20_1)).Add(_vcInt32s20_2.Sub(_v13)).Store(_w1[_i:])
				}
				for _i < _count {
					_v16, _n := simd.LoadInt32sPart(_w2[_i:])
					_v17, _ := simd.LoadInt32sPart(_w3[_i:])
					_v18, _ := simd.LoadInt32sPart(_w4[_i:])
					_v16.Mul(_v17).Add(_v18.Mul(_vcInt32s20_1)).Add(_vcInt32s20_2.Sub(_v16)).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func SumInt32s(a []int32, init int32) int32 {
	sum := init
	{
		_i := 0
		if len(a) >= 64 {
			_a1_0 := simd.BroadcastInt32s(0)
			_a1_1 := simd.BroadcastInt32s(0)
			_a1_2 := simd.BroadcastInt32s(0)
			_a1_3 := simd.BroadcastInt32s(0)
			_lanes := _a1_0.Len()
			for ; _i <= len(a)-4*_lanes; _i += 4 * _lanes {
				_v1 := simd.LoadInt32s(a[_i:])
				_a1_0 = _a1_0.Add(_v1)
				_v2 := simd.LoadInt32s(a[_i+_lanes:])
				_a1_1 = _a1_1.Add(_v2)
				_v3 := simd.LoadInt32s(a[_i+2*_lanes:])
				_a1_2 = _a1_2.Add(_v3)
				_v4 := simd.LoadInt32s(a[_i+3*_lanes:])
				_a1_3 = _a1_3.Add(_v4)
			}
			for ; _i <= len(a)-_lanes; _i += _lanes {
				_v5 := simd.LoadInt32s(a[_i:])
				_a1_0 = _a1_0.Add(_v5)
			}
			_a1_0 = _a1_0.Add(_a1_1)
			_a1_0 = _a1_0.Add(_a1_2)
			_a1_0 = _a1_0.Add(_a1_3)
			var _buf1 [16]int32
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

func DotInt32s(a, b []int32) int32 {
	var sum int32
	if len(a) > 0 {
		_ = b[len(a)-1]
		{
			_i := 0
			if len(a) >= 64 {
				_a1_0 := simd.BroadcastInt32s(0)
				_a1_1 := simd.BroadcastInt32s(0)
				_a1_2 := simd.BroadcastInt32s(0)
				_a1_3 := simd.BroadcastInt32s(0)
				_lanes := _a1_0.Len()
				for ; _i <= len(a)-4*_lanes; _i += 4 * _lanes {
					_v1 := simd.LoadInt32s(a[_i:])
					_v2 := simd.LoadInt32s(b[_i:])
					_a1_0 = _a1_0.Add(_v1.Mul(_v2))
					_v3 := simd.LoadInt32s(a[_i+_lanes:])
					_v4 := simd.LoadInt32s(b[_i+_lanes:])
					_a1_1 = _a1_1.Add(_v3.Mul(_v4))
					_v5 := simd.LoadInt32s(a[_i+2*_lanes:])
					_v6 := simd.LoadInt32s(b[_i+2*_lanes:])
					_a1_2 = _a1_2.Add(_v5.Mul(_v6))
					_v7 := simd.LoadInt32s(a[_i+3*_lanes:])
					_v8 := simd.LoadInt32s(b[_i+3*_lanes:])
					_a1_3 = _a1_3.Add(_v7.Mul(_v8))
				}
				for ; _i <= len(a)-_lanes; _i += _lanes {
					_v9 := simd.LoadInt32s(a[_i:])
					_v10 := simd.LoadInt32s(b[_i:])
					_a1_0 = _a1_0.Add(_v9.Mul(_v10))
				}
				_a1_0 = _a1_0.Add(_a1_1)
				_a1_0 = _a1_0.Add(_a1_2)
				_a1_0 = _a1_0.Add(_a1_3)
				var _buf1 [16]int32
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

func MinUint16s(a []uint16) uint16 {
	m := uint16(65535)
	{
		_i := 0
		if len(a) >= 128 {
			_a1_0 := simd.BroadcastUint16s(65535)
			_a1_1 := simd.BroadcastUint16s(65535)
			_a1_2 := simd.BroadcastUint16s(65535)
			_a1_3 := simd.BroadcastUint16s(65535)
			_lanes := _a1_0.Len()
			for ; _i <= len(a)-4*_lanes; _i += 4 * _lanes {
				_v1 := simd.LoadUint16s(a[_i:])
				_a1_0 = _a1_0.Min(_v1)
				_v2 := simd.LoadUint16s(a[_i+_lanes:])
				_a1_1 = _a1_1.Min(_v2)
				_v3 := simd.LoadUint16s(a[_i+2*_lanes:])
				_a1_2 = _a1_2.Min(_v3)
				_v4 := simd.LoadUint16s(a[_i+3*_lanes:])
				_a1_3 = _a1_3.Min(_v4)
			}
			for ; _i <= len(a)-_lanes; _i += _lanes {
				_v5 := simd.LoadUint16s(a[_i:])
				_a1_0 = _a1_0.Min(_v5)
			}
			_a1_0 = _a1_0.Min(_a1_1)
			_a1_0 = _a1_0.Min(_a1_2)
			_a1_0 = _a1_0.Min(_a1_3)
			var _buf1 [32]uint16
			_a1_0.Store(_buf1[:])
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
			for ; _i <= len(a)-4*_lanes; _i += 4 * _lanes {
				_v1 := simd.LoadFloat32s(a[_i:])
				_a1_0 = _a1_0.Add(_v1)
				_v2 := simd.LoadFloat32s(a[_i+_lanes:])
				_a1_1 = _a1_1.Add(_v2)
				_v3 := simd.LoadFloat32s(a[_i+2*_lanes:])
				_a1_2 = _a1_2.Add(_v3)
				_v4 := simd.LoadFloat32s(a[_i+3*_lanes:])
				_a1_3 = _a1_3.Add(_v4)
			}
			for ; _i <= len(a)-_lanes; _i += _lanes {
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
				for ; _i <= len(a)-4*_lanes; _i += 4 * _lanes {
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
				for ; _i <= len(a)-_lanes; _i += _lanes {
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
			for ; _i <= len(a)-4*_lanes; _i += 4 * _lanes {
				_v1 := simd.LoadFloat32s(a[_i:])
				_a1_0 = _a1_0.Mul(_v1)
				_v2 := simd.LoadFloat32s(a[_i+_lanes:])
				_a1_1 = _a1_1.Mul(_v2)
				_v3 := simd.LoadFloat32s(a[_i+2*_lanes:])
				_a1_2 = _a1_2.Mul(_v3)
				_v4 := simd.LoadFloat32s(a[_i+3*_lanes:])
				_a1_3 = _a1_3.Mul(_v4)
			}
			for ; _i <= len(a)-_lanes; _i += _lanes {
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
				for ; _i <= len(a)-4*_lanes; _i += 4 * _lanes {
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
				for ; _i <= len(a)-_lanes; _i += _lanes {
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
			{
				_w1 := dst[lo:][:(hi - lo):(hi - lo)]
				_count := len(_w1)
				_w2, _w3 := a[lo:][:_count:_count], b[lo:][:_count:_count]
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
			{
				_w1 := dst[1:][:((len(dst) - 1) - 1):((len(dst) - 1) - 1)]
				_count := len(_w1)
				_w2 := a[1:][:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_v1.Mul(_vcFloat32s29).Store(_c1)
					_v2 := simd.LoadFloat32s(_c2[_lanes:])
					_v2.Mul(_vcFloat32s29).Store(_c1[_lanes:])
					_v3 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v3.Mul(_vcFloat32s29).Store(_c1[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v4.Mul(_vcFloat32s29).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w2[_i:])
					_v5.Mul(_vcFloat32s29).Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v6.Mul(_vcFloat32s29).StorePart(_w1[_i:])
					_i += _n
				}
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
			{
				_w1 := dst[lo:][:(hi - lo):(hi - lo)]
				_count := len(_w1)
				_w2 := a[lo:][:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c1)
					_v2 := simd.LoadFloat32s(_c2)
					_v1.Add(_v2).Store(_c1)
					_v3 := simd.LoadFloat32s(_c1[_lanes:])
					_v4 := simd.LoadFloat32s(_c2[_lanes:])
					_v3.Add(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadFloat32s(_c1[2*_lanes:])
					_v6 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v5.Add(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadFloat32s(_c1[3*_lanes:])
					_v8 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v7.Add(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadFloat32s(_w1[_i:])
					_v10 := simd.LoadFloat32s(_w2[_i:])
					_v9.Add(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadFloat32sPart(_w1[_i:])
					_v12, _ := simd.LoadFloat32sPart(_w2[_i:])
					_v11.Add(_v12).StorePart(_w1[_i:])
					_i += _n
				}
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
			{
				_w1 := dst[:(len(dst) - 1):(len(dst) - 1)]
				_count := len(_w1)
				_w2, _w3 := a[1:][:_count:_count], a[:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v2 := simd.LoadInt32s(_c3)
					_v1.Sub(_v2).Store(_c1)
					_v3 := simd.LoadInt32s(_c2[_lanes:])
					_v4 := simd.LoadInt32s(_c3[_lanes:])
					_v3.Sub(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadInt32s(_c2[2*_lanes:])
					_v6 := simd.LoadInt32s(_c3[2*_lanes:])
					_v5.Sub(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadInt32s(_c2[3*_lanes:])
					_v8 := simd.LoadInt32s(_c3[3*_lanes:])
					_v7.Sub(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadInt32s(_w2[_i:])
					_v10 := simd.LoadInt32s(_w3[_i:])
					_v9.Sub(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadInt32sPart(_w2[_i:])
					_v12, _ := simd.LoadInt32sPart(_w3[_i:])
					_v11.Sub(_v12).StorePart(_w1[_i:])
					_i += _n
				}
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
			{
				_w1 := dst[1:][:((len(dst) - 1) - 1):((len(dst) - 1) - 1)]
				_count := len(_w1)
				_w2, _w3, _w4 := a[0:][:_count:_count], a[1:][:_count:_count], a[2:][:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3, _c4 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes], _w4[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v2 := simd.LoadInt32s(_c3)
					_v3 := simd.LoadInt32s(_c4)
					_v1.Add(_v2).Add(_v3).Store(_c1)
					_v4 := simd.LoadInt32s(_c2[_lanes:])
					_v5 := simd.LoadInt32s(_c3[_lanes:])
					_v6 := simd.LoadInt32s(_c4[_lanes:])
					_v4.Add(_v5).Add(_v6).Store(_c1[_lanes:])
					_v7 := simd.LoadInt32s(_c2[2*_lanes:])
					_v8 := simd.LoadInt32s(_c3[2*_lanes:])
					_v9 := simd.LoadInt32s(_c4[2*_lanes:])
					_v7.Add(_v8).Add(_v9).Store(_c1[2*_lanes:])
					_v10 := simd.LoadInt32s(_c2[3*_lanes:])
					_v11 := simd.LoadInt32s(_c3[3*_lanes:])
					_v12 := simd.LoadInt32s(_c4[3*_lanes:])
					_v10.Add(_v11).Add(_v12).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v13 := simd.LoadInt32s(_w2[_i:])
					_v14 := simd.LoadInt32s(_w3[_i:])
					_v15 := simd.LoadInt32s(_w4[_i:])
					_v13.Add(_v14).Add(_v15).Store(_w1[_i:])
				}
				for _i < _count {
					_v16, _n := simd.LoadInt32sPart(_w2[_i:])
					_v17, _ := simd.LoadInt32sPart(_w3[_i:])
					_v18, _ := simd.LoadInt32sPart(_w4[_i:])
					_v16.Add(_v17).Add(_v18).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func ShiftInt32s(dst, a []int32, off int) {
	_vcInt32s33 := simd.BroadcastInt32s(3)
	if _loopvecOverlap(dst, a, 0, len(dst), 0, len(a)) {
		for i := range dst {
			dst[i] = a[i+off] * 3
		}
	} else {
		if len(dst) > 0 {
			_ = a[len(dst)+off-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := a[off:][:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c2)
					_v1.Mul(_vcInt32s33).Store(_c1)
					_v2 := simd.LoadInt32s(_c2[_lanes:])
					_v2.Mul(_vcInt32s33).Store(_c1[_lanes:])
					_v3 := simd.LoadInt32s(_c2[2*_lanes:])
					_v3.Mul(_vcInt32s33).Store(_c1[2*_lanes:])
					_v4 := simd.LoadInt32s(_c2[3*_lanes:])
					_v4.Mul(_vcInt32s33).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadInt32s(_w2[_i:])
					_v5.Mul(_vcInt32s33).Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadInt32sPart(_w2[_i:])
					_v6.Mul(_vcInt32s33).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func AccumRowInt32s(work, a []int32, row, lda int) {
	if _loopvecOverlap(work, a, 0, len(work), 0, len(a)) {
		for j := range work {
			work[j] += a[row*lda+j]
		}
	} else {
		if len(work) > 0 {
			_ = a[len(work)+(row*lda)-1]
			{
				_w1 := work[:len(work):len(work)]
				_count := len(_w1)
				_w2 := a[row*lda:][:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c1)
					_v2 := simd.LoadInt32s(_c2)
					_v1.Add(_v2).Store(_c1)
					_v3 := simd.LoadInt32s(_c1[_lanes:])
					_v4 := simd.LoadInt32s(_c2[_lanes:])
					_v3.Add(_v4).Store(_c1[_lanes:])
					_v5 := simd.LoadInt32s(_c1[2*_lanes:])
					_v6 := simd.LoadInt32s(_c2[2*_lanes:])
					_v5.Add(_v6).Store(_c1[2*_lanes:])
					_v7 := simd.LoadInt32s(_c1[3*_lanes:])
					_v8 := simd.LoadInt32s(_c2[3*_lanes:])
					_v7.Add(_v8).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v9 := simd.LoadInt32s(_w1[_i:])
					_v10 := simd.LoadInt32s(_w2[_i:])
					_v9.Add(_v10).Store(_w1[_i:])
				}
				for _i < _count {
					_v11, _n := simd.LoadInt32sPart(_w1[_i:])
					_v12, _ := simd.LoadInt32sPart(_w2[_i:])
					_v11.Add(_v12).StorePart(_w1[_i:])
					_i += _n
				}
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
			{
				_w1 := a[:(len(a) - 1):(len(a) - 1)]
				_count := len(_w1)
				_w2, _w3 := a[1:][:_count:_count], b[:_count:_count]
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
			{
				_w1 := a[:(len(a) - 1):(len(a) - 1)]
				_count := len(_w1)
				_w2, _w3, _w4 := b[:_count:_count], d[:_count:_count], a[1:][:_count:_count]
				_i, _lanes := 0, simd.BroadcastInt32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2, _c3, _c4 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes], _w3[_i:][:4*_lanes], _w4[_i:][:4*_lanes]
					_v1 := simd.LoadInt32s(_c4)
					_v2 := simd.LoadInt32s(_c2)
					_v3 := simd.LoadInt32s(_c3)
					_v2.Add(_v3).Store(_c1)
					_v4 := simd.LoadInt32s(_c1)
					_v4.Mul(_vcInt32s36_1).Store(_c2)
					_v5 := simd.LoadInt32s(_c2)
					_v1.Mul(_v3).Add(_v5).Store(_c1)
					_v6 := simd.LoadInt32s(_c4[_lanes:])
					_v7 := simd.LoadInt32s(_c2[_lanes:])
					_v8 := simd.LoadInt32s(_c3[_lanes:])
					_v7.Add(_v8).Store(_c1[_lanes:])
					_v9 := simd.LoadInt32s(_c1[_lanes:])
					_v9.Mul(_vcInt32s36_1).Store(_c2[_lanes:])
					_v10 := simd.LoadInt32s(_c2[_lanes:])
					_v6.Mul(_v8).Add(_v10).Store(_c1[_lanes:])
					_v11 := simd.LoadInt32s(_c4[2*_lanes:])
					_v12 := simd.LoadInt32s(_c2[2*_lanes:])
					_v13 := simd.LoadInt32s(_c3[2*_lanes:])
					_v12.Add(_v13).Store(_c1[2*_lanes:])
					_v14 := simd.LoadInt32s(_c1[2*_lanes:])
					_v14.Mul(_vcInt32s36_1).Store(_c2[2*_lanes:])
					_v15 := simd.LoadInt32s(_c2[2*_lanes:])
					_v11.Mul(_v13).Add(_v15).Store(_c1[2*_lanes:])
					_v16 := simd.LoadInt32s(_c4[3*_lanes:])
					_v17 := simd.LoadInt32s(_c2[3*_lanes:])
					_v18 := simd.LoadInt32s(_c3[3*_lanes:])
					_v17.Add(_v18).Store(_c1[3*_lanes:])
					_v19 := simd.LoadInt32s(_c1[3*_lanes:])
					_v19.Mul(_vcInt32s36_1).Store(_c2[3*_lanes:])
					_v20 := simd.LoadInt32s(_c2[3*_lanes:])
					_v16.Mul(_v18).Add(_v20).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v21 := simd.LoadInt32s(_w4[_i:])
					_v22 := simd.LoadInt32s(_w2[_i:])
					_v23 := simd.LoadInt32s(_w3[_i:])
					_v22.Add(_v23).Store(_w1[_i:])
					_v24 := simd.LoadInt32s(_w1[_i:])
					_v24.Mul(_vcInt32s36_1).Store(_w2[_i:])
					_v25 := simd.LoadInt32s(_w2[_i:])
					_v21.Mul(_v23).Add(_v25).Store(_w1[_i:])
				}
				for _i < _count {
					_v26, _n := simd.LoadInt32sPart(_w4[_i:])
					_v27, _ := simd.LoadInt32sPart(_w2[_i:])
					_v28, _ := simd.LoadInt32sPart(_w3[_i:])
					_v27.Add(_v28).StorePart(_w1[_i:])
					_v29, _ := simd.LoadInt32sPart(_w1[_i:])
					_v29.Mul(_vcInt32s36_1).StorePart(_w2[_i:])
					_v30, _ := simd.LoadInt32sPart(_w2[_i:])
					_v26.Mul(_v28).Add(_v30).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func ReLUFloat32s(x []float32) {
	_vcAFloat32s37 := simd.BroadcastFloat32s(0)
	{
		_w1 := x[:len(x):len(x)]
		_count := len(_w1)
		_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
		for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
			_c1 := _w1[_i:][:4*_lanes]
			_v1 := simd.LoadFloat32s(_c1)
			_vcAFloat32s37.IfElse(_v1.Less(_vcAFloat32s37), _v1).Store(_c1)
			_v2 := simd.LoadFloat32s(_c1[_lanes:])
			_vcAFloat32s37.IfElse(_v2.Less(_vcAFloat32s37), _v2).Store(_c1[_lanes:])
			_v3 := simd.LoadFloat32s(_c1[2*_lanes:])
			_vcAFloat32s37.IfElse(_v3.Less(_vcAFloat32s37), _v3).Store(_c1[2*_lanes:])
			_v4 := simd.LoadFloat32s(_c1[3*_lanes:])
			_vcAFloat32s37.IfElse(_v4.Less(_vcAFloat32s37), _v4).Store(_c1[3*_lanes:])
		}
		for ; _i <= _count-_lanes; _i += _lanes {
			_v5 := simd.LoadFloat32s(_w1[_i:])
			_vcAFloat32s37.IfElse(_v5.Less(_vcAFloat32s37), _v5).Store(_w1[_i:])
		}
		for _i < _count {
			_v6, _n := simd.LoadFloat32sPart(_w1[_i:])
			_vcAFloat32s37.IfElse(_v6.Less(_vcAFloat32s37), _v6).StorePart(_w1[_i:])
			_i += _n
		}
	}
}

func LeakyReLUFloat32s(dst, x []float32, alpha float32) {
	_vcAFloat32s38 := simd.BroadcastFloat32s(0)
	_vcBFloat32s38 := simd.BroadcastFloat32s(alpha)
	if _loopvecOverlap(dst, x, 0, len(dst), 0, len(x)) {
		for i := range dst {
			if x[i] > 0 {
				dst[i] = x[i]
			} else {
				dst[i] = alpha * x[i]
			}
		}
	} else {
		if len(dst) > 0 {
			_ = x[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := x[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_v1.IfElse(_v1.Greater(_vcAFloat32s38), _v1.Mul(_vcBFloat32s38)).Store(_c1)
					_v2 := simd.LoadFloat32s(_c2[_lanes:])
					_v2.IfElse(_v2.Greater(_vcAFloat32s38), _v2.Mul(_vcBFloat32s38)).Store(_c1[_lanes:])
					_v3 := simd.LoadFloat32s(_c2[2*_lanes:])
					_v3.IfElse(_v3.Greater(_vcAFloat32s38), _v3.Mul(_vcBFloat32s38)).Store(_c1[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c2[3*_lanes:])
					_v4.IfElse(_v4.Greater(_vcAFloat32s38), _v4.Mul(_vcBFloat32s38)).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w2[_i:])
					_v5.IfElse(_v5.Greater(_vcAFloat32s38), _v5.Mul(_vcBFloat32s38)).Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w2[_i:])
					_v6.IfElse(_v6.Greater(_vcAFloat32s38), _v6.Mul(_vcBFloat32s38)).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func ReLU6Float32s(x []float32) {
	_vcFloat32s39_1 := simd.BroadcastFloat32s(0)
	_vcFloat32s39_3 := simd.BroadcastFloat32s(6)
	{
		_w1 := x[:len(x):len(x)]
		_count := len(_w1)
		_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
		for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
			_c1 := _w1[_i:][:4*_lanes]
			_v1 := simd.LoadFloat32s(_c1)
			_vcFloat32s39_1.IfElse(_v1.Less(_vcFloat32s39_1), _vcFloat32s39_3.IfElse(_v1.Greater(_vcFloat32s39_3), _v1)).Store(_c1)
			_v2 := simd.LoadFloat32s(_c1[_lanes:])
			_vcFloat32s39_1.IfElse(_v2.Less(_vcFloat32s39_1), _vcFloat32s39_3.IfElse(_v2.Greater(_vcFloat32s39_3), _v2)).Store(_c1[_lanes:])
			_v3 := simd.LoadFloat32s(_c1[2*_lanes:])
			_vcFloat32s39_1.IfElse(_v3.Less(_vcFloat32s39_1), _vcFloat32s39_3.IfElse(_v3.Greater(_vcFloat32s39_3), _v3)).Store(_c1[2*_lanes:])
			_v4 := simd.LoadFloat32s(_c1[3*_lanes:])
			_vcFloat32s39_1.IfElse(_v4.Less(_vcFloat32s39_1), _vcFloat32s39_3.IfElse(_v4.Greater(_vcFloat32s39_3), _v4)).Store(_c1[3*_lanes:])
		}
		for ; _i <= _count-_lanes; _i += _lanes {
			_v5 := simd.LoadFloat32s(_w1[_i:])
			_vcFloat32s39_1.IfElse(_v5.Less(_vcFloat32s39_1), _vcFloat32s39_3.IfElse(_v5.Greater(_vcFloat32s39_3), _v5)).Store(_w1[_i:])
		}
		for _i < _count {
			_v6, _n := simd.LoadFloat32sPart(_w1[_i:])
			_vcFloat32s39_1.IfElse(_v6.Less(_vcFloat32s39_1), _vcFloat32s39_3.IfElse(_v6.Greater(_vcFloat32s39_3), _v6)).StorePart(_w1[_i:])
			_i += _n
		}
	}
}

func SignFloat32s(dst, x []float32) {
	_vcFloat32s40_1 := simd.BroadcastFloat32s(0)
	_vcFloat32s40_2 := simd.BroadcastFloat32s(1)
	_vcFloat32s40_4 := simd.BroadcastFloat32s(-1)
	if _loopvecOverlap(dst, x, 0, len(dst), 0, len(x)) {
		for i := range dst {
			if x[i] > 0 {
				dst[i] = 1
			} else if x[i] < 0 {
				dst[i] = -1
			} else {
				dst[i] = 0
			}
		}
	} else {
		if len(dst) > 0 {
			_ = x[len(dst)-1]
			{
				_w1 := dst[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := x[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c2)
					_vcFloat32s40_2.IfElse(_v1.Greater(_vcFloat32s40_1), _vcFloat32s40_4.IfElse(_v1.Less(_vcFloat32s40_1), _vcFloat32s40_1)).Store(_c1)
					_v2 := simd.LoadFloat32s(_c2[_lanes:])
					_vcFloat32s40_2.IfElse(_v2.Greater(_vcFloat32s40_1), _vcFloat32s40_4.IfElse(_v2.Less(_vcFloat32s40_1), _vcFloat32s40_1)).Store(_c1[_lanes:])
					_v3 := simd.LoadFloat32s(_c2[2*_lanes:])
					_vcFloat32s40_2.IfElse(_v3.Greater(_vcFloat32s40_1), _vcFloat32s40_4.IfElse(_v3.Less(_vcFloat32s40_1), _vcFloat32s40_1)).Store(_c1[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c2[3*_lanes:])
					_vcFloat32s40_2.IfElse(_v4.Greater(_vcFloat32s40_1), _vcFloat32s40_4.IfElse(_v4.Less(_vcFloat32s40_1), _vcFloat32s40_1)).Store(_c1[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w2[_i:])
					_vcFloat32s40_2.IfElse(_v5.Greater(_vcFloat32s40_1), _vcFloat32s40_4.IfElse(_v5.Less(_vcFloat32s40_1), _vcFloat32s40_1)).Store(_w1[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w2[_i:])
					_vcFloat32s40_2.IfElse(_v6.Greater(_vcFloat32s40_1), _vcFloat32s40_4.IfElse(_v6.Less(_vcFloat32s40_1), _vcFloat32s40_1)).StorePart(_w1[_i:])
					_i += _n
				}
			}
		}
	}
}

func FillNaNFloat64s(dst []float64) {
	_vcFloat64s41 := simd.BroadcastFloat64s(math.NaN())
	{
		_w1 := dst[:len(dst):len(dst)]
		_count := len(_w1)
		_i, _lanes := 0, simd.BroadcastFloat64s(0).Len()
		for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
			_c1 := _w1[_i:][:4*_lanes]
			_vcFloat64s41.Store(_c1)
			_vcFloat64s41.Store(_c1[_lanes:])
			_vcFloat64s41.Store(_c1[2*_lanes:])
			_vcFloat64s41.Store(_c1[3*_lanes:])
		}
		for ; _i <= _count-_lanes; _i += _lanes {
			_vcFloat64s41.Store(_w1[_i:])
		}
		for _i < _count {
			_n := _vcFloat64s41.StorePart(_w1[_i:])
			_i += _n
		}
	}
}

// Vec holds its data in a field, as most numeric types do.
type Vec struct {
	Values []float32
}

func ScaleFieldFloat32s(v *Vec, a float32) {
	_vcFloat32s42 := simd.BroadcastFloat32s(a)
	if len(v.Values) > 0 {
		_sv_Values := v.Values
		{
			_w1 := _sv_Values[:len(v.Values):len(v.Values)]
			_count := len(_w1)
			_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
			for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
				_c1 := _w1[_i:][:4*_lanes]
				_v1 := simd.LoadFloat32s(_c1)
				_v1.Mul(_vcFloat32s42).Store(_c1)
				_v2 := simd.LoadFloat32s(_c1[_lanes:])
				_v2.Mul(_vcFloat32s42).Store(_c1[_lanes:])
				_v3 := simd.LoadFloat32s(_c1[2*_lanes:])
				_v3.Mul(_vcFloat32s42).Store(_c1[2*_lanes:])
				_v4 := simd.LoadFloat32s(_c1[3*_lanes:])
				_v4.Mul(_vcFloat32s42).Store(_c1[3*_lanes:])
			}
			for ; _i <= _count-_lanes; _i += _lanes {
				_v5 := simd.LoadFloat32s(_w1[_i:])
				_v5.Mul(_vcFloat32s42).Store(_w1[_i:])
			}
			for _i < _count {
				_v6, _n := simd.LoadFloat32sPart(_w1[_i:])
				_v6.Mul(_vcFloat32s42).StorePart(_w1[_i:])
				_i += _n
			}
		}
	}
}

func AddFieldFloat32s(dst, a *Vec) {
	if len(dst.Values) > 0 {
		_sdst_Values, _sa_Values := dst.Values, a.Values
		if _loopvecOverlap(_sdst_Values, _sa_Values, 0, len(_sdst_Values), 0, len(_sa_Values)) {
			for i := range dst.Values {
				dst.Values[i] += a.Values[i]
			}
		} else {
			if len(dst.Values) > 0 {
				_ = _sa_Values[len(dst.Values)-1]
				{
					_w1 := _sdst_Values[:len(dst.Values):len(dst.Values)]
					_count := len(_w1)
					_w2 := _sa_Values[:_count:_count]
					_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
					for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
						_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
						_v1 := simd.LoadFloat32s(_c1)
						_v2 := simd.LoadFloat32s(_c2)
						_v1.Add(_v2).Store(_c1)
						_v3 := simd.LoadFloat32s(_c1[_lanes:])
						_v4 := simd.LoadFloat32s(_c2[_lanes:])
						_v3.Add(_v4).Store(_c1[_lanes:])
						_v5 := simd.LoadFloat32s(_c1[2*_lanes:])
						_v6 := simd.LoadFloat32s(_c2[2*_lanes:])
						_v5.Add(_v6).Store(_c1[2*_lanes:])
						_v7 := simd.LoadFloat32s(_c1[3*_lanes:])
						_v8 := simd.LoadFloat32s(_c2[3*_lanes:])
						_v7.Add(_v8).Store(_c1[3*_lanes:])
					}
					for ; _i <= _count-_lanes; _i += _lanes {
						_v9 := simd.LoadFloat32s(_w1[_i:])
						_v10 := simd.LoadFloat32s(_w2[_i:])
						_v9.Add(_v10).Store(_w1[_i:])
					}
					for _i < _count {
						_v11, _n := simd.LoadFloat32sPart(_w1[_i:])
						_v12, _ := simd.LoadFloat32sPart(_w2[_i:])
						_v11.Add(_v12).StorePart(_w1[_i:])
						_i += _n
					}
				}
			}
		}
	}
}

func HardSwishFloat32s(dst, x []float32) {
	_vcFloat32s44_1 := simd.BroadcastFloat32s(3)
	_vcFloat32s44_2 := simd.BroadcastFloat32s(0)
	_vcFloat32s44_4 := simd.BroadcastFloat32s(6)
	if _loopvecOverlap(x, dst, 0, len(x), 0, len(dst)) {
		for i := range dst {
			t := x[i] + 3
			if t < 0 {
				t = 0
			} else if t > 6 {
				t = 6
			}
			dst[i] = x[i] * t / 6
		}
	} else {
		if len(dst) > 0 {
			_ = x[len(dst)-1]
			{
				_w1 := x[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := dst[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c1)
					_t1 := _v1.Add(_vcFloat32s44_1)
					_t2 := _vcFloat32s44_2.IfElse(_t1.Less(_vcFloat32s44_2), _vcFloat32s44_4.IfElse(_t1.Greater(_vcFloat32s44_4), _t1))
					_v1.Mul(_t2).Div(_vcFloat32s44_4).Store(_c2)
					_v2 := simd.LoadFloat32s(_c1[_lanes:])
					_t3 := _v2.Add(_vcFloat32s44_1)
					_t4 := _vcFloat32s44_2.IfElse(_t3.Less(_vcFloat32s44_2), _vcFloat32s44_4.IfElse(_t3.Greater(_vcFloat32s44_4), _t3))
					_v2.Mul(_t4).Div(_vcFloat32s44_4).Store(_c2[_lanes:])
					_v3 := simd.LoadFloat32s(_c1[2*_lanes:])
					_t5 := _v3.Add(_vcFloat32s44_1)
					_t6 := _vcFloat32s44_2.IfElse(_t5.Less(_vcFloat32s44_2), _vcFloat32s44_4.IfElse(_t5.Greater(_vcFloat32s44_4), _t5))
					_v3.Mul(_t6).Div(_vcFloat32s44_4).Store(_c2[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c1[3*_lanes:])
					_t7 := _v4.Add(_vcFloat32s44_1)
					_t8 := _vcFloat32s44_2.IfElse(_t7.Less(_vcFloat32s44_2), _vcFloat32s44_4.IfElse(_t7.Greater(_vcFloat32s44_4), _t7))
					_v4.Mul(_t8).Div(_vcFloat32s44_4).Store(_c2[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w1[_i:])
					_t9 := _v5.Add(_vcFloat32s44_1)
					_t10 := _vcFloat32s44_2.IfElse(_t9.Less(_vcFloat32s44_2), _vcFloat32s44_4.IfElse(_t9.Greater(_vcFloat32s44_4), _t9))
					_v5.Mul(_t10).Div(_vcFloat32s44_4).Store(_w2[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w1[_i:])
					_t11 := _v6.Add(_vcFloat32s44_1)
					_t12 := _vcFloat32s44_2.IfElse(_t11.Less(_vcFloat32s44_2), _vcFloat32s44_4.IfElse(_t11.Greater(_vcFloat32s44_4), _t11))
					_v6.Mul(_t12).Div(_vcFloat32s44_4).StorePart(_w2[_i:])
					_i += _n
				}
			}
		}
	}
}

func HardSigmoidFloat32s(dst, x []float32) {
	_vcFloat32s45_1 := simd.BroadcastFloat32s(3)
	_vcFloat32s45_2 := simd.BroadcastFloat32s(0)
	_vcFloat32s45_4 := simd.BroadcastFloat32s(6)
	if _loopvecOverlap(x, dst, 0, len(x), 0, len(dst)) {
		for i := range dst {
			t := x[i] + 3
			if t < 0 {
				t = 0
			}
			if t > 6 {
				t = 6
			}
			dst[i] = t / 6
		}
	} else {
		if len(dst) > 0 {
			_ = x[len(dst)-1]
			{
				_w1 := x[:len(dst):len(dst)]
				_count := len(_w1)
				_w2 := dst[:_count:_count]
				_i, _lanes := 0, simd.BroadcastFloat32s(0).Len()
				for ; _i <= _count-4*_lanes; _i += 4 * _lanes {
					_c1, _c2 := _w1[_i:][:4*_lanes], _w2[_i:][:4*_lanes]
					_v1 := simd.LoadFloat32s(_c1)
					_t1 := _v1.Add(_vcFloat32s45_1)
					_t2 := _vcFloat32s45_2.IfElse(_t1.Less(_vcFloat32s45_2), _t1)
					_t3 := _vcFloat32s45_4.IfElse(_t2.Greater(_vcFloat32s45_4), _t2)
					_t3.Div(_vcFloat32s45_4).Store(_c2)
					_v2 := simd.LoadFloat32s(_c1[_lanes:])
					_t4 := _v2.Add(_vcFloat32s45_1)
					_t5 := _vcFloat32s45_2.IfElse(_t4.Less(_vcFloat32s45_2), _t4)
					_t6 := _vcFloat32s45_4.IfElse(_t5.Greater(_vcFloat32s45_4), _t5)
					_t6.Div(_vcFloat32s45_4).Store(_c2[_lanes:])
					_v3 := simd.LoadFloat32s(_c1[2*_lanes:])
					_t7 := _v3.Add(_vcFloat32s45_1)
					_t8 := _vcFloat32s45_2.IfElse(_t7.Less(_vcFloat32s45_2), _t7)
					_t9 := _vcFloat32s45_4.IfElse(_t8.Greater(_vcFloat32s45_4), _t8)
					_t9.Div(_vcFloat32s45_4).Store(_c2[2*_lanes:])
					_v4 := simd.LoadFloat32s(_c1[3*_lanes:])
					_t10 := _v4.Add(_vcFloat32s45_1)
					_t11 := _vcFloat32s45_2.IfElse(_t10.Less(_vcFloat32s45_2), _t10)
					_t12 := _vcFloat32s45_4.IfElse(_t11.Greater(_vcFloat32s45_4), _t11)
					_t12.Div(_vcFloat32s45_4).Store(_c2[3*_lanes:])
				}
				for ; _i <= _count-_lanes; _i += _lanes {
					_v5 := simd.LoadFloat32s(_w1[_i:])
					_t13 := _v5.Add(_vcFloat32s45_1)
					_t14 := _vcFloat32s45_2.IfElse(_t13.Less(_vcFloat32s45_2), _t13)
					_t15 := _vcFloat32s45_4.IfElse(_t14.Greater(_vcFloat32s45_4), _t14)
					_t15.Div(_vcFloat32s45_4).Store(_w2[_i:])
				}
				for _i < _count {
					_v6, _n := simd.LoadFloat32sPart(_w1[_i:])
					_t16 := _v6.Add(_vcFloat32s45_1)
					_t17 := _vcFloat32s45_2.IfElse(_t16.Less(_vcFloat32s45_2), _t16)
					_t18 := _vcFloat32s45_4.IfElse(_t17.Greater(_vcFloat32s45_4), _t17)
					_t18.Div(_vcFloat32s45_4).StorePart(_w2[_i:])
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
