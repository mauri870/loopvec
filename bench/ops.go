//go:build !goexperiment.simd

package bench

import "math"

func AddFloat32s(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func MulFloat32s(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func ScalFloat32s(x []float32, c float32) {
	for i := range x {
		x[i] *= c
	}
}

func AxpyFloat32s(dst, a []float32, alpha float32, b []float32) {
	for i := range dst {
		dst[i] = a[i]*alpha + b[i]
	}
}

func MixFloat32s(dst, a, b []float32, alpha, beta float32) {
	for i := range dst {
		dst[i] = a[i]*alpha + b[i]*beta
	}
}

func NegFloat32s(dst, src []float32) {
	for i := range dst {
		dst[i] = -src[i]
	}
}

func DivFloat32s(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func DaxpyFloat32s(dst, a []float32, alpha float32) {
	for i := range dst {
		dst[i] += a[i] * alpha
	}
}

func FillFloat32s(dst []float32) {
	for i := range dst {
		dst[i] = 2.5
	}
}

func FillUint8s(dst []byte) {
	for i := range dst {
		dst[i] = 7
	}
}

func ReverseIncFloat32s(dst, src []float32) {
	for i := len(dst) - 1; i >= 0; i-- {
		dst[i] = src[i] + 1
	}
}

func CopyFloat32s(dst, src []float32) {
	for i := range dst {
		dst[i] = src[i]
	}
}

func AndNotUint64s(dst, a, b []uint64) {
	for i := range dst {
		dst[i] = a[i] &^ b[i]
	}
}

func AbsFloat32s(dst, a []float32) {
	for i := range dst {
		dst[i] = float32(math.Abs(float64(a[i])))
	}
}

func SqrtFloat32s(dst, a []float32) {
	for i := range dst {
		dst[i] = float32(math.Sqrt(float64(a[i])))
	}
}

func ClampInt32s(dst, a []int32, lo, hi int32) {
	for i := range dst {
		dst[i] = min(max(a[i], lo), hi)
	}
}

func ShrInt32s(dst, a []int32, n uint) {
	for i := range dst {
		dst[i] = a[i] >> n
	}
}

func ShlUint32s(dst, a []uint32, n uint) {
	for i := range dst {
		dst[i] = a[i] << n
	}
}

func NormalizeFloat32s(auth, delta []float32, norm float32) {
	for i := range auth {
		auth[i] /= norm
		delta[i] -= auth[i]
	}
}

func ChainInt32s(a, b, c, e []int32) {
	for i := range a {
		x := b[i] * c[i]
		a[i] = x + e[i]
		b[i] = x - a[i]
	}
}
