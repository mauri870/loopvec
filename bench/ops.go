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

func WideInt32s(dst, a, b, c []int32, k, m int32) {
	for i := range dst {
		dst[i] = a[i]*b[i] + c[i]*k + (k*m - a[i])
	}
}

func SumInt32s(a []int32, init int32) int32 {
	sum := init
	for i := range a {
		sum += a[i]
	}
	return sum
}

func DotInt32s(a, b []int32) int32 {
	var sum int32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func MinUint16s(a []uint16) uint16 {
	m := uint16(65535)
	for i := range a {
		m = min(m, a[i])
	}
	return m
}

func SumFloat32s(a []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i]
	}
	return sum
}

func DotFloat32s(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func ProductFloat32s(a []float32) float32 {
	prod := float32(1)
	for i := range a {
		prod *= a[i]
	}
	return prod
}

func DotFloat64s(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func AddWindowFloat32s(dst, a, b []float32, lo, hi int) {
	for i := lo; i < hi; i++ {
		dst[i] = a[i] + b[i]
	}
}

func ScaleInnerFloat32s(dst, a []float32, k float32) {
	for i := 1; i < len(dst)-1; i++ {
		dst[i] = a[i] * k
	}
}

func IncDownFloat32s(dst, a []float32, hi, lo int) {
	for i := hi - 1; i >= lo; i-- {
		dst[i] += a[i]
	}
}

func DiffInt32s(dst, a []int32) {
	for i := 0; i < len(dst)-1; i++ {
		dst[i] = a[i+1] - a[i]
	}
}

func SmoothInt32s(dst, a []int32) {
	for i := 1; i < len(dst)-1; i++ {
		dst[i] = a[i-1] + a[i] + a[i+1]
	}
}

func ShiftInt32s(dst, a []int32, off int) {
	for i := range dst {
		dst[i] = a[i+off] * 3
	}
}

func AccumRowInt32s(work, a []int32, row, lda int) {
	for j := range work {
		work[j] += a[row*lda+j]
	}
}

func ShiftAddInt32s(a, b []int32) {
	for i := 0; i < len(a)-1; i++ {
		a[i] = a[i+1] + b[i]
	}
}

func ReadAfterStoreInt32s(a, b, d []int32) {
	for i := 0; i < len(a)-1; i++ {
		a[i] = b[i] + d[i]
		b[i] = a[i] * 2
		a[i] = b[i] + a[i+1]*d[i]
	}
}

func ReLUFloat32s(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

func LeakyReLUFloat32s(dst, x []float32, alpha float32) {
	for i := range dst {
		if x[i] > 0 {
			dst[i] = x[i]
		} else {
			dst[i] = alpha * x[i]
		}
	}
}

func ReLU6Float32s(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		} else if x[i] > 6 {
			x[i] = 6
		}
	}
}

func SignFloat32s(dst, x []float32) {
	for i := range dst {
		if x[i] > 0 {
			dst[i] = 1
		} else if x[i] < 0 {
			dst[i] = -1
		} else {
			dst[i] = 0
		}
	}
}
