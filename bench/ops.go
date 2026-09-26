//go:build !goexperiment.simd

package bench

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
