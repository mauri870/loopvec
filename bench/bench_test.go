package bench

import (
	"fmt"
	"testing"
)

var benchSizes = []int{64, 4096, 1 << 20}

func BenchmarkAddFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			src := make([]float32, n)
			b.ResetTimer()
			for b.Loop() {
				AddFloat32s(dst, a, src)
			}
		})
	}
}

func BenchmarkMulFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			src := make([]float32, n)
			b.ResetTimer()
			for b.Loop() {
				MulFloat32s(dst, a, src)
			}
		})
	}
}

func BenchmarkScalFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			x := make([]float32, n)
			b.ResetTimer()
			for b.Loop() {
				ScalFloat32s(x, 2.0)
			}
		})
	}
}

func BenchmarkMixFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			src := make([]float32, n)
			b.ResetTimer()
			for b.Loop() {
				MixFloat32s(dst, a, src, 2.0, 0.5)
			}
		})
	}
}

func BenchmarkAxpyFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			src := make([]float32, n)
			b.ResetTimer()
			for b.Loop() {
				AxpyFloat32s(dst, a, 2.0, src)
			}
		})
	}
}

func BenchmarkNegFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			src := make([]float32, n)
			b.ResetTimer()
			for b.Loop() {
				NegFloat32s(dst, src)
			}
		})
	}
}
