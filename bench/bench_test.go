package bench

import (
	"fmt"
	"testing"
)

var benchSizes = []int{64, 4096, 1 << 20}

// setBytes makes the benchmark report throughput. It counts every slice the
// loop reads or writes once per element: streams is 3 for dst = a + b (two
// reads, one write) and 2 for x *= c (one read, one write). At the largest size
// the slices no longer fit in L2 cache, so that figure measures memory
// bandwidth, not the loop.
func setBytes(b *testing.B, n, elemSize, streams int) {
	b.SetBytes(int64(n * elemSize * streams))
}

func BenchmarkAddFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			src := make([]float32, n)
			setBytes(b, n, 4, 3)
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
			setBytes(b, n, 4, 3)
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
			setBytes(b, n, 4, 2)
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
			setBytes(b, n, 4, 3)
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
			setBytes(b, n, 4, 3)
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
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				NegFloat32s(dst, src)
			}
		})
	}
}

func BenchmarkDivFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			src := make([]float32, n)
			setBytes(b, n, 4, 3)
			b.ResetTimer()
			for b.Loop() {
				DivFloat32s(dst, a, src)
			}
		})
	}
}

func BenchmarkDaxpyFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			setBytes(b, n, 4, 3)
			b.ResetTimer()
			for b.Loop() {
				DaxpyFloat32s(dst, a, 2.0)
			}
		})
	}
}

func BenchmarkFillFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			setBytes(b, n, 4, 1)
			b.ResetTimer()
			for b.Loop() {
				FillFloat32s(dst)
			}
		})
	}
}

func BenchmarkFillUint8s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]byte, n)
			setBytes(b, n, 1, 1)
			b.ResetTimer()
			for b.Loop() {
				FillUint8s(dst)
			}
		})
	}
}

func BenchmarkReverseIncFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			src := make([]float32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				ReverseIncFloat32s(dst, src)
			}
		})
	}
}

func BenchmarkCopyFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			src := make([]float32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				CopyFloat32s(dst, src)
			}
		})
	}
}

func BenchmarkAndNotUint64s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]uint64, n)
			a := make([]uint64, n)
			src := make([]uint64, n)
			setBytes(b, n, 8, 3)
			b.ResetTimer()
			for b.Loop() {
				AndNotUint64s(dst, a, src)
			}
		})
	}
}

func BenchmarkNormalizeFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			auth := make([]float32, n)
			delta := make([]float32, n)
			setBytes(b, n, 4, 4)
			b.ResetTimer()
			for b.Loop() {
				NormalizeFloat32s(auth, delta, 3)
			}
		})
	}
}

func BenchmarkChainInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]int32, n)
			x := make([]int32, n)
			c := make([]int32, n)
			e := make([]int32, n)
			setBytes(b, n, 4, 6)
			b.ResetTimer()
			for b.Loop() {
				ChainInt32s(a, x, c, e)
			}
		})
	}
}

func BenchmarkWideInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]int32, n)
			x := make([]int32, n)
			y := make([]int32, n)
			z := make([]int32, n)
			setBytes(b, n, 4, 4)
			b.ResetTimer()
			for b.Loop() {
				WideInt32s(dst, x, y, z, 3, 5)
			}
		})
	}
}
