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

var sinkInt32 int32

func BenchmarkSumInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]int32, n)
			setBytes(b, n, 4, 1)
			b.ResetTimer()
			for b.Loop() {
				sinkInt32 = SumInt32s(a, 0)
			}
		})
	}
}

func BenchmarkDotInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			x := make([]int32, n)
			y := make([]int32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				sinkInt32 = DotInt32s(x, y)
			}
		})
	}
}

var (
	sinkFloat32 float32
	sinkFloat64 float64
)

func BenchmarkSumFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]float32, n)
			setBytes(b, n, 4, 1)
			b.ResetTimer()
			for b.Loop() {
				sinkFloat32 = SumFloat32s(a)
			}
		})
	}
}

func BenchmarkDotFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			x := make([]float32, n)
			y := make([]float32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				sinkFloat32 = DotFloat32s(x, y)
			}
		})
	}
}

func BenchmarkDotFloat64s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			x := make([]float64, n)
			y := make([]float64, n)
			setBytes(b, n, 8, 2)
			b.ResetTimer()
			for b.Loop() {
				sinkFloat64 = DotFloat64s(x, y)
			}
		})
	}
}

var sinkUint16 uint16

func BenchmarkAbsFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				AbsFloat32s(dst, a)
			}
		})
	}
}

func BenchmarkSqrtFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			a := make([]float32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				SqrtFloat32s(dst, a)
			}
		})
	}
}

func BenchmarkClampInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]int32, n)
			a := make([]int32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				ClampInt32s(dst, a, -100, 100)
			}
		})
	}
}

func BenchmarkShrInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]int32, n)
			a := make([]int32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				ShrInt32s(dst, a, 3)
			}
		})
	}
}

func BenchmarkShlUint32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]uint32, n)
			a := make([]uint32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				ShlUint32s(dst, a, 3)
			}
		})
	}
}

func BenchmarkDiffInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]int32, n)
			a := make([]int32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				DiffInt32s(dst, a)
			}
		})
	}
}

func BenchmarkSmoothInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]int32, n)
			a := make([]int32, n)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				SmoothInt32s(dst, a)
			}
		})
	}
}

func BenchmarkShiftInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]int32, n)
			a := make([]int32, n+3)
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				ShiftInt32s(dst, a, 3)
			}
		})
	}
}

// AccumRowInt32s adds one row of a two-dimensional slice into work.
func BenchmarkAccumRowInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			work := make([]int32, n)
			a := make([]int32, 2*n)
			setBytes(b, n, 4, 3)
			b.ResetTimer()
			for b.Loop() {
				AccumRowInt32s(work, a, 1, n)
			}
		})
	}
}

func BenchmarkMinUint16s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]uint16, n)
			setBytes(b, n, 2, 1)
			b.ResetTimer()
			for b.Loop() {
				sinkUint16 = MinUint16s(a)
			}
		})
	}
}

func BenchmarkProductFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]float32, n)
			for i := range a {
				a[i] = 1
			}
			setBytes(b, n, 4, 1)
			b.ResetTimer()
			for b.Loop() {
				sinkFloat32 = ProductFloat32s(a)
			}
		})
	}
}

func BenchmarkShiftAddInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]int32, n)
			src := make([]int32, n)
			setBytes(b, n, 4, 3)
			b.ResetTimer()
			for b.Loop() {
				ShiftAddInt32s(a, src)
			}
		})
	}
}

func BenchmarkReadAfterStoreInt32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := make([]int32, n)
			src := make([]int32, n)
			d := make([]int32, n)
			setBytes(b, n, 4, 3)
			b.ResetTimer()
			for b.Loop() {
				ReadAfterStoreInt32s(a, src, d)
			}
		})
	}
}

func BenchmarkReLUFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(i%7) - 3
			}
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				ReLUFloat32s(x)
			}
		})
	}
}

func BenchmarkLeakyReLUFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(i%7) - 3
			}
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				LeakyReLUFloat32s(dst, x, 0.01)
			}
		})
	}
}

func BenchmarkReLU6Float32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(i%13) - 3
			}
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				ReLU6Float32s(x)
			}
		})
	}
}

func BenchmarkSignFloat32s(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			dst := make([]float32, n)
			x := make([]float32, n)
			for i := range x {
				x[i] = float32(i%7) - 3
			}
			setBytes(b, n, 4, 2)
			b.ResetTimer()
			for b.Loop() {
				SignFloat32s(dst, x)
			}
		})
	}
}
