package tsvc

import "testing"

func BenchmarkKernels(b *testing.B) {
	for _, k := range Kernels {
		b.Run(k.Name, func(b *testing.B) {
			x := NewArrays()
			k.Setup(x)
			b.SetBytes(k.Bytes)
			b.ResetTimer()
			for b.Loop() {
				k.Run(x)
			}
		})
	}
}
