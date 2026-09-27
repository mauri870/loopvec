package tsvc

import (
	"math"
	"testing"
)

// TestKernelsRun is a scaffolding smoke test: it exercises every kernel in
// the registry and checks the result is finite. Golden-file correctness
// checks land in a later change.
func TestKernelsRun(t *testing.T) {
	for _, k := range Kernels {
		t.Run(k.Name, func(t *testing.T) {
			x := NewArrays()
			k.Setup(x)
			for range k.Reps {
				k.Run(x)
			}

			sum := k.Checksum(x)
			if math.IsNaN(sum) || math.IsInf(sum, 0) {
				t.Fatalf("checksum = %v, want a finite value", sum)
			}

			_ = k.Hash(x)
		})
	}
}
