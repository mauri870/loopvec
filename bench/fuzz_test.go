package bench

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Differentially fuzzes the exported functions against a reference loop
// that mirrors the original scalar source. Under a plain build both sides
// run the same scalar code, so a mismatch means the reference is wrong; run
// under GOEXPERIMENT=simd to check the generated code in ops_simd.go:
// GOEXPERIMENT=simd go test ./bench/ -run Fuzz

// layout places dst (starts[0]) and each other view at dst's start plus its
// relOffset inside one shared backing array, so a nonzero offset produces
// genuine partial overlap between views.
func layout(n int, relOffsets ...int) (starts []int, backingLen int) {
	minOff, maxOff := 0, 0
	for _, o := range relOffsets {
		minOff = min(minOff, o)
		maxOff = max(maxOff, o)
	}
	shift := -minOff
	starts = make([]int, len(relOffsets)+1)
	starts[0] = shift
	for i, o := range relOffsets {
		starts[i+1] = shift + o
	}
	backingLen = shift + maxOff + n
	return starts, backingLen
}

// clampLen keeps fuzzed lengths small so runs stay fast.
func clampLen(n int) int {
	if n < 0 {
		n = -n
	}
	return n % 300
}

// clampOverlap keeps an offset within (-n, n) so layout produces overlap.
func clampOverlap(offset, n int) int {
	if n == 0 {
		return 0
	}
	return offset % n
}

// fuzzFloat32Value occasionally returns NaN, ±0, or ±Inf.
func fuzzFloat32Value(r *rand.Rand) float32 {
	switch r.IntN(20) {
	case 0:
		return float32(math.NaN())
	case 1:
		return math.Float32frombits(1 << 31) // -0
	case 2:
		return 0
	case 3:
		return float32(math.Inf(1))
	case 4:
		return float32(math.Inf(-1))
	default:
		return (r.Float32() - 0.5) * 1e4
	}
}

func randFloat32Backing(seed uint64, n int) []float32 {
	r := rand.New(rand.NewPCG(seed, seed>>32|1))
	s := make([]float32, n)
	for i := range s {
		s[i] = fuzzFloat32Value(r)
	}
	return s
}

// float32Equal is bit-exact, treating any two NaNs as equal.
func float32Equal(got, want float32) bool {
	if math.IsNaN(float64(got)) && math.IsNaN(float64(want)) {
		return true
	}
	return math.Float32bits(got) == math.Float32bits(want)
}

// float32AlmostEqual allows up to maxULP: the generated code's FMA rounds
// once where the reference loop's separate multiply and add round twice.
func float32AlmostEqual(got, want float32, maxULP uint32) bool {
	if math.IsNaN(float64(got)) && math.IsNaN(float64(want)) {
		return true
	}
	gb, wb := math.Float32bits(got), math.Float32bits(want)
	if gb == wb {
		return true
	}
	if gb>>31 != wb>>31 {
		return got == want // only true for +0 == -0
	}
	d := gb - wb
	if wb > gb {
		d = wb - gb
	}
	return d <= maxULP
}

func FuzzAddFloat32s(f *testing.F) {
	f.Add(0, 0, 0, uint64(1))
	f.Add(1, 0, 0, uint64(2))
	f.Add(5, 1, 0, uint64(3))  // dst/a overlap by 1: AddFloat32s(x[1:], x, y)
	f.Add(5, 0, -1, uint64(4)) // dst/b overlap by -1
	f.Add(5, 1, 1, uint64(5))  // dst, a, b all identical
	f.Add(9, 0, 0, uint64(6))  // odd tail
	f.Add(4, 0, 0, uint64(7))  // exactly one vector
	f.Fuzz(func(t *testing.T, nRaw, offARaw, offBRaw int, seed uint64) {
		n := clampLen(nRaw)
		offA := clampOverlap(offARaw, n)
		offB := clampOverlap(offBRaw, n)
		starts, backingLen := layout(n, offA, offB)
		dstStart, aStart, bStart := starts[0], starts[1], starts[2]

		src := randFloat32Backing(seed, backingLen)
		want := append([]float32(nil), src...)
		got := append([]float32(nil), src...)

		dstW, aW, bW := want[dstStart:dstStart+n], want[aStart:aStart+n], want[bStart:bStart+n]
		for i := range dstW {
			dstW[i] = aW[i] + bW[i]
		}

		dstG, aG, bG := got[dstStart:dstStart+n], got[aStart:aStart+n], got[bStart:bStart+n]
		AddFloat32s(dstG, aG, bG)

		for i := range dstW {
			if !float32Equal(dstW[i], dstG[i]) {
				t.Fatalf("AddFloat32s mismatch at %d (n=%d offA=%d offB=%d): got %v (0x%x) want %v (0x%x)",
					i, n, offA, offB, dstG[i], math.Float32bits(dstG[i]), dstW[i], math.Float32bits(dstW[i]))
			}
		}
	})
}

func FuzzNegFloat32s(f *testing.F) {
	f.Add(0, 0, uint64(1))
	f.Add(1, 0, uint64(2))
	f.Add(5, 0, uint64(3))  // dst == src
	f.Add(5, 1, uint64(4))  // partial overlap
	f.Add(5, -1, uint64(5)) // partial overlap, other direction
	f.Add(9, 0, uint64(6))
	f.Fuzz(func(t *testing.T, nRaw, offSrcRaw int, seed uint64) {
		n := clampLen(nRaw)
		offSrc := clampOverlap(offSrcRaw, n)
		starts, backingLen := layout(n, offSrc)
		dstStart, srcStart := starts[0], starts[1]

		src := randFloat32Backing(seed, backingLen)
		want := append([]float32(nil), src...)
		got := append([]float32(nil), src...)

		dstW, srcW := want[dstStart:dstStart+n], want[srcStart:srcStart+n]
		for i := range dstW {
			dstW[i] = -srcW[i]
		}

		dstG, srcG := got[dstStart:dstStart+n], got[srcStart:srcStart+n]
		NegFloat32s(dstG, srcG)

		for i := range dstW {
			if !float32Equal(dstW[i], dstG[i]) {
				t.Fatalf("NegFloat32s mismatch at %d (n=%d off=%d): got %v (0x%x) want %v (0x%x)",
					i, n, offSrc, dstG[i], math.Float32bits(dstG[i]), dstW[i], math.Float32bits(dstW[i]))
			}
		}
	})
}

func FuzzReverseIncFloat32s(f *testing.F) {
	f.Add(0, 0, uint64(1))
	f.Add(1, 0, uint64(2))
	f.Add(5, 0, uint64(3))
	f.Add(5, 1, uint64(4))
	f.Add(5, -1, uint64(5))
	f.Add(9, 0, uint64(6))
	f.Fuzz(func(t *testing.T, nRaw, offSrcRaw int, seed uint64) {
		n := clampLen(nRaw)
		offSrc := clampOverlap(offSrcRaw, n)
		starts, backingLen := layout(n, offSrc)
		dstStart, srcStart := starts[0], starts[1]

		src := randFloat32Backing(seed, backingLen)
		want := append([]float32(nil), src...)
		got := append([]float32(nil), src...)

		// Mirror the original loop's reverse iteration order: with overlap,
		// direction changes which stale writes a later read can see.
		dstW, srcW := want[dstStart:dstStart+n], want[srcStart:srcStart+n]
		for i := len(dstW) - 1; i >= 0; i-- {
			dstW[i] = srcW[i] + 1
		}

		dstG, srcG := got[dstStart:dstStart+n], got[srcStart:srcStart+n]
		ReverseIncFloat32s(dstG, srcG)

		for i := range dstW {
			if !float32Equal(dstW[i], dstG[i]) {
				t.Fatalf("ReverseIncFloat32s mismatch at %d (n=%d off=%d): got %v (0x%x) want %v (0x%x)",
					i, n, offSrc, dstG[i], math.Float32bits(dstG[i]), dstW[i], math.Float32bits(dstW[i]))
			}
		}
	})
}

func FuzzDaxpyFloat32s(f *testing.F) {
	f.Add(0, 0, uint64(1), float32(1))
	f.Add(1, 0, uint64(2), float32(0))
	f.Add(5, 0, uint64(3), float32(2.5)) // dst == a
	f.Add(5, 1, uint64(4), float32(-3))  // partial overlap
	f.Add(9, 0, uint64(5), float32(1.5))
	f.Fuzz(func(t *testing.T, nRaw, offARaw int, seed uint64, alpha float32) {
		n := clampLen(nRaw)
		offA := clampOverlap(offARaw, n)
		starts, backingLen := layout(n, offA)
		dstStart, aStart := starts[0], starts[1]

		src := randFloat32Backing(seed, backingLen)
		want := append([]float32(nil), src...)
		got := append([]float32(nil), src...)

		dstW, aW := want[dstStart:dstStart+n], want[aStart:aStart+n]
		for i := range dstW {
			dstW[i] += aW[i] * alpha
		}

		dstG, aG := got[dstStart:dstStart+n], got[aStart:aStart+n]
		DaxpyFloat32s(dstG, aG, alpha)

		const maxULP = 2
		for i := range dstW {
			if !float32AlmostEqual(dstW[i], dstG[i], maxULP) {
				t.Fatalf("DaxpyFloat32s mismatch at %d (n=%d offA=%d alpha=%v): got %v (0x%x) want %v (0x%x)",
					i, n, offA, alpha, dstG[i], math.Float32bits(dstG[i]), dstW[i], math.Float32bits(dstW[i]))
			}
		}
	})
}
