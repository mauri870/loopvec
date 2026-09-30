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

// clampOverlap keeps an offset within (-2n, 2n). An offset shorter than n makes
// the views overlap, which the generated code sends to the scalar loop; a longer
// one leaves them disjoint, which is the only case that runs the vector loop.
func clampOverlap(offset, n int) int {
	if n == 0 {
		return 0
	}
	return offset % (2 * n)
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

// fmaTolerance bounds how far a fused multiply-add can be from the same
// operation done in two roundings: about two ULPs of the larger of the addend
// and the product. Measured against the result it would be far too tight, since
// the result is much smaller than either term when they cancel.
func fmaTolerance(addend, product float32) float64 {
	larger := math.Max(math.Abs(float64(addend)), math.Abs(float64(product)))
	return 2 * (float64(math.Nextafter32(float32(larger), float32(math.Inf(1)))) - larger)
}

func FuzzAddFloat32s(f *testing.F) {
	f.Add(0, 0, 0, uint64(1))
	f.Add(1, 0, 0, uint64(2))
	f.Add(5, 1, 0, uint64(3))     // dst/a overlap by 1: AddFloat32s(x[1:], x, y)
	f.Add(5, 0, -1, uint64(4))    // dst/b overlap by -1
	f.Add(5, 1, 1, uint64(5))     // dst, a, b all identical
	f.Add(9, 0, 0, uint64(6))     // odd tail
	f.Add(4, 0, 0, uint64(7))     // exactly one vector
	f.Add(9, 11, 12, uint64(8))   // all three disjoint
	f.Add(37, -40, 50, uint64(9)) // disjoint, several vectors
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
	f.Add(9, 11, uint64(7))   // disjoint
	f.Add(37, -40, uint64(8)) // disjoint, several vectors
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
	f.Add(9, 11, uint64(7))
	f.Add(37, -40, uint64(8))
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
	f.Add(9, 11, uint64(6), float32(2))      // disjoint
	f.Add(37, -40, uint64(7), float32(-1.5)) // disjoint, several vectors
	f.Add(61, -74, uint64(238), float32(-3)) // cancellation: fused and unfused differ by 4 ULPs
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

		// Overlapping views take the scalar loop, which is exactly the reference. On
		// disjoint views the generated code fuses the multiply and the add into one
		// rounding, which can differ from two roundings by a ULP of the larger
		// term, and by many ULPs of the result when the terms cancel.
		for i := range dstW {
			if float32Equal(dstG[i], dstW[i]) {
				continue
			}
			bound := fmaTolerance(src[dstStart+i], src[aStart+i]*alpha)
			if !(math.Abs(float64(dstG[i])-float64(dstW[i])) <= bound) {
				t.Fatalf("DaxpyFloat32s mismatch at %d (n=%d offA=%d alpha=%v): got %v (0x%x) want %v (0x%x), bound %v",
					i, n, offA, alpha, dstG[i], math.Float32bits(dstG[i]), dstW[i], math.Float32bits(dstW[i]), bound)
			}
		}
	})
}

func FuzzCopyFloat32s(f *testing.F) {
	f.Add(0, 0, uint64(1))
	f.Add(1, 0, uint64(2))
	f.Add(5, 0, uint64(3))  // dst == src
	f.Add(5, 1, uint64(4))  // partial overlap: the loop repeats an element
	f.Add(5, -1, uint64(5)) // partial overlap, other direction
	f.Add(9, 0, uint64(6))
	f.Add(9, 11, uint64(7))   // disjoint
	f.Add(37, -40, uint64(8)) // disjoint, several vectors
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
			dstW[i] = srcW[i]
		}

		dstG, srcG := got[dstStart:dstStart+n], got[srcStart:srcStart+n]
		CopyFloat32s(dstG, srcG)

		for i := range want {
			if !float32Equal(got[i], want[i]) {
				t.Fatalf("CopyFloat32s mismatch at %d (n=%d off=%d): got %v (0x%x) want %v (0x%x)",
					i, n, offSrc, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	})
}

func randUint64Backing(seed uint64, n int) []uint64 {
	r := rand.New(rand.NewPCG(seed, seed>>32|1))
	s := make([]uint64, n)
	for i := range s {
		s[i] = r.Uint64()
	}
	return s
}

func FuzzAndNotUint64s(f *testing.F) {
	f.Add(0, 0, 0, uint64(1))
	f.Add(1, 0, 0, uint64(2))
	f.Add(5, 0, 0, uint64(3))     // dst == a == b
	f.Add(5, 1, 0, uint64(4))     // a partially overlaps dst
	f.Add(5, 0, -1, uint64(5))    // b partially overlaps dst
	f.Add(9, 2, -3, uint64(6))    // both overlap
	f.Add(9, 11, -12, uint64(7))  // disjoint
	f.Add(37, -40, 50, uint64(8)) // disjoint, several vectors
	f.Fuzz(func(t *testing.T, nRaw, offARaw, offBRaw int, seed uint64) {
		n := clampLen(nRaw)
		offA, offB := clampOverlap(offARaw, n), clampOverlap(offBRaw, n)
		starts, backingLen := layout(n, offA, offB)
		dstStart, aStart, bStart := starts[0], starts[1], starts[2]

		src := randUint64Backing(seed, backingLen)
		want := append([]uint64(nil), src...)
		got := append([]uint64(nil), src...)

		dstW, aW, bW := want[dstStart:dstStart+n], want[aStart:aStart+n], want[bStart:bStart+n]
		for i := range dstW {
			dstW[i] = aW[i] &^ bW[i]
		}

		AndNotUint64s(got[dstStart:dstStart+n], got[aStart:aStart+n], got[bStart:bStart+n])

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("AndNotUint64s mismatch at %d (n=%d offA=%d offB=%d): got %#x want %#x", i, n, offA, offB, got[i], want[i])
			}
		}
	})
}

func FuzzAbsSqrtFloat32s(f *testing.F) {
	f.Add(0, 0, uint64(1))
	f.Add(1, 0, uint64(2))
	f.Add(5, 0, uint64(3))  // dst == a
	f.Add(5, 1, uint64(4))  // partial overlap
	f.Add(5, -1, uint64(5)) // partial overlap, other direction
	f.Add(9, 0, uint64(6))
	f.Add(9, 11, uint64(7))   // disjoint
	f.Add(37, -40, uint64(8)) // disjoint, several vectors
	f.Fuzz(func(t *testing.T, nRaw, offRaw int, seed uint64) {
		n := clampLen(nRaw)
		off := clampOverlap(offRaw, n)
		starts, backingLen := layout(n, off)
		dstStart, srcStart := starts[0], starts[1]

		for _, sqrt := range []bool{false, true} {
			src := randFloat32Backing(seed, backingLen)
			want := append([]float32(nil), src...)
			got := append([]float32(nil), src...)

			dstW, srcW := want[dstStart:dstStart+n], want[srcStart:srcStart+n]
			for i := range dstW {
				if sqrt {
					dstW[i] = float32(math.Sqrt(float64(srcW[i])))
				} else {
					dstW[i] = float32(math.Abs(float64(srcW[i])))
				}
			}

			dstG, srcG := got[dstStart:dstStart+n], got[srcStart:srcStart+n]
			if sqrt {
				SqrtFloat32s(dstG, srcG)
			} else {
				AbsFloat32s(dstG, srcG)
			}

			for i := range want {
				if !float32Equal(got[i], want[i]) {
					t.Fatalf("sqrt=%v mismatch at %d (n=%d off=%d): got %v (0x%x) want %v (0x%x)",
						sqrt, i, n, off, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
				}
			}
		}
	})
}

func FuzzClampInt32s(f *testing.F) {
	f.Add(0, 0, uint64(1), int32(0), int32(0))
	f.Add(5, 0, uint64(2), int32(-10), int32(10))
	f.Add(5, 1, uint64(3), int32(-1), int32(1))      // partial overlap
	f.Add(9, -1, uint64(4), int32(5), int32(-5))     // lo above hi
	f.Add(9, 11, uint64(5), int32(-100), int32(100)) // disjoint
	f.Add(37, -40, uint64(6), int32(-7), int32(7))   // disjoint, several vectors
	f.Fuzz(func(t *testing.T, nRaw, offRaw int, seed uint64, lo, hi int32) {
		n := clampLen(nRaw)
		off := clampOverlap(offRaw, n)
		starts, backingLen := layout(n, off)
		dstStart, srcStart := starts[0], starts[1]

		src := make([]int32, backingLen)
		for i, v := range randUint64Backing(seed, backingLen) {
			src[i] = int32(v)
		}
		want := append([]int32(nil), src...)
		got := append([]int32(nil), src...)

		dstW, srcW := want[dstStart:dstStart+n], want[srcStart:srcStart+n]
		for i := range dstW {
			dstW[i] = min(max(srcW[i], lo), hi)
		}
		ClampInt32s(got[dstStart:dstStart+n], got[srcStart:srcStart+n], lo, hi)

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mismatch at %d (n=%d off=%d lo=%d hi=%d): got %d want %d", i, n, off, lo, hi, got[i], want[i])
			}
		}
	})
}

func FuzzShiftInt32s(f *testing.F) {
	f.Add(0, 0, uint64(1), uint(0))
	f.Add(5, 0, uint64(2), uint(3))
	f.Add(5, 1, uint64(3), uint(31)) // partial overlap
	f.Add(9, 0, uint64(4), uint(32))
	f.Add(9, 0, uint64(5), uint(1<<40))
	f.Add(9, 11, uint64(6), uint(5))    // disjoint
	f.Add(37, -40, uint64(7), uint(31)) // disjoint, several vectors
	f.Fuzz(func(t *testing.T, nRaw, offRaw int, seed uint64, count uint) {
		n := clampLen(nRaw)
		off := clampOverlap(offRaw, n)
		starts, backingLen := layout(n, off)
		dstStart, srcStart := starts[0], starts[1]

		backing := randUint64Backing(seed, backingLen)
		signed := make([]int32, backingLen)
		unsigned := make([]uint32, backingLen)
		for i, v := range backing {
			signed[i], unsigned[i] = int32(v), uint32(v)
		}
		wantS, wantU := append([]int32(nil), signed...), append([]uint32(nil), unsigned...)
		gotS, gotU := append([]int32(nil), signed...), append([]uint32(nil), unsigned...)

		for i := range n {
			wantS[dstStart+i] = wantS[srcStart+i] >> count
			wantU[dstStart+i] = wantU[srcStart+i] << count
		}
		ShrInt32s(gotS[dstStart:dstStart+n], gotS[srcStart:srcStart+n], count)
		ShlUint32s(gotU[dstStart:dstStart+n], gotU[srcStart:srcStart+n], count)

		for i := range wantS {
			if gotS[i] != wantS[i] || gotU[i] != wantU[i] {
				t.Fatalf("mismatch at %d (n=%d off=%d count=%d): int32 >> got %d want %d, uint32 << got %#x want %#x",
					i, n, off, count, gotS[i], wantS[i], gotU[i], wantU[i])
			}
		}
	})
}

func FuzzNormalizeFloat32s(f *testing.F) {
	f.Add(0, 0, uint64(1), float32(2))
	f.Add(5, 0, uint64(2), float32(3)) // auth == delta
	f.Add(5, 1, uint64(3), float32(0)) // partial overlap
	f.Add(9, 11, uint64(4), float32(-1.5))
	f.Add(9, 11, uint64(5), float32(0))       // disjoint, division by zero
	f.Add(37, -40, uint64(6), float32(1e-30)) // disjoint, several vectors
	f.Add(37, 40, uint64(7), float32(math.Inf(1)))
	f.Fuzz(func(t *testing.T, nRaw, offRaw int, seed uint64, norm float32) {
		n := clampLen(nRaw)
		off := clampOverlap(offRaw, n)
		starts, backingLen := layout(n, off)
		authStart, deltaStart := starts[0], starts[1]

		src := randFloat32Backing(seed, backingLen)
		want := append([]float32(nil), src...)
		got := append([]float32(nil), src...)

		// The second statement reads what the first stored.
		for i := range n {
			want[authStart+i] /= norm
			want[deltaStart+i] -= want[authStart+i]
		}
		NormalizeFloat32s(got[authStart:authStart+n], got[deltaStart:deltaStart+n], norm)

		for i := range want {
			if !float32Equal(got[i], want[i]) {
				t.Fatalf("mismatch at %d (n=%d off=%d norm=%v): got %v (0x%x) want %v (0x%x)",
					i, n, off, norm, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	})
}

func FuzzChainInt32s(f *testing.F) {
	f.Add(0, 0, 0, 0, uint64(1))
	f.Add(5, 0, 0, 0, uint64(2))       // every view the same
	f.Add(5, 1, 2, 3, uint64(3))       // partial overlap
	f.Add(9, 10, -10, 20, uint64(4))   // disjoint
	f.Add(37, -40, 80, 40, uint64(5))  // disjoint, several vectors
	f.Add(37, 40, -80, -40, uint64(6)) // disjoint
	f.Fuzz(func(t *testing.T, nRaw, offBRaw, offCRaw, offERaw int, seed uint64) {
		n := clampLen(nRaw)
		// Four views need room to be disjoint, which the two-view clampOverlap
		// range does not leave.
		clamp := func(offset int) int {
			if n == 0 {
				return 0
			}
			return offset % (4 * n)
		}
		offB, offC, offE := clamp(offBRaw), clamp(offCRaw), clamp(offERaw)
		starts, backingLen := layout(n, offB, offC, offE)
		aStart, bStart, cStart, eStart := starts[0], starts[1], starts[2], starts[3]

		backing := randUint64Backing(seed, backingLen)
		src := make([]int32, backingLen)
		for i, v := range backing {
			src[i] = int32(v)
		}
		want := append([]int32(nil), src...)
		got := append([]int32(nil), src...)

		// The temporary is computed before the two stores, and the second store
		// reads what the first wrote.
		for i := range n {
			x := want[bStart+i] * want[cStart+i]
			want[aStart+i] = x + want[eStart+i]
			want[bStart+i] = x - want[aStart+i]
		}
		ChainInt32s(got[aStart:aStart+n], got[bStart:bStart+n], got[cStart:cStart+n], got[eStart:eStart+n])

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mismatch at %d (n=%d offB=%d offC=%d offE=%d): got %d want %d", i, n, offB, offC, offE, got[i], want[i])
			}
		}
	})
}

func FuzzWideInt32s(f *testing.F) {
	f.Add(0, 0, 0, 0, uint64(1), int32(3), int32(5))
	f.Add(5, 0, 0, 0, uint64(2), int32(-7), int32(0)) // every view the same
	f.Add(5, 1, 2, 3, uint64(3), int32(2), int32(9))  // partial overlap
	f.Add(9, 10, -10, 20, uint64(4), int32(1<<30), int32(4))
	f.Add(37, -40, 80, 40, uint64(5), int32(-1), int32(-1<<31)) // several vectors
	f.Fuzz(func(t *testing.T, nRaw, offARaw, offBRaw, offCRaw int, seed uint64, k, m int32) {
		n := clampLen(nRaw)
		clamp := func(offset int) int {
			if n == 0 {
				return 0
			}
			return offset % (4 * n)
		}
		offA, offB, offC := clamp(offARaw), clamp(offBRaw), clamp(offCRaw)
		starts, backingLen := layout(n, offA, offB, offC)
		dstStart, aStart, bStart, cStart := starts[0], starts[1], starts[2], starts[3]

		backing := randUint64Backing(seed, backingLen)
		src := make([]int32, backingLen)
		for i, v := range backing {
			src[i] = int32(v)
		}
		want := append([]int32(nil), src...)
		got := append([]int32(nil), src...)

		for i := range n {
			want[dstStart+i] = want[aStart+i]*want[bStart+i] + want[cStart+i]*k + (k*m - want[aStart+i])
		}
		WideInt32s(got[dstStart:dstStart+n], got[aStart:aStart+n], got[bStart:bStart+n], got[cStart:cStart+n], k, m)

		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mismatch at %d (n=%d offA=%d offB=%d offC=%d k=%d m=%d): got %d want %d", i, n, offA, offB, offC, k, m, got[i], want[i])
			}
		}
	})
}
