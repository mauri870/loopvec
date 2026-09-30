package tsvc

import (
	"hash"
	"hash/fnv"
	"math"
)

// sumA matches TSVC_2's sum_a, accumulated in float64 so the checksum itself
// adds as little rounding as possible.
func sumA(x *Arrays) float64 {
	var sum float64
	for _, v := range x.A {
		sum += float64(v)
	}
	return sum
}

// sumX matches TSVC_2's sum_x.
func sumX(x *Arrays) float64 {
	var sum float64
	for _, v := range x.X {
		sum += float64(v)
	}
	return sum
}

// sumAA matches TSVC_2's sum_aa.
func sumAA(x *Arrays) float64 {
	var sum float64
	for _, row := range x.AA {
		for _, v := range row {
			sum += float64(v)
		}
	}
	return sum
}

// hashA is an FNV-64a hash over the bit pattern of every element of A. A sum
// can hide swapped or compensating errors; the hash can't.
func hashA(x *Arrays) uint64 {
	h := fnv.New64a()
	for _, v := range x.A {
		hashFloat32(h, v)
	}
	return h.Sum64()
}

// hashX is hashA's equivalent for X.
func hashX(x *Arrays) uint64 {
	h := fnv.New64a()
	for _, v := range x.X {
		hashFloat32(h, v)
	}
	return h.Sum64()
}

// hashAA is hashA's equivalent for AA.
func hashAA(x *Arrays) uint64 {
	h := fnv.New64a()
	for _, row := range x.AA {
		for _, v := range row {
			hashFloat32(h, v)
		}
	}
	return h.Sum64()
}

func hashFloat32(h hash.Hash, v float32) {
	bits := math.Float32bits(v)
	h.Write([]byte{byte(bits), byte(bits >> 8), byte(bits >> 16), byte(bits >> 24)})
}

// sum1d adds v in float64, like sumA.
func sum1d(v []float32) float64 {
	var sum float64
	for _, e := range v {
		sum += float64(e)
	}
	return sum
}

// sum2d adds every element of m.
func sum2d(m [][]float32) float64 {
	var sum float64
	for _, row := range m {
		sum += sum1d(row)
	}
	return sum
}

// hash1d is an FNV-64a hash over the bit pattern of every element of the
// slices, in order.
func hash1d(slices ...[]float32) uint64 {
	h := fnv.New64a()
	for _, v := range slices {
		for _, e := range v {
			hashFloat32(h, e)
		}
	}
	return h.Sum64()
}

// hash2d is hash1d over the rows of the matrices, in order.
func hash2d(matrices ...[][]float32) uint64 {
	h := fnv.New64a()
	for _, m := range matrices {
		for _, row := range m {
			for _, e := range row {
				hashFloat32(h, e)
			}
		}
	}
	return h.Sum64()
}

// The checksums below match TSVC_2's calc_checksum forms sum_flat_2d_array,
// sum_a() + sum_b(), sum_a() + sum_b() + sum_c(), sum_aa_bb, sum_a_aa, sum_xx,
// and sum_half_xx. A kernel that returns a value is checked by that value.

func sumFlat2DArray(x *Arrays) float64 { return sum1d(x.FLAT_2D_ARRAY) }
func hashFlat2DArray(x *Arrays) uint64 { return hash1d(x.FLAT_2D_ARRAY) }
func sumAB(x *Arrays) float64          { return sum1d(x.A) + sum1d(x.B) }
func hashAB(x *Arrays) uint64          { return hash1d(x.A, x.B) }
func sumABC(x *Arrays) float64         { return sum1d(x.A) + sum1d(x.B) + sum1d(x.C) }
func hashABC(x *Arrays) uint64         { return hash1d(x.A, x.B, x.C) }
func sumAABB(x *Arrays) float64        { return sum2d(x.AA) + sum2d(x.BB) }
func hashAABB(x *Arrays) uint64        { return hash2d(x.AA, x.BB) }
func sumAAndAA(x *Arrays) float64      { return sum1d(x.A) + sum2d(x.AA) }
func hashAAndAA(x *Arrays) uint64 {
	h := fnv.New64a()
	for _, e := range x.A {
		hashFloat32(h, e)
	}
	for _, row := range x.AA {
		for _, e := range row {
			hashFloat32(h, e)
		}
	}
	return h.Sum64()
}
func sumR(x *Arrays) float64 { return float64(x.R) }
func hashR(x *Arrays) uint64 { return uint64(math.Float32bits(x.R)) }

// xx names a Len1D-element window of flat_2d_array, which kernels s421 through
// s424 alias with a pointer named xx: at offset 0, 4, and 63.
func xxWindow(x *Arrays, offset int) []float32 { return x.FLAT_2D_ARRAY[offset : offset+Len1D] }

func sumXX0(x *Arrays) float64  { return sum1d(xxWindow(x, 0)) }
func hashXX0(x *Arrays) uint64  { return hash1d(xxWindow(x, 0)) }
func sumXX4(x *Arrays) float64  { return sum1d(xxWindow(x, 4)) }
func hashXX4(x *Arrays) uint64  { return hash1d(xxWindow(x, 4)) }
func sumXX63(x *Arrays) float64 { return sum1d(xxWindow(x, 63)) }
func hashXX63(x *Arrays) uint64 { return hash1d(xxWindow(x, 63)) }

// sumHalfXX is sum_half_xx for s1421, where xx points at b[Len1D/2].
func sumHalfXX(x *Arrays) float64 { return sum1d(x.B[Len1D/2:]) }
func hashHalfXX(x *Arrays) uint64 { return hash1d(x.B[Len1D/2:]) }
