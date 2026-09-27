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
