package tsvc

// Exactness says how tightly a kernel's SIMD result must match its scalar
// golden.
type Exactness int

const (
	// ExactBits requires the hash and sum to match exactly.
	ExactBits Exactness = iota
	// ExactFused allows the relative error a compiler-contracted FMA would
	// introduce; the hash is not compared.
	ExactFused
	// ExactReassoc allows the error of regrouping a floating-point sum or
	// product, as loopvec's -fp-reassoc does: the vector loop adds in a different
	// order, so the result is within the summation error bound of the scalar one
	// but not within a single fused multiply-add of it. The hash is not compared.
	ExactReassoc
)

func (e Exactness) String() string {
	switch e {
	case ExactBits:
		return "bits"
	case ExactFused:
		return "fused"
	case ExactReassoc:
		return "reassoc"
	default:
		return "unknown"
	}
}
