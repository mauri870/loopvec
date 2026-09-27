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
)

func (e Exactness) String() string {
	switch e {
	case ExactBits:
		return "bits"
	case ExactFused:
		return "fused"
	default:
		return "unknown"
	}
}
