package loopir

// Reason explains why a candidate loop was not vectorized. The vocabulary is
// deliberately small and coarse: it names which stage of matching first
// rejected the loop, not the exact AST shape that failed. The same strings
// are meant to back a future -explain mode.
type Reason string

const (
	// ReasonUnsupportedClauses covers a for-loop whose init, condition, post,
	// or iteration bound doesn't match a supported shape: non-zero/non-len-1
	// start, a step other than ++/--, a bound that isn't len(slice) or a
	// simple int expression, and so on.
	ReasonUnsupportedClauses Reason = "loop clauses (start, step, direction, or bound) do not match a supported shape"
	// ReasonUnsupportedDestination covers a destination that isn't a plain
	// slice[i] element: an offset (dst[i+1]), a stride (dst[2*i]), or a
	// nested index (dst[i][j]).
	ReasonUnsupportedDestination Reason = "destination is not a simple, same-index slice element"
	// ReasonUnsupportedBody covers a loop body that isn't a single indexed
	// assignment: multiple statements, or a nested loop.
	ReasonUnsupportedBody Reason = "loop body is not a single indexed assignment"
	// ReasonUnsupportedOperand covers a right-hand side that isn't a
	// same-index slice read, the range value variable, or a plain
	// identifier/literal scalar.
	ReasonUnsupportedOperand Reason = "operand is not a same-index slice, range value, or plain scalar"
	// ReasonUnsupportedType covers an element type simd has no vector type
	// for at all (for example, string).
	ReasonUnsupportedType Reason = "element type not supported by simd"
	// ReasonUnsupportedOp covers an operation simd doesn't implement for an
	// otherwise-supported element type (for example, int64 multiply).
	ReasonUnsupportedOp Reason = "operation not supported by simd for this element type"
	// ReasonMethodSkipped covers a loop inside a method, skipped unless
	// -methods is passed (see analysis.Analyze's doc comment).
	ReasonMethodSkipped Reason = "loops inside methods are skipped without -methods"
	// ReasonZeroFillSkipped covers dst[i] = 0: the compiler already turns
	// this into memclr, which beats a vector loop, so it's left alone on
	// purpose rather than for lack of support.
	ReasonZeroFillSkipped Reason = "zero fill is left to the compiler's memclr"
	// ReasonUnrecognized is used when a kernel or function has no reported
	// candidate loop at all: nothing in it looked like an indexed-store loop.
	ReasonUnrecognized Reason = "not recognized"
)
