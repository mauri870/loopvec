package loopir

// Reason explains why a candidate loop was not vectorized. The vocabulary is
// deliberately small and coarse: it names which stage of matching first
// rejected the loop, not the exact AST shape that failed. The same strings
// are meant to back a future -explain mode.
type Reason string

const (
	// ReasonNotCountedLoop covers a for statement that is not a counted loop
	// at all: a missing init, condition, or post, or an init that does not
	// declare a single loop variable.
	ReasonNotCountedLoop Reason = "loop is not a counted for loop (init; condition; post)"
	// ReasonUnsupportedRange covers a range statement that is not one of
	// for i := range s, for i, v := range s, or for i := range n: a key that
	// is not a new identifier, or a value variable on an integer range.
	ReasonUnsupportedRange Reason = "range clause is not i := range s, i, v := range s, or i := range n"
	// ReasonUnsupportedStart covers a counting loop that does not start at 0,
	// or at len(s)-1 when counting down.
	ReasonUnsupportedStart Reason = "loop start is not 0 (or len(s)-1 when counting down)"
	// ReasonUnsupportedCondition covers a loop condition that is not
	// i < limit, or i >= limit when counting down.
	ReasonUnsupportedCondition Reason = "loop condition is not i < limit (or i >= 0 when counting down)"
	// ReasonUnsupportedStep covers a step other than i++, or i-- when
	// counting down.
	ReasonUnsupportedStep Reason = "loop step is not i++ (or i-- when counting down)"
	// ReasonUnsupportedBound covers a limit that is not len(slice), an int
	// variable, or a positive integer constant (or 0 when counting down).
	ReasonUnsupportedBound Reason = "loop limit is not len(slice), an int variable, or a positive integer constant"
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

// Stage names the part of loopvec that produced r: "analysis" for a loop
// skipped before it is examined, "plan" for an operation simd cannot express,
// and "lower" for everything about the loop's own shape.
func (r Reason) Stage() string {
	switch r {
	case ReasonMethodSkipped:
		return "analysis"
	case ReasonUnsupportedOp:
		return "plan"
	}
	return "lower"
}
