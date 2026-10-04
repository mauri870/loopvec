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
	// ReasonUnsupportedStart covers a counting loop whose start is not a
	// loop-invariant integer expression (a constant, an int variable, len of a
	// slice, and sums and products of those), or is a negative constant.
	ReasonUnsupportedStart Reason = "loop start is not a loop-invariant integer expression"
	// ReasonUnsupportedCondition covers a loop condition that is not
	// i < limit or i <= limit, or i >= limit or i > limit when counting down.
	ReasonUnsupportedCondition Reason = "loop condition is not i < limit, i <= limit, i >= limit, or i > limit"
	// ReasonUnsupportedStep covers a step other than i++, or i-- when
	// counting down.
	ReasonUnsupportedStep Reason = "loop step is not i++ (or i-- when counting down)"
	// ReasonUnsupportedBound covers a limit that is not a loop-invariant integer
	// expression, is a constant that is not positive, or is written so that
	// adding one to it could overflow (a <= limit, or the start of a loop that
	// counts down, that is not a constant or something minus a constant).
	ReasonUnsupportedBound Reason = "loop limit is not a loop-invariant integer expression"
	// ReasonUnsupportedDestination covers a destination that isn't a plain
	// slice[i] element: an offset (dst[i+1]), a stride (dst[2*i]), or a
	// nested index (dst[i][j]).
	ReasonUnsupportedDestination Reason = "destination is not a simple, same-index slice element"
	// ReasonOffsetOfStored covers a slice that is read at an offset from the loop
	// index (a[i+1]) and also written in the loop, in a way the dependence test
	// cannot clear: the offset is not a constant, or the loop counts down.
	ReasonOffsetOfStored Reason = "a slice is read at an offset and written in the same loop"
	// ReasonCarriedDependence covers a slice read at an earlier index than the
	// iteration writes (a[i] = a[i-1] + x): iteration i reads what iteration i-1
	// wrote, so the iterations cannot run as a vector.
	ReasonCarriedDependence Reason = "an iteration reads an element an earlier iteration wrote"
	// ReasonUnsupportedIf covers an if that is not a select: a condition that is
	// not a comparison of loop values (joined by && and ||), an init statement, or
	// a branch that is not one assignment to an element of the same slice.
	ReasonUnsupportedIf Reason = "if is not a select between assignments to one element"
	// ReasonConditionalStore covers an if without an else whose condition does not
	// read the element it stores. The vector loop would write the unchanged lanes
	// back, which the scalar loop never touches; when the condition reads the
	// element anyway the loop already depends on it, so the write-back changes
	// nothing a correct program can observe.
	ReasonConditionalStore Reason = "a store under a condition may write elements the scalar loop leaves alone"
	// ReasonConditionalLoad covers a slice read only under a condition: the vector
	// loop reads it for every element, so a short slice would panic where the
	// scalar loop did not.
	ReasonConditionalLoad Reason = "a slice is read only under a condition"
	// ReasonUnsupportedBody covers a loop body that isn't a sequence of
	// assignments: an if, a nested loop, or any other statement.
	ReasonUnsupportedBody Reason = "loop body is not a sequence of indexed assignments"
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
	// ReasonLiveTemp covers a local declared before the loop and assigned in it
	// that is also used outside the loop: the vector loop would not leave it
	// holding the last iteration's value.
	ReasonLiveTemp Reason = "a local assigned in the loop is used outside it"
	// ReasonUnsupportedReduction covers a fold into a local with an operator
	// that cannot be regrouped, such as -=, /=, &^=, or a shift.
	ReasonUnsupportedReduction Reason = "loop accumulates with an operator that cannot be regrouped"
	// ReasonFloatReduction covers a floating-point sum or product: folding into
	// several vector lanes and combining them regroups the operations, and
	// float addition is not associative. It is rewritten with -fp-reassoc.
	ReasonFloatReduction Reason = "float reduction would combine the elements in a different order (-fp-reassoc allows it)"
	// ReasonMixedTypes covers a body whose assignments write slices of
	// different element types: one vector type is used for the whole loop.
	ReasonMixedTypes Reason = "assignments write slices of different element types"
	// ReasonTooManySlices covers a body that touches more slices than the
	// runtime overlap checks are budgeted for.
	ReasonTooManySlices Reason = "loop touches more than 8 distinct slices"
	// ReasonShortLoop covers a loop whose constant trip count is below the measured
	// break-even, 8 elements, or 4 full vectors for a reduction: the scalar loop is
	// faster than the vector loop plus its setup.
	ReasonShortLoop Reason = "constant trip count is too small for the vector loop to pay off"
	// ReasonFloatMinMax covers min and max of floating-point values: the simd
	// method is the hardware instruction, which on amd64 returns the second
	// operand for a NaN and does not order -0 below +0, where Go's min and max
	// propagate the NaN and order the zeros. arm64 matches Go, so the result
	// would depend on the machine.
	ReasonFloatMinMax Reason = "float min and max differ from Go's on some architectures (NaN and signed zero)"
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
	case ReasonUnsupportedOp, ReasonFloatMinMax, ReasonFloatReduction, ReasonMixedTypes, ReasonTooManySlices, ReasonShortLoop:
		return "plan"
	}
	return "lower"
}
