package tsvc

// Expectation says what loopvec should do with a kernel. It separates a loop
// loopvec cannot vectorize yet from one it must leave alone, so a coverage
// count does not reward vectorizing a loop that has a real dependence.
type Expectation int

const (
	// ExpectVectorize marks a loop a vectorizer can vectorize. Leaving it alone
	// is a gap in loopvec.
	ExpectVectorize Expectation = iota
	// ExpectDecline marks a loop loopvec must leave alone: it carries a
	// dependence between iterations, or it needs a shape loopvec does not
	// support on purpose, such as a gather. Vectorizing it is a bug, not an
	// improvement.
	ExpectDecline
	// ExpectSkip marks a loop that something else handles better, such as a
	// copy or a zero fill. Vectorizing it is a bug for the same reason.
	ExpectSkip
)

func (e Expectation) String() string {
	switch e {
	case ExpectVectorize:
		return "vectorize"
	case ExpectDecline:
		return "decline"
	case ExpectSkip:
		return "skip"
	default:
		return "unknown"
	}
}
