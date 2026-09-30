package tsvc

// Category groups kernels by the TSVC_2 section they come from.
type Category int

const (
	CategoryDependence Category = iota
	// CategoryControl is TSVC_2's "control loops": the plain vector operations
	// (assignment, plus, times, and combinations) whose rate the other loops are
	// judged against.
	CategoryControl
)
