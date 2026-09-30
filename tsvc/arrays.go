package tsvc

// Len1D and Len2D match TSVC_2's LEN_1D and LEN_2D.
const (
	Len1D = 32000
	Len2D = 256
)

// Arrays holds the operands shared by the kernels, named as TSVC_2 names them.
// Each 1-D field has Len1D elements; each 2-D field is Len2D rows of Len2D
// columns backed by one contiguous slice; FLAT_2D_ARRAY is TSVC_2's
// flat_2d_array, Len2D*Len2D elements. S1 and S2 are the scalars, and N1 and N3
// the integers, that TSVC_2's main passes to some kernels. R receives the value
// a kernel returns, such as a reduction.
type Arrays struct {
	A, B, C, D, E, X []float32
	AA, BB, CC       [][]float32
	FLAT_2D_ARRAY    []float32
	S1, S2           float32
	N1, N3           int
	R                float32
}

// NewArrays allocates a fresh Arrays holding the values TSVC_2's init function
// sets once at startup: 1 in a through e and x, 1/(j+1) along every row of aa,
// bb, and cc, and zero in flat_2d_array. TSVC_2 lets each kernel's own
// initialisation override some of them and leaves the rest, along with whatever
// the previous kernel left behind; a kernel here starts from the baseline alone.
// Callers run a kernel's Setup before its Run.
func NewArrays() *Arrays {
	x := &Arrays{
		A: make([]float32, Len1D),
		B: make([]float32, Len1D),
		C: make([]float32, Len1D),
		D: make([]float32, Len1D),
		E: make([]float32, Len1D),
		X: make([]float32, Len1D),

		AA: newMatrix(Len2D, Len2D),
		BB: newMatrix(Len2D, Len2D),
		CC: newMatrix(Len2D, Len2D),

		FLAT_2D_ARRAY: make([]float32, Len2D*Len2D),

		S1: 1,
		S2: 2,
		N1: 1,
		N3: 1,
	}
	for _, v := range [][]float32{x.A, x.B, x.C, x.D, x.E, x.X} {
		fill(v, 1)
	}
	for _, m := range [][][]float32{x.AA, x.BB, x.CC} {
		recipMatrix(m)
	}
	return x
}

func newMatrix(rows, cols int) [][]float32 {
	backing := make([]float32, rows*cols)
	m := make([][]float32, rows)
	for i := range m {
		m[i] = backing[i*cols : (i+1)*cols]
	}
	return m
}
