package tsvc

// Len1D and Len2D match TSVC_2's LEN_1D and LEN_2D.
const (
	Len1D = 32000
	Len2D = 256
)

// Arrays holds the operands shared by the kernels. Each 1-D field has Len1D
// elements; each 2-D field is Len2D rows of Len2D columns backed by one
// contiguous slice.
type Arrays struct {
	A, B, C, D, E []float32
	AA, BB, CC    [][]float32
}

// NewArrays allocates a fresh, zeroed Arrays. Callers run a kernel's Setup
// before its Run.
func NewArrays() *Arrays {
	return &Arrays{
		A: make([]float32, Len1D),
		B: make([]float32, Len1D),
		C: make([]float32, Len1D),
		D: make([]float32, Len1D),
		E: make([]float32, Len1D),

		AA: newMatrix(Len2D, Len2D),
		BB: newMatrix(Len2D, Len2D),
		CC: newMatrix(Len2D, Len2D),
	}
}

func newMatrix(rows, cols int) [][]float32 {
	backing := make([]float32, rows*cols)
	m := make([][]float32, rows)
	for i := range m {
		m[i] = backing[i*cols : (i+1)*cols]
	}
	return m
}
