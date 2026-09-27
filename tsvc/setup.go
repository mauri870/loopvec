package tsvc

// fill sets every element of x to v, matching TSVC_2's set_1d_array with
// stride 1.
func fill(x []float32, v float32) {
	for i := range x {
		x[i] = v
	}
}

// recip sets x[i] = 1/(i+1), matching TSVC_2's SET1D_RECIP_IDX.
func recip(x []float32) {
	for i := range x {
		x[i] = 1 / float32(i+1)
	}
}

// recipSq sets x[i] = 1/(i+1)^2, matching TSVC_2's SET1D_RECIP_IDX_SQ.
func recipSq(x []float32) {
	for i := range x {
		x[i] = 1 / float32((i+1)*(i+1))
	}
}

func fillMatrix(m [][]float32, v float32) {
	for _, row := range m {
		fill(row, v)
	}
}

func recipMatrix(m [][]float32) {
	for _, row := range m {
		recip(row)
	}
}

func recipSqMatrix(m [][]float32) {
	for _, row := range m {
		recipSq(row)
	}
}

// setupS000 matches TSVC_2's initialise_arrays for s000: a = 1+i, b = 2+i,
// c = 3+i, d = 4+i, e = 5+i.
func setupS000(x *Arrays) {
	for i := range x.A {
		x.A[i] = float32(1 + i)
		x.B[i] = float32(2 + i)
		x.C[i] = float32(3 + i)
		x.D[i] = float32(4 + i)
		x.E[i] = float32(5 + i)
	}
}

// setupS111 matches TSVC_2's initialise_arrays for s111: a = 1, b..e = recipSq.
func setupS111(x *Arrays) {
	fill(x.A, 1)
	recipSq(x.B)
	recipSq(x.C)
	recipSq(x.D)
	recipSq(x.E)
}

// setupS112 matches TSVC_2's initialise_arrays for s112: a = 1, b = recipSq.
func setupS112(x *Arrays) {
	fill(x.A, 1)
	recipSq(x.B)
}

// setupS113 matches TSVC_2's initialise_arrays for s113: a = 1, b = recipSq.
func setupS113(x *Arrays) {
	fill(x.A, 1)
	recipSq(x.B)
}

// setupS114 matches TSVC_2's initialise_arrays for s114: aa = recip per row,
// bb = recipSq per row.
func setupS114(x *Arrays) {
	recipMatrix(x.AA)
	recipSqMatrix(x.BB)
}

// setupS115 matches TSVC_2's initialise_arrays for s115: a = 1;
// aa, bb, cc = 0.000001.
func setupS115(x *Arrays) {
	fill(x.A, 1)
	fillMatrix(x.AA, 0.000001)
	fillMatrix(x.BB, 0.000001)
	fillMatrix(x.CC, 0.000001)
}
