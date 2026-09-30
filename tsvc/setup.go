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

// setupVpv matches TSVC_2's initialise_arrays for va, vif, and vpv: a = 0,
// b = recipSq.
func setupVpv(x *Arrays) {
	fill(x.A, 0)
	recipSq(x.B)
}

// setupVtv matches TSVC_2's initialise_arrays for vtv: a = 1, b = 1.
func setupVtv(x *Arrays) {
	fill(x.A, 1)
	fill(x.B, 1)
}

// setupVpvtv matches TSVC_2's initialise_arrays for vpvtv: a = 1, b and c =
// recip.
func setupVpvtv(x *Arrays) {
	fill(x.A, 1)
	recip(x.B)
	recip(x.C)
}

// setupVpvts matches TSVC_2's initialise_arrays for vpvts: a = 1, b = recipSq.
func setupVpvts(x *Arrays) {
	fill(x.A, 1)
	recipSq(x.B)
}

// setupVpvpv matches TSVC_2's initialise_arrays for vpvpv: a = recipSq, b = 1,
// c = -1.
func setupVpvpv(x *Arrays) {
	recipSq(x.A)
	fill(x.B, 1)
	fill(x.C, -1)
}

// setupVtvtv matches TSVC_2's initialise_arrays for vtvtv: a = 1, b = 2,
// c = 0.5.
func setupVtvtv(x *Arrays) {
	fill(x.A, 1)
	fill(x.B, 2)
	fill(x.C, 0.5)
}

// setupVbor matches TSVC_2's initialise_arrays for vbor: a through e = recip,
// aa = recip per row. TSVC_2 passes a value to set_1d_array for c, d, and e,
// but a negative stride selects the reciprocal fill and ignores it.
func setupVbor(x *Arrays) {
	recip(x.A)
	recip(x.B)
	recip(x.C)
	recip(x.D)
	recip(x.E)
	recipMatrix(x.AA)
}
