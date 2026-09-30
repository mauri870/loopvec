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

// setupBaseline is for the kernels TSVC_2's initialise_arrays has no branch for.
// They run on the arrays init set at startup, which NewArrays reproduces.
func setupBaseline(x *Arrays) {}

// setupS118 matches TSVC_2's initialise_arrays for s118: a = 1, bb =
// 0.000001.
func setupS118(x *Arrays) {
	fill(x.A, 1)
	fillMatrix(x.BB, 0.000001)
}

// setupS119 matches TSVC_2's initialise_arrays for s119, s231: aa = 1, bb =
// recipSq per row.
func setupS119(x *Arrays) {
	fillMatrix(x.AA, 1)
	recipSqMatrix(x.BB)
}

// setupS124 matches TSVC_2's initialise_arrays for s124: a = 0, b = 1, c = 1,
// d = recip, e = recip.
func setupS124(x *Arrays) {
	fill(x.A, 0)
	fill(x.B, 1)
	fill(x.C, 1)
	recip(x.D)
	recip(x.E)
}

// setupS125 matches TSVC_2's initialise_arrays for s125: flat_2d_array = 0,
// aa = 1, bb = 0.5, cc = 2.
func setupS125(x *Arrays) {
	fill(x.FLAT_2D_ARRAY, 0)
	fillMatrix(x.AA, 1)
	fillMatrix(x.BB, 0.5)
	fillMatrix(x.CC, 2)
}

// setupS127 matches TSVC_2's initialise_arrays for s127, s211, s243, s251: a
// = 0, b = 1, c = recip, d = recip, e = recip.
func setupS127(x *Arrays) {
	fill(x.A, 0)
	fill(x.B, 1)
	recip(x.C)
	recip(x.D)
	recip(x.E)
}

// setupS128 matches TSVC_2's initialise_arrays for s128: a = 0, b = 2, c = 1,
// d = 1.
func setupS128(x *Arrays) {
	fill(x.A, 0)
	fill(x.B, 2)
	fill(x.C, 1)
	fill(x.D, 1)
}

// setupS132 matches TSVC_2's initialise_arrays for s132: aa = 1, b = recip, c
// = recip.
func setupS132(x *Arrays) {
	fillMatrix(x.AA, 1)
	recip(x.B)
	recip(x.C)
}

// setupS141 matches TSVC_2's initialise_arrays for s141: flat_2d_array = 1,
// bb = recipSq per row.
func setupS141(x *Arrays) {
	fill(x.FLAT_2D_ARRAY, 1)
	recipSqMatrix(x.BB)
}

// setupS222 matches TSVC_2's initialise_arrays for s222, s252: a = 0, b = 1,
// c = 1.
func setupS222(x *Arrays) {
	fill(x.A, 0)
	fill(x.B, 1)
	fill(x.C, 1)
}

// setupS235 matches TSVC_2's initialise_arrays for s235: a = 1, b = recip, c
// = recip, aa = 1, bb = recipSq per row.
func setupS235(x *Arrays) {
	fill(x.A, 1)
	recip(x.B)
	recip(x.C)
	fillMatrix(x.AA, 1)
	recipSqMatrix(x.BB)
}

// setupS241 matches TSVC_2's initialise_arrays for s241: a = 1, b = 1, c = 1,
// d = 1.
func setupS241(x *Arrays) {
	fill(x.A, 1)
	fill(x.B, 1)
	fill(x.C, 1)
	fill(x.D, 1)
}

// setupS254 matches TSVC_2's initialise_arrays for s254, s255, s291: a = 0, b
// = 1.
func setupS254(x *Arrays) {
	fill(x.A, 0)
	fill(x.B, 1)
}

// setupS257 matches TSVC_2's initialise_arrays for s257: a = 1, aa = 2, bb =
// 1.
func setupS257(x *Arrays) {
	fill(x.A, 1)
	fillMatrix(x.AA, 2)
	fillMatrix(x.BB, 1)
}

// setupS273 matches TSVC_2's initialise_arrays for s273: a = 1, b = 1, c = 1,
// d = 0.000001, e = recip.
func setupS273(x *Arrays) {
	fill(x.A, 1)
	fill(x.B, 1)
	fill(x.C, 1)
	fill(x.D, 0.000001)
	recip(x.E)
}

// setupS276 matches TSVC_2's initialise_arrays for s276: a = 1, b = recip, c
// = recip, d = recip.
func setupS276(x *Arrays) {
	fill(x.A, 1)
	recip(x.B)
	recip(x.C)
	recip(x.D)
}

// setupS293 matches TSVC_2's initialise_arrays for s293, s311, s314, s315,
// s316, s3111, vsumr: a = recip.
func setupS293(x *Arrays) {
	recip(x.A)
}

// setupS2101 matches TSVC_2's initialise_arrays for s2101: aa = 1, bb = recip
// per row, cc = recip per row.
func setupS2101(x *Arrays) {
	fillMatrix(x.AA, 1)
	recipMatrix(x.BB)
	recipMatrix(x.CC)
}

// setupS2102 matches TSVC_2's initialise_arrays for s2102: aa = 0.
func setupS2102(x *Arrays) {
	fillMatrix(x.AA, 0)
}

// setupS312 matches TSVC_2's initialise_arrays for s312: a = 1.000001.
func setupS312(x *Arrays) {
	fill(x.A, 1.000001)
}

// setupS313 matches TSVC_2's initialise_arrays for s313, s352, vdotr: a =
// recip, b = recip.
func setupS313(x *Arrays) {
	recip(x.A)
	recip(x.B)
}

// setupS319 matches TSVC_2's initialise_arrays for s319: a = 0, b = 0, c =
// recip, d = recip, e = recip.
func setupS319(x *Arrays) {
	fill(x.A, 0)
	fill(x.B, 0)
	recip(x.C)
	recip(x.D)
	recip(x.E)
}

// setupS3113 matches TSVC_2's initialise_arrays for s3113: a = recip,
// a[LEN_1D-1] = -2.
func setupS3113(x *Arrays) {
	recip(x.A)
	x.A[Len1D-1] = -2
}

// setupS331 matches TSVC_2's initialise_arrays for s331: a = recip,
// a[LEN_1D-1] = -1.
func setupS331(x *Arrays) {
	recip(x.A)
	x.A[Len1D-1] = -1
}

// setupS351 matches TSVC_2's initialise_arrays for s351: a = 1, b = 1, c[0] =
// 1.
func setupS351(x *Arrays) {
	fill(x.A, 1)
	fill(x.B, 1)
	x.C[0] = 1
}

// setupS421 matches TSVC_2's initialise_arrays for s421: a = recipSq,
// flat_2d_array[:len1d] = 1.
func setupS421(x *Arrays) {
	recipSq(x.A)
	fill(x.FLAT_2D_ARRAY[:Len1D], 1)
}

// setupS1421 matches TSVC_2's initialise_arrays for s1421: b = 1.
func setupS1421(x *Arrays) {
	fill(x.B, 1)
}

// setupS422 matches TSVC_2's initialise_arrays for s422, s424:
// flat_2d_array[:len1d] = 1, a = recipSq, flat_2d_array[:len1d] = 0.
func setupS422(x *Arrays) {
	fill(x.FLAT_2D_ARRAY[:Len1D], 1)
	recipSq(x.A)
	fill(x.FLAT_2D_ARRAY[:Len1D], 0)
}

// setupS423 matches TSVC_2's initialise_arrays for s423:
// flat_2d_array[:len1d] = 0, a = recipSq, flat_2d_array[:len1d] = 1.
func setupS423(x *Arrays) {
	fill(x.FLAT_2D_ARRAY[:Len1D], 0)
	recipSq(x.A)
	fill(x.FLAT_2D_ARRAY[:Len1D], 1)
}

// setupS441 matches TSVC_2's initialise_arrays for s441: a = 1, b = recip, c
// = recip, d[:len1d/3] = -1, d[len1d/3 : len1d/3+len1d/3] = 0, d[2*len1d/3 :
// 2*len1d/3+len1d/3+1] = 1.
func setupS441(x *Arrays) {
	fill(x.A, 1)
	recip(x.B)
	recip(x.C)
	fill(x.D[:Len1D/3], -1)
	fill(x.D[Len1D/3:Len1D/3+Len1D/3], 0)
	fill(x.D[2*Len1D/3:2*Len1D/3+Len1D/3+1], 1)
}

// setupS452 matches TSVC_2's initialise_arrays for s452: a = 0, b = 1, c =
// 0.000001.
func setupS452(x *Arrays) {
	fill(x.A, 0)
	fill(x.B, 1)
	fill(x.C, 0.000001)
}

// setupS4117 matches TSVC_2's initialise_arrays for s4117: a = 0, b = 1, c =
// recip, d = recip.
func setupS4117(x *Arrays) {
	fill(x.A, 0)
	fill(x.B, 1)
	recip(x.C)
	recip(x.D)
}

// setupS315 matches TSVC_2's s315, which fills a before it runs the loop rather
// than in initialise_arrays: a[i] = (i*7) % Len1D.
func setupS315(x *Arrays) {
	for i := range x.A {
		x.A[i] = float32((i * 7) % Len1D)
	}
}
