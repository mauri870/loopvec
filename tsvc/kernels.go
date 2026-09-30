package tsvc

// s000: linear dependence testing, no dependence.
//
//tsvc:kernel category=dependence reps=2 setup=s000 checksum=a exact=bits expect=vectorize
func s000(a, b []float32) {
	for i := range a {
		a[i] = b[i] + 1
	}
}

// s111: no dependence, stride 2 (a[i] reads only even elements, which are never
// written).
//
//tsvc:kernel category=dependence reps=2 setup=s111 checksum=a exact=bits expect=vectorize
func s111(a, b []float32) {
	for i := 1; i < len(a); i += 2 {
		a[i] = a[i-1] + b[i]
	}
}

// s1111: no dependence, jump in data access.
//
//tsvc:kernel category=dependence reps=2 setup=s111 checksum=a exact=fused expect=vectorize
func s1111(a, b, c, d []float32) {
	for i := 0; i < len(a)/2; i++ {
		a[2*i] = c[i]*b[i] + d[i]*b[i] + c[i]*c[i] + d[i]*b[i] + d[i]*c[i]
	}
}

// s112: loop reversal. Counting down, a[i+1] is written after a[i+1] was read by
// the previous iteration, so the dependence is an anti-dependence and reading a
// whole vector before writing it gives the scalar result.
//
//tsvc:kernel category=dependence reps=3 setup=s112 checksum=a exact=bits expect=vectorize
func s112(a, b []float32) {
	for i := len(a) - 2; i >= 0; i-- {
		a[i+1] = a[i] + b[i]
	}
}

// s1112: loop reversal, no dependence.
//
//tsvc:kernel category=dependence reps=3 setup=s112 checksum=a exact=bits expect=vectorize
func s1112(a, b []float32) {
	for i := len(a) - 1; i >= 0; i-- {
		a[i] = b[i] + 1
	}
}

// s113: a[i] = a[0] with no actual dependence cycle: a[0] is never written
// because the loop starts at 1.
//
//tsvc:kernel category=dependence reps=4 setup=s113 checksum=a exact=bits expect=vectorize
func s113(a, b []float32) {
	for i := 1; i < len(a); i++ {
		a[i] = a[0] + b[i]
	}
}

// s1113: one-iteration dependency on a[len(a)/2], still vectorizable. Iterations
// after len(a)/2 read the value it wrote, so vectorizing means splitting the loop
// at that index.
//
//tsvc:kernel category=dependence reps=2 setup=s113 checksum=a exact=bits expect=vectorize
func s1113(a, b []float32) {
	for i := range a {
		a[i] = a[len(a)/2] + b[i]
	}
}

// s114: transpose, jump in data access. Expected to be declined: the column read
// aa[j][i] is a gather, which loopvec does not do on purpose.
//
//tsvc:kernel category=dependence reps=1 setup=s114 checksum=aa exact=bits expect=decline
func s114(aa, bb [][]float32) {
	for i := range aa {
		for j := range i {
			aa[i][j] = aa[j][i] + bb[i][j]
		}
	}
}

// s115: triangular saxpy loop. The inner loop reads a[j], which it never writes
// because it only touches i > j, so it is a plain axpy.
//
//tsvc:kernel category=dependence reps=1 setup=s115 checksum=a exact=fused expect=vectorize
func s115(a []float32, aa [][]float32) {
	for j := range aa {
		for i := j + 1; i < len(aa); i++ {
			a[i] -= aa[j][i] * a[j]
		}
	}
}

// s1115: triangular saxpy loop with a transposed read. Expected to be declined:
// cc[j][i] is a column read, a gather.
//
//tsvc:kernel category=dependence reps=1 setup=s115 checksum=aa exact=fused expect=decline
func s1115(aa, bb, cc [][]float32) {
	for i := range aa {
		for j := range aa {
			aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
		}
	}
}

// va: vector assignment. Expected to be skipped: a copy is what copy() is for.
//
//tsvc:kernel category=control reps=4 setup=vpv checksum=a exact=bits expect=skip
func va(a, b []float32) {
	for i := range a {
		a[i] = b[i]
	}
}

// vif: vector if. The store is conditional, which takes a masked store or a
// blend to vectorize.
//
//tsvc:kernel category=control reps=4 setup=vpv checksum=a exact=bits expect=vectorize
func vif(a, b []float32) {
	for i := range a {
		if b[i] > 0 {
			a[i] = b[i]
		}
	}
}

// vpv: vector plus vector.
//
//tsvc:kernel category=control reps=4 setup=vpv checksum=a exact=bits expect=vectorize
func vpv(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// vtv: vector times vector.
//
//tsvc:kernel category=control reps=4 setup=vtv checksum=a exact=bits expect=vectorize
func vtv(a, b []float32) {
	for i := range a {
		a[i] *= b[i]
	}
}

// vpvtv: vector plus vector times vector.
//
//tsvc:kernel category=control reps=4 setup=vpvtv checksum=a exact=fused expect=vectorize
func vpvtv(a, b, c []float32) {
	for i := range a {
		a[i] += b[i] * c[i]
	}
}

// vpvts: vector plus vector times scalar. TSVC_2 passes s1 (1.0) through a
// void pointer and reads it back as an int, which yields 1065353216; the port
// uses the intended value.
//
//tsvc:kernel category=control reps=4 setup=vpvts checksum=a exact=fused expect=vectorize
func vpvts(a, b []float32, s1 float32) {
	for i := range a {
		a[i] += b[i] * s1
	}
}

// vpvpv: vector plus vector plus vector.
//
//tsvc:kernel category=control reps=4 setup=vpvpv checksum=a exact=bits expect=vectorize
func vpvpv(a, b, c []float32) {
	for i := range a {
		a[i] += b[i] + c[i]
	}
}

// vtvtv: vector times vector times vector.
//
//tsvc:kernel category=control reps=4 setup=vtvtv checksum=a exact=bits expect=vectorize
func vtvtv(a, b, c []float32) {
	for i := range a {
		a[i] = a[i] * b[i] * c[i]
	}
}

// vbor: basic operations rate, all combinations of three of six values: 59
// flops for 6 loads and 1 store. The body is several statements, so it takes
// multi-statement support to vectorize.
//
//tsvc:kernel category=control reps=4 setup=vbor checksum=x exact=fused expect=vectorize
func vbor(a, b, c, d, e []float32, aa [][]float32, x []float32) {
	for i := range Len2D {
		a1, b1, c1, d1, e1, f1 := a[i], b[i], c[i], d[i], e[i], aa[0][i]
		a1 = a1*b1*c1 + a1*b1*d1 + a1*b1*e1 + a1*b1*f1 +
			a1*c1*d1 + a1*c1*e1 + a1*c1*f1 + a1*d1*e1 +
			a1*d1*f1 + a1*e1*f1
		b1 = b1*c1*d1 + b1*c1*e1 + b1*c1*f1 + b1*d1*e1 +
			b1*d1*f1 + b1*e1*f1
		c1 = c1*d1*e1 + c1*d1*f1 + c1*e1*f1
		d1 = d1 * e1 * f1
		x[i] = a1 * b1 * c1 * d1
	}
}
