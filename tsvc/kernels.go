package tsvc

// s000: linear dependence testing, no dependence.
//
//tsvc:kernel category=dependence reps=2 setup=s000 checksum=a exact=bits
func s000(a, b []float32) {
	for i := range a {
		a[i] = b[i] + 1
	}
}

// s111: no dependence, stride 2.
//
//tsvc:kernel category=dependence reps=2 setup=s111 checksum=a exact=bits
func s111(a, b []float32) {
	for i := 1; i < len(a); i += 2 {
		a[i] = a[i-1] + b[i]
	}
}

// s1111: no dependence, jump in data access.
//
//tsvc:kernel category=dependence reps=2 setup=s111 checksum=a exact=fused
func s1111(a, b, c, d []float32) {
	for i := 0; i < len(a)/2; i++ {
		a[2*i] = c[i]*b[i] + d[i]*b[i] + c[i]*c[i] + d[i]*b[i] + d[i]*c[i]
	}
}

// s112: loop reversal.
//
//tsvc:kernel category=dependence reps=3 setup=s112 checksum=a exact=bits
func s112(a, b []float32) {
	for i := len(a) - 2; i >= 0; i-- {
		a[i+1] = a[i] + b[i]
	}
}

// s1112: loop reversal, no dependence.
//
//tsvc:kernel category=dependence reps=3 setup=s112 checksum=a exact=bits
func s1112(a, b []float32) {
	for i := len(a) - 1; i >= 0; i-- {
		a[i] = b[i] + 1
	}
}

// s113: a[i] = a[0] with no actual dependence cycle.
//
//tsvc:kernel category=dependence reps=4 setup=s113 checksum=a exact=bits
func s113(a, b []float32) {
	for i := 1; i < len(a); i++ {
		a[i] = a[0] + b[i]
	}
}

// s1113: one-iteration dependency on a[len(a)/2], still vectorizable.
//
//tsvc:kernel category=dependence reps=2 setup=s113 checksum=a exact=bits
func s1113(a, b []float32) {
	for i := range a {
		a[i] = a[len(a)/2] + b[i]
	}
}

// s114: transpose, jump in data access.
//
//tsvc:kernel category=dependence reps=1 setup=s114 checksum=aa exact=bits
func s114(aa, bb [][]float32) {
	for i := range aa {
		for j := range i {
			aa[i][j] = aa[j][i] + bb[i][j]
		}
	}
}

// s115: triangular saxpy loop.
//
//tsvc:kernel category=dependence reps=1 setup=s115 checksum=a exact=fused
func s115(a []float32, aa [][]float32) {
	for j := range aa {
		for i := j + 1; i < len(aa); i++ {
			a[i] -= aa[j][i] * a[j]
		}
	}
}

// s1115: triangular saxpy loop with a transposed read.
//
//tsvc:kernel category=dependence reps=1 setup=s115 checksum=aa exact=fused
func s1115(aa, bb, cc [][]float32) {
	for i := range aa {
		for j := range aa {
			aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
		}
	}
}
