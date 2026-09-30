package tsvc

import "math"

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

// s1115: triangular saxpy loop with a transposed read. clang vectorizes it, so
// it is a gap, not a decline, although cc[j][i] is a column read.
//
//tsvc:kernel category=dependence reps=1 setup=s115 checksum=aa exact=fused expect=vectorize
func s1115(aa, bb, cc [][]float32) {
	for i := range aa {
		for j := range aa {
			aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
		}
	}
}

// s118: linear dependence testing, potential dot product recursion.
// The inner loop is a dot product with a[i-j-1] on the same array.
//
//tsvc:kernel category=dependence reps=2 setup=s118 checksum=a exact=fused expect=vectorize
func s118(a []float32, bb [][]float32) {
	for i := 1; i < len(bb); i++ {
		for j := 0; j <= i-1; j++ {
			a[i] += bb[j][i] * a[i-j-1]
		}
	}
}

// s119: linear dependence testing, no dependence - vectorizable.
//
//tsvc:kernel category=dependence reps=2 setup=s119 checksum=aa exact=bits expect=vectorize
func s119(aa, bb [][]float32) {
	for i := 1; i < len(aa); i++ {
		for j := 1; j < len(aa); j++ {
			aa[i][j] = aa[i-1][j-1] + bb[i][j]
		}
	}
}

// s1119: linear dependence testing, no dependence - vectorizable.
//
//tsvc:kernel category=dependence reps=2 setup=baseline checksum=aa exact=bits expect=vectorize
func s1119(aa, bb [][]float32) {
	for i := 1; i < len(aa); i++ {
		for j := range aa {
			aa[i][j] = aa[i-1][j] + bb[i][j]
		}
	}
}

// s121: induction variable recognition, loop with possible ambiguity because of
// scalar store.
//
//tsvc:kernel category=induction reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s121(a, b []float32) {
	var j int
	for i := 0; i < len(a)-1; i++ {
		j = i + 1
		a[i] = a[j] + b[i]
	}
}

// s122: induction variable recognition, variable lower and upper bound, and
// stride, reverse data access and jump in data access.
// n1 and n3 are the integers TSVC_2's main passes, both 1.
//
//tsvc:kernel category=induction reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s122(a, b []float32, n1, n3 int) {
	var j, k int
	j = 1
	k = 0
	for i := n1 - 1; i < len(a); i += n3 {
		k += j
		a[i] += b[len(a)-k]
	}
}

// s124: induction variable recognition, induction variable under both sides of
// if (same value).
//
//tsvc:kernel category=induction reps=2 setup=s124 checksum=a exact=fused expect=vectorize
func s124(a, b, c, d, e []float32) {
	j := -1
	for i := range a {
		if b[i] > 0 {
			j++
			a[j] = b[i] + d[i]*e[i]
		} else {
			j++
			a[j] = c[i] + d[i]*e[i]
		}
	}
}

// s125: induction variable recognition, induction variable in two loops;
// collapsing possible.
//
//tsvc:kernel category=induction reps=2 setup=s125 checksum=flat_2d_array exact=fused expect=vectorize
func s125(aa, bb, cc [][]float32, flat_2d_array []float32) {
	k := -1
	for i := range aa {
		for j := range aa {
			k++
			flat_2d_array[k] = aa[i][j] + bb[i][j]*cc[i][j]
		}
	}
}

// s127: induction variable recognition, induction variable with multiple
// increments.
//
//tsvc:kernel category=induction reps=2 setup=s127 checksum=a exact=fused expect=vectorize
func s127(a, b, c, d, e []float32) {
	j := -1
	for i := 0; i < len(a)/2; i++ {
		j++
		a[j] = b[i] + c[i]*d[i]
		j++
		a[j] = b[i] + d[i]*e[i]
	}
}

// s128: induction variables, coupled induction variables, jump in data access.
//
//tsvc:kernel category=induction reps=2 setup=s128 checksum=a+b exact=bits expect=vectorize
func s128(a, b, c, d []float32) {
	var j, k int
	j = -1
	for i := 0; i < len(a)/2; i++ {
		k = j + 1
		a[i] = b[k] - d[i]
		j = k + 1
		b[k] = a[i] + c[k]
	}
}

// s131: global data flow analysis, forward substitution.
//
//tsvc:kernel category=dataflow reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s131(a, b []float32) {
	m := 1
	for i := 0; i < len(a)-1; i++ {
		a[i] = a[i+m] + b[i]
	}
}

// s132: global data flow analysis, loop with multiple dimension ambiguous
// subscripts.
//
//tsvc:kernel category=dataflow reps=2 setup=s132 checksum=aa exact=fused expect=vectorize
func s132(aa [][]float32, b, c []float32) {
	m := 0
	j := m
	k := m + 1
	for i := 1; i < len(aa); i++ {
		aa[j][i] = aa[k][i-1] + b[i]*c[1]
	}
}

// s141: nonlinear dependence testing, walk a row in a symmetric packed array,
// element a(i,j) for (int j>i) stored in location j*(j-1)/2+i.
//
//tsvc:kernel category=nonlinear reps=2 setup=s141 checksum=flat_2d_array exact=bits expect=vectorize
func s141(bb [][]float32, flat_2d_array []float32) {
	var k int
	for i := range bb {
		k = (i+1)*((i+1)-1)/2 + (i + 1) - 1
		for j := i; j < len(bb); j++ {
			flat_2d_array[k] += bb[j][i]
			k += j + 1
		}
	}
}

// s162: control flow, deriving assertions.
// TSVC_2 passes n1 through a void pointer as k.
//
//tsvc:kernel category=controlflow reps=2 setup=vpvtv checksum=a exact=fused expect=vectorize
func s162(a, b, c []float32, n1 int) {
	k := n1
	if k > 0 {
		for i := 0; i < len(a)-1; i++ {
			a[i] = a[i+k] + b[i]*c[i]
		}
	}
}

// s171: symbolics, symbolic dependence tests.
// TSVC_2 passes n1 through a void pointer as inc.
//
//tsvc:kernel category=symbolics reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s171(a, b []float32, n1 int) {
	inc := n1
	for i := range a {
		a[i*inc] += b[i]
	}
}

// s172: symbolics, vectorizable if n3 .ne. 0.
// n1 and n3 are the integers TSVC_2's main passes, both 1.
//
//tsvc:kernel category=symbolics reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s172(a, b []float32, n1, n3 int) {
	for i := n1 - 1; i < len(a); i += n3 {
		a[i] += b[i]
	}
}

// s173: symbolics, expression in loop bounds and subscripts.
//
//tsvc:kernel category=symbolics reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s173(a, b []float32) {
	k := len(a) / 2
	for i := 0; i < len(a)/2; i++ {
		a[i+k] = a[i] + b[i]
	}
}

// s174: symbolics, loop with subscript that may seem ambiguous.
// TSVC_2's main passes M = LEN_1D/2.
//
//tsvc:kernel category=symbolics reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s174(a, b []float32) {
	M := len(a) / 2
	for i := range M {
		a[i+M] = a[i] + b[i]
	}
}

// s175: symbolics, symbolic dependence tests.
// TSVC_2 passes n1 through a void pointer as inc.
//
//tsvc:kernel category=symbolics reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s175(a, b []float32, n1 int) {
	inc := n1
	for i := 0; i < len(a)-1; i += inc {
		a[i] = a[i+inc] + b[i]
	}
}

// s176: symbolics, convolution.
//
//tsvc:kernel category=symbolics reps=2 setup=vpvtv checksum=a exact=fused expect=vectorize
func s176(a, b, c []float32) {
	m := len(a) / 2
	for j := 0; j < len(a)/2; j++ {
		for i := range m {
			a[i] += b[i+m-j-1] * c[j]
		}
	}
}

// s211: statement reordering, statement reordering allows vectorization.
//
//tsvc:kernel category=reordering reps=2 setup=s127 checksum=a+b exact=fused expect=vectorize
func s211(a, b, c, d, e []float32) {
	for i := 1; i < len(a)-1; i++ {
		a[i] = b[i-1] + c[i]*d[i]
		b[i] = b[i+1] - e[i]*d[i]
	}
}

// s1221: run-time symbolic resolution.
//
//tsvc:kernel category=symbolics reps=2 setup=baseline checksum=a+b exact=bits expect=vectorize
func s1221(a, b []float32) {
	for i := 4; i < len(a); i++ {
		b[i] = b[i-4] + a[i]
	}
}

// s222: loop distribution, partial loop vectorizatio recurrence in middle.
//
//tsvc:kernel category=distribution reps=2 setup=s222 checksum=a+b exact=fused expect=vectorize
func s222(a, b, c, e []float32) {
	for i := 1; i < len(a); i++ {
		a[i] += b[i] * c[i]
		e[i] = e[i-1] * e[i-1]
		a[i] -= b[i] * c[i]
	}
}

// s231: loop interchange, loop with data dependency.
//
//tsvc:kernel category=interchange reps=2 setup=s119 checksum=aa exact=bits expect=vectorize
func s231(aa, bb [][]float32) {
	for i := range aa {
		for j := 1; j < len(aa); j++ {
			aa[j][i] = aa[j-1][i] + bb[j][i]
		}
	}
}

// s2233: loop interchange, interchanging with one of two inner loops.
//
//tsvc:kernel category=interchange reps=2 setup=baseline checksum=aa+bb exact=bits expect=vectorize
func s2233(aa, bb, cc [][]float32) {
	for i := 1; i < len(aa); i++ {
		for j := 1; j < len(aa); j++ {
			aa[j][i] = aa[j-1][i] + cc[j][i]
		}
		for j := 1; j < len(aa); j++ {
			bb[i][j] = bb[i-1][j] + cc[i][j]
		}
	}
}

// s235: loop interchanging, imperfectly nested loops.
//
//tsvc:kernel category=interchange reps=2 setup=s235 checksum=a+b exact=fused expect=vectorize
func s235(a, b, c []float32, aa, bb [][]float32) {
	for i := range aa {
		a[i] += b[i] * c[i]
		for j := 1; j < len(aa); j++ {
			aa[j][i] = aa[j-1][i] + bb[j][i]*a[i]
		}
	}
}

// s241: node splitting, preloading necessary to allow vectorization.
//
//tsvc:kernel category=nodesplitting reps=2 setup=s241 checksum=a+b exact=fused expect=vectorize
func s241(a, b, c, d []float32) {
	for i := 0; i < len(a)-1; i++ {
		a[i] = b[i] * c[i] * d[i]
		b[i] = a[i] * a[i+1] * d[i]
	}
}

// s243: node splitting, false dependence cycle breaking.
//
//tsvc:kernel category=nodesplitting reps=2 setup=s127 checksum=a+b exact=fused expect=vectorize
func s243(a, b, c, d, e []float32) {
	for i := 0; i < len(a)-1; i++ {
		a[i] = b[i] + c[i]*d[i]
		b[i] = a[i] + d[i]*e[i]
		a[i] = b[i] + a[i+1]*d[i]
	}
}

// s2244: node splitting, cycle with ture and anti dependency.
//
//tsvc:kernel category=nodesplitting reps=2 setup=baseline checksum=a+b exact=bits expect=vectorize
func s2244(a, b, c, e []float32) {
	for i := 0; i < len(a)-1; i++ {
		a[i+1] = b[i] + e[i]
		a[i] = b[i] + c[i]
	}
}

// s251: scalar and array expansion, scalar expansion.
//
//tsvc:kernel category=expansion reps=2 setup=s127 checksum=a exact=fused expect=vectorize
func s251(a, b, c, d []float32) {
	var s float32
	for i := range a {
		s = b[i] + c[i]*d[i]
		a[i] = s * s
	}
}

// s1251: scalar and array expansion, scalar expansion.
//
//tsvc:kernel category=expansion reps=2 setup=baseline checksum=a exact=fused expect=vectorize
func s1251(a, b, c, d, e []float32) {
	var s float32
	for i := range a {
		s = b[i] + c[i]
		b[i] = a[i] + d[i]
		a[i] = s * e[i]
	}
}

// s3251: scalar and array expansion, scalar expansion.
//
//tsvc:kernel category=expansion reps=2 setup=baseline checksum=a exact=fused expect=vectorize
func s3251(a, b, c, d, e []float32) {
	for i := 0; i < len(a)-1; i++ {
		a[i+1] = b[i] + c[i]
		b[i] = c[i] * e[i]
		d[i] = a[i] * e[i]
	}
}

// s252: scalar and array expansion, loop with ambiguous scalar temporary.
//
//tsvc:kernel category=expansion reps=2 setup=s222 checksum=a exact=bits expect=vectorize
func s252(a, b, c []float32) {
	var t, s float32
	t = 0
	for i := range a {
		s = b[i] * c[i]
		a[i] = s + t
		t = s
	}
}

// s254: scalar and array expansion, carry around variable.
//
//tsvc:kernel category=expansion reps=2 setup=s254 checksum=a exact=bits expect=vectorize
func s254(a, b []float32) {
	var x float32
	x = b[len(a)-1]
	for i := range a {
		a[i] = (b[i] + x) * 0.5
		x = b[i]
	}
}

// s255: scalar and array expansion, carry around variables, 2 levels.
//
//tsvc:kernel category=expansion reps=2 setup=s254 checksum=a exact=bits expect=vectorize
func s255(a, b []float32) {
	var x, y float32
	x = b[len(a)-1]
	y = b[len(a)-2]
	for i := range a {
		a[i] = (b[i] + x + y) * 0.333
		y = x
		x = b[i]
	}
}

// s257: scalar and array expansion, array expansion.
//
//tsvc:kernel category=expansion reps=2 setup=s257 checksum=a+aa exact=bits expect=vectorize
func s257(a []float32, aa, bb [][]float32) {
	for i := 1; i < len(aa); i++ {
		for j := range aa {
			a[i] = aa[j][i] - a[i-1]
			aa[j][i] = a[i] + bb[j][i]
		}
	}
}

// s271: control flow, loop with singularity handling.
//
//tsvc:kernel category=controlflow reps=2 setup=vpvtv checksum=a exact=fused expect=vectorize
func s271(a, b, c []float32) {
	for i := range a {
		if b[i] > 0 {
			a[i] += b[i] * c[i]
		}
	}
}

// s273: control flow, simple loop with dependent conditional.
//
//tsvc:kernel category=controlflow reps=2 setup=s273 checksum=a+b+c exact=fused expect=vectorize
func s273(a, b, c, d, e []float32) {
	for i := range a {
		a[i] += d[i] * e[i]
		if a[i] < 0 {
			b[i] += d[i] * e[i]
		}
		c[i] += a[i] * d[i]
	}
}

// s2275: loop distribution is needed to be able to interchange.
//
//tsvc:kernel category=distribution reps=2 setup=baseline checksum=aa exact=fused expect=vectorize
func s2275(a, b, c, d []float32, aa, bb, cc [][]float32) {
	for i := range aa {
		for j := range aa {
			aa[j][i] = aa[j][i] + bb[j][i]*cc[j][i]
		}
		a[i] = b[i] + c[i]*d[i]
	}
}

// s276: control flow, if test using loop index.
//
//tsvc:kernel category=controlflow reps=2 setup=s276 checksum=a exact=fused expect=vectorize
func s276(a, b, c, d []float32) {
	mid := len(a) / 2
	for i := range a {
		if i+1 < mid {
			a[i] += b[i] * c[i]
		} else {
			a[i] += b[i] * d[i]
		}
	}
}

// s1279: control flow, vector if/gotos.
//
//tsvc:kernel category=controlflow reps=2 setup=baseline checksum=a+b+c exact=fused expect=vectorize
func s1279(a, b, c, d, e []float32) {
	for i := range a {
		if a[i] < 0 {
			if b[i] > a[i] {
				c[i] += d[i] * e[i]
			}
		}
	}
}

// s2711: control flow, semantic if removal.
//
//tsvc:kernel category=controlflow reps=2 setup=vpvtv checksum=a exact=fused expect=vectorize
func s2711(a, b, c []float32) {
	for i := range a {
		if b[i] != 0 {
			a[i] += b[i] * c[i]
		}
	}
}

// s2712: control flow, if to elemental min.
//
//tsvc:kernel category=controlflow reps=2 setup=vpvtv checksum=a exact=fused expect=vectorize
func s2712(a, b, c []float32) {
	for i := range a {
		if a[i] > b[i] {
			a[i] += b[i] * c[i]
		}
	}
}

// s1281: crossing thresholds, index set splitting, reverse data access.
//
//tsvc:kernel category=thresholds reps=2 setup=baseline checksum=a+b exact=fused expect=vectorize
func s1281(a, b, c, d, e []float32) {
	var x float32
	for i := range a {
		x = b[i]*c[i] + a[i]*d[i] + e[i]
		a[i] = x - 1.0
		b[i] = x
	}
}

// s291: loop peeling, wrap around variable, 1 level.
//
//tsvc:kernel category=peeling reps=2 setup=s254 checksum=a exact=bits expect=vectorize
func s291(a, b []float32) {
	var im1 int
	im1 = len(a) - 1
	for i := range a {
		a[i] = (b[i] + b[im1]) * 0.5
		im1 = i
	}
}

// s293: loop peeling, a(i)=a(0) with actual dependence cycle, loop is
// vectorizable.
//
//tsvc:kernel category=peeling reps=2 setup=s293 checksum=a exact=bits expect=vectorize
func s293(a []float32) {
	for i := range a {
		a[i] = a[0]
	}
}

// s2101: diagonals, main diagonal calculation, jump in data access.
//
//tsvc:kernel category=diagonals reps=2 setup=s2101 checksum=aa exact=fused expect=vectorize
func s2101(aa, bb, cc [][]float32) {
	for i := range aa {
		aa[i][i] += bb[i][i] * cc[i][i]
	}
}

// s2102: diagonals, identity matrix, best results vectorize both inner and outer
// loops.
//
//tsvc:kernel category=diagonals reps=2 setup=s2102 checksum=aa exact=bits expect=vectorize
func s2102(aa [][]float32) {
	for i := range aa {
		for j := range aa {
			aa[j][i] = 0
		}
		aa[i][i] = 1
	}
}

// s311: reductions, sum reduction.
// TSVC_2 checks the unchanged array a here and passes sum to dummy only to
// keep it alive; the port returns the sum and checks it, like the other
// reductions.
//
//tsvc:kernel category=reductions reps=2 setup=s293 checksum=r exact=fused expect=vectorize
func s311(a []float32) float32 {
	var sum float32
	sum = 0
	for i := range a {
		sum += a[i]
	}
	return sum
}

// s312: reductions, product reduction.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=reductions reps=2 setup=s312 checksum=r exact=fused expect=vectorize
func s312(a []float32) float32 {
	var prod float32
	prod = 1
	for i := range a {
		prod *= a[i]
	}
	return prod
}

// s313: reductions, dot product.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=reductions reps=2 setup=s313 checksum=r exact=fused expect=vectorize
func s313(a, b []float32) float32 {
	var dot float32
	dot = 0
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// s314: reductions, if to max reduction.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=reductions reps=2 setup=s293 checksum=r exact=bits expect=vectorize
func s314(a []float32) float32 {
	var x float32
	x = a[0]
	for i := range a {
		if a[i] > x {
			x = a[i]
		}
	}
	return x
}

// s315: reductions, if to max with index reductio 1 dimension.
// TSVC_2 fills a inside the kernel; here that is Setup. Returns index + x + 1.
//
//tsvc:kernel category=reductions reps=2 setup=s315 checksum=r exact=bits expect=vectorize
func s315(a []float32) float32 {
	var x float32
	var index int
	x = a[0]
	index = 0
	for i := range a {
		if a[i] > x {
			x = a[i]
			index = i
		}
	}
	return float32(index) + x + 1
}

// s316: reductions, if to min reduction.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=reductions reps=2 setup=s293 checksum=r exact=bits expect=vectorize
func s316(a []float32) float32 {
	var x float32
	x = a[0]
	for i := 1; i < len(a); i++ {
		if a[i] < x {
			x = a[i]
		}
	}
	return x
}

// s317: reductions, product reductio vectorize with, 1. scalar expansion of
// factor, and product reduction, 2. closed form solution: q = factor**n.
// The product decays to a denormal in float32, as it does in TSVC_2, and sticks
// there, so its value depends on the order of the multiplications: a vectorized
// product will not match it.
//
//tsvc:kernel category=reductions reps=2 setup=baseline checksum=r exact=fused expect=vectorize
func s317() float32 {
	var q float32
	q = 1
	for range Len1D / 2 {
		q *= 0.99
	}
	return q
}

// s319: reductions, coupled reductions.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=reductions reps=2 setup=s319 checksum=r exact=fused expect=vectorize
func s319(a, b, c, d, e []float32) float32 {
	var sum float32
	sum = 0
	for i := range a {
		a[i] = c[i] + d[i]
		sum += a[i]
		b[i] = c[i] + e[i]
		sum += b[i]
	}
	return sum
}

// s3111: reductions, conditional sum reduction.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=reductions reps=2 setup=s293 checksum=r exact=fused expect=vectorize
func s3111(a []float32) float32 {
	var sum float32
	sum = 0
	for i := range a {
		if a[i] > 0 {
			sum += a[i]
		}
	}
	return sum
}

// s3113: reductions, maximum of absolute value.
// ABS is fabsf in TSVC_2. Returns the value TSVC_2 passes to dummy, which the
// checksum reads.
//
//tsvc:kernel category=reductions reps=2 setup=s3113 checksum=r exact=bits expect=vectorize
func s3113(a []float32) float32 {
	var max float32
	max = float32(math.Abs(float64(a[0])))
	for i := range a {
		if float32(math.Abs(float64(a[i]))) > max {
			max = float32(math.Abs(float64(a[i])))
		}
	}
	return max
}

// s331: search loops, if to last-1.
// Returns j+1, as TSVC_2 does.
//
//tsvc:kernel category=search reps=2 setup=s331 checksum=r exact=bits expect=vectorize
func s331(a []float32) float32 {
	j := -1
	for i := range a {
		if a[i] < 0 {
			j = i
		}
	}
	return float32(j + 1)
}

// s351: loop rerolling, unrolled saxpy.
//
//tsvc:kernel category=rerolling reps=2 setup=s351 checksum=a exact=fused expect=vectorize
func s351(a, b, c []float32) {
	alpha := c[0]
	for i := 0; i < len(a); i += 5 {
		a[i] += alpha * b[i]
		a[i+1] += alpha * b[i+1]
		a[i+2] += alpha * b[i+2]
		a[i+3] += alpha * b[i+3]
		a[i+4] += alpha * b[i+4]
	}
}

// s1351: induction pointer recognition.
// Go has no pointer arithmetic, so A, B, and C are slices that advance.
//
//tsvc:kernel category=induction reps=2 setup=baseline checksum=a exact=bits expect=vectorize
func s1351(a, b, c []float32) {
	A := a
	B := b
	C := c
	for range a {
		A[0] = B[0] + C[0]
		A, B, C = A[1:], B[1:], C[1:]
	}
}

// s352: loop rerolling, unrolled dot product.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=rerolling reps=2 setup=s313 checksum=r exact=fused expect=vectorize
func s352(a, b []float32) float32 {
	var dot float32
	dot = 0
	for i := 0; i < len(a); i += 5 {
		dot = dot + a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] + a[i+4]*b[i+4]
	}
	return dot
}

// s421: storage classes and equivalencing, equivalence- no overlap.
// xx and yy are slices of flat_2d_array, where TSVC_2 points them into it.
//
//tsvc:kernel category=equivalencing reps=2 setup=s421 checksum=xx0 exact=bits expect=vectorize
func s421(a, flat_2d_array []float32) {
	xx := flat_2d_array
	yy := xx
	for i := 0; i < len(a)-1; i++ {
		xx[i] = yy[i+1] + a[i]
	}
}

// s1421: storage classes and equivalencing, equivalence- no overlap.
// xx is a slice of the second half of b.
//
//tsvc:kernel category=equivalencing reps=2 setup=s1421 checksum=half_xx exact=bits expect=vectorize
func s1421(a, b []float32) {
	xx := b[len(b)/2:]
	for i := 0; i < len(b)/2; i++ {
		b[i] = xx[i] + a[i]
	}
}

// s422: storage classes and equivalencing, common and equivalence statement,
// anti-dependence, threshold of 4.
// xx is a slice of flat_2d_array, where TSVC_2 points it into it.
//
//tsvc:kernel category=equivalencing reps=2 setup=s422 checksum=xx4 exact=bits expect=vectorize
func s422(a, flat_2d_array []float32) {
	xx := flat_2d_array[4:]
	for i := range a {
		xx[i] = flat_2d_array[i+8] + a[i]
	}
}

// s423: storage classes and equivalencing, common and equivalenced variables -
// with anti-dependence, do this again here.
// xx is a slice of flat_2d_array, where TSVC_2 points it into it.
//
//tsvc:kernel category=equivalencing reps=2 setup=s423 checksum=flat_2d_array exact=bits expect=vectorize
func s423(a, flat_2d_array []float32) {
	vl := 64
	xx := flat_2d_array[vl:]
	for i := 0; i < len(a)-1; i++ {
		flat_2d_array[i+1] = xx[i] + a[i]
	}
}

// s424: storage classes and equivalencing, common and equivalenced variables -
// overlap, vectorizeable in strips of 64 or less, do this again here.
// xx is a slice of flat_2d_array, where TSVC_2 points it into it.
//
//tsvc:kernel category=equivalencing reps=2 setup=s422 checksum=xx63 exact=bits expect=vectorize
func s424(a, flat_2d_array []float32) {
	vl := 63
	xx := flat_2d_array[vl:]
	for i := 0; i < len(a)-1; i++ {
		xx[i+1] = flat_2d_array[i] + a[i]
	}
}

// s431: parameters, parameter statement.
//
//tsvc:kernel category=parameters reps=2 setup=vpvts checksum=a exact=bits expect=vectorize
func s431(a, b []float32) {
	k1 := 1
	k2 := 2
	k := 2*k1 - k2
	for i := range a {
		a[i] = a[i+k] + b[i]
	}
}

// s441: non-logical if's, arithmetic if.
//
//tsvc:kernel category=nonlogicalifs reps=2 setup=s441 checksum=a exact=fused expect=vectorize
func s441(a, b, c, d []float32) {
	for i := range a {
		if d[i] < 0 {
			a[i] += b[i] * c[i]
		} else if d[i] == 0 {
			a[i] += b[i] * b[i]
		} else {
			a[i] += c[i] * c[i]
		}
	}
}

// s443: non-logical if's, arithmetic if.
// TSVC_2 writes the branches with gotos; they are an if and an else.
//
//tsvc:kernel category=nonlogicalifs reps=2 setup=vpvtv checksum=a exact=fused expect=vectorize
func s443(a, b, c, d []float32) {
	for i := range a {
		if d[i] <= 0 {
			a[i] += b[i] * c[i]
		} else {
			a[i] += b[i] * b[i]
		}
	}
}

// s452: intrinsic functions, seq function.
//
//tsvc:kernel category=intrinsics reps=2 setup=s452 checksum=a exact=fused expect=vectorize
func s452(a, b, c []float32) {
	for i := range a {
		a[i] = b[i] + c[i]*float32(i+1)
	}
}

// s453: induction varibale recognition.
//
//tsvc:kernel category=induction reps=2 setup=vpv checksum=a exact=fused expect=vectorize
func s453(a, b []float32) {
	var s float32
	s = 0
	for i := range a {
		s += 2
		a[i] = s * b[i]
	}
}

// s4117: indirect addressing, seq function.
//
//tsvc:kernel category=indirect reps=2 setup=s4117 checksum=a exact=fused expect=vectorize
func s4117(a, b, c, d []float32) {
	for i := range a {
		a[i] = b[i] + c[i/2]*d[i]
	}
}

// va: control loops, vector assignment. A copy: loopvec emits the copy builtin,
// where the compiler leaves the loop as a scalar load and store per element.
//
//tsvc:kernel category=control reps=4 setup=vpv checksum=a exact=bits expect=vectorize
func va(a, b []float32) {
	for i := range a {
		a[i] = b[i]
	}
}

// vif: control loops, vector if. The store is conditional, which takes a masked
// store or a blend to vectorize.
//
//tsvc:kernel category=control reps=4 setup=vpv checksum=a exact=bits expect=vectorize
func vif(a, b []float32) {
	for i := range a {
		if b[i] > 0 {
			a[i] = b[i]
		}
	}
}

// vpv: control loops, vector plus vector.
//
//tsvc:kernel category=control reps=4 setup=vpv checksum=a exact=bits expect=vectorize
func vpv(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// vtv: control loops, vector times vector.
//
//tsvc:kernel category=control reps=4 setup=vtv checksum=a exact=bits expect=vectorize
func vtv(a, b []float32) {
	for i := range a {
		a[i] *= b[i]
	}
}

// vpvtv: control loops, vector plus vector times vector.
//
//tsvc:kernel category=control reps=4 setup=vpvtv checksum=a exact=fused expect=vectorize
func vpvtv(a, b, c []float32) {
	for i := range a {
		a[i] += b[i] * c[i]
	}
}

// vpvts: control loops, vector plus vector times scalar. TSVC_2 passes s1 (1.0)
// through a void pointer and reads it back as an int, which yields 1065353216;
// the port passes the intended 1.0.
//
//tsvc:kernel category=control reps=4 setup=vpvts checksum=a exact=fused expect=vectorize
func vpvts(a, b []float32, s1 float32) {
	for i := range a {
		a[i] += b[i] * s1
	}
}

// vpvpv: control loops, vector plus vector plus vector.
//
//tsvc:kernel category=control reps=4 setup=vpvpv checksum=a exact=bits expect=vectorize
func vpvpv(a, b, c []float32) {
	for i := range a {
		a[i] += b[i] + c[i]
	}
}

// vtvtv: control loops, vector times vector times vector.
//
//tsvc:kernel category=control reps=4 setup=vtvtv checksum=a exact=bits expect=vectorize
func vtvtv(a, b, c []float32) {
	for i := range a {
		a[i] = a[i] * b[i] * c[i]
	}
}

// vsumr: control loops, vector sum reduction.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=control reps=2 setup=s293 checksum=r exact=fused expect=vectorize
func vsumr(a []float32) float32 {
	var sum float32
	sum = 0
	for i := range a {
		sum += a[i]
	}
	return sum
}

// vdotr: control loops, vector dot product reduction.
// Returns the value TSVC_2 passes to dummy, which the checksum reads.
//
//tsvc:kernel category=control reps=2 setup=s313 checksum=r exact=fused expect=vectorize
func vdotr(a, b []float32) float32 {
	var dot float32
	dot = 0
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// vbor: control loops, basic operations rates, isolate arithmetic from memory
// traffic: all combinations of three, 59 flops for 6 loads and 1 store. The body
// is several statements, so it takes multi-statement support to vectorize.
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
