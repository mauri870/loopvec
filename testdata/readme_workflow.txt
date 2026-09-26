# The README workflow: preview, apply with -split, then verify with and
# without GOEXPERIMENT=simd.

# 1. Preview changes
loopvec -d ./mypkg/...
cmpenv stderr stderr.want
stdout -count=1 '^\+\+\+ \S+/mypkg/ops\.go\t'
! exists mypkg/ops_simd.go

# 2. Apply: creates ops_simd.go and guards ops.go
loopvec -split ./mypkg/...
! stdout .
cmpenv stderr stderr.want
exists mypkg/ops_simd.go

# 3. Verify correctness and benchmark, scalar then simd
exec go test ./mypkg/...
stdout -count=1 '^ok  	example\.com/test/mypkg	'
env GOEXPERIMENT=simd
exec go test ./mypkg/...
stdout -count=1 '^ok  	example\.com/test/mypkg	'
exec go test -run=^$ -bench=. -benchtime=1x ./mypkg/
stdout -count=1 '^BenchmarkAddFloat32s'

-- go.mod --
module example.com/test

go 1.27.1
-- mypkg/ops.go --
package mypkg

func AddFloat32s(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- mypkg/ops_test.go --
package mypkg

import "testing"

func TestAddFloat32s(t *testing.T) {
	dst := make([]float32, 19)
	a := make([]float32, 19)
	b := make([]float32, 19)
	for i := range a {
		a[i], b[i] = float32(i), float32(2*i)
	}
	AddFloat32s(dst, a, b)
	for i := range dst {
		if dst[i] != float32(3*i) {
			t.Fatalf("dst[%d] = %v, want %v", i, dst[i], 3*i)
		}
	}
}

func BenchmarkAddFloat32s(b *testing.B) {
	dst := make([]float32, 1024)
	x := make([]float32, 1024)
	y := make([]float32, 1024)
	for b.Loop() {
		AddFloat32s(dst, x, y)
	}
}
-- stderr.want --
loopvec: rewrote 1 loop(s) in $WORK/mypkg/ops.go
