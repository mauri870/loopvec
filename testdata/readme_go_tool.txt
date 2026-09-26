# "As a go tool": loopvec declared as a tool dependency and run through
# "go tool loopvec". A workspace points at this checkout instead of a release.

go work init .
go work use $LOOPVEC_SRC

exec go tool loopvec -d ./...
cmpenv stderr stderr.want
stdout -count=1 '^\+\+\+ \S+/mypkg/ops\.go\t'

exec go tool loopvec -split ./mypkg/...
! stdout .
cmpenv stderr stderr.want
exists mypkg/ops_simd.go

-- go.mod --
module example.com/test

go 1.27.1

tool github.com/mauri870/loopvec
-- mypkg/ops.go --
package mypkg

func AddInt32s(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
-- stderr.want --
loopvec: rewrote 1 loop(s) in $WORK/mypkg/ops.go
