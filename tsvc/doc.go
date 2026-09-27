// Package tsvc ports the first 10 "s" kernels from TSVC_2
// (https://github.com/UoB-HPC/TSVC_2) to Go, in source order: s000, s111,
// s1111, s112, s1112, s113, s1113, s114, s115, s1115. It measures how many of
// them loopvec rewrites; it does not change loopvec's rewriting.
//
// The loop shapes are kept exactly as TSVC_2 wrote them. See LICENSE-TSVC for
// the license covering the ported kernels.
package tsvc

//go:generate go tool stringer -type=Category -trimprefix=Category
//go:generate go run ./internal/tsvcgen -o zz_registry.go kernels.go
