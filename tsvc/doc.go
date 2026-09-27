// Package tsvc ports the TSVC_2 (https://github.com/UoB-HPC/TSVC_2) to Go.
//
// The loop shapes are kept exactly as TSVC_2 wrote them. See LICENSE-TSVC for
// the license covering the ported kernels.
package tsvc

//go:generate go tool stringer -type=Category -trimprefix=Category
//go:generate go run ./internal/tsvcgen -o zz_registry.go kernels.go
