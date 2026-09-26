//go:build !unix

package main

// lockFile is a no-op where advisory file locks are unavailable; concurrent
// wrapper processes then compute the shared state redundantly, which is only
// slower.
func lockFile(string) (func(), error) { return func() {}, nil }
