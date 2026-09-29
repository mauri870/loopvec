//go:build !linux

package main

// parentCommand is unavailable here: the process table is only read on Linux.
func parentCommand() (dir string, args []string, ok bool) {
	return "", nil, false
}
