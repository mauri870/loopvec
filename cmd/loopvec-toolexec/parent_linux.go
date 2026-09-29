//go:build linux

package main

import (
	"os"
	"strconv"
	"strings"
)

// parentCommand returns the working directory and command line of the process
// that started this one, which for a toolexec wrapper is the go command.
func parentCommand() (dir string, args []string, ok bool) {
	proc := "/proc/" + strconv.Itoa(os.Getppid())
	raw, err := os.ReadFile(proc + "/cmdline")
	if err != nil {
		return "", nil, false
	}
	dir, err = os.Readlink(proc + "/cwd")
	if err != nil {
		return "", nil, false
	}
	return dir, strings.Split(strings.TrimSuffix(string(raw), "\x00"), "\x00"), true
}
