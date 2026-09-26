// Command loopvec-toolexec is a go build -toolexec wrapper that vectorizes
// element-wise loops with the simd package while the go command compiles
// every package of the build, standard library included.
//
// Usage:
//
//	GOEXPERIMENT=simd go build -toolexec=$(which loopvec-toolexec) ./...
//
// The wrapper only rewrites packages when GOEXPERIMENT enables simd. Packages
// that simd depends on cannot import it without creating an import cycle and
// are left untouched, as is any package whose rewritten form fails to
// type-check.
//
// Environment:
//
//	LOOPVEC_TOOLEXEC_LOG       append one line per rewritten or rejected package to this file
//	LOOPVEC_TOOLEXEC_LOG_PKGS  comma-separated import paths; restrict the log to these packages
//	LOOPVEC_TOOLEXEC_DEBUG     also log why each package was left as it was
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// nestedEnv marks the go commands the wrapper itself starts, which must not be
// rewritten again.
const nestedEnv = "LOOPVEC_TOOLEXEC_NESTED"

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: go build -toolexec=loopvec-toolexec [packages]")
		os.Exit(2)
	}
	os.Exit(run(os.Args[1], os.Args[2:]))
}

func run(tool string, args []string) int {
	name := strings.TrimSuffix(filepath.Base(tool), ".exe")
	if len(args) == 1 && args[0] == "-V=full" {
		return printVersion(tool, name)
	}
	if os.Getenv(nestedEnv) == "" {
		switch name {
		case "compile":
			return compile(tool, args)
		case "link":
			return link(tool, args)
		}
	}
	return execTool(tool, args)
}

// execTool runs tool with args, wired to this process's stdio, and returns its
// exit code.
func execTool(tool string, args []string) int {
	cmd := exec.Command(tool, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}
		fmt.Fprintln(os.Stderr, "loopvec-toolexec:", err)
		return 1
	}
	return 0
}

// printVersion answers the go command's "-V=full" probe. The go command folds
// the answer into the build ID of everything the tool produces, so compile and
// link report the wrapper's own content hash to invalidate cached objects
// whenever the wrapper changes.
func printVersion(tool, name string) int {
	out, err := exec.Command(tool, "-V=full").Output()
	if err != nil {
		return execTool(tool, []string{"-V=full"})
	}
	line := strings.TrimSpace(string(out))
	if name != "compile" && name != "link" {
		fmt.Println(line)
		return 0
	}
	hash := selfHash()
	fields := strings.Fields(line)
	if last := len(fields) - 1; last >= 0 && strings.HasPrefix(fields[last], "buildID=") {
		// Development toolchains end the line with buildID=...; the go command
		// requires that field to stay last and derives the tool's identity from
		// its content part alone, so the hash goes inside it.
		fields[last] += "-loopvec" + hash
		fmt.Println(strings.Join(fields, " "))
		return 0
	}
	// Releases are identified by the whole line.
	fmt.Println(line + " loopvec=" + hash)
	return 0
}

func selfHash() string {
	exe, err := os.Executable()
	if err != nil {
		return "unknown"
	}
	f, err := os.Open(exe)
	if err != nil {
		return "unknown"
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "unknown"
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// logf appends one line to $LOOPVEC_TOOLEXEC_LOG for package pkg, honoring
// $LOOPVEC_TOOLEXEC_LOG_PKGS.
func logf(pkg, format string, args ...any) {
	path := os.Getenv("LOOPVEC_TOOLEXEC_LOG")
	if path == "" {
		return
	}
	if only := os.Getenv("LOOPVEC_TOOLEXEC_LOG_PKGS"); only != "" {
		match := false
		for _, p := range strings.Split(only, ",") {
			if strings.TrimSpace(p) == pkg {
				match = true
			}
		}
		if !match {
			return
		}
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, format+"\n", args...)
}

// debugf is logf for the reasons a package is left alone, enabled by
// $LOOPVEC_TOOLEXEC_DEBUG.
func debugf(pkg, format string, args ...any) {
	if os.Getenv("LOOPVEC_TOOLEXEC_DEBUG") != "" {
		logf(pkg, format, args...)
	}
}
