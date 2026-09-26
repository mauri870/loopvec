package main_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

// TestScripts builds the real loopvec binary and drives it through the
// scripts in testdata/*.txt. Each script runs in its own temporary module
// containing the files from the script's txtar section.
func TestScripts(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "loopvec")
	build := exec.Command("go", "build", "-o", binary, ".")
	build.Env = testEnv(t)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building loopvec: %v\n%s", err, out)
	}

	cmds := scripttest.DefaultCmds()
	cmds["loopvec"] = script.Program(binary, nil, 0)
	cmds["go"] = script.Program("go", nil, 0)

	engine := &script.Engine{
		Conds: scripttest.DefaultConds(),
		Cmds:  cmds,
		Quiet: !testing.Verbose(),
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// LOOPVEC_SRC lets scripts point a Go workspace at this checkout so that
	// "go tool loopvec" resolves without a published release.
	env := append(testEnv(t), "LOOPVEC_SRC="+wd)
	scripttest.Test(t, context.Background(), engine, env, "testdata/*.txt")
}

// testEnv returns the process environment without GOEXPERIMENT (scripts opt in
// with "env GOEXPERIMENT=simd") and with GOTOOLCHAIN pinned to the toolchain
// declared in go.mod, regardless of the caller's setting.
func testEnv(t *testing.T) []string {
	t.Helper()
	var env []string
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, "GOEXPERIMENT=") || strings.HasPrefix(kv, "GOTOOLCHAIN=") {
			continue
		}
		env = append(env, kv)
	}
	return append(env, "GOTOOLCHAIN="+goModToolchain(t))
}

// goModToolchain returns the "toolchain" directive of go.mod, or "go" plus the
// "go" directive when there is none.
func goModToolchain(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatal(err)
	}
	var goVersion string
	for line := range strings.SplitSeq(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		switch fields[0] {
		case "toolchain":
			return fields[1]
		case "go":
			goVersion = fields[1]
		}
	}
	if goVersion == "" {
		t.Fatal("go.mod has neither a toolchain nor a go directive")
	}
	return "go" + goVersion
}
