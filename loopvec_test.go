package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/txtar"

	"github.com/mauri870/loopvec/internal/rewrite"
)

// TestTxtar runs all testdata/*.txtar files.
// Each archive has the following sections:
//
//   - go.mod          : go.mod for the test module
//   - in.go           : input Go source to rewrite
//   - want.go         : expected rewritten source
//   - testmain/main.go: (optional) test program exercising the function
//   - testmain/output : (optional) expected stdout of the test program
func TestTxtar(t *testing.T) {
	matches, err := filepath.Glob("testdata/*.txtar")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no txtar files found in testdata/")
	}
	for _, path := range matches {
		name := strings.TrimSuffix(filepath.Base(path), ".txtar")
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			runTxtar(t, path)
		})
	}
}

func runTxtar(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	ar := txtar.Parse(data)

	files := archiveFiles(ar)

	goModSrc, ok := files["go.mod"]
	if !ok {
		t.Fatal("txtar missing go.mod section")
	}
	inSrc, ok := files["in.go"]
	if !ok {
		t.Fatal("txtar missing in.go section")
	}
	wantSrc, ok := files["want.go"]
	if !ok {
		t.Fatal("txtar missing want.go section")
	}

	// 1. Rewrite the input source.
	got, err := rewriteSource(inSrc)
	if err != nil {
		t.Fatalf("rewrite failed: %v", err)
	}

	// 2. Compare with the expected output.
	gotNorm := normalizeSource(t, got)
	wantNorm := normalizeSource(t, wantSrc)
	if gotNorm != wantNorm {
		t.Errorf("rewrite mismatch:\ngot:\n%s\nwant:\n%s", gotNorm, wantNorm)
	}

	// 3. If testmain is present, verify behavioral equivalence.
	testMainSrc, hasTestMain := files["testmain/main.go"]
	expectedOutput, hasOutput := files["testmain/output"]
	if !hasTestMain || !hasOutput {
		return
	}

	modName := parseModuleName(goModSrc)
	if modName == "" {
		t.Fatal("could not parse module name from go.mod")
	}

	// Run scalar version.
	scalarOut, err := runTestProgram(t, modName, goModSrc, inSrc, testMainSrc, false)
	if err != nil {
		t.Fatalf("scalar run failed: %v", err)
	}

	// Run simd version.
	simdOut, err := runTestProgram(t, modName, goModSrc, got, testMainSrc, true)
	if err != nil {
		t.Fatalf("simd run failed: %v", err)
	}

	// Both should match the expected output.
	want := strings.TrimSpace(string(expectedOutput))
	if strings.TrimSpace(scalarOut) != want {
		t.Errorf("scalar output mismatch:\ngot:  %q\nwant: %q", scalarOut, want)
	}
	if strings.TrimSpace(simdOut) != want {
		t.Errorf("simd output mismatch:\ngot:  %q\nwant: %q", simdOut, want)
	}
}

// rewriteSource parses and rewrites a single Go source file.
// It uses a minimal type-checker that handles only built-in types.
func rewriteSource(src []byte) ([]byte, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "in.go", src, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}

	info := &types.Info{
		Types: make(map[ast.Expr]types.TypeAndValue),
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
	}

	conf := types.Config{Importer: nil}
	// Type errors are expected when import "simd" is not resolvable; proceed anyway.
	_, _ = conf.Check("test", fset, []*ast.File{file}, info)

	result, err := rewrite.File(fset, file, info, src)
	if err != nil {
		return nil, err
	}
	return result.Src, nil
}

// runTestProgram creates a temp module, writes pkg.go and testmain/main.go,
// compiles and runs the test program. If simd is true, GOEXPERIMENT=simd is set.
func runTestProgram(t *testing.T, modName string, goModSrc, pkgSrc, mainSrc []byte, simd bool) (string, error) {
	t.Helper()
	dir := t.TempDir()

	// Write go.mod.
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), goModSrc, 0o644); err != nil {
		return "", err
	}

	// Write the package file at the module root (same dir as go.mod).
	if err := os.WriteFile(filepath.Join(dir, "pkg.go"), pkgSrc, 0o644); err != nil {
		return "", err
	}

	// Write testmain/main.go.
	mainDir := filepath.Join(dir, "testmain")
	if err := os.MkdirAll(mainDir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(mainDir, "main.go"), mainSrc, 0o644); err != nil {
		return "", err
	}

	// Write a go.mod for testmain that replaces the module with the local dir.
	testMainMod := fmt.Sprintf("module %s/testmain\n\ngo 1.27.1\n\nrequire %s v0.0.0\n\nreplace %s => ..\n", modName, modName, modName)
	if err := os.WriteFile(filepath.Join(mainDir, "go.mod"), []byte(testMainMod), 0o644); err != nil {
		return "", err
	}

	// Run go mod tidy in mainDir.
	tidyCmd := exec.Command("go", "mod", "tidy")
	tidyCmd.Dir = mainDir
	tidyCmd.Env = append(os.Environ(), "GOTOOLCHAIN=go1.27.1")
	if out, err := tidyCmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("go mod tidy: %w\n%s", err, out)
	}

	// Build and run.
	runCmd := exec.Command("go", "run", ".")
	runCmd.Dir = mainDir
	env := append(os.Environ(), "GOTOOLCHAIN=go1.27.1")
	if simd {
		env = append(env, "GOEXPERIMENT=simd")
	}
	runCmd.Env = env

	out, err := runCmd.Output()
	if err != nil {
		if exitErr, ok := errors.AsType[*exec.ExitError](err); ok {
			return "", fmt.Errorf("run failed: %w\n%s", err, exitErr.Stderr)
		}
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// normalizeSource formats Go source for comparison.
func normalizeSource(t *testing.T, src []byte) string {
	t.Helper()
	formatted, err := format.Source(src)
	if err != nil {
		// Return as-is if formatting fails; comparison may still work.
		return string(src)
	}
	return string(formatted)
}

// archiveFiles returns a map of filename → contents from a txtar archive.
func archiveFiles(ar *txtar.Archive) map[string][]byte {
	m := make(map[string][]byte, len(ar.Files))
	for _, f := range ar.Files {
		m[f.Name] = f.Data
	}
	return m
}

// parseModuleName extracts the module name from a go.mod file.
func parseModuleName(gomod []byte) string {
	for line := range strings.SplitSeq(string(gomod), "\n") {
		line = strings.TrimSpace(line)
		if after, ok := strings.CutPrefix(line, "module "); ok {
			return strings.TrimSpace(after)
		}
	}
	return ""
}
