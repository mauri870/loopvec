package main

import (
	"os"
	"path/filepath"
	"strings"
)

// noTargetCheckEnv, when set, turns off testsSimdDependency, leaving only the
// per-package protection against test variants (recordVariant).
const noTargetCheckEnv = "LOOPVEC_TOOLEXEC_NO_TARGET_CHECK"

// testsSimdDependency reports whether the go command that started this build is
// running go test on a package that simd depends on.
//
// Testing such a package makes the go command compile a test variant of it, and
// rebuild the packages that import it against that variant. simd is not among
// them, so any rewritten package, which imports simd, would link against a
// build of the package that simd was not compiled with, and the linker rejects
// the program. recordVariant protects the packages built against a variant, but
// the go command compiles packages concurrently and the variant may not exist
// yet when another package is compiled, so the only safe answer is to rewrite
// nothing in such a build. The go command's own command line is the only place
// that says what is being tested, so it is read from the process table; when
// that is not possible, the answer is no and recordVariant is all there is.
func (b *build) testsSimdDependency(skip map[string]bool) bool {
	out, err := b.cached("testtarget", func() ([]byte, error) {
		found, err := b.testTargetsSimdDependency(func() (map[string]bool, error) { return skip, nil })
		if found {
			return []byte("yes"), err
		}
		return []byte("no"), err
	})
	return err == nil && string(out) == "yes"
}

// probeTestsSimdDependency is testsSimdDependency for the go command's -V=full
// probe, which runs before there is a build directory to share the answer
// through. The answer is part of the wrapper's build identity: rewriting nothing
// in such a build is not enough, since the go command would still take
// rewritten packages that an earlier build left in its cache, and those link
// simd against the other build of the tested package.
func probeTestsSimdDependency(tool string) bool {
	b := &build{tool: tool}
	found, err := b.testTargetsSimdDependency(func() (map[string]bool, error) {
		out, err := b.listSkip()
		if err != nil {
			return nil, err
		}
		return importPaths(out), nil
	})
	return err == nil && found
}

// testTargetsSimdDependency reports whether the go command that started this
// process is go test on a package in the skip set. The set is only computed for
// a go test command line.
func (b *build) testTargetsSimdDependency(skipSet func() (map[string]bool, error)) (bool, error) {
	if os.Getenv(noTargetCheckEnv) != "" {
		return false, nil
	}
	dir, args, ok := parentCommand()
	if !ok {
		return false, nil
	}
	patterns, ok := testPatterns(args)
	if !ok {
		return false, nil
	}
	skip, err := skipSet()
	if err != nil {
		return false, err
	}
	// A flag's value, as in -run TestX, is indistinguishable from a package
	// pattern here. go list -e resolves it to something that is not a simd
	// dependency, or reports an error, and a value that does name one only
	// costs the rewrites.
	out, err := b.goCommandIn(dir, append([]string{"list", "-e", "-f", "{{.ImportPath}}"}, patterns...)...)
	if err != nil {
		return false, err
	}
	for path := range strings.FieldsSeq(string(out)) {
		if skip[path] {
			return true, nil
		}
	}
	return false, nil
}

// testPatterns returns the package patterns of a go test command line: its
// arguments that are not flags, and the packages named by -coverpkg, which are
// also compiled as variants. It reports false when args is not go test.
func testPatterns(args []string) ([]string, bool) {
	if len(args) < 2 || strings.TrimSuffix(filepath.Base(args[0]), ".exe") != "go" || args[1] != "test" {
		return nil, false
	}
	var patterns []string
	rest := args[2:]
	for i := 0; i < len(rest); i++ {
		arg := rest[i]
		if arg == "-args" {
			break
		}
		if !strings.HasPrefix(arg, "-") {
			patterns = append(patterns, arg)
			continue
		}
		name, value, hasValue := strings.Cut(strings.TrimLeft(arg, "-"), "=")
		if name != "coverpkg" {
			continue
		}
		if !hasValue && i+1 < len(rest) {
			i++
			value = rest[i]
		}
		patterns = append(patterns, strings.Split(value, ",")...)
	}
	if len(patterns) == 0 {
		patterns = []string{"."}
	}
	return patterns, true
}
