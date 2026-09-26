package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
)

var (
	toolOnce    sync.Once
	simdEnabled bool
	toolVersion string
)

// probeTool runs the compiler's -V=full once. Its version string advertises
// the experiments it was built with.
func probeTool(tool string) {
	toolOnce.Do(func() {
		out, err := exec.Command(tool, "-V=full").Output()
		if err != nil {
			return
		}
		toolVersion = string(out)
		simdEnabled = strings.Contains(toolVersion, "simd")
	})
}

// simdOn reports whether the compiler that tool points to was built with
// GOEXPERIMENT=simd.
func simdOn(tool string) bool {
	probeTool(tool)
	return simdEnabled
}

var releaseRE = regexp.MustCompile(`go1\.[0-9]+`)

// toolchainRelease returns the Go release (such as go1.28) of the compiler
// probed by simdOn.
func toolchainRelease() string { return releaseRE.FindString(toolVersion) }

// builtRelease returns the Go release this program was built with.
func builtRelease() string { return releaseRE.FindString(runtime.Version()) }

// build holds the per-build state shared by the wrapper processes of one go
// command invocation. It lives in the go command's work directory, so it is
// discarded together with the build.
type build struct {
	tool string
	work string
	dir  string
}

// newBuild derives the work directory from the importcfg path the go command
// passes to compile and link ($WORK/bNNN/importcfg).
func newBuild(tool, importcfgPath string) (*build, error) {
	work := filepath.Dir(filepath.Dir(importcfgPath))
	dir := filepath.Join(work, "loopvec-toolexec")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &build{tool: tool, work: work, dir: dir}, nil
}

// goCommand runs the go command belonging to the toolchain that tool is part
// of, marked so the wrapper does not rewrite its compiles again.
func (b *build) goCommand(args ...string) ([]byte, error) {
	goroot := filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(b.tool)))) // GOROOT/pkg/tool/GOOS_GOARCH/compile
	cmd := exec.Command(filepath.Join(goroot, "bin", "go"), args...)
	cmd.Dir = b.dir
	cmd.Env = append(os.Environ(), nestedEnv+"=1", "GOTOOLCHAIN=local")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("go %s: %w\n%s", strings.Join(args, " "), err, stderr.String())
	}
	return out, nil
}

// skipSet returns the packages that must not be rewritten: everything simd and
// the runtime depend on. Rewritten code imports simd, so rewriting any of them
// would create an import cycle.
func (b *build) skipSet() (map[string]bool, error) {
	out, err := b.cached("skip", func() ([]byte, error) {
		return b.goCommand("list", "-deps", "-f", "{{.ImportPath}}", "simd", "runtime")
	})
	if err != nil {
		return nil, err
	}
	set := map[string]bool{}
	for _, path := range strings.Fields(string(out)) {
		set[path] = true
	}
	return set, nil
}

// exports returns the export data archives of simd and everything it depends
// on. They are built through this wrapper so they are the very objects the
// enclosing build uses for the shared packages.
func (b *build) exports() (map[string]string, error) {
	self, err := os.Executable()
	if err != nil {
		return nil, err
	}
	out, err := b.cached("exports", func() ([]byte, error) {
		return b.goCommand("list", "-deps", "-export", "-toolexec="+self,
			"-f", "{{.ImportPath}}={{.Export}}", "simd")
	})
	if err != nil {
		return nil, err
	}
	exports := map[string]string{}
	for _, line := range strings.Split(string(out), "\n") {
		path, file, ok := strings.Cut(strings.TrimSpace(line), "=")
		if ok && file != "" {
			exports[path] = file
		}
	}
	return exports, nil
}

// cached computes name once per build. Concurrent wrapper processes wait on a
// lock instead of repeating the work, and a failure is remembered as well.
func (b *build) cached(name string, compute func() ([]byte, error)) ([]byte, error) {
	unlock, err := lockFile(filepath.Join(b.dir, name+".lock"))
	if err != nil {
		return nil, err
	}
	defer unlock()
	path := filepath.Join(b.dir, name)
	if data, err := os.ReadFile(path + ".err"); err == nil {
		return nil, fmt.Errorf("%s", data)
	}
	if data, err := os.ReadFile(path); err == nil {
		return data, nil
	}
	data, err := compute()
	if err != nil {
		_ = os.WriteFile(path+".err", []byte(err.Error()), 0o644)
		return nil, err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return nil, err
	}
	return data, nil
}
