package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/mauri870/loopvec/internal/rewrite"
)

// compile runs the compiler, first swapping in vectorized sources when the
// package has eligible loops and the rewritten package still type-checks.
// Any doubt falls back to the original arguments, so the wrapper can only
// change what gets compiled, never make a working build fail.
func compile(tool string, args []string) int {
	recordVariant(tool, args, parseCompileArgs(args))
	if rewritten, ok := tryRewrite(tool, args); ok {
		return execTool(tool, rewritten)
	}
	return execTool(tool, args)
}

// compileArgs locates the parts of a compile command line the wrapper needs.
type compileArgs struct {
	pkg       string
	lang      string
	importcfg int   // index in args of the importcfg path, or -1
	files     []int // indexes in args of the Go source files
	// unsupported is set for build modes whose objects the nested simd build
	// would not match (race, msan, asan, shared, dynlink, -trimpath).
	unsupported bool
}

func parseCompileArgs(args []string) compileArgs {
	c := compileArgs{importcfg: -1}
	first := len(args)
	for first > 0 && strings.HasSuffix(args[first-1], ".go") && !strings.HasPrefix(args[first-1], "-") {
		first--
	}
	for i := first; i < len(args); i++ {
		c.files = append(c.files, i)
	}
	for i := 0; i < first; i++ {
		arg := args[i]
		hasValue := i+1 < first
		switch {
		case arg == "-p" && hasValue:
			c.pkg = args[i+1]
		case arg == "-importcfg" && hasValue:
			c.importcfg = i + 1
		case arg == "-lang" && hasValue:
			c.lang = args[i+1]
		case strings.HasPrefix(arg, "-lang="):
			c.lang = strings.TrimPrefix(arg, "-lang=")
		case arg == "-race", arg == "-msan", arg == "-asan", arg == "-shared", arg == "-dynlink":
			c.unsupported = true
		case arg == "-trimpath" && hasValue && strings.Contains(args[i+1], ";"):
			c.unsupported = true
		}
	}
	return c
}

// source is one Go file of the package being compiled.
type source struct {
	path      string // as passed to the compiler
	src       []byte
	rewritten []byte // nil when the file is compiled as is
	loops     int
}

func tryRewrite(tool string, args []string) (out []string, ok bool) {
	c := parseCompileArgs(args)
	// Whatever goes wrong while analyzing or rewriting, the original package
	// still compiles: never let the wrapper fail a build.
	defer func() {
		if r := recover(); r != nil {
			logf(c.pkg, "kept %s: internal error: %v", c.pkg, r)
			out, ok = nil, false
		}
	}()
	if c.pkg == "" || c.importcfg < 0 || len(c.files) == 0 {
		return nil, false
	}
	if c.unsupported || !simdOn(tool) {
		debugf(c.pkg, "skipped %s: simd unavailable or unsupported build mode", c.pkg)
		return nil, false
	}
	if want := toolchainRelease(); want != builtRelease() {
		// go/types can only read export data written by the Go release it was
		// built with.
		debugf(c.pkg, "skipped %s: loopvec-toolexec was built with %s but the toolchain is %s", c.pkg, builtRelease(), want)
		return nil, false
	}
	b, err := newBuild(tool, args[c.importcfg])
	if err != nil {
		return nil, false
	}
	skip, err := b.skipSet()
	if err != nil {
		logf(c.pkg, "kept %s: %v", c.pkg, firstLine(err))
		return nil, false
	}
	if skip[c.pkg] {
		debugf(c.pkg, "skipped %s: simd depends on it", c.pkg)
		return nil, false
	}

	sources, fset, files, ok := parseSources(args, c)
	if !ok {
		debugf(c.pkg, "skipped %s: cgo or unreadable sources", c.pkg)
		return nil, false
	}
	cfg, err := readImportcfg(args[c.importcfg])
	if err != nil {
		return nil, false
	}
	info, errs := check(fset, files, c, cfg)
	if len(errs) > 0 {
		logf(c.pkg, "kept %s: does not type-check: %v", c.pkg, errs[0])
		return nil, false
	}

	total := 0
	for i, s := range sources {
		res, err := rewrite.File(fset, files[i], info, s.src, rewrite.Options{Compiler: true})
		if err != nil || res.Rewrites == 0 {
			continue
		}
		abs, err := filepath.Abs(s.path)
		if err != nil {
			continue
		}
		s.rewritten = append([]byte("//line "+abs+":1:1\n"), res.Src...)
		s.loops = res.Rewrites
		total += res.Rewrites
	}
	if total == 0 {
		debugf(c.pkg, "skipped %s: no vectorizable loops", c.pkg)
		return nil, false
	}

	exports, err := b.exports()
	if err != nil {
		logf(c.pkg, "kept %s: %v", c.pkg, firstLine(err))
		return nil, false
	}
	if dep := variantDependency(b, cfg); dep != "" {
		debugf(c.pkg, "skipped %s: built against a test variant of %s", c.pkg, dep)
		return nil, false
	}
	patched := cfg.with(exports)
	total, rejected, err := verify(sources, c, patched)
	if rejected != nil {
		logf(c.pkg, "kept %s: rewritten source rejected: %v", c.pkg, rejected)
	}
	if err != nil || total == 0 {
		return nil, false
	}

	// A package and its test variant are compiled concurrently under the same
	// import path, so every compile gets its own directory.
	dir, err := os.MkdirTemp(b.dir, strings.NewReplacer("/", "_", ".", "_").Replace(c.pkg)+"-")
	if err != nil {
		return nil, false
	}
	newArgs := append([]string(nil), args...)
	for i, s := range sources {
		if s.rewritten == nil {
			continue
		}
		path := filepath.Join(dir, filepath.Base(s.path))
		if err := os.WriteFile(path, s.rewritten, 0o644); err != nil {
			return nil, false
		}
		newArgs[c.files[i]] = path
	}
	cfgPath := filepath.Join(dir, "importcfg")
	if err := cfg.writeExtended(args[c.importcfg], cfgPath, exports); err != nil {
		return nil, false
	}
	newArgs[c.importcfg] = cfgPath
	logf(c.pkg, "rewrote %d loop(s) in %s", total, c.pkg)
	return newArgs, true
}

// recordVariant notes the archive of a package simd depends on when it is a
// test variant: compiled together with its test files, or rebuilt against
// another such variant. The go command does this to test the package and
// rebuilds the importers it knows about, but simd is not among them: code
// importing simd would link against two different builds of the same package,
// which the linker rejects. tryRewrite leaves any package built against a
// recorded archive alone.
func recordVariant(tool string, args []string, c compileArgs) {
	hasTests := false
	for _, i := range c.files {
		if strings.HasSuffix(args[i], "_test.go") {
			hasTests = true
		}
	}
	out := flagValue(args, "-o")
	if out < 0 || c.importcfg < 0 || c.pkg == "" || c.unsupported || !simdOn(tool) {
		return
	}
	b, err := newBuild(tool, args[c.importcfg])
	if err != nil {
		return
	}
	skip, err := b.skipSet()
	if err != nil || !skip[c.pkg] {
		return
	}
	if !hasTests {
		cfg, err := readImportcfg(args[c.importcfg])
		if err != nil || variantDependency(b, cfg) == "" {
			return
		}
	}
	f, err := os.OpenFile(filepath.Join(b.dir, "variants"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintln(f, args[out])
}

// variantDependency returns a package that cfg provides from a recorded test
// variant archive, or "".
func variantDependency(b *build, cfg *importcfg) string {
	data, err := os.ReadFile(filepath.Join(b.dir, "variants"))
	if err != nil {
		return ""
	}
	variants := map[string]bool{}
	for line := range strings.FieldsSeq(string(data)) {
		variants[line] = true
	}
	var found []string
	for path, file := range cfg.packageFile {
		if variants[file] {
			found = append(found, path)
		}
	}
	sort.Strings(found)
	if len(found) == 0 {
		return ""
	}
	return found[0]
}

// parseSources reads and parses the package's files. Packages that use cgo are
// left alone: the compiler is handed generated code whose types the wrapper
// cannot reconstruct.
func parseSources(args []string, c compileArgs) ([]*source, *token.FileSet, []*ast.File, bool) {
	fset := token.NewFileSet()
	var sources []*source
	var files []*ast.File
	for _, i := range c.files {
		path := args[i]
		base := filepath.Base(path)
		if strings.HasSuffix(base, ".cgo1.go") || base == "_cgo_gotypes.go" {
			return nil, nil, nil, false
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return nil, nil, nil, false
		}
		file, err := parser.ParseFile(fset, path, src, parser.ParseComments)
		if err != nil {
			return nil, nil, nil, false
		}
		sources = append(sources, &source{path: path, src: src})
		files = append(files, file)
	}
	return sources, fset, files, true
}

// check type-checks files as package c.pkg, importing from cfg's export data.
func check(fset *token.FileSet, files []*ast.File, c compileArgs, cfg *importcfg) (*types.Info, []error) {
	info := &types.Info{
		Types: make(map[ast.Expr]types.TypeAndValue),
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
	}
	var errs []error
	goarch := os.Getenv("GOARCH")
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	conf := types.Config{
		Importer:  cfg.importer(fset),
		GoVersion: c.lang,
		Sizes:     types.SizesFor("gc", goarch),
		Error:     func(err error) { errs = append(errs, err) },
	}
	_, _ = conf.Check(c.pkg, fset, files, info)
	return info, errs
}

// verify type-checks the package with the rewritten files in place. Files that
// cause errors are reverted to their original source; if the package still does
// not check, verify fails. It returns the number of loops that remain rewritten
// and the first error that made it revert a file, if any.
func verify(sources []*source, c compileArgs, cfg *importcfg) (total int, rejected, err error) {
	for attempt := 0; ; attempt++ {
		fset := token.NewFileSet()
		files := make([]*ast.File, len(sources))
		bad := map[string]bool{}
		var firstErr error
		for i, s := range sources {
			src := s.src
			if s.rewritten != nil {
				src = s.rewritten
			}
			file, err := parser.ParseFile(fset, s.path, src, parser.ParseComments)
			if err != nil {
				bad[s.path] = true
				if firstErr == nil {
					firstErr = err
				}
				continue
			}
			files[i] = file
		}
		if len(bad) == 0 {
			_, errs := check(fset, files, c, cfg)
			for _, err := range errs {
				if firstErr == nil {
					firstErr = err
				}
				if terr, ok := err.(types.Error); ok {
					bad[terr.Fset.PositionFor(terr.Pos, false).Filename] = true
				}
			}
		}
		if firstErr == nil {
			for _, s := range sources {
				if s.rewritten != nil {
					total += s.loops
				}
			}
			return total, rejected, nil
		}
		if rejected == nil {
			rejected = firstErr
		}
		reverted := false
		for _, s := range sources {
			if s.rewritten != nil && bad[s.path] {
				s.rewritten, s.loops = nil, 0
				reverted = true
			}
		}
		if !reverted || attempt >= 1 {
			return 0, rejected, firstErr
		}
	}
}

func firstLine(err error) string {
	line, _, _ := strings.Cut(err.Error(), "\n")
	return fmt.Sprint(line)
}
