// Package rewrite rewrites loops identified by the analysis package to use simd.
package rewrite

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"sort"
	"strings"

	"golang.org/x/tools/go/ast/astutil"

	"github.com/mauri870/loopvec/internal/analysis"
	"github.com/mauri870/loopvec/internal/loopir"
)

// Result holds the rewritten file source.
type Result struct {
	Src      []byte
	Rewrites int
}

// Options controls how File rewrites a source file.
type Options struct {
	// AllowMethods enables rewriting of loops inside methods; see
	// analysis.Analyze for the caveat.
	AllowMethods bool
	// Compiler is set when the source is rewritten inside the build itself
	// (-toolexec). The go command has already selected the file for the current
	// build, so its build constraints are not consulted and no
	// //go:build goexperiment.simd constraint is added.
	Compiler bool
}

// File rewrites all vectorizable loops in src, returning the modified source.
// fset and file must correspond to the parsed src. info must have type
// information populated for the file.
func File(fset *token.FileSet, file *ast.File, info *types.Info, src []byte, opts Options) (Result, error) {
	if !opts.Compiler && HasBuildConstraint(file) {
		return Result{Src: src}, nil
	}
	plans := analysis.Analyze(file, info, opts.AllowMethods)
	if len(plans) == 0 {
		return Result{Src: src}, nil
	}

	tf := fset.File(file.Pos())

	// Collect replacements: map from loop node position to replacement text.
	type replacement struct {
		start token.Pos
		end   token.Pos
		text  string
	}
	var replacements []replacement
	var extraStmts []struct {
		pos  token.Pos
		text string
	}
	needsOverlapHelper := false

	for _, plan := range plans {
		stmtNode := plan.Loop.Node

		repl, pre, err := generateReplacement(plan, fset, len(replacements))
		if err != nil {
			continue
		}

		// Distinctly-named slices sharing a caller's backing array with the
		// destination can overlap it at an offset (dst == src[1:], say), which
		// the vectorized loop's block-at-a-time loads/stores does not
		// preserve the scalar loop's element-by-element order for. Guard
		// with a runtime check and fall back to the original scalar loop
		// when any pair could overlap, the same way LLVM/GCC/HotSpot's own
		// vectorizers version a loop on an unprovable aliasing check rather
		// than require static proof. See AGENTS.md's correctness gaps.
		if len(plan.Others) > 0 {
			origStart, origEnd := tf.Offset(stmtNode.Pos()), tf.Offset(stmtNode.End())
			repl = wrapWithOverlapCheck(plan, string(src[origStart:origEnd]), repl)
			needsOverlapHelper = true
		}

		replacements = append(replacements, replacement{
			start: stmtNode.Pos(),
			end:   stmtNode.End(),
			text:  repl,
		})
		if pre != "" {
			extraStmts = append(extraStmts, struct {
				pos  token.Pos
				text string
			}{stmtNode.Pos(), pre})
		}
	}

	if len(replacements) == 0 {
		return Result{Src: src}, nil
	}

	// Sort replacements in reverse order so positions stay valid.
	sort.Slice(replacements, func(i, j int) bool {
		return replacements[i].start > replacements[j].start
	})
	sort.Slice(extraStmts, func(i, j int) bool {
		return extraStmts[i].pos > extraStmts[j].pos
	})

	// Apply text replacements to src.
	result := make([]byte, len(src))
	copy(result, src)

	extraIdx := 0
	for _, r := range replacements {
		startOff := tf.Offset(r.start)
		endOff := tf.Offset(r.end)

		// Insert any pre-statements (like broadcast variable declarations) before the loop.
		var preText strings.Builder
		for extraIdx < len(extraStmts) && extraStmts[extraIdx].pos == r.start {
			preText.WriteString(extraStmts[extraIdx].text + "\n")
			extraIdx++
		}

		replacement := preText.String() + r.text
		result = append(result[:startOff], append([]byte(replacement), result[endOff:]...)...)
	}

	// Add simd import if not present.
	result, err := ensureImport(result, "simd")
	if err != nil {
		return Result{}, err
	}

	if needsOverlapHelper {
		result, err = ensureImport(result, "unsafe")
		if err != nil {
			return Result{}, err
		}
		result = ensureOverlapHelper(result)
	}

	// Add build tag if not present.
	if !opts.Compiler {
		result = ensureBuildTag(result, "goexperiment.simd")
	}

	// Format the result.
	formatted, err := format.Source(result)
	if err != nil {
		return Result{}, fmt.Errorf("format error: %w\nsource:\n%s", err, result)
	}

	// Rename blank identifier parameters to avoid a gotip GOEXPERIMENT=simd
	// compiler bug where _ params in simd-using functions produce
	// "cannot use _ as value or type".
	formatted = fixBlankParams(fset, formatted)

	return Result{Src: formatted, Rewrites: len(replacements)}, nil
}

// HasBuildConstraint reports whether file carries a //go:build or // +build
// line before its package clause. Such files are left untouched.
func HasBuildConstraint(file *ast.File) bool {
	for _, group := range file.Comments {
		if group.Pos() >= file.Package {
			break
		}
		for _, comment := range group.List {
			if constraint.IsGoBuild(comment.Text) || constraint.IsPlusBuild(comment.Text) {
				return true
			}
		}
	}
	return false
}

// wrapWithOverlapCheck guards simdText behind a runtime check that
// the destination's backing memory doesn't overlap any of plan.Others, falling back to
// origText (the loop's own original source) when it might. This is the
// same technique LLVM, GCC, and (since 2025) HotSpot's C2 use for loops
// they can't statically prove don't alias: a bounded runtime check plus a
// scalar fallback, not a static proof requirement.
func wrapWithOverlapCheck(plan *loopir.Plan, origText, simdText string) string {
	var cond strings.Builder
	for i, other := range plan.Others {
		if i > 0 {
			cond.WriteString(" || ")
		}
		fmt.Fprintf(&cond, "_loopvecOverlap(%s, %s)", plan.Dst.Name, other.Name)
	}
	return "if " + cond.String() + " {\n" + origText + "\n} else {\n" + simdText + "\n}"
}

// overlapHelperSrc is _loopvecOverlap's source, appended once per rewritten
// file that needs it.
const overlapHelperSrc = `
func _loopvecOverlap[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(a[0])
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(b[0])
	return aStart < bEnd && bStart < aEnd
}
`

// ensureOverlapHelper appends _loopvecOverlap's definition to src if not
// already present.
func ensureOverlapHelper(src []byte) []byte {
	if bytes.Contains(src, []byte("func _loopvecOverlap")) {
		return src
	}
	return append(src, []byte(overlapHelperSrc)...)
}

// fixBlankParams renames blank identifier (_) function parameters in src to
// _p0, _p1, … to work around a gotip GOEXPERIMENT=simd compiler bug:
// blank params in functions that contain simd code cause the SIMD lowering
// pass to emit "cannot use _ as value or type". The rename is safe because
// the parameters are never referenced by name in the body.
func fixBlankParams(fset *token.FileSet, src []byte) []byte {
	file, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		return src
	}

	counter := 0
	changed := false
	ast.Inspect(file, func(n ast.Node) bool {
		fd, ok := n.(*ast.FuncDecl)
		if !ok {
			return true
		}
		if fd.Type.Params == nil {
			return true
		}
		for _, field := range fd.Type.Params.List {
			for _, name := range field.Names {
				if name.Name == "_" {
					name.Name = fmt.Sprintf("_p%d", counter)
					counter++
					changed = true
				}
			}
		}
		return true
	})

	if !changed {
		return src
	}

	var buf bytes.Buffer
	if err := format.Node(&buf, fset, file); err != nil {
		return src
	}
	return buf.Bytes()
}

// ensureImport adds an import of path to the source if not already present.
func ensureImport(src []byte, path string) ([]byte, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	astutil.AddImport(fset, file, path)
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, file); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ensureBuildTag adds //go:build goexperiment.simd to the file if not already present.
func ensureBuildTag(src []byte, tag string) []byte {
	s := string(src)
	constraint := "//go:build " + tag
	if strings.Contains(s, constraint) {
		return src
	}
	// Insert after the package declaration line.
	idx := strings.Index(s, "package ")
	if idx < 0 {
		return src
	}
	// Find end of that line.
	nl := strings.Index(s[idx:], "\n")
	if nl < 0 {
		return src
	}
	insertAt := idx + nl + 1
	return []byte(s[:insertAt] + "\n" + constraint + "\n" + s[insertAt:])
}
