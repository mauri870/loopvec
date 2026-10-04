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
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/astutil"

	"github.com/mauri870/loopvec/internal/analysis"
	"github.com/mauri870/loopvec/internal/loopir"
)

// Result holds the rewritten file source.
type Result struct {
	Src      []byte
	Rewrites int
	// DefinesHelper is set when Src defines _loopvecOverlap. A package may define
	// it only once, so the caller passes this on to the files after this one.
	DefinesHelper bool
}

// Options controls how File rewrites a source file.
type Options struct {
	// AllowMethods enables rewriting of loops inside methods; see
	// analysis.Analyze for the caveat.
	AllowMethods bool
	// FloatReassoc allows regrouping a floating-point sum or product, which can
	// change the result in the last bits; see loopir.Options.
	FloatReassoc bool
	// HelperDefined says an earlier file of the same package already defines
	// _loopvecOverlap, so this file uses it and does not define it again.
	HelperDefined bool
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
	plans := analysis.Analyze(file, info, analysis.Options{AllowMethods: opts.AllowMethods, FloatReassoc: opts.FloatReassoc})
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
	needsSimd := false

	for _, plan := range plans {
		stmtNode := plan.Loop.Node

		repl, pre, err := generateReplacement(plan, fset, len(replacements))
		if err != nil {
			continue
		}
		needsSimd = needsSimd || !plan.Copy

		// Distinctly-named slices sharing a caller's backing array with the
		// destination can overlap it at an offset (dst == src[1:], say), which
		// the vectorized loop's block-at-a-time loads/stores does not
		// preserve the scalar loop's element-by-element order for. Guard
		// with a runtime check and fall back to the original scalar loop
		// when any pair could overlap, the same way LLVM/GCC/HotSpot's own
		// vectorizers version a loop on an unprovable aliasing check rather
		// than require static proof. See AGENTS.md's correctness gaps.
		if len(plan.Overlaps) > 0 {
			origStart, origEnd := tf.Offset(stmtNode.Pos()), tf.Offset(stmtNode.End())
			repl = wrapWithOverlapCheck(plan, string(src[origStart:origEnd]), repl)
			needsOverlapHelper = true
		}
		if len(plan.Hoists) > 0 {
			repl = hoistSlices(plan, repl)
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

	// Add simd import if not present. A file whose loops all became copy calls
	// does not use it.
	var err error
	if needsSimd {
		result, err = ensureImport(result, "simd")
		if err != nil {
			return Result{}, err
		}
	}

	definesHelper := needsOverlapHelper && !opts.HelperDefined
	if definesHelper {
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

	return Result{Src: formatted, Rewrites: len(replacements), DefinesHelper: definesHelper}, nil
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

// hoistSlices wraps text, which refers to the slices a loop reaches through a selector
// by their Names, in a block that first assigns each from its source expression. The
// loop would not have read the fields at all if it ran no iterations (x may be a nil
// pointer), so the block is entered only when it runs at least one.
func hoistSlices(plan *loopir.Plan, text string) string {
	names := make([]string, len(plan.Hoists))
	srcs := make([]string, len(plan.Hoists))
	for i, h := range plan.Hoists {
		names[i], srcs[i] = h.Name, h.Src
	}
	assign := strings.Join(names, ", ") + " := " + strings.Join(srcs, ", ")
	if plan.NonEmpty {
		return "{\n" + assign + "\n" + text + "\n}"
	}
	return "if " + plan.Limit + " > " + plan.Start + " {\n" + assign + "\n" + text + "\n}"
}

// wrapWithOverlapCheck guards simdText behind a runtime check that no stored
// slice's backing memory overlaps another slice in the loop (plan.Overlaps),
// falling back to origText (the loop's own original source) when it might. This is the
// same technique LLVM, GCC, and (since 2025) HotSpot's C2 use for loops
// they can't statically prove don't alias: a bounded runtime check plus a
// scalar fallback, not a static proof requirement.
func wrapWithOverlapCheck(plan *loopir.Plan, origText, simdText string) string {
	var cond strings.Builder
	for i, pair := range plan.Overlaps {
		if i > 0 {
			cond.WriteString(" || ")
		}
		if plan.Bound == "" && plan.Start == "0" {
			// The loop runs over whole slices, so comparing whole slices covers
			// every element it touches.
			fmt.Fprintf(&cond, "_loopvecOverlap(%s, %s)", pair[0].Name, pair[1].Name)
			continue
		}
		aLo, aHi := window(plan, pair[0])
		bLo, bHi := window(plan, pair[1])
		fmt.Fprintf(&cond, "_loopvecOverlap(%s, %s, %s, %s, %s, %s)", pair[0].Name, pair[1].Name, aLo, aHi, bLo, bHi)
	}
	return "if " + cond.String() + " {\n" + origText + "\n} else {\n" + simdText + "\n}"
}

// window is the range of indexes of ref the loop touches, as source text: from
// the first iteration plus the lowest offset it is accessed at to the limit plus
// the highest. A loop over part of a slice then does not conflict with another
// slice that overlaps only the rest of it.
func window(plan *loopir.Plan, ref *loopir.Ref) (lo, hi string) {
	lowest, highest := 0, 0
	var variable []string
	for _, access := range plan.Checked {
		if access.Ref.Obj != ref.Obj || access.Off == "" {
			continue
		}
		if v, err := strconv.Atoi(access.Off); err == nil {
			lowest, highest = min(lowest, v), max(highest, v)
		} else {
			variable = append(variable, access.Off)
		}
	}
	if len(variable) == 0 {
		return addOffset(plan.Start, offsetText(lowest)), addOffset(plan.Limit, offsetText(highest))
	}
	lowArgs := strings.Join(append([]string{strconv.Itoa(lowest)}, variable...), ", ")
	highArgs := strings.Join(append([]string{strconv.Itoa(highest)}, variable...), ", ")
	return addOffset(plan.Start, "min("+lowArgs+")"), addOffset(plan.Limit, "max("+highArgs+")")
}

// offsetText is v as an offset for addOffset, "" for zero.
func offsetText(v int) string {
	if v == 0 {
		return ""
	}
	return strconv.Itoa(v)
}

// overlapHelperSrc is _loopvecOverlap's source, appended once per rewritten
// file that needs it.
const overlapHelperSrc = `
func _loopvecOverlap[T any](a, b []T, window ...int) bool {
	aLo, aHi, bLo, bHi := 0, len(a), 0, len(b)
	if len(window) == 4 {
		aLo, aHi, bLo, bHi = window[0], window[1], window[2], window[3]
	}
	if aLo >= aHi || bLo >= bHi {
		return false
	}
	size := unsafe.Sizeof(a[0])
	aBase := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bBase := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	aStart, aEnd := aBase+uintptr(aLo)*size, aBase+uintptr(aHi)*size
	bStart, bEnd := bBase+uintptr(bLo)*size, bBase+uintptr(bHi)*size
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
