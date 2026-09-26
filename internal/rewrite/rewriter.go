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
	if !opts.Compiler && hasBuildConstraint(file) {
		return Result{Src: src}, nil
	}
	loops := analysis.Analyze(file, info, opts.AllowMethods)
	if len(loops) == 0 {
		return Result{Src: src}, nil
	}

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

	for _, loop := range loops {
		var stmtNode ast.Node
		if loop.RangeStmt != nil {
			stmtNode = loop.RangeStmt
		} else {
			stmtNode = loop.ForStmt
		}

		repl, pre, err := generateReplacement(loop, fset, info, len(replacements))
		if err != nil {
			continue
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
	tf := fset.File(file.Pos())
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

// hasBuildConstraint reports whether file carries a //go:build or // +build
// line before its package clause. Such files are left untouched.
func hasBuildConstraint(file *ast.File) bool {
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

// generateReplacement returns (loopText, preText, error) for a vectorizable loop.
// preText is optional code to insert before the loop (e.g., broadcast variable).
// idx is a per-file unique counter used to avoid name collisions when multiple
// broadcast variables are declared in the same function scope.
func generateReplacement(loop analysis.Loop, fset *token.FileSet, info *types.Info, idx int) (string, string, error) {
	simdType := loop.SimdTypeName()
	if simdType == "" {
		return "", "", fmt.Errorf("unsupported element type")
	}
	loadFn := "Load" + simdType + "Part"
	broadcastFn := "Broadcast" + simdType

	if loop.IsExprTree {
		return generateExprTree(loop, fset, simdType, loadFn, broadcastFn, idx)
	}

	if loop.IsUnary {
		return generateUnary(loop, simdType, loadFn)
	}

	opMethod := loop.OpMethod()

	var loopBuf bytes.Buffer
	var preBuf bytes.Buffer

	lenExpr := fmt.Sprintf("len(%s)", loop.DstSlice)

	vcName := fmt.Sprintf("_vc%s%d", simdType, idx)

	if loop.Scalar != nil && loop.Src1Slice == "" {
		// Fill broadcast: dst[i] = constant
		var scalarBuf bytes.Buffer
		if err := format.Node(&scalarBuf, fset, loop.Scalar); err != nil {
			return "", "", err
		}
		scalarText := scalarBuf.String()

		fmt.Fprintf(&preBuf, "%s := simd.%s(%s)", vcName, broadcastFn, scalarText)

		fmt.Fprintf(&loopBuf, "for _i := 0; _i < %s; {\n", lenExpr)
		fmt.Fprintf(&loopBuf, "\t_n := %s.StorePart(%s[_i:])\n", vcName, loop.DstSlice)
		fmt.Fprintf(&loopBuf, "\t_i += _n\n")
		fmt.Fprint(&loopBuf, "}")
	} else if loop.Scalar != nil {
		// Scalar broadcast with load: dst[i] op= scalar  or  dst[i] = src[i] op scalar
		var scalarBuf bytes.Buffer
		if err := format.Node(&scalarBuf, fset, loop.Scalar); err != nil {
			return "", "", err
		}
		scalarText := scalarBuf.String()

		fmt.Fprintf(&preBuf, "%s := simd.%s(%s)", vcName, broadcastFn, scalarText)

		loadSlice := loop.Src1Slice
		fmt.Fprintf(&loopBuf, "for _i := 0; _i < %s; {\n", lenExpr)
		fmt.Fprintf(&loopBuf, "\t_v1, _n := simd.%s(%s[_i:])\n", loadFn, loadSlice)
		fmt.Fprintf(&loopBuf, "\t_v1.%s(%s).StorePart(%s[_i:])\n", opMethod, vcName, loop.DstSlice)
		fmt.Fprintf(&loopBuf, "\t_i += _n\n")
		fmt.Fprint(&loopBuf, "}")
	} else if loop.Src2Slice == "" {
		// Should not happen if src2 and scalar are both empty, but guard.
		return "", "", fmt.Errorf("no source operand")
	} else if loop.Src1Slice == loop.DstSlice {
		// In-place: dst[i] op= src2[i]
		fmt.Fprintf(&loopBuf, "for _i := 0; _i < %s; {\n", lenExpr)
		fmt.Fprintf(&loopBuf, "\t_v1, _n := simd.%s(%s[_i:])\n", loadFn, loop.DstSlice)
		fmt.Fprintf(&loopBuf, "\t_v2, _ := simd.%s(%s[_i:])\n", loadFn, loop.Src2Slice)
		fmt.Fprintf(&loopBuf, "\t_v1.%s(_v2).StorePart(%s[_i:])\n", opMethod, loop.DstSlice)
		fmt.Fprintf(&loopBuf, "\t_i += _n\n")
		fmt.Fprint(&loopBuf, "}")
	} else {
		// dst[i] = src1[i] op src2[i]
		fmt.Fprintf(&loopBuf, "for _i := 0; _i < %s; {\n", lenExpr)
		fmt.Fprintf(&loopBuf, "\t_v1, _n := simd.%s(%s[_i:])\n", loadFn, loop.Src1Slice)
		fmt.Fprintf(&loopBuf, "\t_v2, _ := simd.%s(%s[_i:])\n", loadFn, loop.Src2Slice)
		fmt.Fprintf(&loopBuf, "\t_v1.%s(_v2).StorePart(%s[_i:])\n", opMethod, loop.DstSlice)
		fmt.Fprintf(&loopBuf, "\t_i += _n\n")
		fmt.Fprint(&loopBuf, "}")
	}

	return loopBuf.String(), preBuf.String(), nil
}

// generateUnary generates a simd loop for dst[i] = -src[i] or dst[i] = ^src[i].
func generateUnary(loop analysis.Loop, simdType, loadFn string) (string, string, error) {
	var loopBuf bytes.Buffer
	lenExpr := fmt.Sprintf("len(%s)", loop.DstSlice)
	fmt.Fprintf(&loopBuf, "for _i := 0; _i < %s; {\n", lenExpr)
	fmt.Fprintf(&loopBuf, "\t_v1, _n := simd.%s(%s[_i:])\n", loadFn, loop.Src1Slice)
	fmt.Fprintf(&loopBuf, "\t_v1.%s().StorePart(%s[_i:])\n", loop.OpMethod(), loop.DstSlice)
	fmt.Fprintf(&loopBuf, "\t_i += _n\n")
	fmt.Fprint(&loopBuf, "}")
	return loopBuf.String(), "", nil
}

// generateExprTree generates a simd loop for a depth-2 binary expression tree:
//
//	dst[i] = (Src1[i] Op Src2OrScalar) Op2 Src3OrScalar2
//	or (InnerOnRight): Src3OrScalar2 Op2 (Src1[i] Op Src2OrScalar)
//
// Float32s/Float64s with Op==Mul and Op2==Add use MulAdd (FMA).
func generateExprTree(loop analysis.Loop, fset *token.FileSet, simdType, loadFn, broadcastFn string, idx int) (string, string, error) {
	var loopBuf, preBuf bytes.Buffer

	// Pre-loop broadcast declarations. Names include idx to avoid collisions
	// when multiple loops in the same scope broadcast the same simd type.
	var innerBc, outerBc string
	if loop.Scalar != nil {
		var buf bytes.Buffer
		if err := format.Node(&buf, fset, loop.Scalar); err != nil {
			return "", "", err
		}
		innerBc = fmt.Sprintf("_vcA%s%d", simdType, idx)
		fmt.Fprintf(&preBuf, "%s := simd.%s(%s)", innerBc, broadcastFn, buf.String())
	}
	if loop.Scalar2 != nil {
		if preBuf.Len() > 0 {
			preBuf.WriteByte('\n')
		}
		var buf bytes.Buffer
		if err := format.Node(&buf, fset, loop.Scalar2); err != nil {
			return "", "", err
		}
		outerBc = fmt.Sprintf("_vcB%s%d", simdType, idx)
		fmt.Fprintf(&preBuf, "%s := simd.%s(%s)", outerBc, broadcastFn, buf.String())
	}

	isFloat := simdType == "Float32s" || simdType == "Float64s"
	canFMA := isFloat && loop.Op == analysis.OpMul && loop.Op2 == analysis.OpAdd && !loop.InnerOnRight

	lenExpr := fmt.Sprintf("len(%s)", loop.DstSlice)
	fmt.Fprintf(&loopBuf, "for _i := 0; _i < %s; {\n", lenExpr)
	fmt.Fprintf(&loopBuf, "\t_v1, _n := simd.%s(%s[_i:])\n", loadFn, loop.Src1Slice)

	// Inner's second operand.
	var innerY string
	if loop.Src2Slice != "" {
		fmt.Fprintf(&loopBuf, "\t_v2, _ := simd.%s(%s[_i:])\n", loadFn, loop.Src2Slice)
		innerY = "_v2"
	} else {
		innerY = innerBc
	}

	// Outer operand.
	var outerX string
	if loop.Src3Slice != "" {
		v := "_v3"
		if loop.Src2Slice == "" {
			v = "_v2" // safe to reuse slot when inner second operand is a broadcast
		}
		fmt.Fprintf(&loopBuf, "\t%s, _ := simd.%s(%s[_i:])\n", v, loadFn, loop.Src3Slice)
		outerX = v
	} else {
		outerX = outerBc
	}

	// outerExpr is the final outer operand. When OuterIsMul the outer slice is
	// multiplied by its scalar broadcast before being used.
	outerExpr := outerX
	if loop.OuterIsMul {
		outerExpr = outerX + ".Mul(" + outerBc + ")"
	}

	if canFMA {
		fmt.Fprintf(&loopBuf, "\t_v1.MulAdd(%s, %s).StorePart(%s[_i:])\n", innerY, outerExpr, loop.DstSlice)
	} else if loop.InnerOnRight {
		// outerExpr Op2 (v1 Op innerY)
		fmt.Fprintf(&loopBuf, "\t%s.%s(_v1.%s(%s)).StorePart(%s[_i:])\n",
			outerExpr, loop.Op2Method(), loop.OpMethod(), innerY, loop.DstSlice)
	} else {
		// (v1 Op innerY) Op2 outerExpr
		fmt.Fprintf(&loopBuf, "\t_v1.%s(%s).%s(%s).StorePart(%s[_i:])\n",
			loop.OpMethod(), innerY, loop.Op2Method(), outerExpr, loop.DstSlice)
	}

	fmt.Fprintf(&loopBuf, "\t_i += _n\n")
	fmt.Fprint(&loopBuf, "}")
	return loopBuf.String(), preBuf.String(), nil
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
