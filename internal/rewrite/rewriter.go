// Package rewrite rewrites loops identified by the analysis package to use simd.
package rewrite

import (
	"bytes"
	"fmt"
	"go/ast"
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

// File rewrites all vectorizable loops in src, returning the modified source.
// fset and file must correspond to the parsed src. info must have type
// information populated for the file.
func File(fset *token.FileSet, file *ast.File, info *types.Info, src []byte) (Result, error) {
	loops := analysis.Analyze(file, info)
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

		repl, pre, err := generateReplacement(loop, fset, info)
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
	result = ensureBuildTag(result, "goexperiment.simd")

	// Format the result.
	formatted, err := format.Source(result)
	if err != nil {
		return Result{}, fmt.Errorf("format error: %w\nsource:\n%s", err, result)
	}

	return Result{Src: formatted, Rewrites: len(replacements)}, nil
}

// generateReplacement returns (loopText, preText, error) for a vectorizable loop.
// preText is optional code to insert before the loop (e.g., broadcast variable).
func generateReplacement(loop analysis.Loop, fset *token.FileSet, info *types.Info) (string, string, error) {
	simdType := loop.SimdTypeName()
	if simdType == "" {
		return "", "", fmt.Errorf("unsupported element type")
	}
	loadFn := "Load" + simdType + "Part"
	broadcastFn := "Broadcast" + simdType

	if loop.IsExprTree {
		return generateExprTree(loop, fset, simdType, loadFn, broadcastFn)
	}

	opMethod := loop.OpMethod()

	var loopBuf bytes.Buffer
	var preBuf bytes.Buffer

	lenExpr := fmt.Sprintf("len(%s)", loop.DstSlice)

	if loop.Scalar != nil && loop.Src1Slice == "" {
		// Fill broadcast: dst[i] = constant
		var scalarBuf bytes.Buffer
		if err := format.Node(&scalarBuf, fset, loop.Scalar); err != nil {
			return "", "", err
		}
		scalarText := scalarBuf.String()

		fmt.Fprintf(&preBuf, "_vc%s := simd.%s(%s)", simdType, broadcastFn, scalarText)

		fmt.Fprintf(&loopBuf, "for _i := 0; _i < %s; {\n", lenExpr)
		fmt.Fprintf(&loopBuf, "\t_n := _vc%s.StorePart(%s[_i:])\n", simdType, loop.DstSlice)
		fmt.Fprintf(&loopBuf, "\t_i += _n\n")
		fmt.Fprint(&loopBuf, "}")
	} else if loop.Scalar != nil {
		// Scalar broadcast with load: dst[i] op= scalar  or  dst[i] = src[i] op scalar
		var scalarBuf bytes.Buffer
		if err := format.Node(&scalarBuf, fset, loop.Scalar); err != nil {
			return "", "", err
		}
		scalarText := scalarBuf.String()

		fmt.Fprintf(&preBuf, "_vc%s := simd.%s(%s)", simdType, broadcastFn, scalarText)

		loadSlice := loop.Src1Slice
		fmt.Fprintf(&loopBuf, "for _i := 0; _i < %s; {\n", lenExpr)
		fmt.Fprintf(&loopBuf, "\t_v1, _n := simd.%s(%s[_i:])\n", loadFn, loadSlice)
		fmt.Fprintf(&loopBuf, "\t_v1.%s(_vc%s).StorePart(%s[_i:])\n", opMethod, simdType, loop.DstSlice)
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

// generateExprTree generates a simd loop for a depth-2 binary expression tree:
//
//	dst[i] = (Src1[i] Op Src2OrScalar) Op2 Src3OrScalar2
//	or (InnerOnRight): Src3OrScalar2 Op2 (Src1[i] Op Src2OrScalar)
//
// Float32s/Float64s with Op==Mul and Op2==Add use MulAdd (FMA).
func generateExprTree(loop analysis.Loop, fset *token.FileSet, simdType, loadFn, broadcastFn string) (string, string, error) {
	var loopBuf, preBuf bytes.Buffer

	// Pre-loop broadcast declarations.
	var innerBc, outerBc string
	if loop.Scalar != nil {
		var buf bytes.Buffer
		if err := format.Node(&buf, fset, loop.Scalar); err != nil {
			return "", "", err
		}
		innerBc = "_vcA" + simdType
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
		outerBc = "_vcB" + simdType
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
