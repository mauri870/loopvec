package analysis

import (
	"go/ast"
	"go/token"
	"go/types"

	"github.com/mauri870/loopvec/internal/loopir"
)

// Candidate is one loop that at least has the shape of an element-wise
// indexed-store loop. Loops analysis.Analyze accepts have Vectorized set;
// Reason explains every other one.
type Candidate struct {
	File       string
	Line       int
	Func       string
	Vectorized bool
	Reason     loopir.Reason
}

// AnalyzeExplain walks file like Analyze, but reports every candidate loop —
// not just the ones that got vectorized — with a reason when one didn't.
// A loop with no indexed-assignment body at all isn't a candidate and isn't
// reported; see hasIndexedAssignment.
func AnalyzeExplain(fset *token.FileSet, file *ast.File, info *types.Info, allowMethods bool) []Candidate {
	var out []Candidate
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		isMethod := fn.Recv != nil
		results := loopir.NamedResults(fn, info)
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			var body *ast.BlockStmt
			switch stmt := n.(type) {
			case *ast.RangeStmt:
				body = stmt.Body
			case *ast.ForStmt:
				body = stmt.Body
			default:
				return true
			}
			if !hasIndexedAssignment(body) {
				return true
			}
			pos := fset.Position(n.Pos())
			c := Candidate{File: pos.Filename, Line: pos.Line, Func: fn.Name.Name}

			if isMethod && !allowMethods {
				c.Reason = loopir.ReasonMethodSkipped
				out = append(out, c)
				return true
			}

			plan, reason := analyzeLoop(n.(ast.Stmt), info, results)
			c.Vectorized = plan != nil
			c.Reason = reason
			out = append(out, c)
			return true
		})
	}
	return out
}

// hasIndexedAssignment reports whether body assigns to a slice-indexed
// expression somewhere outside a nested loop or function literal, or updates a
// variable from one (sum += a[i]), the minimum for a loop to be reported as a
// candidate at all. A body that is more than one assignment, or wraps it in an
// if, is still a candidate: AnalyzeExplain says why it is not vectorized. A
// nested loop is a candidate of its own, so the search does not enter it. It
// doesn't check that the index matches the loop variable -- that's exactly the
// kind of rejection AnalyzeExplain reports a reason for.
func hasIndexedAssignment(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.ForStmt, *ast.RangeStmt, *ast.FuncLit:
			return false
		case *ast.AssignStmt:
			for _, lhs := range n.Lhs {
				if _, ok := ast.Unparen(lhs).(*ast.IndexExpr); ok {
					found = true
				}
			}
			if _, ok := n.Lhs[0].(*ast.Ident); ok && n.Tok != token.DEFINE {
				for _, rhs := range n.Rhs {
					ast.Inspect(rhs, func(n ast.Node) bool {
						if _, ok := n.(*ast.IndexExpr); ok {
							found = true
						}
						return !found
					})
				}
			}
		}
		return !found
	})
	return found
}
