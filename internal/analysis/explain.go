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

			plan, reason := analyzeLoop(n.(ast.Stmt), info)
			c.Vectorized = plan != nil
			c.Reason = reason
			out = append(out, c)
			return true
		})
	}
	return out
}

// hasIndexedAssignment reports whether body is a single statement assigning
// to a slice-indexed expression, the minimum shape for a loop to be reported
// as a candidate at all. It doesn't check that the index matches the loop
// variable — that's exactly the kind of rejection AnalyzeExplain reports a
// reason for.
func hasIndexedAssignment(body *ast.BlockStmt) bool {
	if len(body.List) != 1 {
		return false
	}
	assign, ok := body.List[0].(*ast.AssignStmt)
	if !ok || len(assign.Lhs) != 1 {
		return false
	}
	_, ok = ast.Unparen(assign.Lhs[0]).(*ast.IndexExpr)
	return ok
}
