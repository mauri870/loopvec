package analysis

import (
	"go/ast"
	"go/token"
	"go/types"
)

// Reason explains why a candidate loop was not vectorized. The vocabulary is
// deliberately small and coarse: it names which stage of matching first
// rejected the loop, not the exact AST shape that failed. The same strings
// are meant to back a future -explain mode.
type Reason string

const (
	// ReasonUnsupportedClauses covers a for-loop whose init, condition, post,
	// or iteration bound doesn't match a supported shape: non-zero/non-len-1
	// start, a step other than ++/--, a bound that isn't len(slice) or a
	// simple int expression, and so on.
	ReasonUnsupportedClauses Reason = "loop clauses (start, step, direction, or bound) do not match a supported shape"
	// ReasonUnsupportedDestination covers a destination that isn't a plain
	// slice[i] element: an offset (dst[i+1]), a stride (dst[2*i]), or a
	// nested index (dst[i][j]).
	ReasonUnsupportedDestination Reason = "destination is not a simple, same-index slice element"
	// ReasonUnsupportedBody covers a loop body that isn't a single indexed
	// assignment: multiple statements, or a nested loop.
	ReasonUnsupportedBody Reason = "loop body is not a single indexed assignment"
	// ReasonUnsupportedOperand covers a right-hand side that isn't a
	// same-index slice read, the range value variable, or a plain
	// identifier/literal scalar.
	ReasonUnsupportedOperand Reason = "operand is not a same-index slice, range value, or plain scalar"
	// ReasonUnsupportedType covers an element type simd has no vector type
	// for at all (for example, string).
	ReasonUnsupportedType Reason = "element type not supported by simd"
	// ReasonUnsupportedOp covers an operation simd doesn't implement for an
	// otherwise-supported element type (for example, int64 multiply).
	ReasonUnsupportedOp Reason = "operation not supported by simd for this element type"
	// ReasonMethodSkipped covers a loop inside a method, skipped unless
	// -methods is passed (see Analyze's doc comment).
	ReasonMethodSkipped Reason = "loops inside methods are skipped without -methods"
	// ReasonZeroFillSkipped covers dst[i] = 0: the compiler already turns
	// this into memclr, which beats a vector loop, so it's left alone on
	// purpose rather than for lack of support.
	ReasonZeroFillSkipped Reason = "zero fill is left to the compiler's memclr"
	// ReasonUnrecognized is used when a kernel or function has no reported
	// candidate loop at all: nothing in it looked like an indexed-store loop.
	ReasonUnrecognized Reason = "not recognized"
)

// Candidate is one loop that at least has the shape of an element-wise
// indexed-store loop. Loops analysis.Analyze accepts have Vectorized set;
// Reason explains every other one.
type Candidate struct {
	File       string
	Line       int
	Func       string
	Vectorized bool
	Reason     Reason
}

// reasonSink records the first rejection reason set for a candidate loop.
// A nil *reasonSink is valid and silently discards every set, so the
// existing analyze* functions used by Analyze need no behavior change.
type reasonSink struct{ reason Reason }

func (s *reasonSink) set(r Reason) {
	if s != nil && s.reason == "" {
		s.reason = r
	}
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
				c.Reason = ReasonMethodSkipped
				out = append(out, c)
				return true
			}

			var why reasonSink
			var loop Loop
			var loopOK bool
			switch stmt := n.(type) {
			case *ast.RangeStmt:
				loop, loopOK = analyzeRange(stmt, info, &why)
			case *ast.ForStmt:
				loop, loopOK = analyzeFor(stmt, info, &why)
			}
			switch {
			case !loopOK:
				c.Reason = why.reason
				if c.Reason == "" {
					c.Reason = ReasonUnsupportedOperand
				}
			case !simdSupportsOp(loop.ElemType, loop.Op) || (loop.IsExprTree && !simdSupportsOp(loop.ElemType, loop.Op2)):
				c.Reason = ReasonUnsupportedOp
			default:
				c.Vectorized = true
			}
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
