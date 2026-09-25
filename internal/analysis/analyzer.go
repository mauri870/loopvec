// Package analysis detects loops that can be vectorized using the simd package.
package analysis

import (
	"go/ast"
	"go/token"
	"go/types"
)

// Op is a binary operation supported by simd.
type Op int

const (
	OpAdd Op = iota
	OpSub
	OpMul
	OpAnd
	OpOr
	OpXor
)

// Loop describes a vectorizable loop found in the source.
type Loop struct {
	// ForStmt is the for statement node.
	ForStmt *ast.ForStmt
	// RangeStmt is the range statement node (mutually exclusive with ForStmt).
	RangeStmt *ast.RangeStmt
	// IndexVar is the name of the loop index variable.
	IndexVar string
	// DstSlice is the name of the destination slice.
	DstSlice string
	// Src1Slice is the name of the first source slice (may equal DstSlice for in-place ops).
	Src1Slice string
	// Src2Slice is the name of the second source slice (empty for scalar ops).
	Src2Slice string
	// Scalar is the scalar expression for broadcast ops (nil for slice-slice ops).
	Scalar ast.Expr
	// Op is the binary operation.
	Op Op
	// ElemType is the slice element type.
	ElemType types.Type
}

// simdElemType returns the simd type name for a given element type, or empty string if not supported.
func simdElemType(t types.Type) string {
	basic, ok := t.Underlying().(*types.Basic)
	if !ok {
		return ""
	}
	switch basic.Kind() {
	case types.Int8:
		return "Int8s"
	case types.Int16:
		return "Int16s"
	case types.Int32:
		return "Int32s"
	case types.Int64:
		return "Int64s"
	case types.Uint8:
		return "Uint8s"
	case types.Uint16:
		return "Uint16s"
	case types.Uint32:
		return "Uint32s"
	case types.Uint64:
		return "Uint64s"
	case types.Float32:
		return "Float32s"
	case types.Float64:
		return "Float64s"
	}
	return ""
}

// SimdTypeName returns the simd vector type name for the loop's element type, or empty string.
func (l *Loop) SimdTypeName() string {
	return simdElemType(l.ElemType)
}

// OpMethod returns the simd method name for the loop's operation.
func (l *Loop) OpMethod() string {
	switch l.Op {
	case OpAdd:
		return "Add"
	case OpSub:
		return "Sub"
	case OpMul:
		return "Mul"
	case OpAnd:
		return "And"
	case OpOr:
		return "Or"
	case OpXor:
		return "Xor"
	}
	return ""
}

// tokenOpToLoopOp converts a token.Token binary op to a Loop Op.
func tokenOpToLoopOp(tok token.Token) (Op, bool) {
	switch tok {
	case token.ADD, token.ADD_ASSIGN:
		return OpAdd, true
	case token.SUB, token.SUB_ASSIGN:
		return OpSub, true
	case token.MUL, token.MUL_ASSIGN:
		return OpMul, true
	case token.AND, token.AND_ASSIGN:
		return OpAnd, true
	case token.OR, token.OR_ASSIGN:
		return OpOr, true
	case token.XOR, token.XOR_ASSIGN:
		return OpXor, true
	}
	return 0, false
}

// Analyze walks a file's AST and returns all vectorizable loops.
func Analyze(file *ast.File, info *types.Info) []Loop {
	var loops []Loop
	ast.Inspect(file, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.RangeStmt:
			if l, ok := analyzeRange(stmt, info); ok {
				loops = append(loops, l)
			}
		case *ast.ForStmt:
			if l, ok := analyzeFor(stmt, info); ok {
				loops = append(loops, l)
			}
		}
		return true
	})
	return loops
}

// analyzeRange checks if a range loop is vectorizable.
// Accepted pattern:
//
//	for i := range dst { dst[i] = src1[i] op src2[i] }
//	for i := range dst { dst[i] op= src[i] }
//	for i := range dst { dst[i] op= scalar }
func analyzeRange(stmt *ast.RangeStmt, info *types.Info) (Loop, bool) {
	// Must have a single key variable (no value).
	keyIdent, ok := stmt.Key.(*ast.Ident)
	if !ok {
		return Loop{}, false
	}
	if stmt.Value != nil && stmt.Value != ast.NewIdent("_") {
		if ident, ok := stmt.Value.(*ast.Ident); !ok || ident.Name != "_" {
			return Loop{}, false
		}
	}
	// Range target must be a slice identifier.
	rangeIdent, ok := stmt.X.(*ast.Ident)
	if !ok {
		return Loop{}, false
	}

	return analyzeBody(stmt, nil, keyIdent.Name, rangeIdent.Name, stmt.Body, info)
}

// analyzeFor checks if a three-clause for loop is vectorizable.
// Accepted pattern:
//
//	for i := 0; i < len(s); i++ { s[i] = ... }
func analyzeFor(stmt *ast.ForStmt, info *types.Info) (Loop, bool) {
	if stmt.Init == nil || stmt.Cond == nil || stmt.Post == nil {
		return Loop{}, false
	}
	// Init: i := 0
	initAssign, ok := stmt.Init.(*ast.AssignStmt)
	if !ok || initAssign.Tok != token.DEFINE || len(initAssign.Lhs) != 1 || len(initAssign.Rhs) != 1 {
		return Loop{}, false
	}
	indexIdent, ok := initAssign.Lhs[0].(*ast.Ident)
	if !ok {
		return Loop{}, false
	}
	initLit, ok := initAssign.Rhs[0].(*ast.BasicLit)
	if !ok || initLit.Value != "0" {
		return Loop{}, false
	}

	// Cond: i < len(s)
	cond, ok := stmt.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.LSS {
		return Loop{}, false
	}
	condLhs, ok := cond.X.(*ast.Ident)
	if !ok || condLhs.Name != indexIdent.Name {
		return Loop{}, false
	}
	lenCall, ok := cond.Y.(*ast.CallExpr)
	if !ok {
		return Loop{}, false
	}
	lenIdent, ok := lenCall.Fun.(*ast.Ident)
	if !ok || lenIdent.Name != "len" || len(lenCall.Args) != 1 {
		return Loop{}, false
	}
	sliceIdent, ok := lenCall.Args[0].(*ast.Ident)
	if !ok {
		return Loop{}, false
	}

	// Post: i++
	incStmt, ok := stmt.Post.(*ast.IncDecStmt)
	if !ok || incStmt.Tok != token.INC {
		return Loop{}, false
	}
	incIdent, ok := incStmt.X.(*ast.Ident)
	if !ok || incIdent.Name != indexIdent.Name {
		return Loop{}, false
	}

	return analyzeBody(nil, stmt, indexIdent.Name, sliceIdent.Name, stmt.Body, info)
}

// analyzeBody checks the loop body for vectorizable assignments.
func analyzeBody(rangeStmt *ast.RangeStmt, forStmt *ast.ForStmt, indexVar, boundsSlice string, body *ast.BlockStmt, info *types.Info) (Loop, bool) {
	if len(body.List) != 1 {
		return Loop{}, false
	}

	loop := Loop{
		RangeStmt: rangeStmt,
		ForStmt:   forStmt,
		IndexVar:  indexVar,
	}

	assignStmt, ok := body.List[0].(*ast.AssignStmt)
	if !ok || len(assignStmt.Lhs) != 1 || len(assignStmt.Rhs) != 1 {
		return Loop{}, false
	}

	if assignStmt.Tok == token.ASSIGN {
		// dst[i] = src1[i] op src2[i]
		lhsIndex, ok := asSliceIndex(assignStmt.Lhs[0], indexVar)
		if !ok {
			return Loop{}, false
		}
		loop.DstSlice = lhsIndex

		binExpr, ok := assignStmt.Rhs[0].(*ast.BinaryExpr)
		if !ok {
			return Loop{}, false
		}
		op, ok := tokenOpToLoopOp(binExpr.Op)
		if !ok {
			return Loop{}, false
		}
		loop.Op = op

		src1, ok := asSliceIndex(binExpr.X, indexVar)
		if !ok {
			return Loop{}, false
		}
		loop.Src1Slice = src1

		if src2, ok := asSliceIndex(binExpr.Y, indexVar); ok {
			loop.Src2Slice = src2
		} else {
			loop.Scalar = binExpr.Y
		}
	} else {
		// dst[i] op= src[i]  or  dst[i] op= scalar
		op, ok := tokenOpToLoopOp(assignStmt.Tok)
		if !ok {
			return Loop{}, false
		}
		loop.Op = op

		lhsSlice, ok := asSliceIndex(assignStmt.Lhs[0], indexVar)
		if !ok {
			return Loop{}, false
		}
		loop.DstSlice = lhsSlice
		loop.Src1Slice = lhsSlice

		if src2, ok := asSliceIndex(assignStmt.Rhs[0], indexVar); ok {
			loop.Src2Slice = src2
		} else {
			loop.Scalar = assignStmt.Rhs[0]
		}
	}

	// Resolve element type via type info.
	elemType, ok := resolveSliceElemType(boundsSlice, body, info)
	if !ok {
		return Loop{}, false
	}
	if simdElemType(elemType) == "" {
		return Loop{}, false
	}
	loop.ElemType = elemType

	return loop, true
}

// asSliceIndex checks if expr is of the form slice[index] where index matches the given name.
// Returns the slice identifier name on success.
func asSliceIndex(expr ast.Expr, indexVar string) (string, bool) {
	indexExpr, ok := expr.(*ast.IndexExpr)
	if !ok {
		return "", false
	}
	sliceIdent, ok := indexExpr.X.(*ast.Ident)
	if !ok {
		return "", false
	}
	idxIdent, ok := indexExpr.Index.(*ast.Ident)
	if !ok {
		return "", false
	}
	if idxIdent.Name != indexVar {
		return "", false
	}
	return sliceIdent.Name, true
}

// resolveSliceElemType finds a slice identifier in scope and returns its element type.
func resolveSliceElemType(sliceName string, body *ast.BlockStmt, info *types.Info) (types.Type, bool) {
	for expr, tv := range info.Types {
		ident, ok := expr.(*ast.Ident)
		if !ok || ident.Name != sliceName {
			continue
		}
		slice, ok := tv.Type.(*types.Slice)
		if !ok {
			continue
		}
		return slice.Elem(), true
	}
	return nil, false
}
