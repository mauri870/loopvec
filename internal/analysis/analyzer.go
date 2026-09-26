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
	OpDiv
	OpNeg // unary: -x
	OpNot // unary: ^x
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

	// Depth-2 expression tree: dst[i] = (Src1[i] Op Src2OrScalar) Op2 Src3OrScalar2
	// or (when InnerOnRight) Src3OrScalar2 Op2 (Src1[i] Op Src2OrScalar).
	// Float32s/Float64s with Op==Mul and Op2==Add use MulAdd (FMA).
	IsExprTree   bool
	InnerOnRight bool
	Op2          Op
	Src3Slice    string
	Scalar2      ast.Expr
	// OuterIsMul is set when the outer operand is itself a multiplication:
	// (Src3Slice[i] * Scalar2). Enables patterns like
	// dst[i] = a[i]*alpha + b[i]*beta → _v1.MulAdd(_vcAlpha, _v2.Mul(_vcBeta)).
	OuterIsMul bool

	// IsUnary is true for loops of the form dst[i] = -src[i] or dst[i] = ^src[i].
	IsUnary bool
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

// simdSupportsOp reports whether the simd package supports op for the given element type.
// int64 and uint64 have no Mul; only float32 and float64 have Div.
func simdSupportsOp(t types.Type, op Op) bool {
	basic, ok := t.Underlying().(*types.Basic)
	if !ok {
		return true
	}
	switch op {
	case OpMul:
		return basic.Kind() != types.Int64 && basic.Kind() != types.Uint64
	case OpDiv:
		return basic.Kind() == types.Float32 || basic.Kind() == types.Float64
	}
	return true
}

// OpMethod returns the simd method name for the loop's inner operation.
func (l *Loop) OpMethod() string {
	return opMethod(l.Op)
}

// Op2Method returns the simd method name for the loop's outer operation (depth-2 trees).
func (l *Loop) Op2Method() string {
	return opMethod(l.Op2)
}

func opMethod(op Op) string {
	switch op {
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
	case OpDiv:
		return "Div"
	case OpNeg:
		return "Neg"
	case OpNot:
		return "Not"
	}
	return ""
}

// negSupported reports whether the simd package implements Neg for t.
// Unsigned integer types do not have Neg.
func negSupported(t types.Type) bool {
	basic, ok := t.Underlying().(*types.Basic)
	if !ok {
		return false
	}
	switch basic.Kind() {
	case types.Uint8, types.Uint16, types.Uint32, types.Uint64:
		return false
	}
	return simdElemType(t) != ""
}

// notSupported reports whether the simd package implements Not for t.
// Floating-point types do not have Not.
func notSupported(t types.Type) bool {
	basic, ok := t.Underlying().(*types.Basic)
	if !ok {
		return false
	}
	switch basic.Kind() {
	case types.Float32, types.Float64:
		return false
	}
	return simdElemType(t) != ""
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
	case token.QUO, token.QUO_ASSIGN:
		return OpDiv, true
	}
	return 0, false
}

// Analyze walks a file's AST and returns all vectorizable loops.
// When allowMethods is false, loops inside methods (functions with a receiver)
// are skipped because GOEXPERIMENT=simd has a compiler bug with method bodies
// in simd-tagged files (https://github.com/golang/go/issues/80657).
// Pass allowMethods=true only when using a toolchain that has the fix
// (Go 1.28+ / gotip with CL 839405).
func Analyze(file *ast.File, info *types.Info, allowMethods bool) []Loop {
	var loops []Loop
	inMethod := false
	ast.Inspect(file, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		if fn, ok := n.(*ast.FuncDecl); ok {
			inMethod = fn.Recv != nil
			return true
		}
		if inMethod && !allowMethods {
			return true
		}
		switch stmt := n.(type) {
		case *ast.RangeStmt:
			if l, ok := analyzeRange(stmt, info); ok &&
				simdSupportsOp(l.ElemType, l.Op) &&
				(!l.IsExprTree || simdSupportsOp(l.ElemType, l.Op2)) {
				loops = append(loops, l)
			}
		case *ast.ForStmt:
			if l, ok := analyzeFor(stmt, info); ok &&
				simdSupportsOp(l.ElemType, l.Op) &&
				(!l.IsExprTree || simdSupportsOp(l.ElemType, l.Op2)) {
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
//	for i := range dst { dst[i] = constant }   (fill broadcast)
//	for i, v := range src { dst[i] = v op scalar }  (two-variable range)
func analyzeRange(stmt *ast.RangeStmt, info *types.Info) (Loop, bool) {
	// Key must be an identifier.
	keyIdent, ok := stmt.Key.(*ast.Ident)
	if !ok {
		return Loop{}, false
	}
	// Value may be absent, blank, or a named variable (two-variable range).
	var valueVar string
	if stmt.Value != nil {
		ident, ok := stmt.Value.(*ast.Ident)
		if !ok {
			return Loop{}, false
		}
		if ident.Name != "_" {
			valueVar = ident.Name
		}
	}
	// Range target must be a slice identifier.
	rangeIdent, ok := stmt.X.(*ast.Ident)
	if !ok {
		return Loop{}, false
	}

	return analyzeBody(stmt, nil, keyIdent.Name, rangeIdent, valueVar, stmt.Body, info)
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

	return analyzeBody(nil, stmt, indexIdent.Name, sliceIdent, "", stmt.Body, info)
}

// innerSliceAndOther normalizes a binary sub-expression so that the slice[i]
// operand is returned first. For commutative ops the slice may be on either side.
// Returns ok=false if neither operand is a slice[i] access.
func innerSliceAndOther(expr *ast.BinaryExpr, op Op, indexVar string) (sliceName string, other ast.Expr, ok bool) {
	if s, ok := asSliceIndex(expr.X, indexVar); ok {
		return s, expr.Y, true
	}
	switch op {
	case OpAdd, OpMul, OpAnd, OpOr, OpXor:
		if s, ok := asSliceIndex(expr.Y, indexVar); ok {
			return s, expr.X, true
		}
	}
	return "", nil, false
}

// buildExprTree constructs a depth-2 Loop from an inner binary expression,
// an outer leaf operand, and the outer op. innerOnRight marks non-commutative
// outer ops where the inner result is the right operand (e.g. "leaf Sub inner").
func buildExprTree(innerBin *ast.BinaryExpr, outerLeaf ast.Expr, outerOp Op, innerOnRight bool, indexVar string, proto Loop) (Loop, bool) {
	innerOp, ok := tokenOpToLoopOp(innerBin.Op)
	if !ok {
		return Loop{}, false
	}
	src1, innerOther, ok := innerSliceAndOther(innerBin, innerOp, indexVar)
	if !ok {
		return Loop{}, false
	}

	var src2 string
	var scalar ast.Expr
	if s, ok2 := asSliceIndex(innerOther, indexVar); ok2 {
		src2 = s
	} else {
		switch innerOther.(type) {
		case *ast.BasicLit, *ast.Ident:
			scalar = innerOther
		default:
			return Loop{}, false
		}
	}

	var src3 string
	var scalar2 ast.Expr
	var outerIsMul bool
	if s, ok2 := asSliceIndex(outerLeaf, indexVar); ok2 {
		src3 = s
	} else if outerBin, ok2 := outerLeaf.(*ast.BinaryExpr); ok2 {
		// Outer operand is itself a binary expr; only accept slice[i]*scalar.
		outerInnerOp, ok3 := tokenOpToLoopOp(outerBin.Op)
		if !ok3 || outerInnerOp != OpMul {
			return Loop{}, false
		}
		outerSrc, outerOther, ok3 := innerSliceAndOther(outerBin, OpMul, indexVar)
		if !ok3 {
			return Loop{}, false
		}
		switch outerOther.(type) {
		case *ast.BasicLit, *ast.Ident:
		default:
			return Loop{}, false
		}
		src3 = outerSrc
		scalar2 = outerOther
		outerIsMul = true
	} else {
		switch outerLeaf.(type) {
		case *ast.BasicLit, *ast.Ident:
			scalar2 = outerLeaf
		default:
			return Loop{}, false
		}
	}

	proto.IsExprTree = true
	proto.InnerOnRight = innerOnRight
	proto.Op = innerOp
	proto.Op2 = outerOp
	proto.Src1Slice = src1
	proto.Src2Slice = src2
	proto.Scalar = scalar
	proto.Src3Slice = src3
	proto.Scalar2 = scalar2
	proto.OuterIsMul = outerIsMul
	return proto, true
}

// tryExprTree attempts to parse a binary RHS as a depth-2 expression tree,
// returning the populated Loop on success.
func tryExprTree(binExpr *ast.BinaryExpr, indexVar string, proto Loop) (Loop, bool) {
	outerOp, ok := tokenOpToLoopOp(binExpr.Op)
	if !ok {
		return Loop{}, false
	}

	// Inner on left: (A op B) outerOp C
	if innerBin, ok := binExpr.X.(*ast.BinaryExpr); ok {
		if l, ok := buildExprTree(innerBin, binExpr.Y, outerOp, false, indexVar, proto); ok {
			return l, true
		}
	}

	// Inner on right: A outerOp (B op C)
	if innerBin, ok := binExpr.Y.(*ast.BinaryExpr); ok {
		innerOnRight := true
		switch outerOp {
		case OpAdd, OpMul, OpAnd, OpOr, OpXor:
			// Commutative outer: normalize so inner is always on the left.
			innerOnRight = false
		}
		if l, ok := buildExprTree(innerBin, binExpr.X, outerOp, innerOnRight, indexVar, proto); ok {
			return l, true
		}
	}

	return Loop{}, false
}

// analyzeBody checks the loop body for vectorizable assignments.
// boundsIdent is the identifier node for the slice that determines the loop bounds (used for type lookup).
// valueVar is the range value variable name (e.g. "v" in "for i, v := range src"), or "".
func analyzeBody(rangeStmt *ast.RangeStmt, forStmt *ast.ForStmt, indexVar string, boundsIdent *ast.Ident, valueVar string, body *ast.BlockStmt, info *types.Info) (Loop, bool) {
	boundsSlice := boundsIdent.Name
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

	// asValueVar returns boundsSlice if expr is the range value variable.
	asValueVar := func(expr ast.Expr) (string, bool) {
		if valueVar == "" {
			return "", false
		}
		ident, ok := expr.(*ast.Ident)
		if !ok || ident.Name != valueVar {
			return "", false
		}
		return boundsSlice, true
	}

	if assignStmt.Tok == token.ASSIGN {
		lhsIndex, ok := asSliceIndex(assignStmt.Lhs[0], indexVar)
		if !ok {
			return Loop{}, false
		}
		loop.DstSlice = lhsIndex

		if binExpr, ok := assignStmt.Rhs[0].(*ast.BinaryExpr); ok {
			// Try depth-2 expression tree before depth-1: one operand is itself binary.
			if depth2, ok := tryExprTree(binExpr, indexVar, loop); ok {
				elemType, ok := resolveSliceElemType(boundsIdent, info)
				if !ok || simdElemType(elemType) == "" {
					return Loop{}, false
				}
				depth2.ElemType = elemType
				return depth2, true
			}

			// dst[i] = X op Y where X/Y may be slice[i] or range value var or scalar
			op, ok := tokenOpToLoopOp(binExpr.Op)
			if !ok {
				return Loop{}, false
			}
			loop.Op = op

			if src1, ok := asSliceIndex(binExpr.X, indexVar); ok {
				loop.Src1Slice = src1
			} else if src1, ok := asValueVar(binExpr.X); ok {
				loop.Src1Slice = src1
			} else {
				return Loop{}, false
			}

			if src2, ok := asSliceIndex(binExpr.Y, indexVar); ok {
				loop.Src2Slice = src2
			} else if src2, ok := asValueVar(binExpr.Y); ok {
				loop.Src2Slice = src2
			} else {
				loop.Scalar = binExpr.Y
			}
		} else if _, isSliceIdx := asSliceIndex(assignStmt.Rhs[0], indexVar); isSliceIdx {
			// dst[i] = src[i]  (plain copy — skip for now)
			return Loop{}, false
		} else if _, isValueVar := asValueVar(assignStmt.Rhs[0]); isValueVar {
			// dst[i] = v  (range value copy — skip for now)
			return Loop{}, false
		} else if ident, ok := assignStmt.Rhs[0].(*ast.Ident); ok && ident.Name == indexVar {
			// dst[i] = i  (index fill — not a constant, skip)
			return Loop{}, false
		} else if _, ok := assignStmt.Rhs[0].(*ast.BasicLit); ok {
			// dst[i] = literal constant  (fill broadcast; Src1Slice left empty)
			loop.Scalar = assignStmt.Rhs[0]
		} else if unaryExpr, ok := assignStmt.Rhs[0].(*ast.UnaryExpr); ok {
			// dst[i] = -src[i]  or  dst[i] = ^src[i]
			var unaryOp Op
			switch unaryExpr.Op {
			case token.SUB:
				unaryOp = OpNeg
			case token.XOR:
				unaryOp = OpNot
			default:
				return Loop{}, false
			}
			src, ok := asSliceIndex(unaryExpr.X, indexVar)
			if !ok {
				return Loop{}, false
			}
			loop.Src1Slice = src
			loop.Op = unaryOp
			loop.IsUnary = true
		} else {
			// Complex or non-constant expression — skip
			return Loop{}, false
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
		} else if src2, ok := asValueVar(assignStmt.Rhs[0]); ok {
			// dst[i] op= v  where v is the range value var (represents boundsSlice[i])
			loop.Src2Slice = src2
		} else if binRHS, isBin := assignStmt.Rhs[0].(*ast.BinaryExpr); isBin {
			// dst[i] outerOp= (a[i] innerOp scalar), in-place depth-2 tree.
			// Expands to: dst[i] = dst[i] outerOp (a[i] innerOp scalar),
			// treating the LHS slice as the outer operand.
			innerOnRight := true
			switch op {
			case OpAdd, OpMul, OpAnd, OpOr, OpXor:
				innerOnRight = false // commutative outer: normalize inner to the left
			}
			depth2, ok2 := buildExprTree(binRHS, assignStmt.Lhs[0], op, innerOnRight, indexVar, loop)
			if !ok2 {
				return Loop{}, false
			}
			elemType2, ok2 := resolveSliceElemType(boundsIdent, info)
			if !ok2 || simdElemType(elemType2) == "" {
				return Loop{}, false
			}
			depth2.ElemType = elemType2
			return depth2, true
		} else {
			// Only accept simple scalars: a literal or a plain identifier.
			// Compound expressions (v*alpha, f(x), etc.) are not safe to hoist.
			switch assignStmt.Rhs[0].(type) {
			case *ast.BasicLit, *ast.Ident:
				loop.Scalar = assignStmt.Rhs[0]
			default:
				return Loop{}, false
			}
		}
	}

	// Resolve element type via the bounds identifier node directly — avoids
	// false matches when multiple functions have same-named slices of different types.
	elemType, ok := resolveSliceElemType(boundsIdent, info)
	if !ok {
		return Loop{}, false
	}
	if simdElemType(elemType) == "" {
		return Loop{}, false
	}
	loop.ElemType = elemType

	if loop.IsUnary {
		switch loop.Op {
		case OpNeg:
			if !negSupported(elemType) {
				return Loop{}, false
			}
		case OpNot:
			if !notSupported(elemType) {
				return Loop{}, false
			}
		}
	}

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

// resolveSliceElemType returns the element type of the slice named by ident.
// Using the specific AST node avoids false matches when multiple functions
// have same-named slices of different types.
func resolveSliceElemType(ident *ast.Ident, info *types.Info) (types.Type, bool) {
	tv, ok := info.Types[ident]
	if !ok {
		return nil, false
	}
	slice, ok := tv.Type.(*types.Slice)
	if !ok {
		return nil, false
	}
	return slice.Elem(), true
}
