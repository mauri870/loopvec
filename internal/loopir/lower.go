package loopir

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
)

// Lower canonicalizes a for or range statement into a Loop. When the loop is
// not one loopvec understands, it returns the reason.
//
// Accepted headers:
//
//	for i := range s              for i, v := range s
//	for i := range n              for i := 0; i < n; i++
//	for i := 0; i < len(s); i++   for i := len(s) - 1; i >= 0; i--
//
// where n is an int variable or a positive integer constant. The body must be
// one assignment to dst[i] whose right-hand side is built from same-index
// slice reads, loop-invariant scalars, and the operators simd provides.
func Lower(stmt ast.Stmt, info *types.Info) (*Loop, Reason) {
	l := &lowerer{info: info}
	switch s := stmt.(type) {
	case *ast.RangeStmt:
		return l.rangeLoop(s)
	case *ast.ForStmt:
		return l.forLoop(s)
	}
	return nil, ReasonNotCountedLoop
}

type lowerer struct {
	info *types.Info
	// iv is the loop index variable.
	iv types.Object
	// valueVar is the range value variable and valueRef the slice it ranges over;
	// valueVar is nil when the loop has no value variable.
	valueVar types.Object
	valueRef *Ref
}

func (l *lowerer) rangeLoop(s *ast.RangeStmt) (*Loop, Reason) {
	key, ok := s.Key.(*ast.Ident)
	if !ok || s.Tok != token.DEFINE {
		return nil, ReasonUnsupportedRange
	}
	l.iv = l.info.Defs[key]
	if l.iv == nil {
		return nil, ReasonUnsupportedRange
	}
	var valueIdent *ast.Ident
	if s.Value != nil {
		valueIdent, ok = s.Value.(*ast.Ident)
		if !ok {
			return nil, ReasonUnsupportedRange
		}
		if valueIdent.Name == "_" {
			valueIdent = nil
		}
	}

	// A slice identifier ranges over its elements; anything else must be an
	// integer limit (for i := range n).
	if ident, ok := s.X.(*ast.Ident); ok && l.isSlice(ident) {
		ref := l.ref(ident)
		if valueIdent != nil {
			l.valueVar = l.info.Defs[valueIdent]
			l.valueRef = ref
		}
		return l.body(s, s.Body, Trip{Kind: TripLen, Slice: ref}, 1)
	}
	if s.Value != nil {
		return nil, ReasonUnsupportedRange
	}
	trip, ok := l.intTrip(s.X)
	if !ok {
		return nil, ReasonUnsupportedBound
	}
	return l.body(s, s.Body, trip, 1)
}

// forLoop handles for i := 0; i < limit; i++ and
// for i := len(s) - 1; i >= 0; i--.
//
// The reverse form is accepted for the same reason the forward one is safe:
// every slice is read or written at exactly index i, so different iterations
// never touch the same element and the direction cannot change the result.
// The rewritten loop always runs forward.
//
// The comparison picks the direction, and the start, step, and limit are then
// checked in that order, so the reason names the first part that is wrong.
func (l *lowerer) forLoop(s *ast.ForStmt) (*Loop, Reason) {
	if s.Init == nil || s.Cond == nil || s.Post == nil {
		return nil, ReasonNotCountedLoop
	}
	init, ok := s.Init.(*ast.AssignStmt)
	if !ok || init.Tok != token.DEFINE || len(init.Lhs) != 1 || len(init.Rhs) != 1 {
		return nil, ReasonNotCountedLoop
	}
	key, ok := init.Lhs[0].(*ast.Ident)
	if !ok {
		return nil, ReasonNotCountedLoop
	}
	l.iv = l.info.Defs[key]
	if l.iv == nil {
		return nil, ReasonNotCountedLoop
	}
	cond, ok := s.Cond.(*ast.BinaryExpr)
	if !ok {
		return nil, ReasonUnsupportedCondition
	}
	if x, ok := cond.X.(*ast.Ident); !ok || l.info.Uses[x] != l.iv {
		return nil, ReasonUnsupportedCondition
	}

	switch cond.Op {
	case token.LSS:
		return l.forward(s, init.Rhs[0], cond.Y)
	case token.GEQ:
		return l.reverse(s, init.Rhs[0], cond.Y)
	}
	return nil, ReasonUnsupportedCondition
}

// forward handles for i := 0; i < limit; i++, where limit is either len(slice)
// or an integer expression.
func (l *lowerer) forward(s *ast.ForStmt, start, limit ast.Expr) (*Loop, Reason) {
	if lit, ok := start.(*ast.BasicLit); !ok || lit.Value != "0" {
		return nil, ReasonUnsupportedStart
	}
	if !l.step(s.Post, token.INC) {
		return nil, ReasonUnsupportedStep
	}
	if call, ok := limit.(*ast.CallExpr); ok {
		ref, ok := l.lenOf(call)
		if !ok {
			return nil, ReasonUnsupportedBound
		}
		return l.body(s, s.Body, Trip{Kind: TripLen, Slice: ref}, 1)
	}
	trip, ok := l.intTrip(limit)
	if !ok {
		return nil, ReasonUnsupportedBound
	}
	return l.body(s, s.Body, trip, 1)
}

// reverse handles for i := len(s) - 1; i >= 0; i--.
func (l *lowerer) reverse(s *ast.ForStmt, start, limit ast.Expr) (*Loop, Reason) {
	sub, ok := start.(*ast.BinaryExpr)
	if !ok || sub.Op != token.SUB {
		return nil, ReasonUnsupportedStart
	}
	if one, ok := sub.Y.(*ast.BasicLit); !ok || one.Value != "1" {
		return nil, ReasonUnsupportedStart
	}
	call, ok := sub.X.(*ast.CallExpr)
	if !ok {
		return nil, ReasonUnsupportedStart
	}
	ref, ok := l.lenOf(call)
	if !ok {
		return nil, ReasonUnsupportedStart
	}
	if !l.step(s.Post, token.DEC) {
		return nil, ReasonUnsupportedStep
	}
	if lit, ok := limit.(*ast.BasicLit); !ok || lit.Value != "0" {
		return nil, ReasonUnsupportedBound
	}
	return l.body(s, s.Body, Trip{Kind: TripLen, Slice: ref}, -1)
}

// step reports whether post is i++ (tok INC) or i-- (tok DEC) on the loop
// index.
func (l *lowerer) step(post ast.Stmt, tok token.Token) bool {
	incDec, ok := post.(*ast.IncDecStmt)
	if !ok || incDec.Tok != tok {
		return false
	}
	ident, ok := incDec.X.(*ast.Ident)
	return ok && l.info.Uses[ident] == l.iv
}

// lenOf matches the builtin call len(s) where s is a slice identifier.
func (l *lowerer) lenOf(call *ast.CallExpr) (*Ref, bool) {
	fn, ok := call.Fun.(*ast.Ident)
	if !ok || len(call.Args) != 1 {
		return nil, false
	}
	if _, builtin := l.info.Uses[fn].(*types.Builtin); !builtin || fn.Name != "len" {
		return nil, false
	}
	arg, ok := call.Args[0].(*ast.Ident)
	if !ok || !l.isSlice(arg) {
		return nil, false
	}
	return l.ref(arg), true
}

// intTrip accepts a loop limit that is a variable of type int or a positive
// integer constant, so it is the same value every time the loop condition
// would evaluate it.
func (l *lowerer) intTrip(limit ast.Expr) (Trip, bool) {
	trip := Trip{Kind: TripInt, Limit: limit}
	switch expr := limit.(type) {
	case *ast.BasicLit:
		if expr.Kind != token.INT {
			return Trip{}, false
		}
	case *ast.Ident:
		tv, ok := l.info.Types[expr]
		if !ok || l.info.Uses[expr] == l.iv {
			return Trip{}, false
		}
		basic, ok := tv.Type.(*types.Basic)
		if !ok || (basic.Kind() != types.Int && basic.Kind() != types.UntypedInt) {
			return Trip{}, false
		}
		if tv.Value == nil {
			return trip, true
		}
	default:
		return Trip{}, false
	}
	tv, ok := l.info.Types[limit]
	if !ok || tv.Value == nil {
		return Trip{}, false
	}
	n, exact := constant.Int64Val(tv.Value)
	if !exact || n <= 0 {
		return Trip{}, false
	}
	trip.Const = true
	return trip, true
}

// body lowers the loop body, a single assignment.
func (l *lowerer) body(node ast.Stmt, block *ast.BlockStmt, trip Trip, step int64) (*Loop, Reason) {
	if len(block.List) != 1 {
		return nil, ReasonUnsupportedBody
	}
	assign, ok := block.List[0].(*ast.AssignStmt)
	if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
		return nil, ReasonUnsupportedBody
	}

	var dst *Ref
	var val Value
	if assign.Tok == token.ASSIGN {
		dst, ok = l.index(assign.Lhs[0])
		if !ok {
			return nil, ReasonUnsupportedDestination
		}
		var reason Reason
		val, reason = l.value(assign.Rhs[0])
		if reason != "" {
			return nil, reason
		}
		switch v := val.(type) {
		case *Load:
			// dst[i] = src[i] is a copy, which copy() does better.
			return nil, ReasonUnsupportedOperand
		case *Invariant:
			if l.isZero(v.Expr) {
				return nil, ReasonZeroFillSkipped
			}
		}
	} else {
		// dst[i] op= rhs is dst[i] = dst[i] op rhs.
		op, ok := binaryOp(assign.Tok)
		if !ok {
			return nil, ReasonUnsupportedOperand
		}
		dst, ok = l.index(assign.Lhs[0])
		if !ok {
			return nil, ReasonUnsupportedDestination
		}
		rhs, reason := l.value(assign.Rhs[0])
		if reason != "" {
			return nil, reason
		}
		val = &Binary{Op: op, X: &Load{Ref: dst}, Y: rhs}
	}

	if !l.withinLimits(val) {
		return nil, ReasonUnsupportedOperand
	}
	if SimdType(dst.Elem) == "" {
		return nil, ReasonUnsupportedType
	}
	return &Loop{
		Node: node,
		Ind:  Induction{Var: l.iv, Step: step, Trip: trip},
		Body: []Store{{Dst: dst, Val: val}},
	}, ""
}

// withinLimits reports whether val is a shape the rest of loopvec is tested
// on. Lowering builds any expression tree, but the operand naming in the
// emitter and the -toolexec wrapper (which rewrites every package in a build,
// so a wider shape newly rewrites code in the standard library, including
// constant-time crypto) are only exercised on these shapes:
//
//	dst[i] = literal
//	dst[i] = -src[i]  or  ^src[i]
//	dst[i] = src[i] op (src[i] or scalar)
//	dst[i] = (src[i] op (src[i] or scalar)) op outer
//
// where outer is src[i], a scalar, or src[i]*scalar, and a scalar in a
// two-level tree is an identifier or a literal. The range value variable is
// an operand only in the single-operation form. Anything else is rejected
// rather than rewritten.
func (l *lowerer) withinLimits(val Value) bool {
	switch v := val.(type) {
	case *Invariant:
		_, literal := v.Expr.(*ast.BasicLit)
		return literal
	case *Unary:
		_, load := v.X.(*Load)
		return load
	case *Binary:
		if Depth(v) == 1 {
			_, load := v.X.(*Load)
			return load
		}
		return l.twoLevel(v)
	}
	return false
}

// twoLevel reports whether v is an operation with one operand of the form
// src[i] op (src[i] or scalar) and the other src[i], a scalar, or
// src[i]*scalar.
func (l *lowerer) twoLevel(v *Binary) bool {
	return (l.inner(v.X) && l.outer(v.Y)) || (l.inner(v.Y) && l.outer(v.X))
}

func (l *lowerer) inner(v Value) bool {
	b, ok := v.(*Binary)
	if !ok {
		return false
	}
	if l.plainLoad(b.X) {
		return l.plainLoad(b.Y) || l.plainScalar(b.Y)
	}
	// A scalar first is only reordered for a commutative operation.
	return b.Op.Commutative() && l.plainScalar(b.X) && l.plainLoad(b.Y)
}

func (l *lowerer) outer(v Value) bool {
	if l.plainLoad(v) || l.plainScalar(v) {
		return true
	}
	b, ok := v.(*Binary)
	if !ok || b.Op != OpMul {
		return false
	}
	return (l.plainLoad(b.X) && l.plainScalar(b.Y)) || (l.plainScalar(b.X) && l.plainLoad(b.Y))
}

// plainLoad reports whether v is a read of src[i] that is not the range value
// variable.
func (l *lowerer) plainLoad(v Value) bool {
	load, ok := v.(*Load)
	return ok && load.Ref != l.valueRef
}

// plainScalar reports whether v is an identifier or a literal.
func (l *lowerer) plainScalar(v Value) bool {
	inv, ok := v.(*Invariant)
	if !ok {
		return false
	}
	switch inv.Expr.(type) {
	case *ast.Ident, *ast.BasicLit:
		return true
	}
	return false
}

// value lowers an expression evaluated once per iteration.
func (l *lowerer) value(expr ast.Expr) (Value, Reason) {
	expr = ast.Unparen(expr)
	if ref, ok := l.load(expr); ok {
		return &Load{Ref: ref}, ""
	}
	if l.invariant(expr) {
		return &Invariant{Expr: expr}, ""
	}
	switch e := expr.(type) {
	case *ast.BinaryExpr:
		op, ok := binaryOp(e.Op)
		if !ok {
			return nil, ReasonUnsupportedOperand
		}
		x, reason := l.value(e.X)
		if reason != "" {
			return nil, reason
		}
		y, reason := l.value(e.Y)
		if reason != "" {
			return nil, reason
		}
		return &Binary{Op: op, X: x, Y: y}, ""
	case *ast.UnaryExpr:
		var op Op
		switch e.Op {
		case token.SUB:
			op = OpNeg
		case token.XOR:
			op = OpNot
		default:
			return nil, ReasonUnsupportedOperand
		}
		x, reason := l.value(e.X)
		if reason != "" {
			return nil, reason
		}
		return &Unary{Op: op, X: x}, ""
	}
	return nil, ReasonUnsupportedOperand
}

// load matches src[i], or the range value variable, which is the same read.
func (l *lowerer) load(expr ast.Expr) (*Ref, bool) {
	if ident, ok := expr.(*ast.Ident); ok {
		if l.valueVar != nil && l.info.Uses[ident] == l.valueVar {
			return l.valueRef, true
		}
		return nil, false
	}
	return l.index(expr)
}

// index matches slice[i] where slice is an identifier and i the loop index.
func (l *lowerer) index(expr ast.Expr) (*Ref, bool) {
	index, ok := ast.Unparen(expr).(*ast.IndexExpr)
	if !ok {
		return nil, false
	}
	base, ok := index.X.(*ast.Ident)
	if !ok || !l.isSlice(base) {
		return nil, false
	}
	pos, ok := index.Index.(*ast.Ident)
	if !ok || l.info.Uses[pos] != l.iv {
		return nil, false
	}
	return l.ref(base), true
}

// invariant reports whether expr has the same value on every iteration and is
// safe to evaluate before the loop, where it runs even if the loop does not:
// a variable, a constant, or a conversion or negation of one. Anything that
// can panic or has side effects, such as a call, an element read, or an
// integer division, is not.
func (l *lowerer) invariant(expr ast.Expr) bool {
	if tv, ok := l.info.Types[expr]; ok && tv.Value != nil {
		return true
	}
	switch e := expr.(type) {
	case *ast.ParenExpr:
		return l.invariant(e.X)
	case *ast.Ident:
		obj, ok := l.info.Uses[e].(*types.Var)
		if !ok || obj == l.iv || obj == l.valueVar {
			return false
		}
		_, basic := obj.Type().Underlying().(*types.Basic)
		return basic
	case *ast.UnaryExpr:
		switch e.Op {
		case token.ADD, token.SUB, token.XOR:
			return l.invariant(e.X)
		}
	case *ast.CallExpr:
		fun, ok := l.info.Types[e.Fun]
		if !ok || !fun.IsType() || len(e.Args) != 1 {
			return false
		}
		_, basic := fun.Type.Underlying().(*types.Basic)
		return basic && l.invariant(e.Args[0])
	}
	return false
}

// isZero reports whether expr is a numeric constant equal to zero.
func (l *lowerer) isZero(expr ast.Expr) bool {
	tv, ok := l.info.Types[expr]
	if !ok || tv.Value == nil {
		return false
	}
	switch tv.Value.Kind() {
	case constant.Int, constant.Float:
		return constant.Sign(tv.Value) == 0
	}
	return false
}

// isSlice reports whether ident is a slice.
func (l *lowerer) isSlice(ident *ast.Ident) bool {
	tv, ok := l.info.Types[ident]
	if !ok {
		return false
	}
	_, ok = tv.Type.Underlying().(*types.Slice)
	return ok
}

// ref returns the Ref for a slice identifier.
func (l *lowerer) ref(ident *ast.Ident) *Ref {
	obj := l.info.Uses[ident]
	if obj == nil {
		obj = l.info.Defs[ident]
	}
	slice := l.info.Types[ident].Type.Underlying().(*types.Slice)
	return &Ref{Name: ident.Name, Obj: obj, Elem: slice.Elem()}
}

// binaryOp maps a binary or op-assign token to an Op.
func binaryOp(tok token.Token) (Op, bool) {
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
