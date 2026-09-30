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
// a sequence of assignments to dst[i], each with a right-hand side built from
// same-index slice reads, loop-invariant scalars, and the operators simd
// provides.
//
// results holds the named result variables of the enclosing functions (see
// NamedResults): a bare return reads them, so a temporary cannot be one.
func Lower(stmt ast.Stmt, info *types.Info, results map[types.Object]bool) (*Loop, Reason) {
	l := &lowerer{info: info, node: stmt, results: results}
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
	// node is the loop being lowered, and results the named result variables of
	// the functions around it.
	node    ast.Stmt
	results map[types.Object]bool
	// assigned holds the local variables the body assigns. They change during
	// the loop, so none of them is loop-invariant.
	assigned map[types.Object]bool
	// temps maps a local to its temporary, once the Let that defines it is
	// lowered.
	temps map[types.Object]*Temp
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

// NamedResults returns the named result variables of fn and of every function
// literal inside it.
func NamedResults(fn *ast.FuncDecl, info *types.Info) map[types.Object]bool {
	results := map[types.Object]bool{}
	ast.Inspect(fn, func(n ast.Node) bool {
		if typ, ok := n.(*ast.FuncType); ok && typ.Results != nil {
			for _, field := range typ.Results.List {
				for _, name := range field.Names {
					if obj := info.Defs[name]; obj != nil {
						results[obj] = true
					}
				}
			}
		}
		return true
	})
	return results
}

// body lowers the loop body, a sequence of assignments.
func (l *lowerer) body(node ast.Stmt, block *ast.BlockStmt, trip Trip, step int64) (*Loop, Reason) {
	if len(block.List) == 0 {
		return nil, ReasonUnsupportedBody
	}
	l.assigned = map[types.Object]bool{}
	l.temps = map[types.Object]*Temp{}
	for _, stmt := range block.List {
		assign, ok := stmt.(*ast.AssignStmt)
		if !ok {
			continue
		}
		for _, lhs := range assign.Lhs {
			if id, ok := ast.Unparen(lhs).(*ast.Ident); ok {
				if obj := l.info.Defs[id]; obj != nil {
					l.assigned[obj] = true
				} else if obj := l.info.Uses[id]; obj != nil {
					l.assigned[obj] = true
				}
			}
		}
	}
	// A limit the body assigns would change while the loop runs.
	if id, ok := trip.Limit.(*ast.Ident); ok && l.assigned[l.info.Uses[id]] {
		return nil, ReasonUnsupportedBound
	}

	body := make([]Stmt, 0, len(block.List))
	for _, stmt := range block.List {
		assign, ok := stmt.(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
			return nil, ReasonUnsupportedBody
		}
		if id, ok := ast.Unparen(assign.Lhs[0]).(*ast.Ident); ok {
			let, reason := l.let(assign, id)
			if reason != "" {
				return nil, reason
			}
			body = append(body, let)
			continue
		}
		store, reason := l.store(assign)
		if reason != "" {
			return nil, reason
		}
		body = append(body, store)
	}

	for _, stmt := range body {
		switch stmt := stmt.(type) {
		case Store:
			switch v := stmt.Val.(type) {
			case *Load:
				// dst[i] = src[i] is a copy, emitted as the copy builtin. Copying a
				// slice onto itself does nothing.
				if v.Ref.Obj == stmt.Dst.Obj {
					return nil, ReasonUnsupportedOperand
				}
			case *Invariant:
				// A loop that only zeroes is left to memclr; among other statements
				// the zero is one more broadcast store.
				if len(body) == 1 && l.isZero(v.Expr) {
					return nil, ReasonZeroFillSkipped
				}
			}
			if SimdType(stmt.Dst.Elem) == "" {
				return nil, ReasonUnsupportedType
			}
		case Let:
			if stmt.Temp.Uses == 0 {
				return nil, ReasonUnsupportedBody
			}
		}
	}

	// The range value variable holds the element as it was when the iteration
	// began, but the vector loop reloads the slice after an earlier store to it.
	for k, stmt := range body {
		for _, leaf := range Leaves(stmtValue(stmt)) {
			load, ok := leaf.(*Load)
			if !ok || !load.FromRange {
				continue
			}
			for _, earlier := range body[:k] {
				if store, ok := earlier.(Store); ok && store.Dst.Obj == load.Ref.Obj {
					return nil, ReasonUnsupportedOperand
				}
			}
		}
	}

	return &Loop{
		Node: node,
		Ind:  Induction{Var: l.iv, Step: step, Trip: trip},
		Body: body,
	}, ""
}

// stmtValue is the expression a statement computes.
func stmtValue(stmt Stmt) Value {
	switch stmt := stmt.(type) {
	case Store:
		return stmt.Val
	case Let:
		return stmt.Val
	}
	return nil
}

// let lowers x := value, or x = value for a local declared before the loop. The
// temporary is written before it is read in every iteration, so no iteration sees
// another's value, and a local declared outside the loop is accepted only when
// nothing outside the loop uses it, so the value it would hold afterwards is
// never observed.
func (l *lowerer) let(assign *ast.AssignStmt, id *ast.Ident) (Let, Reason) {
	var obj types.Object
	switch assign.Tok {
	case token.DEFINE:
		obj = l.info.Defs[id]
	case token.ASSIGN:
		obj = l.info.Uses[id]
	default:
		return Let{}, ReasonUnsupportedBody
	}
	v, ok := obj.(*types.Var)
	if !ok || obj == l.iv || obj == l.valueVar || l.temps[obj] != nil {
		return Let{}, ReasonUnsupportedBody
	}
	if assign.Tok == token.ASSIGN && !l.deadOutsideLoop(v) {
		return Let{}, ReasonLiveTemp
	}
	val, reason := l.value(assign.Rhs[0])
	if reason != "" {
		return Let{}, reason
	}
	temp := &Temp{Var: v, Name: id.Name}
	l.temps[obj] = temp
	return Let{Temp: temp, Val: val}, ""
}

// deadOutsideLoop reports whether v, declared before the loop, is a local that
// nothing outside the loop reads or writes, and that is not a result the function
// returns.
func (l *lowerer) deadOutsideLoop(v *types.Var) bool {
	if v.IsField() || l.results[v] || v.Pos() >= l.node.Pos() {
		return false
	}
	if v.Pkg() == nil || v.Parent() == v.Pkg().Scope() {
		return false
	}
	for id, obj := range l.info.Uses {
		if obj == v && (id.Pos() < l.node.Pos() || id.Pos() >= l.node.End()) {
			return false
		}
	}
	return true
}

// store lowers one assignment to dst[i].
func (l *lowerer) store(assign *ast.AssignStmt) (Store, Reason) {
	var op Op
	switch assign.Tok {
	case token.ASSIGN, token.SHL_ASSIGN, token.SHR_ASSIGN:
	case token.DEFINE:
		return Store{}, ReasonUnsupportedBody
	default:
		var ok bool
		if op, ok = binaryOp(assign.Tok); !ok {
			return Store{}, ReasonUnsupportedOperand
		}
	}
	dst, ok := l.index(assign.Lhs[0])
	if !ok {
		return Store{}, ReasonUnsupportedDestination
	}

	var val Value
	var reason Reason
	switch assign.Tok {
	case token.ASSIGN:
		val, reason = l.value(assign.Rhs[0])
	case token.SHL_ASSIGN, token.SHR_ASSIGN:
		// dst[i] <<= count is dst[i] = dst[i] << count.
		shift := token.SHL
		if assign.Tok == token.SHR_ASSIGN {
			shift = token.SHR
		}
		val, reason = l.shiftOf(shift, &Load{Ref: dst}, assign.Rhs[0])
	default:
		// dst[i] op= rhs is dst[i] = dst[i] op rhs.
		var rhs Value
		if rhs, reason = l.value(assign.Rhs[0]); reason == "" {
			val = &Binary{Op: op, X: &Load{Ref: dst}, Y: rhs}
		}
	}
	if reason != "" {
		return Store{}, reason
	}
	return Store{Dst: dst, Val: val}, ""
}

// value lowers an expression evaluated once per iteration.
func (l *lowerer) value(expr ast.Expr) (Value, Reason) {
	expr = ast.Unparen(expr)
	if load, ok := l.load(expr); ok {
		return load, ""
	}
	if ident, ok := expr.(*ast.Ident); ok {
		if temp := l.temps[l.info.Uses[ident]]; temp != nil {
			temp.Uses++
			return &Use{Temp: temp}, ""
		}
	}
	if l.invariant(expr) {
		return &Invariant{Expr: expr}, ""
	}
	switch e := expr.(type) {
	case *ast.CallExpr:
		return l.call(e)
	case *ast.BinaryExpr:
		if e.Op == token.SHL || e.Op == token.SHR {
			return l.shift(e.Op, e.X, e.Y)
		}
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

// call lowers min, max, and the math functions simd has a method for: Abs, Sqrt,
// Min, and Max. Go has no float32 versions, so they are written
// float32(math.Abs(float64(x))); that is Abs on float32 vectors too, since Abs
// is exact and Sqrt through float64 rounds to the same float32 as the hardware
// (a float64 has more than twice the bits of a float32 significand), and Min and
// Max only choose an operand. Any other call is not rewritten: it may have side
// effects, and an invariant is evaluated before the loop even when it does not
// run.
func (l *lowerer) call(call *ast.CallExpr) (Value, Reason) {
	if fun, ok := l.info.Types[call.Fun]; ok && fun.IsType() {
		// float32(math.F(float64(x), ...))
		if len(call.Args) == 1 && isKind(fun.Type, types.Float32) {
			if inner, ok := ast.Unparen(call.Args[0]).(*ast.CallExpr); ok {
				if op, arity, ok := l.mathFunc(inner); ok {
					return l.operation(op, arity, inner.Args, true)
				}
			}
		}
		return nil, ReasonUnsupportedOperand
	}
	if op, arity, ok := l.mathFunc(call); ok {
		return l.operation(op, arity, call.Args, false)
	}
	if fn, ok := call.Fun.(*ast.Ident); ok && len(call.Args) >= 2 {
		if _, builtin := l.info.Uses[fn].(*types.Builtin); builtin && (fn.Name == "min" || fn.Name == "max") {
			op := OpMin
			if fn.Name == "max" {
				op = OpMax
			}
			return l.operation(op, len(call.Args), call.Args, false)
		}
	}
	return nil, ReasonUnsupportedOperand
}

// mathFunc matches math.Abs, math.Sqrt, math.Min, and math.Max, returning the
// operation and its number of arguments.
func (l *lowerer) mathFunc(call *ast.CallExpr) (op Op, arity int, ok bool) {
	sel, isSel := call.Fun.(*ast.SelectorExpr)
	if !isSel {
		return 0, 0, false
	}
	fn, isFunc := l.info.Uses[sel.Sel].(*types.Func)
	if !isFunc || fn.Pkg() == nil || fn.Pkg().Path() != "math" {
		return 0, 0, false
	}
	switch fn.Name() {
	case "Abs":
		return OpAbs, 1, true
	case "Sqrt":
		return OpSqrt, 1, true
	case "Min":
		return OpMin, 2, true
	case "Max":
		return OpMax, 2, true
	}
	return 0, 0, false
}

// operation lowers op applied to args. With narrow, each argument must be
// float64(x) for a float32 x, and x is what is lowered. More than two arguments
// of min or max fold from the left.
func (l *lowerer) operation(op Op, arity int, args []ast.Expr, narrow bool) (Value, Reason) {
	if len(args) != arity {
		return nil, ReasonUnsupportedOperand
	}
	values := make([]Value, len(args))
	for i, arg := range args {
		arg = ast.Unparen(arg)
		if narrow {
			conv, ok := arg.(*ast.CallExpr)
			if !ok || len(conv.Args) != 1 {
				return nil, ReasonUnsupportedOperand
			}
			fun, ok := l.info.Types[conv.Fun]
			inner := ast.Unparen(conv.Args[0])
			if !ok || !fun.IsType() || !isKind(fun.Type, types.Float64) || !isKind(l.info.Types[inner].Type, types.Float32) {
				return nil, ReasonUnsupportedOperand
			}
			arg = inner
		}
		value, reason := l.value(arg)
		if reason != "" {
			return nil, reason
		}
		values[i] = value
	}
	if len(values) == 1 {
		return &Unary{Op: op, X: values[0]}, ""
	}
	result := values[0]
	for _, next := range values[1:] {
		result = &Binary{Op: op, X: result, Y: next}
	}
	return result, ""
}

// shift lowers x << count and x >> count.
func (l *lowerer) shift(tok token.Token, x, count ast.Expr) (Value, Reason) {
	value, reason := l.value(x)
	if reason != "" {
		return nil, reason
	}
	return l.shiftOf(tok, value, count)
}

// shiftOf shifts an already lowered value. The count must be a non-negative
// constant or a variable of an unsigned type: a negative signed count makes Go
// panic, and evaluating it before the loop would not.
func (l *lowerer) shiftOf(tok token.Token, x Value, count ast.Expr) (Value, Reason) {
	count = ast.Unparen(count)
	ok := false
	if tv, isConst := l.info.Types[count]; isConst && tv.Value != nil {
		_, ok = constant.Uint64Val(tv.Value)
	} else if ident, isIdent := count.(*ast.Ident); isIdent {
		if obj, isVar := l.info.Uses[ident].(*types.Var); isVar && obj != l.iv && obj != l.valueVar && !l.assigned[obj] {
			basic, isBasic := obj.Type().Underlying().(*types.Basic)
			ok = isBasic && basic.Info()&types.IsUnsigned != 0
		}
	}
	if !ok {
		return nil, ReasonUnsupportedOperand
	}
	op := OpShl
	if tok == token.SHR {
		op = OpShr
	}
	return &Shift{Op: op, X: x, Count: count}, ""
}

// isKind reports whether t is the predeclared type of the given kind.
func isKind(t types.Type, kind types.BasicKind) bool {
	basic, ok := t.(*types.Basic)
	return ok && basic.Kind() == kind
}

// load matches src[i], or the range value variable, which is the same read.
func (l *lowerer) load(expr ast.Expr) (*Load, bool) {
	if ident, ok := expr.(*ast.Ident); ok {
		if l.valueVar != nil && l.info.Uses[ident] == l.valueVar {
			return &Load{Ref: l.valueRef, FromRange: true}, true
		}
		return nil, false
	}
	ref, ok := l.index(expr)
	if !ok {
		return nil, false
	}
	return &Load{Ref: ref}, true
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
// a variable, a constant, a conversion or negation of one, or a sum, difference,
// product, or bitwise combination of two. Anything that can panic or has side
// effects, such as a call, an element read, or an integer division, is not.
func (l *lowerer) invariant(expr ast.Expr) bool {
	if tv, ok := l.info.Types[expr]; ok && tv.Value != nil {
		return true
	}
	switch e := expr.(type) {
	case *ast.ParenExpr:
		return l.invariant(e.X)
	case *ast.Ident:
		obj, ok := l.info.Uses[e].(*types.Var)
		if !ok || obj == l.iv || obj == l.valueVar || l.assigned[obj] {
			return false
		}
		_, basic := obj.Type().Underlying().(*types.Basic)
		return basic
	case *ast.UnaryExpr:
		switch e.Op {
		case token.ADD, token.SUB, token.XOR:
			return l.invariant(e.X)
		}
	case *ast.BinaryExpr:
		switch e.Op {
		case token.ADD, token.SUB, token.MUL, token.AND, token.OR, token.XOR, token.AND_NOT:
			return l.invariant(e.X) && l.invariant(e.Y)
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
	case token.AND_NOT, token.AND_NOT_ASSIGN:
		return OpAndNot, true
	}
	return 0, false
}
