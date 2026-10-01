package loopir

import "go/types"

// maxSlices is the most distinct slices a loop may touch. Each stored slice is
// checked at run time against every other slice in the loop, so the number of
// checks grows with it.
const maxSlices = 8

// Plan is a Loop that simd can express, with the decisions the emitted code
// depends on.
type Plan struct {
	Loop *Loop
	// SimdType is the simd vector type name, such as Float32s. Every store in
	// the loop uses it.
	SimdType string
	// Stmts are the loop's statements in source order.
	Stmts []PlanStmt
	// Full is set when the loop folds into an accumulator (has a Reduce). It is
	// emitted with full-width loads, a scalar loop for what is left, and the
	// accumulators combined afterwards, since a zero-padded partial load is not
	// the identity of every operation.
	Full bool
	// Limit is the iteration limit as source text.
	Limit string
	// TripRef is the slice whose length limits the loop, when one does.
	TripRef *Ref
	// Copy is set when the loop only copies one slice onto another. It is
	// emitted as the copy builtin, which needs no simd at all.
	Copy bool
	// Bound is the iteration limit when it is not len of the stored slice: an
	// int variable or constant, or len of another slice. Empty means every store
	// is to the slice that limits the loop, and its length is the limit.
	Bound string
	// BoundConst is set when Bound is a positive integer constant.
	BoundConst bool
	// Checked lists every slice whose length must be verified against the
	// limit before the loop runs: for each store in order, its destination and
	// then each slice it reads, in the order the loads are emitted.
	Checked []*Ref
	// Overlaps lists the pairs of distinct slices that need a runtime overlap
	// check: every pair where at least one is stored, since two slice variables
	// may share memory at an offset.
	Overlaps [][2]*Ref
}

// PlanStmt is a statement whose value has been normalized; see Normalize. It
// stores to Dst, defines Temp, or folds into Acc with Op, whichever is set.
type PlanStmt struct {
	Dst  *Ref
	Temp *Temp
	Acc  types.Object
	Op   Op
	Root Value
}

// Options selects behavior that changes results or depends on the toolchain.
type Options struct {
	// FloatReassoc allows a floating-point sum or product to be regrouped: the
	// vector loop adds in several lanes and accumulators and combines them at
	// the end, so the result can differ from the scalar loop in the last bits,
	// as a Go compiler is never allowed to do on its own.
	FloatReassoc bool
}

// NewPlan checks that simd implements every operation in l for its element
// type and derives the rest of the plan.
func NewPlan(l *Loop, opts Options) (*Plan, Reason) {
	var elem types.Type
	for _, stmt := range l.Body {
		if store, ok := stmt.(Store); ok {
			elem = store.Dst.Elem
			break
		}
		if reduce, ok := stmt.(Reduce); ok {
			elem = reduce.Acc.Type()
			break
		}
	}
	p := &Plan{Loop: l, SimdType: SimdType(elem)}
	for _, stmt := range l.Body {
		var planned PlanStmt
		switch stmt := stmt.(type) {
		case Store:
			if !types.Identical(stmt.Dst.Elem, elem) {
				return nil, ReasonMixedTypes
			}
			planned = PlanStmt{Dst: stmt.Dst, Root: Normalize(stmt.Val)}
		case Let:
			if !types.Identical(stmt.Temp.Var.Type(), elem) {
				return nil, ReasonMixedTypes
			}
			planned = PlanStmt{Temp: stmt.Temp, Root: Normalize(stmt.Val)}
		case Reduce:
			if !types.Identical(stmt.Acc.Type(), elem) {
				return nil, ReasonMixedTypes
			}
			if isFloat(elem) {
				switch {
				case stmt.Op == OpMin || stmt.Op == OpMax:
					return nil, ReasonFloatMinMax
				case !opts.FloatReassoc:
					return nil, ReasonFloatReduction
				}
			}
			if !supports(elem, stmt.Op) {
				return nil, ReasonUnsupportedOp
			}
			planned = PlanStmt{Acc: stmt.Acc, Op: stmt.Op, Root: Normalize(stmt.Val)}
			p.Full = true
		}
		if reason := checkOps(elem, planned.Root); reason != "" {
			return nil, reason
		}
		p.Stmts = append(p.Stmts, planned)
	}
	if len(p.Stmts) == 1 && p.Stmts[0].Dst != nil {
		_, p.Copy = p.Stmts[0].Root.(*Load)
	}

	trip := l.Ind.Trip
	switch trip.Kind {
	case TripLen:
		// The generated loop stops at the length of the stored slice, so when
		// another slice limits the loop, or a second slice is stored, the limit
		// must be spelled out or a longer slice would run past the data.
		for _, stmt := range p.Stmts {
			if stmt.Dst != nil && stmt.Dst.Obj != trip.Slice.Obj {
				p.Bound = "len(" + trip.Slice.Name + ")"
				break
			}
		}
	case TripInt:
		p.Bound = types.ExprString(trip.Limit)
		p.BoundConst = trip.Const
	}
	if trip.Kind == TripLen {
		p.TripRef = trip.Slice
		p.Limit = "len(" + trip.Slice.Name + ")"
	} else {
		p.Limit = p.Bound
	}

	stored := map[types.Object]bool{}
	for _, stmt := range p.Stmts {
		if stmt.Dst != nil {
			stored[stmt.Dst.Obj] = true
			p.Checked = appendRef(p.Checked, stmt.Dst)
		}
		for _, leaf := range Leaves(stmt.Root) {
			if load, ok := leaf.(*Load); ok {
				p.Checked = appendRef(p.Checked, load.Ref)
			}
		}
	}
	if len(p.Checked) > maxSlices {
		return nil, ReasonTooManySlices
	}
	for i, first := range p.Checked {
		for _, second := range p.Checked[i+1:] {
			if stored[first.Obj] || stored[second.Obj] {
				p.Overlaps = append(p.Overlaps, [2]*Ref{first, second})
			}
		}
	}
	return p, ""
}

// checkOps returns why simd cannot express an operation in v on the element
// type, or "".
func checkOps(elem types.Type, v Value) Reason {
	var op Op
	var operands []Value
	switch v := v.(type) {
	case *Unary:
		op, operands = v.Op, []Value{v.X}
	case *Binary:
		op, operands = v.Op, []Value{v.X, v.Y}
	case *Shift:
		op, operands = v.Op, []Value{v.X}
	default:
		return ""
	}
	if (op == OpMin || op == OpMax) && isFloat(elem) {
		return ReasonFloatMinMax
	}
	if !supports(elem, op) {
		return ReasonUnsupportedOp
	}
	for _, operand := range operands {
		if reason := checkOps(elem, operand); reason != "" {
			return reason
		}
	}
	return ""
}

func isFloat(t types.Type) bool {
	basic, ok := t.(*types.Basic)
	return ok && basic.Info()&types.IsFloat != 0
}

func appendRef(refs []*Ref, ref *Ref) []*Ref {
	for _, have := range refs {
		if have.Obj == ref.Obj {
			return refs
		}
	}
	return append(refs, ref)
}

// Depth is the height of the expression tree: 0 for a leaf.
func Depth(v Value) int {
	switch v := v.(type) {
	case *Shift:
		return 1 + Depth(v.X)
	case *Unary:
		return 1 + Depth(v.X)
	case *Binary:
		return 1 + max(Depth(v.X), Depth(v.Y))
	}
	return 0
}

// Normalize puts the operands of every commutative operation in the order the
// emitter expects: the operand with more work under it first, and a broadcast
// scalar after a vector. Emission then always produces the receiver-first form
// (a.Mul(b).Add(c)), and a multiply feeding an add is always the left operand
// where it can be fused.
func Normalize(v Value) Value {
	switch v := v.(type) {
	case *Shift:
		return &Shift{Op: v.Op, X: Normalize(v.X), Count: v.Count}
	case *Unary:
		return &Unary{Op: v.Op, X: Normalize(v.X)}
	case *Binary:
		x, y := Normalize(v.X), Normalize(v.Y)
		if v.Op.Commutative() && shouldSwap(x, y) {
			x, y = y, x
		}
		return &Binary{Op: v.Op, X: x, Y: y}
	}
	return v
}

func shouldSwap(x, y Value) bool {
	_, xInvariant := x.(*Invariant)
	_, yInvariant := y.(*Invariant)
	if xInvariant && !yInvariant {
		return true
	}
	return Depth(y) > Depth(x)
}

// Children returns the operands of v in evaluation order: the operand with
// more work under it first, so the loads feeding it are numbered first.
func Children(v Value) []Value {
	switch v := v.(type) {
	case *Shift:
		return []Value{v.X}
	case *Unary:
		return []Value{v.X}
	case *Binary:
		if Depth(v.Y) > Depth(v.X) {
			return []Value{v.Y, v.X}
		}
		return []Value{v.X, v.Y}
	}
	return nil
}

// Leaves returns the loads and invariants under v in evaluation order.
func Leaves(v Value) []Value {
	children := Children(v)
	if children == nil {
		return []Value{v}
	}
	var out []Value
	for _, child := range children {
		out = append(out, Leaves(child)...)
	}
	return out
}
