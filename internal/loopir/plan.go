package loopir

import "go/types"

// Plan is a Loop that simd can express, with the decisions the emitted code
// depends on.
type Plan struct {
	Loop *Loop
	// SimdType is the simd vector type name, such as Float32s.
	SimdType string
	// Dst and Root are the store's destination and its normalized value; see
	// Normalize.
	Dst  *Ref
	Root Value
	// Bound is the iteration limit when it is not len(Dst): an int variable
	// or constant, or len of another slice. Empty means len(Dst).
	Bound string
	// BoundConst is set when Bound is a positive integer constant.
	BoundConst bool
	// Checked lists every slice whose length must be verified against the
	// limit before the loop runs: Dst first, then each slice read, in the
	// order the loads are emitted.
	Checked []*Ref
	// Others lists the slices read that are not Dst. Each may alias Dst at an
	// offset and needs a runtime overlap check.
	Others []*Ref
}

// NewPlan checks that simd implements every operation in l for its element
// type and derives the rest of the plan.
func NewPlan(l *Loop) (*Plan, Reason) {
	store := l.Body[0]
	p := &Plan{
		Loop:     l,
		SimdType: SimdType(store.Dst.Elem),
		Dst:      store.Dst,
		Root:     Normalize(store.Val),
	}
	if !p.opsSupported(p.Root) {
		return nil, ReasonUnsupportedOp
	}

	trip := l.Ind.Trip
	switch trip.Kind {
	case TripLen:
		// The generated loop stops at len(Dst), so any other bounding slice
		// must be spelled out or a longer Dst would run past the data.
		if trip.Slice.Obj != p.Dst.Obj {
			p.Bound = "len(" + trip.Slice.Name + ")"
		}
	case TripInt:
		p.Bound = types.ExprString(trip.Limit)
		p.BoundConst = trip.Const
	}

	p.Checked = []*Ref{p.Dst}
	for _, leaf := range Leaves(p.Root) {
		load, ok := leaf.(*Load)
		if !ok {
			continue
		}
		p.Checked = appendRef(p.Checked, load.Ref)
		if load.Ref.Obj != p.Dst.Obj {
			p.Others = appendRef(p.Others, load.Ref)
		}
	}
	return p, ""
}

func (p *Plan) opsSupported(v Value) bool {
	switch v := v.(type) {
	case *Unary:
		return supports(p.Dst.Elem, v.Op) && p.opsSupported(v.X)
	case *Binary:
		return supports(p.Dst.Elem, v.Op) && p.opsSupported(v.X) && p.opsSupported(v.Y)
	}
	return true
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
