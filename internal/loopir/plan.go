package loopir

import (
	"go/types"
	"strconv"
)

// maxSlices is the most distinct slices a loop may touch. Each stored slice is
// checked at run time against every other slice in the loop, so the number of
// checks grows with it.
const maxSlices = 8

// minTrip is the fewest iterations a loop with a constant trip count must have to be
// worth rewriting. Measured crossover on AVX-512: the vector loop (with the per-call
// dispatch the compiler adds around simd code) breaks even at about 8 elements.
const minTrip = 8

// MinReduceVectors is how many of the widest vectors a reduction needs before the vector
// loop is used. Setting up the accumulators and combining the lanes costs about 6ns
// however short the loop, more than the scalar loop for fewer than roughly forty int32
// elements on AVX-512, and the scalar loop that follows runs anything the vector loop
// does not. The bound is in elements so that a short loop pays nothing for the vector
// width.
const MinReduceVectors = 4

// MaxLanes is the most elements of type t a vector holds: simd vectors are at most 512
// bits.
func MaxLanes(t types.Type) int {
	switch t.Underlying().(*types.Basic).Kind() {
	case types.Int8, types.Uint8:
		return 64
	case types.Int16, types.Uint16:
		return 32
	case types.Int32, types.Uint32, types.Float32:
		return 16
	}
	return 8
}

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
	// Bound is the iteration limit when it is not len of the stored slice. Empty
	// means every store is to the slice that limits the loop, and its length is
	// the limit.
	Bound string
	// Start is where the loop starts, as source text; "0" is the usual case.
	Start string
	// NonEmpty is set when the loop is known to run at least once.
	NonEmpty bool
	// Checked lists every slice whose length must be verified against the
	// limit before the loop runs: for each store in order, its destination and
	// then each slice it reads, in the order the loads are emitted. A slice read
	// at several offsets is listed once for each.
	Checked []Access
	// Hoists lists the slices the loop reaches through a selector (x.f), in order: the
	// emitted code assigns each Name from Src once before the loop.
	Hoists []Hoist
	// Overlaps lists the pairs of distinct slices that need a runtime overlap
	// check: every pair where at least one is stored, since two slice variables
	// may share memory at an offset.
	Overlaps [][2]*Ref
}

// Hoist declares Name as a copy of the slice expression Src.
type Hoist struct{ Name, Src string }

// Access is a slice and the offset from the loop index it is read at, "" for none.
type Access struct {
	Ref *Ref
	Off string
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
		// A comparison can read a slice of another type than the one stored.
		for _, leaf := range Leaves(planned.Root) {
			if load, ok := leaf.(*Load); ok && !types.Identical(load.Ref.Elem, elem) {
				return nil, ReasonMixedTypes
			}
		}
		p.Stmts = append(p.Stmts, planned)
	}
	if len(p.Stmts) == 1 && p.Stmts[0].Dst != nil {
		if load, ok := p.Stmts[0].Root.(*Load); ok {
			p.Copy = !negativeSource(l.Ind.Trip.Start, load.Off)
		}
	}

	trip := l.Ind.Trip
	// A loop that runs a few times, known at compile time, is faster as it is.
	if from, errFrom := strconv.ParseInt(trip.Start, 10, 64); errFrom == nil {
		if to, errTo := strconv.ParseInt(trip.Limit, 10, 64); errTo == nil {
			least := int64(minTrip)
			if p.Full {
				least = int64(MinReduceVectors * MaxLanes(elem))
			}
			if to-from < least {
				return nil, ReasonShortLoop
			}
		}
	}
	p.Start, p.Limit, p.NonEmpty, p.TripRef = trip.Start, trip.Limit, trip.NonEmpty, trip.Slice
	// The generated loop stops at the length of the stored slice, so unless the
	// limit is exactly that, the limit is spelled out on every slice or a longer
	// one would be read or written past it.
	for _, stmt := range p.Stmts {
		if stmt.Dst != nil && (trip.Slice == nil || stmt.Dst.Obj != trip.Slice.Obj) {
			p.Bound = trip.Limit
			break
		}
	}

	stored := map[types.Object]bool{}
	var refs []*Ref
	for _, stmt := range p.Stmts {
		if stmt.Dst != nil {
			stored[stmt.Dst.Obj] = true
			p.Checked = appendAccess(p.Checked, Access{Ref: stmt.Dst})
			refs = appendRef(refs, stmt.Dst)
		}
		for _, leaf := range Leaves(stmt.Root) {
			if load, ok := leaf.(*Load); ok {
				p.Checked = appendAccess(p.Checked, Access{Ref: load.Ref, Off: load.Off})
				refs = appendRef(refs, load.Ref)
			}
		}
	}
	if len(refs) > maxSlices {
		return nil, ReasonTooManySlices
	}
	for _, ref := range refs {
		if ref.Src != "" {
			p.Hoists = append(p.Hoists, Hoist{Name: ref.Name, Src: ref.Src})
		}
	}
	for i, first := range refs {
		for _, second := range refs[i+1:] {
			if stored[first.Obj] || stored[second.Obj] {
				p.Overlaps = append(p.Overlaps, [2]*Ref{first, second})
			}
		}
	}
	return p, ""
}

// negativeSource reports whether a copy would slice its source at a negative
// constant, which does not compile: the loop starts at start and reads at offset
// off, and both are constants that add up to less than zero. Such a loop panics
// when it runs, which the ordinary vector loop does as well.
func negativeSource(start, off string) bool {
	s, errStart := strconv.ParseInt(start, 10, 64)
	o, errOff := strconv.ParseInt(off, 10, 64)
	return errStart == nil && errOff == nil && s+o < 0
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
	case *Compare:
		op, operands = v.Op, []Value{v.X, v.Y}
	case *Logic:
		for _, operand := range []Value{v.X, v.Y} {
			if reason := checkOps(elem, operand); reason != "" {
				return reason
			}
		}
		return ""
	case *Select:
		if !supports(elem, OpIfElse) {
			return ReasonUnsupportedOp
		}
		for _, operand := range []Value{v.Cond, v.Then, v.Else} {
			if reason := checkOps(elem, operand); reason != "" {
				return reason
			}
		}
		return ""
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

func appendAccess(accesses []Access, access Access) []Access {
	for _, have := range accesses {
		if have.Ref.Obj == access.Ref.Obj && have.Off == access.Off {
			return accesses
		}
	}
	return append(accesses, access)
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
	case *Compare:
		return 1 + max(Depth(v.X), Depth(v.Y))
	case *Logic:
		return 1 + max(Depth(v.X), Depth(v.Y))
	case *Select:
		return 1 + max(Depth(v.Cond), Depth(v.Then), Depth(v.Else))
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
	case *Compare:
		return &Compare{Op: v.Op, X: Normalize(v.X), Y: Normalize(v.Y)}
	case *Logic:
		return &Logic{Op: v.Op, X: Normalize(v.X), Y: Normalize(v.Y)}
	case *Select:
		return &Select{Cond: Normalize(v.Cond), Then: Normalize(v.Then), Else: Normalize(v.Else)}
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
	case *Compare:
		return []Value{v.X, v.Y}
	case *Logic:
		return []Value{v.X, v.Y}
	case *Select:
		return []Value{v.Cond, v.Then, v.Else}
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
