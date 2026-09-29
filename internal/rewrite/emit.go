package rewrite

import (
	"bytes"
	"fmt"
	"go/format"
	"go/token"
	"strings"

	"github.com/mauri870/loopvec/internal/loopir"
)

// generateReplacement returns (loopText, preText, error) for a planned loop.
// preText is code to insert before the loop: the broadcast of each
// loop-invariant operand. idx is a per-file unique counter used to avoid name
// collisions when several loops in the same function scope broadcast the same
// simd type.
func generateReplacement(plan *loopir.Plan, fset *token.FileSet, idx int) (string, string, error) {
	e := &emitter{plan: plan, loads: map[*loopir.Load]string{}, broadcasts: broadcastNames(plan, idx)}

	var pre strings.Builder
	for _, leaf := range loopir.Leaves(plan.Root) {
		inv, ok := leaf.(*loopir.Invariant)
		if !ok {
			continue
		}
		var text bytes.Buffer
		if err := format.Node(&text, fset, inv.Expr); err != nil {
			return "", "", err
		}
		if pre.Len() > 0 {
			pre.WriteByte('\n')
		}
		fmt.Fprintf(&pre, "%s := simd.Broadcast%s(%s)", e.broadcasts[inv], plan.SimdType, text.String())
	}
	return e.checked(e.loop()), pre.String(), nil
}

type emitter struct {
	plan       *loopir.Plan
	loads      map[*loopir.Load]string
	broadcasts map[*loopir.Invariant]string
}

// limit is the loop's iteration limit.
func (e *emitter) limit() string {
	if e.plan.Bound != "" {
		return e.plan.Bound
	}
	return "len(" + e.plan.Dst.Name + ")"
}

// operand slices ref from the current position, capped at the limit when it is
// not len(Dst).
func (e *emitter) operand(ref *loopir.Ref) string {
	if e.plan.Bound != "" {
		return ref.Name + "[_i:" + e.plan.Bound + "]"
	}
	return ref.Name + "[_i:]"
}

// loop generates the vector loop: load each operand a partial vector at a
// time, compute, store, and advance by the number of lanes the first load
// returned.
func (e *emitter) loop() string {
	var b strings.Builder
	fmt.Fprintf(&b, "for _i := 0; _i < %s; {\n", e.limit())
	count := 0
	for _, leaf := range loopir.Leaves(e.plan.Root) {
		load, ok := leaf.(*loopir.Load)
		if !ok {
			continue
		}
		count++
		name := fmt.Sprintf("_v%d", count)
		e.loads[load] = name
		lanes := "_"
		if count == 1 {
			lanes = "_n"
		}
		fmt.Fprintf(&b, "\t%s, %s := simd.Load%sPart(%s)\n", name, lanes, e.plan.SimdType, e.operand(load.Ref))
	}
	store := fmt.Sprintf("%s.StorePart(%s)", e.expr(e.plan.Root), e.operand(e.plan.Dst))
	if count == 0 {
		// Nothing was loaded, so the store reports how many lanes it wrote.
		store = "_n := " + store
	}
	fmt.Fprintf(&b, "\t%s\n", store)
	b.WriteString("\t_i += _n\n")
	b.WriteString("}")
	return b.String()
}

// expr renders v as a chain of simd method calls.
func (e *emitter) expr(v loopir.Value) string {
	switch v := v.(type) {
	case *loopir.Load:
		return e.loads[v]
	case *loopir.Invariant:
		return e.broadcasts[v]
	case *loopir.Unary:
		return e.expr(v.X) + "." + opMethod(v.Op) + "()"
	case *loopir.Binary:
		// A multiply feeding an add is one fused multiply-add. The spec allows
		// the fusion, but the result can differ from the scalar loop in the
		// last bit.
		if mul, ok := v.X.(*loopir.Binary); ok && v.Op == loopir.OpAdd && mul.Op == loopir.OpMul && e.isFloat() {
			return fmt.Sprintf("%s.MulAdd(%s, %s)", e.expr(mul.X), e.expr(mul.Y), e.expr(v.Y))
		}
		return fmt.Sprintf("%s.%s(%s)", e.expr(v.X), opMethod(v.Op), e.expr(v.Y))
	}
	return ""
}

func (e *emitter) isFloat() bool {
	return e.plan.SimdType == "Float32s" || e.plan.SimdType == "Float64s"
}

// checked wraps loopText so the operand lengths are verified before the loop
// runs, and a loop with a limit of zero is skipped. Every slice's length is
// checked against the limit up front, so a slice shorter than the limit still
// panics with an index error, as the original loop would, instead of the simd
// load silently returning a short, zero-padded partial result. The
// destination needs that check only with an explicit limit: otherwise
// len(Dst) is the limit and it trivially satisfies it.
func (e *emitter) checked(loopText string) string {
	limit := e.limit()
	var checks strings.Builder
	for _, ref := range e.plan.Checked {
		if e.plan.Bound == "" && ref.Obj == e.plan.Dst.Obj {
			continue
		}
		fmt.Fprintf(&checks, "_ = %s[%s-1]\n", ref.Name, limit)
	}
	if checks.Len() == 0 {
		return loopText
	}
	if e.plan.BoundConst {
		return checks.String() + loopText
	}
	return "if " + limit + " > 0 {\n" + checks.String() + loopText + "\n}"
}

// broadcastNames names the broadcast of each invariant operand. A single
// broadcast in a simple loop is _vc<Type><n>. In a two-level expression the
// operands under the operand evaluated first are _vcA<Type><n> and the rest
// _vcB<Type><n>. Anything else is numbered.
func broadcastNames(plan *loopir.Plan, idx int) map[*loopir.Invariant]string {
	var invariants []*loopir.Invariant
	for _, leaf := range loopir.Leaves(plan.Root) {
		if inv, ok := leaf.(*loopir.Invariant); ok {
			invariants = append(invariants, inv)
		}
	}
	names := map[*loopir.Invariant]string{}
	depth := loopir.Depth(plan.Root)
	switch {
	case len(invariants) == 1 && depth <= 1:
		names[invariants[0]] = fmt.Sprintf("_vc%s%d", plan.SimdType, idx)
		return names
	case depth == 2:
		first := map[loopir.Value]bool{}
		for _, leaf := range loopir.Leaves(loopir.Children(plan.Root)[0]) {
			first[leaf] = true
		}
		used := map[string]bool{}
		unique := true
		for _, inv := range invariants {
			letter := "B"
			if first[inv] {
				letter = "A"
			}
			unique = unique && !used[letter]
			used[letter] = true
			names[inv] = fmt.Sprintf("_vc%s%s%d", letter, plan.SimdType, idx)
		}
		if unique {
			return names
		}
	}
	for n, inv := range invariants {
		names[inv] = fmt.Sprintf("_vc%s%d_%d", plan.SimdType, idx, n+1)
	}
	return names
}

// opMethod returns the simd method that implements op.
func opMethod(op loopir.Op) string {
	switch op {
	case loopir.OpAdd:
		return "Add"
	case loopir.OpSub:
		return "Sub"
	case loopir.OpMul:
		return "Mul"
	case loopir.OpAnd:
		return "And"
	case loopir.OpOr:
		return "Or"
	case loopir.OpXor:
		return "Xor"
	case loopir.OpDiv:
		return "Div"
	case loopir.OpNeg:
		return "Neg"
	case loopir.OpNot:
		return "Not"
	}
	return ""
}
