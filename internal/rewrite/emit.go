package rewrite

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"math/big"
	"strconv"
	"strings"

	"github.com/mauri870/loopvec/internal/loopir"
)

// generateReplacement returns (loopText, preText, error) for a planned loop.
// preText is code to insert before the loop: the broadcast of each
// loop-invariant operand. idx is a per-file unique counter used to avoid name
// collisions when several loops in the same function scope broadcast the same
// simd type.
func generateReplacement(plan *loopir.Plan, fset *token.FileSet, idx int) (string, string, error) {
	if plan.Copy {
		e := &emitter{plan: plan}
		return e.checked(e.copyCall()), "", nil
	}
	e := &emitter{plan: plan, fset: fset, loads: map[*loopir.Load]string{}, temps: map[*loopir.Temp]string{}, broadcasts: broadcastNames(plan, idx)}

	var pre strings.Builder
	for _, inv := range invariants(plan) {
		var text bytes.Buffer
		if err := format.Node(&text, fset, inv.Expr); err != nil {
			return "", "", err
		}
		if pre.Len() > 0 {
			pre.WriteByte('\n')
		}
		fmt.Fprintf(&pre, "%s := simd.Broadcast%s(%s)", e.broadcasts[inv], plan.SimdType, text.String())
	}
	if plan.Full {
		text, err := e.fullLoop()
		if err != nil {
			return "", "", err
		}
		return e.checked(text), pre.String(), nil
	}
	return e.checked(e.loop()), pre.String(), nil
}

type emitter struct {
	plan       *loopir.Plan
	fset       *token.FileSet
	loads      map[*loopir.Load]string
	temps      map[*loopir.Temp]string
	broadcasts map[*loopir.Invariant]string
	// loadCount and tempCount number the vectors loaded and the temporaries
	// defined, across every copy of the loop body, so no two names collide.
	loadCount, tempCount int
}

// copyCall renders a copy loop as the copy builtin. The length checks around it
// (see checked) guarantee the source has at least as many elements as the loop
// runs, so copy moves exactly that many.
func (e *emitter) copyCall() string {
	store := e.plan.Stmts[0]
	load := store.Root.(*loopir.Load)
	destination, source := store.Dst.Name, load.Ref.Name
	switch {
	case e.plan.Start != "0":
		destination += "[" + e.plan.Start + ":" + e.plan.Bound + "]"
		source += "[" + addOffset(e.plan.Start, load.Off) + ":]"
	case e.plan.Bound != "":
		destination += "[:" + e.plan.Bound + "]"
		if load.Off != "" {
			source += "[" + load.Off + ":]"
		}
	case load.Off != "":
		source += "[" + load.Off + ":]"
	}
	return fmt.Sprintf("copy(%s, %s)", destination, source)
}

// limit is the loop's iteration limit.
func (e *emitter) limit() string {
	return e.plan.Limit
}

// count is the number of iterations, limit - start.
func (e *emitter) count() string {
	if e.plan.Start == "0" {
		return e.plan.Limit
	}
	return operand(e.plan.Limit) + " - " + operand(e.plan.Start)
}

// operand parenthesizes an expression that has an operator at its top level, so
// it can be an operand of another.
func operand(expr string) string {
	if strings.ContainsAny(expr, " +-*/") {
		return "(" + expr + ")"
	}
	return expr
}

// operand slices ref from position pos, capped at the limit when it is not
// len(Dst). A slice read at an offset is not capped: the cap is the limit, and
// the elements it needs end past it.
func (e *emitter) operand(ref *loopir.Ref, pos, off string) string {
	if e.plan.Bound != "" && off == "" {
		return ref.Name + "[" + pos + ":" + e.plan.Bound + "]"
	}
	return ref.Name + "[" + addOffset(pos, off) + ":]"
}

// addOffset is the expression pos + off, where off is "" for zero, a constant, or
// an expression.
func addOffset(pos, off string) string {
	if p, err := strconv.ParseInt(pos, 10, 64); err == nil {
		if o, err := strconv.ParseInt(off, 10, 64); err == nil {
			return strconv.FormatInt(p+o, 10)
		}
	}
	switch {
	case off == "":
		return pos
	case off[0] == '-' && !strings.ContainsAny(off[1:], " +-*/"):
		return pos + off
	case strings.ContainsAny(off, " +-*/"):
		return pos + "+(" + off + ")"
	}
	return pos + "+" + off
}

// lastIndex is the expression limit + off - 1, the last index read at offset off.
func lastIndex(limit, off string) string {
	if off == "" {
		return operand(limit) + "-1"
	}
	if v, err := strconv.ParseInt(off, 10, 64); err == nil {
		switch c := v - 1; {
		case c == 0:
			return limit
		case c > 0:
			return operand(limit) + "+" + strconv.FormatInt(c, 10)
		default:
			return operand(limit) + strconv.FormatInt(c, 10)
		}
	}
	return operand(limit) + addOffset("", off) + "-1"
}

// loop generates the vector loop: for each statement in order, load each operand
// a partial vector at a time, compute, and store it or keep it in a temporary.
// The loop advances by the number of lanes the first load reported, or the first
// store when nothing was loaded before it. Running one statement over the whole
// vector before the next is the scalar order, because every access is at the loop
// index.
func (e *emitter) loop() string {
	var b strings.Builder
	fmt.Fprintf(&b, "for _i := %s; _i < %s; {\n", e.plan.Start, e.limit())
	e.statements(&b, false, 0, 1)
	b.WriteString("\t_i += _n\n")
	b.WriteString("}")
	return b.String()
}

// statements writes the body of the vector loop. A slice already loaded is not
// loaded again until a store to it, since distinct slices are known not to
// overlap. With full, loads and stores cover a whole vector and a fold goes into
// its accumulator; otherwise they are partial and the first one defines _n.
// Copy is which of copies unrolled bodies this is: it works on the elements
// copy vectors past _i, and folds into its own accumulators.
func (e *emitter) statements(b *strings.Builder, full bool, copy, copies int) {
	offset := "_i"
	if copy > 0 {
		offset = fmt.Sprintf("_i+%d*_lanes", copy)
		if copy == 1 {
			offset = "_i+_lanes"
		}
	}
	haveLanes := false
	type key struct {
		obj types.Object
		off string
	}
	loaded := map[key]string{}
	emitLoad := func(load *loopir.Load) {
		if name, ok := loaded[key{load.Ref.Obj, load.Off}]; ok {
			e.loads[load] = name
			return
		}
		e.loadCount++
		name := fmt.Sprintf("_v%d", e.loadCount)
		loaded[key{load.Ref.Obj, load.Off}] = name
		e.loads[load] = name
		if full {
			fmt.Fprintf(b, "\t%s := simd.Load%s(%s)\n", name, e.plan.SimdType, e.operand(load.Ref, offset, load.Off))
			return
		}
		lanes := "_"
		if !haveLanes {
			lanes, haveLanes = "_n", true
		}
		fmt.Fprintf(b, "\t%s, %s := simd.Load%sPart(%s)\n", name, lanes, e.plan.SimdType, e.operand(load.Ref, offset, load.Off))
	}
	// A slice stored in the loop and read ahead of the index (a[i+1]) is read
	// before any store of the body: the scalar loop reads the value from before
	// the loop, because the iteration that writes it has not run yet, and a
	// store earlier in the body would already have changed it in the vector.
	stored := map[types.Object]bool{}
	for _, stmt := range e.plan.Stmts {
		if stmt.Dst != nil {
			stored[stmt.Dst.Obj] = true
		}
	}
	for _, stmt := range e.plan.Stmts {
		for _, leaf := range loopir.Leaves(stmt.Root) {
			if load, ok := leaf.(*loopir.Load); ok && stored[load.Ref.Obj] && load.Off != "" {
				emitLoad(load)
			}
		}
	}
	accs := 0
	for _, stmt := range e.plan.Stmts {
		for _, leaf := range loopir.Leaves(stmt.Root) {
			if load, ok := leaf.(*loopir.Load); ok {
				emitLoad(load)
			}
		}
		switch {
		case stmt.Temp != nil:
			e.tempCount++
			name := fmt.Sprintf("_t%d", e.tempCount)
			e.temps[stmt.Temp] = name
			fmt.Fprintf(b, "\t%s := %s\n", name, e.expr(stmt.Root))
		case stmt.Acc != nil:
			accs++
			acc := accName(accs, copy, copies)
			fmt.Fprintf(b, "\t%s = %s\n", acc, e.fold(stmt, acc))
		case full:
			fmt.Fprintf(b, "\t%s.Store(%s)\n", e.expr(stmt.Root), e.operand(stmt.Dst, offset, ""))
			delete(loaded, key{stmt.Dst.Obj, ""})
		default:
			text := fmt.Sprintf("%s.StorePart(%s)", e.expr(stmt.Root), e.operand(stmt.Dst, offset, ""))
			if !haveLanes {
				// Nothing was loaded, so the store reports how many lanes it wrote.
				text, haveLanes = "_n := "+text, true
			}
			fmt.Fprintf(b, "\t%s\n", text)
			delete(loaded, key{stmt.Dst.Obj, ""})
		}
	}
}

// accName names the vector accumulator of the n'th fold (counting from 1) in
// copy of copies unrolled bodies.
func accName(n, copy, copies int) string {
	if copies == 1 {
		return fmt.Sprintf("_a%d", n)
	}
	return fmt.Sprintf("_a%d_%d", n, copy)
}

// fold is the expression that folds the statement's value into the accumulator
// acc. A float sum of a product is one fused multiply-add.
func (e *emitter) fold(stmt loopir.PlanStmt, acc string) string {
	if mul, ok := stmt.Root.(*loopir.Binary); ok && stmt.Op == loopir.OpAdd && mul.Op == loopir.OpMul && e.isFloat() {
		return fmt.Sprintf("%s.MulAdd(%s, %s)", e.expr(mul.X), e.expr(mul.Y), acc)
	}
	return fmt.Sprintf("%s.%s(%s)", acc, stmt.Op.Method(), e.expr(stmt.Root))
}

// minVectors is how many of the widest vectors a reduction needs before the
// vector loop is used. Setting up the accumulators and combining the lanes costs
// about 6ns however short the loop, more than the scalar loop for fewer than
// roughly forty int32 elements on AVX-512, and the scalar loop that follows runs
// anything the vector loop does not. The bound is in elements so that a short
// loop pays nothing for the vector width.
const minVectors = 4

// fullLoop generates a loop that folds into accumulators. The vector loop runs
// while a whole vector remains, every accumulator starts at the identity of its
// operation, and afterwards the lanes are combined into the scalar accumulator
// (simd has no horizontal reduction on every toolchain, so they go through an
// array). What is left, fewer than one vector, runs the original body. A partial
// load would zero-pad, which is the identity for a sum but not for a product,
// min, max, or and.
func (e *emitter) fullLoop() (string, error) {
	var b strings.Builder
	b.WriteString("{\n")
	var folds []loopir.PlanStmt
	for _, stmt := range e.plan.Stmts {
		if stmt.Acc != nil {
			folds = append(folds, stmt)
		}
	}
	copies := unroll(folds)
	fmt.Fprintf(&b, "_i := %s\n", e.plan.Start)
	fmt.Fprintf(&b, "if %s >= %d {\n", e.count(), minVectors*maxLanes(folds[0].Acc.Type()))
	for n, fold := range folds {
		for k := range copies {
			fmt.Fprintf(&b, "%s := simd.Broadcast%s(%s)\n", accName(n+1, k, copies), e.plan.SimdType, identity(fold.Acc.Type(), fold.Op))
		}
	}
	fmt.Fprintf(&b, "_lanes := %s.Len()\n", accName(1, 0, copies))
	if copies > 1 {
		fmt.Fprintf(&b, "for ; _i+%d*_lanes <= %s; _i += %d*_lanes {\n", copies, e.limit(), copies)
		for k := range copies {
			e.statements(&b, true, k, copies)
		}
		b.WriteString("}\n")
	}
	fmt.Fprintf(&b, "for ; _i+_lanes <= %s; _i += _lanes {\n", e.limit())
	e.statements(&b, true, 0, copies)
	b.WriteString("}\n")
	for n, fold := range folds {
		first := accName(n+1, 0, copies)
		for k := 1; k < copies; k++ {
			fmt.Fprintf(&b, "%s = %s.%s(%s)\n", first, first, fold.Op.Method(), accName(n+1, k, copies))
		}
		name := fold.Acc.Name()
		fmt.Fprintf(&b, "var _buf%d [%d]%s\n%s.Store(_buf%[1]d[:])\n", n+1, maxLanes(fold.Acc.Type()), fold.Acc.Type(), first)
		fmt.Fprintf(&b, "for _, _x := range _buf%d[:_lanes] {\n%s = %s\n}\n", n+1, name, combine(fold.Op, name, "_x"))
	}
	b.WriteString("}\n")

	loop := e.plan.Loop
	index := loop.Ind.Var.Name()
	if index == "_" {
		// for _, v := range x has no index to continue from.
		index = "_k"
	}
	fmt.Fprintf(&b, "for %[1]s := _i; %[1]s < %[2]s; %[1]s++ {\n", index, e.limit())
	if loop.ValueVar != nil {
		fmt.Fprintf(&b, "%s := %s[%s]\n", loop.ValueVar.Name(), loop.ValueRef.Name, index)
	}
	var block *ast.BlockStmt
	switch node := loop.Node.(type) {
	case *ast.RangeStmt:
		block = node.Body
	case *ast.ForStmt:
		block = node.Body
	}
	for _, stmt := range block.List {
		var text bytes.Buffer
		if err := format.Node(&text, e.fset, stmt); err != nil {
			return "", err
		}
		b.Write(text.Bytes())
		b.WriteByte('\n')
	}
	b.WriteString("}\n}")
	return b.String(), nil
}

// unroll is how many bodies the main loop of a reduction runs per iteration, each
// into its own accumulators. A float add takes several cycles and depends on the
// previous one, so a single accumulator leaves the vector unit idle; the copies
// overlap those latencies. Measured on a float dot product, two copies nearly
// double the speed of one and four add a few percent more; an integer add takes
// one cycle and copies did not help. The accumulators have to stay in registers.
func unroll(folds []loopir.PlanStmt) int {
	if t := folds[0].Acc.Type().Underlying().(*types.Basic); t.Info()&types.IsFloat == 0 {
		return 1
	}
	return max(1, min(4, 8/len(folds)))
}

// identity is the value that leaves op unchanged, as source text for an element
// of type t.
func identity(t types.Type, op loopir.Op) string {
	basic := t.Underlying().(*types.Basic)
	var bits int
	switch basic.Kind() {
	case types.Int8, types.Uint8:
		bits = 8
	case types.Int16, types.Uint16:
		bits = 16
	case types.Int32, types.Uint32:
		bits = 32
	default:
		bits = 64
	}
	unsigned := basic.Info()&types.IsUnsigned != 0
	top := new(big.Int).Lsh(big.NewInt(1), uint(bits))
	switch op {
	case loopir.OpMul:
		return "1"
	case loopir.OpAnd:
		if unsigned {
			return new(big.Int).Sub(top, big.NewInt(1)).String()
		}
		return "-1"
	case loopir.OpMin:
		if unsigned {
			return new(big.Int).Sub(top, big.NewInt(1)).String()
		}
		return new(big.Int).Sub(new(big.Int).Rsh(top, 1), big.NewInt(1)).String()
	case loopir.OpMax:
		if unsigned {
			return "0"
		}
		return new(big.Int).Neg(new(big.Int).Rsh(top, 1)).String()
	}
	return "0"
}

// maxLanes is the most elements of type t a vector holds: simd vectors are at
// most 512 bits.
func maxLanes(t types.Type) int {
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

// combine is the scalar expression that folds x into acc with op.
func combine(op loopir.Op, acc, x string) string {
	switch op {
	case loopir.OpMin:
		return "min(" + acc + ", " + x + ")"
	case loopir.OpMax:
		return "max(" + acc + ", " + x + ")"
	}
	return acc + " " + map[loopir.Op]string{
		loopir.OpAdd: "+", loopir.OpMul: "*", loopir.OpAnd: "&", loopir.OpOr: "|", loopir.OpXor: "^",
	}[op] + " " + x
}

// expr renders v as a chain of simd method calls.
func (e *emitter) expr(v loopir.Value) string {
	switch v := v.(type) {
	case *loopir.Load:
		return e.loads[v]
	case *loopir.Invariant:
		return e.broadcasts[v]
	case *loopir.Use:
		return e.temps[v.Temp]
	case *loopir.Shift:
		return fmt.Sprintf("%s.%s(uint64(%s))", e.expr(v.X), v.Op.Method(), types.ExprString(v.Count))
	case *loopir.Unary:
		return e.expr(v.X) + "." + v.Op.Method() + "()"
	case *loopir.Binary:
		// A multiply feeding an add is one fused multiply-add. The spec allows
		// the fusion, but the result can differ from the scalar loop in the
		// last bit.
		if mul, ok := v.X.(*loopir.Binary); ok && v.Op == loopir.OpAdd && mul.Op == loopir.OpMul && e.isFloat() {
			return fmt.Sprintf("%s.MulAdd(%s, %s)", e.expr(mul.X), e.expr(mul.Y), e.expr(v.Y))
		}
		return fmt.Sprintf("%s.%s(%s)", e.expr(v.X), v.Op.Method(), e.expr(v.Y))
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
	for _, access := range e.plan.Checked {
		if e.plan.Bound == "" && access.Off == "" && e.plan.TripRef != nil && access.Ref.Obj == e.plan.TripRef.Obj {
			continue
		}
		fmt.Fprintf(&checks, "_ = %s[%s]\n", access.Ref.Name, lastIndex(limit, access.Off))
	}
	if checks.Len() == 0 {
		return loopText
	}
	if e.plan.NonEmpty {
		return checks.String() + loopText
	}
	return "if " + limit + " > " + e.plan.Start + " {\n" + checks.String() + loopText + "\n}"
}

// invariants returns the loop-invariant operands of every store in evaluation
// order.
func invariants(plan *loopir.Plan) []*loopir.Invariant {
	var out []*loopir.Invariant
	for _, store := range plan.Stmts {
		for _, leaf := range loopir.Leaves(store.Root) {
			if inv, ok := leaf.(*loopir.Invariant); ok {
				out = append(out, inv)
			}
		}
	}
	return out
}

// broadcastNames names the broadcast of each invariant operand. A single
// broadcast in a simple loop is _vc<Type><n>. In a two-level expression the
// operands under the operand evaluated first are _vcA<Type><n> and the rest
// _vcB<Type><n>. Anything else, including every loop with several stores, is
// numbered.
func broadcastNames(plan *loopir.Plan, idx int) map[*loopir.Invariant]string {
	all := invariants(plan)
	names := map[*loopir.Invariant]string{}
	if len(plan.Stmts) == 1 {
		root := plan.Stmts[0].Root
		depth := loopir.Depth(root)
		switch {
		case len(all) == 1 && depth <= 1:
			names[all[0]] = fmt.Sprintf("_vc%s%d", plan.SimdType, idx)
			return names
		case depth == 2:
			first := map[loopir.Value]bool{}
			for _, leaf := range loopir.Leaves(loopir.Children(root)[0]) {
				first[leaf] = true
			}
			used := map[string]bool{}
			unique := true
			for _, inv := range all {
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
	}
	for n, inv := range all {
		names[inv] = fmt.Sprintf("_vc%s%d_%d", plan.SimdType, idx, n+1)
	}
	return names
}
