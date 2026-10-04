// Package loopir is the loop IR loopvec vectorizes from.
//
// A loop goes through three stages: Lower canonicalizes a for or range
// statement into a Loop, NewPlan decides whether simd can express it and what
// checks the rewritten code needs, and the rewrite package emits source text
// from the Plan. The stages are separate so each can report its own Reason.
package loopir

import (
	"go/ast"
	"go/types"
)

// Op is an operation simd can perform lane-wise.
type Op int

const (
	OpAdd Op = iota
	OpSub
	OpMul
	OpAnd
	OpOr
	OpXor
	OpDiv
	OpAndNot // x &^ y
	OpMin    // min(x, y)
	OpMax    // max(x, y)
	OpShl    // x << count, see Shift
	OpShr    // x >> count, see Shift
	OpNeg    // unary: -x
	OpNot    // unary: ^x
	OpAbs    // unary: math.Abs(x)
	OpSqrt   // unary: math.Sqrt(x)
	OpEq     // x == y, a mask
	OpNe     // x != y
	OpLt     // x < y
	OpLe     // x <= y
	OpGt     // x > y
	OpGe     // x >= y
	OpIfElse // x where the mask is set, y elsewhere
	OpMaskAnd
	OpMaskOr
)

// Method returns the name of the simd method that implements op.
func (op Op) Method() string {
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
	case OpAndNot:
		return "AndNot"
	case OpMin:
		return "Min"
	case OpMax:
		return "Max"
	case OpShl:
		return "ShiftAllLeft"
	case OpShr:
		return "ShiftAllRight"
	case OpNeg:
		return "Neg"
	case OpNot:
		return "Not"
	case OpAbs:
		return "Abs"
	case OpSqrt:
		return "Sqrt"
	case OpEq:
		return "Equal"
	case OpNe:
		return "NotEqual"
	case OpLt:
		return "Less"
	case OpLe:
		return "LessEqual"
	case OpGt:
		return "Greater"
	case OpGe:
		return "GreaterEqual"
	case OpIfElse:
		return "IfElse"
	case OpMaskAnd:
		return "And"
	case OpMaskOr:
		return "Or"
	}
	return ""
}

// Commutative reports whether swapping the operands of op leaves the result
// unchanged.
func (op Op) Commutative() bool {
	switch op {
	case OpAdd, OpMul, OpAnd, OpOr, OpXor, OpMin, OpMax:
		return true
	}
	return false
}

// Ref names a slice: a variable, or a chain of field selections such as x.f.
type Ref struct {
	// Name is how the emitted vector code refers to the slice.
	Name string
	// Src is the source text of the slice expression when it is not a plain variable
	// (x.f); the emitted code copies it into Name before the loop, which is the same
	// slice on every iteration. Empty for a variable.
	Src  string
	Obj  types.Object
	Elem types.Type
}

// Text is the slice as the source writes it.
func (r *Ref) Text() string {
	if r.Src != "" {
		return r.Src
	}
	return r.Name
}

// Trip is the range of iterations of a loop, written as the forward loop runs
// them: Start <= i < Limit. A loop that counts down, or is written with <=, is
// stated this way too, so the rest of loopvec sees one shape.
type Trip struct {
	// Start and Limit are loop-invariant integer expressions, as source text.
	// Start is "0" for the common case.
	Start, Limit string
	// Slice is the slice whose length is Limit, when Limit is exactly len(Slice).
	Slice *Ref
	// NonEmpty is set when both are constants and Start < Limit, so the loop runs
	// at least once.
	NonEmpty bool
}

// Induction describes how the loop index moves.
type Induction struct {
	Var  types.Object
	Step int64 // +1 or -1
	Trip Trip
}

// Loop is a canonical counted loop whose body is a sequence of same-index
// stores and per-iteration temporaries.
type Loop struct {
	// Node is the for or range statement, for its source span.
	Node ast.Stmt
	Ind  Induction
	Body []Stmt
	// ValueVar is the range value variable, and ValueRef the slice it ranges
	// over, when the loop has one that is used.
	ValueVar types.Object
	ValueRef *Ref
}

// Stmt is a statement of a loop body: a Store or a Let.
type Stmt interface{ isStmt() }

// Store writes Val to Dst[i].
type Store struct {
	Dst *Ref
	Val Value
}

// Let defines a temporary, x := Val or x = Val, that later statements read.
type Let struct {
	Temp *Temp
	Val  Value
}

// Reduce folds Val into the local Acc with Op, once per iteration: Acc op= Val.
// The loop body does not otherwise read or write Acc, so the folds can happen in
// any order and grouping. Op is associative and commutative: OpAdd, OpMul, OpAnd,
// OpOr, OpXor, OpMin or OpMax.
type Reduce struct {
	Acc types.Object
	Op  Op
	Val Value
}

func (Store) isStmt()  {}
func (Let) isStmt()    {}
func (Reduce) isStmt() {}

// Temp is a scalar local written by one Let and read by later statements of the
// same iteration. The vector loop holds it as a vector. A local declared before
// the loop is accepted only when nothing outside the loop uses it.
type Temp struct {
	Var  types.Object
	Name string
	// Uses counts the reads in the body.
	Uses int
}

// Value is an element-wise expression evaluated once per iteration.
type Value interface{ isValue() }

// Load reads Ref[i+Off]. Off is source text for an integer expression that is the
// same on every iteration, "" for zero. A slice that is also stored in the loop is
// read only at a constant positive offset: iteration i then reads the element a
// later iteration writes, and the vector loop, which loads the chunk before any
// store of the body, sees the same old value.
// FromRange is set when the source is the range value variable, which holds the
// element as it was when the iteration began: a store to the same slice earlier in
// the body does not change it.
type Load struct {
	Ref       *Ref
	Off       string
	FromRange bool
}

// Invariant is an expression that has the same value on every iteration and
// is evaluated once, before the loop, and broadcast.
type Invariant struct{ Expr ast.Expr }

// Unary applies Op to X.
type Unary struct {
	Op Op
	X  Value
}

// Binary applies Op to X and Y.
type Binary struct {
	Op   Op
	X, Y Value
}

// Shift shifts X by Count bits, a scalar that is the same on every iteration.
// Unlike the operand of a Binary it is not broadcast: simd takes it as an
// argument. It is a non-negative constant or an unsigned variable, so evaluating
// it cannot panic, where a negative signed count would.
type Shift struct {
	Op    Op // OpShl or OpShr
	X     Value
	Count ast.Expr
}

// Compare is a lane-wise comparison of X and Y, a mask. Op is OpEq through OpGe.
type Compare struct {
	Op   Op
	X, Y Value
}

// Logic combines two masks with OpMaskAnd or OpMaskOr. Go's && and || evaluate the
// right side only when the left does not decide the result; both sides here are
// evaluated, which is the same because lowering only accepts a right side that
// reads nothing the left did not.
type Logic struct {
	Op   Op
	X, Y Value
}

// Select is Then in the lanes where Cond is set and Else in the others: an
// if/else whose branches each store the same element. Both branches are evaluated
// for every lane, so lowering only accepts one that cannot fail where the scalar
// loop would not have run it.
type Select struct {
	Cond, Then, Else Value
}

// Use reads a temporary.
type Use struct{ Temp *Temp }

func (*Shift) isValue()     {}
func (*Use) isValue()       {}
func (*Load) isValue()      {}
func (*Invariant) isValue() {}
func (*Unary) isValue()     {}
func (*Binary) isValue()    {}
func (*Compare) isValue()   {}
func (*Logic) isValue()     {}
func (*Select) isValue()    {}
