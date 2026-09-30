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
	OpNeg    // unary: -x
	OpNot    // unary: ^x
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
	case OpNeg:
		return "Neg"
	case OpNot:
		return "Not"
	}
	return ""
}

// Commutative reports whether swapping the operands of op leaves the result
// unchanged.
func (op Op) Commutative() bool {
	switch op {
	case OpAdd, OpMul, OpAnd, OpOr, OpXor:
		return true
	}
	return false
}

// Ref names a slice variable.
type Ref struct {
	Name string
	Obj  types.Object
	Elem types.Type
}

// TripKind says what limits a loop.
type TripKind int

const (
	// TripLen limits the loop to len(Slice).
	TripLen TripKind = iota
	// TripInt limits the loop to an int variable or a positive constant.
	TripInt
)

// Trip is the iteration limit of a loop.
type Trip struct {
	Kind TripKind
	// Slice is the slice whose length limits the loop (TripLen).
	Slice *Ref
	// Limit is the int variable or constant that limits the loop (TripInt).
	Limit ast.Expr
	// Const is set when Limit is a positive integer constant.
	Const bool
}

// Induction describes how the loop index moves.
type Induction struct {
	Var  types.Object
	Step int64 // +1 or -1
	Trip Trip
}

// Loop is a canonical counted loop whose body is a sequence of same-index
// stores.
type Loop struct {
	// Node is the for or range statement, for its source span.
	Node ast.Stmt
	Ind  Induction
	Body []Store
}

// Store writes Val to Dst[i].
type Store struct {
	Dst *Ref
	Val Value
}

// Value is an element-wise expression evaluated once per iteration.
type Value interface{ isValue() }

// Load reads Ref[i].
type Load struct{ Ref *Ref }

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

func (*Load) isValue()      {}
func (*Invariant) isValue() {}
func (*Unary) isValue()     {}
func (*Binary) isValue()    {}
