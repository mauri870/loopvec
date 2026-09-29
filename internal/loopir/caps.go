package loopir

import "go/types"

// SimdType returns the simd vector type name for element type t, or the empty
// string when simd has none. Only the predeclared numeric types qualify: a
// named type such as type Celsius float32 needs conversions the rewrite does
// not emit.
func SimdType(t types.Type) string {
	basic, ok := t.(*types.Basic)
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

// supports reports whether simd implements op for element type t. There is no
// 64-bit integer multiply or integer divide, no negation of unsigned
// integers, and no bitwise not of floats.
func supports(t types.Type, op Op) bool {
	kind := t.(*types.Basic).Kind()
	switch op {
	case OpMul:
		return kind != types.Int64 && kind != types.Uint64
	case OpDiv:
		return kind == types.Float32 || kind == types.Float64
	case OpNeg:
		switch kind {
		case types.Uint8, types.Uint16, types.Uint32, types.Uint64:
			return false
		}
	case OpNot:
		return kind != types.Float32 && kind != types.Float64
	}
	return true
}
