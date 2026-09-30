package loopir

import "go/types"

//go:generate go run ./gen -o caps_gen.go

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

// supports reports whether simd has the method that implements op on the
// vector type of element type t.
func supports(t types.Type, op Op) bool {
	return simdMethods[SimdType(t)][op.Method()]
}
