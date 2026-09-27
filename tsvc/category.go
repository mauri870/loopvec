package tsvc

// Category groups kernels by the TSVC_2 section they come from.
type Category int

const (
	CategoryDependence Category = iota
)

func (c Category) String() string {
	switch c {
	case CategoryDependence:
		return "dependence"
	default:
		return "unknown"
	}
}
