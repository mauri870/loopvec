package tsvc

// Kernel describes one ported TSVC_2 loop: how to set up its inputs, how to
// run it, and how to check its result.
type Kernel struct {
	Name     string
	Category Category
	Reps     int
	Exact    Exactness
	Setup    func(x *Arrays)
	Run      func(x *Arrays)
	Checksum func(x *Arrays) float64
	Hash     func(x *Arrays) uint64
}

// Kernels lists the 10 ported kernels, in TSVC_2 source order. It is
// hand-written for now; a later change generates it from the //tsvc:kernel
// directives in kernels.go.
var Kernels = []Kernel{
	{
		Name: "s000", Category: CategoryDependence, Reps: 2, Exact: ExactBits,
		Setup: setupS000, Checksum: sumA, Hash: hashA,
		Run: func(x *Arrays) { s000(x.A, x.B) },
	},
	{
		Name: "s111", Category: CategoryDependence, Reps: 2, Exact: ExactBits,
		Setup: setupS111, Checksum: sumA, Hash: hashA,
		Run: func(x *Arrays) { s111(x.A, x.B) },
	},
	{
		Name: "s1111", Category: CategoryDependence, Reps: 2, Exact: ExactFused,
		Setup: setupS111, Checksum: sumA, Hash: hashA,
		Run: func(x *Arrays) { s1111(x.A, x.B, x.C, x.D) },
	},
	{
		Name: "s112", Category: CategoryDependence, Reps: 3, Exact: ExactBits,
		Setup: setupS112, Checksum: sumA, Hash: hashA,
		Run: func(x *Arrays) { s112(x.A, x.B) },
	},
	{
		Name: "s1112", Category: CategoryDependence, Reps: 3, Exact: ExactBits,
		Setup: setupS112, Checksum: sumA, Hash: hashA,
		Run: func(x *Arrays) { s1112(x.A, x.B) },
	},
	{
		Name: "s113", Category: CategoryDependence, Reps: 4, Exact: ExactBits,
		Setup: setupS113, Checksum: sumA, Hash: hashA,
		Run: func(x *Arrays) { s113(x.A, x.B) },
	},
	{
		Name: "s1113", Category: CategoryDependence, Reps: 2, Exact: ExactBits,
		Setup: setupS113, Checksum: sumA, Hash: hashA,
		Run: func(x *Arrays) { s1113(x.A, x.B) },
	},
	{
		Name: "s114", Category: CategoryDependence, Reps: 1, Exact: ExactBits,
		Setup: setupS114, Checksum: sumAA, Hash: hashAA,
		Run: func(x *Arrays) { s114(x.AA, x.BB) },
	},
	{
		Name: "s115", Category: CategoryDependence, Reps: 1, Exact: ExactFused,
		Setup: setupS115, Checksum: sumA, Hash: hashA,
		Run: func(x *Arrays) { s115(x.A, x.AA) },
	},
	{
		Name: "s1115", Category: CategoryDependence, Reps: 1, Exact: ExactFused,
		Setup: setupS115, Checksum: sumAA, Hash: hashAA,
		Run: func(x *Arrays) { s1115(x.AA, x.BB, x.CC) },
	},
}
