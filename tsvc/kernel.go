package tsvc

// Kernel describes one ported TSVC_2 loop: how to set up its inputs, how to
// run it, and how to check its result.
type Kernel struct {
	Name     string
	Category Category
	Reps     int
	Exact    Exactness
	// Expect says whether loopvec should vectorize the kernel.
	Expect Expectation
	// Bytes is the size, in bytes, of every array Run reads or writes once
	// per element. It feeds b.SetBytes so benchmarks report throughput.
	Bytes    int64
	Setup    func(x *Arrays)
	Run      func(x *Arrays)
	Checksum func(x *Arrays) float64
	Hash     func(x *Arrays) uint64
}
