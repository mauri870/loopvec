package tsvc

import (
	"flag"
	"math"
	"testing"
)

var update = flag.Bool("update", false, "rewrite golden files")

func TestGolden(t *testing.T) {
	golden, err := readGolden(goldenPath())
	if err != nil {
		if !*update {
			t.Fatalf("readGolden: %v", err)
		}
		golden = make(map[string]Result)
	}

	for _, k := range Kernels {
		t.Run(k.Name, func(t *testing.T) {
			x := NewArrays()
			k.Setup(x)
			for range k.Reps {
				k.Run(x)
			}

			got := Result{Sum: k.Checksum(x), Hash: k.Hash(x)}
			if *update {
				golden[k.Name] = got
				return
			}

			want, ok := golden[k.Name]
			if !ok {
				t.Fatalf("no golden for %s; run with -update", k.Name)
			}
			check(t, k.Exact, want, got)
		})
	}

	if *update {
		if err := writeGolden(goldenPath(), golden); err != nil {
			t.Fatalf("writeGolden: %v", err)
		}
	}
}

// check compares a kernel's result against its golden according to its
// exactness. ExactBits requires the hash and sum to match exactly.
// ExactFused allows the relative error a compiler-contracted FMA would
// introduce, and ExactReassoc the error of regrouping a sum; both ignore the
// hash.
func check(t *testing.T, exact Exactness, want, got Result) {
	switch exact {
	case ExactBits:
		if want.Hash != got.Hash || want.Sum != got.Sum {
			t.Errorf("sum = %v, hash = %#x; want sum = %v, hash = %#x", got.Sum, got.Hash, want.Sum, want.Hash)
		}
	case ExactFused:
		if e := relErr(want.Sum, got.Sum); e > 1e-5 {
			t.Errorf("sum = %v; want %v (relative error %v > 1e-5)", got.Sum, want.Sum, e)
		}
	case ExactReassoc:
		if e := relErr(want.Sum, got.Sum); e > reassocTolerance {
			t.Errorf("sum = %v; want %v (relative error %v > %v)", got.Sum, want.Sum, e, reassocTolerance)
		}
	default:
		t.Fatalf("unknown exactness %v", exact)
	}
}

// reassocTolerance bounds the relative error of regrouping a sum or product of
// the 32000 float32 elements of a TSVC array. Recursive summation of n terms is
// within n*u of the exact result, u = 2^-24, which is 2e-3 for n = 32000, and
// both the scalar and the regrouped loop satisfy it. Measured against the scalar
// goldens the errors are 2e-7 for the sums, 1e-4 for the dot products, and 5e-4
// for the product.
const reassocTolerance = 2e-3

func relErr(want, got float64) float64 {
	if want == 0 {
		return math.Abs(got)
	}
	return math.Abs(got-want) / math.Abs(want)
}
