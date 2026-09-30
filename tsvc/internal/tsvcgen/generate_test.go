package main

import (
	"go/token"
	"strings"
	"testing"
)

func TestToEntryUnknownParameter(t *testing.T) {
	d := directive{
		FuncName: "s000",
		Params:   []string{"a", "zz"},
		Category: "dependence",
		Reps:     2,
		Setup:    "s000",
		Checksum: "a",
		Exact:    "bits",
		Expect:   "vectorize",
		Pos:      token.Position{Filename: "kernels.go", Line: 5},
	}
	_, err := toEntry(d)
	if err == nil {
		t.Fatal("toEntry: want error for unknown parameter, got nil")
	}
	if want := `parameter "zz" has no matching Arrays field`; !strings.Contains(err.Error(), want) {
		t.Fatalf("toEntry error = %q, want to contain %q", err, want)
	}
}

func TestToEntryUnknownExpect(t *testing.T) {
	d := directive{
		FuncName: "s000",
		Params:   []string{"a", "b"},
		Category: "dependence",
		Reps:     2,
		Setup:    "s000",
		Checksum: "a",
		Exact:    "bits",
		Expect:   "maybe",
		Pos:      token.Position{Filename: "kernels.go", Line: 5},
	}
	_, err := toEntry(d)
	if err == nil {
		t.Fatal("toEntry: want error for unknown expect, got nil")
	}
	if want := `unknown expect "maybe"`; !strings.Contains(err.Error(), want) {
		t.Fatalf("toEntry error = %q, want to contain %q", err, want)
	}
}

func TestGenerate(t *testing.T) {
	src := `package tsvc

//tsvc:kernel category=dependence reps=2 setup=s000 checksum=a exact=bits expect=vectorize
func s000(a, b []float32) {}

//tsvc:kernel category=dependence reps=1 setup=s114 checksum=aa exact=bits expect=decline
func s114(aa, bb [][]float32) {}
`
	fset := token.NewFileSet()
	ds, err := parseDirectives(fset, "kernels.go", []byte(src))
	if err != nil {
		t.Fatalf("parseDirectives: %v", err)
	}

	out, err := generate(ds)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	got := string(out)
	for _, want := range []string{
		`Name: "s000"`, `Expect: ExpectVectorize`, `Run: func(x *Arrays) { s000(x.A, x.B) }`,
		`Name: "s114"`, `Expect: ExpectDecline`, `Run: func(x *Arrays) { s114(x.AA, x.BB) }`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("generated output missing %q, got:\n%s", want, got)
		}
	}
}
