package main

import (
	"go/token"
	"strings"
	"testing"
)

func TestParseDirectivesErrors(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want string
	}{
		{
			name: "unknown key",
			src: `package tsvc

//tsvc:kernel category=dependence reps=2 setup=s000 checksum=a exact=bits bogus=1
func s000(a, b []float32) {}
`,
			want: `unknown directive key "bogus"`,
		},
		{
			name: "duplicate name",
			src: `package tsvc

//tsvc:kernel category=dependence category=dependence reps=2 setup=s000 checksum=a exact=bits
func s000(a, b []float32) {}
`,
			want: `duplicate directive key "category"`,
		},
		{
			name: "missing key",
			src: `package tsvc

//tsvc:kernel category=dependence setup=s000 checksum=a exact=bits
func s000(a, b []float32) {}
`,
			want: `missing directive key "reps"`,
		},
		{
			name: "malformed token",
			src: `package tsvc

//tsvc:kernel category reps=2 setup=s000 checksum=a exact=bits
func s000(a, b []float32) {}
`,
			want: `malformed directive token "category"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fset := token.NewFileSet()
			_, err := parseDirectives(fset, "kernels.go", []byte(tt.src))
			if err == nil {
				t.Fatalf("parseDirectives: want error containing %q, got nil", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("parseDirectives error = %q, want to contain %q", err, tt.want)
			}
		})
	}
}

func TestParseDirectivesOK(t *testing.T) {
	src := `package tsvc

// s000: linear dependence testing, no dependence.
//
//tsvc:kernel category=dependence reps=2 setup=s000 checksum=a exact=bits
func s000(a, b []float32) {}
`
	fset := token.NewFileSet()
	ds, err := parseDirectives(fset, "kernels.go", []byte(src))
	if err != nil {
		t.Fatalf("parseDirectives: %v", err)
	}
	if len(ds) != 1 {
		t.Fatalf("len(ds) = %d, want 1", len(ds))
	}
	d := ds[0]
	if d.FuncName != "s000" {
		t.Errorf("FuncName = %q, want s000", d.FuncName)
	}
	if got, want := d.Params, []string{"a", "b"}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("Params = %v, want %v", got, want)
	}
	if d.Category != "dependence" || d.Reps != 2 || d.Setup != "s000" || d.Checksum != "a" || d.Exact != "bits" {
		t.Errorf("directive = %+v, want the fields parsed from the source", d)
	}
}
