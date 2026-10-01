// Package analysis detects loops that can be vectorized using the simd package.
package analysis

import (
	"go/ast"
	"go/types"

	"github.com/mauri870/loopvec/internal/loopir"
)

// Options controls what Analyze and AnalyzeExplain accept.
type Options struct {
	// AllowMethods enables loops inside methods; see Analyze.
	AllowMethods bool
	// FloatReassoc allows regrouping a floating-point sum or product; see
	// loopir.Options.
	FloatReassoc bool
}

// Plan returns the planning options o implies.
func (o Options) Plan() loopir.Options {
	return loopir.Options{FloatReassoc: o.FloatReassoc}
}

// Analyze walks a file's AST and returns a plan for every vectorizable loop.
//
// Only loops inside function bodies are considered. A loop in a package-level
// variable initializer, such as var t = func() []byte { ... }(), runs as part
// of the compiler-generated package initializer, where the simd operations are
// not resolved and the program fails to link.
//
// When allowMethods is false, loops inside methods (functions with a receiver)
// are skipped because GOEXPERIMENT=simd has a compiler bug with method bodies
// in simd-tagged files (https://github.com/golang/go/issues/80657).
// Pass allowMethods=true only when using a toolchain that has the fix
// (Go 1.28+ / gotip with CL 839405).
func Analyze(file *ast.File, info *types.Info, opts Options) []*loopir.Plan {
	var plans []*loopir.Plan
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil || (fn.Recv != nil && !opts.AllowMethods) {
			continue
		}
		results := loopir.NamedResults(fn, info)
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if stmt, ok := n.(ast.Stmt); ok {
				if plan, _ := analyzeLoop(stmt, info, results, opts.Plan()); plan != nil {
					plans = append(plans, plan)
				}
			}
			return true
		})
	}
	return plans
}

// analyzeLoop lowers stmt, which must be a for or range statement, and plans
// it. It returns the reason when the loop is not vectorizable.
func analyzeLoop(stmt ast.Stmt, info *types.Info, results map[types.Object]bool, plan loopir.Options) (*loopir.Plan, loopir.Reason) {
	switch stmt.(type) {
	case *ast.RangeStmt, *ast.ForStmt:
	default:
		return nil, ""
	}
	loop, reason := loopir.Lower(stmt, info, results)
	if reason != "" {
		return nil, reason
	}
	return loopir.NewPlan(loop, plan)
}
