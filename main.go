// Command loopvec analyzes Go source files and rewrites element-wise loops
// to use the simd package for hardware vectorization.
//
// Usage:
//
//	loopvec [-w] [packages...]
//
// Without -w, loopvec prints the rewritten source to stdout.
// With -w, loopvec writes changes back to the source files.
package main

import (
	"flag"
	"fmt"
	"go/token"
	"os"

	"golang.org/x/tools/go/packages"

	"github.com/mauri870/loopvec/internal/rewrite"
)

var writeBack = flag.Bool("w", false, "write result to source files")

func main() {
	flag.Parse()
	patterns := flag.Args()
	if len(patterns) == 0 {
		patterns = []string{"."}
	}

	if err := run(patterns); err != nil {
		fmt.Fprintln(os.Stderr, "loopvec:", err)
		os.Exit(1)
	}
}

func run(patterns []string) error {
	fset := token.NewFileSet()
	cfg := &packages.Config{
		Mode: packages.NeedSyntax |
			packages.NeedTypesInfo |
			packages.NeedTypes |
			packages.NeedFiles |
			packages.NeedImports,
		Fset: fset,
	}
	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}

	hasErr := false
	for _, pkg := range pkgs {
		if packages.PrintErrors([]*packages.Package{pkg}) > 0 {
			hasErr = true
			continue
		}
		if err := processPkg(fset, pkg); err != nil {
			fmt.Fprintf(os.Stderr, "loopvec: %s: %v\n", pkg.ID, err)
			hasErr = true
		}
	}
	if hasErr {
		return fmt.Errorf("errors occurred")
	}
	return nil
}

func processPkg(fset *token.FileSet, pkg *packages.Package) error {
	info := pkg.TypesInfo
	if info == nil {
		return fmt.Errorf("no type info available")
	}

	for _, file := range pkg.Syntax {
		tf := fset.File(file.Pos())
		if tf == nil {
			continue
		}
		path := tf.Name()

		src, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read %s: %w", path, err)
		}

		result, err := rewrite.File(fset, file, info, src)
		if err != nil {
			return fmt.Errorf("rewrite %s: %w", path, err)
		}
		if result.Rewrites == 0 {
			continue
		}

		fmt.Fprintf(os.Stderr, "loopvec: rewrote %d loop(s) in %s\n", result.Rewrites, path)

		if *writeBack {
			if err := os.WriteFile(path, result.Src, 0o644); err != nil {
				return err
			}
		} else {
			if _, err := os.Stdout.Write(result.Src); err != nil {
				return err
			}
		}
	}
	return nil
}
