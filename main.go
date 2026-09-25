// Command loopvec analyzes Go source files and rewrites element-wise loops
// to use the simd package for hardware vectorization.
//
// Usage:
//
//	loopvec [-split | -w | -d] [packages...]
//
// Without flags, loopvec prints the rewritten source to stdout.
// With -split, loopvec writes the simd variant to file_simd.go and adds
// //go:build !goexperiment.simd to the original file.
// With -w, loopvec writes changes back to the source files in place.
// With -d, loopvec prints a unified diff for each changed file.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"go/token"
	"os"
	"os/exec"
	"strings"

	"golang.org/x/tools/go/packages"

	"github.com/mauri870/loopvec/internal/rewrite"
)

var (
	writeBack = flag.Bool("w", false, "write result to source files in place")
	splitMode = flag.Bool("split", false, "write simd variant to file_simd.go and add //go:build !goexperiment.simd to original")
	diffMode  = flag.Bool("d", false, "display unified diff instead of rewritten source")
)

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

		switch {
		case *splitMode:
			simdPath := strings.TrimSuffix(path, ".go") + "_simd.go"
			if err := os.WriteFile(simdPath, result.Src, 0o644); err != nil {
				return err
			}
			tagged := addBuildTag(src, "!goexperiment.simd")
			if err := os.WriteFile(path, tagged, 0o644); err != nil {
				return err
			}
		case *writeBack:
			if err := os.WriteFile(path, result.Src, 0o644); err != nil {
				return err
			}
		case *diffMode:
			if err := printDiff(path, src, result.Src); err != nil {
				return err
			}
		default:
			if _, err := os.Stdout.Write(result.Src); err != nil {
				return err
			}
		}
	}
	return nil
}

// addBuildTag inserts a //go:build constraint into src if not already present.
// format.Source moves it to the correct position (before the package clause).
func addBuildTag(src []byte, tag string) []byte {
	constraint := "//go:build " + tag
	s := string(src)
	if strings.Contains(s, constraint) {
		return src
	}
	idx := strings.Index(s, "package ")
	if idx < 0 {
		return src
	}
	nl := strings.Index(s[idx:], "\n")
	if nl < 0 {
		return src
	}
	insertAt := idx + nl + 1
	modified := s[:insertAt] + "\n" + constraint + "\n" + s[insertAt:]
	formatted, err := format.Source([]byte(modified))
	if err != nil {
		return []byte(modified)
	}
	return formatted
}

// printDiff prints a unified diff between orig and rewritten for path,
// using the system diff command.
func printDiff(path string, orig, rewritten []byte) error {
	origFile, err := os.CreateTemp("", "loopvec-orig-*.go")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(origFile.Name()) }()
	if _, err := origFile.Write(orig); err != nil {
		_ = origFile.Close()
		return err
	}
	_ = origFile.Close()

	newFile, err := os.CreateTemp("", "loopvec-new-*.go")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(newFile.Name()) }()
	if _, err := newFile.Write(rewritten); err != nil {
		_ = newFile.Close()
		return err
	}
	_ = newFile.Close()

	out, _ := exec.Command("diff", "-u", origFile.Name(), newFile.Name()).Output()
	out = bytes.ReplaceAll(out, []byte(origFile.Name()), []byte(path))
	out = bytes.ReplaceAll(out, []byte(newFile.Name()), []byte(path))
	_, err = os.Stdout.Write(out)
	return err
}
