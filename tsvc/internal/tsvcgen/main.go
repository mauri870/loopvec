// Command tsvcgen generates the Kernels registry for package tsvc from the
// //tsvc:kernel directives in its source files.
//
// Usage:
//
//	tsvcgen -o zz_registry.go kernels.go
package main

import (
	"flag"
	"fmt"
	"go/token"
	"os"
)

var out = flag.String("o", "", "output file")

func main() {
	flag.Parse()
	if *out == "" || flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: tsvcgen -o file.go sources...")
		os.Exit(2)
	}

	if err := run(*out, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "tsvcgen:", err)
		os.Exit(1)
	}
}

func run(out string, sources []string) error {
	fset := token.NewFileSet()
	var directives []directive
	for _, name := range sources {
		src, err := os.ReadFile(name)
		if err != nil {
			return err
		}
		ds, err := parseDirectives(fset, name, src)
		if err != nil {
			return err
		}
		directives = append(directives, ds...)
	}

	data, err := generate(directives)
	if err != nil {
		return err
	}
	return os.WriteFile(out, data, 0o644)
}
