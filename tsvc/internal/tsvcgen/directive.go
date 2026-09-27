package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"slices"
	"strconv"
	"strings"
)

// directivePrefix marks a kernel directive comment, formatted like a
// //go:generate line so gofmt leaves it alone and it doesn't show in godoc.
const directivePrefix = "//tsvc:kernel"

// directiveKeys are the only keys a //tsvc:kernel comment may set, all of
// which are required.
var directiveKeys = []string{"category", "reps", "setup", "checksum", "exact"}

// directive holds one parsed //tsvc:kernel comment together with the
// signature of the function it annotates.
type directive struct {
	FuncName string
	Params   []string
	Category string
	Reps     int
	Setup    string
	Checksum string
	Exact    string
	Pos      token.Position
}

// parseDirectives collects every //tsvc:kernel directive in src.
func parseDirectives(fset *token.FileSet, filename string, src []byte) ([]directive, error) {
	file, err := parser.ParseFile(fset, filename, src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	var directives []directive
	for _, decl := range file.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Doc == nil {
			continue
		}
		for _, c := range fd.Doc.List {
			text, ok := strings.CutPrefix(c.Text, directivePrefix)
			if !ok {
				continue
			}
			d, err := parseDirectiveLine(strings.TrimSpace(text), fset.Position(c.Pos()))
			if err != nil {
				return nil, err
			}
			d.FuncName = fd.Name.Name
			d.Params = paramNames(fd.Type.Params)
			directives = append(directives, d)
			break
		}
	}
	return directives, nil
}

func paramNames(fl *ast.FieldList) []string {
	var names []string
	if fl == nil {
		return names
	}
	for _, f := range fl.List {
		for _, n := range f.Names {
			names = append(names, n.Name)
		}
	}
	return names
}

func parseDirectiveLine(text string, pos token.Position) (directive, error) {
	fields := make(map[string]string)
	for _, tok := range strings.Fields(text) {
		key, value, ok := strings.Cut(tok, "=")
		if !ok {
			return directive{}, fmt.Errorf("%s: malformed directive token %q", pos, tok)
		}
		if !slices.Contains(directiveKeys, key) {
			return directive{}, fmt.Errorf("%s: unknown directive key %q", pos, key)
		}
		if _, dup := fields[key]; dup {
			return directive{}, fmt.Errorf("%s: duplicate directive key %q", pos, key)
		}
		fields[key] = value
	}
	for _, key := range directiveKeys {
		if _, ok := fields[key]; !ok {
			return directive{}, fmt.Errorf("%s: missing directive key %q", pos, key)
		}
	}

	reps, err := strconv.Atoi(fields["reps"])
	if err != nil {
		return directive{}, fmt.Errorf("%s: invalid reps %q: %w", pos, fields["reps"], err)
	}

	return directive{
		Category: fields["category"],
		Reps:     reps,
		Setup:    fields["setup"],
		Checksum: fields["checksum"],
		Exact:    fields["exact"],
		Pos:      pos,
	}, nil
}
