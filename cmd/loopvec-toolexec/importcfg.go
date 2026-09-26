package main

import (
	"bufio"
	"fmt"
	"go/importer"
	"go/token"
	"go/types"
	"io"
	"maps"
	"os"
	"sort"
	"strings"
)

// importcfg is the parsed form of the file the go command hands to compile and
// link: which archive provides each package, and which import paths are
// remapped (vendoring).
type importcfg struct {
	packageFile map[string]string
	importMap   map[string]string
}

func readImportcfg(path string) (*importcfg, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	cfg := &importcfg{packageFile: map[string]string{}, importMap: map[string]string{}}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(nil, 1<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		verb, rest, _ := strings.Cut(line, " ")
		key, value, ok := strings.Cut(rest, "=")
		if !ok {
			continue
		}
		switch verb {
		case "packagefile":
			cfg.packageFile[key] = value
		case "importmap":
			cfg.importMap[key] = value
		}
	}
	return cfg, scanner.Err()
}

// with returns a copy of cfg that also provides the given packages. Packages
// cfg already provides keep their archive.
func (c *importcfg) with(extra map[string]string) *importcfg {
	out := &importcfg{packageFile: map[string]string{}, importMap: c.importMap}
	maps.Copy(out.packageFile, c.packageFile)
	for k, v := range extra {
		if _, ok := out.packageFile[k]; !ok {
			out.packageFile[k] = v
		}
	}
	return out
}

// writeExtended writes to dst a copy of the importcfg file src, byte for byte,
// with a packagefile line appended for each package of extra that cfg (the
// parsed form of src) does not provide. The file is never re-serialized: the go
// command puts directives in it, such as modinfo, that this program does not
// model and must not lose.
func (c *importcfg) writeExtended(src, dst string, extra map[string]string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	if len(data) > 0 && data[len(data)-1] != '\n' {
		data = append(data, '\n')
	}
	var lines []string
	for path, file := range extra {
		if _, ok := c.packageFile[path]; !ok {
			lines = append(lines, fmt.Sprintf("packagefile %s=%s\n", path, file))
		}
	}
	sort.Strings(lines)
	for _, line := range lines {
		data = append(data, line...)
	}
	return os.WriteFile(dst, data, 0o644)
}

// importer type-checks imports from the export data the archives provide, the
// same data the compiler itself reads.
func (c *importcfg) importer(fset *token.FileSet) types.Importer {
	lookup := func(path string) (io.ReadCloser, error) {
		if mapped, ok := c.importMap[path]; ok {
			path = mapped
		}
		file, ok := c.packageFile[path]
		if !ok {
			return nil, fmt.Errorf("no export data for %q", path)
		}
		return os.Open(file)
	}
	return importer.ForCompiler(fset, "gc", lookup)
}
