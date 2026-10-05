package main

import (
	"bufio"
	"bytes"
	"fmt"
	"go/token"
	"go/types"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/gcexportdata"
)

// privateExportData reports whether the archive at file holds the compiler's
// private export data format. The compiler writes it for its own use and the
// standard library's importer refuses to read it, so a wrapper that
// type-checks a package cannot use the archives the go command passes to the
// compiler and needs the export data the go command writes for tools.
func privateExportData(file string) bool {
	f, err := os.Open(file)
	if err != nil {
		return false
	}
	defer func() { _ = f.Close() }()
	buf := make([]byte, 32<<10)
	n, _ := io.ReadFull(f, buf)
	buf = buf[:n]
	i := bytes.Index(buf, []byte("\n$$B\n"))
	return i >= 0 && i+5 < len(buf) && buf[i+5] == 'p'
}

// publicExports returns, for each of paths, a file with export data that
// tools can read: what go list -export writes. They are found with one go
// command run in dir (the package being compiled, so its module's dependencies
// resolve) for the paths no earlier wrapper process of this build looked up.
func (b *build) publicExports(dir string, paths []string) (map[string]string, error) {
	cache := filepath.Join(b.dir, "public")
	known, err := b.readPublic(cache)
	if err != nil {
		return nil, err
	}
	var missing []string
	for _, p := range paths {
		if _, ok := known[p]; !ok {
			missing = append(missing, p)
		}
	}
	if len(missing) == 0 {
		return known, nil
	}
	args := append([]string{"list", "-e", "-export", "-f", "{{.ImportPath}}={{.Export}}", "--"}, missing...)
	out, err := b.goCommandIn(dir, args...)
	if err != nil {
		return nil, err
	}
	var lines []string
	for line := range strings.SplitSeq(string(out), "\n") {
		path, file, ok := strings.Cut(strings.TrimSpace(line), "=")
		if ok && file != "" {
			known[path] = file
			lines = append(lines, path+"="+file+"\n")
		}
	}
	// Another process may have looked up the same paths; a repeated line is harmless.
	unlock, err := lockFile(cache + ".lock")
	if err != nil {
		return known, nil
	}
	defer unlock()
	if f, err := os.OpenFile(cache, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
		_, _ = f.WriteString(strings.Join(lines, ""))
		_ = f.Close()
	}
	return known, nil
}

func (b *build) readPublic(cache string) (map[string]string, error) {
	known := map[string]string{}
	unlock, err := lockFile(cache + ".lock")
	if err != nil {
		return nil, err
	}
	defer unlock()
	data, err := os.ReadFile(cache)
	if err != nil {
		return known, nil
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		if path, file, ok := strings.Cut(line, "="); ok {
			known[path] = file
		}
	}
	return known, nil
}

// exportImporter type-checks imports from export data files of the format go
// list -export writes.
type exportImporter struct {
	fset      *token.FileSet
	files     map[string]string
	importMap map[string]string
	pkgs      map[string]*types.Package
}

func (i *exportImporter) Import(path string) (*types.Package, error) {
	return i.ImportFrom(path, "", 0)
}

func (i *exportImporter) ImportFrom(path, _ string, _ types.ImportMode) (*types.Package, error) {
	if path == "unsafe" {
		return types.Unsafe, nil
	}
	if mapped, ok := i.importMap[path]; ok {
		path = mapped
	}
	if pkg := i.pkgs[path]; pkg != nil && pkg.Complete() {
		return pkg, nil
	}
	file, ok := i.files[path]
	if !ok {
		return nil, fmt.Errorf("no export data for %q", path)
	}
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return gcexportdata.Read(bufio.NewReader(f), i.fset, i.pkgs, path)
}

// failingImporter reports why export data could not be found.
type failingImporter struct{ err error }

func (f failingImporter) Import(string) (*types.Package, error) { return nil, f.err }

// paths returns the packages cfg provides, sorted.
func (c *importcfg) paths() []string {
	paths := make([]string, 0, len(c.packageFile))
	for p := range c.packageFile {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths
}

// usesPrivateExportData reports whether any archive cfg names is in the
// compiler's private format. The go command builds them all with one compiler,
// so one is enough; files that are not archives (the export data tools read) are
// skipped.
func (c *importcfg) usesPrivateExportData() bool {
	for _, p := range c.paths() {
		if privateExportData(c.packageFile[p]) {
			return true
		}
	}
	return false
}

// publicImporter imports from tool-readable export data found through the go
// command.
func (c *importcfg) publicImporter(fset *token.FileSet) types.Importer {
	files, err := c.build.publicExports(c.dir, c.paths())
	if err != nil {
		return failingImporter{fmt.Errorf("export data for tools: %w", err)}
	}
	return &exportImporter{fset: fset, files: files, importMap: c.importMap, pkgs: map[string]*types.Package{}}
}
