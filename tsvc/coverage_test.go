package tsvc

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// jsonCandidate mirrors loopvec's -json line format: file, line, func,
// whether it was vectorized, and (if not) why.
type jsonCandidate struct {
	File       string `json:"file"`
	Line       int    `json:"line"`
	Func       string `json:"func"`
	Vectorized bool   `json:"vectorized"`
	Reason     string `json:"reason,omitempty"`
}

// coverageResult is what TestCoverage records for one kernel.
type coverageResult struct {
	Vectorized bool
	Reason     string
}

const notRecognized = "not recognized"

// TestCoverage builds the real loopvec binary and runs -json against this
// package, then checks the result against testdata/coverage.txt: which of the
// ported kernels loopvec actually rewrites today, and why not for the rest.
// Run with -update to record a deliberate change (a fix that vectorizes a
// kernel, or a change that no longer does).
//
// Each kernel carries an expectation. A kernel expected to be vectorized that
// is not is a gap; one that must be declined or skipped and is left alone is
// correct. A kernel that must be declined or skipped and is vectorized fails
// the test even under -update: it is a bug, not an improvement.
func TestCoverage(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "loopvec")
	build := exec.Command("go", "build", "-o", binary, "github.com/mauri870/loopvec")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building loopvec: %v\n%s", err, out)
	}

	cmd := exec.Command(binary, "-json", ".")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("loopvec -json .: %v\n%s", err, stderr.String())
	}

	results := make(map[string]coverageResult)
	scanner := bufio.NewScanner(&stdout)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var c jsonCandidate
		if err := json.Unmarshal(line, &c); err != nil {
			t.Fatalf("parsing -json line %q: %v", line, err)
		}
		if existing, ok := results[c.Func]; ok && existing.Vectorized {
			continue // keep the first vectorized candidate for a function
		}
		results[c.Func] = coverageResult{Vectorized: c.Vectorized, Reason: c.Reason}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("reading loopvec -json output: %v", err)
	}

	got := make(map[string]coverageResult, len(Kernels))
	for _, k := range Kernels {
		r, ok := results[k.Name]
		if !ok {
			r = coverageResult{Reason: notRecognized}
		}
		got[k.Name] = r
	}

	for _, k := range Kernels {
		if k.Expect != ExpectVectorize && got[k.Name].Vectorized {
			t.Fatalf("%s is expected to be left alone (expect=%s) but is vectorized; if that is intended, change its expect directive in kernels.go", k.Name, k.Expect)
		}
	}
	t.Logf("%s", summary(got))

	const path = "testdata/coverage.txt"
	if *update {
		if err := os.WriteFile(path, renderCoverage(got), 0o644); err != nil {
			t.Fatalf("writing %s: %v", path, err)
		}
		return
	}

	want, err := readCoverage(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}

	for _, k := range Kernels {
		g, w := got[k.Name], want[k.Name]
		switch {
		case w.Vectorized && !g.Vectorized:
			t.Errorf("regression: %s was vectorized, now isn't (%s)", k.Name, g.Reason)
		case !w.Vectorized && g.Vectorized:
			t.Errorf("improvement: %s is now vectorized; run 'make tsvc-coverage-update' to record it", k.Name)
		case !g.Vectorized && g.Reason != w.Reason:
			t.Logf("%s: reason changed: %q -> %q", k.Name, w.Reason, g.Reason)
		}
	}
}

// summary counts the kernels by outcome: vectorized, a gap (expected to be
// vectorized but not), and left alone as expected (declined or skipped).
func summary(results map[string]coverageResult) string {
	var vectorized, gaps, declined, skipped int
	for _, k := range Kernels {
		switch {
		case results[k.Name].Vectorized:
			vectorized++
		case k.Expect == ExpectVectorize:
			gaps++
		case k.Expect == ExpectDecline:
			declined++
		default:
			skipped++
		}
	}
	return fmt.Sprintf("vectorized %d, gaps %d, declined %d, skipped %d (of %d)",
		vectorized, gaps, declined, skipped, len(Kernels))
}

// renderCoverage renders results in tsvc/kernels.go source order (matching
// every other table in this package), not alphabetically: "s112"/"s1112"
// stay adjacent, which lexicographic order would split up. The expectation is
// copied from the kernel's directive so the file reads on its own.
func renderCoverage(results map[string]coverageResult) []byte {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# loopvec TSVC coverage: %s\n", summary(results))
	for _, k := range Kernels {
		r := results[k.Name]
		if r.Vectorized {
			fmt.Fprintf(&buf, "%s\t%s\tyes\n", k.Name, k.Expect)
			continue
		}
		fmt.Fprintf(&buf, "%s\t%s\tno\t%s\n", k.Name, k.Expect, r.Reason)
	}
	return buf.Bytes()
}

// readCoverage parses the format renderCoverage produces.
func readCoverage(path string) (map[string]coverageResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	results := make(map[string]coverageResult)
	for line := range strings.Lines(string(data)) {
		line = strings.TrimRight(line, "\n")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.SplitN(line, "\t", 4)
		if len(fields) < 3 {
			return nil, fmt.Errorf("malformed coverage line %q", line)
		}
		r := coverageResult{Vectorized: fields[2] == "yes"}
		if len(fields) == 4 {
			r.Reason = fields[3]
		}
		results[fields[0]] = r
	}
	return results, nil
}
