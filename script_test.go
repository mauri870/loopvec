package main_test

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/txtar"
	"rsc.io/script"
	"rsc.io/script/scripttest"
)

// update regenerates the testdata/*.txt sections that cmp/cmpenv compare
// against (stderr.want, ops_simd.go.want, ...) from the script's actual
// output instead of failing the test on a mismatch. Wired into "make
// test-update".
var update = flag.Bool("update", false, "regenerate testdata/*.txt golden sections")

// TestScripts builds the real loopvec binary and drives it through the
// scripts in testdata/*.txt. Each script runs in its own temporary module
// containing the files from the script's txtar section.
func TestScripts(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "loopvec")
	build := exec.Command("go", "build", "-o", binary, ".")
	build.Env = testEnv(t)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building loopvec: %v\n%s", err, out)
	}

	toolexec := filepath.Join(t.TempDir(), "loopvec-toolexec")
	buildToolexec := exec.Command("go", "build", "-o", toolexec, "./cmd/loopvec-toolexec")
	buildToolexec.Env = testEnv(t)
	if out, err := buildToolexec.CombinedOutput(); err != nil {
		t.Fatalf("building loopvec-toolexec: %v\n%s", err, out)
	}

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// LOOPVEC_SRC lets scripts point a Go workspace at this checkout so that
	// "go tool loopvec" resolves without a published release.
	// LOOPVEC_TOOLEXEC is the built -toolexec wrapper.
	env := append(testEnv(t), "LOOPVEC_SRC="+wd, "LOOPVEC_TOOLEXEC="+toolexec)

	if *update {
		updateScripts(t, env, binary, toolexec, "testdata/*.txt")
		return
	}

	cmds := scripttest.DefaultCmds()
	cmds["loopvec"] = script.Program(binary, nil, 0)
	cmds["go"] = script.Program("go", nil, 0)
	cmds["gotip"] = script.Program("gotip", nil, 0)

	engine := &script.Engine{
		Conds: testConds(env),
		Cmds:  cmds,
		Quiet: !testing.Verbose(),
	}
	scripttest.Test(t, context.Background(), engine, env, "testdata/*.txt")
}

// updateScripts reruns every script matching pattern with cmp/cmpenv
// replaced by goldenRecorder.cmd, which never fails and instead records what
// the golden section (the cmp/cmpenv second argument, e.g. "ops_simd.go.want"
// or "stderr.want") should now contain. Once a script finishes, the recorded
// sections are spliced back into its txtar archive and the file is rewritten
// in place.
func updateScripts(t *testing.T, env []string, binary, toolexec, pattern string) {
	files, err := filepath.Glob(pattern)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no testdata")
	}
	for _, file := range files {
		t.Run(strings.TrimSuffix(filepath.Base(file), ".txt"), func(t *testing.T) {
			rec := &goldenRecorder{sections: map[string]string{}}

			cmds := scripttest.DefaultCmds()
			cmds["loopvec"] = script.Program(binary, nil, 0)
			cmds["go"] = script.Program("go", nil, 0)
			cmds["gotip"] = script.Program("gotip", nil, 0)
			cmds["cmp"] = rec.cmd(false)
			cmds["cmpenv"] = rec.cmd(true)
			// scripttest.DefaultCmds's "skip" returns an unexported error type
			// that only scripttest.Test knows how to turn into t.Skip. Replace
			// it with a local skip command carrying skipUpdateError instead.
			cmds["skip"] = skipUpdateCmd()

			engine := &script.Engine{
				Conds: testConds(env),
				Cmds:  cmds,
				Quiet: !testing.Verbose(),
			}

			a, err := txtar.ParseFile(file)
			if err != nil {
				t.Fatal(err)
			}

			s, err := script.NewState(context.Background(), t.TempDir(), env)
			if err != nil {
				t.Fatal(err)
			}
			initUpdateScriptDirs(t, s)
			if err := s.ExtractFiles(a); err != nil {
				t.Fatal(err)
			}

			log := new(strings.Builder)
			runErr := engine.Execute(s, file, bufio.NewReader(bytes.NewReader(a.Comment)), log)
			if closeErr := s.CloseAndWait(log); runErr == nil {
				runErr = closeErr
			}
			if log.Len() > 0 {
				t.Log(strings.TrimSuffix(log.String(), "\n"))
			}
			if skip, ok := errors.AsType[skipUpdateError](runErr); ok {
				if skip.msg == "" {
					t.Skip("SKIP")
				} else {
					t.Skipf("SKIP: %v", skip.msg)
				}
			}
			if runErr != nil {
				t.Fatalf("running script: %v", runErr)
			}

			if rec.apply(a) {
				if err := os.WriteFile(file, txtar.Format(a), 0o644); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

// skipUpdateError a local equivalent of scripttest's unexported skipError, so
// a "skip" command's error can be recognized with errors.As from this package.
type skipUpdateError struct{ msg string }

func (s skipUpdateError) Error() string {
	if s.msg == "" {
		return "skip"
	}
	return s.msg
}

// skipUpdateCmd mirrors scripttest.Skip, returning skipUpdateError instead
// of scripttest's unexported skipError.
func skipUpdateCmd() script.Cmd {
	return script.Command(
		script.CmdUsage{
			Summary: "skip the current test",
			Args:    "[msg]",
		},
		func(_ *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) > 1 {
				return nil, script.ErrUsage
			}
			if len(args) == 0 {
				return nil, skipUpdateError{""}
			}
			return nil, skipUpdateError{args[0]}
		})
}

// initUpdateScriptDirs mirrors the unexported setup scripttest.Test performs
// (WORK and a per-OS temp dir env var) so scripts behave identically under
// -update.
func initUpdateScriptDirs(t *testing.T, s *script.State) {
	t.Helper()
	work := s.Getwd()
	if err := s.Setenv("WORK", work); err != nil {
		t.Fatal(err)
	}
	tmp := filepath.Join(work, "tmp")
	if err := os.MkdirAll(tmp, 0o777); err != nil {
		t.Fatal(err)
	}
	tmpVar := "TMPDIR"
	switch runtime.GOOS {
	case "windows":
		tmpVar = "TMP"
	}
	if err := s.Setenv(tmpVar, tmp); err != nil {
		t.Fatal(err)
	}
}

// goldenRecorder implements cmp/cmpenv for -update: instead of failing on a
// mismatch, it records what the golden (second) argument's txtar section
// should now contain.
type goldenRecorder struct {
	sections map[string]string
}

// envRefPattern matches the $VAR references os.Expand recognizes, used to
// find which environment variables a golden file already references.
var envRefPattern = regexp.MustCompile(`\$([A-Za-z_][A-Za-z0-9_]*)`)

func (r *goldenRecorder) cmd(env bool) script.Cmd {
	return script.Command(
		script.CmdUsage{
			Args:    "[-q] file1 file2",
			Summary: "record golden file2 from actual file1 (testing -update only)",
		},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) > 0 && args[0] == "-q" {
				args = args[1:]
			}
			if len(args) != 2 {
				return nil, script.ErrUsage
			}
			name1, name2 := args[0], args[1]

			var actual string
			switch name1 {
			case "stdout":
				actual = s.Stdout()
			case "stderr":
				actual = s.Stderr()
			default:
				data, err := os.ReadFile(s.Path(name1))
				if err != nil {
					return nil, err
				}
				actual = string(data)
			}

			want, err := os.ReadFile(s.Path(name2))
			if err != nil {
				return nil, err
			}

			newWant := actual
			if env {
				newWant = reverseExpandEnv(s, string(want), actual)
			}
			r.sections[name2] = newWant
			return nil, nil
		})
}

// reverseExpandEnv inverts the substitution cmpenv applies to the golden
// text before comparing: for every $VAR already referenced in origWant, it
// replaces VAR's current resolved value inside actual with the literal
// "$VAR" placeholder, so the regenerated golden still expands correctly
// against a fresh $WORK on the next run.
func reverseExpandEnv(s *script.State, origWant, actual string) string {
	seen := map[string]bool{}
	var names []string
	for _, m := range envRefPattern.FindAllStringSubmatch(origWant, -1) {
		if !seen[m[1]] {
			seen[m[1]] = true
			names = append(names, m[1])
		}
	}
	sort.Slice(names, func(i, j int) bool {
		vi, _ := s.LookupEnv(names[i])
		vj, _ := s.LookupEnv(names[j])
		return len(vi) > len(vj)
	})
	for _, name := range names {
		val, ok := s.LookupEnv(name)
		if !ok || val == "" {
			continue
		}
		actual = strings.ReplaceAll(actual, val, "$"+name)
	}
	return actual
}

// apply splices the recorded sections into a and reports whether anything
// changed.
func (r *goldenRecorder) apply(a *txtar.Archive) bool {
	changed := false
	for i := range a.Files {
		content, ok := r.sections[a.Files[i].Name]
		if !ok {
			continue
		}
		if data := []byte(content); !bytes.Equal(a.Files[i].Data, data) {
			a.Files[i].Data = data
			changed = true
		}
	}
	return changed
}

// testConds is scripttest.DefaultConds plus [tip], active when the go command the
// scripts run is a development toolchain. A few behaviors differ there on purpose
// (the method crash of go1.27.1 is fixed on tip), and those scripts guard the
// toolchain-specific part with it.
func testConds(env []string) map[string]script.Cond {
	conds := scripttest.DefaultConds()
	cmd := exec.Command("go", "version")
	cmd.Env = env
	out, _ := cmd.Output()
	conds["tip"] = script.BoolCondition("the go command is a development toolchain", strings.Contains(string(out), "devel"))
	return conds
}

// testEnv returns the process environment without GOEXPERIMENT (scripts opt in
// with "env GOEXPERIMENT=simd") and with GOTOOLCHAIN pinned to the toolchain
// declared in go.mod, regardless of the caller's setting, except that
// GOTOOLCHAIN=local is kept: the job that tests a development toolchain puts it
// first on PATH and must not have it replaced by the pinned release.
func testEnv(t *testing.T) []string {
	t.Helper()
	var env []string
	for _, kv := range os.Environ() {
		if strings.HasPrefix(kv, "GOEXPERIMENT=") || strings.HasPrefix(kv, "GOTOOLCHAIN=") {
			continue
		}
		env = append(env, kv)
	}
	if os.Getenv("GOTOOLCHAIN") == "local" {
		return append(env, "GOTOOLCHAIN=local")
	}
	return append(env, "GOTOOLCHAIN="+goModToolchain(t))
}

// goModToolchain returns the "toolchain" directive of go.mod, or "go" plus the
// "go" directive when there is none.
func goModToolchain(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatal(err)
	}
	var goVersion string
	for line := range strings.SplitSeq(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		switch fields[0] {
		case "toolchain":
			return fields[1]
		case "go":
			goVersion = fields[1]
		}
	}
	if goVersion == "" {
		t.Fatal("go.mod has neither a toolchain nor a go directive")
	}
	return "go" + goVersion
}
