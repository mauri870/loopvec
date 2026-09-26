package main

import (
	"fmt"
	"os"
)

// link runs the linker with simd and its dependencies added to the importcfg.
// Rewritten packages import simd, which the go command never planned for, so
// the linker would not otherwise find it.
func link(tool string, args []string) int {
	idx := flagValue(args, "-importcfg")
	if idx < 0 || !simdOn(tool) {
		return execTool(tool, args)
	}
	patched, err := patchLinkImportcfg(tool, args[idx])
	if err != nil {
		logf("", "link: %v", err)
		return execTool(tool, args)
	}
	args = append([]string(nil), args...)
	args[idx] = patched
	return execTool(tool, args)
}

func patchLinkImportcfg(tool, path string) (string, error) {
	b, err := newBuild(tool, path)
	if err != nil {
		return "", err
	}
	exports, err := b.exports()
	if err != nil {
		return "", err
	}
	cfg, err := readImportcfg(path)
	if err != nil {
		return "", err
	}
	// Links run in parallel within one build, so each needs its own file.
	out, err := os.CreateTemp(b.dir, "importcfg.link-")
	if err != nil {
		return "", err
	}
	if err := out.Close(); err != nil {
		return "", err
	}
	patched := out.Name()
	if err := cfg.writeExtended(path, patched, exports); err != nil {
		return "", fmt.Errorf("writing %s: %w", patched, err)
	}
	return patched, nil
}

// flagValue returns the index of the value following flag in args, or -1.
func flagValue(args []string, flag string) int {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return i + 1
		}
	}
	return -1
}
