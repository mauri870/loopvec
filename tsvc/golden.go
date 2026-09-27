package tsvc

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
)

// Result is one kernel's checksum, as stored in a golden file.
type Result struct {
	Sum  float64 `json:"sum"`
	Hash uint64  `json:"hash"`
}

// goldenPath returns the golden file for the running architecture. Goldens
// are per-GOARCH because the scalar build already differs between
// architectures: the Go compiler fuses x*y+z into an FMA on arm64 but not on
// baseline amd64.
func goldenPath() string {
	return filepath.Join("testdata", "golden_"+runtime.GOARCH+".json")
}

func readGolden(path string) (map[string]Result, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	golden := make(map[string]Result)
	if err := json.Unmarshal(data, &golden); err != nil {
		return nil, err
	}
	return golden, nil
}

func writeGolden(path string, golden map[string]Result) error {
	data, err := json.MarshalIndent(golden, "", "\t")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}
