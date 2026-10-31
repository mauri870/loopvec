package main

import "testing"

func TestRelease(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{"go1.27.3", "go1.27.3"},
		{"go1.27.3-X:simd", "go1.27.3"},
		{"go1.27.3-X:simd,fieldtrack", "go1.27.3"},
		{"compile version go1.27.3", "go1.27.3"},
		{"compile version go1.27.3 X:simd\n", "go1.27.3"},
		{"go1.28rc1-X:simd", "go1.28rc1"},
		{"compile version go1.28rc1 X:simd", "go1.28rc1"},
		{"devel go1.28-6ae0b1f8a9 Tue Oct 27 10:00:00 2026 -0700 X:simd", "go1.28-6ae0b1f8a9"},
		{"compile version devel go1.28-6ae0b1f8a9 Tue Oct 27 10:00:00 2026 -0700 X:simd buildID=x/y", "go1.28-6ae0b1f8a9"},
		{"go1.27.3-acme X:simd", "go1.27.3-acme"},
		{"compile version go1.27.3-acme X:simd", "go1.27.3-acme"},
		{"", ""},
		{"devel +6ae0b1f8a9", ""},
	} {
		if got := release(tt.in); got != tt.want {
			t.Errorf("release(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
