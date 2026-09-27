export GOTOOLCHAIN := go1.27.1

.DEFAULT_GOAL := build

.PHONY: build install test fix fmt lint ci bench bench-regen bench-fuzz fuzz-slow generate tsvc-test tsvc-test-qemu-arm64 tsvc-update tsvc-coverage-update tsvc-bench test-update

test: build
	go test ./... -count=1 -race

bench-fuzz:
	GOEXPERIMENT=simd go test ./bench/ -count=1 -run Fuzz

fuzz-slow:
	GOEXPERIMENT=simd go test ./bench/ -run '^$$' -fuzz=FuzzAddFloat32s -fuzztime=60s
	GOEXPERIMENT=simd go test ./bench/ -run '^$$' -fuzz=FuzzNegFloat32s -fuzztime=60s
	GOEXPERIMENT=simd go test ./bench/ -run '^$$' -fuzz=FuzzReverseIncFloat32s -fuzztime=60s
	GOEXPERIMENT=simd go test ./bench/ -run '^$$' -fuzz=FuzzDaxpyFloat32s -fuzztime=60s

fix:
	go fix ./...
	$(MAKE) fmt

fmt:
	go fmt ./...
	go vet ./...

lint:
	go fix -diff ./...
	go tool golangci-lint run ./...
	go tool deadcode $(shell go list ./... | grep -vE '/bench$$|/tsvc$$')

build:
	go build ./...
	go build -o bin/ . ./cmd/loopvec-toolexec

install:
	go install . ./cmd/loopvec-toolexec

ci: test-update test bench-fuzz tsvc-test tsvc-test-qemu-arm64
	$(MAKE) fmt
	git diff --exit-code
	$(MAKE) lint

bench:
	go test -bench=. -benchtime=100ms -count=10 ./bench/ > /tmp/bench_scalar.txt
	GOEXPERIMENT=simd go test -bench=. -benchtime=100ms -count=10 ./bench/ > /tmp/bench_simd.txt
	benchstat /tmp/bench_scalar.txt /tmp/bench_simd.txt

bench-regen: build
	go run . -split ./bench/

generate:
	go generate ./tsvc/...

tsvc-test: generate build
	go test -count=1 ./tsvc/...
	GOEXPERIMENT=simd go test -toolexec="$(CURDIR)/bin/loopvec-toolexec" -run TestGolden ./tsvc/

tsvc-test-qemu-arm64: generate build
	GOARCH=arm64 GOOS=linux CGO_ENABLED=0 go test -exec=qemu-aarch64-static ./tsvc/ -run TestGolden
	GOARCH=arm64 GOOS=linux CGO_ENABLED=0 GOEXPERIMENT=simd go test -toolexec="$(CURDIR)/bin/loopvec-toolexec" -exec=qemu-aarch64-static -run TestGolden ./tsvc/

tsvc-update: generate
	go test ./tsvc/ -run TestGolden -update
	GOARCH=arm64 GOOS=linux CGO_ENABLED=0 GOEXPERIMENT=simd go test -toolexec="$(CURDIR)/bin/loopvec-toolexec" -exec=qemu-aarch64-static ./tsvc -run TestGolden -update

# tsvc-coverage-update is deliberately separate from tsvc-update to catch coverage regressions
tsvc-coverage-update: generate
	go test -count=1 ./tsvc/ -run TestCoverage -update

tsvc-bench: generate build
	go test -run '^$$' -bench . -count=10 ./tsvc/ > /tmp/tsvc_bench_scalar.txt
	GOEXPERIMENT=simd go test -run '^$$' -bench . -count=10 -toolexec="$(CURDIR)/bin/loopvec-toolexec" ./tsvc/ > /tmp/tsvc_bench_simd.txt
	benchstat /tmp/tsvc_bench_scalar.txt /tmp/tsvc_bench_simd.txt

test-update: build
	go test . -run TestScripts -update
	$(MAKE) tsvc-update tsvc-coverage-update
