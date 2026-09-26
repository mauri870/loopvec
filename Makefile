export GOTOOLCHAIN := go1.27.1

.DEFAULT_GOAL := build

.PHONY: build install test fix fmt lint ci bench bench-regen

test:
	go test ./... -count=1 -race

fix:
	go fix ./...
	$(MAKE) fmt

fmt:
	go fmt ./...
	go vet ./...

lint:
	go fix -diff ./...
	go tool golangci-lint run ./...
	go tool deadcode $(shell go list ./... | grep -v /bench)

build:
	go build ./...
	go build -o bin/ . ./cmd/loopvec-toolexec

install:
	go install . ./cmd/loopvec-toolexec

ci: test
	$(MAKE) fmt
	git diff --exit-code

bench:
	go test -bench=. -count=10 ./bench/ > /tmp/bench_scalar.txt
	GOEXPERIMENT=simd go test -bench=. -count=10 ./bench/ > /tmp/bench_simd.txt
	benchstat /tmp/bench_scalar.txt /tmp/bench_simd.txt

bench-regen: build
	go run . -split ./bench/
