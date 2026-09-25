export GOTOOLCHAIN := go1.27.1

.DEFAULT_GOAL := build

.PHONY: build test fix fmt lint ci

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
	go tool deadcode ./...

build:
	go build ./...

ci: test
	$(MAKE) fmt
	git diff --exit-code
