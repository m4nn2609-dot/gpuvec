.PHONY: all build test bench clean

all: build

build:
	go build -o bin/bench ./cmd/bench

test:
	go test -v -race ./...

bench: build
	@mkdir -p data results
	./bin/bench -data ./data -out ./results -name bruteforce_l2 -k 10 -nq 1000 -warmup 3 -runs 5 -batch 1 -metric L2

bench-cosine: build
	@mkdir -p data results
	./bin/bench -data ./data -out ./results -name bruteforce_cosine -k 10 -nq 1000 -warmup 3 -runs 5 -batch 1 -metric Cosine

bench-batch: build
	@mkdir -p data results
	./bin/bench -data ./data -out ./results -name bruteforce_batch32 -k 10 -nq 1000 -warmup 3 -runs 5 -batch 32 -metric L2

bench-openloop: build
	@mkdir -p data results
	./bin/bench -data ./data -out ./results -name bruteforce_openloop -k 10 -nq 1000 -openloop -rate 500 -duration 10s -metric L2

clean:
	rm -rf bin results data

# Download dataset only
download:
	@mkdir -p data
	go run ./cmd/bench -data ./data -nq 0 2>&1 | head -20