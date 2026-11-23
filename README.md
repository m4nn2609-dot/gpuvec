# gpuvec — GPU-Accelerated Vector Search Engine

[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go)](https://golang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A rigorously benchmarked GPU vector search engine written in **Go (control plane) + CUDA C++ (compute plane)**. Designed for honest performance engineering: showing exactly where GPU search wins and loses against CPU baselines, with trustworthy numbers.

> **Status**: Active development — Phase 1 complete, Phase 2 in progress (blocked on CUDA toolkit install)

---

## 🎯 Project Goals

| Goal | Approach |
|------|----------|
| **Measurement rigor** | Warmup runs, 5+ repeats, median + spread, pinned GPU clocks/temp, exact commit/driver/CUDA version per result |
| **Honest analysis** | Publish negative results: where GPU loses (tiny corpora, single queries) and why |
| **Reproducible benchmarks** | Machine-readable JSON/CSV, script to regenerate all plots from raw data |
| **Roofline analysis** | Achieved memory bandwidth % of peak, arithmetic intensity vs batch size, Nsight Compute profiles |
| **Crossover analysis** | Latency/throughput vs corpus size & batch size for CPU BF, CPU HNSW, GPU BF, GPU IVF |
| **Correctness first** | Differential testing vs CPU oracle, fuzz tests, `go test -race`, `compute-sanitizer` |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Go Control Plane                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ gRPC API     │  │ Dynamic      │  │ Pinned Buffer Pool   │  │
│  │ Request Queue│→ │ Batcher      │→ │ (double-buffered)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │               │                     │                 │
│         ▼               ▼                     ▼                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Stream Manager (2 CUDA streams)             │  │
│  │  Stream 0: H2D ──► Kernel ──► D2H                        │  │
│  │  Stream 1:       H2D ──► Kernel ──► D2H  (overlapped)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ cgo (1 call per batch)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CUDA Compute Plane                         │
│  • Brute force (exact)          • IVF-Flat                     │
│  • Fused top-k (warp shuffle)   • IVF-PQ + re-ranking (stretch)│
│  • cuBLAS only for coarse matmul                                │
└─────────────────────────────────────────────────────────────────┘
```

### Index Interface (pluggable backends)

```go
type Index interface {
    Build(vectors []vector.Vector) error
    Search(ctx context.Context, queries []vector.Vector, k int) (SearchResults, error)
    Name() string
    Stats() Stats
    Close() error
}
```

**Backends (planned):**
1. ✅ CPU Brute Force (exact, oracle)
2. 🔄 GPU Brute Force (exact) — *in progress*
3. ⬜ GPU IVF-Flat
4. ⬜ GPU IVF-PQ with raw-vector re-ranking
5. ⬜ CPU HNSW (own or hnswlib bindings)

---

## 📊 Benchmark Rigor (The Differentiator)

### Datasets
- **Primary**: SIFT1M (128-dim, L2) with ground truth
- **Secondary**: Real embedding dataset (384-dim, cosine, normalized)
- **Large**: GIST1M (960-dim) via fp16/out-of-core streaming

### Metrics
| Metric | Method |
|--------|--------|
| Recall@k vs QPS | ann-benchmarks style Pareto curves (sweep nprobe/ef/batch/corpus) |
| Latency p50/p99/p99.9 | Open-loop Poisson arrival generator (no waiting for responses) |
| Per-stage breakdown | Queue wait, H2D, Kernel, D2H, Total |
| Roofline | Achieved BW % of RTX 3050 peak, arithmetic intensity vs batch size |

### Method Hygiene
- ✅ Warmup runs, ≥5 repeats, report median + spread
- ✅ Pin/log GPU clocks and temperature
- ✅ Record exact commit, driver, CUDA version per result
- ✅ Machine-readable results (JSON/CSV) + plot regeneration script
- ✅ CI-style recall regression test on small dataset

---

## 📁 Project Structure

```
gpuvec/
├── cmd/
│   ├── bench/           # Benchmark runner (closed-loop + open-loop)
│   └── gen_test_data/   # Synthetic dataset generator for CI
├── internal/
│   ├── index/           # Index interface + backends
│   │   ├── index.go          # Index interface, SearchResults, Stats
│   │   ├── bruteforce.go     # CPU brute force (oracle)
│   │   ├── gpu_bruteforce.go # GPU brute force (Phase 2)
│   │   └── *_test.go         # Differential tests vs oracle
│   ├── dataset/         # SIFT1M loader, fvecs/ivecs parsers
│   ├── bench/           # Benchmark harness (latency histograms, recall)
│   ├── gpu/             # CUDA bindings (pinned mem, streams, buffer pool)
│   │   ├── bruteforce.cu     # CUDA kernel + host launcher
│   │   └── gpu.go            # Go bindings (cgo)
│   └── batcher/         # Dynamic batcher, request queue (Phase 3)
├── pkg/
│   └── vector/          # Vector math (L2, cosine, fvecs I/O)
└── results/             # Benchmark outputs (JSON, CSV)
```

---

## 🚀 Current Progress

| Phase | Status | Description |
|-------|--------|-------------|
| **1** | ✅ **Complete** | Repo skeleton, Index interface, CPU brute force oracle, SIFT1M loader, benchmark harness (closed-loop + open-loop), differential tests, CSV/JSON output |
| **2** | 🔄 **In Progress** | GPU brute force (naive kernel), cgo bindings, pinned memory, streams — *blocked on CUDA toolkit install* |
| **3** | ⬜ **Planned** | Dynamic batcher, pinned buffer pool, double-buffered streams, per-stage latency instrumentation |
| **4** | ⬜ **Planned** | Fused top-k kernel (warp shuffle merge → shared mem block merge), Nsight Compute profiling |
| **5** | ⬜ **Planned** | IVF-Flat: k-means training, cluster layout, nprobe sweep, recall vs QPS curves |
| **6** | ⬜ **Planned** | CPU HNSW + FAISS/hnswlib baselines, crossover analysis |
| **7** | ⬜ **Planned** | Open-loop load generator, p99/p99.9, per-stage breakdown |
| **8** | ⬜ **Planned** | Roofline analysis, README with honest results |
| **9** | ⬜ **Stretch** | Out-of-core IVF, growing/sealed segments, IVF-PQ re-ranking |

---

## 🖥 Hardware & Environment

| Constraint | Value |
|------------|-------|
| GPU | RTX 3050 Laptop (4 GB VRAM) |
| OS | Windows 11 + WSL2 (Ubuntu) |
| CUDA | 12.4 (installing) |
| Go | 1.23+ |
| VRAM Limit | **4 GB — design constraint, not a bug** |

> **Thermal throttling is likely.** Benchmarks pin/record clocks and log GPU temperature.

---

## 📦 Quick Start (once CUDA is installed)

```bash
# Build
make build

# Generate tiny test dataset (100 base, 10 queries, dim=16)
./bin/gen_test_data ./data

# CPU baseline
./bin/bench -data ./data -out ./results -name cpu_bf -k 10 -nq 1000 -warmup 3 -runs 5 -batch 1 -metric L2 -backend cpu

# GPU brute force (batched)
./bin/bench -data ./data -out ./results -name gpu_bf -k 10 -nq 1000 -warmup 3 -runs 5 -batch 32 -metric L2 -backend gpu

# Open-loop tail latency
./bin/bench -data ./data -out ./results -name gpu_open -k 10 -nq 1000 -openloop -rate 1000 -duration 10s -backend gpu

# Run tests
make test
```

---

## 📈 Example Benchmark Output

```text
=== Benchmark Result ===
Run:           gpu_bf_batch32
Index:         GPU-BruteForce-L2
Dataset:       SIFT1M
K:             10
Queries:       1000
Batch size:    32
Latency P50:   245.32 µs
Latency P99:   892.14 µs
Latency P99.9: 1.42 ms
Throughput:    42,847.15 QPS
Recall@10:     1.0000
Index memory:  512.00 MB
```

Results saved to `results/*.json` and `results/summary.csv` for plotting.

---

## 🧪 Correctness Engineering

- **Differential testing**: Every backend tested against CPU brute force oracle
- **Fuzz tests**: Top-k ties, NaN/Inf, k > N, empty clusters, duplicate vectors
- **Race detection**: `go test -race` passes
- **CUDA sanitizer**: `compute-sanitizer` integration planned
- **Recall regression**: CI-style test on small dataset

---

## 📚 Key Design Decisions (for interviews)

| Decision | Rationale |
|----------|-----------|
| **Go + CUDA via cgo** | Go for control plane (gRPC, batching, metrics); CUDA for compute. Single cgo call per batch. |
| **Pinned host memory** | Eliminates page faults, enables async H2D/D2D, 2-3× transfer speedup |
| **Double-buffered streams** | Overlap H2D of batch N+1 with Kernel of batch N with D2H of batch N-1 |
| **Fused top-k kernel** | Never write all N distances to global memory. Per-thread register top-k → warp shuffle → shared mem merge |
| **Precomputed norms** | `||a-b||² = ||a||² + ||b||² - 2(a·b)` — no sqrt in hot path |
| **Cosine = normalize + dot** | Single code path for both metrics |
| **IVF clusters contiguous + offsets** | Coalesced reads during fine search |
| **4 GB VRAM constraint** | Forces out-of-core design; makes streaming a feature, not a workaround |

---

## 🗺 Roadmap

### Near Term (Phases 2-4)
- [ ] GPU brute force working & profiled
- [ ] Batching + double-buffered pipeline
- [ ] Fused top-k kernel with Nsight optimization

### Medium Term (Phases 5-8)
- [ ] IVF-Flat with k-means training
- [ ] CPU HNSW + FAISS baselines
- [ ] Crossover analysis (the "money graph")
- [ ] Roofline analysis + honest README

### Stretch (Phase 9+)
- [ ] Out-of-core IVF (index in host RAM, stream clusters to GPU)
- [ ] Segment design: growing (brute) + sealed (IVF) + background merge
- [ ] IVF-PQ with re-ranking (tens of millions of vectors)
- [ ] Tree-sitter code search layer

---

## 🤝 Contributing

This is a portfolio project for demonstration purposes. Issues/PRs welcome for:
- Benchmark methodology improvements
- CUDA kernel optimizations
- Additional dataset integrations
- Documentation corrections

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**CS/ML Student** — Building a portfolio project to demonstrate systems & performance engineering skills. Every design decision is defensible in an interview.

> *"The differentiator is measurement rigor and honest analysis: showing exactly where GPU search wins and loses against CPU baselines, with trustworthy numbers."*

---

**Last updated**: November 2024 — Phase 1 complete, Phase 2 pending CUDA toolkit