package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/user/gpuvec/internal/bench"
	"github.com/user/gpuvec/internal/dataset"
	"github.com/user/gpuvec/internal/index"
)

func main() {
	var (
		dataDir      = flag.String("data", "./data", "dataset directory")
		outputDir    = flag.String("out", "./results", "output directory")
		runName      = flag.String("name", "bruteforce", "run name")
		k            = flag.Int("k", 10, "top-k")
		nQueries     = flag.Int("nq", 1000, "number of queries (0 = all)")
		warmupRuns   = flag.Int("warmup", 3, "warmup runs")
		measureRuns  = flag.Int("runs", 5, "measurement runs")
		batchSize    = flag.Int("batch", 1, "batch size")
		metric       = flag.String("metric", "L2", "metric: L2 or Cosine")
		backend      = flag.String("backend", "cpu", "backend: cpu or gpu")
		openLoop     = flag.Bool("openloop", false, "run open-loop benchmark")
		arrivalRate  = flag.Float64("rate", 1000, "arrival rate (QPS) for open-loop")
		duration     = flag.Duration("duration", 10*time.Second, "duration for open-loop")
		subsetBase   = flag.Int("subset-base", 0, "subset base vectors (0 = all)")
		subsetQuery  = flag.Int("subset-query", 0, "subset queries (0 = all)")
	)
	flag.Parse()

	runtime.GOMAXPROCS(runtime.NumCPU())

	// Load dataset
	fmt.Printf("Loading SIFT1M from %s...\n", *dataDir)
	ds, err := dataset.LoadSIFT1M(*dataDir)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}
	fmt.Printf("Loaded: %d base, %d queries, dim=%d\n", len(ds.Base), len(ds.Queries), ds.Dim)

	// Apply subset if requested
	if *subsetBase > 0 || *subsetQuery > 0 {
		ds = ds.Subset(*subsetBase, *subsetQuery)
		fmt.Printf("Using subset: %d base, %d queries\n", len(ds.Base), len(ds.Queries))
	}

	// Normalize if cosine
	if *metric == "Cosine" {
		fmt.Println("Normalizing vectors for cosine similarity...")
		ds.NormalizeL2()
	}

	// Create index
	var idx index.Index
	var met index.Metric
	if *metric == "L2" {
		met = index.MetricL2
	} else {
		met = index.MetricCosine
	}
	switch *backend {
	case "cpu":
		idx = index.NewBruteForce(met)
	case "gpu":
		idx = index.NewGPUBruteForce(met)
	default:
		log.Fatalf("Unknown backend: %s (use cpu or gpu)", *backend)
	}

	// Build index
	fmt.Println("Building index...")
	start := time.Now()
	if err := idx.Build(ds.Base); err != nil {
		log.Fatalf("Build failed: %v", err)
	}
	fmt.Printf("Build time: %v\n", time.Since(start))
	fmt.Printf("Index stats: %+v\n", idx.Stats())

	// Run benchmark
	cfg := bench.BenchmarkConfig{
		Index:       idx,
		Dataset:     ds,
		K:           *k,
		NQueries:    *nQueries,
		WarmupRuns:  *warmupRuns,
		MeasureRuns: *measureRuns,
		BatchSize:   *batchSize,
		OutputDir:   *outputDir,
		RunName:     *runName,
	}

	var result *bench.BenchmarkResult
	if *openLoop {
		fmt.Printf("Running open-loop benchmark: rate=%.0f QPS, duration=%v\n", *arrivalRate, *duration)
		result, err = bench.RunOpenLoopBenchmark(cfg, *arrivalRate, *duration)
	} else {
		fmt.Printf("Running closed-loop benchmark: %d runs, batch=%d\n", *measureRuns, *batchSize)
		result, err = bench.RunBenchmark(cfg)
	}
	if err != nil {
		log.Fatalf("Benchmark failed: %v", err)
	}

	// Print summary
	printResult(result)
	
	// Verify recall against ground truth (sanity check)
	verifyRecall(idx, ds, *k, *nQueries)

	// Close index (important for GPU to release resources)
	if err := idx.Close(); err != nil {
		log.Printf("Close failed: %v", err)
	}
}

func printResult(r *bench.BenchmarkResult) {
	fmt.Println("\n=== Benchmark Result ===")
	fmt.Printf("Run:           %s\n", r.RunName)
	fmt.Printf("Index:         %s\n", r.IndexName)
	fmt.Printf("Dataset:       %s\n", r.DatasetName)
	fmt.Printf("K:             %d\n", r.K)
	fmt.Printf("Queries:       %d\n", r.NQueries)
	fmt.Printf("Batch size:    %d\n", r.BatchSize)
	fmt.Printf("Timestamp:     %s\n", r.Timestamp)
	fmt.Printf("Git commit:    %s\n", r.GitCommit)
	fmt.Printf("Go version:    %s\n", r.GoVersion)
	fmt.Println()
	fmt.Printf("Latency P50:   %s\n", formatLatency(r.LatencyP50))
	fmt.Printf("Latency P90:   %s\n", formatLatency(r.LatencyP90))
	fmt.Printf("Latency P95:   %s\n", formatLatency(r.LatencyP95))
	fmt.Printf("Latency P99:   %s\n", formatLatency(r.LatencyP99))
	fmt.Printf("Latency P99.9: %s\n", formatLatency(r.LatencyP999))
	fmt.Printf("Latency Mean:  %s\n", formatLatency(r.LatencyMean))
	fmt.Printf("Latency Std:   %.2f µs\n", r.LatencyStd/1000)
	fmt.Println()
	fmt.Printf("Throughput:    %.2f QPS\n", r.QPS)
	fmt.Println()
	fmt.Printf("Recall@%d:     %.4f\n", r.K, r.RecallAtK)
	fmt.Printf("Recall@1:      %.4f\n", r.RecallAt1)
	fmt.Printf("Recall@10:     %.4f\n", r.RecallAt10)
	fmt.Printf("Recall@100:    %.4f\n", r.RecallAt100)
	fmt.Println()
	fmt.Printf("Index memory:  %.2f MB\n", float64(r.IndexStats.MemoryBytes)/1e6)
	fmt.Printf("Build time:    %v\n", r.IndexStats.BuildTime)
}

func formatLatency(ns int64) string {
	if ns < 1000 {
		return fmt.Sprintf("%d ns", ns)
	}
	if ns < 1_000_000 {
		return fmt.Sprintf("%.2f µs", float64(ns)/1000)
	}
	if ns < 1_000_000_000 {
		return fmt.Sprintf("%.2f ms", float64(ns)/1_000_000)
	}
	return fmt.Sprintf("%.2f s", float64(ns)/1_000_000_000)
}

func verifyRecall(idx index.Index, ds *dataset.Dataset, k, nQueries int) {
	if nQueries > len(ds.Queries) {
		nQueries = len(ds.Queries)
	}
	queries := ds.Queries[:nQueries]
	groundTruth := ds.GroundTruth[:nQueries]

	fmt.Println("\n=== Recall Verification ===")
	results, err := idx.Search(context.Background(), queries, k)
	if err != nil {
		log.Printf("Verification search failed: %v", err)
		return
	}

	totalRecall := 0.0
	for i, res := range results.Results {
		gt := groundTruth[i]
		recall := computeRecall(res, gt, k)
		totalRecall += recall
		if recall < 1.0 && i < 5 {
			fmt.Printf("  Query %d: recall=%.4f, got=%v, gt=%v\n", i, recall, getIndices(res), gt[:k])
		}
	}
	avgRecall := totalRecall / float64(nQueries)
	fmt.Printf("Average Recall@%d: %.4f (expected 1.0 for brute force)\n", k, avgRecall)
	
	if avgRecall < 0.999 {
		fmt.Println("WARNING: Recall < 1.0 for brute force! Possible bug.")
	}
}

func computeRecall(results []index.SearchResult, groundTruth []int32, k int) float64 {
	if k > len(groundTruth) {
		k = len(groundTruth)
	}
	if k > len(results) {
		k = len(results)
	}
	if k == 0 {
		return 0
	}
	gtSet := make(map[int32]bool, k)
	for i := 0; i < k; i++ {
		gtSet[groundTruth[i]] = true
	}
	hits := 0
	for i := 0; i < k; i++ {
		if gtSet[results[i].Index] {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

func getIndices(results []index.SearchResult) []int32 {
	out := make([]int32, len(results))
	for i, r := range results {
		out[i] = r.Index
	}
	return out
}