package bench

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/user/gpuvec/internal/dataset"
	"github.com/user/gpuvec/internal/index"
	"github.com/user/gpuvec/pkg/vector"
)

// BenchmarkConfig holds benchmark configuration.
type BenchmarkConfig struct {
	Index        index.Index
	Dataset      *dataset.Dataset
	K            int
	NQueries     int
	WarmupRuns   int
	MeasureRuns  int
	BatchSize    int
	OutputDir    string
	RunName      string
}

// BenchmarkResult holds the results of a benchmark run.
type BenchmarkResult struct {
	RunName      string             `json:"run_name"`
	IndexName    string             `json:"index_name"`
	DatasetName  string             `json:"dataset_name"`
	K            int                `json:"k"`
	NQueries     int                `json:"n_queries"`
	BatchSize    int                `json:"batch_size"`
	Timestamp    string             `json:"timestamp"`
	GitCommit    string             `json:"git_commit"`
	GoVersion    string             `json:"go_version"`
	
	// Latency stats (nanoseconds)
	LatencyP50   int64              `json:"latency_p50_ns"`
	LatencyP90   int64              `json:"latency_p90_ns"`
	LatencyP95   int64              `json:"latency_p95_ns"`
	LatencyP99   int64              `json:"latency_p99_ns"`
	LatencyP999  int64              `json:"latency_p999_ns"`
	LatencyMean  int64              `json:"latency_mean_ns"`
	LatencyStd   float64            `json:"latency_std_ns"`
	
	// Throughput
	QPS          float64            `json:"qps"`
	
	// Recall
	RecallAtK    float64            `json:"recall_at_k"`
	RecallAt1    float64            `json:"recall_at_1"`
	RecallAt10   float64            `json:"recall_at_10"`
	RecallAt100  float64            `json:"recall_at_100"`
	
	// Per-stage latency (if available)
	StageLatencies map[string]StageStats `json:"stage_latencies,omitempty"`
	
	// Index stats
	IndexStats   index.Stats        `json:"index_stats"`
	
	// Raw latencies for analysis
	RawLatencies []int64            `json:"raw_latencies_ns,omitempty"`
}

// StageStats holds statistics for a pipeline stage.
type StageStats struct {
	P50   int64   `json:"p50_ns"`
	P99   int64   `json:"p99_ns"`
	Mean  int64   `json:"mean_ns"`
	Count int     `json:"count"`
}

// RunBenchmark runs a closed-loop benchmark (wait for each response).
func RunBenchmark(cfg BenchmarkConfig) (*BenchmarkResult, error) {
	if cfg.NQueries > len(cfg.Dataset.Queries) {
		cfg.NQueries = len(cfg.Dataset.Queries)
	}
	if cfg.K > len(cfg.Dataset.Base) {
		cfg.K = len(cfg.Dataset.Base)
	}

	queries := cfg.Dataset.Queries[:cfg.NQueries]
	groundTruth := cfg.Dataset.GroundTruth[:cfg.NQueries]

	// Warmup
	for i := 0; i < cfg.WarmupRuns; i++ {
		_, _ = cfg.Index.Search(context.Background(), queries[:min(100, len(queries))], cfg.K)
	}

	// Measure
	var latencies []int64
	var totalRecall float64
	var totalRecall1, totalRecall10, totalRecall100 float64
	var mu sync.Mutex

	startAll := time.Now()
	for run := 0; run < cfg.MeasureRuns; run++ {
		// Shuffle queries for each run (optional)
		// For reproducibility, we don't shuffle
		
		// Process in batches
		for i := 0; i < len(queries); i += cfg.BatchSize {
			end := min(i+cfg.BatchSize, len(queries))
			batch := queries[i:end]
			
			ctx := context.Background()
			searchStart := time.Now()
			results, err := cfg.Index.Search(ctx, batch, cfg.K)
			searchLatency := time.Since(searchStart)
			
			if err != nil {
				return nil, err
			}
			
			// Record latencies per query
			perQueryLatency := searchLatency.Nanoseconds() / int64(len(batch))
			mu.Lock()
			for range batch {
				latencies = append(latencies, perQueryLatency)
			}
			mu.Unlock()
			
			// Compute recall
			for j, res := range results.Results {
				gt := groundTruth[i+j]
				recall := computeRecall(res, gt, cfg.K)
				mu.Lock()
				totalRecall += recall
				if cfg.K >= 1 {
					totalRecall1 += computeRecallAt(res, gt, 1)
				}
				if cfg.K >= 10 {
					totalRecall10 += computeRecallAt(res, gt, 10)
				}
				if cfg.K >= 100 {
					totalRecall100 += computeRecallAt(res, gt, 100)
				}
				mu.Unlock()
			}
		}
	}
	totalTime := time.Since(startAll)

	// Compute statistics
	result := computeStats(cfg, latencies, totalTime, 
		totalRecall, totalRecall1, totalRecall10, totalRecall100,
		float64(cfg.MeasureRuns*cfg.NQueries))
	
	// Save results
	if cfg.OutputDir != "" {
		if err := saveResult(cfg.OutputDir, result); err != nil {
			return nil, err
		}
	}

	return result, nil
}

// RunOpenLoopBenchmark runs an open-loop benchmark (Poisson arrivals, no waiting).
func RunOpenLoopBenchmark(cfg BenchmarkConfig, arrivalRate float64, duration time.Duration) (*BenchmarkResult, error) {
	if cfg.NQueries > len(cfg.Dataset.Queries) {
		cfg.NQueries = len(cfg.Dataset.Queries)
	}
	queries := cfg.Dataset.Queries[:cfg.NQueries]
	groundTruth := cfg.Dataset.GroundTruth[:cfg.NQueries]

	// Channel for results
	type request struct {
		query     vector.Vector
		gt        []int32
		startTime time.Time
		resultCh  chan *BenchmarkResult
	}

	reqChan := make(chan request, 10000)
	resultChan := make(chan *BenchmarkResult, 10000)

	// Worker pool
	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for req := range reqChan {
				ctx := context.Background()
				results, err := cfg.Index.Search(ctx, []vector.Vector{req.query}, cfg.K)
				latency := time.Since(req.startTime)
				
				recall := 0.0
				if err == nil && len(results.Results) > 0 {
					recall = computeRecall(results.Results[0], req.gt, cfg.K)
				}
				
				resultChan <- &BenchmarkResult{
					LatencyMean: latency.Nanoseconds(),
					RecallAtK:   recall,
					RawLatencies: []int64{latency.Nanoseconds()},
				}
			}
		}()
	}

	// Arrival generator (Poisson process)
	go func() {
		ticker := time.NewTicker(time.Duration(float64(time.Second) / arrivalRate))
		defer ticker.Stop()
		defer close(reqChan)
		
		queryIdx := 0
		start := time.Now()
		for {
			select {
			case <-ticker.C:
				if time.Since(start) >= duration {
					return
				}
				req := request{
					query:     queries[queryIdx%len(queries)],
					gt:        groundTruth[queryIdx%len(groundTruth)],
					startTime: time.Now(),
				}
				select {
				case reqChan <- req:
					queryIdx++
				default:
					// Drop if queue full (load shedding)
				}
			case <-time.After(duration):
				return
			}
		}
	}()

	// Collector
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Aggregate results
	var latencies []int64
	var totalRecall float64
	count := 0
	for r := range resultChan {
		latencies = append(latencies, r.RawLatencies[0])
		totalRecall += r.RecallAtK
		count++
	}

	totalTime := duration
	result := computeStats(cfg, latencies, totalTime, totalRecall, 0, 0, 0, float64(count))
	result.RunName += "-openloop"
	
	if cfg.OutputDir != "" {
		saveResult(cfg.OutputDir, result)
	}
	return result, nil
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
	
	// Build set of ground truth top-k
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

func computeRecallAt(results []index.SearchResult, groundTruth []int32, atK int) float64 {
	if atK > len(groundTruth) {
		atK = len(groundTruth)
	}
	if atK > len(results) {
		atK = len(results)
	}
	if atK == 0 {
		return 0
	}
	gtSet := make(map[int32]bool, atK)
	for i := 0; i < atK; i++ {
		gtSet[groundTruth[i]] = true
	}
	hits := 0
	for i := 0; i < atK; i++ {
		if gtSet[results[i].Index] {
			hits++
		}
	}
	return float64(hits) / float64(atK)
}

func computeStats(cfg BenchmarkConfig, latencies []int64, totalTime time.Duration,
	totalRecall, totalRecall1, totalRecall10, totalRecall100 float64, totalQueries float64) *BenchmarkResult {
	
	if len(latencies) == 0 {
		return &BenchmarkResult{}
	}
	
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	
	n := len(latencies)
	sum := int64(0)
	for _, l := range latencies {
		sum += l
	}
	mean := float64(sum) / float64(n)
	
	// Std deviation
	var variance float64
	for _, l := range latencies {
		diff := float64(l) - mean
		variance += diff * diff
	}
	std := math.Sqrt(variance / float64(n))
	
	percentile := func(p float64) int64 {
		idx := int(float64(n-1) * p)
		return latencies[idx]
	}
	
	qps := totalQueries / totalTime.Seconds()
	
	return &BenchmarkResult{
		RunName:       cfg.RunName,
		IndexName:     cfg.Index.Name(),
		DatasetName:   cfg.Dataset.Name,
		K:             cfg.K,
		NQueries:      cfg.NQueries,
		BatchSize:     cfg.BatchSize,
		Timestamp:     time.Now().Format(time.RFC3339),
		GitCommit:     getGitCommit(),
		GoVersion:     runtime.Version(),
		LatencyP50:    percentile(0.50),
		LatencyP90:    percentile(0.90),
		LatencyP95:    percentile(0.95),
		LatencyP99:    percentile(0.99),
		LatencyP999:   percentile(0.999),
		LatencyMean:   int64(mean),
		LatencyStd:    std,
		QPS:           qps,
		RecallAtK:     totalRecall / totalQueries,
		RecallAt1:     totalRecall1 / totalQueries,
		RecallAt10:    totalRecall10 / totalQueries,
		RecallAt100:   totalRecall100 / totalQueries,
		IndexStats:    cfg.Index.Stats(),
		RawLatencies:  latencies,
	}
}

func getGitCommit() string {
	// Try to get git commit
	// For now return empty
	return ""
}

func saveResult(outputDir string, result *BenchmarkResult) error {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return err
	}
	
	// Save JSON
	jsonPath := filepath.Join(outputDir, result.RunName+".json")
	f, err := os.Create(jsonPath)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		return err
	}
	
	// Save CSV summary
	csvPath := filepath.Join(outputDir, "summary.csv")
	fileExists := true
	if _, err := os.Stat(csvPath); os.IsNotExist(err) {
		fileExists = false
	}
	csvFile, err := os.OpenFile(csvPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer csvFile.Close()
	
	writer := csv.NewWriter(csvFile)
	if !fileExists {
		writer.Write([]string{
			"run_name", "index_name", "dataset", "k", "n_queries", "batch_size",
			"latency_p50_ns", "latency_p99_ns", "latency_p999_ns", "latency_mean_ns",
			"qps", "recall_at_k", "recall_at_1", "recall_at_10", "recall_at_100",
			"index_memory_mb", "build_time_s", "timestamp",
		})
	}
	writer.Write([]string{
		result.RunName, result.IndexName, result.DatasetName,
		fmt.Sprintf("%d", result.K), fmt.Sprintf("%d", result.NQueries),
		fmt.Sprintf("%d", result.BatchSize),
		fmt.Sprintf("%d", result.LatencyP50), fmt.Sprintf("%d", result.LatencyP99),
		fmt.Sprintf("%d", result.LatencyP999), fmt.Sprintf("%d", result.LatencyMean),
		fmt.Sprintf("%.2f", result.QPS),
		fmt.Sprintf("%.4f", result.RecallAtK), fmt.Sprintf("%.4f", result.RecallAt1),
		fmt.Sprintf("%.4f", result.RecallAt10), fmt.Sprintf("%.4f", result.RecallAt100),
		fmt.Sprintf("%.2f", float64(result.IndexStats.MemoryBytes)/1e6),
		fmt.Sprintf("%.2f", result.IndexStats.BuildTime.Seconds()),
		result.Timestamp,
	})
	writer.Flush()
	return writer.Error()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}