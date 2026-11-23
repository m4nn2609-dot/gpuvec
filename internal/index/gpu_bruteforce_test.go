package index

import (
	"context"
	"testing"

	"github.com/user/gpuvec/pkg/vector"
)

func TestGPUBruteForceL2(t *testing.T) {
	// Skip if no GPU available
	if !hasGPU() {
		t.Skip("No GPU available")
	}

	vectors := []vector.Vector{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}

	idx := NewGPUBruteForce(MetricL2)
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer idx.Close()

	query := []vector.Vector{{0, 0, 0}}
	results, err := idx.Search(context.Background(), query, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results.Results) != 1 || len(results.Results[0]) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results.Results[0]))
	}

	// All three basis vectors have distance 1
	for _, r := range results.Results[0] {
		if r.Score < 0.99 || r.Score > 1.01 {
			t.Errorf("Expected distance ~1, got %f", r.Score)
		}
	}
}

func TestGPUBruteForceCosine(t *testing.T) {
	if !hasGPU() {
		t.Skip("No GPU available")
	}

	vectors := []vector.Vector{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	for i := range vectors {
		vectors[i].NormalizeL2()
	}

	idx := NewGPUBruteForce(MetricCosine)
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer idx.Close()

	query := []vector.Vector{{1, 0, 0}}
	query[0].NormalizeL2()

	results, err := idx.Search(context.Background(), query, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results.Results) != 1 || len(results.Results[0]) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results.Results[0]))
	}

	// Cosine similarity with itself should be 1.0
	if results.Results[0][0].Score < 0.99 || results.Results[0][0].Score > 1.01 {
		t.Errorf("Expected score ~1.0 (cosine=1), got %f", results.Results[0][0].Score)
	}
	if results.Results[0][0].Index != 0 {
		t.Errorf("Expected index 0, got %d", results.Results[0][0].Index)
	}
}

func TestGPUBruteForceRecall(t *testing.T) {
	if !hasGPU() {
		t.Skip("No GPU available")
	}

	// Use a larger random dataset
	vectors := make([]vector.Vector, 1000)
	for i := range vectors {
		vectors[i] = make(vector.Vector, 128)
		for j := range vectors[i] {
			vectors[i][j] = float32(i + j)
		}
	}

	idx := NewGPUBruteForce(MetricL2)
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer idx.Close()

	// CPU brute force as oracle
	cpuIdx := NewBruteForce(MetricL2)
	if err := cpuIdx.Build(vectors); err != nil {
		t.Fatalf("CPU Build failed: %v", err)
	}
	defer cpuIdx.Close()

	queries := make([]vector.Vector, 10)
	for i := range queries {
		queries[i] = make(vector.Vector, 128)
		for j := range queries[i] {
			queries[i][j] = float32(i * j)
		}
	}

	gpuResults, err := idx.Search(context.Background(), queries, 10)
	if err != nil {
		t.Fatalf("GPU Search failed: %v", err)
	}

	cpuResults, err := cpuIdx.Search(context.Background(), queries, 10)
	if err != nil {
		t.Fatalf("CPU Search failed: %v", err)
	}

	// Compare results - should be identical for exact search
	for qi := range queries {
		for k := 0; k < 10; k++ {
			if gpuResults.Results[qi][k].Index != cpuResults.Results[qi][k].Index {
				t.Errorf("Mismatch at query %d, k=%d: GPU idx=%d, CPU idx=%d",
					qi, k, gpuResults.Results[qi][k].Index, cpuResults.Results[qi][k].Index)
			}
			if gpuResults.Results[qi][k].Score != cpuResults.Results[qi][k].Score {
				t.Errorf("Score mismatch at query %d, k=%d: GPU=%f, CPU=%f",
					qi, k, gpuResults.Results[qi][k].Score, cpuResults.Results[qi][k].Score)
			}
		}
	}
}

func TestGPUBruteForceBatch(t *testing.T) {
	if !hasGPU() {
		t.Skip("No GPU available")
	}

	vectors := make([]vector.Vector, 1000)
	for i := range vectors {
		vectors[i] = make(vector.Vector, 128)
		for j := range vectors[i] {
			vectors[i][j] = float32(i + j)
		}
	}

	idx := NewGPUBruteForce(MetricL2)
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	defer idx.Close()

	// Batch of queries
	queries := make([]vector.Vector, 32)
	for i := range queries {
		queries[i] = make(vector.Vector, 128)
		for j := range queries[i] {
			queries[i][j] = float32(i * j)
		}
	}

	results, err := idx.Search(context.Background(), queries, 10)
	if err != nil {
		t.Fatalf("Batch search failed: %v", err)
	}

	if len(results.Results) != 32 {
		t.Fatalf("Expected 32 results, got %d", len(results.Results))
	}
	for _, r := range results.Results {
		if len(r) != 10 {
			t.Errorf("Expected 10 results per query, got %d", len(r))
		}
	}
}

func hasGPU() bool {
	// Try to create a minimal index and see if it works
	idx := NewGPUBruteForce(MetricL2)
	err := idx.Build([]vector.Vector{{0}})
	if err != nil {
		return false
	}
	idx.Close()
	return true
}