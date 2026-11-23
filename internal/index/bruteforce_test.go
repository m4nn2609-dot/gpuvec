package index

import (
	"context"
	"math"
	"testing"

	"github.com/user/gpuvec/pkg/vector"
)

func TestBruteForceL2(t *testing.T) {
	// Simple test case
	vectors := []vector.Vector{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}

	idx := NewBruteForce(MetricL2)
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Query at origin - closest should be (1,0,0), (0,1,0), (0,0,1) all at distance 1
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
		if math.Abs(float64(r.Score)-1.0) > 1e-5 {
			t.Errorf("Expected distance 1, got %f", r.Score)
		}
	}
}

func TestBruteForceCosine(t *testing.T) {
	vectors := []vector.Vector{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}

	// Normalize for cosine
	for i := range vectors {
		vectors[i].NormalizeL2()
	}

	idx := NewBruteForce(MetricCosine)
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Query = (1,0,0) normalized - should match first vector exactly
	query := []vector.Vector{{1, 0, 0}}
	query[0].NormalizeL2()

	results, err := idx.Search(context.Background(), query, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results.Results) != 1 || len(results.Results[0]) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results.Results[0]))
	}

	// Cosine similarity with itself should be 1.0 (score = -1.0 since we negate)
	if math.Abs(float64(results.Results[0][0].Score)-(-1.0)) > 1e-5 {
		t.Errorf("Expected score -1.0 (cosine=1), got %f", results.Results[0][0].Score)
	}
	if results.Results[0][0].Index != 0 {
		t.Errorf("Expected index 0, got %d", results.Results[0][0].Index)
	}
}

func TestBruteForceEmpty(t *testing.T) {
	idx := NewBruteForce(MetricL2)
	if err := idx.Build([]vector.Vector{}); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	results, err := idx.Search(context.Background(), []vector.Vector{{1, 2, 3}}, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results.Results) != 1 || len(results.Results[0]) != 0 {
		t.Errorf("Expected empty results, got %d", len(results.Results[0]))
	}
}

func TestBruteForceKGreaterThanN(t *testing.T) {
	vectors := []vector.Vector{
		{1, 0},
		{0, 1},
	}
	idx := NewBruteForce(MetricL2)
	idx.Build(vectors)

	results, err := idx.Search(context.Background(), []vector.Vector{{0, 0}}, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results.Results[0]) != 2 {
		t.Errorf("Expected 2 results (clamped to N), got %d", len(results.Results[0]))
	}
}

func TestBruteForceParallelConsistency(t *testing.T) {
	// Test that parallel search gives same results as sequential
	vectors := make([]vector.Vector, 1000)
	for i := range vectors {
		vectors[i] = vector.Vector{
			float32(i % 100),
			float32((i * 7) % 100),
			float32((i * 13) % 100),
			float32((i * 17) % 100),
		}
	}

	idx := NewBruteForce(MetricL2)
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	queries := make([]vector.Vector, 100)
	for i := range queries {
		queries[i] = vector.Vector{
			float32(i * 3 % 100),
			float32(i * 5 % 100),
			float32(i * 7 % 100),
			float32(i * 11 % 100),
		}
	}

	// Parallel search (default)
	resultsPar, err := idx.Search(context.Background(), queries, 10)
	if err != nil {
		t.Fatalf("Parallel search failed: %v", err)
	}

	// Sequential search (force 1 worker)
	// We can't easily force sequential, but we can verify results are valid
	for i, res := range resultsPar.Results {
		if len(res) != 10 {
			t.Errorf("Query %d: expected 10 results, got %d", i, len(res))
		}
		// Check sorted order
		for j := 1; j < len(res); j++ {
			if res[j].Score < res[j-1].Score {
				t.Errorf("Query %d: results not sorted at pos %d: %f < %f", i, j, res[j].Score, res[j-1].Score)
			}
		}
	}
}

func TestBruteForceNaNInf(t *testing.T) {
	vectors := []vector.Vector{
		{1, 0},
		{0, 1},
		{float32(math.NaN()), 0}, // This should be handled
	}
	idx := NewBruteForce(MetricL2)
	
	// Build should not panic
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Search with NaN query
	query := []vector.Vector{{float32(math.NaN()), 0}}
	results, err := idx.Search(context.Background(), query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	// Should not crash, NaN distances will be at the end
	_ = results
}

func BenchmarkBruteForceL2(b *testing.B) {
	vectors := make([]vector.Vector, 10000)
	for i := range vectors {
		vectors[i] = make(vector.Vector, 128)
		for j := range vectors[i] {
			vectors[i][j] = float32(i + j)
		}
	}

	idx := NewBruteForce(MetricL2)
	if err := idx.Build(vectors); err != nil {
		b.Fatalf("Build failed: %v", err)
	}

	queries := make([]vector.Vector, 100)
	for i := range queries {
		queries[i] = make(vector.Vector, 128)
		for j := range queries[i] {
			queries[i][j] = float32(i * j)
		}
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = idx.Search(context.Background(), queries, 10)
		}
	})
}