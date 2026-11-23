package dataset

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/user/gpuvec/pkg/vector"
)

func TestDatasetSubset(t *testing.T) {
	base := make([]vector.Vector, 100)
	for i := range base {
		base[i] = vector.Vector{float32(i), float32(i + 1)}
	}
	queries := make([]vector.Vector, 10)
	for i := range queries {
		queries[i] = vector.Vector{float32(i * 10), float32(i * 10 + 1)}
	}
	groundTruth := make([][]int32, 10)
	for i := range groundTruth {
		groundTruth[i] = []int32{int32(i), int32(i + 1), int32(i + 2)}
	}

	ds := &Dataset{
		Name:        "Test",
		Base:        base,
		Queries:     queries,
		GroundTruth: groundTruth,
		Dim:         2,
		Metric:      "L2",
	}

	subset := ds.Subset(50, 5)
	if len(subset.Base) != 50 {
		t.Errorf("Expected 50 base vectors, got %d", len(subset.Base))
	}
	if len(subset.Queries) != 5 {
		t.Errorf("Expected 5 queries, got %d", len(subset.Queries))
	}
	if len(subset.GroundTruth) != 5 {
		t.Errorf("Expected 5 ground truth, got %d", len(subset.GroundTruth))
	}
	if subset.Dim != 2 {
		t.Errorf("Expected dim 2, got %d", subset.Dim)
	}
}

func TestDatasetNormalizeL2(t *testing.T) {
	base := []vector.Vector{
		{3, 4},
		{0, 5},
	}
	queries := []vector.Vector{
		{5, 12},
	}

	ds := &Dataset{
		Base:    base,
		Queries: queries,
		Dim:     2,
		Metric:  "L2",
	}

	ds.NormalizeL2()

	// Check base vectors are normalized
	for _, v := range ds.Base {
		var sum float32
		for _, x := range v {
			sum += x * x
		}
		if sum < 0.999 || sum > 1.001 {
			t.Errorf("Vector not normalized: norm^2 = %f", sum)
		}
	}

	// Check queries are normalized
	for _, v := range ds.Queries {
		var sum float32
		for _, x := range v {
			sum += x * x
		}
		if sum < 0.999 || sum > 1.001 {
			t.Errorf("Query not normalized: norm^2 = %f", sum)
		}
	}

	if ds.Metric != "Cosine" {
		t.Errorf("Metric should be Cosine after normalize, got %s", ds.Metric)
	}
}

// TestParseFvecsFile tests parsing with actual file I/O
func TestParseFvecsFile(t *testing.T) {
	// Create temp fvecs file
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.fvecs")

	// Write 3 vectors of dim 4
	dim := int32(4)
	vectors := [][]float32{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}

	f, err := os.Create(filePath)
	if err != nil {
		t.Fatal(err)
	}
	
	for _, vec := range vectors {
		// Write dim then vector
		binary.Write(f, binary.LittleEndian, dim)
		for _, v := range vec {
			binary.Write(f, binary.LittleEndian, v)
		}
	}
	f.Close()

	// Read back
	parsed, err := vector.ReadFvecs(filePath)
	if err != nil {
		t.Fatalf("ReadFvecs failed: %v", err)
	}

	if len(parsed) != 3 {
		t.Fatalf("Expected 3 vectors, got %d", len(parsed))
	}

	expected := []vector.Vector{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}
	for i, v := range parsed {
		for j := range v {
			if v[j] != expected[i][j] {
				t.Errorf("Vector %d[%d]: got %f, want %f", i, j, v[j], expected[i][j])
			}
		}
	}
}

func TestParseIvecsFile(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.ivecs")

	dim := int32(3)
	vectors := [][]int32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}

	f, err := os.Create(filePath)
	if err != nil {
		t.Fatal(err)
	}
	
	for _, vec := range vectors {
		binary.Write(f, binary.LittleEndian, dim)
		for _, v := range vec {
			binary.Write(f, binary.LittleEndian, v)
		}
	}
	f.Close()

	parsed, err := vector.ReadIvecs(filePath)
	if err != nil {
		t.Fatalf("ReadIvecs failed: %v", err)
	}

	if len(parsed) != 3 {
		t.Fatalf("Expected 3 vectors, got %d", len(parsed))
	}

	expected := [][]int32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	for i, v := range parsed {
		for j := range v {
			if v[j] != expected[i][j] {
				t.Errorf("Vector %d[%d]: got %d, want %d", i, j, v[j], expected[i][j])
			}
		}
	}
}