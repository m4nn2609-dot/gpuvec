package vector

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestVectorNormalizeL2(t *testing.T) {
	v := Vector{3, 4}
	v.NormalizeL2()
	expected := Vector{0.6, 0.8}
	for i := range v {
		if math.Abs(float64(v[i]-expected[i])) > 1e-5 {
			t.Errorf("NormalizeL2: got %v, want %v", v, expected)
		}
	}
}

func TestVectorNormalizeL2Zero(t *testing.T) {
	v := Vector{0, 0, 0}
	v.NormalizeL2() // Should not panic
	if v[0] != 0 || v[1] != 0 || v[2] != 0 {
		t.Errorf("Zero vector should remain zero")
	}
}

func TestSquaredL2Norm(t *testing.T) {
	v := Vector{3, 4}
	norm := v.SquaredL2Norm()
	if math.Abs(float64(norm)-25.0) > 1e-5 {
		t.Errorf("SquaredL2Norm: got %f, want 25", norm)
	}
}

func TestDotProduct(t *testing.T) {
	a := Vector{1, 2, 3}
	b := Vector{4, 5, 6}
	dot := a.DotProduct(b)
	if dot != 32 {
		t.Errorf("DotProduct: got %f, want 32", dot)
	}
}

func TestDotProductMismatch(t *testing.T) {
	a := Vector{1, 2}
	b := Vector{1, 2, 3}
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic on dimension mismatch")
		}
	}()
	_ = a.DotProduct(b)
}

func TestSquaredL2Distance(t *testing.T) {
	a := Vector{1, 0}
	b := Vector{0, 1}
	normA := a.SquaredL2Norm() // 1
	normB := b.SquaredL2Norm() // 1
	dist := SquaredL2Distance(a, b, normA, normB)
	// ||a-b||^2 = ||(1,-1)||^2 = 2
	if math.Abs(float64(dist)-2.0) > 1e-5 {
		t.Errorf("SquaredL2Distance: got %f, want 2", dist)
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := Vector{1, 0}
	b := Vector{0, 1}
	sim := CosineSimilarity(a, b)
	if math.Abs(float64(sim)-0.0) > 1e-5 {
		t.Errorf("CosineSimilarity orthogonal: got %f, want 0", sim)
	}

	a = Vector{1, 2, 3}
	b = Vector{1, 2, 3}
	a.NormalizeL2()
	b.NormalizeL2()
	sim = CosineSimilarity(a, b)
	if math.Abs(float64(sim)-1.0) > 1e-5 {
		t.Errorf("CosineSimilarity same: got %f, want 1", sim)
	}
}

func TestParseFvecs(t *testing.T) {
	// Create a fake fvecs file: 2 vectors of dim 3
	// Format per vector: int32 dim, then dim float32
	dim := int32(3)
	vec1 := []float32{1, 2, 3}
	vec2 := []float32{4, 5, 6}
	
	vecSize := 4 + 3*4
	data := make([]byte, 2*vecSize)
	
	// First vector
	binary.LittleEndian.PutUint32(data[0:4], uint32(dim))
	for i, f := range vec1 {
		binary.LittleEndian.PutUint32(data[4+i*4:4+(i+1)*4], math.Float32bits(f))
	}
	
	// Second vector
	offset := vecSize
	binary.LittleEndian.PutUint32(data[offset:offset+4], uint32(dim))
	for i, f := range vec2 {
		binary.LittleEndian.PutUint32(data[offset+4+i*4:offset+4+(i+1)*4], math.Float32bits(f))
	}
	
	parsed, err := ParseFvecs(data)
	if err != nil {
		t.Fatalf("ParseFvecs failed: %v", err)
	}
	
	if len(parsed) != 2 {
		t.Fatalf("Expected 2 vectors, got %d", len(parsed))
	}
	
	expected := []Vector{
		{1, 2, 3},
		{4, 5, 6},
	}
	for i, v := range parsed {
		for j := range v {
			if math.Abs(float64(v[j]-expected[i][j])) > 1e-5 {
				t.Errorf("Vector %d[%d]: got %f, want %f", i, j, v[j], expected[i][j])
			}
		}
	}
}

func TestParseIvecs(t *testing.T) {
	dim := int32(3)
	vec1 := []int32{1, 2, 3}
	vec2 := []int32{4, 5, 6}
	
	vecSize := 4 + 3*4
	data := make([]byte, 2*vecSize)
	
	// First vector
	binary.LittleEndian.PutUint32(data[0:4], uint32(dim))
	for i, v := range vec1 {
		binary.LittleEndian.PutUint32(data[4+i*4:4+(i+1)*4], uint32(v))
	}
	
	// Second vector
	offset := vecSize
	binary.LittleEndian.PutUint32(data[offset:offset+4], uint32(dim))
	for i, v := range vec2 {
		binary.LittleEndian.PutUint32(data[offset+4+i*4:offset+4+(i+1)*4], uint32(v))
	}
	
	parsed, err := ParseIvecs(data)
	if err != nil {
		t.Fatalf("ParseIvecs failed: %v", err)
	}
	
	if len(parsed) != 2 {
		t.Fatalf("Expected 2 vectors, got %d", len(parsed))
	}
	
	expected := [][]int32{
		{1, 2, 3},
		{4, 5, 6},
	}
	for i, v := range parsed {
		for j := range v {
			if v[j] != expected[i][j] {
				t.Errorf("Vector %d[%d]: got %d, want %d", i, j, v[j], expected[i][j])
			}
		}
	}
}