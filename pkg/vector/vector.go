package vector

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"unsafe"
)

// Vector is a float32 slice with alignment guarantees for SIMD/AVX.
type Vector []float32

// Dim returns the dimension of the vector.
func (v Vector) Dim() int { return len(v) }

// NormalizeL2 normalizes the vector in-place to unit L2 norm.
func (v Vector) NormalizeL2() {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return
	}
	inv := 1.0 / float32(math.Sqrt(float64(sum)))
	for i := range v {
		v[i] *= inv
	}
}

// SquaredL2Norm returns the squared L2 norm of the vector.
func (v Vector) SquaredL2Norm() float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return sum
}

// DotProduct computes the dot product of two vectors.
// Panics if dimensions don't match.
func (v Vector) DotProduct(other Vector) float32 {
	if len(v) != len(other) {
		panic("vector dimension mismatch")
	}
	var sum float32
	for i, x := range v {
		sum += x * other[i]
	}
	return sum
}

// SquaredL2Distance computes ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a·b).
// Requires precomputed squared norms for a and b.
func SquaredL2Distance(a, b Vector, normA, normB float32) float32 {
	return normA + normB - 2*a.DotProduct(b)
}

// CosineSimilarity computes cosine similarity assuming vectors are L2-normalized.
// Returns value in [-1, 1].
func CosineSimilarity(a, b Vector) float32 {
	return a.DotProduct(b)
}

// ReadFvecs reads vectors from an fvecs file (SIFT/GIST format).
// Each vector: int32 dim, followed by dim float32 values.
func ReadFvecs(path string) ([]Vector, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseFvecs(data)
}

// ParseFvecs parses fvecs format from a byte slice.
func ParseFvecs(data []byte) ([]Vector, error) {
	if len(data) < 4 {
		return nil, errors.New("file too small")
	}
	dim := int(binary.LittleEndian.Uint32(data[:4]))
	if dim <= 0 {
		return nil, errors.New("invalid dimension")
	}
	vecSize := 4 + dim*4
	if len(data)%vecSize != 0 {
		return nil, errors.New("file size not multiple of vector size")
	}
	n := len(data) / vecSize
	out := make([]Vector, n)
	offset := 0
	for i := 0; i < n; i++ {
		// Verify dimension matches
		d := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		if d != dim {
			return nil, errors.New("dimension mismatch in file")
		}
		v := make(Vector, dim)
		// Use unsafe for fast conversion
		src := data[offset+4 : offset+vecSize]
		dst := unsafe.Slice((*float32)(unsafe.Pointer(&v[0])), dim)
		// binary.Read is slow; use direct memory copy via casting
		for j := 0; j < dim; j++ {
			dst[j] = math.Float32frombits(binary.LittleEndian.Uint32(src[j*4 : j*4+4]))
		}
		out[i] = v
		offset += vecSize
	}
	return out, nil
}

// ReadIvecs reads int32 vectors from an ivecs file (ground truth format).
func ReadIvecs(path string) ([][]int32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseIvecs(data)
}

// ParseIvecs parses ivecs format from a byte slice.
func ParseIvecs(data []byte) ([][]int32, error) {
	if len(data) < 4 {
		return nil, errors.New("file too small")
	}
	dim := int(binary.LittleEndian.Uint32(data[:4]))
	if dim <= 0 {
		return nil, errors.New("invalid dimension")
	}
	vecSize := 4 + dim*4
	if len(data)%vecSize != 0 {
		return nil, errors.New("file size not multiple of vector size")
	}
	n := len(data) / vecSize
	out := make([][]int32, n)
	offset := 0
	for i := 0; i < n; i++ {
		d := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		if d != dim {
			return nil, errors.New("dimension mismatch in file")
		}
		v := make([]int32, dim)
		src := data[offset+4 : offset+vecSize]
		for j := 0; j < dim; j++ {
			v[j] = int32(binary.LittleEndian.Uint32(src[j*4 : j*4+4]))
		}
		out[i] = v
		offset += vecSize
	}
	return out, nil
}