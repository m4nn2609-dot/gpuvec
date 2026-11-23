package index

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/user/gpuvec/internal/gpu"
	"github.com/user/gpuvec/pkg/vector"
)

// C.size_t is used for CUDA allocations
// #include <stddef.h>
import "C"
)

// GPUBruteForceIndex implements exact search on GPU.
type GPUBruteForceIndex struct {
	baseVecs    []vector.Vector
	baseNorms   []float32
	dim         int
	metric      Metric
	buildTime   time.Duration

	// GPU resources
	dBase       *gpu.GPUBuffer
	dBaseNorms  *gpu.GPUBuffer
	stream      *gpu.Stream
	mu          sync.Mutex
	closed      bool
}

// NewGPUBruteForce creates a new GPU brute force index.
func NewGPUBruteForce(metric Metric) *GPUBruteForceIndex {
	return &GPUBruteForceIndex{
		metric: metric,
	}
}

// Build constructs the index: copies vectors to GPU.
func (idx *GPUBruteForceIndex) Build(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrClosed
	}

	start := time.Now()
	n := len(vectors)
	if n == 0 {
		idx.dim = 0
		idx.buildTime = time.Since(start)
		return nil
	}

	dim := vectors[0].Dim()
	idx.dim = dim
	idx.baseVecs = make([]vector.Vector, n)
	idx.baseNorms = make([]float32, n)

	// Copy vectors and compute norms on host
	numWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	chunkSize := (n + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		s := w * chunkSize
		e := min(s+chunkSize, n)
		if s >= e {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				v := vectors[i]
				if v.Dim() != dim {
					panic("dimension mismatch during build")
				}
				idx.baseVecs[i] = make(vector.Vector, dim)
				copy(idx.baseVecs[i], v)
				if idx.metric == MetricL2 {
					idx.baseNorms[i] = v.SquaredL2Norm()
				}
			}
		}(s, e)
	}
	wg.Wait()

	// Pin current thread for CUDA context
	gpu.PinCurrentThread()
	defer gpu.UnpinCurrentThread()

	// Allocate device memory for base vectors and norms
	baseBytes := C.size_t(n * dim * 4)
	normsBytes := C.size_t(n * 4)

	var err error
	idx.dBase, err = gpu.AllocDevice(baseBytes)
	if err != nil {
		return fmt.Errorf("alloc d_base: %w", err)
	}

	idx.dBaseNorms, err = gpu.AllocDevice(normsBytes)
	if err != nil {
		idx.dBase.Free()
		return fmt.Errorf("alloc d_base_norms: %w", err)
	}

	// Create stream
	idx.stream, err = gpu.NewStream()
	if err != nil {
		idx.dBase.Free()
		idx.dBaseNorms.Free()
		return fmt.Errorf("create stream: %w", err)
	}

	// Allocate pinned host buffers for initial upload
	hBase, err := gpu.AllocPinned(baseBytes)
	if err != nil {
		idx.Close()
		return fmt.Errorf("alloc host base: %w", err)
	}
	defer hBase.Free()

	hNorms, err := gpu.AllocPinned(normsBytes)
	if err != nil {
		idx.Close()
		return fmt.Errorf("alloc host norms: %w", err)
	}
	defer hNorms.Free()

	// Flatten vectors into host buffers
	hBaseSlice := hBase.HostSlice()
	hNormsSlice := hNorms.HostSlice()
	offset := 0
	for i := 0; i < n; i++ {
		copy(hBaseSlice[offset:offset+dim], idx.baseVecs[i])
		hNormsSlice[i] = idx.baseNorms[i]
		offset += dim
	}

	// Upload to device
	if err := gpu.MemcpyH2D(idx.dBase, hBase, idx.stream); err != nil {
		idx.Close()
		return fmt.Errorf("H2D base: %w", err)
	}
	if err := gpu.MemcpyH2D(idx.dBaseNorms, hNorms, idx.stream); err != nil {
		idx.Close()
		return fmt.Errorf("H2D norms: %w", err)
	}
	if err := idx.stream.Synchronize(); err != nil {
		idx.Close()
		return fmt.Errorf("sync after upload: %w", err)
	}

	idx.buildTime = time.Since(start)
	return nil
}

// Search performs search on GPU.
func (idx *GPUBruteForceIndex) Search(ctx context.Context, queries []vector.Vector, k int) (SearchResults, error) {
	return idx.searchBatch(ctx, queries, k)
}

func (idx *GPUBruteForceIndex) searchBatch(ctx context.Context, queries []vector.Vector, k int) (SearchResults, error) {
	start := time.Now()
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return SearchResults{}, ErrClosed
	}
	if idx.dBase == nil {
		return SearchResults{}, ErrNotBuilt
	}

	nQueries := len(queries)
	if nQueries == 0 {
		return SearchResults{Results: [][]SearchResult{}, Latency: 0}, nil
	}
	if k > len(idx.baseVecs) {
		k = len(idx.baseVecs)
	}
	if k <= 0 {
		k = 1
	}

	// Pin thread for CUDA
	gpu.PinCurrentThread()
	defer gpu.UnpinCurrentThread()

	// Allocate pinned host buffers for queries and outputs
	queryVecBytes := C.size_t(nQueries * idx.dim * 4)
	queryNormsBytes := C.size_t(nQueries * 4)
	outIdxBytes := C.size_t(nQueries * k * 4)
	outScoreBytes := C.size_t(nQueries * k * 4)

	hQueries, err := gpu.AllocPinned(queryVecBytes)
	if err != nil {
		return SearchResults{}, err
	}
	defer hQueries.Free()

	hQueryNorms, err := gpu.AllocPinned(queryNormsBytes)
	if err != nil {
		return SearchResults{}, err
	}
	defer hQueryNorms.Free()

	hOutIndices, err := gpu.AllocPinned(outIdxBytes)
	if err != nil {
		return SearchResults{}, err
	}
	defer hOutIndices.Free()

	hOutScores, err := gpu.AllocPinned(outScoreBytes)
	if err != nil {
		return SearchResults{}, err
	}
	defer hOutScores.Free()

	// Allocate device buffers for query and output
	dQueries, err := gpu.AllocDevice(queryVecBytes)
	if err != nil {
		return SearchResults{}, err
	}
	defer dQueries.Free()

	dQueryNorms, err := gpu.AllocDevice(queryNormsBytes)
	if err != nil {
		return SearchResults{}, err
	}
	defer dQueryNorms.Free()

	dOutIndices, err := gpu.AllocDevice(outIdxBytes)
	if err != nil {
		return SearchResults{}, err
	}
	defer dOutIndices.Free()

	dOutScores, err := gpu.AllocDevice(outScoreBytes)
	if err != nil {
		return SearchResults{}, err
	}
	defer dOutScores.Free()

	// Prepare query data
	hQueriesSlice := hQueries.HostSlice()
	hQueryNormsSlice := hQueryNorms.HostSlice()
	for i, q := range queries {
		if q.Dim() != idx.dim {
			return SearchResults{}, fmt.Errorf("query %d: dimension mismatch", i)
		}
		copy(hQueriesSlice[i*idx.dim:(i+1)*idx.dim], q)
		if idx.metric == MetricL2 {
			hQueryNormsSlice[i] = q.SquaredL2Norm()
		} else {
			hQueryNormsSlice[i] = 0 // unused for cosine
		}
	}

	// H2D
	if err := gpu.MemcpyH2D(dQueries, hQueries, idx.stream); err != nil {
		return SearchResults{}, err
	}
	if err := gpu.MemcpyH2D(dQueryNorms, hQueryNorms, idx.stream); err != nil {
		return SearchResults{}, err
	}

	// Launch kernel
	metricL2 := idx.metric == MetricL2
	if err := gpu.BruteForceSearch(
		idx.dBase, idx.dBaseNorms,
		dQueries, dQueryNorms,
		dOutIndices, dOutScores,
		len(idx.baseVecs), nQueries, idx.dim, k,
		metricL2,
		idx.stream,
	); err != nil {
		return SearchResults{}, err
	}

	// D2H
	if err := gpu.MemcpyD2H(hOutIndices, dOutIndices, idx.stream); err != nil {
		return SearchResults{}, err
	}
	if err := gpu.MemcpyD2H(hOutScores, dOutScores, idx.stream); err != nil {
		return SearchResults{}, err
	}
	if err := idx.stream.Synchronize(); err != nil {
		return SearchResults{}, err
	}

	// Collect results
	hOutIndicesSlice := hOutIndices.HostInt32Slice()
	hOutScoresSlice := hOutScores.HostSlice()

	results := make([][]SearchResult, nQueries)
	for qi := 0; qi < nQueries; qi++ {
		results[qi] = make([]SearchResult, k)
		for i := 0; i < k; i++ {
			idxVal := hOutIndicesSlice[qi*k+i]
			score := hOutScoresSlice[qi*k+i]
			// For cosine, score is negative dot product; convert back
			if !metricL2 {
				score = -score
			}
			results[qi][i] = SearchResult{Index: idxVal, Score: score}
		}
	}

	return SearchResults{
		Results: results,
		Latency: time.Since(start),
	}, nil
}

func (idx *GPUBruteForceIndex) Name() string {
	if idx.metric == MetricL2 {
		return "GPU-BruteForce-L2"
	}
	return "GPU-BruteForce-Cosine"
}

func (idx *GPUBruteForceIndex) Stats() Stats {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	mem := int64(len(idx.baseVecs)) * int64(idx.dim) * 4
	mem += int64(len(idx.baseNorms)) * 4
	if idx.dBase != nil {
		mem += int64(idx.dBase.sizeBytes)
	}
	if idx.dBaseNorms != nil {
		mem += int64(idx.dBaseNorms.sizeBytes)
	}
	return Stats{
		Name:        idx.Name(),
		NumVectors:  len(idx.baseVecs),
		Dim:         idx.dim,
		MemoryBytes: mem,
		BuildTime:   idx.buildTime,
	}
}

func (idx *GPUBruteForceIndex) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return nil
	}
	idx.closed = true

	var errs []error
	if idx.dBase != nil {
		if err := idx.dBase.Free(); err != nil {
			errs = append(errs, err)
		}
	}
	if idx.dBaseNorms != nil {
		if err := idx.dBaseNorms.Free(); err != nil {
			errs = append(errs, err)
		}
	}
	if idx.stream != nil {
		if err := idx.stream.Destroy(); err != nil {
			errs = append(errs, err)
		}
	}
	gpu.UnpinCurrentThread()

	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

var ErrClosed = &IndexError{Op: "search", Msg: "index closed"}
var ErrNotBuilt = &IndexError{Op: "search", Msg: "index not built"}