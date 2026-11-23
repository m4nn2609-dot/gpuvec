package index

import (
	"context"
	"runtime"
	"sync"
	"time"

	"github.com/user/gpuvec/pkg/vector"
)

// BruteForceIndex is an exact linear scan index (the oracle).
type BruteForceIndex struct {
	vectors     []vector.Vector
	norms       []float32 // precomputed squared L2 norms
	dim         int
	metric      Metric
	buildTime   time.Duration
	mu          sync.RWMutex
}

// Metric defines the distance metric.
type Metric int

const (
	MetricL2 Metric = iota
	MetricCosine
)

// NewBruteForce creates a new brute force index.
func NewBruteForce(metric Metric) *BruteForceIndex {
	return &BruteForceIndex{
		metric: metric,
	}
}

// Build constructs the index by storing vectors and precomputing norms.
func (idx *BruteForceIndex) Build(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	start := time.Now()
	n := len(vectors)
	idx.vectors = make([]vector.Vector, n)
	idx.norms = make([]float32, n)
	if n == 0 {
		idx.dim = 0
		idx.buildTime = time.Since(start)
		return nil
	}
	dim := vectors[0].Dim()
	idx.dim = dim

	// Copy vectors and compute norms in parallel
	numWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	chunkSize := (n + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startIdx := w * chunkSize
		endIdx := min(startIdx+chunkSize, n)
		if startIdx >= endIdx {
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
				// Make a copy to own the memory
				idx.vectors[i] = make(vector.Vector, dim)
				copy(idx.vectors[i], v)
				if idx.metric == MetricL2 {
					idx.norms[i] = v.SquaredL2Norm()
				}
			}
		}(startIdx, endIdx)
	}
	wg.Wait()
	idx.buildTime = time.Since(start)
	return nil
}

// Search performs exact linear scan search.
func (idx *BruteForceIndex) Search(ctx context.Context, queries []vector.Vector, k int) (SearchResults, error) {
	start := time.Now()
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.vectors == nil {
		return SearchResults{}, ErrNotBuilt
	}

	n := len(idx.vectors)
	q := len(queries)
	if k > n {
		k = n
	}

	results := make([][]SearchResult, q)

	// Handle empty index
	if n == 0 {
		return SearchResults{
			Results: results,
			Latency: time.Since(start),
		}, nil
	}

	// Parallelize over queries
	numWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	chunkSize := (q + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startIdx := w * chunkSize
		endIdx := min(startIdx+chunkSize, q)
		if startIdx >= endIdx {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for qi := s; qi < e; qi++ {
				select {
				case <-ctx.Done():
					return
				default:
				}
				results[qi] = idx.searchSingle(queries[qi], k)
			}
		}(startIdx, endIdx)
	}
	wg.Wait()

	return SearchResults{
		Results: results,
		Latency: time.Since(start),
	}, nil
}

func (idx *BruteForceIndex) searchSingle(query vector.Vector, k int) []SearchResult {
	n := len(idx.vectors)
	if query.Dim() != idx.dim {
		panic("query dimension mismatch")
	}

	// Precompute query norm for L2
	var qNorm float32
	if idx.metric == MetricL2 {
		qNorm = query.SquaredL2Norm()
	}

	// Use a simple top-k heap (slice + insertion sort for small k)
	// For k <= 100, insertion sort is faster than heap.
	topK := make([]SearchResult, 0, k)

	for i := 0; i < n; i++ {
		var score float32
		if idx.metric == MetricL2 {
			score = vector.SquaredL2Distance(query, idx.vectors[i], qNorm, idx.norms[i])
		} else {
			score = -vector.CosineSimilarity(query, idx.vectors[i]) // negative for min-heap behavior
		}

		// Insert into top-k
		if len(topK) < k {
			topK = append(topK, SearchResult{Index: int32(i), Score: score})
			// Bubble up
			j := len(topK) - 1
			for j > 0 && topK[j].Score < topK[j-1].Score {
				topK[j], topK[j-1] = topK[j-1], topK[j]
				j--
			}
		} else if score < topK[k-1].Score {
			topK[k-1] = SearchResult{Index: int32(i), Score: score}
			// Bubble down
			j := k - 1
			for j > 0 && topK[j].Score < topK[j-1].Score {
				topK[j], topK[j-1] = topK[j-1], topK[j]
				j--
			}
		}
	}
	return topK
}

// Name returns the index name.
func (idx *BruteForceIndex) Name() string {
	if idx.metric == MetricL2 {
		return "CPU-BruteForce-L2"
	}
	return "CPU-BruteForce-Cosine"
}

// Stats returns index statistics.
func (idx *BruteForceIndex) Stats() Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	mem := int64(len(idx.vectors)) * int64(idx.dim) * 4
	if idx.norms != nil {
		mem += int64(len(idx.norms)) * 4
	}
	return Stats{
		Name:        idx.Name(),
		NumVectors:  len(idx.vectors),
		Dim:         idx.dim,
		MemoryBytes: mem,
		BuildTime:   idx.buildTime,
	}
}

// Close releases resources.
func (idx *BruteForceIndex) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.vectors = nil
	idx.norms = nil
	return nil
}

var ErrNotBuilt = &IndexError{Op: "search", Msg: "index not built"}

type IndexError struct {
	Op  string
	Msg string
}

func (e *IndexError) Error() string { return e.Op + ": " + e.Msg }