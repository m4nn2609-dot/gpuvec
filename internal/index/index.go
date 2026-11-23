package index

import (
	"context"
	"time"

	"github.com/user/gpuvec/pkg/vector"
)

// SearchResult holds a single search result.
type SearchResult struct {
	Index int32
	Score float32
}

// SearchResults holds top-k results for a batch of queries.
type SearchResults struct {
	Results [][]SearchResult // [query][k]
	Latency time.Duration
}

// Index is the interface for all vector search backends.
type Index interface {
	// Build constructs the index from the given vectors.
	// For IVF, this includes k-means training.
	Build(vectors []vector.Vector) error

	// Search searches for top-k nearest neighbors for each query.
	// Returns results and per-query latency breakdown.
	Search(ctx context.Context, queries []vector.Vector, k int) (SearchResults, error)

	// Name returns a human-readable name for this backend.
	Name() string

	// Stats returns backend-specific statistics (memory usage, index size, etc.).
	Stats() Stats

	// Close releases any resources (GPU memory, threads, etc.).
	Close() error
}

// Stats holds index statistics.
type Stats struct {
	Name         string
	NumVectors   int
	Dim          int
	MemoryBytes  int64
	BuildTime    time.Duration
	Extra        map[string]any
}

// SearchConfig holds search-time configuration.
type SearchConfig struct {
	K       int
	NProbe  int // for IVF
	EFSearch int // for HNSW
}

// BatchSearcher extends Index with batch search capability.
// This is for backends that benefit from batching (GPU).
type BatchSearcher interface {
	Index
	// SearchBatch searches multiple queries in a single call, optimized for throughput.
	SearchBatch(ctx context.Context, queries []vector.Vector, k int) (SearchResults, error)
}