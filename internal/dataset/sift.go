package dataset

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/user/gpuvec/pkg/vector"
)

// SIFT1M paths
const (
	SIFTBaseURL   = "ftp://ftp.irisa.fr/local/texmex/corpus/"
	SIFTBaseFile  = "sift_base.fvecs.gz"
	SIFTQueryFile = "sift_query.fvecs.gz"
	SIFTGroundFile = "sift_groundtruth.ivecs.gz"
)

// Dataset holds loaded vectors and ground truth.
type Dataset struct {
	Name        string
	Base        []vector.Vector
	Queries     []vector.Vector
	GroundTruth [][]int32
	Dim         int
	Metric      string // "L2" or "Cosine"
}

// LoadSIFT1M downloads (if needed) and loads the SIFT1M dataset.
func LoadSIFT1M(dataDir string) (*Dataset, error) {
	basePath := filepath.Join(dataDir, "sift_base.fvecs")
	queryPath := filepath.Join(dataDir, "sift_query.fvecs")
	groundPath := filepath.Join(dataDir, "sift_groundtruth.ivecs")

	// Download if missing
	if _, err := os.Stat(basePath); err != nil {
		if err := downloadSIFT1M(dataDir); err != nil {
			return nil, err
		}
	}

	// Load base vectors
	base, err := vector.ReadFvecs(basePath)
	if err != nil {
		return nil, fmt.Errorf("load base: %w", err)
	}

	// Load queries
	queries, err := vector.ReadFvecs(queryPath)
	if err != nil {
		return nil, fmt.Errorf("load queries: %w", err)
	}

	// Load ground truth
	groundTruth, err := vector.ReadIvecs(groundPath)
	if err != nil {
		return nil, fmt.Errorf("load ground truth: %w", err)
	}

	if len(base) == 0 || len(queries) == 0 || len(groundTruth) == 0 {
		return nil, errors.New("empty dataset")
	}

	dim := base[0].Dim()
	for i, v := range base {
		if v.Dim() != dim {
			return nil, fmt.Errorf("base vector %d has wrong dim", i)
		}
	}
	for i, v := range queries {
		if v.Dim() != dim {
			return nil, fmt.Errorf("query vector %d has wrong dim", i)
		}
	}

	return &Dataset{
		Name:        "SIFT1M",
		Base:        base,
		Queries:     queries,
		GroundTruth: groundTruth,
		Dim:         dim,
		Metric:      "L2",
	}, nil
}

// downloadSIFT1M downloads and extracts SIFT1M files.
func downloadSIFT1M(dataDir string) error {
	files := []struct {
		url  string
		name string
	}{
		{SIFTBaseURL + SIFTBaseFile, "sift_base.fvecs"},
		{SIFTBaseURL + SIFTQueryFile, "sift_query.fvecs"},
		{SIFTBaseURL + SIFTGroundFile, "sift_groundtruth.ivecs"},
	}

	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return err
	}

	client := &http.Client{Timeout: 0} // no timeout for large files

	for _, f := range files {
		gzPath := filepath.Join(dataDir, f.name+".gz")
		outPath := filepath.Join(dataDir, f.name)

		if _, err := os.Stat(outPath); err == nil {
			continue // already extracted
		}

		// Download .gz
		fmt.Printf("Downloading %s...\n", f.name)
		resp, err := client.Get(f.url)
		if err != nil {
			return fmt.Errorf("download %s: %w", f.name, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("download %s: status %d", f.name, resp.StatusCode)
		}

		gzFile, err := os.Create(gzPath)
		if err != nil {
			return err
		}
		_, err = io.Copy(gzFile, resp.Body)
		gzFile.Close()
		if err != nil {
			return err
		}

		// Extract
		fmt.Printf("Extracting %s...\n", f.name)
		if err := extractGzip(gzPath, outPath); err != nil {
			return err
		}

		// Clean up .gz
		os.Remove(gzPath)
	}
	return nil
}

func extractGzip(src, dst string) error {
	gzFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer gzFile.Close()

	gzReader, err := gzip.NewReader(gzFile)
	if err != nil {
		return err
	}
	defer gzReader.Close()

	outFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, gzReader)
	return err
}

// Subset returns a subset of the dataset (for quick testing).
func (d *Dataset) Subset(nBase, nQueries int) *Dataset {
	if nBase > len(d.Base) {
		nBase = len(d.Base)
	}
	if nQueries > len(d.Queries) {
		nQueries = len(d.Queries)
	}
	return &Dataset{
		Name:        d.Name + "-subset",
		Base:        d.Base[:nBase],
		Queries:     d.Queries[:nQueries],
		GroundTruth: d.GroundTruth[:nQueries],
		Dim:         d.Dim,
		Metric:      d.Metric,
	}
}

// NormalizeL2 normalizes all vectors in-place for cosine similarity.
func (d *Dataset) NormalizeL2() {
	for i := range d.Base {
		d.Base[i].NormalizeL2()
	}
	for i := range d.Queries {
		d.Queries[i].NormalizeL2()
	}
	d.Metric = "Cosine"
}