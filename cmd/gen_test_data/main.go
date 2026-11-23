package main

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

func main() {
	dataDir := "./data"
	if len(os.Args) > 1 {
		dataDir = os.Args[1]
	}

	os.MkdirAll(dataDir, 0o755)

	// Generate small test dataset: 100 base, 10 queries, dim=16 (tiny for quick test)
	nBase := 100
	nQuery := 10
	dim := 16
	k := 5

	println("Generating base vectors...")
	writeVectors(filepath.Join(dataDir, "sift_base.fvecs"), nBase, dim)

	println("Generating query vectors...")
	writeVectors(filepath.Join(dataDir, "sift_query.fvecs"), nQuery, dim)

	println("Generating ground truth...")
	writeGroundTruth(filepath.Join(dataDir, "sift_groundtruth.ivecs"), nQuery, nBase, k)

	println("Generated test dataset in", dataDir)
}

func writeVectors(path string, n, dim int) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	buf := make([]byte, 4+dim*4)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(buf[:4], uint32(dim))
		for j := 0; j < dim; j++ {
			binary.LittleEndian.PutUint32(buf[4+j*4:4+(j+1)*4], uint32(math.Float32bits(rand.Float32())))
		}
		f.Write(buf)
	}
}

func writeGroundTruth(path string, nQuery, nBase, k int) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	buf := make([]byte, 4+k*4)
	for i := 0; i < nQuery; i++ {
		binary.LittleEndian.PutUint32(buf[:4], uint32(k))
		for j := 0; j < k; j++ {
			binary.LittleEndian.PutUint32(buf[4+j*4:4+(j+1)*4], uint32(rand.Intn(nBase)))
		}
		f.Write(buf)
	}
}