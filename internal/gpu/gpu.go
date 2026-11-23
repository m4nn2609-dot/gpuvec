package gpu

/*
#cgo LDFLAGS: -lcudart
#cgo CFLAGS: -I${SRCDIR}

#include <cuda_runtime.h>
#include <stdint.h>

// Forward declarations for CUDA functions
cudaError_t launch_bruteforce_search(
    const float* d_base,
    const float* d_base_norms,
    const float* d_queries,
    const float* d_query_norms,
    int32_t* d_out_indices,
    float* d_out_scores,
    int n_base,
    int n_queries,
    int dim,
    int k,
    int metric_l2,
    cudaStream_t stream
);

cudaError_t cuda_malloc(void** ptr, size_t bytes);
cudaError_t cuda_free(void* ptr);
cudaError_t cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t stream);
cudaError_t cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t stream);
cudaError_t cuda_stream_create(cudaStream_t* stream);
cudaError_t cuda_stream_destroy(cudaStream_t stream);
cudaError_t cuda_stream_synchronize(cudaStream_t stream);
cudaError_t cuda_device_synchronize();
cudaError_t cuda_malloc_host(void** ptr, size_t bytes);
cudaError_t cuda_free_host(void* ptr);
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// GPUError wraps CUDA errors
type GPUError struct {
	Code C.cudaError_t
	Op   string
}

func (e *GPUError) Error() string {
	return e.Op + ": " + C.GoString(C.cudaGetErrorString(e.Code))
}

func check(err C.cudaError_t, op string) error {
	if err != C.cudaSuccess {
		return &GPUError{Code: err, Op: op}
	}
	return nil
}

// GPUBuffer manages pinned host memory and/or device memory
type GPUBuffer struct {
	hostPtr   unsafe.Pointer
	devicePtr unsafe.Pointer
	sizeBytes C.size_t
	pinned    bool
}

// AllocPinned allocates page-locked (pinned) host memory for fast H2D/D2H
func AllocPinned(sizeBytes C.size_t) (*GPUBuffer, error) {
	var hostPtr unsafe.Pointer
	err := check(C.cuda_malloc_host(&hostPtr, sizeBytes), "cudaMallocHost")
	if err != nil {
		return nil, err
	}
	return &GPUBuffer{
		hostPtr:   hostPtr,
		sizeBytes: sizeBytes,
		pinned:    true,
	}, nil
}

// AllocDevice allocates device memory
func AllocDevice(sizeBytes C.size_t) (*GPUBuffer, error) {
	var devPtr unsafe.Pointer
	err := check(C.cuda_malloc(&devPtr, sizeBytes), "cudaMalloc")
	if err != nil {
		return nil, err
	}
	return &GPUBuffer{
		devicePtr: devPtr,
		sizeBytes: sizeBytes,
		pinned:    false,
	}, nil
}

// Free releases the buffer
func (b *GPUBuffer) Free() error {
	if b.hostPtr != nil && b.pinned {
		return check(C.cuda_free_host(b.hostPtr), "cudaFreeHost")
	}
	if b.devicePtr != nil {
		return check(C.cuda_free(b.devicePtr), "cudaFree")
	}
	return nil
}

// HostSlice returns a Go slice backed by the pinned host memory (float32)
func (b *GPUBuffer) HostSlice() []float32 {
	if b.hostPtr == nil {
		return nil
	}
	return unsafe.Slice((*float32)(b.hostPtr), int(b.sizeBytes/4))
}

// HostInt32Slice returns a Go slice backed by the pinned host memory (int32)
func (b *GPUBuffer) HostInt32Slice() []int32 {
	if b.hostPtr == nil {
		return nil
	}
	return unsafe.Slice((*int32)(b.hostPtr), int(b.sizeBytes/4))
}

// DevicePtr returns the device pointer
func (b *GPUBuffer) DevicePtr() unsafe.Pointer {
	return b.devicePtr
}

func (b *GPUBuffer) SetDevicePtr(ptr unsafe.Pointer) {
	b.devicePtr = ptr
}

// Stream wraps a CUDA stream
type Stream struct {
	stream C.cudaStream_t
}

func NewStream() (*Stream, error) {
	var s C.cudaStream_t
	err := check(C.cuda_stream_create(&s), "cudaStreamCreate")
	if err != nil {
		return nil, err
	}
	return &Stream{stream: s}, nil
}

func (s *Stream) Destroy() error {
	return check(C.cuda_stream_destroy(s.stream), "cudaStreamDestroy")
}

func (s *Stream) Synchronize() error {
	return check(C.cuda_stream_synchronize(s.stream), "cudaStreamSynchronize")
}

func DeviceSynchronize() error {
	return check(C.cuda_device_synchronize(), "cudaDeviceSynchronize")
}

// MemcpyH2D copies host to device asynchronously
func MemcpyH2D(dst, src *GPUBuffer, stream *Stream) error {
	return check(C.cuda_memcpy_h2d(dst.devicePtr, src.hostPtr, src.sizeBytes, stream.stream), "cudaMemcpyH2D")
}

// MemcpyD2H copies device to host asynchronously
func MemcpyD2H(dst, src *GPUBuffer, stream *Stream) error {
	return check(C.cuda_memcpy_d2h(dst.hostPtr, src.devicePtr, src.sizeBytes, stream.stream), "cudaMemcpyD2H")
}

// BruteForceSearch launches the brute force search kernel
func BruteForceSearch(
	base, baseNorms, queries, queryNorms *GPUBuffer,
	outIndices, outScores *GPUBuffer,
	nBase, nQueries, dim, k int,
	metricL2 bool,
	stream *Stream,
) error {
	metric := C.int(0)
	if metricL2 {
		metric = 1
	}
	err := check(C.launch_bruteforce_search(
		(*C.float)(base.devicePtr),
		(*C.float)(baseNorms.devicePtr),
		(*C.float)(queries.devicePtr),
		(*C.float)(queryNorms.devicePtr),
		(*C.int32_t)(outIndices.devicePtr),
		(*C.float)(outScores.devicePtr),
		C.int(nBase),
		C.int(nQueries),
		C.int(dim),
		C.int(k),
		metric,
		stream.stream,
	), "launch_bruteforce_search")
	return err
}

// PinCurrentThread pins the current OS thread (required for CUDA context)
func PinCurrentThread() {
	runtime.LockOSThread()
}

// UnpinCurrentThread unpins the current OS thread
func UnpinCurrentThread() {
	runtime.UnlockOSThread()
}

// BufferPool manages a pool of pinned host buffers for reuse
type BufferPool struct {
	buffers []*GPUBuffer
	inUse   []bool
	size    C.size_t
	mu      chan struct{}
}

func NewBufferPool(numBuffers int, sizeBytes C.size_t) (*BufferPool, error) {
	p := &BufferPool{
		buffers: make([]*GPUBuffer, numBuffers),
		inUse:   make([]bool, numBuffers),
		size:    sizeBytes,
		mu:      make(chan struct{}, 1),
	}
	p.mu <- struct{}{}

	for i := 0; i < numBuffers; i++ {
		buf, err := AllocPinned(sizeBytes)
		if err != nil {
			for j := 0; j < i; j++ {
				p.buffers[j].Free()
			}
			return nil, err
		}
		p.buffers[i] = buf
	}
	return p, nil
}

func (p *BufferPool) Acquire() *GPUBuffer {
	<-p.mu
	defer func() { p.mu <- struct{}{} }()

	for i, inUse := range p.inUse {
		if !inUse {
			p.inUse[i] = true
			return p.buffers[i]
		}
	}
	buf, _ := AllocPinned(p.size)
	return buf
}

func (p *BufferPool) Release(buf *GPUBuffer) {
	<-p.mu
	defer func() { p.mu <- struct{}{} }()

	for i, b := range p.buffers {
		if b == buf {
			p.inUse[i] = false
			return
		}
	}
	buf.Free()
}

func (p *BufferPool) Close() error {
	<-p.mu
	defer func() { p.mu <- struct{}{} }()

	var firstErr error
	for _, buf := range p.buffers {
		if err := buf.Free(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}