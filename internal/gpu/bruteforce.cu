#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

extern "C" {

// Kernel: brute force search for batch of queries
// Each block handles one query
// Each thread computes distances for a subset of base vectors
// Uses shared memory for top-k reduction within block
__global__ void bruteforce_search_kernel(
    const float* __restrict__ base,     // [n_base, dim]
    const float* __restrict__ base_norms, // [n_base] squared L2 norms
    const float* __restrict__ queries,  // [n_queries, dim]
    const float* __restrict__ query_norms, // [n_queries] squared L2 norms
    int32_t* __restrict__ out_indices,  // [n_queries, k]
    float* __restrict__ out_scores,     // [n_queries, k]
    int n_base,
    int n_queries,
    int dim,
    int k,
    int metric_l2  // 1 = L2, 0 = Cosine (dot product)
) {
    extern __shared__ float smem[];
    // Shared memory layout:
    // float block_scores[num_threads * k]
    // int32_t block_indices[num_threads * k]

    int query_idx = blockIdx.x;
    if (query_idx >= n_queries) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const float* query = queries + query_idx * dim;
    float q_norm = query_norms[query_idx];

    // Each thread maintains its own top-k in registers
    float thread_scores[32];  // Max k=32 for register allocation
    int32_t thread_indices[32];
    int thread_k = 0;

    // Initialize with +inf (for L2, smaller is better)
    for (int i = 0; i < k; i++) {
        thread_scores[i] = __int_as_float(0x7f800000);  // +inf
        thread_indices[i] = -1;
    }

    // Stride through base vectors
    for (int base_idx = tid; base_idx < n_base; base_idx += num_threads) {
        const float* base_vec = base + base_idx * dim;
        float b_norm = base_norms[base_idx];

        // Compute distance
        float score;
        if (metric_l2) {
            // L2: ||q - b||^2 = ||q||^2 + ||b||^2 - 2(q·b)
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot += query[d] * base_vec[d];
            }
            score = q_norm + b_norm - 2.0f * dot;
        } else {
            // Cosine: negative dot product (so smaller = better for top-k merge)
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot += query[d] * base_vec[d];
            }
            score = -dot;  // negative because we want max cosine = min score
        }

        // Insert into thread-local top-k (insertion sort)
        if (thread_k < k) {
            thread_scores[thread_k] = score;
            thread_indices[thread_k] = base_idx;
            thread_k++;
            // Bubble up
            for (int j = thread_k - 1; j > 0 && thread_scores[j] < thread_scores[j-1]; j--) {
                float tmp_s = thread_scores[j];
                thread_scores[j] = thread_scores[j-1];
                thread_scores[j-1] = tmp_s;
                int32_t tmp_i = thread_indices[j];
                thread_indices[j] = thread_indices[j-1];
                thread_indices[j-1] = tmp_i;
            }
        } else if (score < thread_scores[k-1]) {
            thread_scores[k-1] = score;
            thread_indices[k-1] = base_idx;
            // Bubble down
            for (int j = k-1; j > 0 && thread_scores[j] < thread_scores[j-1]; j--) {
                float tmp_s = thread_scores[j];
                thread_scores[j] = thread_scores[j-1];
                thread_scores[j-1] = tmp_s;
                int32_t tmp_i = thread_indices[j];
                thread_indices[j] = thread_indices[j-1];
                thread_indices[j-1] = tmp_i;
            }
        }
    }

    // Store thread top-k to shared memory
    float* block_scores = smem;
    int32_t* block_indices = (int32_t*)(smem + k * num_threads);

    for (int i = 0; i < thread_k; i++) {
        block_scores[tid * k + i] = thread_scores[i];
        block_indices[tid * k + i] = thread_indices[i];
    }
    // Pad with +inf
    for (int i = thread_k; i < k; i++) {
        block_scores[tid * k + i] = __int_as_float(0x7f800000);
        block_indices[tid * k + i] = -1;
    }
    __syncthreads();

    // Block-level merge: single thread (tid=0) merges all thread top-k arrays
    if (tid == 0) {
        float merged_scores[32];
        int32_t merged_indices[32];
        for (int i = 0; i < k; i++) {
            merged_scores[i] = __int_as_float(0x7f800000);
            merged_indices[i] = -1;
        }
        int merged_k = 0;

        for (int t = 0; t < num_threads; t++) {
            for (int i = 0; i < k; i++) {
                float s = block_scores[t * k + i];
                int32_t idx = block_indices[t * k + i];
                if (idx == -1) break;

                if (merged_k < k) {
                    merged_scores[merged_k] = s;
                    merged_indices[merged_k] = idx;
                    merged_k++;
                    for (int j = merged_k - 1; j > 0 && merged_scores[j] < merged_scores[j-1]; j--) {
                        float tmp_s = merged_scores[j];
                        merged_scores[j] = merged_scores[j-1];
                        merged_scores[j-1] = tmp_s;
                        int32_t tmp_i = merged_indices[j];
                        merged_indices[j] = merged_indices[j-1];
                        merged_indices[j-1] = tmp_i;
                    }
                } else if (s < merged_scores[k-1]) {
                    merged_scores[k-1] = s;
                    merged_indices[k-1] = idx;
                    for (int j = k-1; j > 0 && merged_scores[j] < merged_scores[j-1]; j--) {
                        float tmp_s = merged_scores[j];
                        merged_scores[j] = merged_scores[j-1];
                        merged_scores[j-1] = tmp_s;
                        int32_t tmp_i = merged_indices[j];
                        merged_indices[j] = merged_indices[j-1];
                        merged_indices[j-1] = tmp_i;
                    }
                }
            }
        }

        // Write final results
        for (int i = 0; i < k; i++) {
            out_indices[query_idx * k + i] = merged_indices[i];
            out_scores[query_idx * k + i] = merged_scores[i];
        }
    }
}

// Host wrapper to launch kernel
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
) {
    // Block size: 256 threads per query
    // Grid size: n_queries blocks
    dim3 grid(n_queries);
    dim3 block(256);

    // Shared memory: k * block.x * sizeof(float) + k * block.x * sizeof(int32_t)
    size_t smem_bytes = k * block.x * (sizeof(float) + sizeof(int32_t));

    bruteforce_search_kernel<<<grid, block, smem_bytes, stream>>>(
        d_base, d_base_norms, d_queries, d_query_norms,
        d_out_indices, d_out_scores,
        n_base, n_queries, dim, k, metric_l2
    );

    return cudaGetLastError();
}

// Memory management helpers
cudaError_t cuda_malloc(void** ptr, size_t bytes) {
    return cudaMalloc(ptr, bytes);
}

cudaError_t cuda_free(void* ptr) {
    return cudaFree(ptr);
}

cudaError_t cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
}

cudaError_t cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
}

cudaError_t cuda_stream_create(cudaStream_t* stream) {
    return cudaStreamCreate(stream);
}

cudaError_t cuda_stream_destroy(cudaStream_t stream) {
    return cudaStreamDestroy(stream);
}

cudaError_t cuda_stream_synchronize(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

cudaError_t cuda_device_synchronize() {
    return cudaDeviceSynchronize();
}

cudaError_t cuda_malloc_host(void** ptr, size_t bytes) {
    return cudaMallocHost(ptr, bytes);
}

cudaError_t cuda_free_host(void* ptr) {
    return cudaFreeHost(ptr);
}

} // extern "C"