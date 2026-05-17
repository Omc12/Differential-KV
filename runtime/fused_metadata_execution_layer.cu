#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
    // Fused Metadata Generation CUDA Kernel
    // Natively parses attention confidence scores and produces sparse block index maps 
    // entirely on the GPU, avoiding any expensive host-device CPU synchronization roundtrips.
    __global__ void fused_metadata_kernel(
        const float* __restrict__ attention_scores, // Shape: [B, H, S_k]
        int* __restrict__ sparse_indices,           // Shape: [B, S_q, num_sparse_blocks]
        int B, int H, int S_q, int S_k, int num_sparse_blocks, int block_size,
        float confidence_threshold
    ) {
        int b = blockIdx.x;
        int q = threadIdx.x + blockIdx.y * blockDim.x;

        if (q >= S_q) return;

        // Output index cursor for this query
        int out_base = b * S_q * num_sparse_blocks + q * num_sparse_blocks;

        // Traverse keys and select blocks exceeding confidence threshold natively
        int collected_count = 0;
        int num_key_blocks = S_k / block_size;

        // Scan attention scores for the first head (or aggregate across heads)
        int score_base = b * H * S_k + 0 * S_k; // query head 0 score for selection

        for (int kb = 0; kb < num_key_blocks && collected_count < num_sparse_blocks; ++kb) {
            // Fetch score for this block head
            float val = attention_scores[score_base + kb * block_size];
            
            // Fused confidence routing threshold check
            if (val >= confidence_threshold) {
                sparse_indices[out_base + collected_count] = kb;
                collected_count++;
            }
        }

        // Fill remaining slots with padding index (-1)
        for (int i = collected_count; i < num_sparse_blocks; ++i) {
            sparse_indices[out_base + i] = -1;
        }
    }

    // Exported function for ctypes
    __declspec(dllexport) void launch_fused_metadata(
        const float* attention_scores,
        int* sparse_indices,
        int B, int H, int S_q, int S_k, int num_sparse_blocks, int block_size,
        float confidence_threshold
    ) {
        dim3 grid(B, (S_q + 127) / 128);
        dim3 block(128, 1, 1);

        fused_metadata_kernel<<<grid, block>>>(
            attention_scores, sparse_indices, B, H, S_q, S_k, num_sparse_blocks, block_size, confidence_threshold
        );
        cudaDeviceSynchronize();
    }
}
