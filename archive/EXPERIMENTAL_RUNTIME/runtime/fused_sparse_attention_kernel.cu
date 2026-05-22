#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C" {
    // Fused Sparse Attention CUDA Kernel
    __global__ void fused_sparse_attention_kernel(
        const float* __restrict__ Q,          // Shape: [B, H, S_q, D]
        const float* __restrict__ K,          // Shape: [B, H, S_k, D]
        const float* __restrict__ V,          // Shape: [B, H, S_k, D]
        const int* __restrict__ sparse_indices, // Shape: [B, S_q, num_sparse_blocks]
        float* __restrict__ O,                // Shape: [B, H, S_q, D]
        int B, int H, int S_q, int S_k, int D, int num_sparse_blocks, int block_size,
        float scale
    ) {
        int b = blockIdx.x;
        int h = blockIdx.y;
        int q = threadIdx.x + blockIdx.z * blockDim.x; // Query index

        if (q >= S_q) return;

        // Fetch query vector for this thread
        // Q: [B, H, S_q, D]
        // Offset: b * (H * S_q * D) + h * (S_q * D) + q * D
        int q_base = b * H * S_q * D + h * S_q * D + q * D;
        
        // Output base
        int o_base = b * H * S_q * D + h * S_q * D + q * D;

        // Local query buffer
        float q_local[128]; // Supports up to head_dim=128
        for (int d = 0; d < D && d < 128; ++d) {
            q_local[d] = Q[q_base + d];
        }

        // Shared memory or registers for attention computation
        // Loop over key blocks
        float scores[64]; // Max 64 sparse blocks supported in registry
        float max_score = -1e20f;

        // Traverse indices natively on GPU
        // sparse_indices: [B, S_q, num_sparse_blocks]
        int index_base = b * S_q * num_sparse_blocks + q * num_sparse_blocks;

        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            int block_idx = sparse_indices[index_base + sb];
            if (block_idx < 0 || block_idx * block_size >= S_k) {
                scores[sb] = -1e20f;
                continue;
            }

            // Compute dot product of Q and K for the head element of block
            // In a block, compute the attention dot product
            float sum_val = 0.0f;
            int k_base = b * H * S_k * D + h * S_k * D + (block_idx * block_size) * D;

            for (int d = 0; d < D && d < 128; ++d) {
                sum_val += q_local[d] * K[k_base + d];
            }
            sum_val *= scale;
            scores[sb] = sum_val;
            if (sum_val > max_score) {
                max_score = sum_val;
            }
        }

        // Compute Softmax denominators
        float exp_sum = 0.0f;
        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            if (scores[sb] > -1e19f) {
                scores[sb] = expf(scores[sb] - max_score);
                exp_sum += scores[sb];
            } else {
                scores[sb] = 0.0f;
            }
        }

        // Multiply with V and write to output projection
        float accum[128] = {0.0f};
        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            if (scores[sb] > 0.0f) {
                float att_weight = scores[sb] / (exp_sum + 1e-6f);
                int block_idx = sparse_indices[index_base + sb];
                int v_base = b * H * S_k * D + h * S_k * D + (block_idx * block_size) * D;

                for (int d = 0; d < D && d < 128; ++d) {
                    accum[d] += att_weight * V[v_base + d];
                }
            }
        }

        // Write contiguous output values
        for (int d = 0; d < D && d < 128; ++d) {
            O[o_base + d] = accum[d];
        }
    }

    // Exported function for ctypes
    __declspec(dllexport) void launch_fused_sparse_attention(
        const float* Q,
        const float* K,
        const float* V,
        const int* sparse_indices,
        float* O,
        int B, int H, int S_q, int S_k, int D, int num_sparse_blocks, int block_size,
        float scale
    ) {
        dim3 grid(B, H, (S_q + 127) / 128);
        dim3 block(128, 1, 1);
        
        fused_sparse_attention_kernel<<<grid, block>>>(
            Q, K, V, sparse_indices, O,
            B, H, S_q, S_k, D, num_sparse_blocks, block_size, scale
        );
        cudaDeviceSynchronize();
    }
}
