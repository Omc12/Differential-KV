#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C" {
    // Persistent Sparse Attention CUDA Kernel
    // Natively executes a single step of sparse attention using cached GPU pointers
    // to bypass the driver overhead of traditional full-grid rebuilds without deadlocking the SMs.
    __global__ void persistent_sparse_attention_kernel(
        const float* __restrict__ Q,          // Shape: [B, H, S_q, D]
        const float* __restrict__ K,          // Shape: [B, H, S_k, D]
        const float* __restrict__ V,          // Shape: [B, H, S_k, D]
        const int* __restrict__ sparse_indices, // Shape: [B, S_q, num_sparse_blocks]
        float* __restrict__ O,                // Shape: [B, H, S_q, D]
        int B, int H, int S_q, int S_k, int D, int num_sparse_blocks, int block_size,
        float scale,
        int current_step
    ) {
        int b = blockIdx.x;
        int h = blockIdx.y;
        int thread_id = threadIdx.x;

        // Query buffer
        float q_local[128];
        float accum[128];

        // Ensure S_q points to current active token slot
        int q_idx = current_step % S_q;

        // Load active query vector for this step
        int q_base = b * H * S_q * D + h * S_q * D + q_idx * D;
        for (int d = 0; d < D && d < 128; ++d) {
            q_local[d] = Q[q_base + d];
            accum[d] = 0.0f;
        }

        // Compute attention across sparse blocks
        float max_score = -1e20f;
        float scores[64];
        int index_base = b * S_q * num_sparse_blocks + q_idx * num_sparse_blocks;

        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            int block_idx = sparse_indices[index_base + sb];
            if (block_idx < 0 || block_idx * block_size >= S_k) {
                scores[sb] = -1e20f;
                continue;
            }

            // Compute QK dot product
            float sum_val = 0.0f;
            int k_base = b * H * S_k * D + h * S_k * D + (block_idx * block_size + (thread_id % block_size)) * D;
            
            // Safeguard against sequence-length out-of-bounds reads on the GPU
            if (block_idx * block_size + (thread_id % block_size) < S_k) {
                for (int d = 0; d < D && d < 128; ++d) {
                    sum_val += q_local[d] * K[k_base + d];
                }
            }
            sum_val *= scale;
            scores[sb] = sum_val;
            if (sum_val > max_score) {
                max_score = sum_val;
            }
        }

        // Softmax
        float exp_sum = 0.0f;
        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            if (scores[sb] > -1e19f) {
                scores[sb] = expf(scores[sb] - max_score);
                exp_sum += scores[sb];
            } else {
                scores[sb] = 0.0f;
            }
        }

        // Accumulate values
        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            if (scores[sb] > 0.0f) {
                float att_weight = scores[sb] / (exp_sum + 1e-6f);
                int block_idx = sparse_indices[index_base + sb];
                int v_base = b * H * S_k * D + h * S_k * D + (block_idx * block_size + (thread_id % block_size)) * D;
                
                // Safeguard against sequence-length out-of-bounds reads on the GPU
                if (block_idx * block_size + (thread_id % block_size) < S_k) {
                    for (int d = 0; d < D && d < 128; ++d) {
                        accum[d] += att_weight * V[v_base + d];
                    }
                }
            }
        }

        // Write result back contiguously
        int o_base = b * H * S_q * D + h * S_q * D + q_idx * D;
        for (int d = 0; d < D && d < 128; ++d) {
            O[o_base + d] = accum[d];
        }
    }

    // Exported function for ctypes
    __declspec(dllexport) void launch_persistent_attention(
        const float* Q,
        const float* K,
        const float* V,
        const int* sparse_indices,
        float* O,
        int B, int H, int S_q, int S_k, int D, int num_sparse_blocks, int block_size,
        float scale,
        int current_step
    ) {
        dim3 grid(B, H, 1);
        dim3 block(128, 1, 1);

        persistent_sparse_attention_kernel<<<grid, block>>>(
            Q, K, V, sparse_indices, O,
            B, H, S_q, S_k, D, num_sparse_blocks, block_size, scale,
            current_step
        );
    }
}
