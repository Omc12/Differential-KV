#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C" {
    // Shared Memory Sparse Tile CUDA Kernel
    // Stages block-tiled Keys/Values natively into GPU Shared Memory to minimize DRAM memory latency stalls.
    __global__ void shared_memory_sparse_tile_kernel(
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
        int q = threadIdx.x + blockIdx.z * blockDim.x;

        if (q >= S_q) return;

        // Thread local query registers
        float q_local[128];
        int q_base = b * H * S_q * D + h * S_q * D + q * D;
        for (int d = 0; d < D && d < 128; ++d) {
            q_local[d] = Q[q_base + d];
        }

        // Shared memory tiling workspace (16 threads cooperatively load sparse tile data)
        // Stages a block size of 16 keys/values with dimension 128
        __shared__ float s_key_tile[16][128];
        __shared__ float s_val_tile[16][128];

        float max_score = -1e20f;
        float scores[64];
        int index_base = b * S_q * num_sparse_blocks + q * num_sparse_blocks;

        // Loop over the scheduled blocks
        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            int block_idx = sparse_indices[index_base + sb];
            if (block_idx < 0 || block_idx * block_size >= S_k) {
                scores[sb] = -1e20f;
                continue;
            }

            // Cooperative Shared Memory loading
            // Load key/value block elements cooperatively to local shared memory tile
            int thread_id = threadIdx.x;
            int load_idx = thread_id % block_size; // cooperatively load elements of the block

            int k_base = b * H * S_k * D + h * S_k * D + (block_idx * block_size + load_idx) * D;
            for (int d = 0; d < D && d < 128; ++d) {
                // Safeguard against sequence-length out-of-bounds reads on the GPU
                if (block_idx * block_size + load_idx < S_k) {
                    s_key_tile[load_idx][d] = K[k_base + d];
                    s_val_tile[load_idx][d] = V[k_base + d];
                } else {
                    s_key_tile[load_idx][d] = 0.0f;
                    s_val_tile[load_idx][d] = 0.0f;
                }
            }
            __syncthreads(); // Ensure cooperative load is completed

            // Perform dot product using the shared memory cache
            float sum_val = 0.0f;
            for (int d = 0; d < D && d < 128; ++d) {
                sum_val += q_local[d] * s_key_tile[load_idx][d];
            }
            sum_val *= scale;
            scores[sb] = sum_val;
            if (sum_val > max_score) {
                max_score = sum_val;
            }
            __syncthreads(); // Sync before next tile iteration overwrite
        }

        // Fused Softmax denoms
        float exp_sum = 0.0f;
        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            if (scores[sb] > -1e19f) {
                scores[sb] = expf(scores[sb] - max_score);
                exp_sum += scores[sb];
            } else {
                scores[sb] = 0.0f;
            }
        }

        // Value aggregation
        float accum[128] = {0.0f};
        for (int sb = 0; sb < num_sparse_blocks && sb < 64; ++sb) {
            if (scores[sb] > 0.0f) {
                float att_weight = scores[sb] / (exp_sum + 1e-6f);
                int block_idx = sparse_indices[index_base + sb];
                int thread_id = threadIdx.x;
                int load_idx = thread_id % block_size;

                // Read from local shared memory cache first
                for (int d = 0; d < D && d < 128; ++d) {
                    accum[d] += att_weight * s_val_tile[load_idx][d];
                }
            }
        }

        // Write output
        int o_base = b * H * S_q * D + h * S_q * D + q * D;
        for (int d = 0; d < D && d < 128; ++d) {
            O[o_base + d] = accum[d];
        }
    }

    // Exported function for ctypes
    __declspec(dllexport) void launch_shared_memory_sparse_tile(
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

        shared_memory_sparse_tile_kernel<<<grid, block>>>(
            Q, K, V, sparse_indices, O,
            B, H, S_q, S_k, D, num_sparse_blocks, block_size, scale
        );
        cudaDeviceSynchronize();
    }
}
