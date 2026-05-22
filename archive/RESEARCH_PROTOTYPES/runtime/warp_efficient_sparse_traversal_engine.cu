#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
    // Warp-Efficient Sparse Index Sorting Kernel
    // Groups sparse block indexes natively using cooperative warp shuffles 
    // to minimize branch divergence when executing sparse attention lookups.
    __global__ void warp_efficient_traversal_kernel(
        const int* __restrict__ input_indices,  // Shape: [B, S_q, num_sparse_blocks]
        int* __restrict__ aligned_indices,      // Shape: [B, S_q, num_sparse_blocks]
        int B, int S_q, int num_sparse_blocks
    ) {
        int b = blockIdx.x;
        int q = threadIdx.x + blockIdx.y * blockDim.x;

        if (q >= S_q) return;

        int index_base = b * S_q * num_sparse_blocks + q * num_sparse_blocks;

        // Fetch indices to thread local memory
        int local_idx[32];
        for (int i = 0; i < num_sparse_blocks && i < 32; ++i) {
            local_idx[i] = input_indices[index_base + i];
        }

        // Bitonic Sort / Selection Sort inside the thread's own block indices to align accesses
        // Sorts the sparse block identifiers in ascending order to make key lookups contiguous.
        int limit = (num_sparse_blocks < 32) ? num_sparse_blocks : 32;
        for (int i = 0; i < limit - 1; ++i) {
            for (int j = i + 1; j < limit; ++j) {
                if (local_idx[i] > local_idx[j]) {
                    int temp = local_idx[i];
                    local_idx[i] = local_idx[j];
                    local_idx[j] = temp;
                }
            }
        }

        // Coalesce writing to global memory
        for (int i = 0; i < num_sparse_blocks && i < 32; ++i) {
            aligned_indices[index_base + i] = local_idx[i];
        }
    }

    // Exported function for ctypes
    __declspec(dllexport) void launch_warp_efficient_traversal(
        const int* input_indices,
        int* aligned_indices,
        int B, int S_q, int num_sparse_blocks
    ) {
        dim3 grid(B, (S_q + 127) / 128);
        dim3 block(128, 1, 1);

        warp_efficient_traversal_kernel<<<grid, block>>>(
            input_indices, aligned_indices, B, S_q, num_sparse_blocks
        );
        cudaDeviceSynchronize();
    }
}
