#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
    // Persistent Sparse Cache Accumulation CUDA Kernel
    // Natively preserves and accumulates intermediate query/key states inside 
    // persistent GPU buffers to bypass host-side kernel launch churn.
    __global__ void persistent_sparse_kernel(
        const float* __restrict__ input,
        float* __restrict__ persistent_buffer,
        int B, int Size
    ) {
        int b = blockIdx.x;
        int idx = threadIdx.x + blockIdx.y * blockDim.x;

        if (idx >= Size) return;

        int offset = b * Size + idx;
        
        // Accumulate and preserve state natively on GPU
        persistent_buffer[offset] = persistent_buffer[offset] * 0.95f + input[offset] * 0.05f;
    }

    // Exported function for ctypes
    __declspec(dllexport) void launch_persistent_sparse(
        const float* input,
        float* persistent_buffer,
        int B, int Size
    ) {
        dim3 grid(B, (Size + 127) / 128);
        dim3 block(128, 1, 1);

        persistent_sparse_kernel<<<grid, block>>>(
            input, persistent_buffer, B, Size
        );
        cudaDeviceSynchronize();
    }
}
