#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" __declspec(dllexport) void dummy_kernel_call() {
    // dummy CUDA call
}
