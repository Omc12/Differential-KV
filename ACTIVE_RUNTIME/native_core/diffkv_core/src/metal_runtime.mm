#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_runtime.hpp"
#include "diffkv_metallib.hpp"
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <iostream>
#include <stdexcept>
#include <tuple>

namespace diffkv {

// Pipeline State Manager for Metal Decode Kernel
class MetalDecodePipeline {
public:
    static MetalDecodePipeline& getInstance() {
        static MetalDecodePipeline instance;
        return instance;
    }

    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipelineState = nil;
    bool initialized = false;

    MetalDecodePipeline() {
        @autoreleasepool {
            // Get PyTorch's active MPS device
            device = at::mps::MPSDevice::getInstance()->device();
            if (!device) {
                std::cerr << "[DiffKV Metal] Failed to retrieve MPS MTLDevice!" << std::endl;
                return;
            }

            // Create dispatch data from the embedded C++ metallib byte array
            dispatch_data_t data = dispatch_data_create(
                diffkv_metallib, 
                diffkv_metallib_len, 
                nil, 
                DISPATCH_DATA_DESTRUCTOR_DEFAULT
            );

            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
            if (error || !library) {
                std::cerr << "[DiffKV Metal] Failed to load embedded Metal library: " 
                          << (error ? [[error localizedDescription] UTF8String] : "Unknown error") 
                          << std::endl;
                return;
            }

            id<MTLFunction> function = [library newFunctionWithName:@"decode_attention_metal_kernel"];
            if (!function) {
                std::cerr << "[DiffKV Metal] Kernel function 'decode_attention_metal_kernel' not found!" << std::endl;
                return;
            }

            pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
            if (error || !pipelineState) {
                std::cerr << "[DiffKV Metal] Failed to create compute pipeline state: " 
                          << (error ? [[error localizedDescription] UTF8String] : "Unknown error") 
                          << std::endl;
                return;
            }

            initialized = true;
        }
    }
};

bool is_metal_available() {
    return MetalDecodePipeline::getInstance().initialized;
}

std::tuple<torch::Tensor, torch::Tensor> decode_attention_metal(
    const torch::Tensor& Q,
    const torch::Tensor& U_pool,
    const torch::Tensor& U_scale_pool,
    const torch::Tensor& VK_pool,
    const torch::Tensor& VV_pool,
    const torch::Tensor& anchors_K,
    const torch::Tensor& anchors_V,
    const torch::Tensor& seq_lens,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
) {
    auto& mps_pipeline = MetalDecodePipeline::getInstance();
    if (!mps_pipeline.initialized) {
        throw std::runtime_error("DiffKV Metal compute pipeline not initialized!");
    }

    const auto device = Q.device();
    const int D = Q.size(1);
    const int K_slots = slot_indices.size(0);

    // If no slots are active, return zero outputs immediately
    if (K_slots == 0) {
        auto out = torch::zeros({n_q_heads, D}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
        auto lse = torch::full({n_q_heads}, -std::numeric_limits<float>::infinity(), torch::TensorOptions().dtype(torch::kFloat32).device(device));
        return {out, lse};
    }

    // Pre-allocate output tensors
    auto out = torch::empty({n_q_heads, D}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    auto lse = torch::empty({n_q_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    @autoreleasepool {
        // Get the active PyTorch MPS stream to queue the execution
        at::mps::MPSStream* mps_stream = at::mps::getCurrentMPSStream();
        // End active coalescing encoder to prevent command buffer conflicts
        mps_stream->endKernelCoalescing();

        // Retrieve internal Metal storage buffers from ATen tensors
        id<MTLBuffer> buf_q = at::native::mps::getMTLBufferStorage(Q);
        id<MTLBuffer> buf_u = at::native::mps::getMTLBufferStorage(U_pool);
        id<MTLBuffer> buf_u_scale = at::native::mps::getMTLBufferStorage(U_scale_pool);
        id<MTLBuffer> buf_vk = at::native::mps::getMTLBufferStorage(VK_pool);
        id<MTLBuffer> buf_vv = at::native::mps::getMTLBufferStorage(VV_pool);
        id<MTLBuffer> buf_ak = at::native::mps::getMTLBufferStorage(anchors_K);
        id<MTLBuffer> buf_av = at::native::mps::getMTLBufferStorage(anchors_V);
        id<MTLBuffer> buf_slens = at::native::mps::getMTLBufferStorage(seq_lens);
        id<MTLBuffer> buf_slots = at::native::mps::getMTLBufferStorage(slot_indices);
        id<MTLBuffer> buf_out = at::native::mps::getMTLBufferStorage(out);
        id<MTLBuffer> buf_lse = at::native::mps::getMTLBufferStorage(lse);

        // Compute byte offsets from storage offsets
        size_t off_q = Q.storage_offset() * Q.element_size();
        size_t off_u = U_pool.storage_offset() * U_pool.element_size();
        size_t off_u_scale = U_scale_pool.storage_offset() * U_scale_pool.element_size();
        size_t off_vk = VK_pool.storage_offset() * VK_pool.element_size();
        size_t off_vv = VV_pool.storage_offset() * VV_pool.element_size();
        size_t off_ak = anchors_K.storage_offset() * anchors_K.element_size();
        size_t off_av = anchors_V.storage_offset() * anchors_V.element_size();
        size_t off_slens = seq_lens.storage_offset() * seq_lens.element_size();
        size_t off_slots = slot_indices.storage_offset() * slot_indices.element_size();
        size_t off_out = out.storage_offset() * out.element_size();
        size_t off_lse = lse.storage_offset() * lse.element_size();

        id<MTLCommandBuffer> commandBuffer = mps_stream->commandBuffer();
        if (!commandBuffer) {
            throw std::runtime_error("Failed to retrieve active PyTorch MPS command buffer!");
        }

        // Create a compute command encoder from the active command buffer
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            throw std::runtime_error("Failed to create MTLComputeCommandEncoder!");
        }

        [encoder setComputePipelineState:mps_pipeline.pipelineState];

        // Bind raw Metal buffers
        [encoder setBuffer:buf_q offset:off_q atIndex:0];
        [encoder setBuffer:buf_u offset:off_u atIndex:1];
        [encoder setBuffer:buf_u_scale offset:off_u_scale atIndex:2];
        [encoder setBuffer:buf_vk offset:off_vk atIndex:3];
        [encoder setBuffer:buf_vv offset:off_vv atIndex:4];
        [encoder setBuffer:buf_ak offset:off_ak atIndex:5];
        [encoder setBuffer:buf_av offset:off_av atIndex:6];
        [encoder setBuffer:buf_slens offset:off_slens atIndex:7];
        [encoder setBuffer:buf_slots offset:off_slots atIndex:8];
        [encoder setBuffer:buf_out offset:off_out atIndex:9];
        [encoder setBuffer:buf_lse offset:off_lse atIndex:10];

        // Bind uniform parameters directly as bytes
        int S_max = U_pool.size(1);
        [encoder setBytes:&n_q_heads length:sizeof(int) atIndex:11];
        [encoder setBytes:&n_kv_heads length:sizeof(int) atIndex:12];
        [encoder setBytes:&rank length:sizeof(int) atIndex:13];
        [encoder setBytes:&S_max length:sizeof(int) atIndex:14];
        [encoder setBytes:&K_slots length:sizeof(int) atIndex:15];
        [encoder setBytes:&D length:sizeof(int) atIndex:16];
        [encoder setBytes:&scale length:sizeof(float) atIndex:17];

        // Threadgroup grid: [n_q_heads, 1, 1] (1 threadgroup per head)
        MTLSize grid = MTLSizeMake(n_q_heads, 1, 1);
        
        // Threadgroup size: 64 threads (covers typical warp sizes for maximum occupancy)
        MTLSize threadgroup = MTLSizeMake(64, 1, 1);

        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        // Enqueue the stream execution but do NOT call commit/waitUntilCompleted.
        // This keeps execution perfectly pipeline-aligned with other PyTorch ops.
    }

    return {out, lse};
}

} // namespace diffkv
