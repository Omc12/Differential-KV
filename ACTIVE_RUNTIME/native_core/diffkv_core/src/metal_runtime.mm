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

struct AttentionParams {
    int32_t n_q_heads;
    int32_t n_kv_heads;
    int32_t rank;
    int32_t S_max;
    int32_t K;
    int32_t D;
    float scale;
    int32_t has_rope;
    int32_t max_residual;
    int32_t L_dense;
};

std::tuple<torch::Tensor, torch::Tensor> decode_attention_metal(
    const torch::Tensor& Q,
    const torch::Tensor& U_pool,
    const torch::Tensor& U_scale_pool,
    const torch::Tensor& VK_pool,
    const torch::Tensor& VV_pool,
    const torch::Tensor& anchors_K,
    const torch::Tensor& anchors_V,
    const torch::Tensor& seq_lens,
    const torch::Tensor& scales,
    const torch::Tensor& cos_anc,
    const torch::Tensor& sin_anc,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank,
    // Residual and Fact Anchor Override buffers (Track D)
    const torch::Tensor& res_pos_K,
    const torch::Tensor& res_val_K,
    const torch::Tensor& res_pos_V,
    const torch::Tensor& res_val_V,
    const torch::Tensor& fact_pos,
    const torch::Tensor& fact_val_K,
    const torch::Tensor& fact_val_V,
    // Dense window buffers
    const torch::Tensor& dense_K,
    const torch::Tensor& dense_V,
    const torch::Tensor& cos_dense,
    const torch::Tensor& sin_dense
) {
    auto& mps_pipeline = MetalDecodePipeline::getInstance();
    if (!mps_pipeline.initialized) {
        throw std::runtime_error("DiffKV Metal compute pipeline not initialized!");
    }

    const auto device = Q.device();
    const int D = Q.size(1);
    const int K_slots = slot_indices.size(0);
    const int L_dense = (dense_K.defined() && dense_K.numel() > 0) ? dense_K.size(-2) : 0;

    // If no slots are active and dense window is empty, return zero outputs immediately
    if (K_slots == 0 && L_dense == 0) {
        auto out = torch::zeros({n_q_heads, D}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
        auto lse = torch::full({n_q_heads}, -std::numeric_limits<float>::infinity(), torch::TensorOptions().dtype(torch::kFloat32).device(device));
        return {out, lse};
    }

    // Pre-allocate output tensors
    auto out = torch::empty({n_q_heads, D}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    auto lse = torch::empty({n_q_heads}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    @autoreleasepool {
        // Ensure all input tensors are contiguous in memory before passing to Metal
        // We must perform all contiguous checks and PyTorch tensor operations FIRST
        // before synchronizing the stream to ensure PyTorch does not open new command encoders afterwards.
        auto Q_c = Q.is_contiguous() ? Q : Q.contiguous();
        auto U_c = U_pool.is_contiguous() ? U_pool : U_pool.contiguous();
        auto U_scale_c = U_scale_pool.is_contiguous() ? U_scale_pool : U_scale_pool.contiguous();
        auto VK_c = VK_pool.is_contiguous() ? VK_pool : VK_pool.contiguous();
        auto VV_c = VV_pool.is_contiguous() ? VV_pool : VV_pool.contiguous();
        auto AK_c = anchors_K.is_contiguous() ? anchors_K : anchors_K.contiguous();
        auto AV_c = anchors_V.is_contiguous() ? anchors_V : anchors_V.contiguous();
        auto slens_c = seq_lens.is_contiguous() ? seq_lens : seq_lens.contiguous();
        auto scales_c = scales.is_contiguous() ? scales : scales.contiguous();
        auto slots_c = slot_indices.is_contiguous() ? slot_indices : slot_indices.contiguous();
        // cos_anc / sin_anc: [K, D] float32. May be empty if RoPE info unavailable.
        bool has_rope = (cos_anc.defined() && cos_anc.numel() > 0);
        auto cos_c = has_rope ? (cos_anc.is_contiguous() ? cos_anc : cos_anc.contiguous())
                              : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
        auto sin_c = has_rope ? (sin_anc.is_contiguous() ? sin_anc : sin_anc.contiguous())
                              : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
        int has_rope_flag = has_rope ? 1 : 0;

        // Prepare contiguous tensors for dense window
        bool has_dense = (dense_K.defined() && dense_K.numel() > 0 && dense_V.defined() && dense_V.numel() > 0);
        auto dense_K_c = has_dense ? (dense_K.is_contiguous() ? dense_K : dense_K.contiguous())
                                   : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        auto dense_V_c = has_dense ? (dense_V.is_contiguous() ? dense_V : dense_V.contiguous())
                                   : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        bool has_dense_rope = has_dense && cos_dense.defined() && cos_dense.numel() > 0;
        auto cos_dense_c = has_dense_rope ? (cos_dense.is_contiguous() ? cos_dense : cos_dense.contiguous())
                                          : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
        auto sin_dense_c = has_dense_rope ? (sin_dense.is_contiguous() ? sin_dense : sin_dense.contiguous())
                                          : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));

        // Track D: Prepare contiguous tensors for Residual and Fact Overrides
        bool has_res = (res_pos_K.defined() && res_pos_K.numel() > 0 && res_val_K.defined() && res_val_K.numel() > 0);
        auto res_pos_K_c = has_res ? (res_pos_K.is_contiguous() ? res_pos_K : res_pos_K.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt16).device(Q.device()));
        auto res_val_K_c = has_res ? (res_val_K.is_contiguous() ? res_val_K : res_val_K.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        auto res_pos_V_c = has_res ? (res_pos_V.is_contiguous() ? res_pos_V : res_pos_V.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt16).device(Q.device()));
        auto res_val_V_c = has_res ? (res_val_V.is_contiguous() ? res_val_V : res_val_V.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        int max_res_val = has_res ? static_cast<int>(res_pos_K_c.size(1)) : 0;

        bool has_fact = (fact_pos.defined() && fact_pos.numel() > 0 && fact_val_K.defined() && fact_val_K.numel() > 0);
        auto fact_pos_c = has_fact ? (fact_pos.is_contiguous() ? fact_pos : fact_pos.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt16).device(Q.device()));
        auto fact_val_K_c = has_fact ? (fact_val_K.is_contiguous() ? fact_val_K : fact_val_K.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));
        auto fact_val_V_c = has_fact ? (fact_val_V.is_contiguous() ? fact_val_V : fact_val_V.contiguous()) : torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));

        // Get the active PyTorch MPS stream to queue the execution
        at::mps::MPSStream* mps_stream = at::mps::getCurrentMPSStream();
        // Commit any active command encoder and flush the stream to get a clean command buffer.
        // Doing this AFTER the contiguous operations commits any command encoder PyTorch opened.
        mps_stream->synchronize(at::mps::SyncType::COMMIT);

        // Retrieve internal Metal storage buffers from ATen tensors
        id<MTLBuffer> buf_q = at::native::mps::getMTLBufferStorage(Q_c);
        id<MTLBuffer> buf_u = at::native::mps::getMTLBufferStorage(U_c);
        id<MTLBuffer> buf_u_scale = at::native::mps::getMTLBufferStorage(U_scale_c);
        id<MTLBuffer> buf_vk = at::native::mps::getMTLBufferStorage(VK_c);
        id<MTLBuffer> buf_vv = at::native::mps::getMTLBufferStorage(VV_c);
        id<MTLBuffer> buf_ak = at::native::mps::getMTLBufferStorage(AK_c);
        id<MTLBuffer> buf_av = at::native::mps::getMTLBufferStorage(AV_c);
        id<MTLBuffer> buf_slens = at::native::mps::getMTLBufferStorage(slens_c);
        id<MTLBuffer> buf_scales = at::native::mps::getMTLBufferStorage(scales_c);
        id<MTLBuffer> buf_slots = at::native::mps::getMTLBufferStorage(slots_c);
        id<MTLBuffer> buf_out = at::native::mps::getMTLBufferStorage(out);
        id<MTLBuffer> buf_lse = at::native::mps::getMTLBufferStorage(lse);
        id<MTLBuffer> buf_cos = at::native::mps::getMTLBufferStorage(cos_c);
        id<MTLBuffer> buf_sin = at::native::mps::getMTLBufferStorage(sin_c);

        id<MTLBuffer> buf_dense_k = at::native::mps::getMTLBufferStorage(dense_K_c);
        id<MTLBuffer> buf_dense_v = at::native::mps::getMTLBufferStorage(dense_V_c);
        id<MTLBuffer> buf_cos_dense = at::native::mps::getMTLBufferStorage(cos_dense_c);
        id<MTLBuffer> buf_sin_dense = at::native::mps::getMTLBufferStorage(sin_dense_c);

        id<MTLBuffer> buf_res_pos_K = at::native::mps::getMTLBufferStorage(res_pos_K_c);
        id<MTLBuffer> buf_res_val_K = at::native::mps::getMTLBufferStorage(res_val_K_c);
        id<MTLBuffer> buf_res_pos_V = at::native::mps::getMTLBufferStorage(res_pos_V_c);
        id<MTLBuffer> buf_res_val_V = at::native::mps::getMTLBufferStorage(res_val_V_c);
        id<MTLBuffer> buf_fact_pos   = at::native::mps::getMTLBufferStorage(fact_pos_c);
        id<MTLBuffer> buf_fact_val_K = at::native::mps::getMTLBufferStorage(fact_val_K_c);
        id<MTLBuffer> buf_fact_val_V = at::native::mps::getMTLBufferStorage(fact_val_V_c);

        // Compute byte offsets from storage offsets
        size_t off_q = Q_c.storage_offset() * Q_c.element_size();
        size_t off_u = U_c.storage_offset() * U_c.element_size();
        size_t off_u_scale = U_scale_c.storage_offset() * U_scale_c.element_size();
        size_t off_vk = VK_c.storage_offset() * VK_c.element_size();
        size_t off_vv = VV_c.storage_offset() * VV_c.element_size();
        size_t off_ak = AK_c.storage_offset() * AK_c.element_size();
        size_t off_av = AV_c.storage_offset() * AV_c.element_size();
        size_t off_slens = slens_c.storage_offset() * slens_c.element_size();
        size_t off_scales = scales_c.storage_offset() * scales_c.element_size();
        size_t off_slots = slots_c.storage_offset() * slots_c.element_size();
        size_t off_out = out.storage_offset() * out.element_size();
        size_t off_lse = lse.storage_offset() * lse.element_size();
        size_t off_cos = cos_c.storage_offset() * cos_c.element_size();
        size_t off_sin = sin_c.storage_offset() * sin_c.element_size();

        size_t off_dense_k = dense_K_c.storage_offset() * dense_K_c.element_size();
        size_t off_dense_v = dense_V_c.storage_offset() * dense_V_c.element_size();
        size_t off_cos_dense = cos_dense_c.storage_offset() * cos_dense_c.element_size();
        size_t off_sin_dense = sin_dense_c.storage_offset() * sin_dense_c.element_size();

        size_t off_res_pos_K = res_pos_K_c.storage_offset() * res_pos_K_c.element_size();
        size_t off_res_val_K = res_val_K_c.storage_offset() * res_val_K_c.element_size();
        size_t off_res_pos_V = res_pos_V_c.storage_offset() * res_pos_V_c.element_size();
        size_t off_res_val_V = res_val_V_c.storage_offset() * res_val_V_c.element_size();
        size_t off_fact_pos   = fact_pos_c.storage_offset() * fact_pos_c.element_size();
        size_t off_fact_val_K = fact_val_K_c.storage_offset() * fact_val_K_c.element_size();
        size_t off_fact_val_V = fact_val_V_c.storage_offset() * fact_val_V_c.element_size();

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

        // Bind uniform parameters directly as bytes using AttentionParams struct
        int S_max = U_c.size(1);
        AttentionParams params;
        params.n_q_heads = n_q_heads;
        params.n_kv_heads = n_kv_heads;
        params.rank = rank;
        params.S_max = S_max;
        params.K = K_slots;
        params.D = D;
        params.scale = scale;
        params.has_rope = has_rope_flag;
        params.max_residual = max_res_val;
        params.L_dense = L_dense;

        [encoder setBytes:&params length:sizeof(AttentionParams) atIndex:11];
        
        [encoder setBuffer:buf_scales offset:off_scales atIndex:12];
        // RoPE buffers: cos_anc [K, D] float32 and sin_anc [K, D] float32
        [encoder setBuffer:buf_cos offset:off_cos atIndex:13];
        [encoder setBuffer:buf_sin offset:off_sin atIndex:14];

        // Bind Track D Residual and Fact Override Buffers
        [encoder setBuffer:buf_res_pos_K offset:off_res_pos_K atIndex:15];
        [encoder setBuffer:buf_res_val_K offset:off_res_val_K atIndex:16];
        [encoder setBuffer:buf_res_pos_V offset:off_res_pos_V atIndex:17];
        [encoder setBuffer:buf_res_val_V offset:off_res_val_V atIndex:18];
        [encoder setBuffer:buf_fact_pos   offset:off_fact_pos   atIndex:19];
        [encoder setBuffer:buf_fact_val_K offset:off_fact_val_K atIndex:20];
        [encoder setBuffer:buf_fact_val_V offset:off_fact_val_V atIndex:21];

        // Bind dense window buffers
        [encoder setBuffer:buf_dense_k offset:off_dense_k atIndex:22];
        [encoder setBuffer:buf_dense_v offset:off_dense_v atIndex:23];
        [encoder setBuffer:buf_cos_dense offset:off_cos_dense atIndex:24];
        [encoder setBuffer:buf_sin_dense offset:off_sin_dense atIndex:25];

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
