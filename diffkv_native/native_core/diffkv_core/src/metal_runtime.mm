#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "native_core/diffkv_core/include/metal_runtime.hpp"
#include "native_core/diffkv_core/src/diffkv_metallib.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace diffkv {

class MetalDecodePipeline {
public:
    static MetalDecodePipeline& getInstance() {
        static MetalDecodePipeline instance;
        return instance;
    }

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipelineState = nil;
    id<MTLBuffer> dummyRopeBuf = nil;
    bool initialized = false;

    MetalDecodePipeline() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                std::cerr << "[DiffKV Metal] Failed to retrieve system default MTLDevice!" << std::endl;
                return;
            }
            queue = [device newCommandQueue];
            if (!queue) {
                std::cerr << "[DiffKV Metal] Failed to create MTLCommandQueue!" << std::endl;
                return;
            }

            // Dummy buffer for RoPE to prevent exceptions on GPU when RoPE is disabled
            dummyRopeBuf = [device newBufferWithLength:65536 options:MTLResourceStorageModeShared];

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

bool has_metal() {
#ifdef __APPLE__
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil;
#else
    return false;
#endif
}

bool is_metal_available() {
    return MetalDecodePipeline::getInstance().initialized;
}

static id<MTLBuffer> wrap_pointer(id<MTLDevice> device, const void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return nil;
    size_t aligned_len = (bytes + 4095) & ~4095;
    id<MTLBuffer> buf = nil;
    if (((uintptr_t)ptr % 4096) == 0) {
        buf = [device newBufferWithBytesNoCopy:(void*)ptr
                                        length:aligned_len
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
    }
    if (!buf) {
        buf = [device newBufferWithBytes:ptr
                                  length:bytes
                                 options:MTLResourceStorageModeShared];
    }
    return buf;
}

static id<MTLBuffer> wrap_output_pointer(id<MTLDevice> device, void* ptr, size_t bytes, void** out_temp_ptr) {
    *out_temp_ptr = nullptr;
    if (!ptr || bytes == 0) return nil;
    if (((uintptr_t)ptr % 4096) == 0) {
        size_t aligned_len = (bytes + 4095) & ~4095;
        return [device newBufferWithBytesNoCopy:ptr
                                         length:aligned_len
                                        options:MTLResourceStorageModeShared
                                    deallocator:nil];
    } else {
        void* temp_mem = nullptr;
        int ret = posix_memalign(&temp_mem, 4096, (bytes + 4095) & ~4095);
        if (ret != 0) {
            std::cerr << "[Metal Attention] Error: posix_memalign failed!" << std::endl;
            return nil;
        }
        *out_temp_ptr = temp_mem;
        return [device newBufferWithBytesNoCopy:temp_mem
                                         length:(bytes + 4095) & ~4095
                                        options:MTLResourceStorageModeShared
                                    deallocator:^(void* pointer, NSUInteger length) {
                                        free(pointer);
                                    }];
    }
}

static inline float fp16_to_fp32(uint16_t h) {
    _Float16 f;
    std::memcpy(&f, &h, sizeof(f));
    return (float)f;
}

// CPU Fallback logic
static void decode_attention_cpu_fallback(
    const float*    Q,
    const int8_t*   U_pool,
    const float*    U_scale_pool,
    const uint16_t* VK_pool,
    const uint16_t* VV_pool,
    const uint16_t* anchors_K,
    const uint16_t* anchors_V,
    const int32_t*  seq_lens,
    const uint16_t* scales,
    const float*    cos_anc,
    const float*    sin_anc,
    const int32_t*  slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank,
    int K_active,
    int S_max,
    int D,
    float* out,
    float* lse
) {
    const int g = n_q_heads / n_kv_heads;
    const int half_d = D / 2;

    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;
        const float* q_h = Q + h * D;

        float m_h = -1e30f;
        float d_h = 0.0f;
        float* o_h = out + h * D;
        std::memset(o_h, 0, D * sizeof(float));

        for (int k = 0; k < K_active; ++k) {
            int slot_id = slot_indices[k];
            int S_k = seq_lens[slot_id];
            if (S_k < 0) S_k = 0;

            float scale_u = U_scale_pool[slot_id];
            float block_scale = fp16_to_fp32(scales[slot_id]);

            // 1. Compute Anchor score
            float score_anc = 0.0f;
            for (int d = 0; d < D; ++d) {
                float raw_ak = fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
                float ak_rot = raw_ak;
                if (cos_anc && sin_anc) {
                    float c = cos_anc[k * D + d];
                    float s = sin_anc[k * D + d];
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    float raw_partner = fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner]);
                    float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                    ak_rot = raw_ak * c + rot_contrib * s;
                }
                score_anc += q_h[d] * ak_rot;
            }
            float s_anc_scaled = score_anc * scale;

            // 2. Query projection
            std::vector<float> q_proj(rank, 0.0f);
            for (int r = 0; r < rank; ++r) {
                float proj_val = 0.0f;
                int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    float raw_vk = fp16_to_fp32(VK_pool[base_vk_offset + d]);
                    float vk_rot = raw_vk;
                    if (cos_anc && sin_anc) {
                        float c = cos_anc[k * D + d];
                        float s = sin_anc[k * D + d];
                        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        float raw_vk_partner = fp16_to_fp32(VK_pool[base_vk_offset + partner]);
                        float rot_contrib = (d < half_d) ? -raw_vk_partner : raw_vk_partner;
                        vk_rot = raw_vk * c + rot_contrib * s;
                    }
                    proj_val += q_h[d] * vk_rot;
                }
                q_proj[r] = proj_val;
            }

            // 3. Delta scores
            std::vector<float> s_delta_vec(S_k, 0.0f);
            for (int t = 0; t < S_k; ++t) {
                float delta_sum = 0.0f;
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    delta_sum += q_proj[r] * static_cast<float>(U_pool[u_offset + r]);
                }
                s_delta_vec[t] = (delta_sum * scale_u * block_scale + score_anc) * scale;
            }

            // Local max score in block k
            float M_local = s_anc_scaled;
            for (int t = 0; t < S_k; ++t) {
                if (s_delta_vec[t] > M_local) {
                    M_local = s_delta_vec[t];
                }
            }

            // Exponentials
            float E_anc = std::exp(s_anc_scaled - M_local);
            float E_sum = E_anc;
            std::vector<float> w_token_vec(S_k);
            for (int t = 0; t < S_k; ++t) {
                float E_t = std::exp(s_delta_vec[t] - M_local);
                E_sum += E_t;
                w_token_vec[t] = E_t;
            }

            // Online softmax update stats
            float m_new = std::max(m_h, M_local);
            float alpha = std::exp(m_h - m_new);
            float beta = std::exp(M_local - m_new);
            float d_new = d_h * alpha + E_sum * beta;

            // 4. Value contribution
            std::vector<float> W_proj(rank, 0.0f);
            for (int t = 0; t < S_k; ++t) {
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    W_proj[r] += w_token_vec[t] * static_cast<float>(U_pool[u_offset + r]) * scale_u;
                }
            }

            float w_total_anc = E_anc;
            for (int t = 0; t < S_k; ++t) {
                w_total_anc += w_token_vec[t];
            }

            for (int d = 0; d < D; ++d) {
                float av_val = fp16_to_fp32(anchors_V[slot_id * n_kv_heads * D + kv_head * D + d]);
                float V_local_d = w_total_anc * av_val;

                float svd_v = 0.0f;
                int base_vv_offset = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                for (int r = 0; r < rank; ++r) {
                    svd_v += W_proj[r] * fp16_to_fp32(VV_pool[base_vv_offset + r * n_kv_heads * D]);
                }
                V_local_d += svd_v * block_scale;

                o_h[d] = o_h[d] * alpha + V_local_d * beta;
            }

            m_h = m_new;
            d_h = d_new;
        }

        // Final normalization
        if (d_h > 0.0f) {
            float inv_d = 1.0f / d_h;
            for (int d = 0; d < D; ++d) {
                o_h[d] *= inv_d;
            }
        }
        lse[h] = m_h + std::log(std::max(d_h, 1e-9f));
    }
}

void decode_attention_metal(
    const float*    Q,
    const int8_t*   U_pool,
    const float*    U_scale_pool,
    const uint16_t* VK_pool,
    const uint16_t* VV_pool,
    const uint16_t* anchors_K,
    const uint16_t* anchors_V,
    const int32_t*  seq_lens,
    const uint16_t* scales,
    const float*    cos_anc,
    const float*    sin_anc,
    const int32_t*  slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank,
    int K_active,
    int N_pool,
    int S_max,
    int D,
    float* out,
    float* lse
) {
    auto& mps_pipeline = MetalDecodePipeline::getInstance();
    if (!mps_pipeline.initialized) {
        decode_attention_cpu_fallback(Q, U_pool, U_scale_pool, VK_pool, VV_pool, anchors_K, anchors_V,
                                      seq_lens, scales, cos_anc, sin_anc, slot_indices, scale,
                                      n_q_heads, n_kv_heads, rank, K_active, S_max, D, out, lse);
        return;
    }

    if (K_active == 0) {
        std::memset(out, 0, n_q_heads * D * sizeof(float));
        for (int h = 0; h < n_q_heads; ++h) lse[h] = -1e30f;
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [mps_pipeline.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:mps_pipeline.pipelineState];

        id<MTLBuffer> buf_q = wrap_pointer(mps_pipeline.device, Q, n_q_heads * D * sizeof(float));
        id<MTLBuffer> buf_u = wrap_pointer(mps_pipeline.device, U_pool, N_pool * S_max * rank * sizeof(int8_t));
        id<MTLBuffer> buf_u_scale = wrap_pointer(mps_pipeline.device, U_scale_pool, N_pool * sizeof(float));
        id<MTLBuffer> buf_vk = wrap_pointer(mps_pipeline.device, VK_pool, N_pool * rank * n_kv_heads * D * sizeof(uint16_t));
        id<MTLBuffer> buf_vv = wrap_pointer(mps_pipeline.device, VV_pool, N_pool * rank * n_kv_heads * D * sizeof(uint16_t));
        id<MTLBuffer> buf_ak = wrap_pointer(mps_pipeline.device, anchors_K, N_pool * n_kv_heads * D * sizeof(uint16_t));
        id<MTLBuffer> buf_av = wrap_pointer(mps_pipeline.device, anchors_V, N_pool * n_kv_heads * D * sizeof(uint16_t));
        id<MTLBuffer> buf_slens = wrap_pointer(mps_pipeline.device, seq_lens, N_pool * sizeof(int32_t));
        id<MTLBuffer> buf_slots = wrap_pointer(mps_pipeline.device, slot_indices, K_active * sizeof(int32_t));
        id<MTLBuffer> buf_scales = wrap_pointer(mps_pipeline.device, scales, N_pool * sizeof(uint16_t));

        void* temp_out_ptr = nullptr;
        id<MTLBuffer> buf_out = wrap_output_pointer(mps_pipeline.device, out, n_q_heads * D * sizeof(float), &temp_out_ptr);

        void* temp_lse_ptr = nullptr;
        id<MTLBuffer> buf_lse = wrap_output_pointer(mps_pipeline.device, lse, n_q_heads * sizeof(float), &temp_lse_ptr);

        id<MTLBuffer> buf_cos = nil;
        id<MTLBuffer> buf_sin = nil;
        int has_rope_flag = 0;
        if (cos_anc && sin_anc) {
            buf_cos = wrap_pointer(mps_pipeline.device, cos_anc, K_active * D * sizeof(float));
            buf_sin = wrap_pointer(mps_pipeline.device, sin_anc, K_active * D * sizeof(float));
            has_rope_flag = 1;
        } else {
            buf_cos = mps_pipeline.dummyRopeBuf;
            buf_sin = mps_pipeline.dummyRopeBuf;
        }

        [encoder setBuffer:buf_q offset:0 atIndex:0];
        [encoder setBuffer:buf_u offset:0 atIndex:1];
        [encoder setBuffer:buf_u_scale offset:0 atIndex:2];
        [encoder setBuffer:buf_vk offset:0 atIndex:3];
        [encoder setBuffer:buf_vv offset:0 atIndex:4];
        [encoder setBuffer:buf_ak offset:0 atIndex:5];
        [encoder setBuffer:buf_av offset:0 atIndex:6];
        [encoder setBuffer:buf_slens offset:0 atIndex:7];
        [encoder setBuffer:buf_slots offset:0 atIndex:8];
        [encoder setBuffer:buf_out offset:0 atIndex:9];
        [encoder setBuffer:buf_lse offset:0 atIndex:10];

        int32_t n_q_heads_i32 = n_q_heads;
        int32_t n_kv_heads_i32 = n_kv_heads;
        int32_t rank_i32 = rank;
        int32_t S_max_i32 = S_max;
        int32_t K_i32 = K_active;
        int32_t D_i32 = D;
        float scale_f32 = scale;

        [encoder setBytes:&n_q_heads_i32 length:sizeof(n_q_heads_i32) atIndex:11];
        [encoder setBytes:&n_kv_heads_i32 length:sizeof(n_kv_heads_i32) atIndex:12];
        [encoder setBytes:&rank_i32 length:sizeof(rank_i32) atIndex:13];
        [encoder setBytes:&S_max_i32 length:sizeof(S_max_i32) atIndex:14];
        [encoder setBytes:&K_i32 length:sizeof(K_i32) atIndex:15];
        [encoder setBytes:&D_i32 length:sizeof(D_i32) atIndex:16];
        [encoder setBytes:&scale_f32 length:sizeof(scale_f32) atIndex:17];
        [encoder setBuffer:buf_scales offset:0 atIndex:18];
        [encoder setBuffer:buf_cos offset:0 atIndex:19];
        [encoder setBuffer:buf_sin offset:0 atIndex:20];
        [encoder setBytes:&has_rope_flag length:sizeof(has_rope_flag) atIndex:21];

        MTLSize grid = MTLSizeMake(n_q_heads, 1, 1);
        MTLSize threadgroup = MTLSizeMake(64, 1, 1);

        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (temp_out_ptr != nullptr) {
            std::memcpy(out, temp_out_ptr, n_q_heads * D * sizeof(float));
        }
        if (temp_lse_ptr != nullptr) {
            std::memcpy(lse, temp_lse_ptr, n_q_heads * sizeof(float));
        }
    }
}

} // namespace diffkv
