#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "diffkv_attention.hpp"
#include <iostream>
#include <cmath>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipeline = nil;
static id<MTLBuffer> g_dummy_rope_buf = nil;

static void init_metal_runtime() {
    if (g_device != nil) return;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        std::cerr << "[Metal Attention] Error: MTLCreateSystemDefaultDevice returned nil!" << std::endl;
        return;
    }
    g_queue = [g_device newCommandQueue];

    // Allocate dummy non-nil MTLBuffer for RoPE to prevent GPU exceptions
    // 128 (max K) * 128 (max D) * sizeof(float) = 65536 bytes
    g_dummy_rope_buf = [g_device newBufferWithLength:65536 options:MTLResourceStorageModeShared];

    NSError* error = nil;
    // Load shader source from diffkv_native/shaders/diffkv_decode.metal
    NSString* path = @"../shaders/diffkv_decode.metal";
    NSString* source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!source) {
        // Fallback: look at sibling directory path
        path = @"shaders/diffkv_decode.metal";
        source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    }
    if (!source) {
        std::cerr << "[Metal Attention] Error: Could not load diffkv_decode.metal shader file!" << std::endl;
        return;
    }
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        std::cerr << "[Metal Attention] Error compiling Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"decode_attention_metal_kernel"];
    g_pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!g_pipeline) {
        std::cerr << "[Metal Attention] Error creating Compute Pipeline State: " << [[error localizedDescription] UTF8String] << std::endl;
    } else {
        std::cout << "[Metal Attention] Successfully initialized Metal runtime and compiled shader library." << std::endl;
    }
}

static id<MTLBuffer> wrap_tensor(struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return nil;
    size_t bytes = ggml_nbytes(tensor);
    // Align length to page size (4096) for newBufferWithBytesNoCopy
    size_t aligned_len = (bytes + 4095) & ~4095;
    id<MTLBuffer> buf = nil;
    if (((uintptr_t)tensor->data % 4096) == 0) {
        buf = [g_device newBufferWithBytesNoCopy:tensor->data
                                          length:aligned_len
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
    }
    if (!buf) {
        // Fallback: copy data to a new shared buffer
        buf = [g_device newBufferWithBytes:tensor->data
                                    length:bytes
                                   options:MTLResourceStorageModeShared];
    }
    return buf;
}

static id<MTLBuffer> wrap_output_tensor(struct ggml_tensor* tensor, void** out_temp_ptr) {
    *out_temp_ptr = nullptr;
    if (!tensor || !tensor->data) return nil;
    size_t bytes = ggml_nbytes(tensor);
    if (((uintptr_t)tensor->data % 4096) == 0) {
        size_t aligned_len = (bytes + 4095) & ~4095;
        return [g_device newBufferWithBytesNoCopy:tensor->data
                                           length:aligned_len
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
    } else {
        // Allocate page-aligned temporary memory
        void* temp_mem = nullptr;
        int ret = posix_memalign(&temp_mem, 4096, (bytes + 4095) & ~4095);
        if (ret != 0) {
            std::cerr << "[Metal Attention] Error: posix_memalign failed!" << std::endl;
            return nil;
        }
        *out_temp_ptr = temp_mem;
        return [g_device newBufferWithBytesNoCopy:temp_mem
                                           length:(bytes + 4095) & ~4095
                                          options:MTLResourceStorageModeShared
                                      deallocator:^(void* pointer, NSUInteger length) {
                                          free(pointer);
                                      }];
    }
}

static bool verify_attention_cpu(
    const float* q_data,              // [n_q_heads * D]
    const int32_t* slots,             // [K]
    const float* metal_output,        // [n_q_heads * D]
    diffkv::DiffKVKVEngine* kv_engine,
    int n_q_heads, int n_kv_heads, int rank, int S_max, int K, int D, float scale,
    bool has_rope, float rope_freq_base
) {
    // Read pool tensors to host for reference calculation
    std::vector<int8_t> U(ggml_nelements(kv_engine->get_U()));
    ggml_backend_tensor_get(kv_engine->get_U(), U.data(), 0, U.size() * sizeof(int8_t));

    std::vector<ggml_fp16_t> U_scale(ggml_nelements(kv_engine->get_U_scale()));
    ggml_backend_tensor_get(kv_engine->get_U_scale(), U_scale.data(), 0, U_scale.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> VK(ggml_nelements(kv_engine->get_VK()));
    ggml_backend_tensor_get(kv_engine->get_VK(), VK.data(), 0, VK.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> VV(ggml_nelements(kv_engine->get_VV()));
    ggml_backend_tensor_get(kv_engine->get_VV(), VV.data(), 0, VV.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> anchors_K(ggml_nelements(kv_engine->get_anchors_K()));
    ggml_backend_tensor_get(kv_engine->get_anchors_K(), anchors_K.data(), 0, anchors_K.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> anchors_V(ggml_nelements(kv_engine->get_anchors_V()));
    ggml_backend_tensor_get(kv_engine->get_anchors_V(), anchors_V.data(), 0, anchors_V.size() * sizeof(ggml_fp16_t));

    std::vector<int32_t> seq_lens(ggml_nelements(kv_engine->get_seq_lens()));
    ggml_backend_tensor_get(kv_engine->get_seq_lens(), seq_lens.data(), 0, seq_lens.size() * sizeof(int32_t));

    std::vector<ggml_fp16_t> scales(ggml_nelements(kv_engine->get_scales()));
    ggml_backend_tensor_get(kv_engine->get_scales(), scales.data(), 0, scales.size() * sizeof(ggml_fp16_t));

    std::vector<int32_t> anchor_positions(ggml_nelements(kv_engine->get_anchor_positions()));
    ggml_backend_tensor_get(kv_engine->get_anchor_positions(), anchor_positions.data(), 0, anchor_positions.size() * sizeof(int32_t));

    std::vector<float> cpu_output(n_q_heads * D, 0.0f);
    const int g = n_q_heads / n_kv_heads;
    const int half_d = D / 2;

    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;

        float max_score = -1e30f;
        
        struct SlotScoreInfo {
            float anchor_score;
            std::vector<float> token_scores;
            std::vector<float> q_proj;
        };
        std::vector<SlotScoreInfo> slot_infos(K);

        for (int k = 0; k < K; ++k) {
            int slot_id = slots[k];
            int slen = seq_lens[slot_id];
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);
            int anchor_pos = anchor_positions[slot_id];

            // 1. Anchor score (rotated by RoPE)
            float score_anc = 0.0f;
            for (int d = 0; d < D; ++d) {
                float raw_ak = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
                float ak_rot = raw_ak;
                if (has_rope) {
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    float raw_partner = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner]);
                    float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                    int idx = (d < half_d) ? d : (d - half_d);
                    float theta = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
                    float angle = anchor_pos * theta;
                    ak_rot = raw_ak * std::cos(angle) + rot_contrib * std::sin(angle);
                }
                score_anc += q_data[h * D + d] * ak_rot;
            }
            slot_infos[k].anchor_score = score_anc;

            float s_anc_scaled = score_anc * scale;
            if (s_anc_scaled > max_score) max_score = s_anc_scaled;

            // 2. Query projection (using rotated VK)
            std::vector<float> q_proj(rank, 0.0f);
            for (int r = 0; r < rank; ++r) {
                float proj_val = 0.0f;
                int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    float raw_vk = ggml_fp16_to_fp32(VK[base_vk_offset + d]);
                    float vk_rot = raw_vk;
                    if (has_rope) {
                        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        float raw_partner = ggml_fp16_to_fp32(VK[base_vk_offset + partner]);
                        float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                        int idx = (d < half_d) ? d : (d - half_d);
                        float theta = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
                        float angle = anchor_pos * theta;
                        vk_rot = raw_vk * std::cos(angle) + rot_contrib * std::sin(angle);
                    }
                    proj_val += q_data[h * D + d] * vk_rot;
                }
                q_proj[r] = proj_val;
            }
            slot_infos[k].q_proj = q_proj;

            // 3. Delta scores
            slot_infos[k].token_scores.resize(slen);
            for (int t = 0; t < slen; ++t) {
                float delta_sum = 0.0f;
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    delta_sum += q_proj[r] * static_cast<float>(U[u_offset + r]);
                }
                float t_score = (delta_sum * scale_u * block_scale + score_anc) * scale;
                slot_infos[k].token_scores[t] = t_score;
                if (t_score > max_score) max_score = t_score;
            }
        }

        // Softmax denominator
        double sum_exp = 0.0;
        for (int k = 0; k < K; ++k) {
            sum_exp += std::exp(slot_infos[k].anchor_score * scale - max_score);
            for (float s : slot_infos[k].token_scores) {
                sum_exp += std::exp(s - max_score);
            }
        }

        // Pass 2: Accumulate values
        std::vector<double> accum_val(D, 0.0);
        for (int k = 0; k < K; ++k) {
            int slot_id = slots[k];
            int slen = seq_lens[slot_id];
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);

            double w_anc = std::exp(slot_infos[k].anchor_score * scale - max_score) / sum_exp;
            double sum_w_tokens = 0.0;
            std::vector<double> w_proj(rank, 0.0);

            for (int t = 0; t < slen; ++t) {
                double w_t = std::exp(slot_infos[k].token_scores[t] - max_score) / sum_exp;
                sum_w_tokens += w_t;
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    w_proj[r] += w_t * static_cast<float>(U[u_offset + r]) * scale_u;
                }
            }

            double w_total_anc = w_anc + sum_w_tokens;

            for (int d = 0; d < D; ++d) {
                double av_val = ggml_fp16_to_fp32(anchors_V[slot_id * n_kv_heads * D + kv_head * D + d]);
                accum_val[d] += w_total_anc * av_val;

                double svd_v_contrib = 0.0;
                int base_vv_offset = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                for (int r = 0; r < rank; ++r) {
                    double vv_val = ggml_fp16_to_fp32(VV[base_vv_offset + r * n_kv_heads * D]);
                    svd_v_contrib += w_proj[r] * vv_val;
                }
                accum_val[d] += svd_v_contrib * block_scale;
            }
        }

        for (int d = 0; d < D; ++d) {
            cpu_output[h * D + d] = static_cast<float>(accum_val[d]);
        }
    }

    float max_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        float diff = std::abs(cpu_output[i] - metal_output[i]);
        if (diff > max_diff) max_diff = diff;
        sum_sq_diff += diff * diff;
    }
    float rmse = std::sqrt(sum_sq_diff / cpu_output.size());
    std::printf("[Verification] CPU vs Metal Max Diff: %e, RMSE: %e\n", max_diff, rmse);
    return max_diff < 1e-4f;
}

namespace diffkv {

void execute_metal_attention(
    struct ggml_tensor * dst,
    const struct ggml_tensor * Q,
    struct ggml_tensor * slot_indices,
    CustomAttnUserData * data
) {
    init_metal_runtime();
    if (!g_pipeline) {
        std::cerr << "[Metal Attention] Error: Compute pipeline not initialized!" << std::endl;
        return;
    }

    int n_q_heads = data->n_q_heads;
    int n_kv_heads = data->n_kv_heads;
    int rank = data->rank;
    int S_max = data->S_max;
    int K = data->K;
    int D = data->D;
    float scale = data->scale;
    bool has_rope = data->has_rope;
    float rope_freq_base = data->rope_freq_base;
    DiffKVKVEngine* kv_engine = data->kv_engine;

    std::vector<float> out_sparse(n_q_heads * D, 0.0f);
    std::vector<float> lse_sparse(n_q_heads, -1e30f);

    @autoreleasepool {
        id<MTLBuffer> lse_buf = nil;

        if (K > 0) {
            id<MTLCommandBuffer> commandBuffer = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:g_pipeline];

            // 1. Bind buffer arguments
            id<MTLBuffer> q_buf = wrap_tensor((struct ggml_tensor*)Q);
            id<MTLBuffer> u_pool_buf = wrap_tensor(kv_engine->get_U());
            id<MTLBuffer> u_scale_buf = wrap_tensor(kv_engine->get_U_scale());
            id<MTLBuffer> vk_pool_buf = wrap_tensor(kv_engine->get_VK());
            id<MTLBuffer> vv_pool_buf = wrap_tensor(kv_engine->get_VV());
            id<MTLBuffer> anchors_k_buf = wrap_tensor(kv_engine->get_anchors_K());
            id<MTLBuffer> anchors_v_buf = wrap_tensor(kv_engine->get_anchors_V());
            id<MTLBuffer> seq_lens_buf = wrap_tensor(kv_engine->get_seq_lens());
            id<MTLBuffer> slot_indices_buf = wrap_tensor(slot_indices);

            void* temp_out_ptr = nullptr;
            id<MTLBuffer> out_buf = wrap_output_tensor(dst, &temp_out_ptr);

            lse_buf = [g_device newBufferWithLength:n_q_heads * sizeof(float)
                                                          options:MTLResourceStorageModeShared];

            id<MTLBuffer> scales_buf = wrap_tensor(kv_engine->get_scales());

            [encoder setBuffer:q_buf offset:0 atIndex:0];
            [encoder setBuffer:u_pool_buf offset:0 atIndex:1];
            [encoder setBuffer:u_scale_buf offset:0 atIndex:2];
            [encoder setBuffer:vk_pool_buf offset:0 atIndex:3];
            [encoder setBuffer:vv_pool_buf offset:0 atIndex:4];
            [encoder setBuffer:anchors_k_buf offset:0 atIndex:5];
            [encoder setBuffer:anchors_v_buf offset:0 atIndex:6];
            [encoder setBuffer:seq_lens_buf offset:0 atIndex:7];
            [encoder setBuffer:slot_indices_buf offset:0 atIndex:8];
            [encoder setBuffer:out_buf offset:0 atIndex:9];
            [encoder setBuffer:lse_buf offset:0 atIndex:10];

            // Bind uniform values
            int32_t n_q_heads_i32 = n_q_heads;
            int32_t n_kv_heads_i32 = n_kv_heads;
            int32_t rank_i32 = rank;
            int32_t S_max_i32 = S_max;
            int32_t K_i32 = K;
            int32_t D_i32 = D;
            float scale_f32 = scale;
            int32_t has_rope_i32 = has_rope ? 1 : 0;

            [encoder setBytes:&n_q_heads_i32 length:sizeof(n_q_heads_i32) atIndex:11];
            [encoder setBytes:&n_kv_heads_i32 length:sizeof(n_kv_heads_i32) atIndex:12];
            [encoder setBytes:&rank_i32 length:sizeof(rank_i32) atIndex:13];
            [encoder setBytes:&S_max_i32 length:sizeof(S_max_i32) atIndex:14];
            [encoder setBytes:&K_i32 length:sizeof(K_i32) atIndex:15];
            [encoder setBytes:&D_i32 length:sizeof(D_i32) atIndex:16];
            [encoder setBytes:&scale_f32 length:sizeof(scale_f32) atIndex:17];
            [encoder setBuffer:scales_buf offset:0 atIndex:18];

            // Bind RoPE cosine/sine buffers (null/empty if has_rope is false)
            id<MTLBuffer> cos_anc_buf = g_dummy_rope_buf;
            id<MTLBuffer> sin_anc_buf = g_dummy_rope_buf;
            if (has_rope) {
                cos_anc_buf = [g_device newBufferWithLength:K * D * sizeof(float) options:MTLResourceStorageModeShared];
                sin_anc_buf = [g_device newBufferWithLength:K * D * sizeof(float) options:MTLResourceStorageModeShared];
                
                const int32_t* slots_ptr = (const int32_t*)slot_indices->data;
                float* cos_anc_ptr = (float*)cos_anc_buf.contents;
                float* sin_anc_ptr = (float*)sin_anc_buf.contents;
                
                // Read actual anchor positions from the engine's anchor_positions tensor
                std::vector<int32_t> anchor_positions_host(ggml_nelements(kv_engine->get_anchor_positions()));
                ggml_backend_tensor_get(kv_engine->get_anchor_positions(), anchor_positions_host.data(), 0, anchor_positions_host.size() * sizeof(int32_t));
                
                for (int k = 0; k < K; ++k) {
                    int slot_id = slots_ptr[k];
                    // Use the actual sequence position of the anchor token (not slot_id * 64)
                    int anchor_idx = anchor_positions_host[slot_id];
                    
                    for (int d = 0; d < D; ++d) {
                        int half_d = D / 2;
                        int idx = (d < half_d) ? d : (d - half_d);
                        float theta = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
                        float angle = anchor_idx * theta;
                        cos_anc_ptr[k * D + d] = std::cos(angle);
                        sin_anc_ptr[k * D + d] = std::sin(angle);
                    }
                }
            }
            [encoder setBuffer:cos_anc_buf offset:0 atIndex:19];
            [encoder setBuffer:sin_anc_buf offset:0 atIndex:20];
            [encoder setBytes:&has_rope_i32 length:sizeof(has_rope_i32) atIndex:21];
            float rope_freq_base_f32 = rope_freq_base;
            [encoder setBytes:&rope_freq_base_f32 length:sizeof(rope_freq_base_f32) atIndex:22];

            // Bind anchor_positions: [N_pool] int32 with actual sequence positions
            id<MTLBuffer> anchor_positions_buf = wrap_tensor(kv_engine->get_anchor_positions());
            [encoder setBuffer:anchor_positions_buf offset:0 atIndex:23];

            // Dispatch threads: 1 Threadgroup per Query Head, 64 threads per Threadgroup
            MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);
            MTLSize numThreadgroups = MTLSizeMake(n_q_heads, 1, 1);
            [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];

            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            // Copy temporary output memory back if it was allocated
            if (temp_out_ptr != nullptr) {
                memcpy(dst->data, temp_out_ptr, ggml_nbytes(dst));
            }
            
            // Read sparse output and LSE back
            memcpy(out_sparse.data(), dst->data, n_q_heads * D * sizeof(float));
            memcpy(lse_sparse.data(), lse_buf.contents, n_q_heads * sizeof(float));

            // Run verification of sparse attention
            verify_attention_cpu(
                (const float*)Q->data,
                (const int32_t*)slot_indices->data,
                (const float*)out_sparse.data(),
                kv_engine,
                n_q_heads, n_kv_heads, rank, S_max, K, D, scale,
                has_rope, rope_freq_base
            );
        }

        // ── Dense Window Attention & LSE Combine on CPU ──
        bool has_dense = (data->active_block_tokens > 0 && data->active_k_dense != nullptr && data->active_v_dense != nullptr);
        if (has_dense) {
            int active_block_tokens = data->active_block_tokens;
            int active_slot = data->active_slot;
            const float* active_k_dense = data->active_k_dense;
            const float* active_v_dense = data->active_v_dense;
            const int g = n_q_heads / n_kv_heads;
            const int half_d = D / 2;

            // Read anchor positions from engine
            std::vector<int32_t> anchor_positions_host(ggml_nelements(kv_engine->get_anchor_positions()));
            ggml_backend_tensor_get(kv_engine->get_anchor_positions(), anchor_positions_host.data(), 0, anchor_positions_host.size() * sizeof(int32_t));
            int anchor_pos = anchor_positions_host[active_slot];

            // Read anchors_K and anchors_V from engine
            std::vector<ggml_fp16_t> anchors_K_fp16(ggml_nelements(kv_engine->get_anchors_K()));
            ggml_backend_tensor_get(kv_engine->get_anchors_K(), anchors_K_fp16.data(), 0, anchors_K_fp16.size() * sizeof(ggml_fp16_t));
            std::vector<ggml_fp16_t> anchors_V_fp16(ggml_nelements(kv_engine->get_anchors_V()));
            ggml_backend_tensor_get(kv_engine->get_anchors_V(), anchors_V_fp16.data(), 0, anchors_V_fp16.size() * sizeof(ggml_fp16_t));

            const float* Q_ptr = (const float*)Q->data;
            std::vector<float> final_output(n_q_heads * D, 0.0f);

            for (int h = 0; h < n_q_heads; ++h) {
                int kv_head = h / g;

                // Reconstruct K and V for the active block dense tokens
                std::vector<float> K_dense_rot(active_block_tokens * D);
                std::vector<float> V_dense(active_block_tokens * D);

                for (int t = 0; t < active_block_tokens; ++t) {
                    int pos = anchor_pos + t;

                    for (int d = 0; d < D; ++d) {
                        float raw_k, raw_v;
                        if (t == 0) {
                            raw_k = ggml_fp16_to_fp32(anchors_K_fp16[active_slot * n_kv_heads * D + kv_head * D + d]);
                            raw_v = ggml_fp16_to_fp32(anchors_V_fp16[active_slot * n_kv_heads * D + kv_head * D + d]);
                        } else {
                            // active_k_dense shape: [64, F] where F = n_kv_heads * D
                            int offset = (t - 1) * n_kv_heads * D + kv_head * D;
                            raw_k = ggml_fp16_to_fp32(anchors_K_fp16[active_slot * n_kv_heads * D + kv_head * D + d]) + active_k_dense[offset + d];
                            raw_v = ggml_fp16_to_fp32(anchors_V_fp16[active_slot * n_kv_heads * D + kv_head * D + d]) + active_v_dense[offset + d];
                        }
                        V_dense[t * D + d] = raw_v;

                        if (has_rope) {
                            int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                            float raw_partner;
                            if (t == 0) {
                                raw_partner = ggml_fp16_to_fp32(anchors_K_fp16[active_slot * n_kv_heads * D + kv_head * D + partner]);
                            } else {
                                int offset = (t - 1) * n_kv_heads * D + kv_head * D;
                                raw_partner = ggml_fp16_to_fp32(anchors_K_fp16[active_slot * n_kv_heads * D + kv_head * D + partner]) + active_k_dense[offset + partner];
                            }
                            float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                            int idx = (d < half_d) ? d : (d - half_d);
                            float theta = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
                            float angle = pos * theta;
                            K_dense_rot[t * D + d] = raw_k * std::cos(angle) + rot_contrib * std::sin(angle);
                        } else {
                            K_dense_rot[t * D + d] = raw_k;
                        }
                    }
                }

                // Compute query-key dot products
                std::vector<float> scores(active_block_tokens);
                float max_score = -1e30f;
                for (int t = 0; t < active_block_tokens; ++t) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += Q_ptr[h * D + d] * K_dense_rot[t * D + d];
                    }
                    scores[t] = dot * scale;
                    if (scores[t] > max_score) {
                        max_score = scores[t];
                    }
                }

                // Log-Sum-Exp
                float sum_exp = 0.0f;
                for (int t = 0; t < active_block_tokens; ++t) {
                    sum_exp += std::exp(scores[t] - max_score);
                }
                float lse_dense = max_score + std::log(std::max(sum_exp, 1e-9f));

                // Compute dense attention output vector
                std::vector<float> out_dense(D, 0.0f);
                for (int t = 0; t < active_block_tokens; ++t) {
                    float w_t = std::exp(scores[t] - lse_dense);
                    for (int d = 0; d < D; ++d) {
                        out_dense[d] += w_t * V_dense[t * D + d];
                    }
                }

                // Combine with sparse attention if sparse blocks exist
                if (K > 0) {
                    float lse_sparse_val = lse_sparse[h];
                    float lse_max = std::max(lse_dense, lse_sparse_val);
                    float w_dense = std::exp(lse_dense - lse_max);
                    float w_sparse = std::exp(lse_sparse_val - lse_max);
                    float denom = w_dense + w_sparse;

                    for (int d = 0; d < D; ++d) {
                        final_output[h * D + d] = (out_dense[d] * w_dense + out_sparse[h * D + d] * w_sparse) / std::max(denom, 1e-9f);
                    }
                } else {
                    for (int d = 0; d < D; ++d) {
                        final_output[h * D + d] = out_dense[d];
                    }
                }
            }
            memcpy(dst->data, final_output.data(), n_q_heads * D * sizeof(float));
        } else {
            // No dense tokens, copy sparse output directly
            if (K > 0) {
                memcpy(dst->data, out_sparse.data(), n_q_heads * D * sizeof(float));
            } else {
                memset(dst->data, 0, n_q_heads * D * sizeof(float));
            }
        }
    }
}

void custom_attention_op_callback(
    struct ggml_tensor * dst,
    const struct ggml_tensor * Q,
    const struct ggml_tensor * slot_indices,
    int ith,
    int nth,
    void * userdata
) {
    if (ith != 0) return;

    CustomAttnUserData * data = static_cast<CustomAttnUserData*>(userdata);
    execute_metal_attention(
        dst, Q, (struct ggml_tensor*)slot_indices, data
    );
}

} // namespace diffkv
