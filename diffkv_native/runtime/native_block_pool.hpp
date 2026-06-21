#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>

#include "native_core/diffkv_core/include/block_state.hpp"
#include "runtime/page_aligned_allocator.hpp"

namespace diffkv {

class NativeBlockPool {
public:
    NativeBlockPool();
    ~NativeBlockPool();

    bool initialize(int n_slots, int rank, int head_dim, int kv_heads, int desc_dim, ggml_backend_buffer_type_t buft, int S_max = 64);

    int get_rank() const { return rank_; }
    int get_S_max() const { return S_max_; }

    // Getters for GGML tensors
    struct ggml_tensor * get_U() { return U_; }
    struct ggml_tensor * get_U_f16() { return U_f16_; }   // f16 mirror of U (native ggml gather)
    struct ggml_tensor * get_U_scale() { return U_scale_; }
    struct ggml_tensor * get_VK() { return VK_; }
    struct ggml_tensor * get_VV() { return VV_; }
    // Native-attn precomputed RoPE'd keys (rotated at each block's fixed anchor position).
    // Filled at upload when native attn is enabled, so the in-graph subgraph can gather f16
    // and dot with the in-graph query (rotated at the current pos) with NO in-graph RoPE and
    // NO i32 position gather. nullptr unless DIFFKV_NATIVE_ATTN is set.
    struct ggml_tensor * get_VK_rot() { return VK_rot_; }
    struct ggml_tensor * get_anchorK_rot() { return anchorK_rot_; }
    // [S_max, n_slots] f16 additive bias: 0 for valid tokens (t < seq_len), -inf for padding.
    struct ggml_tensor * get_valid_mask() { return valid_mask_; }
    bool native_attn_enabled() const { return native_attn_; }
    // Must be called once after initialize(), before any upload_slot(), to supply RoPE params.
    void set_rope_config(bool has_rope, float rope_freq_base) { has_rope_ = has_rope; rope_freq_base_ = rope_freq_base; }
    float get_rope_freq_base() const { return rope_freq_base_; }
    bool  get_has_rope() const { return has_rope_; }
    struct ggml_tensor * get_anchors_K() { return anchors_K_; }
    struct ggml_tensor * get_anchors_V() { return anchors_V_; }
    struct ggml_tensor * get_seq_lens() { return seq_lens_; }
    struct ggml_tensor * get_scales() { return scales_; }
    struct ggml_tensor * get_desc_matrix() { return desc_matrix_; }
    struct ggml_tensor * get_anchor_positions() { return anchor_positions_; }

    DiffKVBlockStateTable & get_state_table() { return state_table_; }

    // Slot allocation & free-list
    int allocate_slot();
    void free_slot(int slot_id);
    void reset_slots();
    int get_free_slots_count();
    void zero_all_tensors();

    // Upload/Download specific slot
    void upload_slot(int slot_id);
    void download_slot(int slot_id);

    int get_pool_version() const { return pool_version_.load(); }
    void increment_pool_version() { pool_version_.fetch_add(1); }

    bool slot_device_has_data(int slot_id) const {
        if (slot_id < 0 || slot_id >= (int)slot_device_has_data_.size()) return false;
        return slot_device_has_data_[slot_id];
    }

    // Host mirror getters
    const int8_t* get_host_U() const { return host_U_.data(); }
    int8_t* get_host_U() { return host_U_.data(); }
    const ggml_fp16_t* get_host_U_scale() const { return host_U_scale_.data(); }
    ggml_fp16_t* get_host_U_scale() { return host_U_scale_.data(); }
    const ggml_fp16_t* get_host_VK() const { return host_VK_.data(); }
    ggml_fp16_t* get_host_VK() { return host_VK_.data(); }
    const ggml_fp16_t* get_host_VV() const { return host_VV_.data(); }
    ggml_fp16_t* get_host_VV() { return host_VV_.data(); }
    const ggml_fp16_t* get_host_anchors_K() const { return host_anchors_K_.data(); }
    ggml_fp16_t* get_host_anchors_K() { return host_anchors_K_.data(); }
    const ggml_fp16_t* get_host_anchors_V() const { return host_anchors_V_.data(); }
    ggml_fp16_t* get_host_anchors_V() { return host_anchors_V_.data(); }
    const int32_t* get_host_seq_lens() const { return host_seq_lens_.data(); }
    int32_t* get_host_seq_lens() { return host_seq_lens_.data(); }
    const ggml_fp16_t* get_host_scales() const { return host_scales_.data(); }
    ggml_fp16_t* get_host_scales() { return host_scales_.data(); }
    const int32_t* get_host_anchor_positions() const { return host_anchor_positions_.data(); }
    int32_t* get_host_anchor_positions() { return host_anchor_positions_.data(); }
    const float* get_host_desc_matrix() const { return host_desc_matrix_.data(); }
    float* get_host_desc_matrix() { return host_desc_matrix_.data(); }

    // Precomputed RoPE-rotated key buffers (fp32); filled by upload_slot when
    // native_attn_ is enabled. Returns nullptr when disabled or not yet allocated.
    // Layout: host_VK_rot_     [slot * rank * kv_heads * D + r * kv_heads * D + kv * D + d]
    //         host_anchorK_rot_[slot * kv_heads * D + kv * D + d]
    const float* get_host_VK_rot() const      { return host_VK_rot_.empty()      ? nullptr : host_VK_rot_.data(); }
    float*       get_host_VK_rot()            { return host_VK_rot_.empty()      ? nullptr : host_VK_rot_.data(); }
    const float* get_host_anchorK_rot() const { return host_anchorK_rot_.empty() ? nullptr : host_anchorK_rot_.data(); }
    float*       get_host_anchorK_rot()       { return host_anchorK_rot_.empty() ? nullptr : host_anchorK_rot_.data(); }

    // F9: per-block sparse residuals (exact corrections for the highest-error tokens
    // the low-rank SVD failed to capture — e.g. digits). MAX_RESIDUAL tokens per K/V.
    static constexpr int MAX_RESIDUAL = 8;
    int32_t* get_host_res_K_pos() { return host_res_K_pos_.data(); }
    int32_t* get_host_res_V_pos() { return host_res_V_pos_.data(); }
    ggml_fp16_t* get_host_res_K_val() { return host_res_K_val_.data(); }
    ggml_fp16_t* get_host_res_V_val() { return host_res_V_val_.data(); }
    const int32_t* get_host_res_K_pos() const { return host_res_K_pos_.data(); }
    const int32_t* get_host_res_V_pos() const { return host_res_V_pos_.data(); }
    const ggml_fp16_t* get_host_res_K_val() const { return host_res_K_val_.data(); }
    const ggml_fp16_t* get_host_res_V_val() const { return host_res_V_val_.data(); }

private:
    int n_slots_ = 0;
    int rank_ = 0;
    int head_dim_ = 0;
    int kv_heads_ = 0;
    int desc_dim_ = 0;
    int S_max_ = 64;

    struct ggml_context * pool_ctx_ = nullptr;
    struct ggml_backend_buffer * pool_buffer_ = nullptr;

    // Pool Tensors
    struct ggml_tensor * U_ = nullptr;
    struct ggml_tensor * U_f16_ = nullptr;   // f16 mirror of U (int8 values cast to f16) for native ggml-metal gather/matmul
    struct ggml_tensor * U_scale_ = nullptr;
    struct ggml_tensor * VK_ = nullptr;
    struct ggml_tensor * VV_ = nullptr;
    struct ggml_tensor * VK_rot_ = nullptr;       // [head_dim, kv_heads, rank, n_slots] FP32, RoPE'd at anchor pos (native attn only)
    struct ggml_tensor * anchorK_rot_ = nullptr;  // [head_dim, kv_heads, n_slots]       FP32, RoPE'd at anchor pos (native attn only)
    struct ggml_tensor * valid_mask_ = nullptr;   // [S_max, n_slots] f16 additive -inf padding bias (native attn only)
    struct ggml_tensor * anchors_K_ = nullptr;
    struct ggml_tensor * anchors_V_ = nullptr;
    struct ggml_tensor * seq_lens_ = nullptr;
    struct ggml_tensor * scales_ = nullptr;
    struct ggml_tensor * desc_matrix_ = nullptr;
    struct ggml_tensor * anchor_positions_ = nullptr;  // [n_slots] int32: actual sequence position of each block's anchor

    // Host-side mirror buffers
    std::vector<int8_t, PageAlignedAllocator<int8_t>> host_U_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_U_scale_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_VK_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_VV_;
    std::vector<float, PageAlignedAllocator<float>> host_VK_rot_;       // fp32 (precision-match CPU)
    std::vector<float, PageAlignedAllocator<float>> host_anchorK_rot_;  // fp32 (precision-match CPU)
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_valid_mask_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_anchors_K_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_anchors_V_;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> host_seq_lens_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_scales_;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> host_anchor_positions_;
    std::vector<float, PageAlignedAllocator<float>> host_desc_matrix_;

    // F9 residual host buffers: positions [n_slots*MAX_RESIDUAL] int32 (-1 = unused),
    // values [n_slots*MAX_RESIDUAL*kv_heads*head_dim] fp16 (exact delta residual).
    std::vector<int32_t, PageAlignedAllocator<int32_t>> host_res_K_pos_;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> host_res_V_pos_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_res_K_val_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_res_V_val_;

    DiffKVBlockStateTable state_table_;

    // Native-attn config / state
    bool native_attn_ = false;        // set from DIFFKV_NATIVE_ATTN in initialize()
    bool has_rope_ = true;            // set via set_rope_config()
    float rope_freq_base_ = 1000000.0f;

    // Slot allocator state
    std::vector<int> free_slots_;
    std::mutex slot_mutex_;
    std::atomic<int> pool_version_{0};
    std::vector<bool> slot_device_has_data_;
};

} // namespace diffkv
