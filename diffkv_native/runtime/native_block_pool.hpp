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

#include <cstdlib>

namespace diffkv {

// ── Pool K rotation scheme (single source of truth for ingest + every decode path) ──
// Historically two sessions left the ingest and the decode paths on DIFFERENT RoPE
// schemes (ingest pre-rotated K at the within-block offset while the exact decode
// paths rotated the reconstruction again at the absolute token position), silently
// double-rotating every compressed token. All rotation decisions now read this one
// mode so ingest and decode cannot disagree again.
//   POOL_ROT_RAW     — pool holds raw K; decode rotates at true token positions.
//   POOL_ROT_WITHIN  — legacy: pool pre-rotated at within-block offsets; decode adds
//                      the block-start rotation (only the approximate paths compose
//                      this correctly).
//   POOL_ROT_ABS     — DEFAULT: pool holds K fully rotated at its absolute sequence
//                      position (exactly what the MLX reference stores). Decode does
//                      NO rotation for pool content (anchors, U·VK deltas, residuals),
//                      which also makes the cheap project-then-attend score exact.
enum PoolRotMode { POOL_ROT_RAW = 0, POOL_ROT_WITHIN = 1, POOL_ROT_ABS = 2 };

inline PoolRotMode pool_rot_mode() {
    static const PoolRotMode mode = []() {
        if (std::getenv("DIFFKV_NO_ROTATE_AT_INGEST")) return POOL_ROT_RAW;
        if (const char* e = std::getenv("DIFFKV_POOL_ABS_ROT")) {
            if (std::string(e) == "0") return POOL_ROT_WITHIN;
        }
        return POOL_ROT_ABS;
    }();
    return mode;
}

struct SlotHostBuffer {
    std::vector<int8_t, PageAlignedAllocator<int8_t>> U;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> U_row_scale;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> VK;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> VV;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> anchors_K;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> anchors_V;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> token_positions;
    std::vector<float, PageAlignedAllocator<float>> desc_matrix;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> res_K_pos;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> res_V_pos;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> res_K_val;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> res_V_val;
};

class NativeBlockPool {
public:
    NativeBlockPool();
    ~NativeBlockPool();

    bool initialize(int n_slots, int rank, int head_dim, int kv_heads, int desc_dim, ggml_backend_buffer_type_t buft, int S_max = 64, ggml_type kv_type = GGML_TYPE_Q8_0);

    int get_rank() const { return rank_; }
    int get_S_max() const { return S_max_; }
    int get_head_dim() const { return head_dim_; }
    int get_kv_heads() const { return kv_heads_; }
    int get_desc_dim() const { return desc_dim_; }
    int get_n_slots() const { return n_slots_; }
    ggml_type get_kv_type() const { return kv_type_; }

    // Getters for GGML tensors
    struct ggml_tensor * get_U() { return U_; }
    struct ggml_tensor * get_U_f16() { return U_f16_; }   // f16 mirror of U (native ggml gather)
    struct ggml_tensor * get_U_scale() { return U_scale_; }
    struct ggml_tensor * get_U_row_scale() { return U_row_scale_; }  // [S_max,n_slots] per-token int8 scale
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
    struct ggml_tensor * get_token_positions() { return token_positions_; }  // [S_max,n_slots] i32 true pos/token
    struct ggml_tensor * get_res_K_pos() { return res_K_pos_; }
    struct ggml_tensor * get_res_V_pos() { return res_V_pos_; }
    struct ggml_tensor * get_res_K_val() { return res_K_val_; }
    struct ggml_tensor * get_res_V_val() { return res_V_val_; }


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

    // Host mirror lazy allocation helpers
    SlotHostBuffer* ensure_slot_buffer(int slot_id);
    void free_slot_buffer(int slot_id);

    // Host mirror getters (per slot)
    const int8_t* get_host_U(int slot_id) const;
    int8_t* get_host_U(int slot_id);
    const ggml_fp16_t* get_host_U_row_scale(int slot_id) const;
    ggml_fp16_t* get_host_U_row_scale(int slot_id);
    const ggml_fp16_t* get_host_VK(int slot_id) const;
    ggml_fp16_t* get_host_VK(int slot_id);
    const ggml_fp16_t* get_host_VV(int slot_id) const;
    ggml_fp16_t* get_host_VV(int slot_id);
    const ggml_fp16_t* get_host_anchors_K(int slot_id) const;
    ggml_fp16_t* get_host_anchors_K(int slot_id);
    const ggml_fp16_t* get_host_anchors_V(int slot_id) const;
    ggml_fp16_t* get_host_anchors_V(int slot_id);
    const int32_t* get_host_token_positions(int slot_id) const;
    int32_t* get_host_token_positions(int slot_id);
    const float* get_host_desc_matrix(int slot_id) const;
    float* get_host_desc_matrix(int slot_id);

    // Flat tiny host getters
    const ggml_fp16_t* get_host_U_scale() const { return host_U_scale_.data(); }
    ggml_fp16_t* get_host_U_scale() { return host_U_scale_.data(); }
    const int32_t* get_host_seq_lens() const { return host_seq_lens_.data(); }
    int32_t* get_host_seq_lens() { return host_seq_lens_.data(); }
    const ggml_fp16_t* get_host_scales() const { return host_scales_.data(); }
    ggml_fp16_t* get_host_scales() { return host_scales_.data(); }
    const int32_t* get_host_anchor_positions() const { return host_anchor_positions_.data(); }
    int32_t* get_host_anchor_positions() { return host_anchor_positions_.data(); }

    // Precomputed RoPE-rotated key buffers (fp32); filled by upload_slot when
    // native_attn_ is enabled. Returns nullptr when disabled or not yet allocated.
    // Layout: host_VK_rot_     [slot * rank * kv_heads * D + r * kv_heads * D + kv * D + d]
    //         host_anchorK_rot_[slot * kv_heads * D + kv * D + d]
    const float* get_host_VK_rot() const      { return nullptr; }
    float*       get_host_VK_rot()            { return nullptr; }
    const float* get_host_anchorK_rot() const { return nullptr; }
    float*       get_host_anchorK_rot()       { return nullptr; }

    // F9: per-block sparse residuals (exact corrections for the highest-error tokens
    // the low-rank SVD failed to capture — e.g. digits). MAX_RESIDUAL tokens per K/V.
    static int MAX_RESIDUAL;
    int32_t* get_host_res_K_pos(int slot_id);
    int32_t* get_host_res_V_pos(int slot_id);
    ggml_fp16_t* get_host_res_K_val(int slot_id);
    ggml_fp16_t* get_host_res_V_val(int slot_id);
    const int32_t* get_host_res_K_pos(int slot_id) const;
    const int32_t* get_host_res_V_pos(int slot_id) const;
    const ggml_fp16_t* get_host_res_K_val(int slot_id) const;
    const ggml_fp16_t* get_host_res_V_val(int slot_id) const;

private:
    ggml_backend_buffer_type_t buft_ = nullptr;
    int n_allocated_slots_ = 0;
    bool grow_gpu_pool_impl(int min_slots_needed);
    bool grow_gpu_pool(int min_slots_needed);
    void upload_slot_impl(int slot_id);

    int n_slots_ = 0;
    int rank_ = 0;
    int head_dim_ = 0;
    int kv_heads_ = 0;
    int desc_dim_ = 0;
    int S_max_ = 64;
    ggml_type kv_type_ = GGML_TYPE_Q8_0;

    struct ggml_context * pool_ctx_ = nullptr;
    struct ggml_backend_buffer * pool_buffer_ = nullptr;

    // Pool Tensors
    struct ggml_tensor * U_ = nullptr;
    struct ggml_tensor * U_f16_ = nullptr;   // f16 mirror of U (int8 values cast to f16) for native ggml-metal gather/matmul
    struct ggml_tensor * U_scale_ = nullptr;
    struct ggml_tensor * U_row_scale_ = nullptr;  // [S_max, n_slots] f16 per-token int8 scale
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
    struct ggml_tensor * token_positions_ = nullptr;   // [S_max, n_slots] int32: true seq position of each delta token (RoPE)
    struct ggml_tensor * res_K_pos_ = nullptr;
    struct ggml_tensor * res_V_pos_ = nullptr;
    struct ggml_tensor * res_K_val_ = nullptr;
    struct ggml_tensor * res_V_val_ = nullptr;


    // Flat tiny host-side buffers
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_U_scale_;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> host_seq_lens_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_scales_;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> host_anchor_positions_;

    // Per-slot lazy allocated host-side buffers
    std::vector<std::unique_ptr<SlotHostBuffer>> slot_host_buffers_;

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
