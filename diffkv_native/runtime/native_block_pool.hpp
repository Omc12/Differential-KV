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

    bool initialize(int n_slots, int rank, int head_dim, int kv_heads, int desc_dim, ggml_backend_buffer_type_t buft);

    int get_rank() const { return rank_; }

    // Getters for GGML tensors
    struct ggml_tensor * get_U() { return U_; }
    struct ggml_tensor * get_U_scale() { return U_scale_; }
    struct ggml_tensor * get_VK() { return VK_; }
    struct ggml_tensor * get_VV() { return VV_; }
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

private:
    int n_slots_ = 0;
    int rank_ = 0;
    int head_dim_ = 0;
    int kv_heads_ = 0;
    int desc_dim_ = 0;

    struct ggml_context * pool_ctx_ = nullptr;
    struct ggml_backend_buffer * pool_buffer_ = nullptr;

    // Pool Tensors
    struct ggml_tensor * U_ = nullptr;
    struct ggml_tensor * U_scale_ = nullptr;
    struct ggml_tensor * VK_ = nullptr;
    struct ggml_tensor * VV_ = nullptr;
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
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_anchors_K_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_anchors_V_;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> host_seq_lens_;
    std::vector<ggml_fp16_t, PageAlignedAllocator<ggml_fp16_t>> host_scales_;
    std::vector<int32_t, PageAlignedAllocator<int32_t>> host_anchor_positions_;
    std::vector<float, PageAlignedAllocator<float>> host_desc_matrix_;

    DiffKVBlockStateTable state_table_;

    // Slot allocator state
    std::vector<int> free_slots_;
    std::mutex slot_mutex_;
};

} // namespace diffkv
