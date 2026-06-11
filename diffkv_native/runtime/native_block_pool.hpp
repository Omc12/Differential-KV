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

namespace diffkv {

class NativeBlockPool {
public:
    NativeBlockPool();
    ~NativeBlockPool();

    bool initialize(int n_slots, int rank, int head_dim, int kv_heads, int desc_dim, ggml_backend_buffer_type_t buft);

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

    DiffKVBlockStateTable state_table_;

    // Slot allocator state
    std::vector<int> free_slots_;
    std::mutex slot_mutex_;
};

} // namespace diffkv
