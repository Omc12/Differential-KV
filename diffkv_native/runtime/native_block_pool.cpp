#include "runtime/native_block_pool.hpp"
#include "ggml-alloc.h"
#include <iostream>
#include <algorithm>

namespace diffkv {

NativeBlockPool::NativeBlockPool() {}

NativeBlockPool::~NativeBlockPool() {
    if (pool_buffer_) {
        ggml_backend_buffer_free(pool_buffer_);
    }
    if (pool_ctx_) {
        ggml_free(pool_ctx_);
    }
}

bool NativeBlockPool::initialize(int n_slots, int rank, int head_dim, int kv_heads, int desc_dim, ggml_backend_buffer_type_t buft) {
    n_slots_ = n_slots;
    rank_ = rank;
    head_dim_ = head_dim;
    kv_heads_ = kv_heads;
    desc_dim_ = desc_dim;

    std::cerr << "[NativeBlockPool] Initializing Block Pool with " << n_slots << " slots, SVD rank " << rank << " ..." << std::endl;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024, // 1MB metadata context
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,            // allocate actual data in backend buffer
    };

    pool_ctx_ = ggml_init(params);
    if (!pool_ctx_) {
        std::cerr << "[NativeBlockPool] Error: Failed to initialize pool ggml_context!" << std::endl;
        return false;
    }

    const int S_max = 64; // Block size

    // Create tensor descriptors in the pool context
    U_           = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_I8,  rank, S_max, n_slots);
    U_scale_     = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_F16, n_slots);
    
    // VK and VV shape: [head_dim, kv_heads, rank, n_slots]
    VK_          = ggml_new_tensor_4d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, rank, n_slots);
    VV_          = ggml_new_tensor_4d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, rank, n_slots);
    
    // anchors shape: [head_dim, kv_heads, n_slots]
    anchors_K_   = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, n_slots);
    anchors_V_   = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, n_slots);
    
    seq_lens_    = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_I32, n_slots);
    scales_      = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_F16, n_slots);
    
    desc_matrix_     = ggml_new_tensor_2d(pool_ctx_, GGML_TYPE_F32, desc_dim, n_slots);
    anchor_positions_ = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_I32, n_slots);

    // Assign names for debug visibility
    ggml_set_name(U_,           "pool.U");
    ggml_set_name(U_scale_,     "pool.U_scale");
    ggml_set_name(VK_,          "pool.V_K");
    ggml_set_name(VV_,          "pool.V_V");
    ggml_set_name(anchors_K_,   "pool.anchors_K");
    ggml_set_name(anchors_V_,   "pool.anchors_V");
    ggml_set_name(seq_lens_,      "pool.seq_lens");
    ggml_set_name(scales_,        "pool.scales");
    ggml_set_name(desc_matrix_,   "pool.desc_matrix");
    ggml_set_name(anchor_positions_, "pool.anchor_positions");

    // Allocate buffer for actual data in the backend memory type
    pool_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(pool_ctx_, buft);
    if (!pool_buffer_) {
        std::cerr << "[NativeBlockPool] Error: Failed to allocate pool tensors in backend buffer!" << std::endl;
        return false;
    }

    std::cerr << "[NativeBlockPool] Allocated " 
              << ggml_backend_buffer_get_size(pool_buffer_) / (1024 * 1024) 
              << " MB on backend buffer type: " 
              << ggml_backend_buft_name(buft) << std::endl;

    // Initialize all slot states to Freed
    for (int i = 0; i < n_slots; ++i) {
        state_table_.force_invalidate(i); // sets state to Invalid
        state_table_.transition(i, BlockState::Invalid, BlockState::Freed);
    }

    // Populate free list
    free_slots_.clear();
    for (int i = 0; i < n_slots; ++i) {
        free_slots_.push_back(i);
    }

    return true;
}

int NativeBlockPool::allocate_slot() {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    if (free_slots_.empty()) {
        return -1;
    }
    int slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
}

void NativeBlockPool::free_slot(int slot_id) {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    if (std::find(free_slots_.begin(), free_slots_.end(), slot_id) == free_slots_.end()) {
        free_slots_.push_back(slot_id);
    }
    state_table_.force_invalidate(slot_id);
    if (state_table_.get(slot_id) == BlockState::Invalid) {
        state_table_.transition(slot_id, BlockState::Invalid, BlockState::Freed);
    }
}

void NativeBlockPool::reset_slots() {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    free_slots_.clear();
    for (int i = 0; i < n_slots_; ++i) {
        free_slots_.push_back(i);
    }
}

int NativeBlockPool::get_free_slots_count() {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    return free_slots_.size();
}

} // namespace diffkv
