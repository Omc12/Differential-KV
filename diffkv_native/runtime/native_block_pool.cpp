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

    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
        std::cerr << "[NativeBlockPool] Initializing Block Pool with " << n_slots << " slots, SVD rank " << rank << " ..." << std::endl;
    }

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

    host_U_.resize(ggml_nelements(U_), 0);
    host_U_scale_.resize(ggml_nelements(U_scale_), ggml_fp32_to_fp16(0.0f));
    host_VK_.resize(ggml_nelements(VK_), ggml_fp32_to_fp16(0.0f));
    host_VV_.resize(ggml_nelements(VV_), ggml_fp32_to_fp16(0.0f));
    host_anchors_K_.resize(ggml_nelements(anchors_K_), ggml_fp32_to_fp16(0.0f));
    host_anchors_V_.resize(ggml_nelements(anchors_V_), ggml_fp32_to_fp16(0.0f));
    host_seq_lens_.resize(ggml_nelements(seq_lens_), 0);
    host_scales_.resize(ggml_nelements(scales_), ggml_fp32_to_fp16(0.0f));
    host_anchor_positions_.resize(ggml_nelements(anchor_positions_), 0);
    host_desc_matrix_.resize(ggml_nelements(desc_matrix_), 0.0f);

    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
        std::cerr << "[NativeBlockPool] Allocated " 
                  << ggml_backend_buffer_get_size(pool_buffer_) / (1024 * 1024) 
                  << " MB on backend buffer type: " 
                  << ggml_backend_buft_name(buft) << std::endl;
    }

    // Initialize all slot states to Freed
    for (int i = 0; i < n_slots; ++i) {
        state_table_.force_invalidate(i); // sets state to Invalid
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
        state_table_.force_invalidate(slot_id);
        free_slots_.push_back(slot_id);
    }
}

void NativeBlockPool::reset_slots() {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    free_slots_.clear();
    for (int i = 0; i < n_slots_; ++i) {
        state_table_.force_invalidate(i);
        free_slots_.push_back(i);
    }
}

int NativeBlockPool::get_free_slots_count() {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    return free_slots_.size();
}

void NativeBlockPool::zero_all_tensors() {
    std::fill(host_U_.begin(), host_U_.end(), 0);
    std::fill(host_U_scale_.begin(), host_U_scale_.end(), ggml_fp32_to_fp16(0.0f));
    std::fill(host_VK_.begin(), host_VK_.end(), ggml_fp32_to_fp16(0.0f));
    std::fill(host_VV_.begin(), host_VV_.end(), ggml_fp32_to_fp16(0.0f));
    std::fill(host_anchors_K_.begin(), host_anchors_K_.end(), ggml_fp32_to_fp16(0.0f));
    std::fill(host_anchors_V_.begin(), host_anchors_V_.end(), ggml_fp32_to_fp16(0.0f));
    std::fill(host_seq_lens_.begin(), host_seq_lens_.end(), 0);
    std::fill(host_scales_.begin(), host_scales_.end(), ggml_fp32_to_fp16(0.0f));
    std::fill(host_anchor_positions_.begin(), host_anchor_positions_.end(), 0);
    std::fill(host_desc_matrix_.begin(), host_desc_matrix_.end(), 0.0f);

    if (U_) {
        std::vector<int8_t> zeros(ggml_nelements(U_), 0);
        ggml_backend_tensor_set(U_, zeros.data(), 0, zeros.size() * sizeof(int8_t));
    }
    if (U_scale_) {
        std::vector<ggml_fp16_t> zeros(ggml_nelements(U_scale_), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(U_scale_, zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));
    }
    if (VK_) {
        std::vector<ggml_fp16_t> zeros(ggml_nelements(VK_), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(VK_, zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));
    }
    if (VV_) {
        std::vector<ggml_fp16_t> zeros(ggml_nelements(VV_), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(VV_, zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));
    }
    if (anchors_K_) {
        std::vector<ggml_fp16_t> zeros(ggml_nelements(anchors_K_), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(anchors_K_, zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));
    }
    if (anchors_V_) {
        std::vector<ggml_fp16_t> zeros(ggml_nelements(anchors_V_), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(anchors_V_, zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));
    }
    if (seq_lens_) {
        std::vector<int32_t> zeros(ggml_nelements(seq_lens_), 0);
        ggml_backend_tensor_set(seq_lens_, zeros.data(), 0, zeros.size() * sizeof(int32_t));
    }
    if (scales_) {
        std::vector<ggml_fp16_t> zeros(ggml_nelements(scales_), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(scales_, zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));
    }
    if (desc_matrix_) {
        std::vector<float> zeros(ggml_nelements(desc_matrix_), 0.0f);
        ggml_backend_tensor_set(desc_matrix_, zeros.data(), 0, zeros.size() * sizeof(float));
    }
    if (anchor_positions_) {
        std::vector<int32_t> zeros(ggml_nelements(anchor_positions_), 0);
        ggml_backend_tensor_set(anchor_positions_, zeros.data(), 0, zeros.size() * sizeof(int32_t));
    }
}

void NativeBlockPool::upload_slot(int slot_id) {
    if (slot_id < 0 || slot_id >= n_slots_) return;
    
    ggml_backend_tensor_set(U_, host_U_.data() + slot_id * rank_ * 64, slot_id * U_->nb[2], rank_ * 64 * sizeof(int8_t));
    ggml_backend_tensor_set(U_scale_, host_U_scale_.data() + slot_id, slot_id * U_scale_->nb[0], sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(VK_, host_VK_.data() + slot_id * head_dim_ * kv_heads_ * rank_, slot_id * VK_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(VV_, host_VV_.data() + slot_id * head_dim_ * kv_heads_ * rank_, slot_id * VV_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(anchors_K_, host_anchors_K_.data() + slot_id * head_dim_ * kv_heads_, slot_id * anchors_K_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(anchors_V_, host_anchors_V_.data() + slot_id * head_dim_ * kv_heads_, slot_id * anchors_V_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(seq_lens_, host_seq_lens_.data() + slot_id, slot_id * seq_lens_->nb[0], sizeof(int32_t));
    ggml_backend_tensor_set(scales_, host_scales_.data() + slot_id, slot_id * scales_->nb[0], sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(anchor_positions_, host_anchor_positions_.data() + slot_id, slot_id * anchor_positions_->nb[0], sizeof(int32_t));
}

void NativeBlockPool::download_slot(int slot_id) {
    if (slot_id < 0 || slot_id >= n_slots_) return;
    
    ggml_backend_tensor_get(U_, host_U_.data() + slot_id * rank_ * 64, slot_id * U_->nb[2], rank_ * 64 * sizeof(int8_t));
    ggml_backend_tensor_get(U_scale_, host_U_scale_.data() + slot_id, slot_id * U_scale_->nb[0], sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(VK_, host_VK_.data() + slot_id * head_dim_ * kv_heads_ * rank_, slot_id * VK_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(VV_, host_VV_.data() + slot_id * head_dim_ * kv_heads_ * rank_, slot_id * VV_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(anchors_K_, host_anchors_K_.data() + slot_id * head_dim_ * kv_heads_, slot_id * anchors_K_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(anchors_V_, host_anchors_V_.data() + slot_id * head_dim_ * kv_heads_, slot_id * anchors_V_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(seq_lens_, host_seq_lens_.data() + slot_id, slot_id * seq_lens_->nb[0], sizeof(int32_t));
    ggml_backend_tensor_get(scales_, host_scales_.data() + slot_id, slot_id * scales_->nb[0], sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(anchor_positions_, host_anchor_positions_.data() + slot_id, slot_id * anchor_positions_->nb[0], sizeof(int32_t));
}

} // namespace diffkv
