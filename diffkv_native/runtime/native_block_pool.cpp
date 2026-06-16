#include "runtime/native_block_pool.hpp"
#include "ggml-alloc.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace diffkv {

namespace {
// Rotate a single head_dim vector by NEOX RoPE at absolute position `pos`.
// Matches the kernel's K/anchor rotation exactly (diffkv_attention.cpp lines 84-91 / 152-159):
//   out[d] = raw[d]*cos(angle) + rot_contrib*sin(angle)
// with partner = d±half_d, rot_contrib = (d<half) ? -raw[partner] : +raw[partner],
//      idx = d % half_d, theta = freq_base^(-2*idx/D), angle = pos*theta.
inline void rope_rotate_vec(const ggml_fp16_t* in, ggml_fp16_t* out, int D, float pos, float freq_base) {
    const int half_d = D / 2;
    for (int d = 0; d < D; ++d) {
        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
        float raw  = ggml_fp16_to_fp32(in[d]);
        float rawp = ggml_fp16_to_fp32(in[partner]);
        float rot_contrib = (d < half_d) ? -rawp : rawp;
        int idx = (d < half_d) ? d : (d - half_d);
        float theta = 1.0f / std::pow(freq_base, (2.0f * idx) / D);
        float angle = pos * theta;
        out[d] = ggml_fp32_to_fp16(raw * std::cos(angle) + rot_contrib * std::sin(angle));
    }
}
// fp32-output overload: the native pool stores RoPE'd K as fp32 so the precomputed rotation
// matches the CPU reference's runtime fp32 rotation (storing fp16 loses precision at large
// anchor positions ~500+ and cascades into token flips on degenerate repetitive prompts).
inline void rope_rotate_vec(const ggml_fp16_t* in, float* out, int D, float pos, float freq_base) {
    const int half_d = D / 2;
    for (int d = 0; d < D; ++d) {
        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
        float raw  = ggml_fp16_to_fp32(in[d]);
        float rawp = ggml_fp16_to_fp32(in[partner]);
        float rot_contrib = (d < half_d) ? -rawp : rawp;
        int idx = (d < half_d) ? d : (d - half_d);
        float theta = 1.0f / std::pow(freq_base, (2.0f * idx) / D);
        float angle = pos * theta;
        out[d] = raw * std::cos(angle) + rot_contrib * std::sin(angle);
    }
}
} // namespace

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

    // Native ggml-metal attention path (gated). When enabled, we allocate + fill the
    // precomputed RoPE'd key tensors so the in-graph subgraph never has to rope/gather i32.
    native_attn_ = true; // ENABLED BY DEFAULT!
    if (const char* e = std::getenv("DIFFKV_NATIVE_ATTN")) {
        native_attn_ = (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "yes" || std::string(e) == "on");
    }

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
    U_f16_       = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F16, rank, S_max, n_slots);  // native-ggml mirror
    U_scale_     = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_F16, n_slots);
    
    // VK and VV shape: [head_dim, kv_heads, rank, n_slots]
    VK_          = ggml_new_tensor_4d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, rank, n_slots);
    VV_          = ggml_new_tensor_4d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, rank, n_slots);
    
    // anchors shape: [head_dim, kv_heads, n_slots]
    anchors_K_   = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, n_slots);
    anchors_V_   = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, n_slots);

    // Native-attn precomputed RoPE'd keys (same shapes as VK_ / anchors_K_). Only allocated
    // when native attn is enabled, to keep the default path RAM-conservative.
    if (native_attn_) {
        VK_rot_      = ggml_new_tensor_4d(pool_ctx_, GGML_TYPE_F32, head_dim, kv_heads, rank, n_slots);
        anchorK_rot_ = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F32, head_dim, kv_heads, n_slots);
        valid_mask_  = ggml_new_tensor_2d(pool_ctx_, GGML_TYPE_F16, S_max, n_slots);
    }
    
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
    if (VK_rot_)      ggml_set_name(VK_rot_,      "pool.V_K_rot");
    if (anchorK_rot_) ggml_set_name(anchorK_rot_, "pool.anchors_K_rot");
    if (valid_mask_)  ggml_set_name(valid_mask_,  "pool.valid_mask");

    // Allocate buffer for actual data in the backend memory type
    pool_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(pool_ctx_, buft);
    if (!pool_buffer_) {
        std::cerr << "[NativeBlockPool] Error: Failed to allocate pool tensors in backend buffer!" << std::endl;
        return false;
    }

    host_U_.resize(ggml_nelements(U_), 0);
    host_U_f16_.resize(ggml_nelements(U_f16_), ggml_fp32_to_fp16(0.0f));
    host_U_scale_.resize(ggml_nelements(U_scale_), ggml_fp32_to_fp16(0.0f));
    host_VK_.resize(ggml_nelements(VK_), ggml_fp32_to_fp16(0.0f));
    host_VV_.resize(ggml_nelements(VV_), ggml_fp32_to_fp16(0.0f));
    host_anchors_K_.resize(ggml_nelements(anchors_K_), ggml_fp32_to_fp16(0.0f));
    host_anchors_V_.resize(ggml_nelements(anchors_V_), ggml_fp32_to_fp16(0.0f));
    host_seq_lens_.resize(ggml_nelements(seq_lens_), 0);
    host_scales_.resize(ggml_nelements(scales_), ggml_fp32_to_fp16(0.0f));
    host_anchor_positions_.resize(ggml_nelements(anchor_positions_), 0);
    host_desc_matrix_.resize(ggml_nelements(desc_matrix_), 0.0f);
    if (VK_rot_)      host_VK_rot_.resize(ggml_nelements(VK_rot_), 0.0f);
    if (anchorK_rot_) host_anchorK_rot_.resize(ggml_nelements(anchorK_rot_), 0.0f);
    if (valid_mask_)  host_valid_mask_.resize(ggml_nelements(valid_mask_), ggml_fp32_to_fp16(-INFINITY));

    // F9: residual host buffers (CPU-only path for now; -1 positions = unused).
    const int F = kv_heads * head_dim;
    host_res_K_pos_.resize((size_t)n_slots * MAX_RESIDUAL, -1);
    host_res_V_pos_.resize((size_t)n_slots * MAX_RESIDUAL, -1);
    host_res_K_val_.resize((size_t)n_slots * MAX_RESIDUAL * F, ggml_fp32_to_fp16(0.0f));
    host_res_V_val_.resize((size_t)n_slots * MAX_RESIDUAL * F, ggml_fp32_to_fp16(0.0f));

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

    slot_device_has_data_.resize(n_slots, false);
    pool_version_.store(0);
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
        host_seq_lens_[slot_id] = 0;
    }
}

void NativeBlockPool::reset_slots() {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    free_slots_.clear();
    for (int i = 0; i < n_slots_; ++i) {
        state_table_.force_invalidate(i);
        free_slots_.push_back(i);
        host_seq_lens_[i] = 0;
    }
    std::fill(slot_device_has_data_.begin(), slot_device_has_data_.end(), false);
    increment_pool_version();
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
    if (U_f16_) {
        std::fill(host_U_f16_.begin(), host_U_f16_.end(), ggml_fp32_to_fp16(0.0f));
        std::vector<ggml_fp16_t> zeros(ggml_nelements(U_f16_), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(U_f16_, zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));
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
    if (VK_rot_) {
        std::fill(host_VK_rot_.begin(), host_VK_rot_.end(), 0.0f);
        std::vector<float> zeros(ggml_nelements(VK_rot_), 0.0f);
        ggml_backend_tensor_set(VK_rot_, zeros.data(), 0, zeros.size() * sizeof(float));
    }
    if (anchorK_rot_) {
        std::fill(host_anchorK_rot_.begin(), host_anchorK_rot_.end(), 0.0f);
        std::vector<float> zeros(ggml_nelements(anchorK_rot_), 0.0f);
        ggml_backend_tensor_set(anchorK_rot_, zeros.data(), 0, zeros.size() * sizeof(float));
    }
    if (valid_mask_) {
        // Default to fully-masked (-inf); upload_slot fills the valid prefix per slot.
        std::fill(host_valid_mask_.begin(), host_valid_mask_.end(), ggml_fp32_to_fp16(-INFINITY));
        std::vector<ggml_fp16_t> neg(ggml_nelements(valid_mask_), ggml_fp32_to_fp16(-INFINITY));
        ggml_backend_tensor_set(valid_mask_, neg.data(), 0, neg.size() * sizeof(ggml_fp16_t));
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
    std::fill(slot_device_has_data_.begin(), slot_device_has_data_.end(), false);
    increment_pool_version();
}

void NativeBlockPool::upload_slot(int slot_id) {
    if (slot_id < 0 || slot_id >= n_slots_) return;
    if (std::getenv("DIFFKV_DBG_ROT")) {
        static std::atomic<int> cnt{0};
        if (cnt.fetch_add(1) < 5)
            std::cerr << "[DBG_UP] upload_slot(" << slot_id << ") native_attn_=" << native_attn_
                      << " VK_rot_=" << (void*)VK_rot_ << " has_rope_=" << has_rope_ << std::endl;
    }
    
    ggml_backend_tensor_set(U_, host_U_.data() + slot_id * rank_ * 64, slot_id * U_->nb[2], rank_ * 64 * sizeof(int8_t));
    // f16 mirror of U (native ggml gather): cast this slot's int8 values to f16.
    {
        const int8_t* src = host_U_.data() + slot_id * rank_ * 64;
        ggml_fp16_t* dst = host_U_f16_.data() + slot_id * rank_ * 64;
        for (int i = 0; i < rank_ * 64; ++i) dst[i] = ggml_fp32_to_fp16((float)src[i]);
        ggml_backend_tensor_set(U_f16_, dst, slot_id * U_f16_->nb[2], rank_ * 64 * sizeof(ggml_fp16_t));
    }
    ggml_backend_tensor_set(U_scale_, host_U_scale_.data() + slot_id, slot_id * U_scale_->nb[0], sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(VK_, host_VK_.data() + slot_id * head_dim_ * kv_heads_ * rank_, slot_id * VK_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(VV_, host_VV_.data() + slot_id * head_dim_ * kv_heads_ * rank_, slot_id * VV_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(anchors_K_, host_anchors_K_.data() + slot_id * head_dim_ * kv_heads_, slot_id * anchors_K_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(anchors_V_, host_anchors_V_.data() + slot_id * head_dim_ * kv_heads_, slot_id * anchors_V_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(seq_lens_, host_seq_lens_.data() + slot_id, slot_id * seq_lens_->nb[0], sizeof(int32_t));
    ggml_backend_tensor_set(scales_, host_scales_.data() + slot_id, slot_id * scales_->nb[0], sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(anchor_positions_, host_anchor_positions_.data() + slot_id, slot_id * anchor_positions_->nb[0], sizeof(int32_t));

    // Native attn: precompute this slot's RoPE'd keys at its (fixed) anchor position so the
    // in-graph subgraph can gather f16 and dot with the in-graph query — no in-graph rope.
    if (native_attn_ && VK_rot_ && anchorK_rot_) {
        const float pos = (float)host_anchor_positions_[slot_id];
        const int D = head_dim_;
        // anchors_K_rot: one head_dim vector per kv_head
        for (int kv = 0; kv < kv_heads_; ++kv) {
            const size_t off = (size_t)slot_id * kv_heads_ * D + (size_t)kv * D;
            const ggml_fp16_t* src = host_anchors_K_.data() + off;
            float* dst = host_anchorK_rot_.data() + off;
            if (has_rope_) rope_rotate_vec(src, dst, D, pos, rope_freq_base_);
            else for (int d = 0; d < D; ++d) dst[d] = ggml_fp16_to_fp32(src[d]);
        }
        ggml_backend_tensor_set(anchorK_rot_, host_anchorK_rot_.data() + (size_t)slot_id * kv_heads_ * D,
                                slot_id * anchorK_rot_->nb[2], (size_t)kv_heads_ * D * sizeof(float));
        // VK_rot: one head_dim vector per (rank, kv_head)
        for (int r = 0; r < rank_; ++r) {
            for (int kv = 0; kv < kv_heads_; ++kv) {
                const size_t off = (size_t)slot_id * rank_ * kv_heads_ * D + (size_t)r * kv_heads_ * D + (size_t)kv * D;
                const ggml_fp16_t* src = host_VK_.data() + off;
                float* dst = host_VK_rot_.data() + off;
                if (has_rope_) rope_rotate_vec(src, dst, D, pos, rope_freq_base_);
                else for (int d = 0; d < D; ++d) dst[d] = ggml_fp16_to_fp32(src[d]);
            }
        }
        ggml_backend_tensor_set(VK_rot_, host_VK_rot_.data() + (size_t)slot_id * rank_ * kv_heads_ * D,
                                slot_id * VK_rot_->nb[3], (size_t)rank_ * kv_heads_ * D * sizeof(float));
        static std::atomic<int> rotcnt{0};
        if (std::getenv("DIFFKV_DBG_ROT") && rotcnt.fetch_add(1) < 4) {
            double nvk = 0, nvkr = 0, nak = 0, nakr = 0;
            const ggml_fp16_t* vk = host_VK_.data() + (size_t)slot_id * rank_ * kv_heads_ * D;
            for (int i = 0; i < rank_ * kv_heads_ * D; ++i) { double a = ggml_fp16_to_fp32(vk[i]); nvk += a*a; double b = host_VK_rot_[(size_t)slot_id*rank_*kv_heads_*D + i]; nvkr += b*b; }
            const ggml_fp16_t* ak = host_anchors_K_.data() + (size_t)slot_id * kv_heads_ * D;
            for (int i = 0; i < kv_heads_ * D; ++i) { double a = ggml_fp16_to_fp32(ak[i]); nak += a*a; double b = host_anchorK_rot_[(size_t)slot_id*kv_heads_*D + i]; nakr += b*b; }
            std::cerr << "[DBG_ROT] slot " << slot_id << " pos=" << pos << " |VK|=" << std::sqrt(nvk) << " |VK_rot|=" << std::sqrt(nvkr) << " |aK|=" << std::sqrt(nak) << " |aK_rot|=" << std::sqrt(nakr) << " seq_len=" << host_seq_lens_[slot_id] << std::endl;
        }
    }
    if (native_attn_ && valid_mask_) {
        const int S_max = 64;
        const int slen = host_seq_lens_[slot_id];
        ggml_fp16_t* m = host_valid_mask_.data() + (size_t)slot_id * S_max;
        for (int t = 0; t < S_max; ++t) m[t] = ggml_fp32_to_fp16(t < slen ? 0.0f : -INFINITY);
        ggml_backend_tensor_set(valid_mask_, m, slot_id * valid_mask_->nb[1], (size_t)S_max * sizeof(ggml_fp16_t));
    }
    slot_device_has_data_[slot_id] = (host_seq_lens_[slot_id] > 0);
    increment_pool_version();
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
