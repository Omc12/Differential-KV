#include "runtime/native_block_pool.hpp"
#include "ggml-alloc.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace diffkv {

namespace {
// Rotate a single head_dim vector by NeoX RoPE at absolute position `pos`.
// Processes (d, d+half_d) pairs together: one std::pow + one cos/sin per pair
// instead of two std::pow + two cos/sin per pair in the old element-wise loop.
// Rotation formula (matches runtime attention and the Metal kernel):
//   out[d]        = x * cos - y * sin
//   out[d+half_d] = y * cos + x * sin
// where x = in[d], y = in[d+half_d], angle = pos / freq_base^(2d/D).
inline void rope_rotate_vec(const ggml_fp16_t* in, ggml_fp16_t* out, int D, float pos, float freq_base) {
    const int half_d = D / 2;
    for (int d = 0; d < half_d; ++d) {
        float theta = 1.0f / std::pow(freq_base, (2.0f * d) / D);
        float angle = pos * theta;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        float x = ggml_fp16_to_fp32(in[d]);
        float y = ggml_fp16_to_fp32(in[d + half_d]);
        out[d]          = ggml_fp32_to_fp16(x * cos_a - y * sin_a);
        out[d + half_d] = ggml_fp32_to_fp16(y * cos_a + x * sin_a);
    }
}
// fp32-output overload: host_anchorK_rot_ / host_VK_rot_ are fp32 to avoid precision
// loss at large positions (~500+) which cascades into token flips on repetitive prompts.
inline void rope_rotate_vec(const ggml_fp16_t* in, float* out, int D, float pos, float freq_base) {
    const int half_d = D / 2;
    for (int d = 0; d < half_d; ++d) {
        float theta = 1.0f / std::pow(freq_base, (2.0f * d) / D);
        float angle = pos * theta;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        float x = ggml_fp16_to_fp32(in[d]);
        float y = ggml_fp16_to_fp32(in[d + half_d]);
        out[d]          = x * cos_a - y * sin_a;
        out[d + half_d] = y * cos_a + x * sin_a;
    }
}

inline void quantize_q8_0(const ggml_fp16_t* src, void* dst, int n) {
    typedef struct {
        ggml_fp16_t d; // scale
        int8_t qs[32]; // values
    } block_q8_0_t;
    block_q8_0_t* d_ptr = (block_q8_0_t*)dst;
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; ++b) {
        float max_val = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float v = std::abs(ggml_fp16_to_fp32(src[b * 32 + i]));
            if (v > max_val) max_val = v;
        }
        float scale = max_val / 127.0f;
        d_ptr[b].d = ggml_fp32_to_fp16(scale);
        float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int i = 0; i < 32; ++i) {
            float v = ggml_fp16_to_fp32(src[b * 32 + i]);
            int q = std::round(v * inv_scale);
            d_ptr[b].qs[i] = std::max(-127, std::min(127, (int)q));
        }
    }
}

inline void dequantize_q8_0(const void* src, ggml_fp16_t* dst, int n) {
    typedef struct {
        ggml_fp16_t d;
        int8_t qs[32];
    } block_q8_0_t;
    const block_q8_0_t* s_ptr = (const block_q8_0_t*)src;
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; ++b) {
        float d = ggml_fp16_to_fp32(s_ptr[b].d);
        for (int i = 0; i < 32; ++i) {
            dst[b * 32 + i] = ggml_fp32_to_fp16(s_ptr[b].qs[i] * d);
        }
    }
}
inline void quantize_row(ggml_type type, const ggml_fp16_t* src, void* dst, int n) {
    std::vector<float> src_f32(n);
    for (int i = 0; i < n; ++i) {
        src_f32[i] = ggml_fp16_to_fp32(src[i]);
    }
    ggml_quantize_chunk(type, src_f32.data(), dst, 0, 1, n, nullptr);
}
} // anonymous namespace


NativeBlockPool::NativeBlockPool() {}

NativeBlockPool::~NativeBlockPool() {
    if (pool_buffer_) {
        ggml_backend_buffer_free(pool_buffer_);
    }
    if (pool_ctx_) {
        ggml_free(pool_ctx_);
    }
}

bool NativeBlockPool::initialize(int n_slots, int rank, int head_dim, int kv_heads, int desc_dim, ggml_backend_buffer_type_t buft, int S_max, ggml_type kv_type) {
    n_slots_ = n_slots;
    rank_ = rank;
    head_dim_ = head_dim;
    kv_heads_ = kv_heads;
    desc_dim_ = desc_dim;
    S_max_ = S_max;
    buft_ = buft;
    kv_type_ = kv_type;
    n_allocated_slots_ = std::min(16, n_slots);

    // Native ggml-metal attention path (gated). When enabled, we allocate + fill the
    // precomputed RoPE'd key tensors so the in-graph subgraph never has to rope/gather i32.
    native_attn_ = false;
    if (const char* env_native = std::getenv("DIFFKV_NATIVE_ATTN")) {
        if (std::strcmp(env_native, "1") == 0 || std::strcmp(env_native, "true") == 0 || std::strcmp(env_native, "yes") == 0 || std::strcmp(env_native, "on") == 0) {
            native_attn_ = true;
        }
    }
    bool skip_lowrank = false;

    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
        std::cerr << "[NativeBlockPool] Initializing Block Pool with " << n_slots << " slots (initially allocating " << n_allocated_slots_ << " slots), SVD rank " << rank << " (skip_lowrank=" << skip_lowrank << ") ..." << std::endl;
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

    // Create tensor descriptors in the pool context
    if (!skip_lowrank) {
        U_           = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_I8,  rank, S_max_, n_allocated_slots_);
        U_f16_       = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F16, rank, S_max_, n_allocated_slots_);  // native-ggml mirror
        U_scale_     = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_F16, n_allocated_slots_);
        U_row_scale_ = ggml_new_tensor_2d(pool_ctx_, GGML_TYPE_F16, S_max_, n_allocated_slots_);
        VK_          = ggml_new_tensor_4d(pool_ctx_, kv_type_, head_dim, kv_heads, rank, n_allocated_slots_);
        VV_          = ggml_new_tensor_4d(pool_ctx_, kv_type_, head_dim, kv_heads, rank, n_allocated_slots_);
    }
    
    // anchors shape: [head_dim, kv_heads, n_allocated_slots_]
    // Always F16 regardless of kv_type_: all Metal shaders (diffkv_decode.metal,
    // ggml-metal.metal) declare anchors_K/V as `half*`. Quantizing them would corrupt
    // both the CPU custom op and the fused kernel. Only VK_/VV_ are quantized.
    anchors_K_   = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, n_allocated_slots_);
    anchors_V_   = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F16, head_dim, kv_heads, n_allocated_slots_);

    // Native-attn precomputed RoPE'd keys (same shapes as VK_ / anchors_K_). Only allocated
    // when native attn is enabled, to keep the default path RAM-conservative.
    if (native_attn_ && !skip_lowrank) {
        VK_rot_      = ggml_new_tensor_4d(pool_ctx_, GGML_TYPE_F32, head_dim, kv_heads, rank, n_allocated_slots_);
        anchorK_rot_ = ggml_new_tensor_3d(pool_ctx_, GGML_TYPE_F32, head_dim, kv_heads, n_allocated_slots_);
        valid_mask_  = ggml_new_tensor_2d(pool_ctx_, GGML_TYPE_F16, S_max_, n_allocated_slots_);
    }
    
    seq_lens_    = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_I32, n_allocated_slots_);
    scales_      = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_F16, n_allocated_slots_);
    
    desc_matrix_     = ggml_new_tensor_2d(pool_ctx_, GGML_TYPE_F32, desc_dim, n_allocated_slots_);
    anchor_positions_ = ggml_new_tensor_1d(pool_ctx_, GGML_TYPE_I32, n_allocated_slots_);
    token_positions_  = ggml_new_tensor_2d(pool_ctx_, GGML_TYPE_I32, S_max_, n_allocated_slots_);  // true seq pos per delta token (RoPE fix)

    // Assign names for debug visibility
    if (U_)           ggml_set_name(U_,           "pool.U");
    if (U_scale_)     ggml_set_name(U_scale_,     "pool.U_scale");
    if (U_row_scale_) ggml_set_name(U_row_scale_, "pool.U_row_scale");
    if (VK_)          ggml_set_name(VK_,          "pool.V_K");
    if (VV_)          ggml_set_name(VV_,          "pool.V_V");
    ggml_set_name(anchors_K_,   "pool.anchors_K");
    ggml_set_name(anchors_V_,   "pool.anchors_V");
    ggml_set_name(seq_lens_,      "pool.seq_lens");
    ggml_set_name(scales_,        "pool.scales");
    ggml_set_name(desc_matrix_,   "pool.desc_matrix");
    ggml_set_name(anchor_positions_, "pool.anchor_positions");
    ggml_set_name(token_positions_,  "pool.token_positions");
    if (VK_rot_)      ggml_set_name(VK_rot_,      "pool.V_K_rot");
    if (anchorK_rot_) ggml_set_name(anchorK_rot_, "pool.anchors_K_rot");
    if (valid_mask_)  ggml_set_name(valid_mask_,  "pool.valid_mask");

    // Allocate buffer for actual data in the backend memory type
    pool_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(pool_ctx_, buft);
    if (!pool_buffer_) {
        std::cerr << "[NativeBlockPool] Error: Failed to allocate pool tensors in backend buffer!" << std::endl;
        return false;
    }

    host_U_scale_.resize(n_slots, ggml_fp32_to_fp16(0.0f));
    host_seq_lens_.resize(n_slots, 0);
    host_scales_.resize(n_slots, ggml_fp32_to_fp16(0.0f));
    host_anchor_positions_.resize(n_slots, 0);

    slot_host_buffers_.resize(n_slots);

    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
        std::cerr << "[NativeBlockPool] Allocated " 
                  << ggml_backend_buffer_get_size(pool_buffer_) / (1024 * 1024) 
                  << " MB on backend buffer type: " 
                  << ggml_backend_buft_name(buft) << std::endl;
    }

    // Initialize all slot states to Freed, allocating from 0 upwards
    for (int i = n_slots - 1; i >= 0; --i) {
        state_table_.force_invalidate(i); // sets state to Invalid
        free_slots_.push_back(i);
    }

    // Initialize valid mask on GPU if allocated
    if (valid_mask_) {
        std::vector<ggml_fp16_t> neg(ggml_nelements(valid_mask_), ggml_fp32_to_fp16(-INFINITY));
        ggml_backend_tensor_set(valid_mask_, neg.data(), 0, neg.size() * sizeof(ggml_fp16_t));
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

    if (!grow_gpu_pool_impl(slot + 1)) {
        std::cerr << "[NativeBlockPool] Error: Failed to grow GPU pool to " << (slot + 1) << " slots!" << std::endl;
    }

    return slot;
}

void NativeBlockPool::free_slot(int slot_id) {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    if (std::find(free_slots_.begin(), free_slots_.end(), slot_id) == free_slots_.end()) {
        state_table_.force_invalidate(slot_id);
        free_slots_.push_back(slot_id);
        host_seq_lens_[slot_id] = 0;
        free_slot_buffer(slot_id);
        if (slot_id >= 0 && slot_id < (int)slot_device_has_data_.size()) {
            slot_device_has_data_[slot_id] = false;
        }
    }
}

void NativeBlockPool::reset_slots() {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    free_slots_.clear();
    for (int i = n_slots_ - 1; i >= 0; --i) {
        state_table_.force_invalidate(i);
        free_slots_.push_back(i);
        host_seq_lens_[i] = 0;
        free_slot_buffer(i);
    }
    std::fill(slot_device_has_data_.begin(), slot_device_has_data_.end(), false);
    increment_pool_version();
}

int NativeBlockPool::get_free_slots_count() {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    return free_slots_.size();
}

void NativeBlockPool::zero_all_tensors() {
    std::fill(host_U_scale_.begin(), host_U_scale_.end(), ggml_fp32_to_fp16(0.0f));
    std::fill(host_seq_lens_.begin(), host_seq_lens_.end(), 0);
    std::fill(host_scales_.begin(), host_scales_.end(), ggml_fp32_to_fp16(0.0f));
    std::fill(host_anchor_positions_.begin(), host_anchor_positions_.end(), 0);

    for (int i = 0; i < n_slots_; ++i) {
        free_slot_buffer(i);
    }

    if (U_) {
        std::vector<char> zeros(ggml_nbytes(U_), 0);
        ggml_backend_tensor_set(U_, zeros.data(), 0, zeros.size());
    }
    if (U_f16_) {
        std::vector<char> zeros(ggml_nbytes(U_f16_), 0);
        ggml_backend_tensor_set(U_f16_, zeros.data(), 0, zeros.size());
    }
    if (U_scale_) {
        std::vector<char> zeros(ggml_nbytes(U_scale_), 0);
        ggml_backend_tensor_set(U_scale_, zeros.data(), 0, zeros.size());
    }
    if (U_row_scale_) {
        std::vector<char> zeros(ggml_nbytes(U_row_scale_), 0);
        ggml_backend_tensor_set(U_row_scale_, zeros.data(), 0, zeros.size());
    }
    if (VK_) {
        std::vector<char> zeros(ggml_nbytes(VK_), 0);
        ggml_backend_tensor_set(VK_, zeros.data(), 0, zeros.size());
    }
    if (VV_) {
        std::vector<char> zeros(ggml_nbytes(VV_), 0);
        ggml_backend_tensor_set(VV_, zeros.data(), 0, zeros.size());
    }
    if (anchors_K_) {
        std::vector<char> zeros(ggml_nbytes(anchors_K_), 0);
        ggml_backend_tensor_set(anchors_K_, zeros.data(), 0, zeros.size());
    }
    if (anchors_V_) {
        std::vector<char> zeros(ggml_nbytes(anchors_V_), 0);
        ggml_backend_tensor_set(anchors_V_, zeros.data(), 0, zeros.size());
    }
    if (VK_rot_) {
        std::vector<char> zeros(ggml_nbytes(VK_rot_), 0);
        ggml_backend_tensor_set(VK_rot_, zeros.data(), 0, zeros.size());
    }
    if (anchorK_rot_) {
        std::vector<char> zeros(ggml_nbytes(anchorK_rot_), 0);
        ggml_backend_tensor_set(anchorK_rot_, zeros.data(), 0, zeros.size());
    }
    if (valid_mask_) {
        std::vector<ggml_fp16_t> neg(ggml_nelements(valid_mask_), ggml_fp32_to_fp16(-INFINITY));
        ggml_backend_tensor_set(valid_mask_, neg.data(), 0, neg.size() * sizeof(ggml_fp16_t));
    }
    if (seq_lens_) {
        std::vector<char> zeros(ggml_nbytes(seq_lens_), 0);
        ggml_backend_tensor_set(seq_lens_, zeros.data(), 0, zeros.size());
    }
    if (scales_) {
        std::vector<char> zeros(ggml_nbytes(scales_), 0);
        ggml_backend_tensor_set(scales_, zeros.data(), 0, zeros.size());
    }
    if (desc_matrix_) {
        std::vector<char> zeros(ggml_nbytes(desc_matrix_), 0);
        ggml_backend_tensor_set(desc_matrix_, zeros.data(), 0, zeros.size());
    }
    if (anchor_positions_) {
        std::vector<char> zeros(ggml_nbytes(anchor_positions_), 0);
        ggml_backend_tensor_set(anchor_positions_, zeros.data(), 0, zeros.size());
    }
    if (token_positions_) {
        std::vector<char> zeros(ggml_nbytes(token_positions_), 0);
        ggml_backend_tensor_set(token_positions_, zeros.data(), 0, zeros.size());
    }
    std::fill(slot_device_has_data_.begin(), slot_device_has_data_.end(), false);
    increment_pool_version();
}

bool NativeBlockPool::grow_gpu_pool_impl(int min_slots_needed) {
    if (min_slots_needed <= n_allocated_slots_) {
        return true;
    }
    
    int new_allocated_slots = n_allocated_slots_;
    if (new_allocated_slots == 0) {
        new_allocated_slots = 16;
    }
    while (new_allocated_slots < min_slots_needed) {
        new_allocated_slots *= 2;
    }
    if (new_allocated_slots > n_slots_) {
        new_allocated_slots = n_slots_;
    }
    if (new_allocated_slots <= n_allocated_slots_) {
        return min_slots_needed <= n_allocated_slots_;
    }

    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
        std::cerr << "[NativeBlockPool] Growing GPU pool allocation: " 
                  << n_allocated_slots_ << " -> " << new_allocated_slots 
                  << " slots (needed: " << min_slots_needed << ")" << std::endl;
    }

    struct ggml_context * old_ctx = pool_ctx_;
    struct ggml_backend_buffer * old_buffer = pool_buffer_;
    bool skip_lowrank = false;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024, // 1MB metadata context
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,            // allocate actual data in backend buffer
    };

    struct ggml_context * new_ctx = ggml_init(params);
    if (!new_ctx) {
        std::cerr << "[NativeBlockPool] Error: Failed to initialize new pool ggml_context during grow!" << std::endl;
        return false;
    }

    struct ggml_tensor * new_U = nullptr;
    struct ggml_tensor * new_U_f16 = nullptr;
    struct ggml_tensor * new_U_scale = nullptr;
    struct ggml_tensor * new_U_row_scale = nullptr;
    struct ggml_tensor * new_VK = nullptr;
    struct ggml_tensor * new_VV = nullptr;
    struct ggml_tensor * new_VK_rot = nullptr;
    struct ggml_tensor * new_anchorK_rot = nullptr;
    struct ggml_tensor * new_valid_mask = nullptr;

    if (!skip_lowrank) {
        new_U           = ggml_new_tensor_3d(new_ctx, GGML_TYPE_I8,  rank_, S_max_, new_allocated_slots);
        new_U_f16       = ggml_new_tensor_3d(new_ctx, GGML_TYPE_F16, rank_, S_max_, new_allocated_slots);
        new_U_scale     = ggml_new_tensor_1d(new_ctx, GGML_TYPE_F16, new_allocated_slots);
        new_U_row_scale = ggml_new_tensor_2d(new_ctx, GGML_TYPE_F16, S_max_, new_allocated_slots);
        new_VK          = ggml_new_tensor_4d(new_ctx, kv_type_, head_dim_, kv_heads_, rank_, new_allocated_slots);
        new_VV          = ggml_new_tensor_4d(new_ctx, kv_type_, head_dim_, kv_heads_, rank_, new_allocated_slots);
    }
    
    struct ggml_tensor * new_anchors_K   = ggml_new_tensor_3d(new_ctx, GGML_TYPE_F16, head_dim_, kv_heads_, new_allocated_slots);
    struct ggml_tensor * new_anchors_V   = ggml_new_tensor_3d(new_ctx, GGML_TYPE_F16, head_dim_, kv_heads_, new_allocated_slots);

    if (native_attn_ && !skip_lowrank) {
        new_VK_rot      = ggml_new_tensor_4d(new_ctx, GGML_TYPE_F32, head_dim_, kv_heads_, rank_, new_allocated_slots);
        new_anchorK_rot = ggml_new_tensor_3d(new_ctx, GGML_TYPE_F32, head_dim_, kv_heads_, new_allocated_slots);
        new_valid_mask  = ggml_new_tensor_2d(new_ctx, GGML_TYPE_F16, S_max_, new_allocated_slots);
    }
    
    struct ggml_tensor * new_seq_lens    = ggml_new_tensor_1d(new_ctx, GGML_TYPE_I32, new_allocated_slots);
    struct ggml_tensor * new_scales      = ggml_new_tensor_1d(new_ctx, GGML_TYPE_F16, new_allocated_slots);
    struct ggml_tensor * new_desc_matrix     = ggml_new_tensor_2d(new_ctx, GGML_TYPE_F32, desc_dim_, new_allocated_slots);
    struct ggml_tensor * new_anchor_positions = ggml_new_tensor_1d(new_ctx, GGML_TYPE_I32, new_allocated_slots);
    struct ggml_tensor * new_token_positions  = ggml_new_tensor_2d(new_ctx, GGML_TYPE_I32, S_max_, new_allocated_slots);

    if (new_U)           ggml_set_name(new_U,           "pool.U");
    if (new_U_f16)       ggml_set_name(new_U_f16,       "pool.U_f16");
    if (new_U_scale)     ggml_set_name(new_U_scale,     "pool.U_scale");
    if (new_U_row_scale) ggml_set_name(new_U_row_scale, "pool.U_row_scale");
    if (new_VK)          ggml_set_name(new_VK,          "pool.V_K");
    if (new_VV)          ggml_set_name(new_VV,          "pool.V_V");
    ggml_set_name(new_anchors_K,   "pool.anchors_K");
    ggml_set_name(new_anchors_V,   "pool.anchors_V");
    ggml_set_name(new_seq_lens,      "pool.seq_lens");
    ggml_set_name(new_scales,        "pool.scales");
    ggml_set_name(new_desc_matrix,   "pool.desc_matrix");
    ggml_set_name(new_anchor_positions, "pool.anchor_positions");
    ggml_set_name(new_token_positions,  "pool.token_positions");
    if (new_VK_rot)      ggml_set_name(new_VK_rot,      "pool.V_K_rot");
    if (new_anchorK_rot) ggml_set_name(new_anchorK_rot, "pool.anchors_K_rot");
    if (new_valid_mask)  ggml_set_name(new_valid_mask,  "pool.valid_mask");

    struct ggml_backend_buffer * new_buffer = ggml_backend_alloc_ctx_tensors_from_buft(new_ctx, buft_);
    if (!new_buffer) {
        std::cerr << "[NativeBlockPool] Error: Failed to allocate new pool tensors in backend buffer during grow!" << std::endl;
        ggml_free(new_ctx);
        return false;
    }

    pool_ctx_ = new_ctx;
    pool_buffer_ = new_buffer;
    n_allocated_slots_ = new_allocated_slots;

    U_ = new_U;
    U_f16_ = new_U_f16;
    U_scale_ = new_U_scale;
    U_row_scale_ = new_U_row_scale;
    VK_ = new_VK;
    VV_ = new_VV;
    anchors_K_ = new_anchors_K;
    anchors_V_ = new_anchors_V;
    seq_lens_ = new_seq_lens;
    scales_ = new_scales;
    desc_matrix_ = new_desc_matrix;
    anchor_positions_ = new_anchor_positions;
    token_positions_ = new_token_positions;
    VK_rot_ = new_VK_rot;
    anchorK_rot_ = new_anchorK_rot;
    valid_mask_ = new_valid_mask;

    if (valid_mask_) {
        std::vector<ggml_fp16_t> neg(ggml_nelements(valid_mask_), ggml_fp32_to_fp16(-INFINITY));
        ggml_backend_tensor_set(valid_mask_, neg.data(), 0, neg.size() * sizeof(ggml_fp16_t));
    }

    for (int s = 0; s < n_slots_; ++s) {
        if (slot_device_has_data_[s]) {
            upload_slot_impl(s);
        }
    }

    if (old_buffer) {
        ggml_backend_buffer_free(old_buffer);
    }
    if (old_ctx) {
        ggml_free(old_ctx);
    }

    increment_pool_version();
    return true;
}

bool NativeBlockPool::grow_gpu_pool(int min_slots_needed) {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    return grow_gpu_pool_impl(min_slots_needed);
}

void NativeBlockPool::upload_slot_impl(int slot_id) {
    SlotHostBuffer* slot_buf = slot_host_buffers_[slot_id].get();
    
    if (U_ && slot_buf) {
        ggml_backend_tensor_set(U_, slot_buf->U.data(), slot_id * U_->nb[2], rank_ * S_max_ * sizeof(int8_t));
    }
    // f16 mirror of U (native ggml gather): cast this slot's int8 values to f16.
    if (U_f16_ && slot_buf) {
        const int8_t* src = slot_buf->U.data();
        std::vector<ggml_fp16_t> temp_U_f16(rank_ * S_max_);
        for (int i = 0; i < rank_ * S_max_; ++i) temp_U_f16[i] = ggml_fp32_to_fp16((float)src[i]);
        ggml_backend_tensor_set(U_f16_, temp_U_f16.data(), slot_id * U_f16_->nb[2], rank_ * S_max_ * sizeof(ggml_fp16_t));
    }
    if (U_scale_) {
        ggml_backend_tensor_set(U_scale_, host_U_scale_.data() + slot_id, slot_id * U_scale_->nb[0], sizeof(ggml_fp16_t));
    }
    if (U_row_scale_ && slot_buf) {
        ggml_backend_tensor_set(U_row_scale_, slot_buf->U_row_scale.data(), slot_id * U_row_scale_->nb[1], (size_t)S_max_ * sizeof(ggml_fp16_t));
    }
    if (VK_ && slot_buf) {
        if (ggml_is_quantized(kv_type_)) {
            int n = head_dim_ * kv_heads_ * rank_;
            std::vector<char> qbuf(ggml_row_size(kv_type_, n));
            quantize_row(kv_type_, slot_buf->VK.data(), qbuf.data(), n);
            ggml_backend_tensor_set(VK_, qbuf.data(), slot_id * VK_->nb[3], qbuf.size());
        } else {
            ggml_backend_tensor_set(VK_, slot_buf->VK.data(), slot_id * VK_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
        }
    }
    if (VV_ && slot_buf) {
        if (ggml_is_quantized(kv_type_)) {
            int n = head_dim_ * kv_heads_ * rank_;
            std::vector<char> qbuf(ggml_row_size(kv_type_, n));
            quantize_row(kv_type_, slot_buf->VV.data(), qbuf.data(), n);
            ggml_backend_tensor_set(VV_, qbuf.data(), slot_id * VV_->nb[3], qbuf.size());
        } else {
            ggml_backend_tensor_set(VV_, slot_buf->VV.data(), slot_id * VV_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
        }
    }
    if (slot_buf) {
        // anchors_K_ / anchors_V_ are ALWAYS F16 (see allocation above). Never quantize them:
        // every Metal shader reads these as `half*`. Upload raw f16 unconditionally.
        ggml_backend_tensor_set(anchors_K_, slot_buf->anchors_K.data(), slot_id * anchors_K_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(anchors_V_, slot_buf->anchors_V.data(), slot_id * anchors_V_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
    }
    ggml_backend_tensor_set(seq_lens_, host_seq_lens_.data() + slot_id, slot_id * seq_lens_->nb[0], sizeof(int32_t));
    ggml_backend_tensor_set(scales_, host_scales_.data() + slot_id, slot_id * scales_->nb[0], sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(anchor_positions_, host_anchor_positions_.data() + slot_id, slot_id * anchor_positions_->nb[0], sizeof(int32_t));
    if (desc_matrix_ && slot_buf) {
        ggml_backend_tensor_set(desc_matrix_, slot_buf->desc_matrix.data(), slot_id * desc_matrix_->nb[1], desc_dim_ * sizeof(float));
    }
    if (token_positions_ && slot_buf)
        ggml_backend_tensor_set(token_positions_, slot_buf->token_positions.data(), slot_id * token_positions_->nb[1], (size_t)S_max_ * sizeof(int32_t));

    // Native attn: precompute this slot's RoPE'd keys at its (fixed) anchor position so the
    // in-graph subgraph can gather f16 and dot with the in-graph query — no in-graph rope.
    if (native_attn_ && VK_rot_ && anchorK_rot_ && slot_buf) {
        const float pos = (float)host_anchor_positions_[slot_id];
        const int D = head_dim_;
        // anchors_K_rot: one head_dim vector per kv_head
        std::vector<float> tmp_anchorK_rot(kv_heads_ * D);
        for (int kv = 0; kv < kv_heads_; ++kv) {
            const size_t off = (size_t)kv * D;
            const ggml_fp16_t* src = slot_buf->anchors_K.data() + off;
            float* dst = tmp_anchorK_rot.data() + off;
            if (has_rope_) rope_rotate_vec(src, dst, D, pos, rope_freq_base_);
            else for (int d = 0; d < D; ++d) dst[d] = ggml_fp16_to_fp32(src[d]);
        }
        ggml_backend_tensor_set(anchorK_rot_, tmp_anchorK_rot.data(),
                                slot_id * anchorK_rot_->nb[2], (size_t)kv_heads_ * D * sizeof(float));
        // VK_rot: rotate VK vectors at anchor_pos and upload directly to GPU tensor.
        {
            // Compute rotation into a temporary buffer and upload to GPU.
            std::vector<float> tmp_rot(rank_ * kv_heads_ * D);
            for (int r = 0; r < rank_; ++r) {
                for (int kv = 0; kv < kv_heads_; ++kv) {
                    const size_t off = (size_t)r * kv_heads_ * D + (size_t)kv * D;
                    const ggml_fp16_t* src = slot_buf->VK.data() + off;
                    float* dst = tmp_rot.data() + off;
                    if (has_rope_) rope_rotate_vec(src, dst, D, pos, rope_freq_base_);
                    else for (int d = 0; d < D; ++d) dst[d] = ggml_fp16_to_fp32(src[d]);
                }
            }
            ggml_backend_tensor_set(VK_rot_, tmp_rot.data(),
                                    slot_id * VK_rot_->nb[3], (size_t)rank_ * kv_heads_ * D * sizeof(float));
        }
    }
    if (native_attn_ && valid_mask_) {
        const int S_max = S_max_;
        const int slen = host_seq_lens_[slot_id];
        std::vector<ggml_fp16_t> tmp_mask(S_max);
        for (int t = 0; t < S_max; ++t) tmp_mask[t] = ggml_fp32_to_fp16(t < slen ? 0.0f : -INFINITY);
        ggml_backend_tensor_set(valid_mask_, tmp_mask.data(), slot_id * valid_mask_->nb[1], (size_t)S_max * sizeof(ggml_fp16_t));
    }
}

void NativeBlockPool::upload_slot(int slot_id) {
    if (slot_id < 0 || slot_id >= n_slots_) return;
    
    std::lock_guard<std::mutex> lock(slot_mutex_);
    if (!grow_gpu_pool_impl(slot_id + 1)) {
        std::cerr << "[NativeBlockPool] Error: Failed to grow GPU pool to " << (slot_id + 1) << " slots!" << std::endl;
        return;
    }
    
    upload_slot_impl(slot_id);
    slot_device_has_data_[slot_id] = (host_seq_lens_[slot_id] > 0);
    increment_pool_version();
}

void NativeBlockPool::download_slot(int slot_id) {
    if (slot_id < 0 || slot_id >= n_slots_) return;
    
    std::lock_guard<std::mutex> lock(slot_mutex_);
    SlotHostBuffer* slot_buf = ensure_slot_buffer(slot_id);
    if (!slot_buf) return;
    
    if (U_ && slot_id < n_allocated_slots_) {
        ggml_backend_tensor_get(U_, slot_buf->U.data(), slot_id * U_->nb[2], rank_ * S_max_ * sizeof(int8_t));
    }
    if (U_scale_ && slot_id < n_allocated_slots_) {
        ggml_backend_tensor_get(U_scale_, host_U_scale_.data() + slot_id, slot_id * U_scale_->nb[0], sizeof(ggml_fp16_t));
    }
    if (U_row_scale_ && slot_id < n_allocated_slots_) {
        ggml_backend_tensor_get(U_row_scale_, slot_buf->U_row_scale.data(), slot_id * U_row_scale_->nb[1], (size_t)S_max_ * sizeof(ggml_fp16_t));
    }
    if (VK_ && slot_id < n_allocated_slots_) {
        if (kv_type_ == GGML_TYPE_Q8_0) {
            std::vector<char> qbuf(head_dim_ * kv_heads_ * rank_ / 32 * 34);
            ggml_backend_tensor_get(VK_, qbuf.data(), slot_id * VK_->nb[3], qbuf.size());
            dequantize_q8_0(qbuf.data(), slot_buf->VK.data(), head_dim_ * kv_heads_ * rank_);
        } else {
            ggml_backend_tensor_get(VK_, slot_buf->VK.data(), slot_id * VK_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
        }
    }
    if (VV_ && slot_id < n_allocated_slots_) {
        if (kv_type_ == GGML_TYPE_Q8_0) {
            std::vector<char> qbuf(head_dim_ * kv_heads_ * rank_ / 32 * 34);
            ggml_backend_tensor_get(VV_, qbuf.data(), slot_id * VV_->nb[3], qbuf.size());
            dequantize_q8_0(qbuf.data(), slot_buf->VV.data(), head_dim_ * kv_heads_ * rank_);
        } else {
            ggml_backend_tensor_get(VV_, slot_buf->VV.data(), slot_id * VV_->nb[3], head_dim_ * kv_heads_ * rank_ * sizeof(ggml_fp16_t));
        }
    }
    if (anchors_K_ && slot_id < n_allocated_slots_) {
        if (kv_type_ == GGML_TYPE_Q8_0) {
            std::vector<char> qbuf(kv_heads_ * head_dim_ / 32 * 34);
            ggml_backend_tensor_get(anchors_K_, qbuf.data(), slot_id * anchors_K_->nb[2], qbuf.size());
            dequantize_q8_0(qbuf.data(), slot_buf->anchors_K.data(), kv_heads_ * head_dim_);
        } else {
            ggml_backend_tensor_get(anchors_K_, slot_buf->anchors_K.data(), slot_id * anchors_K_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
        }
    }
    if (anchors_V_ && slot_id < n_allocated_slots_) {
        if (kv_type_ == GGML_TYPE_Q8_0) {
            std::vector<char> qbuf(kv_heads_ * head_dim_ / 32 * 34);
            ggml_backend_tensor_get(anchors_V_, qbuf.data(), slot_id * anchors_V_->nb[2], qbuf.size());
            dequantize_q8_0(qbuf.data(), slot_buf->anchors_V.data(), kv_heads_ * head_dim_);
        } else {
            ggml_backend_tensor_get(anchors_V_, slot_buf->anchors_V.data(), slot_id * anchors_V_->nb[2], head_dim_ * kv_heads_ * sizeof(ggml_fp16_t));
        }
    }
    if (seq_lens_ && slot_id < n_allocated_slots_) {
        ggml_backend_tensor_get(seq_lens_, host_seq_lens_.data() + slot_id, slot_id * seq_lens_->nb[0], sizeof(int32_t));
    }
    if (scales_ && slot_id < n_allocated_slots_) {
        ggml_backend_tensor_get(scales_, host_scales_.data() + slot_id, slot_id * scales_->nb[0], sizeof(ggml_fp16_t));
    }
    if (anchor_positions_ && slot_id < n_allocated_slots_) {
        ggml_backend_tensor_get(anchor_positions_, host_anchor_positions_.data() + slot_id, slot_id * anchor_positions_->nb[0], sizeof(int32_t));
    }
}

SlotHostBuffer* NativeBlockPool::ensure_slot_buffer(int slot_id) {
    if (slot_id < 0 || slot_id >= n_slots_) return nullptr;
    if (!slot_host_buffers_[slot_id]) {
        auto buf = std::make_unique<SlotHostBuffer>();
        if (U_) buf->U.resize(rank_ * S_max_, 0);
        if (U_row_scale_) buf->U_row_scale.resize(S_max_, ggml_fp32_to_fp16(0.0f));
        if (VK_) buf->VK.resize(rank_ * kv_heads_ * head_dim_, ggml_fp32_to_fp16(0.0f));
        if (VV_) buf->VV.resize(rank_ * kv_heads_ * head_dim_, ggml_fp32_to_fp16(0.0f));
        buf->anchors_K.resize(kv_heads_ * head_dim_, ggml_fp32_to_fp16(0.0f));
        buf->anchors_V.resize(kv_heads_ * head_dim_, ggml_fp32_to_fp16(0.0f));
        buf->token_positions.resize(S_max_, 0);
        buf->desc_matrix.resize(desc_dim_, 0.0f);
        
        buf->res_K_pos.resize(MAX_RESIDUAL, -1);
        buf->res_V_pos.resize(MAX_RESIDUAL, -1);
        buf->res_K_val.resize(MAX_RESIDUAL * kv_heads_ * head_dim_, ggml_fp32_to_fp16(0.0f));
        buf->res_V_val.resize(MAX_RESIDUAL * kv_heads_ * head_dim_, ggml_fp32_to_fp16(0.0f));
        
        slot_host_buffers_[slot_id] = std::move(buf);
    }
    return slot_host_buffers_[slot_id].get();
}

void NativeBlockPool::free_slot_buffer(int slot_id) {
    if (slot_id >= 0 && slot_id < n_slots_) {
        slot_host_buffers_[slot_id].reset();
    }
}

const int8_t* NativeBlockPool::get_host_U(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->U.data();
}
int8_t* NativeBlockPool::get_host_U(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->U.data() : nullptr;
}
const ggml_fp16_t* NativeBlockPool::get_host_U_row_scale(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->U_row_scale.data();
}
ggml_fp16_t* NativeBlockPool::get_host_U_row_scale(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->U_row_scale.data() : nullptr;
}
const ggml_fp16_t* NativeBlockPool::get_host_VK(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->VK.data();
}
ggml_fp16_t* NativeBlockPool::get_host_VK(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->VK.data() : nullptr;
}
const ggml_fp16_t* NativeBlockPool::get_host_VV(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->VV.data();
}
ggml_fp16_t* NativeBlockPool::get_host_VV(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->VV.data() : nullptr;
}
const ggml_fp16_t* NativeBlockPool::get_host_anchors_K(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->anchors_K.data();
}
ggml_fp16_t* NativeBlockPool::get_host_anchors_K(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->anchors_K.data() : nullptr;
}
const ggml_fp16_t* NativeBlockPool::get_host_anchors_V(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->anchors_V.data();
}
ggml_fp16_t* NativeBlockPool::get_host_anchors_V(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->anchors_V.data() : nullptr;
}
const int32_t* NativeBlockPool::get_host_token_positions(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->token_positions.data();
}
int32_t* NativeBlockPool::get_host_token_positions(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->token_positions.data() : nullptr;
}
const float* NativeBlockPool::get_host_desc_matrix(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->desc_matrix.data();
}
float* NativeBlockPool::get_host_desc_matrix(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->desc_matrix.data() : nullptr;
}

int32_t* NativeBlockPool::get_host_res_K_pos(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->res_K_pos.data() : nullptr;
}
int32_t* NativeBlockPool::get_host_res_V_pos(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->res_V_pos.data() : nullptr;
}
ggml_fp16_t* NativeBlockPool::get_host_res_K_val(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->res_K_val.data() : nullptr;
}
ggml_fp16_t* NativeBlockPool::get_host_res_V_val(int slot_id) {
    auto* buf = ensure_slot_buffer(slot_id);
    return buf ? buf->res_V_val.data() : nullptr;
}
const int32_t* NativeBlockPool::get_host_res_K_pos(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->res_K_pos.data();
}
const int32_t* NativeBlockPool::get_host_res_V_pos(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->res_V_pos.data();
}
const ggml_fp16_t* NativeBlockPool::get_host_res_K_val(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->res_K_val.data();
}
const ggml_fp16_t* NativeBlockPool::get_host_res_V_val(int slot_id) const {
    if (slot_id < 0 || slot_id >= n_slots_ || !slot_host_buffers_[slot_id]) return nullptr;
    return slot_host_buffers_[slot_id]->res_V_val.data();
}

} // namespace diffkv
