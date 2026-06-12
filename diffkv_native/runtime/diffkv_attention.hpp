#pragma once

#include "ggml.h"
#include "runtime/native_block_pool.hpp"

#include <cstdlib>
#include <new>
#include <vector>

namespace diffkv {

template <typename T>
struct PageAlignedAllocator {
    using value_type = T;
    PageAlignedAllocator() noexcept {}
    template <typename U> PageAlignedAllocator(const PageAlignedAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(n * sizeof(T), 4096);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, 4096, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) noexcept {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

template <typename T, typename U>
bool operator==(const PageAlignedAllocator<T>&, const PageAlignedAllocator<U>&) { return true; }

template <typename T, typename U>
bool operator!=(const PageAlignedAllocator<T>&, const PageAlignedAllocator<U>&) { return false; }

typedef std::vector<float, PageAlignedAllocator<float>> AlignedFloatVector;
typedef std::vector<int32_t, PageAlignedAllocator<int32_t>> AlignedInt32Vector;

struct CustomAttnUserData {
    NativeBlockPool * kv_engine;
    std::string session_id;
    int layer_idx = -1;
    struct ggml_tensor * slot_indices;
    int n_q_heads;
    int n_kv_heads;
    int rank;
    int S_max;
    int K;
    int D;
    float scale;
    bool has_rope;
    float rope_freq_base;
    float * active_k_dense = nullptr;            // mutable — callback appends current token
    float * active_v_dense = nullptr;            // mutable — callback appends current token
    int32_t * active_positions_dense = nullptr;  // mutable — callback writes current_pos
    int active_block_tokens;
    int active_slot;
    bool ignore_c = false;
    int current_pos = 0;  // sequence position of the current decode token (set in main.cpp)

    // Cached Metal buffers to avoid per-token allocations
    void * mtl_dense_k = nullptr;
    void * mtl_dense_v = nullptr;
    void * mtl_dense_pos = nullptr;
    void * mtl_slot_indices = nullptr;
    void * mtl_output_buf = nullptr;
    void * mtl_lse_buf = nullptr;

    // Cached Metal buffers for static pool and query tensors
    void * mtl_q_buf = nullptr;
    void * mtl_u_pool = nullptr;
    void * mtl_u_scale = nullptr;
    void * mtl_vk_pool = nullptr;
    void * mtl_vv_pool = nullptr;
    void * mtl_anchors_k = nullptr;
    void * mtl_anchors_v = nullptr;
    void * mtl_seq_lens = nullptr;
    void * mtl_scales = nullptr;
    void * mtl_anc_pos = nullptr;

    CustomAttnUserData();
    ~CustomAttnUserData();

    // Prevent copy constructor and copy assignment to avoid double-free of cached buffers
    CustomAttnUserData(const CustomAttnUserData&) = delete;
    CustomAttnUserData& operator=(const CustomAttnUserData&) = delete;

    // Support move semantics
    CustomAttnUserData(CustomAttnUserData&& other) noexcept;
    CustomAttnUserData& operator=(CustomAttnUserData&& other) noexcept;
};

// Metal-accelerated decode attention.
// Handles both sparse compressed blocks (Project-Then-Attend) and dense window
// tokens (buffers 22-25) in a single GPU dispatch.
// Outputs fully combined sparse + dense result into dst.
// Parameters:
//   dense_K, dense_V : CPU float buffers [T_dense × n_kv × D]
//   dense_pos        : CPU int32 buffer  [T_dense]  (actual sequence positions)
//   T_dense          : number of dense window tokens to process
void execute_metal_attention(
    struct ggml_tensor * dst,
    const struct ggml_tensor * Q,
    struct ggml_tensor * slot_indices,
    CustomAttnUserData * data,
    float* lse_out,
    const float*   dense_K,
    const float*   dense_V,
    const int32_t* dense_pos,
    int            T_dense
);

// GGML callback for custom attention operator
void custom_attention_op_callback(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    const struct ggml_tensor * b,
    const struct ggml_tensor * c,
    int ith,
    int nth,
    void * userdata
);

} // namespace diffkv
