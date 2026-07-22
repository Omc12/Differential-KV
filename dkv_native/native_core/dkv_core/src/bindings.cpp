// src/bindings.cpp
// C API bridge for dkv_native.
//
// In ACTIVE_RUNTIME, this file was a PyBind11 module (dkv_core.cpython-*.so)
// that exposed C++ internals to Python. In dkv_native (llama.cpp-based),
// there is no Python layer — everything is linked directly into the binary.
//
// This file instead provides:
//   1. A stable C-linkage API for any external callers (future use)
//   2. Runtime capability detection functions
//   3. Version information
//
// Original: ACTIVE_RUNTIME/native_core/dkv_core/src/bindings.cpp
// Translation: PyBind11 module → C API (extern "C" linkage)

#include "native_core/dkv_core/include/block_state.hpp"
#include "native_core/dkv_core/include/srl_router.hpp"
#include "native_core/dkv_core/include/decode_attention.hpp"
#include "native_core/config.hpp"
#include "native_core/mac_utils.hpp"

#ifdef __APPLE__
#include "native_core/dkv_core/include/compressor_cpu.hpp"
#include "native_core/dkv_core/include/metal_runtime.hpp"
#endif

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>

// ── Version ───────────────────────────────────────────────────────────────────

extern "C" {

const char* dkv_version() {
    return "dkv_native-1.0.0-llama.cpp";
}

// ── Capability detection ──────────────────────────────────────────────────────

int dkv_has_metal() {
#ifdef __APPLE__
    return dkv::has_metal() ? 1 : 0;
#else
    return 0;
#endif
}

int dkv_has_cuda() {
    // dkv_native is macOS-first; CUDA not supported in this build.
    return 0;
}

const char* dkv_best_device() {
    return dkv::get_best_device().c_str();
}

// ── BlockStateTable C API ─────────────────────────────────────────────────────

// Thin wrappers around DKVBlockStateTable for external C callers.
// (Internal C++ code uses the class directly.)

void* dkv_state_table_create() {
    return new dkv::DKVBlockStateTable();
}

void dkv_state_table_destroy(void* handle) {
    delete reinterpret_cast<dkv::DKVBlockStateTable*>(handle);
}

int dkv_state_table_transition(void* handle,
                                   uint32_t block_id,
                                   int expected_state,
                                   int desired_state) {
    auto* tbl = reinterpret_cast<dkv::DKVBlockStateTable*>(handle);
    bool ok = tbl->transition(
        block_id,
        static_cast<dkv::BlockState>(expected_state),
        static_cast<dkv::BlockState>(desired_state));
    return ok ? 1 : 0;
}

void dkv_state_table_invalidate(void* handle, uint32_t block_id) {
    auto* tbl = reinterpret_cast<dkv::DKVBlockStateTable*>(handle);
    tbl->force_invalidate(block_id);
}

int dkv_state_table_get(void* handle, uint32_t block_id) {
    auto* tbl = reinterpret_cast<dkv::DKVBlockStateTable*>(handle);
    return static_cast<int>(tbl->get(block_id));
}

// ── SRL Router C API ──────────────────────────────────────────────────────────

// compute_query_desc: Q [H_q * D] → desc [desc_dim]
void dkv_compute_query_desc(
    const float* Q, int H_q, int D,
    const float* W_proj, int desc_dim,
    float* out_desc
) {
    dkv::compute_query_desc(Q, H_q, D, W_proj, desc_dim, out_desc);
}

// semantic_search_topk: q_desc [desc_dim] × desc_matrix [N * desc_dim] → top-k indices
void dkv_semantic_search_topk(
    const float* q_desc,
    const float* desc_matrix,
    int N, int desc_dim, int k,
    int32_t* out_indices,
    float*   out_scores
) {
    dkv::semantic_search_topk(q_desc, desc_matrix, N, desc_dim, k,
                                  out_indices, out_scores);
}

// anchor_screen: rerank M candidates down to k_keep
void dkv_anchor_screen(
    const float*   Q,
    int            H_q, int D,
    const float*   anchors_K,
    int            N_pool, int kv_heads,
    const int32_t* candidate_slots, int M,
    float          scale, int k_keep,
    int32_t*       out_slots
) {
    dkv::anchor_screen(Q, H_q, D, anchors_K, N_pool, kv_heads,
                           candidate_slots, M, scale, k_keep, out_slots);
}

// ── Decode Attention C API ────────────────────────────────────────────────────

void dkv_decode_attention(
    const float*    Q,
    const int8_t*   U_pool,
    const float*    U_scale_pool,
    const uint16_t* VK_pool,
    const uint16_t* VV_pool,
    const uint16_t* anchors_K,
    const uint16_t* anchors_V,
    const int32_t*  seq_lens,
    const uint16_t* scales,
    const float*    cos_anc,
    const float*    sin_anc,
    const int32_t*  slot_indices,
    int K_active,
    int N_pool, int S_max, int R,
    int H_q, int kv_heads, int D,
    float scale,
    float* out
) {
    dkv::decode_attention(Q, U_pool, U_scale_pool, VK_pool, VV_pool,
                              anchors_K, anchors_V, seq_lens, scales,
                              cos_anc, sin_anc, slot_indices,
                              K_active, N_pool, S_max, R, H_q, kv_heads, D,
                              scale, out);
}

// ── Memory / diagnostics ──────────────────────────────────────────────────────

void dkv_get_memory_mb(double* out_allocated, double* out_reserved, double* out_rss) {
    auto info = dkv::get_memory_info();
    if (out_allocated) *out_allocated = info.allocated_mb;
    if (out_reserved)  *out_reserved  = info.reserved_mb;
    if (out_rss)       *out_rss       = info.rss_mb;
}

} // extern "C"
