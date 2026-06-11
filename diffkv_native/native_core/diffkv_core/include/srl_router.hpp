// diffkv_native/include/srl_router.hpp
// C++ implementation of the SRL (Semantic Relevance Layer) routing hot path.
//
// These functions replace the Python hot-path calls in:
//   native_core/srl/query_router.py  — two_level_gate(), compute_query_descriptor()
//   native_core/srl/semantic_index.py — SemanticIndex.search()
//   native_core/srl/chunk_descriptor.py — compute_query_descriptor()
//
// All functions operate on raw float* arrays using Accelerate BLAS (cblas_sgemv,
// cblas_sgemm). They are safe to call from C++ without holding the Python GIL.
//
// Thread safety:
//   - All functions are stateless (no shared mutable state).
//   - Safe to call from multiple threads as long as the arrays are not mutated
//     concurrently. On Apple Silicon, arrays are unified-memory — reads are thread-safe.

#pragma once
#include <cstdint>

namespace diffkv {

// ── compute_query_desc ────────────────────────────────────────────────────────
// Computes a compact [desc_dim] descriptor for a query tensor.
//
// Replaces: compute_query_descriptor() in native_core/srl/chunk_descriptor.py
//
// Args:
//   Q         : [H_q, D]        query matrix (all heads, float32), row-major
//   H_q       : number of query heads
//   D         : head dimension
//   W_proj    : [desc_dim, D]   random projection matrix (float32, row-normalized)
//   desc_dim  : descriptor dimension
//   out_desc  : [desc_dim]      output descriptor, L2-normalized (caller-allocated)
//
// Algorithm:
//   1. Mean-pool over all query heads: q_mean = sum(Q[h,:], h) / H_q   → [D]
//   2. Project into descriptor space:  desc = W_proj @ q_mean            → [desc_dim]
//   3. L2-normalize to unit sphere (matches block descriptors)
//
// Cost: 1 mean-reduction + 1 cblas_sgemv (D→desc_dim) — sub-0.05ms on M3 Pro.
void compute_query_desc(
    const float* Q,        // [H_q, D]
    int          H_q,
    int          D,
    const float* W_proj,   // [desc_dim, D]
    int          desc_dim,
    float*       out_desc  // [desc_dim]
);

// ── semantic_search_topk ──────────────────────────────────────────────────────
// Dot-product search over descriptor matrix: returns top-k row indices.
//
// Replaces: SemanticIndex.search() in native_core/srl/semantic_index.py
//
// Args:
//   q_desc      : [desc_dim]    float32, L2-normalized query descriptor
//   desc_matrix : [N, desc_dim] float32 — pool.desc_matrix for all slots
//   N           : number of rows in desc_matrix (pool size)
//   desc_dim    : descriptor dimension
//   k           : number of top slots to return (clamped to N if k > N)
//   out_indices : [min(k,N)] int32 — row indices into desc_matrix (pool slot IDs)
//   out_scores  : [min(k,N)] float32 — corresponding dot-product scores (desc order)
//
// Cost: 1 cblas_sgemv [1, desc_dim] × [desc_dim, N] + partial_sort(k) — ~0.1ms for N=1000.
void semantic_search_topk(
    const float*   q_desc,
    const float*   desc_matrix,
    int            N,
    int            desc_dim,
    int            k,
    int32_t*       out_indices,
    float*         out_scores
);

// ── anchor_screen ─────────────────────────────────────────────────────────────
// Level-1 anchor reranking — cheap dot-product screening over candidate slots.
//
// Replaces: two_level_gate() in native_core/srl/query_router.py
//
// Loads anchors_K[candidate_slots] and reranks by mean-query dot product.
// This is the fast "Level 1" gate that filters K_semantic + lexical + graph
// candidates down to the final K slots before the expensive attention kernel.
//
// Args:
//   Q               : [H_q, D]            float32 — all query heads
//   H_q             : number of query heads
//   D               : head dimension
//   anchors_K       : [N_pool, kv_heads, D] float32 — pool.anchors_K (full pool)
//   N_pool          : total number of pool slots
//   kv_heads        : number of KV heads
//   candidate_slots : [M] int32 — candidate pool slot IDs (already merged from
//                     semantic + lexical + graph + recency paths)
//   M               : number of candidates
//   scale           : attention scale factor (1/sqrt(head_dim))
//   k_keep          : number of slots to keep (clamped to M if k_keep >= M)
//   out_slots       : [min(k_keep, M)] int32 — filtered slot IDs, ordered by
//                     descending anchor score (caller-allocated)
//
// Cost: gather [M, kv_heads, D] + mean [M, D] + dot [M] + partial_sort(k_keep).
//       For M=150, D=64: ~0.1ms on M3 Pro.
void anchor_screen(
    const float*   Q,
    int            H_q,
    int            D,
    const float*   anchors_K,
    int            N_pool,
    int            kv_heads,
    const int32_t* candidate_slots,
    int            M,
    float          scale,
    int            k_keep,
    int32_t*       out_slots
);

} // namespace diffkv
