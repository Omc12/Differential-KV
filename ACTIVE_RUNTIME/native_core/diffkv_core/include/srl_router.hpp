// diffkv_core/include/srl_router.hpp
// C++ implementation of the SRL (Semantic Relevance Layer) routing hot path.
//
// These functions replace the Python hot-path calls in:
//   native_core/srl/query_router.py  — two_level_gate(), compute_query_descriptor()
//   native_core/srl/semantic_index.py — SemanticIndex.search()
//   native_core/srl/chunk_descriptor.py — compute_query_descriptor()
//
// All functions operate on ATen tensors and use the ATen C++ API internally.
// They are safe to call from C++ without holding the Python GIL once the input
// tensors have been detached/contiguous-ized by the Python callsite.
//
// Thread safety:
//   - All functions are stateless (no shared mutable state).
//   - Safe to call from multiple threads as long as the tensors are not mutated
//     concurrently. On MPS, tensors are unified-memory — reads are thread-safe.

#pragma once
#include <torch/extension.h>

namespace diffkv {

// ── compute_query_desc ────────────────────────────────────────────────────────
// Computes a compact [desc_dim] descriptor for a query tensor.
//
// Replaces: compute_query_descriptor() in native_core/srl/chunk_descriptor.py
//
// Args:
//   Q       : [H, D]           query tensor (all heads, float16 or float32)
//   W_proj  : [desc_dim, D]    random projection matrix (float32, row-normalized)
//
// Returns:
//   [desc_dim] float32 descriptor, L2-normalized (matches block descriptors)
//
// Cost: 1 mean-reduction + 1 matmul (D→desc_dim) — sub-0.05ms on M3 Pro.
torch::Tensor compute_query_desc(
    const torch::Tensor& Q,
    const torch::Tensor& W_proj
);

// ── semantic_search_topk ──────────────────────────────────────────────────────
// Dot-product search over descriptor matrix: returns top-k row indices.
//
// Replaces: SemanticIndex.search() in native_core/srl/semantic_index.py
//
// Args:
//   q_desc       : [desc_dim] float32, normalized
//   desc_matrix  : [N, desc_dim] float32 — pool.desc_matrix for all slots
//   k            : number of top slots to return
//
// Returns:
//   [min(k,N)] int64 tensor — row indices into desc_matrix (= pool slot IDs)
//
// Cost: 1 matmul [1, desc_dim] × [desc_dim, N] + topk(k) — ~0.1ms for N=1000.
torch::Tensor semantic_search_topk(
    const torch::Tensor& q_desc,
    const torch::Tensor& desc_matrix,
    int k
);

// ── anchor_screen ─────────────────────────────────────────────────────────────
// Level-1 anchor reranking — cheap dot-product screening over candidate slots.
//
// Replaces: two_level_gate() in native_core/srl/query_router.py
//
// Loads pool.anchors_K[candidate_slots] and reranks by mean-query dot product.
// This is the fast "Level 1" gate that filters K_semantic + lexical + graph
// candidates down to the final K slots before the expensive attention kernel.
//
// Args:
//   Q               : [H, D]  float16 or float32 — all query heads
//   anchors_K       : [N_pool, kv_heads, D]  — pool.anchors_K (full pool)
//   candidate_slots : [M] int32 or int64 — candidate pool slot IDs (already
//                     merged from semantic + lexical + graph + recency paths)
//   scale           : attention scale factor (1/sqrt(head_dim))
//   k_keep          : number of slots to return
//
// Returns:
//   [min(k_keep, M)] int32 tensor — filtered slot IDs, ordered by anchor score
//
// Cost: gather [M, kv_heads, D] + mean [M, D] + dot [M] + topk(k_keep).
//       For M=150, D=64: ~0.1ms on M3 Pro.
torch::Tensor anchor_screen(
    const torch::Tensor& Q,
    const torch::Tensor& anchors_K,
    const torch::Tensor& candidate_slots,
    float scale,
    int k_keep
);

} // namespace diffkv
