// dkv_core/src/srl_router.cpp
// C++ ATen implementation of SRL routing hot path.
//
// Replaces Python hot path in:
//   native_core/srl/chunk_descriptor.py  — compute_query_descriptor()
//   native_core/srl/semantic_index.py    — SemanticIndex.search()
//   native_core/srl/query_router.py      — two_level_gate()
//
// All operations are expressed via the ATen C++ API (torch::Tensor, torch::mm,
// torch::topk, etc.) so they dispatch to the same backend kernels as PyTorch
// (MPS on Mac, CUDA on GPU, CPU otherwise) without any Python GIL involvement.

#include "srl_router.hpp"
#include <torch/extension.h>
#include <algorithm>
#include <cmath>

namespace dkv {

// ── compute_query_desc ────────────────────────────────────────────────────────
torch::Tensor compute_query_desc(
    const torch::Tensor& Q,      // [H, D]
    const torch::Tensor& W_proj  // [desc_dim, D]
) {
    // 1. Mean-pool over all query heads: [D]
    //    Equivalent to Python: Q.float().mean(dim=0)
    auto q_mean = Q.to(torch::kFloat32).mean(/*dim=*/0);  // [D]

    // 2. Project into descriptor space: [desc_dim]
    //    W_proj is row-normalized at construction; q_mean must be float32.
    auto desc = torch::mv(W_proj.to(torch::kFloat32), q_mean);  // [desc_dim]

    // 3. L2-normalize to unit sphere (matches block descriptors)
    auto norm = desc.norm().clamp_min(1e-8f);
    return (desc / norm).to(torch::kFloat32);  // [desc_dim] float32
}

// ── semantic_search_topk ──────────────────────────────────────────────────────
torch::Tensor semantic_search_topk(
    const torch::Tensor& q_desc,      // [desc_dim]
    const torch::Tensor& desc_matrix, // [N, desc_dim]
    int k
) {
    const int N = static_cast<int>(desc_matrix.size(0));
    if (N == 0) {
        return torch::zeros({0}, torch::TensorOptions()
                                      .dtype(torch::kInt64)
                                      .device(q_desc.device()));
    }

    // Dot product over all descriptors: desc_matrix [N, desc_dim] × q_desc [desc_dim]
    // scores [N] — both should be float32, normalized
    auto scores = torch::mv(
        desc_matrix.to(torch::kFloat32),
        q_desc.to(torch::kFloat32)
    );  // [N]

    int k_clamped = std::min(k, N);

    // topk — returns (values, indices); we need indices (row IDs in desc_matrix)
    // largest=true, sorted=true
    auto [top_vals, top_idx] = scores.topk(k_clamped, /*dim=*/0,
                                            /*largest=*/true, /*sorted=*/true);
    (void)top_vals;
    return top_idx;  // [k_clamped] int64
}

// ── anchor_screen ─────────────────────────────────────────────────────────────
torch::Tensor anchor_screen(
    const torch::Tensor& Q,               // [H, D]
    const torch::Tensor& anchors_K,       // [N_pool, kv_heads, D]
    const torch::Tensor& candidate_slots, // [M] int32 or int64
    float scale,
    int k_keep
) {
    const int M = static_cast<int>(candidate_slots.size(0));

    // If already within budget — nothing to screen
    if (M <= k_keep) {
        return candidate_slots.to(torch::kInt32);
    }

    // ── Step 1: Mean-pool query over all heads → [D] ─────────────────────────
    // GQA: averaging over all query heads is a good approximation of attention
    // head diversity for screening purposes (same as Python two_level_gate).
    auto q_mean = Q.to(torch::kFloat32).mean(/*dim=*/0);  // [D]

    // ── Step 2: Gather anchor keys for candidate slots ────────────────────────
    // candidate_slots must be int64 for index_select
    auto slots_long = candidate_slots.to(torch::kInt64);

    // anchors_K [N_pool, kv_heads, D] → gather → [M, kv_heads, D]
    auto anc_K = anchors_K.index_select(0, slots_long).to(torch::kFloat32);

    // Mean over kv_heads: [M, D]
    auto anc_flat = anc_K.mean(/*dim=*/1);  // [M, D]

    // ── Step 3: Dot product with mean query → anchor score [M] ───────────────
    // On MPS, mm requires contiguous 2D tensors (satisfied by mean output)
    auto scores = torch::mv(anc_flat, q_mean) * scale;  // [M]

    // ── Step 3b: Edge-aware routing propagation ──────────────────────────────
    static const bool er_on = []() {
        const char* e = std::getenv("DKV_EDGE_ROUTING");
        return !(e && (std::string(e) == "0" || std::string(e) == "OFF" || std::string(e) == "false"));
    }();
    if (er_on && M >= 3) {
        static const float er_beta = []() {
            const char* e = std::getenv("DKV_EDGE_ROUTE_BETA");
            return e ? std::stof(e) : 0.25f;
        }();
        static const int er_maxnb = []() {
            const char* e = std::getenv("DKV_EDGE_ROUTE_MAXNB");
            return e ? std::stoi(e) : 512;
        }();

        if (M <= er_maxnb) {
            // Flatten anchor keys: [M, kv_heads, D] -> [M, kv_heads * D]
            auto akf = anc_K.reshape({M, -1});
            // Normalize row-wise to get unit vector signatures
            auto akn = akf / (akf.norm(2, -1, true) + 1e-6f);
            // Cosine similarity: [M, M]
            auto A = torch::mm(akn, akn.t());
            // Remove self-loops and keep positive connections only
            A = torch::clamp(A - torch::eye(M, A.options()), 0.0);
            // Row-normalize the adjacency matrix
            A = A / (A.sum(-1, true) + 1e-6f);
            // Diffuse: relevance <- relevance + beta * (A @ relevance)
            auto prop = torch::mv(A, scores);
            scores = scores + er_beta * prop;
        }
    }

    // ── Step 4: Top-k by anchor score ────────────────────────────────────────
    int k_clamped = std::min(k_keep, M);
    auto [top_vals, top_idx] = scores.topk(k_clamped, 0, true, true);
    (void)top_vals;

    // Return the corresponding slot IDs (int32 to match pool.block_indices dtype)
    return candidate_slots.index_select(0, top_idx).to(torch::kInt32);
}

} // namespace dkv
