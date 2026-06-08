// diffkv_core/src/decode_attention.cpp
// Fused Project-Then-Attend decode attention kernel in C++ (ATen API).
//
// Replaces:
//   fused_decode_mps()   in native_core/sparse_decode/triton_fused_decode.py (MPS path)
//   The inner math of native_triton_sparse_attn_decode() for the compressed-history path
//
// Algorithm — Fully Vectorized Project-Then-Attend:
//   1. Gather active blocks from pool: slot_indices → [K, S, R/kv/D] tensors
//   2. GQA-reduce Q: [H_q, D] → [H_kv, D]
//   3. Anchor scores:     s_anc = q_kv @ anc_K_gathered.T * scale   [H_kv, K]
//   4. Q projection:      q_proj = q_kv @ VK_gathered.T_per_head    [H_kv, K, R]
//   5. Delta scores:      s_delta = (q_proj[h] @ U_b.T) * scale     [H_kv, K, S]
//   6. Build score matrix: [H_kv, K*S+K] with masking for padding
//   7. Softmax over all scores → attention weights
//   8. Accumulate V = anc_V * w_anc + U_gathered @ VV_gathered * w_delta
//   9. GQA expand: [H_kv, D] → [H_q, D]
//
// Key insight: By gathering all K blocks at once and working with [K, S, *] tensors,
// we replace K*(S+1) ATen op dispatches with a small constant number of ops.
// This reduces MPS command encoder overhead from O(K*S) to O(1) per layer.
//
// Thread safety: All ops are ATen-dispatched. No Metal API calls.

#include "decode_attention.hpp"
#include <torch/extension.h>
#include <cmath>
#include <limits>

namespace diffkv {

// ── Internal helper: GQA query reduction ─────────────────────────────────────
static torch::Tensor gqa_reduce_q(
    const torch::Tensor& q,
    int n_kv_heads
) {
    const int H_q = static_cast<int>(q.size(0));
    const int D   = static_cast<int>(q.size(1));
    const int g   = H_q / n_kv_heads;

    if (g == 1) return q.to(torch::kFloat32);

    return q.to(torch::kFloat32)
             .reshape({n_kv_heads, g, D})
             .mean(1);  // [n_kv_heads, D]
}

// ── Core implementation ───────────────────────────────────────────────────────
// Returns (out [H_q, D] float16, lse [H_q] float32).
static std::tuple<torch::Tensor, torch::Tensor> _decode_attention_impl(
    const torch::Tensor& Q,            // [H_q, D] float16
    const torch::Tensor& U_pool,       // [N_pool, S_max, R] int8
    const torch::Tensor& U_scale_pool, // [N_pool] float16
    const torch::Tensor& VK_pool,      // [N_pool, R, n_kv_heads, D] float16
    const torch::Tensor& VV_pool,      // [N_pool, R, n_kv_heads, D] float16
    const torch::Tensor& anchors_K,    // [N_pool, n_kv_heads, D] float16
    const torch::Tensor& anchors_V,    // [N_pool, n_kv_heads, D] float16
    const torch::Tensor& seq_lens,     // [N_pool] int32
    const torch::Tensor& slot_indices, // [K_active] int32
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
) {
    const auto device = Q.device();
    const int  D      = static_cast<int>(Q.size(1));
    const int  K      = static_cast<int>(slot_indices.size(0));
    const int  g      = n_q_heads / n_kv_heads;
    const int  S_max  = static_cast<int>(U_pool.size(1));

    // ── Empty slot case ───────────────────────────────────────────────────────
    if (K == 0) {
        auto out = torch::zeros({n_q_heads, D},
            torch::TensorOptions().dtype(torch::kFloat16).device(device));
        auto lse = torch::full({n_q_heads}, -std::numeric_limits<float>::infinity(),
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        return {out, lse};
    }

    auto slots64 = slot_indices.to(torch::kInt64);  // [K]

    // ── GQA Q-reduction: [H_q, D] → [n_kv_heads, D] float32 ─────────────────
    auto q_kv = gqa_reduce_q(Q, n_kv_heads);  // [n_kv_heads, D] float32

    // ── Gather pool tensors for active slots ──────────────────────────────────
    // anchors_K gathered: [K, n_kv_heads, D] float16
    auto anc_K_g = anchors_K.index_select(0, slots64).to(torch::kFloat32);  // [K, n_kv, D]
    auto anc_V_g = anchors_V.index_select(0, slots64).to(torch::kFloat32);  // [K, n_kv, D]

    // U_pool gathered:  [K, S_max, R] int8 → dequantize → [K, S_max, R] float32
    auto U_g_int8 = U_pool.index_select(0, slots64);                        // [K, S_max, R]
    auto u_scales = U_scale_pool.index_select(0, slots64).to(torch::kFloat32); // [K]
    // Dequantize: broadcast scale [K, 1, 1]
    auto U_g = U_g_int8.to(torch::kFloat32)
                        .mul_(u_scales.view({K, 1, 1}));                     // [K, S_max, R]

    // VK_pool gathered: [K, R, n_kv_heads, D] float16 → float32
    auto VK_g = VK_pool.index_select(0, slots64).to(torch::kFloat32);       // [K, R, n_kv, D]
    auto VV_g = VV_pool.index_select(0, slots64).to(torch::kFloat32);       // [K, R, n_kv, D]

    // seq_lens gathered: [K] int32
    auto slen_g = seq_lens.index_select(0, slots64).to(torch::kInt32);      // [K]

    // ── Step 1: Anchor scores — [n_kv_heads, K] ───────────────────────────────
    // anc_K_g: [K, n_kv, D] → permute → [n_kv, K, D]
    // q_kv [n_kv, D] × anc_K [n_kv, D, K] → [n_kv, K]
    auto anc_K_perm = anc_K_g.permute({1, 0, 2});  // [n_kv, K, D]
    auto s_anc = torch::bmm(
        q_kv.unsqueeze(1),    // [n_kv, 1, D]
        anc_K_perm.permute({0, 2, 1})  // [n_kv, D, K]
    ).squeeze(1).mul_(scale);  // [n_kv, K]

    // ── Step 2: Q projection into SVD subspace — [n_kv_heads, K, R] ──────────
    // VK_g: [K, R, n_kv, D] → permute → [n_kv, K, R, D]
    // q_proj[h, k, r] = q_kv[h] @ VK_g[k, r, h, :]
    // Batched: for each h, for each k:
    //   q_kv[h]: [D], VK_g[k, :, h, :]: [R, D]
    //   q_proj[h, k, :] = VK_g[k, :, h, :] @ q_kv[h] = [R]
    // Vectorized over k:
    //   VK_g[:,: , h, :] shape [K, R, D] → q_kv[h] [D] → [K, R] via mv
    //   Collect over h → [n_kv, K, R]
    // Efficient: reshape VK_g to [K*R, n_kv*D] ... better:
    // VK_g [K, R, n_kv, D] → permute to [n_kv, K*R, D] → mm q_kv.T → [n_kv, K*R] → reshape [n_kv, K, R]
    auto VK_nkv_kr_d = VK_g.permute({2, 0, 1, 3}).reshape({n_kv_heads, K * rank, D});  // [n_kv, K*R, D]
    auto q_proj = torch::bmm(
        VK_nkv_kr_d,                   // [n_kv, K*R, D]
        q_kv.unsqueeze(2)              // [n_kv, D, 1]
    ).squeeze(2).reshape({n_kv_heads, K, rank});  // [n_kv, K, R]

    // ── Step 3: Delta scores — [n_kv_heads, K, S_max] ─────────────────────────
    // For each h: q_proj[h, :, :] [K, R] × U_g [K, S_max, R].T → [K, S_max]
    // Batched over h: stack U_g → [n_kv, K, S_max, R] (via expand) then bmm
    // OR: since U_g is the same for all heads (pooling), we can:
    //   U_g: [K, S_max, R] → [1, K*S_max, R] → expanded [n_kv, K*S_max, R]
    //   q_proj [n_kv, K, R] → [n_kv, K*S_max via broadcast]
    // Better: reshape as [n_kv, K, R] @ [K, R, S_max] per head-block pair.
    // Simplest vectorized form:
    //   q_proj: [n_kv, K, R]
    //   U_g.permute(0,2,1): [K, R, S_max]
    //   → For each h: q_proj[h] [K, R] @ U_g.T [K, R, S_max] → [K, S_max]... can't batch K dim.
    // Use: bmm with shapes [n_kv*K, 1, R] @ [n_kv*K, R, S_max]
    // Or: einsum("hkr,ksr->hks", q_proj, U_g)
    auto s_delta = torch::einsum("hkr,ksr->hks",
                                  {q_proj,           // [n_kv, K, R]
                                   U_g})             // [K, S_max, R]
                        .mul_(scale);                // [n_kv, K, S_max]

    // ── Step 4: Build seq_len mask — [K, S_max] ───────────────────────────────
    // mask[k, t] = 1 if t < seq_lens[k], else 0
    // Used to zero out padding scores before softmax.
    auto t_range = torch::arange(S_max, torch::TensorOptions()
                                               .dtype(torch::kInt32)
                                               .device(device));  // [S_max]
    // slen_g [K] → compare with t_range [S_max] → [K, S_max] bool
    auto mask = (t_range.unsqueeze(0) < slen_g.to(torch::kInt32).unsqueeze(1));  // [K, S_max] bool

    // ── Step 5: Online softmax over (anchor + masked delta) per head ──────────
    // Scores: s_anc [n_kv, K], s_delta [n_kv, K, S_max] (masked)
    //
    // Set masked positions to -inf before combining for joint softmax.
    // Combined: concat anchor (1 token equiv) with deltas along dim -1.
    // Anchor is always "present" (no masking needed).
    //
    // Build joint score matrix: [n_kv, K, 1+S_max] with mask applied to delta part.
    const float NEG_INF_VAL = -1e30f;

    // Apply mask to delta scores: set t >= seq_lens to -inf
    // mask [K, S_max] → expand [n_kv, K, S_max]
    auto mask_expanded = mask.unsqueeze(0).expand({n_kv_heads, K, S_max});
    auto s_delta_masked = s_delta.clone();
    s_delta_masked.masked_fill_(~mask_expanded, NEG_INF_VAL);

    // Concat: s_anc [n_kv, K, 1] + s_delta_masked [n_kv, K, S_max] → [n_kv, K, 1+S_max]
    auto scores_all = torch::cat(
        {s_anc.unsqueeze(2), s_delta_masked},
        /*dim=*/2
    );  // [n_kv, K, 1+S_max]

    // Flatten K blocks: [n_kv, K*(1+S_max)] → softmax over all blocks jointly
    auto scores_flat = scores_all.reshape({n_kv_heads, K * (1 + S_max)});
    auto weights = torch::softmax(scores_flat.to(torch::kFloat32), /*dim=*/1)
                         .reshape({n_kv_heads, K, 1 + S_max});  // [n_kv, K, 1+S_max]

    // ── Step 6: Accumulate output values ──────────────────────────────────────
    // V_anchor:  anc_V_g [K, n_kv, D] → permute → [n_kv, K, D]
    // V_delta_b: U_g [K, S_max, R] @ VV_g[:,: , h, :] → [n_kv, K, S_max, D]
    //
    // anchor weights: weights[:, :, 0]  → [n_kv, K]
    // delta weights:  weights[:, :, 1:] → [n_kv, K, S_max]

    auto w_anc   = weights.select(2, 0);         // [n_kv, K]
    auto w_delta = weights.slice(2, 1, 1 + S_max);  // [n_kv, K, S_max]

    // Anchor value contribution: anc_V_g [K, n_kv, D] → [n_kv, K, D]
    auto anc_V_nkv = anc_V_g.permute({1, 0, 2});  // [n_kv, K, D]
    // Weighted sum over K: [n_kv, K] * [n_kv, K, D] → [n_kv, D]
    auto out_anchor = (w_anc.unsqueeze(2) * anc_V_nkv).sum(1);  // [n_kv, D]

    // Delta value contribution:
    // V_delta = U_g @ VV_g_flat: U_g [K, S_max, R] @ VV_g [K, R, n_kv*D] → [K, S_max, n_kv, D]
    // VV_g: [K, R, n_kv, D] → reshape → [K, R, n_kv*D]
    auto VV_flat = VV_g.reshape({K, rank, n_kv_heads * D});        // [K, R, n_kv*D]
    // V_reconst = U_g @ VV_flat: [K, S_max, R] @ [K, R, n_kv*D] → [K, S_max, n_kv*D]
    auto V_reconst = torch::bmm(U_g, VV_flat)
                         .reshape({K, S_max, n_kv_heads, D})
                         .permute({2, 0, 1, 3});   // [n_kv, K, S_max, D]

    // Apply delta weights and mask (V_reconst already 0 at padded positions via weights=-inf→0)
    // w_delta: [n_kv, K, S_max] → unsqueeze → [n_kv, K, S_max, 1]
    auto out_delta = (w_delta.unsqueeze(3) * V_reconst).sum({1, 2});  // [n_kv, D]

    auto out_kv = (out_anchor + out_delta).to(torch::kFloat16);  // [n_kv, D] float16

    // ── Step 7: GQA expand [n_kv, D] → [H_q, D] ──────────────────────────────
    torch::Tensor out_q;
    if (g == 1) {
        out_q = out_kv;
    } else {
        out_q = out_kv.unsqueeze(1)
                      .expand({n_kv_heads, g, D})
                      .reshape({n_q_heads, D})
                      .contiguous();
    }

    // ── Step 8: LSE — log-sum-exp for LSE-combine with dense window ───────────
    // LSE per head = logsumexp over the joint score distribution
    auto lse_kv = torch::logsumexp(scores_flat.to(torch::kFloat32), /*dim=*/1);  // [n_kv]
    torch::Tensor lse_q;
    if (g == 1) {
        lse_q = lse_kv;
    } else {
        lse_q = lse_kv.unsqueeze(1)
                      .expand({n_kv_heads, g})
                      .reshape({n_q_heads})
                      .contiguous();
    }

    return {out_q, lse_q};
}

// ── Public API ────────────────────────────────────────────────────────────────

torch::Tensor decode_attention_aten(
    const torch::Tensor& Q,
    const torch::Tensor& U_pool,
    const torch::Tensor& U_scale_pool,
    const torch::Tensor& VK_pool,
    const torch::Tensor& VV_pool,
    const torch::Tensor& anchors_K,
    const torch::Tensor& anchors_V,
    const torch::Tensor& seq_lens,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
) {
    auto [out, lse] = _decode_attention_impl(
        Q, U_pool, U_scale_pool, VK_pool, VV_pool,
        anchors_K, anchors_V, seq_lens, slot_indices,
        scale, n_q_heads, n_kv_heads, rank
    );
    (void)lse;
    return out;  // [H_q, D] float16
}

std::tuple<torch::Tensor, torch::Tensor> decode_attention_aten_lse(
    const torch::Tensor& Q,
    const torch::Tensor& U_pool,
    const torch::Tensor& U_scale_pool,
    const torch::Tensor& VK_pool,
    const torch::Tensor& VV_pool,
    const torch::Tensor& anchors_K,
    const torch::Tensor& anchors_V,
    const torch::Tensor& seq_lens,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
) {
    return _decode_attention_impl(
        Q, U_pool, U_scale_pool, VK_pool, VV_pool,
        anchors_K, anchors_V, seq_lens, slot_indices,
        scale, n_q_heads, n_kv_heads, rank
    );
}

} // namespace diffkv
