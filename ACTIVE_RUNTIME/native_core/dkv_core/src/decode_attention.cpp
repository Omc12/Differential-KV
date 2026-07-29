// dkv_core/src/decode_attention.cpp
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

#ifdef DKV_APPLE
#include "metal_runtime.hpp"
#endif

namespace dkv {

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

// ── Internal helper: RoPE rotate_half ────────────────────────────────────────
// Implements the half-dimension swap used in standard Rotary Positional Embedding.
// Input x: [..., D] float32
// Output:  [..., D] float32 — first half negated and swapped with second half
static torch::Tensor rotate_half_cpp(const torch::Tensor& x) {
    const int D    = static_cast<int>(x.size(-1));
    const int half = D / 2;
    auto x1 = x.slice(-1, 0, half);     // [..., D/2]
    auto x2 = x.slice(-1, half, D);     // [..., D/2]
    return torch::cat({-x2, x1}, /*dim=*/-1);  // [..., D]
}

// ── Internal helper: apply RoPE to a set of keys given per-slot cos/sin ──────
// keys:    [K, n_heads, D]          float32
// cos_anc: [K, rotary_dim]          float32  (already squeezed to per-anchor cosine)
// sin_anc: [K, rotary_dim]          float32
// Returns: [K, n_heads, D]          float32  rotated keys
//
// Partial RoPE (Qwen3.5/GLM-style partial_rotary_factor<1.0): cos_anc/sin_anc's
// last dim (rotary_dim) may be smaller than D -- rotate only that leading
// slice and pass the remainder through unrotated. rotate_half_cpp derives its
// own half-width from whatever tensor it's given, so calling it on the
// rotary_dim-wide slice (not the full D-wide `keys`) automatically pairs
// dimensions within [0, rotary_dim) instead of [0, D) -- this was the actual
// bug (every caller used to implicitly assume rotary_dim == D). When
// rotary_dim == D this reduces to exactly the original full-width rotation.
static torch::Tensor apply_rope_to_keys(
    const torch::Tensor& keys,       // [K, n_heads, D]
    const torch::Tensor& cos_anc,    // [K, rotary_dim]
    const torch::Tensor& sin_anc     // [K, rotary_dim]
) {
    const int64_t D = keys.size(-1);
    const int64_t rotary_dim = cos_anc.size(-1);
    auto c = cos_anc.unsqueeze(1);  // [K, 1, rotary_dim]
    auto s = sin_anc.unsqueeze(1);  // [K, 1, rotary_dim]
    if (rotary_dim >= D) {
        return keys * c + rotate_half_cpp(keys) * s;  // [K, n_heads, D]
    }
    auto k_rot  = keys.slice(-1, 0, rotary_dim);   // [K, n_heads, rotary_dim]
    auto k_pass = keys.slice(-1, rotary_dim, D);   // [K, n_heads, D-rotary_dim]
    auto rotated = k_rot * c + rotate_half_cpp(k_rot) * s;
    return torch::cat({rotated, k_pass}, /*dim=*/-1);  // [K, n_heads, D]
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
    const torch::Tensor& scales,       // [N_pool] float16
    const torch::Tensor& cos_anc,      // [K_active, D] float32 — RoPE cosine at anchor positions
    const torch::Tensor& sin_anc,      // [K_active, D] float32 — RoPE sine at anchor positions
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
    // The real per-slot rank width of VK_pool/VV_pool/U_pool, derived the same
    // way S_max is above rather than trusted from the `rank` parameter.
    // DKV_LAYER_ADAPTIVE_RANK (default on) compresses different layers at
    // different ranks, so the pool is allocated at the max across layers and
    // a given layer's blocks may use fewer (or, with content-boost, more)
    // columns than the caller's flat `rank` argument. All the reshape/expand/
    // einsum calls below describe real tensor dimensions and must use this,
    // not `rank` -- using `rank` directly used to throw a shape-mismatch
    // RuntimeError (or, for an einsum, a dimension-mismatch error) on every
    // call where a layer's actual rank differed from the flat default.
    const int  rank_real = static_cast<int>(VK_pool.size(1));
    (void)rank;  // superseded by rank_real; kept in the signature for API compatibility

    // ── Empty slot case ───────────────────────────────────────────────────────
    if (K == 0) {
        auto out = torch::zeros({n_q_heads, D},
            torch::TensorOptions().dtype(torch::kFloat16).device(device));
        auto lse = torch::full({n_q_heads}, -std::numeric_limits<float>::infinity(),
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        return {out, lse};
    }

    auto slots64 = slot_indices.to(torch::kInt64);  // [K]

    // ── Gather pool tensors for active slots ──────────────────────────────────
    auto anc_K_g_raw = anchors_K.index_select(0, slots64).to(torch::kFloat32);  // [K, n_kv_heads, D]
    auto anc_V_g = anchors_V.index_select(0, slots64).to(torch::kFloat32);      // [K, n_kv_heads, D]

    // ── On-the-fly RoPE rotation for anchor keys ─────────────────────────────
    // anchors_K are stored unrotated. Apply RoPE using per-slot cos/sin at anchor positions.
    // cos_anc: [K, D] — rotate each anchor key by its absolute sequence position.
    bool has_rope = (cos_anc.defined() && cos_anc.numel() > 0);
    torch::Tensor anc_K_g;
    if (has_rope) {
        auto cos_f = cos_anc.to(torch::kFloat32);  // [K, D]
        auto sin_f = sin_anc.to(torch::kFloat32);  // [K, D]
        anc_K_g = apply_rope_to_keys(anc_K_g_raw, cos_f, sin_f);  // [K, n_kv_heads, D]
    } else {
        anc_K_g = anc_K_g_raw;
    }

    // U_pool gathered:  [K, S_max, R] int8 → dequantize → [K, S_max, R] float32
    auto U_g_int8 = U_pool.index_select(0, slots64);                        // [K, S_max, R]
    auto u_scales = U_scale_pool.index_select(0, slots64).to(torch::kFloat32); // [K]
    // Dequantize: broadcast scale [K, 1, 1]
    auto U_g = U_g_int8.to(torch::kFloat32)
                        .mul_(u_scales.view({K, 1, 1}));                     // [K, S_max, R]

    // VK_pool gathered: [K, R, n_kv_heads, D] float16 → float32
    auto VK_g_raw = VK_pool.index_select(0, slots64).to(torch::kFloat32);   // [K, R, n_kv_heads, D]
    // ── On-the-fly RoPE rotation for VK (SVD key basis vectors) ─────────────
    // VK basis vectors are stored unrotated. Rotate each [n_kv_heads, D] slice
    // using the corresponding anchor's cos/sin (all basis vectors share the
    // same position as their anchor — valid for the rank-1 approximation).
    torch::Tensor VK_g;
    if (has_rope) {
        auto cos_f = cos_anc.to(torch::kFloat32);  // [K, D]
        auto sin_f = sin_anc.to(torch::kFloat32);  // [K, D]
        // VK_g_raw: [K, R, n_kv_heads, D] — reshape to [K*R, n_kv_heads, D] for batched rotation
        auto VK_flat = VK_g_raw.reshape({K * rank_real, n_kv_heads, D});     // [K*R, n_kv_heads, D]
        // Repeat cos/sin for R ranks: [K, D] → [K*R, D]
        auto cos_rep = cos_f.unsqueeze(1).expand({K, rank_real, D}).reshape({K * rank_real, D});
        auto sin_rep = sin_f.unsqueeze(1).expand({K, rank_real, D}).reshape({K * rank_real, D});
        VK_g = apply_rope_to_keys(VK_flat, cos_rep, sin_rep).reshape({K, rank_real, n_kv_heads, D});
    } else {
        VK_g = VK_g_raw;
    }
    auto VV_g = VV_pool.index_select(0, slots64).to(torch::kFloat32);       // [K, R, n_kv_heads, D]
    auto slen_g = seq_lens.index_select(0, slots64);                        // [K]
    auto scales_g = scales.index_select(0, slots64).to(torch::kFloat32);    // [K]

    // ── GQA key/value repetition to match n_q_heads ──────────────────────────
    // Repetition factor: g = n_q_heads / n_kv_heads
    auto anc_K_expand = anc_K_g.unsqueeze(2).expand({K, n_kv_heads, g, D}).reshape({K, n_q_heads, D});
    auto anc_V_expand = anc_V_g.unsqueeze(2).expand({K, n_kv_heads, g, D}).reshape({K, n_q_heads, D});
    auto VK_expand = VK_g.unsqueeze(3).expand({K, rank_real, n_kv_heads, g, D}).reshape({K, rank_real, n_q_heads, D});
    auto VV_expand = VV_g.unsqueeze(3).expand({K, rank_real, n_kv_heads, g, D}).reshape({K, rank_real, n_q_heads, D});

    // ── Step 1: Anchor scores — [n_q_heads, K] ───────────────────────────────
    // anc_K_expand: [K, n_q_heads, D] → permute → [n_q_heads, K, D]
    // Q [n_q_heads, D] × anc_K [n_q_heads, D, K] → [n_q_heads, K]
    auto anc_K_perm = anc_K_expand.permute({1, 0, 2});  // [n_q_heads, K, D]
    auto s_anc = torch::bmm(
        Q.unsqueeze(1).to(torch::kFloat32),    // [n_q_heads, 1, D]
        anc_K_perm.permute({0, 2, 1})  // [n_q_heads, D, K]
    ).squeeze(1).mul_(scale);  // [n_q_heads, K]

    // ── Step 2: Q projection into SVD subspace — [n_q_heads, K, R] ──────────
    // VK_expand: [K, R, n_q_heads, D] → permute → [n_q_heads, K, R, D]
    // q_proj[h, k, r] = Q[h] @ VK_expand[k, r, h, :]
    // VK_expand [K, R, n_q_heads, D] → permute to [n_q_heads, K*R, D] → mm Q.T → [n_q_heads, K*R] → reshape [n_q_heads, K, R]
    auto VK_nq_kr_d = VK_expand.permute({2, 0, 1, 3}).reshape({n_q_heads, K * rank_real, D});  // [n_q_heads, K*R, D]
    auto q_proj = torch::bmm(
        VK_nq_kr_d,                   // [n_q_heads, K*R, D]
        Q.unsqueeze(2).to(torch::kFloat32)  // [n_q_heads, D, 1]
    ).squeeze(2).reshape({n_q_heads, K, rank_real});  // [n_q_heads, K, R]

    // ── Step 3: Delta scores — [n_q_heads, K, S_max] ─────────────────────────
    // For each h: q_proj[h, :, :] [K, R] × U_g [K, S_max, R].T → [K, S_max]
    // einsum("hkr,ksr->hks", q_proj, U_g)
    auto s_delta = torch::einsum("hkr,ksr->hks",
                                  {q_proj,           // [n_q_heads, K, R]
                                   U_g})             // [K, S_max, R]
                        .mul_(scale);                // [n_q_heads, K, S_max]
    // Scale delta scores by the block-level scales
    s_delta.mul_(scales_g.view({1, K, 1}));
    s_delta.add_(s_anc.unsqueeze(2));

    // ── Step 4: Build seq_len mask — [K, S_max] ───────────────────────────────
    // mask[k, t] = 1 if t < seq_lens[k], else 0
    // Used to zero out padding scores before softmax.
    auto t_range = torch::arange(S_max, torch::TensorOptions()
                                               .dtype(torch::kInt32)
                                               .device(device));  // [S_max]
    // slen_g [K] → compare with t_range [S_max] → [K, S_max] bool
    auto mask = (t_range.unsqueeze(0) < slen_g.to(torch::kInt32).unsqueeze(1));  // [K, S_max] bool

    // ── Step 5: Online softmax over (anchor + masked delta) per head ──────────
    // Scores: s_anc [n_q_heads, K], s_delta [n_q_heads, K, S_max] (masked)
    // Combined: concat anchor (1 token equiv) with deltas along dim -1.
    const float NEG_INF_VAL = -1e30f;

    // Apply mask to delta scores: set t >= seq_lens to -inf
    auto mask_expanded = mask.unsqueeze(0).expand({n_q_heads, K, S_max});
    auto s_delta_masked = s_delta.clone();
    s_delta_masked.masked_fill_(~mask_expanded, NEG_INF_VAL);

    // Concat: s_anc [n_q_heads, K, 1] + s_delta_masked [n_q_heads, K, S_max] → [n_q_heads, K, 1+S_max]
    auto scores_all = torch::cat(
        {s_anc.unsqueeze(2), s_delta_masked},
        /*dim=*/2
    );  // [n_q_heads, K, 1+S_max]

    // Flatten K blocks: [n_q_heads, K*(1+S_max)] → softmax over all blocks jointly
    auto scores_flat = scores_all.reshape({n_q_heads, K * (1 + S_max)});
    auto weights = torch::softmax(scores_flat.to(torch::kFloat32), /*dim=*/1)
                         .reshape({n_q_heads, K, 1 + S_max});  // [H_q, K, 1+S_max]

    // ── Step 6: Accumulate output values ──────────────────────────────────────
    // anchor weights: weights[:, :, 0]  → [H_q, K]
    // delta weights:  weights[:, :, 1:] → [H_q, K, S_max]
    auto w_anc   = weights.select(2, 0);         // [H_q, K]
    auto w_delta = weights.slice(2, 1, 1 + S_max);  // [H_q, K, S_max]

    auto w_block_sum = w_anc + w_delta.sum(2);   // [H_q, K]

    // Anchor value contribution: anc_V_expand [K, H_q, D] → [H_q, K, D]
    auto anc_V_nq = anc_V_expand.permute({1, 0, 2});  // [H_q, K, D]
    // Weighted sum over K: [H_q, K] * [H_q, K, D] → [H_q, D]
    auto out_anchor = (w_block_sum.unsqueeze(2) * anc_V_nq).sum(1);  // [H_q, D]

    // Delta value contribution:
    // VV_expand: [K, R, H_q, D] → reshape → [K, R, H_q*D]
    auto VV_flat = VV_expand.reshape({K, rank_real, n_q_heads * D});  // [K, R, H_q*D]
    // V_reconst = U_g @ VV_flat: [K, S_max, R] @ [K, R, H_q*D] → [K, S_max, H_q*D]
    auto V_reconst = torch::bmm(U_g, VV_flat)
                         .reshape({K, S_max, n_q_heads, D})
                         .permute({2, 0, 1, 3});   // [H_q, K, S_max, D]

    // Apply delta weights, SVD block-level scales, and mask
    auto w_delta_scaled = w_delta * scales_g.view({1, K, 1});
    auto out_delta = (w_delta_scaled.unsqueeze(3) * V_reconst).sum({1, 2});  // [H_q, D]

    auto out_q = (out_anchor + out_delta).to(torch::kFloat16);  // [H_q, D] float16
    auto lse_q = torch::logsumexp(scores_flat.to(torch::kFloat32), /*dim=*/1);  // [H_q]

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
    const torch::Tensor& scales,
    const torch::Tensor& cos_anc,
    const torch::Tensor& sin_anc,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
) {
    auto [out, lse] = _decode_attention_impl(
        Q, U_pool, U_scale_pool, VK_pool, VV_pool,
        anchors_K, anchors_V, seq_lens, scales, cos_anc, sin_anc, slot_indices,
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
    const torch::Tensor& scales,
    const torch::Tensor& cos_anc,
    const torch::Tensor& sin_anc,
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
) {
    return _decode_attention_impl(
        Q, U_pool, U_scale_pool, VK_pool, VV_pool,
        anchors_K, anchors_V, seq_lens, scales, cos_anc, sin_anc, slot_indices,
        scale, n_q_heads, n_kv_heads, rank
    );
}

torch::Tensor fused_decode_attention_combined(
    const torch::Tensor& Q,               // [H_q, D]
    const torch::Tensor& dense_k,         // [1, H_kv, L_dense, D] or [H_kv, L_dense, D]
    const torch::Tensor& dense_v,         // [1, H_kv, L_dense, D] or [H_kv, L_dense, D]
    const torch::Tensor& cos_dense,       // [1, 1, L_dense, D] or [L_dense, D]
    const torch::Tensor& sin_dense,       // [1, 1, L_dense, D] or [L_dense, D]
    const torch::Tensor& U_pool,
    const torch::Tensor& U_scale_pool,
    const torch::Tensor& VK_pool,
    const torch::Tensor& VV_pool,
    const torch::Tensor& anchors_K,
    const torch::Tensor& anchors_V,
    const torch::Tensor& seq_lens,
    const torch::Tensor& scales,
    const torch::Tensor& cos_anc,         // [K_active, D] — RoPE cosine at anchor positions
    const torch::Tensor& sin_anc,         // [K_active, D] — RoPE sine at anchor positions
    const torch::Tensor& slot_indices,
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank,
    // Residual and Fact Anchor Override buffers (Track D)
    const torch::Tensor& res_pos_K,
    const torch::Tensor& res_val_K,
    const torch::Tensor& res_pos_V,
    const torch::Tensor& res_val_V,
    const torch::Tensor& fact_pos,
    const torch::Tensor& fact_val_K,
    const torch::Tensor& fact_val_V,
    // Partial RoPE: forwarded to decode_attention_metal on the MPS branch
    // below (the only branch this function's Python caller ever actually
    // exercises -- it gates the call to device.type=="mps" itself). -1
    // means full rotary; default lives in the header declaration, not here.
    // cos_anc/sin_anc must be Metal's padded-to-D buffer in that case, not
    // the true rotary_dim-wide tensor decode_attention_aten_lse (used only
    // on the never-actually-reached non-MPS branch) expects.
    int rotary_dim,
    // Full-sequence RoPE tables + per-slot anchor positions -- forwarded
    // straight to decode_attention_metal for exact-position residual/fact
    // rotation. Unused by the non-MPS branch (its ATen helpers reconstruct
    // per-token keys directly rather than applying block-level overrides).
    const c10::optional<torch::Tensor>& cos_full,
    const c10::optional<torch::Tensor>& sin_full,
    const c10::optional<torch::Tensor>& anchor_pos
) {
    const auto device = Q.device();
    const int D = Q.size(1);
    const int H_q = n_q_heads;
    const int H_kv = n_kv_heads;
    const int g = H_q / H_kv;

#ifdef DKV_APPLE
    if (device.is_mps()) {
        auto [out_final, lse_final] = decode_attention_metal(
            Q, U_pool, U_scale_pool, VK_pool, VV_pool,
            anchors_K, anchors_V, seq_lens, scales, cos_anc, sin_anc, slot_indices,
            scale, n_q_heads, n_kv_heads, rank,
            res_pos_K, res_val_K, res_pos_V, res_val_V,
            fact_pos, fact_val_K, fact_val_V,
            dense_k, dense_v, cos_dense, sin_dense,
            rotary_dim,
            cos_full, sin_full, anchor_pos
        );
        return out_final;
    }
#endif

    // ── 1. Sparse Attention ──
    torch::Tensor out_sparse, lse_sparse;
    bool has_sparse = (slot_indices.defined() && slot_indices.numel() > 0);
    if (has_sparse) {
        std::tie(out_sparse, lse_sparse) = decode_attention_aten_lse(
            Q, U_pool, U_scale_pool, VK_pool, VV_pool,
            anchors_K, anchors_V, seq_lens, scales, cos_anc, sin_anc, slot_indices,
            scale, n_q_heads, n_kv_heads, rank
        );
    }

    // ── 2. Dense Attention ──
    torch::Tensor out_dense, lse_dense;
    bool has_dense = (dense_k.defined() && dense_k.numel() > 0);
    if (has_dense) {
        auto dk = (dense_k.dim() == 4) ? dense_k.squeeze(0) : dense_k; // [H_kv, L_dense, D]
        auto dv = (dense_v.dim() == 4) ? dense_v.squeeze(0) : dense_v; // [H_kv, L_dense, D]

        // cos_dense and sin_dense may have shape [1, 1, L_dense, rotary_dim] or
        // [L_dense, rotary_dim] -- rotary_dim may be < D (Qwen3.5-style partial
        // RoPE). We squeeze and unsqueeze to get [1, L_dense, rotary_dim] for
        // broadcasting with dk [H_kv, L_dense, D].
        auto cos_view = cos_dense.squeeze().unsqueeze(0); // [1, L_dense, rotary_dim]
        auto sin_view = sin_dense.squeeze().unsqueeze(0); // [1, L_dense, rotary_dim]
        const int64_t rotary_dim = cos_view.size(-1);

        torch::Tensor dense_k_rot;
        if (rotary_dim >= D) {
            auto k_half = torch::empty_like(dk);
            int half_d = D / 2;
            k_half.slice(2, 0, half_d) = -dk.slice(2, half_d, D);
            k_half.slice(2, half_d, D) = dk.slice(2, 0, half_d);
            dense_k_rot = dk.mul(cos_view).addcmul(k_half, sin_view);
        } else {
            // Partial RoPE: rotate only the first rotary_dim dims, pass the
            // remainder through unrotated (same pattern as apply_rope_to_keys).
            auto dk_rot  = dk.slice(2, 0, rotary_dim);
            auto dk_pass = dk.slice(2, rotary_dim, D);
            auto k_half = torch::empty_like(dk_rot);
            int half_r = static_cast<int>(rotary_dim) / 2;
            k_half.slice(2, 0, half_r) = -dk_rot.slice(2, half_r, rotary_dim);
            k_half.slice(2, half_r, rotary_dim) = dk_rot.slice(2, 0, half_r);
            auto rotated = dk_rot.mul(cos_view).addcmul(k_half, sin_view);
            dense_k_rot = torch::cat({rotated, dk_pass}, /*dim=*/2);
        }

        // repeat_kv to match n_q_heads
        auto k_rep = dense_k_rot.unsqueeze(1).expand({H_kv, g, dense_k_rot.size(1), D}).reshape({H_q, dense_k_rot.size(1), D});
        auto v_rep = dv.unsqueeze(1).expand({H_kv, g, dv.size(1), D}).reshape({H_q, dv.size(1), D});

        if (has_dense && !has_sparse) {
            // Fast path: bypass all manual softmax math and call native torch::scaled_dot_product_attention
            // shapes expected: [B, H, L_q, D]
            auto q_sdpa = Q.unsqueeze(0).unsqueeze(2); // [1, H_q, 1, D]
            auto k_sdpa = k_rep.unsqueeze(0); // [1, H_q, L_dense, D]
            auto v_sdpa = v_rep.unsqueeze(0); // [1, H_q, L_dense, D]

            auto out_sdpa = torch::scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                /*attn_mask=*/{},
                /*dropout_p=*/0.0,
                /*is_causal=*/false,
                /*scale=*/scale
            );
            out_dense = out_sdpa.squeeze(2).squeeze(0); // [H_q, D]
        } else {
            // scores = batch matmul (k_rep @ Q.unsqueeze(2)) -> [H_q, L_dense, 1] -> squeeze to [H_q, L_dense]
            auto scores = torch::bmm(k_rep, Q.unsqueeze(2)).squeeze(2).mul(scale);

            // Consolidate manual math to native torch::softmax and torch::logsumexp
            auto w = torch::softmax(scores, -1);
            out_dense = torch::bmm(w.unsqueeze(1), v_rep).squeeze(1); // [H_q, D]
            lse_dense = torch::logsumexp(scores, -1); // [H_q]
        }
    }

    // ── 3. Combine ──
    if (has_dense && has_sparse) {
        auto lse_max = torch::maximum(lse_dense, lse_sparse);
        auto w_dense = torch::exp(lse_dense - lse_max);
        auto w_sparse = torch::exp(lse_sparse - lse_max);
        auto denom = w_dense + w_sparse;

        auto out_final = (out_dense.to(torch::kFloat32) * w_dense.unsqueeze(1) + 
                          out_sparse.to(torch::kFloat32) * w_sparse.unsqueeze(1)) / denom.unsqueeze(1);
        return out_final.to(torch::kFloat16);
    } else if (has_dense) {
        return out_dense.to(torch::kFloat16);
    } else if (has_sparse) {
        return out_sparse.to(torch::kFloat16);
    } else {
        return torch::zeros({H_q, D}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    }
}

} // namespace dkv
