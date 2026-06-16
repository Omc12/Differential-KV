#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <random>
#include <map>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include "runtime/diffkv_model.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../third_party/llama.cpp/ggml/src/ggml-impl.h"
#include "runtime/native_block_pool.hpp"
#include "native_core/srl/diffkv_srl.hpp"
#include "native_core/srl/factual_alignment.hpp"
#include "native_core/srl/query_router.hpp"
#include "runtime/diffkv_attention.hpp"
#include "native_core/compression/async_compressor.hpp"
#include "native_core/kv_runtime_manager.hpp"

using namespace diffkv;

static bool is_native_attn_enabled() {
    const char* e = std::getenv("DIFFKV_NATIVE_ATTN");
    if (e) {
        return (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "yes" || std::string(e) == "on");
    }
    return true; // ENABLED BY DEFAULT!
}

struct ggml_backend_owner {
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ggml_backend_owner() {
        bool use_gpu = false;
        if (const char* env_gpu = std::getenv("DIFFKV_USE_GPU")) {
            if (std::string(env_gpu) == "1") {
                use_gpu = true;
            }
        }
        cpu_backend = ggml_backend_cpu_init();
        if (use_gpu) {
            gpu_backend = ggml_backend_init_best();
            if (!gpu_backend) {
                gpu_backend = cpu_backend;
            }
        } else {
            gpu_backend = cpu_backend;
        }
        std::vector<ggml_backend_t> backends;
        if (gpu_backend && gpu_backend != cpu_backend) {
            backends.push_back(gpu_backend);
        }
        backends.push_back(cpu_backend);
        // Sched graph capacity must cover the largest graph we build. The native sparse-attn
        // path builds a ~32k-node decode graph; an undersized sched corrupts buffer planning.
        size_t sched_size = 8192;
        if (is_native_attn_enabled()) sched_size = 40960;
        sched = ggml_backend_sched_new(backends.data(), NULL, backends.size(), sched_size, false, true);
    }

    ~ggml_backend_owner() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        if (gpu_backend && gpu_backend != cpu_backend) {
            ggml_backend_free(gpu_backend);
        }
        if (cpu_backend) {
            ggml_backend_free(cpu_backend);
        }
    }
};

// Helper to build the Qwen 2.5 dense prefill graph using causal flash attention
// ── CPU RoPE helper ────────────────────────────────────────────────────────
// Applies NeoX-style rotary position embedding to raw K in-place.
// Matches the formula: k_rot[d] = k[d]*cos(theta) + k[d+D/2]*sin(theta)
// for d in [0, D/2), and k_rot[d+D/2] = -k[d]*sin(theta) + k[d+D/2]*cos(theta).
// token_positions: absolute position of each token, length = num_tokens.
// k_raw: flat buffer [num_tokens * kv_heads * head_dim], F32, layout [tok, head, dim].
// k_rotated: output buffer, same shape.
static void apply_rope_neox_cpu_fast(
    const float* k_raw, float* k_rotated,
    const float* cos_table, const float* sin_table,
    int num_tokens, int kv_heads, int head_dim
) {
    const int D    = head_dim;
    const int half = D / 2;
    for (int t = 0; t < num_tokens; ++t) {
        const float* cos_t = cos_table + t * half;
        const float* sin_t = sin_table + t * half;
        for (int h = 0; h < kv_heads; ++h) {
            const float* src = k_raw     + (t * kv_heads + h) * D;
            float*       dst = k_rotated + (t * kv_heads + h) * D;
            for (int i = 0; i < half; ++i) {
                float cos_th = cos_t[i];
                float sin_th = sin_t[i];
                float x      = src[i];
                float y      = src[i + half];
                dst[i]        = x * cos_th - y * sin_th;
                dst[i + half] = y * cos_th + x * sin_th;
            }
        }
    }
}

struct ggml_cgraph * build_prefill_graph(
    struct ggml_context * ctx,
    DiffKVModel & model,
    struct ggml_tensor * input_tokens,
    struct ggml_tensor * positions,
    struct ggml_tensor * mask,
    struct ggml_tensor ** out_logits,
    std::vector<struct ggml_tensor *>* k_layers = nullptr,
    std::vector<struct ggml_tensor *>* v_layers = nullptr,
    bool need_logits = false
) {
    const auto & config = model.get_config();
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // 1. Embedding lookup
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.get_token_embd(), input_tokens);

    for (int l = 0; l < config.n_layer; ++l) {
        const auto & layer = model.get_layers()[l];

        // 2. Attention RMSNorm
        struct ggml_tensor * h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.attn_norm);

        // 3. QKV Projections
        struct ggml_tensor * q = ggml_mul_mat(ctx, layer.wq, h);
        if (layer.bq) q = ggml_add(ctx, q, layer.bq);
        struct ggml_tensor * k = ggml_mul_mat(ctx, layer.wk, h);
        if (layer.bk) k = ggml_add(ctx, k, layer.bk);
        struct ggml_tensor * v = ggml_mul_mat(ctx, layer.wv, h);
        if (layer.bv) v = ggml_add(ctx, v, layer.bv);

        // Export raw V (no RoPE needed for V)
        if (v_layers) (*v_layers)[l] = v;

        // 4. RoPE
        int head_dim = config.n_embd / config.n_head;
        struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, head_dim, config.n_head, q->ne[1]);
        struct ggml_tensor * k_reshaped = ggml_reshape_3d(ctx, k, head_dim, config.n_head_kv, k->ne[1]);

        struct ggml_tensor * q_rope = ggml_rope_ext(ctx, q_reshaped, positions, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        struct ggml_tensor * k_rope = ggml_rope_ext(ctx, k_reshaped, positions, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Export RAW K (before RoPE) — matches ACTIVE_RUNTIME Python unrot_key_states.
        // diffkv_attention.cpp decode callback applies RoPE at query time (has_rope=true).
        if (k_layers) (*k_layers)[l] = k;

        // 5. Permute for Flash Attention
        struct ggml_tensor * q_perm = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        struct ggml_tensor * k_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
        struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, head_dim, config.n_head_kv, v->ne[1]);
        struct ggml_tensor * v_perm = ggml_permute(ctx, v_reshaped, 0, 2, 1, 3);

        // 6. Flash Attention
        float scale_val = 1.0f / std::sqrt((float)head_dim);
        struct ggml_tensor * attn_out_perm = ggml_flash_attn_ext(ctx, q_perm, k_perm, v_perm, mask, scale_val, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out_perm, GGML_PREC_F32);

        // 7. Flatten back to [n_embd, q_len]
        struct ggml_tensor * attn_out = ggml_reshape_2d(ctx, attn_out_perm, config.n_embd, q->ne[1]);

        // 8. Output Projection (WO)
        struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer.wo, attn_out);
        if (layer.bo) attn_proj = ggml_add(ctx, attn_proj, layer.bo);

        // 9. Residual connection
        cur = ggml_add(ctx, cur, attn_proj);

        // 10. FFN RMSNorm
        h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.ffn_norm);

        // 11. FFN SwiGLU
        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, h);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up, h);

        struct ggml_tensor * gate_silu = ggml_silu(ctx, gate);
        struct ggml_tensor * ffn_out   = ggml_mul(ctx, gate_silu, up);
        struct ggml_tensor * ffn_proj  = ggml_mul_mat(ctx, layer.ffn_down, ffn_out);

        // 12. Residual connection
        cur = ggml_add(ctx, cur, ffn_proj);
    }

    // 13. Final RMSNorm
    struct ggml_tensor * final_node = cur;
    if (need_logits) {
        struct ggml_tensor * final_norm = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        final_norm = ggml_mul(ctx, final_norm, model.get_output_norm());

        // Extract ONLY the last token from final_norm: shape [n_embd, 1]
        int chunk_len = input_tokens->ne[0];
        struct ggml_tensor * last_token_norm = ggml_view_1d(ctx, final_norm, config.n_embd, (chunk_len - 1) * config.n_embd * sizeof(float));

        // Multiply last_token_norm [n_embd, 1] by model.get_output() [n_embd, n_vocab] -> logits [n_vocab, 1]
        struct ggml_tensor * logits = ggml_mul_mat(ctx, model.get_output(), last_token_norm);
        *out_logits = logits;
        final_node = logits;
    } else {
        if (out_logits) *out_logits = nullptr;
    }

    ggml_build_forward_expand(gf, final_node);

    // Add views of k_layers and v_layers to prevent graph allocator reuse
    if (k_layers) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*k_layers)[l]) {
                struct ggml_tensor * dummy_k = ggml_view_1d(ctx, (*k_layers)[l], ggml_nelements((*k_layers)[l]), 0);
                ggml_build_forward_expand(gf, dummy_k);
            }
        }
    }
    if (v_layers) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*v_layers)[l]) {
                struct ggml_tensor * dummy_v = ggml_view_1d(ctx, (*v_layers)[l], ggml_nelements((*v_layers)[l]), 0);
                ggml_build_forward_expand(gf, dummy_v);
            }
        }
    }

    return gf;
}

// ── CHUNKED PREFILL WITH FULL-CONTEXT ATTENTION ────────────────────────────────
// Matches ACTIVE_RUNTIME Python: each chunk attends to ALL prior chunks' K/V
// (already RoPE-rotated) plus itself causally. Eliminates the O(N^2) memory
// spike while preserving exact causal attention semantics.
struct ggml_cgraph * build_prefill_ctx_graph(
    struct ggml_context * ctx,
    DiffKVModel & model,
    struct ggml_tensor * input_tokens,   // [chunk_len]
    struct ggml_tensor * positions,      // [chunk_len] — positions for CURRENT chunk only
    struct ggml_tensor * mask,           // [ctx_len, chunk_len] — full mask
    // Per-layer prior KV context tensors (RoPE-rotated, shape [head_dim, kv_heads, prior_len])
    std::vector<struct ggml_tensor *> * prior_k_ctx,
    std::vector<struct ggml_tensor *> * prior_v_ctx,
    struct ggml_tensor ** out_logits,
    std::vector<struct ggml_tensor *>* k_layers = nullptr,
    std::vector<struct ggml_tensor *>* v_layers = nullptr,
    bool need_logits = false
) {
    const auto & config = model.get_config();
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // 1. Embedding lookup
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.get_token_embd(), input_tokens);

    for (int l = 0; l < config.n_layer; ++l) {
        const auto & layer = model.get_layers()[l];

        // 2. Attention RMSNorm
        struct ggml_tensor * h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.attn_norm);

        // 3. QKV Projections
        struct ggml_tensor * q = ggml_mul_mat(ctx, layer.wq, h);
        if (layer.bq) q = ggml_add(ctx, q, layer.bq);
        struct ggml_tensor * k = ggml_mul_mat(ctx, layer.wk, h);
        if (layer.bk) k = ggml_add(ctx, k, layer.bk);
        struct ggml_tensor * v = ggml_mul_mat(ctx, layer.wv, h);
        if (layer.bv) v = ggml_add(ctx, v, layer.bv);

        // Export raw V
        if (v_layers) (*v_layers)[l] = v;

        // 4. RoPE on current chunk only
        int head_dim = config.n_embd / config.n_head;
        struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, head_dim, config.n_head, q->ne[1]);
        struct ggml_tensor * k_reshaped = ggml_reshape_3d(ctx, k, head_dim, config.n_head_kv, k->ne[1]);

        struct ggml_tensor * q_rope = ggml_rope_ext(ctx, q_reshaped, positions, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        struct ggml_tensor * k_rope = ggml_rope_ext(ctx, k_reshaped, positions, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Export RAW (pre-RoPE) K - matching ACTIVE_RUNTIME Python which clones key_states
        // BEFORE apply_rotary_pos_emb and stores unrot_key_states in blocks/active_k_dense.
        // RoPE is re-applied at decode time by diffkv_attention.cpp (has_rope=true).
        // For the prior-context prefill path, caller pre-rotates k_activations in CPU.
        if (k_layers) (*k_layers)[l] = k;  // raw K, shape [n_embd, chunk_len]

        // 5. Permute Q/K/V of current chunk: [head_dim, kv_heads, chunk_len] -> [head_dim, chunk_len, kv_heads]
        struct ggml_tensor * q_perm = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        struct ggml_tensor * k_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
        struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, head_dim, config.n_head_kv, v->ne[1]);
        struct ggml_tensor * v_perm = ggml_permute(ctx, v_reshaped, 0, 2, 1, 3);

        // 6. Concatenate prior context with current chunk along seq dim (dim=1 in permuted layout)
        struct ggml_tensor * k_ctx_perm = k_perm;
        struct ggml_tensor * v_ctx_perm = v_perm;
        bool has_prior = (prior_k_ctx && (*prior_k_ctx)[l] != nullptr);
        if (has_prior) {
            // prior tensors are already [head_dim, kv_heads, prior_len] — permute to [head_dim, prior_len, kv_heads]
            struct ggml_tensor * pk = ggml_permute(ctx, (*prior_k_ctx)[l], 0, 2, 1, 3);
            struct ggml_tensor * pv = ggml_permute(ctx, (*prior_v_ctx)[l], 0, 2, 1, 3);
            k_ctx_perm = ggml_concat(ctx, pk, k_perm, 1);
            v_ctx_perm = ggml_concat(ctx, pv, v_perm, 1);
        }

        // 7. Flash Attention with full context mask
        float scale_val = 1.0f / std::sqrt((float)head_dim);
        struct ggml_tensor * attn_out_perm = ggml_flash_attn_ext(ctx, q_perm, k_ctx_perm, v_ctx_perm, mask, scale_val, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out_perm, GGML_PREC_F32);

        // 8. Flatten back to [n_embd, chunk_len]
        struct ggml_tensor * attn_out = ggml_reshape_2d(ctx, attn_out_perm, config.n_embd, q->ne[1]);

        // 9. Output Projection
        struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer.wo, attn_out);
        if (layer.bo) attn_proj = ggml_add(ctx, attn_proj, layer.bo);

        // 10. Residual
        cur = ggml_add(ctx, cur, attn_proj);

        // 11. FFN RMSNorm
        h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.ffn_norm);

        // 12. FFN SwiGLU
        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, h);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up, h);
        struct ggml_tensor * gate_silu = ggml_silu(ctx, gate);
        struct ggml_tensor * ffn_out   = ggml_mul(ctx, gate_silu, up);
        struct ggml_tensor * ffn_proj  = ggml_mul_mat(ctx, layer.ffn_down, ffn_out);

        // 13. Residual
        cur = ggml_add(ctx, cur, ffn_proj);
    }

    // 14. Final RMSNorm + logits for last chunk
    struct ggml_tensor * final_node = cur;
    if (need_logits) {
        struct ggml_tensor * final_norm = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        final_norm = ggml_mul(ctx, final_norm, model.get_output_norm());
        int chunk_len = input_tokens->ne[0];
        struct ggml_tensor * last_token_norm = ggml_view_1d(ctx, final_norm, config.n_embd, (chunk_len - 1) * config.n_embd * sizeof(float));
        struct ggml_tensor * logits = ggml_mul_mat(ctx, model.get_output(), last_token_norm);
        *out_logits = logits;
        final_node = logits;
    } else {
        if (out_logits) *out_logits = nullptr;
    }

    ggml_build_forward_expand(gf, final_node);

    // Keep output tensors alive
    if (k_layers) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*k_layers)[l]) {
                ggml_build_forward_expand(gf, ggml_view_1d(ctx, (*k_layers)[l], ggml_nelements((*k_layers)[l]), 0));
            }
        }
    }
    if (v_layers) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*v_layers)[l]) {
                ggml_build_forward_expand(gf, ggml_view_1d(ctx, (*v_layers)[l], ggml_nelements((*v_layers)[l]), 0));
            }
        }
    }
    if (prior_k_ctx) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*prior_k_ctx)[l]) {
                ggml_build_forward_expand(gf, ggml_view_1d(ctx, (*prior_k_ctx)[l], ggml_nelements((*prior_k_ctx)[l]), 0));
            }
        }
    }
    if (prior_v_ctx) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*prior_v_ctx)[l]) {
                ggml_build_forward_expand(gf, ggml_view_1d(ctx, (*prior_v_ctx)[l], ggml_nelements((*prior_v_ctx)[l]), 0));
            }
        }
    }

    return gf;
}

// ── Native ggml-metal sparse attention subgraph (gated by DIFFKV_NATIVE_ATTN) ──
// Reproduces the custom-op "approximate" path (diffkv_attention.cpp:143-253) as native
// ggml ops, so sparse decode runs as one fused Metal command stream instead of 24
// per-layer ggml_map_custom3 CPU dispatches. Used only when the factual store is empty
// (factual/NIAH sessions keep the CPU custom op, which also does VSL/biasing). Consumes
// the precomputed pool tensors VK_rot/anchorK_rot/U_f16/valid_mask (RoPE'd at anchor pos).
// Returns attn_out [n_embd, 1].
//
// Per query head h (kv = h/group): standard attention over (1 anchor + S token) entries
// per slot. token entry value = anchorV + su*bs*(U[t]@VV); project-then-attend reuses U
// for both scores (q_proj·U) and the value projection (w·U).
static struct ggml_tensor * build_native_sparse_attn(
    struct ggml_context * ctx,
    struct ggml_tensor * q_rope,          // [D, nq, 1] rotated query
    struct ggml_tensor * k_rope,          // [D, nkv, 1] rotated CURRENT-token key
    struct ggml_tensor * v_cur,           // [nkv*D] CURRENT-token value (raw)
    struct ggml_tensor * dense_kr,        // [nkv*D, MAXD] rotated past-dense keys (f32 input)
    struct ggml_tensor * dense_v,         // [nkv*D, MAXD] past-dense values (f32 input)
    struct ggml_tensor * dense_mask,      // [MAXD] additive 0/-inf validity (f32 input)
    int MAXD,
    struct ggml_tensor * selected_slots,  // [K] i32 (may contain empties/duplicates)
    struct ggml_tensor * dup_tri,         // [K,K] strict-lower-triangular ones (j<k) const input
    struct ggml_tensor * half,            // [1] = 0.5 const input
    diffkv::NativeBlockPool * pool,
    int nq, int nkv, int D, int R, int S, int K, float scale,
    bool ignore_c,
    struct ggml_tensor ** out_dbg = nullptr
) {
    const int group = nq / nkv;
    const int n_slots = pool->get_seq_lens()->ne[0];
    const int F_kv = nkv * D;

    // ── Dedup penalty: the routed slots may repeat (padding); the CPU path dedups,
    // so penalize every NON-first occurrence to -inf. eq[j,k]=1 iff slot[j]==slot[k]
    // (integer ids → step(0.5-|diff|)); prior_count[k]=Σ_{j<k} eq; drop if >0. → [1,1,K]
    struct ggml_tensor* sF = ggml_cast(ctx, selected_slots, GGML_TYPE_F32);                 // [K]
    struct ggml_tensor* AA = ggml_repeat(ctx, ggml_reshape_2d(ctx, sF, K, 1), dup_tri);     // A[j,k]=s[j]
    struct ggml_tensor* BB = ggml_repeat(ctx, ggml_reshape_2d(ctx, sF, 1, K), dup_tri);     // B[j,k]=s[k]
    struct ggml_tensor* adiff = ggml_abs(ctx, ggml_sub(ctx, AA, BB));                        // |s[j]-s[k]|
    struct ggml_tensor* neg_half = ggml_neg(ctx, half);
    struct ggml_tensor* eq = ggml_step(ctx, ggml_add(ctx, ggml_neg(ctx, adiff), half));     // 1 if equal (add1→add: ADD1 not Metal)
    struct ggml_tensor* priorc = ggml_sum_rows(ctx, ggml_mul(ctx, eq, dup_tri));            // [1,K] prior dups
    // DEBUG: DIFFKV_NATIVE_NOSPARSE → drop ALL slots (priorc+0.5 > 0 always) to isolate the dense path.
    static const bool dbg_nosparse = (std::getenv("DIFFKV_NATIVE_NOSPARSE") != nullptr);
    struct ggml_tensor* drop = dbg_nosparse
        ? ggml_step(ctx, ggml_add(ctx, priorc, half))                        // always 1
        : ggml_step(ctx, ggml_add(ctx, priorc, neg_half));                   // 1 if priorc>=1
    struct ggml_tensor* dup_add = ggml_reshape_3d(ctx, ggml_scale(ctx, drop, std::getenv("DIFFKV_DBG_NODEDUP")?0.0f:-1e30f), 1, 1, K); // [1,1,K]

    // ── Gather the K selected slots from each pool tensor (f16 get_rows on Metal). ──
    auto gather = [&](struct ggml_tensor* t, int row_len) -> struct ggml_tensor* {
        struct ggml_tensor* t2d = ggml_reshape_2d(ctx, t, row_len, n_slots);
        return ggml_get_rows(ctx, t2d, selected_slots); // [row_len, K]
    };
    struct ggml_tensor* aKr  = ggml_reshape_3d(ctx, gather(pool->get_anchorK_rot(), D*nkv),     D, nkv, K);       // [D,nkv,K]
    struct ggml_tensor* VKr  = ggml_reshape_4d(ctx, gather(pool->get_VK_rot(),      D*nkv*R),   D, nkv, R, K);    // [D,nkv,R,K]
    struct ggml_tensor* VVs  = ggml_reshape_4d(ctx, gather(pool->get_VV(),          D*nkv*R),   D, nkv, R, K);    // [D,nkv,R,K]
    struct ggml_tensor* aVs  = ggml_reshape_3d(ctx, gather(pool->get_anchors_V(),   D*nkv),     D, nkv, K);       // [D,nkv,K]
    struct ggml_tensor* Usel = ggml_reshape_3d(ctx, gather(pool->get_U_f16(),       R*S),       R, S, K);         // [R,S,K]
    struct ggml_tensor* Msel = gather(pool->get_valid_mask(), S);                                                 // [S,K] additive -inf
    struct ggml_tensor* USf  = ggml_cast(ctx, gather(pool->get_U_scale(), 1), GGML_TYPE_F32);                     // [1,K]
    struct ggml_tensor* BSf  = ggml_cast(ctx, gather(pool->get_scales(),  1), GGML_TYPE_F32);                     // [1,K]

    // Per-slot scalars as [1,1,K] for broadcasting.
    struct ggml_tensor* su    = ggml_reshape_3d(ctx, USf, 1, 1, K);
    struct ggml_tensor* bs    = ggml_reshape_3d(ctx, BSf, 1, 1, K);
    struct ggml_tensor* su_bs = ggml_mul(ctx, su, bs); // [1,1,K]

    // anchor-entry mask = valid_mask row 0 (0 if slot non-empty, -inf if empty) → [1,1,K]
    struct ggml_tensor* anc_mask = ggml_reshape_3d(ctx,
        ggml_cast(ctx, ggml_cont(ctx, ggml_view_2d(ctx, Msel, 1, K, Msel->nb[1], 0)), GGML_TYPE_F32), 1, 1, K);

    struct ggml_tensor* Q2 = ggml_reshape_2d(ctx, q_rope, D, nq); // [D,nq]

    // Buffer-reuse safety: these tensors are computed once and consumed across BOTH kv-head
    // iterations (long lifetime). ggml_backend_sched's buffer reuse can overwrite them with a
    // large-valued intermediate before the 2nd iteration reads them → 113× corrupted output on
    // big-score (repetitive) inputs. Pinning them as outputs stops reuse. The self-test
    // (DIFFKV_SELFTEST, no buffer reuse) proves the MATH is exact, so this is the real fix.
    for (struct ggml_tensor* t : {aKr, VKr, VVs, aVs, Usel, Msel, USf, BSf, su, bs, su_bs, anc_mask, dup_add})
        ggml_set_output(t);

    std::vector<struct ggml_tensor*> kv_outs(nkv);
    for (int kv = 0; kv < nkv; ++kv) {
        // Per-kv views (cont to keep matmuls happy).
        struct ggml_tensor* Qk   = ggml_cont(ctx, ggml_view_2d(ctx, Q2, D, group, Q2->nb[1], (size_t)kv*group*Q2->nb[1])); // [D,group]
        struct ggml_tensor* aKrk = ggml_cont(ctx, ggml_view_2d(ctx, aKr, D, K, aKr->nb[2], (size_t)kv*aKr->nb[1]));        // [D,K]
        struct ggml_tensor* aVsk = ggml_cont(ctx, ggml_view_2d(ctx, aVs, D, K, aVs->nb[2], (size_t)kv*aVs->nb[1]));        // [D,K]
        // VKr/VVs [D,nkv,R,K] → fixed kv → [D,R,K]
        struct ggml_tensor* VKrk = ggml_cont(ctx, ggml_view_3d(ctx, VKr, D, R, K, VKr->nb[2], VKr->nb[3], (size_t)kv*VKr->nb[1])); // [D,R,K]
        struct ggml_tensor* VVsk = ggml_cont(ctx, ggml_view_3d(ctx, VVs, D, R, K, VVs->nb[2], VVs->nb[3], (size_t)kv*VVs->nb[1])); // [D,R,K]

        // dense-window views for this kv head
        struct ggml_tensor* dkr = ggml_cont(ctx, ggml_view_2d(ctx, dense_kr, D, MAXD, dense_kr->nb[1], (size_t)kv*D*dense_kr->nb[0])); // [D,MAXD]
        struct ggml_tensor* dvk = ggml_cont(ctx, ggml_view_2d(ctx, dense_v,  D, MAXD, dense_v->nb[1],  (size_t)kv*D*dense_v->nb[0]));  // [D,MAXD]

        // ── Sparse-pool scores → flat_sp [(S+1)*K, group] ──
        struct ggml_tensor* anc = ggml_mul_mat(ctx, aKrk, Qk); // [K,group] raw anchor score
        struct ggml_tensor* VKrk2 = ggml_reshape_2d(ctx, VKrk, D, R*K);
        struct ggml_tensor* qp = ggml_mul_mat(ctx, VKrk2, Qk);          // [R*K, group]
        qp = ggml_reshape_3d(ctx, qp, R, K, group);                     // [R,K,group]
        qp = ggml_cont(ctx, ggml_permute(ctx, qp, 0, 2, 1, 3));         // [R,group,K]
        struct ggml_tensor* delta = ggml_mul_mat(ctx, Usel, qp);        // [S,group,K]
        struct ggml_tensor* anc_b = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, anc)), 1, group, K); // [1,group,K]
        struct ggml_tensor* ts = ggml_add(ctx, ggml_mul(ctx, delta, su_bs), anc_b); // [S,group,K]
        ts = ggml_scale(ctx, ts, scale);
        ts = ggml_add(ctx, ts, ggml_reshape_3d(ctx, Msel, S, 1, K));                 // +token mask [S,1,K]
        // Anchor entry: ALWAYS included (matches execute_cpu_attention, which adds every selected
        // slot's anchor to the softmax with NO seq_len check — diffkv_attention.cpp:95-97/209).
        // Masking it by valid_mask[0] (seq_len==0) wrongly drops legit anchors of newly-started /
        // stale slots that the CPU reference DOES include → loses the attention "sink" → echo.
        struct ggml_tensor* ae = std::getenv("DIFFKV_MASK_EMPTY_ANCHOR")
            ? ggml_add(ctx, ggml_scale(ctx, anc_b, scale), anc_mask)
            : ggml_scale(ctx, anc_b, scale);                                              // [1,group,K]
        struct ggml_tensor* allsp = ggml_concat(ctx, ts, ae, 0);                     // [S+1,group,K]
        allsp = ggml_add(ctx, allsp, dup_add);                                       // -inf on duplicate slots
        struct ggml_tensor* perm  = ggml_cont(ctx, ggml_permute(ctx, allsp, 0, 2, 1, 3)); // [S+1,K,group]
        struct ggml_tensor* flat_sp = ggml_reshape_2d(ctx, perm, (S+1)*K, group);    // [(S+1)*K, group]

        // ── Dense-window scores ──
        struct ggml_tensor* ds = ggml_scale(ctx, ggml_mul_mat(ctx, dkr, Qk), scale); // [MAXD, group]
        ds = ggml_add(ctx, ds, ggml_reshape_2d(ctx, dense_mask, MAXD, 1));           // + validity mask

        // ── Softmax and split weights (conditional on ignore_c) ──
        struct ggml_tensor* w_sp = nullptr;
        struct ggml_tensor* w_ds = nullptr;
        struct ggml_tensor* w_cs = nullptr;

        if (ignore_c) {
            // DEBUG: DIFFKV_DBG_DENSEOFF → mask dense (-1e30) to isolate the sparse pool.
            static const bool denseoff = (std::getenv("DIFFKV_DBG_DENSEOFF") != nullptr);
            if (denseoff) {
                struct ggml_tensor* bigneg = ggml_scale(ctx, neg_half, 2e30f); // [1] = -1e30
                ds = ggml_add(ctx, ds, bigneg);
            }
            static const bool dsoff = (std::getenv("DIFFKV_DBG_DSOFF") != nullptr);
            if (dsoff) ds = ggml_add(ctx, ds, ggml_scale(ctx, neg_half, 2e30f));  // mask past-dense only

            struct ggml_tensor* allsc = ggml_concat(ctx, flat_sp, ds, 0); // [(S+1)*K + MAXD, group]
            struct ggml_tensor* w = ggml_soft_max(ctx, allsc);            // [(S+1)*K + MAXD, group]

            w_sp = ggml_cont(ctx, ggml_view_2d(ctx, w, (S+1)*K, group, w->nb[1], 0));
            w_ds = ggml_cont(ctx, ggml_view_2d(ctx, w, MAXD,    group, w->nb[1], (size_t)((S+1)*K)*w->nb[0]));
        } else {
            struct ggml_tensor* ckr = ggml_cont(ctx, ggml_view_2d(ctx, k_rope,   D, 1,    k_rope->nb[1],   (size_t)kv*k_rope->nb[1]));     // [D,1] current K
            struct ggml_tensor* cs = ggml_scale(ctx, ggml_mul_mat(ctx, ckr, Qk), scale); // [1, group] current token

            // DEBUG: DIFFKV_DBG_DENSEOFF → mask dense+current (-1e30) to isolate the sparse pool.
            static const bool denseoff = (std::getenv("DIFFKV_DBG_DENSEOFF") != nullptr);
            if (denseoff) {
                struct ggml_tensor* bigneg = ggml_scale(ctx, neg_half, 2e30f); // [1] = -1e30
                ds = ggml_add(ctx, ds, bigneg);
                cs = ggml_add(ctx, cs, bigneg);
            }
            static const bool curoff = (std::getenv("DIFFKV_DBG_CUROFF") != nullptr);
            if (curoff) cs = ggml_add(ctx, cs, ggml_scale(ctx, neg_half, 2e30f)); // mask current token only
            static const bool dsoff = (std::getenv("DIFFKV_DBG_DSOFF") != nullptr);
            if (dsoff) ds = ggml_add(ctx, ds, ggml_scale(ctx, neg_half, 2e30f));  // mask past-dense only

            struct ggml_tensor* allsc = ggml_concat(ctx, ggml_concat(ctx, flat_sp, ds, 0), cs, 0); // [(S+1)*K + MAXD + 1, group]
            struct ggml_tensor* w = ggml_soft_max(ctx, allsc);                                     // [E, group]

            w_sp = ggml_cont(ctx, ggml_view_2d(ctx, w, (S+1)*K, group, w->nb[1], 0));
            w_ds = ggml_cont(ctx, ggml_view_2d(ctx, w, MAXD,    group, w->nb[1], (size_t)((S+1)*K)*w->nb[0]));
            w_cs = ggml_cont(ctx, ggml_view_2d(ctx, w, 1,       group, w->nb[1], (size_t)((S+1)*K+MAXD)*w->nb[0]));
        }

        // ── Sparse value (project-then-attend) ──
        struct ggml_tensor* wsp3 = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, w_sp, S+1, K, group), 0, 2, 1, 3)); // [S+1,group,K]
        struct ggml_tensor* w_tok = ggml_cont(ctx, ggml_view_3d(ctx, wsp3, S, group, K, wsp3->nb[1], wsp3->nb[2], 0));        // [S,group,K]
        struct ggml_tensor* w_anc = ggml_view_3d(ctx, wsp3, 1, group, K, wsp3->nb[1], wsp3->nb[2], (size_t)S*wsp3->nb[0]);    // [1,group,K]
        struct ggml_tensor* w_total = ggml_add(ctx, ggml_sum_rows(ctx, w_tok), w_anc); // [1,group,K]
        struct ggml_tensor* wt2  = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, w_total), group, K))); // [K,group]
        struct ggml_tensor* aVsT = ggml_cont(ctx, ggml_transpose(ctx, aVsk));          // [K,D]
        struct ggml_tensor* term1 = ggml_mul_mat(ctx, aVsT, wt2);                       // [D,group]
        struct ggml_tensor* UselT = ggml_cont(ctx, ggml_transpose(ctx, Usel));          // [S,R,K]
        struct ggml_tensor* wproj = ggml_mul(ctx, ggml_mul_mat(ctx, UselT, w_tok), su); // [R,group,K]
        struct ggml_tensor* VVsT = ggml_cont(ctx, ggml_transpose(ctx, VVsk));           // [R,D,K]
        struct ggml_tensor* t2pre = ggml_mul(ctx, ggml_mul_mat(ctx, VVsT, wproj), bs);  // [D,group,K]
        struct ggml_tensor* t2p = ggml_cont(ctx, ggml_permute(ctx, t2pre, 1, 2, 0, 3)); // [K,D,group]
        struct ggml_tensor* term2 = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, t2p), D, group); // [D,group]

        // ── Dense value: Σ_t w_ds[t]·dvk[:,t] (+ w_cs·cvk if !ignore_c) ──
        struct ggml_tensor* dvkT  = ggml_cont(ctx, ggml_transpose(ctx, dvk));           // [MAXD,D]
        struct ggml_tensor* dense_out = ggml_mul_mat(ctx, dvkT, w_ds);                  // [D,group]
        struct ggml_tensor* final_dense = dense_out;
        if (!ignore_c) {
            struct ggml_tensor* cvk = ggml_cont(ctx, ggml_view_1d(ctx, v_cur, D, (size_t)kv*D*v_cur->nb[0])); // [D] current V
            struct ggml_tensor* cvk2  = ggml_reshape_2d(ctx, cvk, 1, D);                                     // [1,D]
            struct ggml_tensor* cur_out = ggml_mul_mat(ctx, cvk2, w_cs);                                     // [D,group]
            final_dense = ggml_add(ctx, dense_out, cur_out);
        }

        // DEBUG term toggles to localize the sparse magnitude bug.
        static const bool t1off = (std::getenv("DIFFKV_DBG_T1OFF") != nullptr);
        static const bool t2off = (std::getenv("DIFFKV_DBG_T2OFF") != nullptr);
        if (t1off) term1 = ggml_scale(ctx, term1, 0.0f);
        if (t2off) term2 = ggml_scale(ctx, term2, 0.0f);

        kv_outs[kv] = ggml_add(ctx, ggml_add(ctx, term1, term2), final_dense); // [D,group]
    }

    // Assemble [n_embd,1] from the per-head [D,group] outputs. Each kv block occupies a
    // contiguous [D*group] slice (flat index = (kv*group+g)*D + d). We reshape each kv_outs to
    // [D*group,1] (materialised via cont) and concat along dim0 — this yields a REAL [n_embd,1]
    // tensor, NOT a reshape-of-concat VIEW (whose source buffer the sched freed early → the
    // 113× output corruption on large-score inputs).
    struct ggml_tensor* out = ggml_reshape_2d(ctx, ggml_cont(ctx, kv_outs[0]), D*group, 1);
    for (int kv = 1; kv < nkv; ++kv)
        out = ggml_concat(ctx, out, ggml_reshape_2d(ctx, ggml_cont(ctx, kv_outs[kv]), D*group, 1), 0); // [n_embd,1]
    return ggml_cont(ctx, out);
}

// DEBUG (DIFFKV_DBG_CMP): capture layer-0 native q_rope + sparse attn_out so main can diff vs
// execute_cpu_attention in-process (definitive native-vs-CPU input check, immune to warmup noise).
static struct ggml_tensor* g_dbg_qrope = nullptr;
static struct ggml_tensor* g_dbg_attn0 = nullptr;
static struct ggml_tensor* g_dbg_sel0  = nullptr;
static struct ggml_tensor* g_dbg_curk  = nullptr;
static struct ggml_tensor* g_dbg_curv  = nullptr;

// Helper to build the Qwen 2.5 sparse decode forward pass graph with SRL routing and custom Metal attention
struct ggml_cgraph * build_decode_graph(
    struct ggml_context * ctx,
    DiffKVModel & model,
    struct ggml_tensor * input_token,
    struct ggml_tensor * position,
    struct ggml_tensor * W_proj,
    struct ggml_tensor * desc_matrix, // Layer 0 desc_matrix for SRL routing
    struct ggml_tensor * anchors_K,   // Layer 0 anchors_K for SRL routing
    struct ggml_tensor * slots_mask,  // slots_mask to ignore unoccupied slots
    struct ggml_tensor * host_slots,  // Host-computed candidate slots
    int srl_k_semantic,
    int srl_k_keep,
    CustomAttnUserData * userdata,     // Array of size config.n_layer!
    struct ggml_tensor ** out_logits,
    struct ggml_tensor ** out_selected_slots,
    struct ggml_tensor ** out_concat_k = nullptr,
    struct ggml_tensor ** out_concat_v = nullptr,
    bool use_sparse = true,
    int T_past = 0,
    int engage_threshold = 2048,
    struct ggml_tensor ** dense_k_past_inputs = nullptr,
    struct ggml_tensor ** dense_v_past_inputs = nullptr,
    struct ggml_tensor * dense_attn_mask = nullptr,
    struct ggml_tensor ** native_dense_kr = nullptr,   // [n_layer] each [F_kv, native_maxd] rotated past-dense K
    struct ggml_tensor ** native_dense_v = nullptr,    // [n_layer] each [F_kv, native_maxd] past-dense V
    struct ggml_tensor * native_dense_mask = nullptr,  // [native_maxd] validity bias
    int native_maxd = 0,
    struct ggml_tensor * native_dup_tri = nullptr,     // [srl_k_keep,srl_k_keep] strict-lower ones
    struct ggml_tensor * native_half = nullptr,        // [1] = 0.5
    struct ggml_tensor * native_dense_pos = nullptr,   // [native_maxd] i32 past-dense positions
    struct ggml_tensor ** out_dbg_anc = nullptr        // DEBUG: layer-0 anchor scores [K,group]
) {
    const auto & config = model.get_config();
    // Native sparse-attn adds ~40 ggml ops/layer; bump graph capacity well past the
    // 2048 default when it's enabled (the larger node arena is otherwise harmless).
    static const bool native_graph = is_native_attn_enabled();
    struct ggml_cgraph * gf = native_graph
        ? ggml_new_graph_custom(ctx, 32768, false)
        : ggml_new_graph(ctx);

    int F_test = config.n_embd / config.n_head * config.n_head_kv;
    struct ggml_tensor * concat_k = nullptr;
    struct ggml_tensor * concat_v = nullptr;
    if (out_concat_k) {
        concat_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F_test, config.n_layer);
        *out_concat_k = concat_k;
    }
    if (out_concat_v) {
        concat_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F_test, config.n_layer);
        *out_concat_v = concat_v;
    }

    // 1. Embedding lookup: shape [n_embd, 1]
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.get_token_embd(), input_token);

    struct ggml_tensor * selected_slots = nullptr;

    for (int l = 0; l < config.n_layer; ++l) {
        const auto & layer = model.get_layers()[l];

        // 2. Attention RMSNorm
        struct ggml_tensor * h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.attn_norm);

        // 3. QKV Projections
        struct ggml_tensor * q = ggml_mul_mat(ctx, layer.wq, h);
        if (layer.bq) q = ggml_add(ctx, q, layer.bq);
        struct ggml_tensor * k = ggml_mul_mat(ctx, layer.wk, h);
        if (layer.bk) k = ggml_add(ctx, k, layer.bk);
        struct ggml_tensor * v = ggml_mul_mat(ctx, layer.wv, h);
        if (layer.bv) v = ggml_add(ctx, v, layer.bv);

        // Export raw key/value tensors of each layer (before RoPE)
        if (concat_k) {
            struct ggml_tensor * k_view = ggml_view_1d(ctx, concat_k, F_test, l * F_test * sizeof(float));
            struct ggml_tensor * copy_k = ggml_cpy(ctx, k, k_view);
            ggml_build_forward_expand(gf, copy_k);
        }
        if (concat_v) {
            struct ggml_tensor * v_view = ggml_view_1d(ctx, concat_v, F_test, l * F_test * sizeof(float));
            struct ggml_tensor * copy_v = ggml_cpy(ctx, v, v_view);
            ggml_build_forward_expand(gf, copy_v);
        }

        struct ggml_tensor * attn_out = nullptr;

        if (use_sparse) {
            // ── SRL Routing Pipeline at Layer 0 ──
            if (l == 0) {
                int head_dim = config.n_embd / config.n_head;
                // Reshape Q: [896, 1] -> [head_dim, n_head] = [64, 14]
                struct ggml_tensor * Q = ggml_reshape_2d(ctx, q, head_dim, config.n_head);

                // Compute query descriptor: [desc_dim, 1]
                struct ggml_tensor * q_desc = compute_query_desc(ctx, Q, W_proj);

                // Semantic search topk: [srl_k_semantic, 1]
                struct ggml_tensor * sem_slots = semantic_search_topk(ctx, q_desc, desc_matrix, slots_mask, srl_k_semantic);
                struct ggml_tensor * sem_slots_1d = ggml_reshape_1d(ctx, sem_slots, srl_k_semantic);

                // Concatenate semantic and host slots
                struct ggml_tensor * candidate_slots = ggml_concat(ctx, sem_slots_1d, host_slots, 0);

                // Anchor screening: [srl_k_keep]
                float scale = 1.0f / std::sqrt((float)head_dim);
                selected_slots = ggml_cont(ctx, anchor_screen(ctx, Q, anchors_K, candidate_slots, scale, srl_k_keep));

                // Save selected slots to out parameter
                if (out_selected_slots) {
                    *out_selected_slots = selected_slots;
                }

                // Make sure the selected slots are computed in the graph
                ggml_build_forward_expand(gf, selected_slots);
            }

            // Apply RoPE to Q for the custom Metal kernel!
            int head_dim = config.n_embd / config.n_head;
            struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, head_dim, config.n_head, 1);
            struct ggml_tensor * q_rope = ggml_rope_ext(ctx, q_reshaped, position, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            struct ggml_tensor * q_rope_flat = ggml_reshape_1d(ctx, q_rope, config.n_embd);

            // Native ggml-metal sparse attention path (gated). Decided at graph-build time:
            // the factual store is populated during prefill, so if it's empty we can run the
            // fully-native fused subgraph; factual/NIAH sessions keep the CPU custom op (which
            // also drives VSL masking + logit biasing). Static graph → decision is per-build.
            static const bool native_attn_env = is_native_attn_enabled();
            bool factual_empty = true;
            if (userdata && userdata[l].srl_state) {
                factual_empty = static_cast<diffkv::SessionSRLState*>(userdata[l].srl_state)->factual_store.entries.empty();
            }
            bool use_native_attn = native_attn_env && factual_empty &&
                                   userdata && userdata[l].kv_engine &&
                                   userdata[l].kv_engine->native_attn_enabled();

            if (use_native_attn && selected_slots && native_dense_kr && native_dense_v && native_dense_mask) {
                // Current-token key rotated at the current position (dense self-entry).
                struct ggml_tensor * k_reshaped_n = ggml_reshape_3d(ctx, k, head_dim, config.n_head_kv, 1);
                struct ggml_tensor * k_rope_n = ggml_rope_ext(ctx, k_reshaped_n, position, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                // Past-dense keys are uploaded RAW; RoPE them in-graph at their actual positions.
                struct ggml_tensor * dkr3 = ggml_reshape_3d(ctx, native_dense_kr[l], head_dim, config.n_head_kv, native_maxd);
                struct ggml_tensor * dkr_roped = ggml_rope_ext(ctx, dkr3, native_dense_pos, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                struct ggml_tensor * dkr_flat = ggml_reshape_2d(ctx, dkr_roped, head_dim * config.n_head_kv, native_maxd);
                if (std::getenv("DIFFKV_DBG_NOROPE")) dkr_flat = native_dense_kr[l]; // DEBUG: bypass dense rope
                attn_out = build_native_sparse_attn(
                    ctx, q_rope, k_rope_n, v, dkr_flat, native_dense_v[l], native_dense_mask, native_maxd,
                    selected_slots, native_dup_tri, native_half, userdata[l].kv_engine,
                    config.n_head, config.n_head_kv, head_dim,
                    userdata[l].kv_engine->get_rank(), 64, srl_k_keep,
                    1.0f / std::sqrt((float)head_dim),
                    userdata[l].ignore_c,
                    nullptr
                );
                { const char* cl = std::getenv("DIFFKV_DBG_CMP_LAYER"); int cmpL = cl ? atoi(cl) : 0;
                  if (l == cmpL && std::getenv("DIFFKV_DBG_CMP")) {
                    g_dbg_qrope = q_rope; g_dbg_attn0 = attn_out; g_dbg_sel0 = selected_slots;
                    g_dbg_curk = k; g_dbg_curv = v;
                    ggml_set_output(q_rope); ggml_set_output(attn_out); ggml_set_output(selected_slots);
                    ggml_set_output(k); ggml_set_output(v);
                } }
            } else if (userdata && selected_slots) {
                struct ggml_tensor * kv_concat = ggml_concat(ctx, k, v, 0);
                // Reconstruct attention output using the custom Metal kernel!
                struct ggml_tensor * custom_attn = ggml_map_custom3(
                    ctx, q_rope_flat, selected_slots, kv_concat,
                    custom_attention_op_callback, 1, &userdata[l]
                );
                attn_out = custom_attn;
            } else {
                // Fallback placeholder attention
                struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, config.n_embd / config.n_head, 1, config.n_head_kv);
                int group_size = config.n_head / config.n_head_kv;
                struct ggml_tensor * target_repeat = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.n_embd / config.n_head, group_size, config.n_head_kv);
                struct ggml_tensor * v_repeated = ggml_repeat(ctx, v_reshaped, target_repeat);
                attn_out = ggml_reshape_1d(ctx, v_repeated, config.n_embd);
            }
        } else {
            // ── Dense attention fast-path with ggml_flash_attn_ext! ──
            int head_dim_val = config.n_embd / config.n_head;

            // Create the large past input tensors in the graph context
            if (dense_k_past_inputs && dense_v_past_inputs) {
                if (dense_k_past_inputs[l] == nullptr) {
                    dense_k_past_inputs[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim_val, config.n_head_kv, engage_threshold);
                    ggml_set_input(dense_k_past_inputs[l]);
                }
                if (dense_v_past_inputs[l] == nullptr) {
                    dense_v_past_inputs[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim_val, config.n_head_kv, engage_threshold);
                    ggml_set_input(dense_v_past_inputs[l]);
                }
            }

            struct ggml_tensor * k_past = dense_k_past_inputs[l];
            struct ggml_tensor * v_past = dense_v_past_inputs[l];

            // Permute current token
            struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, head_dim_val, config.n_head, 1);
            struct ggml_tensor * k_reshaped = ggml_reshape_3d(ctx, k, head_dim_val, config.n_head_kv, 1);

            struct ggml_tensor * q_rope = ggml_rope_ext(ctx, q_reshaped, position, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            struct ggml_tensor * k_rope = ggml_rope_ext(ctx, k_reshaped, position, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            struct ggml_tensor * q_perm = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            struct ggml_tensor * k_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
            struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, head_dim_val, config.n_head_kv, 1);
            struct ggml_tensor * v_perm = ggml_permute(ctx, v_reshaped, 0, 2, 1, 3);

            // Permute past
            struct ggml_tensor * pk = ggml_permute(ctx, k_past, 0, 2, 1, 3);
            struct ggml_tensor * pv = ggml_permute(ctx, v_past, 0, 2, 1, 3);

            // Concat
            struct ggml_tensor * k_ctx_perm = ggml_concat(ctx, pk, k_perm, 1);
            struct ggml_tensor * v_ctx_perm = ggml_concat(ctx, pv, v_perm, 1);

            // Flash attention
            float scale_val = 1.0f / std::sqrt((float)head_dim_val);
            struct ggml_tensor * attn_out_perm = ggml_flash_attn_ext(ctx, q_perm, k_ctx_perm, v_ctx_perm, dense_attn_mask, scale_val, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out_perm, GGML_PREC_F32);

            attn_out = ggml_reshape_2d(ctx, attn_out_perm, config.n_embd, 1);
        }

        // 5. Output Projection (WO)
        struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer.wo, attn_out);
        if (layer.bo) attn_proj = ggml_add(ctx, attn_proj, layer.bo);

        // 6. Residual connection
        cur = ggml_add(ctx, cur, attn_proj);

        // 7. FFN RMSNorm
        h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.ffn_norm);

        // 8. FFN SwiGLU: (SiLU(gate) * up) * down
        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, h);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up, h);

        struct ggml_tensor * gate_silu = ggml_silu(ctx, gate);
        struct ggml_tensor * ffn_out   = ggml_mul(ctx, gate_silu, up);
        struct ggml_tensor * ffn_proj  = ggml_mul_mat(ctx, layer.ffn_down, ffn_out);

        // 9. Residual connection
        cur = ggml_add(ctx, cur, ffn_proj);
    }

    // 10. Final RMSNorm
    struct ggml_tensor * final_norm = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
    final_norm = ggml_mul(ctx, final_norm, model.get_output_norm());

    // 11. LM Head Output
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.get_output(), final_norm);

    *out_logits = logits;
    ggml_build_forward_expand(gf, logits);

    if (out_selected_slots && *out_selected_slots) {
        struct ggml_tensor * dummy_slots = ggml_view_1d(ctx, *out_selected_slots, (*out_selected_slots)->ne[0], 0);
        ggml_build_forward_expand(gf, dummy_slots);
    }

    if (out_concat_k && *out_concat_k) {
        struct ggml_tensor * dummy_concat_k = ggml_view_1d(ctx, *out_concat_k, (*out_concat_k)->ne[0] * (*out_concat_k)->ne[1], 0);
        ggml_build_forward_expand(gf, dummy_concat_k);
    }
    if (out_concat_v && *out_concat_v) {
        struct ggml_tensor * dummy_concat_v = ggml_view_1d(ctx, *out_concat_v, (*out_concat_v)->ne[0] * (*out_concat_v)->ne[1], 0);
        ggml_build_forward_expand(gf, dummy_concat_v);
    }

    return gf;
}

bool verify_attention_cpu(
    const float* q_data,              // [n_q_heads * D]
    const int32_t* slots,             // [K]
    const float* metal_output,        // [n_q_heads * D]
    NativeBlockPool* kv_engine,
    int n_q_heads, int n_kv_heads, int rank, int S_max, int K, int D, float scale
) {
    // Read pool tensors to host for reference calculation
    std::vector<int8_t> U(ggml_nelements(kv_engine->get_U()));
    ggml_backend_tensor_get(kv_engine->get_U(), U.data(), 0, U.size() * sizeof(int8_t));

    std::vector<ggml_fp16_t> U_scale(ggml_nelements(kv_engine->get_U_scale()));
    ggml_backend_tensor_get(kv_engine->get_U_scale(), U_scale.data(), 0, U_scale.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> VK(ggml_nelements(kv_engine->get_VK()));
    ggml_backend_tensor_get(kv_engine->get_VK(), VK.data(), 0, VK.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> VV(ggml_nelements(kv_engine->get_VV()));
    ggml_backend_tensor_get(kv_engine->get_VV(), VV.data(), 0, VV.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> anchors_K(ggml_nelements(kv_engine->get_anchors_K()));
    ggml_backend_tensor_get(kv_engine->get_anchors_K(), anchors_K.data(), 0, anchors_K.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> anchors_V(ggml_nelements(kv_engine->get_anchors_V()));
    ggml_backend_tensor_get(kv_engine->get_anchors_V(), anchors_V.data(), 0, anchors_V.size() * sizeof(ggml_fp16_t));

    std::vector<int32_t> seq_lens(ggml_nelements(kv_engine->get_seq_lens()));
    ggml_backend_tensor_get(kv_engine->get_seq_lens(), seq_lens.data(), 0, seq_lens.size() * sizeof(int32_t));

    std::vector<ggml_fp16_t> scales(ggml_nelements(kv_engine->get_scales()));
    ggml_backend_tensor_get(kv_engine->get_scales(), scales.data(), 0, scales.size() * sizeof(ggml_fp16_t));

    std::vector<float> cpu_output(n_q_heads * D, 0.0f);
    const int g = n_q_heads / n_kv_heads;

    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;

        float max_score = -1e30f;
        
        struct SlotScoreInfo {
            float anchor_score;
            std::vector<float> token_scores;
            std::vector<float> q_proj;
        };
        std::vector<SlotScoreInfo> slot_infos(K);

        for (int k = 0; k < K; ++k) {
            int slot_id = slots[k];
            int slen = seq_lens[slot_id];
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);

            // 1. Anchor score
            float score_anc = 0.0f;
            for (int d = 0; d < D; ++d) {
                float q_val = q_data[h * D + d];
                float ak_val = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
                score_anc += q_val * ak_val;
            }
            slot_infos[k].anchor_score = score_anc;

            float s_anc_scaled = score_anc * scale;
            if (s_anc_scaled > max_score) max_score = s_anc_scaled;

            // 2. Query projection
            std::vector<float> q_proj(rank, 0.0f);
            for (int r = 0; r < rank; ++r) {
                float proj = 0.0f;
                int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    float q_val = q_data[h * D + d];
                    float vk_val = ggml_fp16_to_fp32(VK[base_vk_offset + d]);
                    proj += q_val * vk_val;
                }
                q_proj[r] = proj;
            }
            slot_infos[k].q_proj = q_proj;

            // 3. Token scores
            slot_infos[k].token_scores.resize(slen);
            for (int t = 0; t < slen; ++t) {
                float delta_sum = 0.0f;
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    delta_sum += q_proj[r] * static_cast<float>(U[u_offset + r]);
                }
                float t_score = (delta_sum * scale_u * block_scale + score_anc) * scale;
                slot_infos[k].token_scores[t] = t_score;
                if (t_score > max_score) max_score = t_score;
            }
        }

        // Compute softmax denominator
        double sum_exp = 0.0;
        for (int k = 0; k < K; ++k) {
            sum_exp += std::exp(slot_infos[k].anchor_score * scale - max_score);
            for (float s : slot_infos[k].token_scores) {
                sum_exp += std::exp(s - max_score);
            }
        }

        // Pass 2: Accumulate values
        std::vector<double> accum_val(D, 0.0);
        for (int k = 0; k < K; ++k) {
            int slot_id = slots[k];
            int slen = seq_lens[slot_id];
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);

            double w_anc = std::exp(slot_infos[k].anchor_score * scale - max_score) / sum_exp;
            double sum_w_tokens = 0.0;
            std::vector<double> w_proj(rank, 0.0);

            for (int t = 0; t < slen; ++t) {
                double w_t = std::exp(slot_infos[k].token_scores[t] - max_score) / sum_exp;
                sum_w_tokens += w_t;
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    w_proj[r] += w_t * static_cast<float>(U[u_offset + r]) * scale_u;
                }
            }

            double w_total_anc = w_anc + sum_w_tokens;

            for (int d = 0; d < D; ++d) {
                double av_val = ggml_fp16_to_fp32(anchors_V[slot_id * n_kv_heads * D + kv_head * D + d]);
                accum_val[d] += w_total_anc * av_val;

                double svd_v_contrib = 0.0;
                int base_vv_offset = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                for (int r = 0; r < rank; ++r) {
                    double vv_val = ggml_fp16_to_fp32(VV[base_vv_offset + r * n_kv_heads * D]);
                    svd_v_contrib += w_proj[r] * vv_val;
                }
                accum_val[d] += svd_v_contrib * block_scale;
            }
        }

        for (int d = 0; d < D; ++d) {
            cpu_output[h * D + d] = static_cast<float>(accum_val[d]);
        }
    }

    float max_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        float diff = std::abs(cpu_output[i] - metal_output[i]);
        if (diff > max_diff) max_diff = diff;
        sum_sq_diff += diff * diff;
    }
    float rmse = std::sqrt(sum_sq_diff / cpu_output.size());
    std::printf("[Verification] CPU vs Metal Max Diff: %e, RMSE: %e\n", max_diff, rmse);
    
    if (max_diff < 1e-4f) {
        std::printf("[Verification] SUCCESS: Metal attention matches CPU reference!\n");
        return true;
    } else {
        std::printf("[Verification] FAILURE: Metal attention does not match CPU reference!\n");
        return false;
    }
}

// Helper function to sample from logits
int32_t sample_logits(const std::vector<float>& logits, float temp, float top_p, std::mt19937& rng) {
    if (logits.empty()) return 0;
    if (temp <= 0.01f) {
        // Greedy argmax
        float max_logit = -1e30f;
        int32_t best_idx = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Apply temperature scaling
    std::vector<double> probs(logits.size());
    double max_logit = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((double)(logits[i] - max_logit) / temp);
        sum += probs[i];
    }
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }

    if (top_p < 1.0f) {
        // Sort indices based on probabilities
        std::vector<size_t> indices(probs.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return probs[a] > probs[b];
        });

        double cum_prob = 0.0;
        bool cut = false;
        for (size_t idx : indices) {
            if (cut) {
                probs[idx] = 0.0;
            } else {
                cum_prob += probs[idx];
                if (cum_prob > top_p) {
                    cut = true;
                }
            }
        }
        // Renormalize
        sum = 0.0;
        for (double p : probs) {
            sum += p;
        }
        if (sum > 0.0) {
            for (size_t i = 0; i < probs.size(); ++i) {
                probs[i] /= sum;
            }
        } else {
            // Fallback: assign 1.0 to the top index
            std::fill(probs.begin(), probs.end(), 0.0);
            probs[indices[0]] = 1.0;
        }
    }

    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

namespace diffkv {
// External (non-static) reference attention used by the self-test.
void execute_cpu_attention(const float*, const int32_t*, float*, float*, NativeBlockPool*,
                           int,int,int,int,int,int,float,bool,float,bool);
void cpu_dense_attention(const float*, const float*, const float*, const int32_t*,
                         int,int,int,int,float,bool,float,int,float*,float*);
}

// DIFFKV_SELFTEST=1: standalone unit test of build_native_sparse_attn vs execute_cpu_attention
// with tiny KNOWN inputs on the CPU backend. Each graph tensor gets its OWN buffer (no sched
// reuse), so a match ⇒ the subgraph MATH is correct (bug is sched buffer-reuse in the full model);
// a mismatch ⇒ a math/op bug, localized here without 24-layer / pool-timing / capture noise.
static void run_native_attn_selftest() {
    using namespace diffkv;
    const int n_slots=4, rank=4, D=8, nkv=2, desc_dim=8, nq=4, K=3, S_max=64, MAXD=8;
    const int Tpast=3;                  // past-dense tokens (current token added separately)
    const float scale = 1.0f/std::sqrt((float)D), freq=1000000.0f;
    setenv("DIFFKV_NATIVE_ATTN","1",1); // pool allocates VK_rot/anchorK_rot/valid_mask/U_f16
    // FULL path test (sparse ∪ dense ∪ current). has_rope=false → no rotation to match.

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    NativeBlockPool pool;
    pool.initialize(n_slots, rank, D, nkv, desc_dim, buft);
    const bool HR = (std::getenv("DIFFKV_SELFTEST_ROPE") != nullptr); // test WITH rotation
    pool.set_rope_config(HR, freq);
    pool.zero_all_tensors();

    std::mt19937 g(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto H=[&](float f){return ggml_fp32_to_fp16(f);};
    for (int s=0;s<K;++s){
        int8_t* U=pool.get_host_U(); ggml_fp16_t* VK=pool.get_host_VK(); ggml_fp16_t* VV=pool.get_host_VV();
        ggml_fp16_t* aK=pool.get_host_anchors_K(); ggml_fp16_t* aV=pool.get_host_anchors_V();
        int32_t* sl=pool.get_host_seq_lens(); ggml_fp16_t* sc=pool.get_host_scales();
        ggml_fp16_t* us=pool.get_host_U_scale(); int32_t* ap=pool.get_host_anchor_positions();
        int seqlen = 20 + s*10;  // longer blocks
        sl[s]=seqlen; sc[s]=H(0.3f+0.1f*s); us[s]=H(0.2f); ap[s]=10+s*7;
        // EXTREME data to reproduce the real failure: massive anchor-K (|aK|~per-elem 30, like
        // Qwen layer-0), and slots 1,2 NEAR-IDENTICAL to slot 0 (mimics repetitive-prompt blocks).
        int src = (s==0)?0:0; (void)src;
        for(int t=0;t<seqlen;++t) for(int r=0;r<rank;++r) U[(size_t)s*S_max*rank + t*rank + r]=(int8_t)((int)((g()+s*7)%21)-10);
        for(int r=0;r<rank;++r) for(int kv=0;kv<nkv;++kv) for(int d=0;d<D;++d){
            VK[(size_t)s*rank*nkv*D + r*nkv*D + kv*D + d]=H(dist(g)*0.5f + (s>0?0.02f*s:0.0f));
            VV[(size_t)s*rank*nkv*D + r*nkv*D + kv*D + d]=H(dist(g)*0.5f);
        }
        for(int kv=0;kv<nkv;++kv) for(int d=0;d<D;++d){
            aK[(size_t)s*nkv*D + kv*D + d]=H(dist(g)*64.0f + (s>0?0.5f*s:0.0f));  // massive K + near-dup
            aV[(size_t)s*nkv*D + kv*D + d]=H(dist(g));
        }
        pool.upload_slot(s);
    }

    std::vector<float> Q(nq*D); for(auto&x:Q)x=dist(g);
    std::vector<int32_t> slots={0,1,2};
    const int F=nkv*D, Tdense=Tpast+1;
    std::vector<float> pastK((size_t)Tpast*F), pastV((size_t)Tpast*F), curK(F), curV(F);
    for(auto&x:pastK)x=dist(g)*0.4f; for(auto&x:pastV)x=dist(g)*0.4f;
    for(auto&x:curK)x=dist(g)*0.4f; for(auto&x:curV)x=dist(g)*0.4f;

    // ── Reference: sparse + dense(Tpast+current) + 3-way LSE combine (has_rope=false) ──
    std::vector<float> outS(nq*D,0.0f), lseS(nq,-1e30f), outD(nq*D,0.0f), lseD(nq,-1e30f);
    execute_cpu_attention(Q.data(), slots.data(), outS.data(), lseS.data(), &pool, nq, nkv, rank, S_max, K, D, scale, HR, freq, true);
    std::vector<float> denseK((size_t)Tdense*F), denseV((size_t)Tdense*F); std::vector<int32_t> dpos(Tdense);
    std::memcpy(denseK.data(), pastK.data(), pastK.size()*4); std::memcpy(denseK.data()+(size_t)Tpast*F, curK.data(), F*4);
    std::memcpy(denseV.data(), pastV.data(), pastV.size()*4); std::memcpy(denseV.data()+(size_t)Tpast*F, curV.data(), F*4);
    const int curpos=Tpast; for(int t=0;t<Tdense;++t)dpos[t]=t;  // past 0..Tpast-1, current=Tpast
    cpu_dense_attention(Q.data(), denseK.data(), denseV.data(), dpos.data(), Tdense, nq, nkv, D, scale, HR, freq, 0, outD.data(), lseD.data());
    std::vector<float> out_ref(nq*D);
    for(int h=0;h<nq;++h){ double lmax=std::max(lseS[h],lseD[h]); double ws=(lseS[h]<=-1e20?0.0:std::exp(lseS[h]-lmax)), wd=(lseD[h]<=-1e20?0.0:std::exp(lseD[h]-lmax)); double den=std::max(ws+wd,1e-9); for(int d=0;d<D;++d) out_ref[h*D+d]=(float)((outS[h*D+d]*ws+outD[h*D+d]*wd)/den); }

    // ── Native graph (FULL path, no DENSEOFF) ──
    ggml_init_params ip={ (size_t)16*1024*1024, nullptr, true };
    ggml_context* ctx=ggml_init(ip);
    ggml_tensor* q_rope=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,nq,1); ggml_set_input(q_rope);
    ggml_tensor* k_rope=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,nkv,1); ggml_set_input(k_rope);
    ggml_tensor* v_cur=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,F); ggml_set_input(v_cur);
    ggml_tensor* dkr=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,F,MAXD); ggml_set_input(dkr);
    ggml_tensor* dv=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,F,MAXD); ggml_set_input(dv);
    ggml_tensor* dmask=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,MAXD); ggml_set_input(dmask);
    ggml_tensor* sel=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,K); ggml_set_input(sel);
    ggml_tensor* tri=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,K); ggml_set_input(tri);
    ggml_tensor* half=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1); ggml_set_input(half);
    ggml_tensor* dpos_t=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,MAXD); ggml_set_input(dpos_t);
    ggml_tensor* cpos_t=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1); ggml_set_input(cpos_t);
    // With rotation: rope the dense K (raw) and current K in-graph, exactly as build_decode_graph does.
    ggml_tensor* dkr_use=dkr, *krope_use=k_rope;
    if (HR) {
        ggml_tensor* d3=ggml_reshape_3d(ctx,dkr,D,nkv,MAXD);
        dkr_use=ggml_reshape_2d(ctx, ggml_rope_ext(ctx,d3,dpos_t,nullptr,D,GGML_ROPE_TYPE_NEOX,0,freq,1.0f,0.0f,1.0f,0.0f,0.0f), F, MAXD);
        krope_use=ggml_rope_ext(ctx, k_rope, cpos_t, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, freq, 1.0f,0.0f,1.0f,0.0f,0.0f);
    }
    ggml_tensor* out = build_native_sparse_attn(ctx, q_rope, krope_use, v_cur, dkr_use, dv, dmask, MAXD,
                                                sel, tri, half, &pool, nq, nkv, D, rank, S_max, K, scale,
                                                false, nullptr);
    ggml_set_output(out);
    ggml_cgraph* gf=ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> dkrbuf((size_t)F*MAXD,0.0f), dvbuf((size_t)F*MAXD,0.0f), dmbuf(MAXD,-INFINITY), trih((size_t)K*K,0.0f);
    std::memcpy(dkrbuf.data(), pastK.data(), pastK.size()*4); std::memcpy(dvbuf.data(), pastV.data(), pastV.size()*4);
    for(int t=0;t<Tpast;++t)dmbuf[t]=0.0f;
    for(int kk=0;kk<K;++kk) for(int j=0;j<kk;++j) trih[j+kk*K]=1.0f;
    float halfv=0.5f;
    ggml_backend_tensor_set(q_rope,Q.data(),0,Q.size()*4);
    ggml_backend_tensor_set(k_rope,curK.data(),0,F*4);
    ggml_backend_tensor_set(v_cur,curV.data(),0,F*4);
    ggml_backend_tensor_set(dkr,dkrbuf.data(),0,dkrbuf.size()*4);
    ggml_backend_tensor_set(dv,dvbuf.data(),0,dvbuf.size()*4);
    ggml_backend_tensor_set(dmask,dmbuf.data(),0,dmbuf.size()*4);
    ggml_backend_tensor_set(sel,slots.data(),0,slots.size()*4);
    ggml_backend_tensor_set(tri,trih.data(),0,trih.size()*4);
    ggml_backend_tensor_set(half,&halfv,0,4);
    std::vector<int32_t> dposb(MAXD,0); for(int t=0;t<Tpast;++t)dposb[t]=dpos[t]; int32_t cposv=curpos;
    ggml_backend_tensor_set(dpos_t,dposb.data(),0,dposb.size()*4);
    ggml_backend_tensor_set(cpos_t,&cposv,0,4);
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> nat(nq*D); ggml_backend_tensor_get(out,nat.data(),0,nat.size()*4);

    double maxd=0,rn=0,nn=0; for(int i=0;i<nq*D;++i){ double d=std::abs((double)nat[i]-out_ref[i]); maxd=std::max(maxd,d); rn+=(double)out_ref[i]*out_ref[i]; nn+=(double)nat[i]*nat[i]; }
    std::cerr<<"[SELFTEST] |native|="<<std::sqrt(nn)<<" |ref|="<<std::sqrt(rn)<<" maxAbsDiff="<<maxd<<(maxd<1e-2*std::sqrt(rn)+1e-3?"  PASS":"  FAIL")<<std::endl;
    std::cerr<<"[SELFTEST] native[0..7]="; for(int i=0;i<8;++i)std::cerr<<nat[i]<<" "; std::cerr<<"\n[SELFTEST] ref   [0..7]="; for(int i=0;i<8;++i)std::cerr<<out_ref[i]<<" "; std::cerr<<std::endl;
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    if (std::getenv("DIFFKV_SELFTEST")) { run_native_attn_selftest(); return 0; }
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_model_path> [prompt]" << std::endl;
        return 1;
    }

    // ── Initialize Best Backend (GPU / CPU) ──────────────────────────────────
    ggml_backend_owner backend_owner;
    if (!backend_owner.sched) {
        std::cerr << "Failed to initialize backend scheduler!" << std::endl;
        return 1;
    }
    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
        std::cerr << "[DiffKV Native] Initialized scheduler with backends: GPU=" 
                  << (backend_owner.gpu_backend ? ggml_backend_name(backend_owner.gpu_backend) : "none") 
                  << ", CPU=" << (backend_owner.cpu_backend ? ggml_backend_name(backend_owner.cpu_backend) : "none") << std::endl;
    }

    ggml_backend_t backend = backend_owner.gpu_backend;
    ggml_backend_sched_t sched = backend_owner.sched;

    std::string model_path = argv[1];
    diffkv::DiffKVModel model;

    if (!model.load_from_file(model_path, backend)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    model.print_info();
    
    // ── SRL configuration (with defaults) ───────────────────────────────────
    int srl_k_semantic = 32;
    int srl_k_lexical = 8;
    int srl_k_graph = 8;
    int srl_k_recency = 8;
    int srl_k_keep = 16;

    // Load from env vars
    if (const char* env = std::getenv("DIFFKV_SRL_K_SEM")) srl_k_semantic = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_LEX")) srl_k_lexical = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_GRAPH")) srl_k_graph = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_RECENCY")) srl_k_recency = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_KEEP")) srl_k_keep = std::stoi(env);

    // Also parse from argv
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--srl-k-semantic" && i + 1 < argc) {
            srl_k_semantic = std::stoi(argv[++i]);
        } else if (arg == "--srl-k-lexical" && i + 1 < argc) {
            srl_k_lexical = std::stoi(argv[++i]);
        } else if (arg == "--srl-k-graph" && i + 1 < argc) {
            srl_k_graph = std::stoi(argv[++i]);
        } else if (arg == "--srl-k-recency" && i + 1 < argc) {
            srl_k_recency = std::stoi(argv[++i]);
        } else if (arg == "--srl-k-keep" && i + 1 < argc) {
            srl_k_keep = std::stoi(argv[++i]);
        }
    }

    int srl_k_host = 1 + srl_k_recency + srl_k_lexical + srl_k_semantic + srl_k_graph;

    std::unordered_set<int32_t> stop_token_ids;
    {
        std::vector<std::string> stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "need",
            "to", "of", "in", "on", "at", "by", "for", "with", "as",
            "and", "or", "but", "if", "then", "that", "this", "it",
            "he", "she", "they", "we", "you", "i", "not", "no",
            ",", ".", ":", ";", "?", "!", "(", ")", "'", "\"", "-", "\n",
            "system", "user", "assistant", "im_start", "im_end"
        };
        for (const auto & word : stop_words) {
            auto t = model.tokenize(word, false);
            for (int32_t tok : t) {
                stop_token_ids.insert(tok);
            }
        }
        for (int i = 0; i < 200; ++i) {
            stop_token_ids.insert(i);
        }
    }

    diffkv::SessionSRLState srl_state;
    // KV prefix cache: tracks how many tokens the binary has already ingested
    // in its KV pool for the current session, matching ACTIVE_RUNTIME's cached_len mechanism.
    int session_cached_len = 0;   // token count already in KV pool (skip re-prefilling these)
    std::vector<int32_t> session_cached_token_ids; // the prefix we verified is resident


    // Backend is already initialized at the start of main
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Initialize NativeBlockPool block pool for all layers
    int n_slots = model.get_config().n_ctx / 64;
    if (const char* env_slots = std::getenv("DIFFKV_MAX_CONTEXT_SLOTS")) {
        n_slots = std::stoi(env_slots);
        std::cerr << "[DiffKV Native] Overriding n_slots from DIFFKV_MAX_CONTEXT_SLOTS: " << n_slots << std::endl;
    }
    srl_k_semantic = std::min(srl_k_semantic, n_slots);
    srl_k_keep = std::min(srl_k_keep, n_slots);
    int rank = 32;
    int head_dim = model.get_config().n_embd / model.get_config().n_head;
    int kv_heads = model.get_config().n_head_kv;
    int desc_dim = 64;
    int n_vocab = model.get_config().n_vocab;
    int n_layers = model.get_config().n_layer;

    int micro_block_size = 64;
    if (const char* env_mbs = std::getenv("DIFFKV_MICRO_BLOCK_SIZE")) {
        micro_block_size = std::stoi(env_mbs);
    }
    float gpu_budget_gb = 2.0f;
    if (const char* env_budget = std::getenv("DIFFKV_GPU_BUDGET_GB")) {
        gpu_budget_gb = std::stof(env_budget);
    }
    size_t gpu_budget_bytes = static_cast<size_t>(gpu_budget_gb * 1024.0f * 1024.0f * 1024.0f);

    diffkv::KVRuntimeManager runtime_manager(rank, micro_block_size, gpu_budget_bytes);
    if (!runtime_manager.initialize(n_slots, head_dim, kv_heads, desc_dim, n_layers, &model, buft)) {
        std::cerr << "Failed to initialize KVRuntimeManager!" << std::endl;
        return 1;
    }
    runtime_manager.get_ingest_manager().set_stop_token_ids(&stop_token_ids);
    runtime_manager.get_ingest_manager().set_session_id("interactive_session");
    auto & kv_engines = runtime_manager.get_engines();

    // Initialize W_proj on host
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> W_proj_host(head_dim * desc_dim);
    for (int r = 0; r < desc_dim; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < head_dim; ++c) {
            float val = dist(gen);
            W_proj_host[r * head_dim + c] = val;
            sum_sq += val * val;
        }
        float norm = std::sqrt(sum_sq) + 1e-8f;
        for (int c = 0; c < head_dim; ++c) {
            W_proj_host[r * head_dim + c] /= norm;
        }
    }

    // Clear pool tensors for all layers
    for (int l = 0; l < n_layers; ++l) {
        // Supply RoPE params for the native-attn precomputed-rotation path (no-op when gated off).
        kv_engines[l]->set_rope_config(true, model.get_config().rope_freq_base);
        kv_engines[l]->zero_all_tensors();
    }

    // Reference SVD compressor from runtime_manager
    auto & compressor = runtime_manager.get_compressor();

    // Allocate persistent dense vectors and tables
    int F_test = kv_heads * head_dim;
    std::vector<diffkv::AlignedFloatVector> active_k_dense(n_layers, diffkv::AlignedFloatVector(16384 * F_test, 0.0f));
    std::vector<diffkv::AlignedFloatVector> active_k_dense_rotated(n_layers, diffkv::AlignedFloatVector(16384 * F_test, 0.0f));
    std::vector<diffkv::AlignedFloatVector> active_v_dense(n_layers, diffkv::AlignedFloatVector(16384 * F_test, 0.0f));
    diffkv::AlignedInt32Vector active_positions_dense(16384, 0);
    int total_positions = 0;
    std::map<int, std::vector<float>> persistent_k_dense;
    std::map<int, std::vector<float>> persistent_v_dense;
    std::vector<std::vector<int32_t>> seq_lens_by_layer(n_layers, std::vector<int32_t>(n_slots, 0));


    // Persistent raw K/V activations (kept per-session so continuation turns can
    // skip the already-compressed prefix and only prefill new tokens)
    // Sized to max context (n_slots * 64 tokens) at runtime below.



    bool interactive = (argc < 3 || std::string(argv[2]) == "-");
    bool is_warmup_run = true;

    // Initialize sampling parameters and random engine
    unsigned int seed = std::random_device{}();
    if (const char* env_seed = std::getenv("DIFFKV_SEED")) {
        seed = std::stoul(env_seed);
    }
    std::mt19937 sample_rng(seed);

    float temperature = 0.7f;
    float top_p = 0.9f;
    float repetition_penalty = 1.15f;
    if (const char* env_temp = std::getenv("DIFFKV_TEMPERATURE")) {
        temperature = std::stof(env_temp);
    }
    if (const char* env_topp = std::getenv("DIFFKV_TOP_P")) {
        top_p = std::stof(env_topp);
    }
    if (const char* env_rep = std::getenv("DIFFKV_REPETITION_PENALTY")) {
        repetition_penalty = std::stof(env_rep);
    }

    // Speed 2: Pre-build alphanumeric cache for vocab tokens
    std::vector<int8_t> alnum_cache(n_vocab, 0);
    for (int32_t tok_id = 0; tok_id < n_vocab; ++tok_id) {
        std::string piece = model.token_to_piece(tok_id);
        bool has_alnum = false;
        for (char c : piece) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                has_alnum = true;
                break;
            }
        }
        alnum_cache[tok_id] = has_alnum ? 1 : 0;
    }


    while (true) {
        std::string prompt;
        int cached_len = 0; // tokens already in KV pool for this request (ACTIVE_RUNTIME prefix skip)
        if (is_warmup_run) {
            prompt = "warmup";
        } else if (interactive) {
            std::cout << "__READY__" << std::endl;
            if (!std::getline(std::cin, prompt)) {
                break;
            }
            if (prompt.rfind("__CACHED__:", 0) == 0) {
                try {
                    cached_len = std::stoi(prompt.substr(11));
                } catch (...) {
                    cached_len = 0;
                }
                if (!std::getline(std::cin, prompt)) {
                    break;
                }
            }
            if (prompt.empty() || prompt == "exit" || prompt == "quit") {
                break;
            }
            // Unescape \\n sequences back to real newlines (encoded by gateway for single-line stdin)
            {
                std::string unescaped;
                unescaped.reserve(prompt.size());
                for (size_t ui = 0; ui < prompt.size(); ++ui) {
                    if (ui + 1 < prompt.size() && prompt[ui] == '\\' && prompt[ui+1] == 'n') {
                        unescaped += '\n';
                        ++ui;
                    } else {
                        unescaped += prompt[ui];
                    }
                }
                prompt = std::move(unescaped);
            }
        } else {
            prompt = argv[2];
        }

        // Always do a full reset + re-prefill from token 0 on every turn.
        //
        // WHY: diffkv_native has no dense KV cache (unlike ACTIVE_RUNTIME's PyTorch model
        // which keeps past_key_values in GPU memory). Our compressed blocks can't be used
        // as prefill prior context without decompression. Skipping the first cached_len
        // tokens during prefill would mean the model attends to NO prior context for those
        // positions — causing it to hallucinate from the new prompt text (the root cause
        // of the 'Random Features' garbage output).
        //
        // Each turn: reset compressed pool, re-prefill full prompt from 0, decode.
        // Speed improvement from prefix-skipping requires implementing decompression-
        // based prior context injection (future work).
        bool do_full_reset = (cached_len == 0);
        if (do_full_reset || is_warmup_run) {
            runtime_manager.reset();

            for (int l = 0; l < n_layers; ++l) {
                std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
            }
            std::fill(active_positions_dense.begin(), active_positions_dense.end(), 0);
            total_positions = 0;
            persistent_k_dense.clear();
            persistent_v_dense.clear();
            srl_state.ordered_slot_ids.clear();
            srl_state.sink_blocks.clear();
            srl_state.inverted_index.clear();
            srl_state.chunk_graph = diffkv::ChunkGraph();
            srl_state.semantic_index = diffkv::SemanticIndex();
            srl_state.recent_generated_tokens.clear();
            srl_state.current_query_tokens.clear();
            srl_state.current_step_slots.clear();
            srl_state.current_step_factual_tokens.clear();
            srl_state.current_step_count = 0;
            srl_state.recent_miss_rate = 0.0f;
            srl_state.k_multiplier = 1.0f;
            srl_state.call_count = 0;
            for (int l = 0; l < n_layers; ++l) {
                std::fill(seq_lens_by_layer[l].begin(), seq_lens_by_layer[l].end(), 0);
            }
            session_cached_len = 0;
            session_cached_token_ids.clear();
        } else {
            // Continuation turn: only reset per-turn SRL state
            srl_state.recent_generated_tokens.clear();
            srl_state.current_query_tokens.clear();
            srl_state.current_step_slots.clear();
            srl_state.current_step_factual_tokens.clear();
            srl_state.current_step_count = 0;
            // Reset query anchor so each response anchors to its own first decode
            // step rather than persisting the previous turn's Q-vector.
            srl_state.factual_anchor_q.clear();
            // Reset entity context so each new response starts with no entity
            // commitment — the model freely picks which entity to lead with.
            srl_state.current_entity_id = -1;
            srl_state.dual_entity_mode = false;
            srl_state.dual_entity_ids.clear();
            srl_state.comparison_entities.clear();
            srl_state.comparison_active_idx = 0;
            srl_state.comparison_covered.clear();
            srl_state.current_step_sequence_entity_ids.clear();
            srl_state.current_step_sequence_is_prime.clear();
            srl_state.current_step_sequence_prefixes.clear();
        }

        // Wrap prompt in Qwen2.5 instruction chat template (skip if gateway already formatted it)
        std::string chat_prompt;
        if (prompt.rfind("<|im_start|", 0) == 0) {
            // Gateway sent a pre-formatted multi-turn conversation — use as-is
            chat_prompt = prompt;
            if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
                std::cerr << "[DiffKV Native] Using pre-formatted chat prompt (" << prompt.size() << " bytes)" << std::endl;
            }
        } else {
            chat_prompt =
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                "<|im_start|>assistant\n";
        }
        
        if (!interactive) {
            std::cerr << "[DiffKV Native] Prompt: \"" << prompt << "\"" << std::endl;
        }

        std::vector<int32_t> prompt_tokens = model.tokenize(chat_prompt, true);
        if (prompt_tokens.empty()) {
            std::cerr << "Error: Tokenization returned empty list!" << std::endl;
            if (!interactive) break;
            continue;
        }

        if (!interactive) {
            std::cerr << "[DiffKV Native] Prompt tokens (" << prompt_tokens.size() << "): ";
            for (int32_t t : prompt_tokens) std::cerr << t << " ";
            std::cerr << "\n";
        }

        int L = prompt_tokens.size();
        if (L > n_slots * 64) {
            std::cerr << "[DiffKV Native] Warning: Prompt tokens length " << L 
                      << " exceeds maximum capacity " << n_slots * 64 
                      << ". Truncating prompt." << std::endl;
            L = n_slots * 64;
            prompt_tokens.resize(L);
        }

        // Prefix verification guard: check mismatch index for token-level prefix verification.
        bool cache_valid = true;
        if (cached_len > 0) {
            int mismatch_idx = 0;
            if (!session_cached_token_ids.empty()) {
                int max_compare = std::min((int)prompt_tokens.size(), (int)session_cached_token_ids.size());
                while (mismatch_idx < max_compare && prompt_tokens[mismatch_idx] == session_cached_token_ids[mismatch_idx]) {
                    mismatch_idx++;
                }
            } else {
                mismatch_idx = 0;
            }

            if (mismatch_idx >= cached_len && cached_len < L) {
                // Fully valid cache prefix, no rollback needed.
            } else if (mismatch_idx > 32 && mismatch_idx < L) {
                // Partial rollback is possible and beneficial!
                if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
                    std::cerr << "[DiffKV Native] Prefix mismatch. Matching prefix length: " << mismatch_idx 
                              << " (cached_len=" << cached_len << "). Performing partial rollback." << std::endl;
                }
                runtime_manager.get_ingest_manager().rollback(mismatch_idx, runtime_manager.get_engines());
                cached_len = mismatch_idx;
                session_cached_len = mismatch_idx;
                if (mismatch_idx < (int)session_cached_token_ids.size()) {
                    session_cached_token_ids.resize(mismatch_idx);
                }
                
                // Clear/rollback SRL state to allow reconstruction for the new branch
                std::unordered_set<int32_t> kept_slots;
                const auto& blocks_layer0 = runtime_manager.get_ingest_manager().get_blocks(0);
                for (const auto& block : blocks_layer0) {
                    if (block->pool_idx != -1) {
                        kept_slots.insert(block->pool_idx);
                    }
                }
                srl_state.rollback_to(mismatch_idx, &kept_slots);
                srl_state.factual_store.clear();
                srl_state.inverted_index.clear();
                srl_state.chunk_graph = diffkv::ChunkGraph();
                srl_state.semantic_index = diffkv::SemanticIndex();
                srl_state.recent_miss_rate = 0.0f;
                srl_state.k_multiplier = 1.0f;
                srl_state.call_count = 0;
            } else {
                // Mismatch index too small, reset everything
                cache_valid = false;
            }
        }
        if (!cache_valid) {
            if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
                std::cerr << "[DiffKV Native] Warning: cached_len mismatch or invalid (" << cached_len 
                          << " vs L=" << L << "). Resetting KV cache." << std::endl;
            }
            cached_len = 0;
            do_full_reset = true;

            runtime_manager.reset();
            for (int l = 0; l < n_layers; ++l) {
                std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
            }
            std::fill(active_positions_dense.begin(), active_positions_dense.end(), 0);
            total_positions = 0;
            persistent_k_dense.clear();
            persistent_v_dense.clear();
            srl_state.ordered_slot_ids.clear();
            srl_state.sink_blocks.clear();
            srl_state.inverted_index.clear();
            srl_state.chunk_graph = diffkv::ChunkGraph();
            srl_state.semantic_index = diffkv::SemanticIndex();
            srl_state.recent_generated_tokens.clear();
            srl_state.current_query_tokens.clear();
            srl_state.current_step_slots.clear();
            srl_state.current_step_factual_tokens.clear();
            srl_state.current_step_count = 0;
            srl_state.recent_miss_rate = 0.0f;
            srl_state.k_multiplier = 1.0f;
            srl_state.call_count = 0;
            for (int l = 0; l < n_layers; ++l) {
                std::fill(seq_lens_by_layer[l].begin(), seq_lens_by_layer[l].end(), 0);
            }
            session_cached_len = 0;
            session_cached_token_ids.clear();
        }

        // ── Adaptive micro-block size (matches Python ACTIVE_RUNTIME logic) ──────
        {
            int raw_target;
            if (L < 256) {
                raw_target = 16;
            } else if (L < 1024) {
                raw_target = 32;
            } else if (L < 4096) {
                raw_target = 64;
            } else if (L < 8192) {
                raw_target = 128;
            } else {
                raw_target = 256;
            }
            int target = std::min(raw_target, micro_block_size);
            int adaptive_mbs = std::max(16, ((target + 15) / 16) * 16);
            if (adaptive_mbs != micro_block_size) {
                std::cerr << "[DiffKV Native] Adaptive micro_block_size: " << micro_block_size
                          << " -> " << adaptive_mbs << " (L=" << L << ")" << std::endl;
            }
            runtime_manager.set_micro_block_size(adaptive_mbs);
        }

        // ── 1. PREFILL PHASE ──
        if (!interactive) {
            std::cerr << "[DiffKV Native] Running Prefill phase in chunks..." << std::endl;
        }

        // Local per-turn raw K/V activation buffers.
        // For continuation turns (cached_len > 0), we only fill offsets [cached_len..L-1]
        // and upload the already-stored prefix K/V from the prior chunk's data (if pos_start > cached_len).
        // This matches ACTIVE_RUNTIME: full prompt is sent, only new tokens are prefilled.
        std::vector<std::vector<float>> k_activations(n_layers, std::vector<float>(L * F_test, 0.0f));
        std::vector<std::vector<float>> v_activations(n_layers, std::vector<float>(L * F_test, 0.0f));
        std::vector<float> prefill_output_logits(n_vocab);

        int chunk_size = 512; // Default to balanced preset size
        if (const char* env_preset = std::getenv("DIFFKV_PRESET")) {
            std::string p(env_preset);
            std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c){ return std::tolower(c); });
            if (p == "low") {
                chunk_size = 256;
            } else if (p == "high") {
                chunk_size = 2048;
            }
        }
        if (const char* env_pcs = std::getenv("DIFFKV_PREFILL_CHUNK_SIZE")) {
            try { chunk_size = std::stoi(env_pcs); } catch (...) {}
        }
        // Decompress prefix blocks from turn 1 if cached_len > 0
        if (cached_len > 0) {
            for (int l = 0; l < n_layers; ++l) {
                auto & blocks = runtime_manager.get_ingest_manager().get_blocks(l);
                for (auto & block : blocks) {
                    if (block->anchor_idx >= cached_len) {
                        break;
                    }
                    int block_len = block->token_count();
                    
                    // Touch block to ensure residency
                    if (block->state == BlockState::CPUResident) {
                        runtime_manager.get_pager().touch(block.get(), runtime_manager.get_engines());
                    }
                    
                    int slot_id = block->pool_idx;
                    if (block->state == BlockState::CompressedResident) {
                        auto & engine = runtime_manager.get_engines()[l];
                        int rank = engine->get_U()->ne[0];
                        float scale_u = ggml_fp16_to_fp32(engine->get_host_U_scale()[slot_id]);
                        float block_scale = ggml_fp16_to_fp32(engine->get_host_scales()[slot_id]);

                        // 1. Compute landmark index (landmark_idx)
                        int landmark_idx = engine->get_host_anchor_positions()[slot_id] - block->anchor_idx;

                        // 2. Compute non-anchor tokens
                        int non_anchor_len = block_len - 1;
                        if (non_anchor_len > 0) {
                            // Pre-convert U to float and scale it
                            std::vector<float> U_float(non_anchor_len * rank);
                            const int8_t* u_src = engine->get_host_U() + (slot_id * 64 * rank);
                            for (int i = 0; i < non_anchor_len * rank; ++i) {
                                U_float[i] = (float)u_src[i] * scale_u;
                            }

                            // Pre-convert VK and VV to float
                            std::vector<float> VK_float(rank * F_test);
                            std::vector<float> VV_float(rank * F_test);
                            const ggml_fp16_t* vk_src = engine->get_host_VK() + (slot_id * rank * F_test);
                            const ggml_fp16_t* vv_src = engine->get_host_VV() + (slot_id * rank * F_test);
                            for (int i = 0; i < rank * F_test; ++i) {
                                VK_float[i] = ggml_fp16_to_fp32(vk_src[i]);
                                VV_float[i] = ggml_fp16_to_fp32(vv_src[i]);
                            }

                            // Pre-convert anchors to float
                            std::vector<float> anchor_k_float(F_test);
                            std::vector<float> anchor_v_float(F_test);
                            const ggml_fp16_t* ak_src = engine->get_host_anchors_K() + (slot_id * F_test);
                            const ggml_fp16_t* av_src = engine->get_host_anchors_V() + (slot_id * F_test);
                            for (int f = 0; f < F_test; ++f) {
                                anchor_k_float[f] = ggml_fp16_to_fp32(ak_src[f]);
                                anchor_v_float[f] = ggml_fp16_to_fp32(av_src[f]);
                            }

                            // Matrix multiplication outputs: delta_K and delta_V
                            std::vector<float> K_delta(non_anchor_len * F_test, 0.0f);
                            std::vector<float> V_delta(non_anchor_len * F_test, 0.0f);

#ifdef __APPLE__
                            // Leverage Accelerate framework's cblas_sgemm for hardware AMX acceleration!
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                        non_anchor_len, F_test, rank,
                                        1.0f, U_float.data(), rank,
                                        VK_float.data(), F_test,
                                        0.0f, K_delta.data(), F_test);

                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                        non_anchor_len, F_test, rank,
                                        1.0f, U_float.data(), rank,
                                        VV_float.data(), F_test,
                                        0.0f, V_delta.data(), F_test);
#else
                            // Optimized CPU fallback loop order: s -> r -> f (sequential memory writes)
                            for (int s = 0; s < non_anchor_len; ++s) {
                                for (int r = 0; r < rank; ++r) {
                                    float u_val = U_float[s * rank + r];
                                    const float* vk_row = &VK_float[r * F_test];
                                    const float* vv_row = &VV_float[r * F_test];
                                    float* k_del_row = &K_delta[s * F_test];
                                    float* v_del_row = &V_delta[s * F_test];
                                    for (int f = 0; f < F_test; ++f) {
                                        k_del_row[f] += u_val * vk_row[f];
                                        v_del_row[f] += u_val * vv_row[f];
                                    }
                                }
                            }
#endif

                            // Add anchor K/V and multiply by block_scale, then store in activations.
                            // Copy anchor to landmark position, and delta tokens to their original positions.
                            for (int t = 0; t < block_len; ++t) {
                                int global_pos = block->anchor_idx + t;
                                if (global_pos >= cached_len) break;

                                if (t == landmark_idx) {
                                    // This is the landmark token, stored as the anchor in the pool
                                    for (int f = 0; f < F_test; ++f) {
                                        k_activations[l][global_pos * F_test + f] = anchor_k_float[f];
                                        v_activations[l][global_pos * F_test + f] = anchor_v_float[f];
                                    }
                                } else {
                                    // Reconstructed from SVD delta
                                    int s = (t == 0) ? (landmark_idx - 1) : (t - 1);
                                    float* k_act_row = &k_activations[l][global_pos * F_test];
                                    float* v_act_row = &v_activations[l][global_pos * F_test];
                                    const float* k_del_row = &K_delta[s * F_test];
                                    const float* v_del_row = &V_delta[s * F_test];
                                    for (int f = 0; f < F_test; ++f) {
                                        k_act_row[f] = anchor_k_float[f] + k_del_row[f] * block_scale;
                                        v_act_row[f] = anchor_v_float[f] + v_del_row[f] * block_scale;
                                    }
                                }
                            }
                        } else {
                            // Single token block (just anchor)
                            int global_pos = block->anchor_idx;
                            if (global_pos < cached_len) {
                                for (int f = 0; f < F_test; ++f) {
                                    k_activations[l][global_pos * F_test + f] = ggml_fp16_to_fp32(engine->get_host_anchors_K()[slot_id * F_test + f]);
                                    v_activations[l][global_pos * F_test + f] = ggml_fp16_to_fp32(engine->get_host_anchors_V()[slot_id * F_test + f]);
                                }
                            }
                        }
                    } else {
                        // DenseResident or Compressing
                        if (l == 0) {
                            std::cerr << "[Decompress Debug] Block starting at " << block->anchor_idx 
                                      << " block_len=" << block_len 
                                      << " active_k_size=" << block->active_k.size() 
                                      << " state=" << (int)block->state << "\n";
                        }
                        for (int t = 0; t < block_len; ++t) {
                            int global_pos = block->anchor_idx + t;
                            if (global_pos >= cached_len) break;
                            
                            if (t == 0) {
                                std::memcpy(k_activations[l].data() + global_pos * F_test, block->anchor_k.data(), F_test * sizeof(float));
                                std::memcpy(v_activations[l].data() + global_pos * F_test, block->anchor_v.data(), F_test * sizeof(float));
                            } else {
                                int s = t - 1;
                                std::memcpy(k_activations[l].data() + global_pos * F_test, block->active_k.data() + s * F_test, F_test * sizeof(float));
                                std::memcpy(v_activations[l].data() + global_pos * F_test, block->active_v.data() + s * F_test, F_test * sizeof(float));
                            }
                        }
                    }
                }
            }
        }

        int pos_start = cached_len;

        // ── CHUNKED PREFILL WITH FULL PRIOR-CONTEXT ATTENTION ──────────────────
        // Matches ACTIVE_RUNTIME Python: chunk c attends to ALL tokens [0..pos_start-1]
        // (via k_activations / v_activations which store RoPE-rotated K / raw V from prior
        //  chunks) plus itself causally.
        //
        // For continuation turns (cached_len > 0): prefill starts at pos_start = cached_len.
        // The first chunk has has_prior = (pos_start > 0) even for turn 2 chunk 1 because
        // pos_start starts at cached_len (e.g. 28 for the "hi" turn). The prior K/V for
        // those cached tokens are now decompressed back into k_activations/v_activations.
        //
        // k_activations[l] stores raw K (before RoPE); v_activations[l] stores raw V.

        size_t max_prior_bytes = (size_t)2 * n_layers * L * F_test * sizeof(float);
        size_t max_graph_bytes = 8 * 1024 * 1024 + max_prior_bytes;

        struct ggml_init_params prefill_params = {
            /*.mem_size   =*/ max_graph_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * prefill_ctx = ggml_init(prefill_params);
        if (!prefill_ctx) {
            std::cerr << "Failed to initialize prefill context!" << std::endl;
            // Always emit sentinels so the gateway doesn't hang waiting for __RESPONSE__
            if (interactive) {
                std::cout << "__RESPONSE__" << std::endl;
                std::cout << "[Error: failed to initialize prefill context]" << std::flush;
                std::cout << "\n__FINISH__" << std::endl;
            }
            if (!interactive) break; else continue;
        }

        while (pos_start < L) {
            int chunk_len = std::min(chunk_size, L - pos_start);
            int ctx_len   = pos_start + chunk_len;  // total KV context length

            ggml_reset(prefill_ctx);

            // ── 2. Create input tensors ────────────────────────────────────────
            struct ggml_tensor * input_tokens_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, chunk_len);
            ggml_set_input(input_tokens_prefill);
            struct ggml_tensor * positions_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, chunk_len);
            ggml_set_input(positions_prefill);

            // Full-context mask: [intra_ctx_len, chunk_len]
            // intra_ctx_len = pos_start + chunk_len (full causal window for this chunk).
            int intra_ctx_len = pos_start + chunk_len;
            struct ggml_tensor * mask_prefill = ggml_new_tensor_2d(prefill_ctx, GGML_TYPE_F16, intra_ctx_len, chunk_len);
            ggml_set_input(mask_prefill);

            // ── 3. Create prior-context tensors for each layer ─────────────────
            // These are used only when pos_start > cached_len (intra-turn prior chunks).
            // The cached prefix tokens [0..cached_len-1] are already in compressed KV pool
            // and are accessed via DiffKV attention during decode — not raw tensors here.
            std::vector<struct ggml_tensor *> prior_k_tensors(n_layers, nullptr);
            std::vector<struct ggml_tensor *> prior_v_tensors(n_layers, nullptr);
            bool has_prior = (pos_start > 0); // raw K/V from prior chunks of this turn
            if (has_prior) {
                int prior_intra_len = pos_start; // all prior intra-turn chunks
                for (int l = 0; l < n_layers; ++l) {
                    // prior_k: [head_dim, kv_heads, prior_intra_len] — raw K from THIS turn's prior chunks
                    prior_k_tensors[l] = ggml_new_tensor_3d(prefill_ctx, GGML_TYPE_F32,
                        head_dim, kv_heads, prior_intra_len);
                    ggml_set_input(prior_k_tensors[l]);
                    prior_v_tensors[l] = ggml_new_tensor_3d(prefill_ctx, GGML_TYPE_F32,
                        head_dim, kv_heads, prior_intra_len);
                    ggml_set_input(prior_v_tensors[l]);
                }
            }

            // ── 4. Build the graph ────────────────────────────────────────────
            struct ggml_tensor * prefill_logits = nullptr;
            std::vector<struct ggml_tensor *> prefill_k_layers(n_layers, nullptr);
            std::vector<struct ggml_tensor *> prefill_v_layers(n_layers, nullptr);
            bool is_last_chunk = (pos_start + chunk_len >= L);

            struct ggml_cgraph * prefill_graph = build_prefill_ctx_graph(
                prefill_ctx, model,
                input_tokens_prefill, positions_prefill, mask_prefill,
                has_prior ? &prior_k_tensors : nullptr,
                has_prior ? &prior_v_tensors : nullptr,
                &prefill_logits,
                &prefill_k_layers, &prefill_v_layers,
                is_last_chunk
            );
            if (is_last_chunk && prefill_logits) {
                ggml_set_output(prefill_logits);
            }

            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, prefill_graph)) {
                std::cerr << "Failed to allocate memory for prefill graph (chunk " << pos_start << " / " << L << ")!" << std::endl;
                break;
            }

            // ── 5. Upload inputs ──────────────────────────────────────────────
            ggml_backend_tensor_set(input_tokens_prefill, prompt_tokens.data() + pos_start, 0, chunk_len * sizeof(int32_t));

            std::vector<int32_t> pos_host(chunk_len);
            for (int i = 0; i < chunk_len; ++i) pos_host[i] = pos_start + i;
            ggml_backend_tensor_set(positions_prefill, pos_host.data(), 0, chunk_len * sizeof(int32_t));

            // Build full-context mask
            int intra_prior = pos_start;
            std::vector<ggml_fp16_t> mask_host(chunk_len * intra_ctx_len, ggml_fp32_to_fp16(0.0f));
            for (int qi = 0; qi < chunk_len; ++qi) {
                for (int kj = intra_prior; kj < intra_ctx_len; ++kj) {
                    int chunk_kj = kj - intra_prior;
                    if (chunk_kj > qi) {
                        mask_host[qi * intra_ctx_len + kj] = ggml_fp32_to_fp16(-INFINITY);
                    }
                }
            }
            ggml_backend_tensor_set(mask_prefill, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));

            // Upload prior K/V context (from this turn's prior chunks)
            // k_activations[l][0..pos_start-1] stores raw K from this turn's prior chunks.
            if (has_prior) {
                int intra_prior_len = pos_start;
                // Positions for prior tokens [0..pos_start-1]
                std::vector<int32_t> prior_positions(intra_prior_len);
                for (int t = 0; t < intra_prior_len; ++t) prior_positions[t] = t;

                // Precompute cos/sin tables for intra-turn prior positions
                int half_dim = head_dim / 2;
                std::vector<float> inv_freq(half_dim);
                for (int i = 0; i < half_dim; ++i) {
                    inv_freq[i] = 1.0f / std::pow(model.get_config().rope_freq_base, 2.0f * i / head_dim);
                }
                std::vector<float> cos_table(intra_prior_len * half_dim);
                std::vector<float> sin_table(intra_prior_len * half_dim);
                for (int t = 0; t < intra_prior_len; ++t) {
                    float pos = (float)prior_positions[t];
                    for (int i = 0; i < half_dim; ++i) {
                        float theta = pos * inv_freq[i];
                        cos_table[t * half_dim + i] = std::cos(theta);
                        sin_table[t * half_dim + i] = std::sin(theta);
                    }
                }

                // RoPE-rotate the intra-turn prior raw K and upload
                std::vector<float> prior_k_rotated(intra_prior_len * F_test);
                for (int l = 0; l < n_layers; ++l) {
                    apply_rope_neox_cpu_fast(
                        k_activations[l].data(),   // raw K from index 0 (= cached_len in prompt)
                        prior_k_rotated.data(),
                        cos_table.data(),
                        sin_table.data(),
                        intra_prior_len, kv_heads, head_dim
                    );
                    ggml_backend_tensor_set(prior_k_tensors[l],
                        prior_k_rotated.data(),
                        0, intra_prior_len * F_test * sizeof(float));
                    ggml_backend_tensor_set(prior_v_tensors[l],
                        v_activations[l].data(),
                        0, intra_prior_len * F_test * sizeof(float));
                }
            }

            // ── 6. Run the graph ──────────────────────────────────────────────
            if (ggml_backend_sched_graph_compute(sched, prefill_graph) != GGML_STATUS_SUCCESS) {
                std::cerr << "Error: Prefill graph compute failed at pos " << pos_start << "!" << std::endl;
                break;
            }

            // ── 7. Capture raw K/V for decode attention and next chunk's prior context ──
            // build_prefill_ctx_graph exports raw K (before RoPE) matching ACTIVE_RUNTIME.
            // k_activations stores raw K → used by decode callback (has_rope=true re-applies RoPE).
            // For next chunk's prior context, apply_rope_neox_cpu will rotate it at upload time.
            std::vector<std::vector<float>> chunk_k(n_layers, std::vector<float>(chunk_len * F_test));
            std::vector<std::vector<float>> chunk_v(n_layers, std::vector<float>(chunk_len * F_test));
            // Store raw K/V at pos_start offset in k_activations
            int local_offset = pos_start;
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_tensor_get(prefill_k_layers[l], chunk_k[l].data(), 0, chunk_len * F_test * sizeof(float));
                ggml_backend_tensor_get(prefill_v_layers[l], chunk_v[l].data(), 0, chunk_len * F_test * sizeof(float));
                std::memcpy(k_activations[l].data() + local_offset * F_test, chunk_k[l].data(), chunk_len * F_test * sizeof(float));
                std::memcpy(v_activations[l].data() + local_offset * F_test, chunk_v[l].data(), chunk_len * F_test * sizeof(float));
            }
            // Ingest chunk into KV manager (raw K + raw V, matching ACTIVE_RUNTIME ingest_streaming)
            runtime_manager.ingest_prefill(chunk_k, chunk_v, chunk_len, pos_start, prompt_tokens, &srl_state);

            if (pos_start + chunk_len >= L && prefill_logits) {
                ggml_backend_tensor_get(prefill_logits, prefill_output_logits.data(), 0, n_vocab * sizeof(float));
            }

            pos_start += chunk_len;
        }

        if (prefill_ctx) {
            ggml_free(prefill_ctx);
        }

        if (!interactive) {
            std::cerr << "[DiffKV Native] Waiting for background SVD compressor to catch up..." << std::endl;
        }
        runtime_manager.wait_for_compressor();
        runtime_manager.update_descriptors(W_proj_host, desc_dim, head_dim);
        // Native attn: push prefill-compressed slots host→device before decode (async SVD
        // only wrote host mirrors). Decode-time blocks are handled inside ingest_decode.
        runtime_manager.sync_device_for_native();

        if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
            std::cerr << "[DEBUG ACTIVATIONS] cached_len=" << cached_len << " L=" << L << "\n";
            std::cerr << "  k_activations[0] at 0 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << k_activations[0][0 * F_test + i];
            std::cerr << "\n  k_activations[0] at 21 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << k_activations[0][21 * F_test + i];
            std::cerr << "\n  k_activations[0] at 77 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << k_activations[0][77 * F_test + i];
            std::cerr << "\n  v_activations[0] at 0 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << v_activations[0][0 * F_test + i];
            std::cerr << "\n  v_activations[0] at 21 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << v_activations[0][21 * F_test + i];
            std::cerr << "\n  v_activations[0] at 77 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << v_activations[0][77 * F_test + i];
            std::cerr << std::endl;
        }

        // Logits already retrieved inside the chunk loop

        int32_t first_decode_token = 0;
        if (interactive) {
            first_decode_token = sample_logits(prefill_output_logits, temperature, top_p, sample_rng);
        } else {
            std::vector<std::pair<float, int>> prefill_top_k;
            prefill_top_k.reserve(n_vocab);
            for (int i = 0; i < n_vocab; ++i) {
                prefill_top_k.push_back({prefill_output_logits[i], i});
            }
            std::partial_sort(prefill_top_k.begin(), prefill_top_k.begin() + 5, prefill_top_k.end(),
                              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });
            first_decode_token = prefill_top_k[0].second;

            std::cerr << "\n[Prefill Phase Top predictions]:\n";
            for (int k = 0; k < 5; ++k) {
                std::cerr << "  " << k << ": \"" << model.token_to_piece(prefill_top_k[k].second) << "\" (id: " << prefill_top_k[k].second << ", logit: " << prefill_top_k[k].first << ")\n";
            }
        }

        // Resources already freed inside the chunk loop

        // ── 2. DECODE PHASE — rebuild decode graph fresh (avoids sched-ctx pointer corruption) ──
        int engage_threshold = 2048;
        if (const char* env_et = std::getenv("DIFFKV_ENGAGE_THRESHOLD")) {
            engage_threshold = std::stoi(env_et);
        }
        bool decode_use_sparse = (L >= engage_threshold);

        if (!decode_use_sparse) {
            int half_dim = head_dim / 2;
            std::vector<float> cos_table_full(L * half_dim);
            std::vector<float> sin_table_full(L * half_dim);
            for (int t = 0; t < L; ++t) {
                for (int i = 0; i < half_dim; ++i) {
                    float theta = (float)t / std::pow(model.get_config().rope_freq_base, (float)(2 * i) / (float)head_dim);
                    cos_table_full[t * half_dim + i] = std::cos(theta);
                    sin_table_full[t * half_dim + i] = std::sin(theta);
                }
            }
            for (int l = 0; l < n_layers; ++l) {
                apply_rope_neox_cpu_fast(
                    k_activations[l].data(),
                    active_k_dense_rotated[l].data(),
                    cos_table_full.data(),
                    sin_table_full.data(),
                    L, kv_heads, head_dim
                );
            }
        }

        if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
            std::cerr << "[DiffKV Native] Building fresh decode graph..." << std::flush;
        }
        struct ggml_init_params decode_params = {
            // Native sparse-attn builds far more op/tensor metadata + a 32k-node graph arena.
            /*.mem_size   =*/ is_native_attn_enabled()
                                  ? (size_t)48 * 1024 * 1024 : (size_t)4 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * decode_ctx = ggml_init(decode_params);
        if (!decode_ctx) {
            std::cerr << "\n[ERROR] Failed to initialize decode context!" << std::endl;
            // Always emit sentinels so the gateway doesn't hang waiting for __RESPONSE__
            if (interactive) {
                std::cout << "__RESPONSE__" << std::endl;
                std::cout << "[Error: failed to initialize decode context]" << std::flush;
                std::cout << "\n__FINISH__" << std::endl;
            }
            if (!interactive) break; else continue;
        }

        struct ggml_tensor * input_token_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(input_token_decode);
        struct ggml_tensor * position_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(position_decode);
        struct ggml_tensor * W_proj_decode = ggml_new_tensor_2d(decode_ctx, GGML_TYPE_F32, head_dim, desc_dim);
        ggml_set_input(W_proj_decode);
        struct ggml_tensor * slots_mask_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_F32, n_slots);
        ggml_set_input(slots_mask_decode);
        struct ggml_tensor * host_slots_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, srl_k_host);
        ggml_set_input(host_slots_decode);
        struct ggml_tensor * dense_attn_mask_decode = ggml_new_tensor_2d(decode_ctx, GGML_TYPE_F16, engage_threshold + 1, 1);
        ggml_set_input(dense_attn_mask_decode);

#ifdef __APPLE__
        bool approx = true;
#else
        bool approx = false;
#endif
        if (const char* env_approx = std::getenv("DIFFKV_MPS_APPROXIMATE_ATTN")) {
            approx = (std::strcmp(env_approx, "1") == 0 || std::strcmp(env_approx, "true") == 0 || std::strcmp(env_approx, "yes") == 0 || std::strcmp(env_approx, "on") == 0);
        }

        std::vector<diffkv::CustomAttnUserData> userdata(n_layers);
        for (int l = 0; l < n_layers; ++l) {
            userdata[l].kv_engine = kv_engines[l].get();
            userdata[l].session_id = "interactive_session";
            userdata[l].layer_idx = l;
            userdata[l].slot_indices = nullptr;
            userdata[l].n_q_heads = model.get_config().n_head;
            userdata[l].n_kv_heads = model.get_config().n_head_kv;
            userdata[l].rank = kv_engines[l]->get_rank();
            userdata[l].S_max = 64;
            userdata[l].K = 0;
            userdata[l].D = head_dim;
            userdata[l].scale = 1.0f / std::sqrt((float)head_dim);
            userdata[l].has_rope = true;
            userdata[l].rope_freq_base = model.get_config().rope_freq_base;
            userdata[l].approximate_attn = approx;
            userdata[l].ignore_c = true;
            userdata[l].srl_state = &srl_state;
            userdata[l].W_proj = W_proj_host.data();
            userdata[l].desc_dim = desc_dim;
        }

        struct ggml_tensor * decode_logits = nullptr;
        struct ggml_tensor * decode_selected_slots = nullptr;
        struct ggml_tensor * decode_concat_k = nullptr;
        struct ggml_tensor * decode_concat_v = nullptr;
        struct ggml_tensor * dbg_anc = nullptr;

        struct ggml_init_params dense_past_params = {
            /*.mem_size   =*/ 4 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * dense_past_ctx = ggml_init(dense_past_params);
        std::vector<struct ggml_tensor *> dense_k_past_inputs(n_layers, nullptr);
        std::vector<struct ggml_tensor *> dense_v_past_inputs(n_layers, nullptr);
        for (int l = 0; l < n_layers; ++l) {
            dense_k_past_inputs[l] = ggml_new_tensor_3d(dense_past_ctx, GGML_TYPE_F32, head_dim, kv_heads, engage_threshold);
            ggml_set_input(dense_k_past_inputs[l]);
            dense_v_past_inputs[l] = ggml_new_tensor_3d(dense_past_ctx, GGML_TYPE_F32, head_dim, kv_heads, engage_threshold);
            ggml_set_input(dense_v_past_inputs[l]);
        }

        // Native sparse-attn dense-window inputs (past rotated K / V + validity mask),
        // persistent (survive graph rebuilds), filled each decode step. Gated.
        const int native_maxd = 2048;
        const bool native_attn_on = is_native_attn_enabled();
        std::vector<struct ggml_tensor *> native_dense_kr(n_layers, nullptr);
        std::vector<struct ggml_tensor *> native_dense_v(n_layers, nullptr);
        struct ggml_tensor * native_dense_mask = nullptr;
        if (native_attn_on) {
            for (int l = 0; l < n_layers; ++l) {
                native_dense_kr[l] = ggml_new_tensor_2d(dense_past_ctx, GGML_TYPE_F32, head_dim * kv_heads, native_maxd);
                ggml_set_input(native_dense_kr[l]);
                native_dense_v[l]  = ggml_new_tensor_2d(dense_past_ctx, GGML_TYPE_F32, head_dim * kv_heads, native_maxd);
                ggml_set_input(native_dense_v[l]);
            }
            native_dense_mask = ggml_new_tensor_1d(dense_past_ctx, GGML_TYPE_F32, native_maxd);
            ggml_set_input(native_dense_mask);
        }
        // Past-dense token positions (for in-graph RoPE of the raw dense K).
        struct ggml_tensor * native_dense_pos = nullptr;
        if (native_attn_on) {
            native_dense_pos = ggml_new_tensor_1d(dense_past_ctx, GGML_TYPE_I32, native_maxd);
            ggml_set_input(native_dense_pos);
        }
        // Dedup constants for the native path: strict-lower-triangular ones + 0.5 scalar.
        struct ggml_tensor * native_dup_tri = nullptr;
        struct ggml_tensor * native_half = nullptr;
        if (native_attn_on) {
            native_dup_tri = ggml_new_tensor_2d(dense_past_ctx, GGML_TYPE_F32, srl_k_keep, srl_k_keep);
            ggml_set_input(native_dup_tri);
            native_half = ggml_new_tensor_1d(dense_past_ctx, GGML_TYPE_F32, 1);
            ggml_set_input(native_half);
        }
        ggml_backend_buffer_t dense_past_buffer = ggml_backend_alloc_ctx_tensors(dense_past_ctx, backend);
        if (native_attn_on) {
            // Fill the dedup constants once (persistent). tri[j,k] = 1 if j<k else 0 (column-major: idx = j + k*K).
            std::vector<float> tri((size_t)srl_k_keep * srl_k_keep, 0.0f);
            for (int k = 0; k < srl_k_keep; ++k)
                for (int j = 0; j < k; ++j)
                    tri[(size_t)j + (size_t)k * srl_k_keep] = 1.0f;
            ggml_backend_tensor_set(native_dup_tri, tri.data(), 0, tri.size() * sizeof(float));
            float halfv = 0.5f;
            ggml_backend_tensor_set(native_half, &halfv, 0, sizeof(float));
        }

        ggml_backend_buffer_t native_decode_buf = nullptr; // no-reuse alloc for the native decode path
        struct ggml_cgraph * decode_graph = build_decode_graph(
            decode_ctx, model, input_token_decode, position_decode, W_proj_decode,
            kv_engines[0]->get_desc_matrix(), kv_engines[0]->get_anchors_K(),
            slots_mask_decode, host_slots_decode,
            srl_k_semantic, srl_k_keep,
            userdata.data(), &decode_logits, &decode_selected_slots,
            &decode_concat_k, &decode_concat_v,
            decode_use_sparse, L, engage_threshold,
            dense_k_past_inputs.data(), dense_v_past_inputs.data(),
            dense_attn_mask_decode,
            native_dense_kr.data(), native_dense_v.data(), native_dense_mask, native_maxd,
            native_dup_tri, native_half, native_dense_pos, &dbg_anc
        );
        if (dbg_anc) ggml_set_output(dbg_anc);
        ggml_set_output(decode_logits);
        if (decode_selected_slots) ggml_set_output(decode_selected_slots);
        if (decode_concat_k) ggml_set_output(decode_concat_k);
        if (decode_concat_v) ggml_set_output(decode_concat_v);

        // Native path: allocate the decode graph with NO buffer reuse (ggml_backend_sched's
        // reuse corrupts the native sparse-attn subgraph on big-score inputs — verified: the
        // subgraph MATH is exact under no-reuse via DIFFKV_SELFTEST). Direct backend compute.
        bool decode_alloc_ok;
        if (native_attn_on) {
            if (native_decode_buf) ggml_backend_buffer_free(native_decode_buf);
            native_decode_buf = ggml_backend_alloc_ctx_tensors(decode_ctx, backend);
            decode_alloc_ok = (native_decode_buf != nullptr);
        } else {
            ggml_backend_sched_reset(sched);
            if (decode_concat_k) ggml_backend_sched_set_tensor_backend(sched, decode_concat_k, backend);
            if (decode_concat_v) ggml_backend_sched_set_tensor_backend(sched, decode_concat_v, backend);
            for (int l = 0; l < n_layers; ++l) {
                if (dense_k_past_inputs[l]) ggml_backend_sched_set_tensor_backend(sched, dense_k_past_inputs[l], backend);
                if (dense_v_past_inputs[l]) ggml_backend_sched_set_tensor_backend(sched, dense_v_past_inputs[l], backend);
            }
            if (dense_attn_mask_decode) ggml_backend_sched_set_tensor_backend(sched, dense_attn_mask_decode, backend);
            decode_alloc_ok = ggml_backend_sched_alloc_graph(sched, decode_graph);
        }
        if (!decode_alloc_ok) {
            // Always emit sentinels so the gateway doesn't hang waiting for __RESPONSE__
            if (interactive) {
                std::cout << "__RESPONSE__" << std::endl;
                std::cout << "[Error: decode graph allocation failed]" << std::flush;
                std::cout << "\n__FINISH__" << std::endl;
            }
            ggml_free(decode_ctx);
            if (!interactive) break; else continue;
        }
        if (decode_use_sparse) {
            ggml_backend_tensor_set(W_proj_decode, W_proj_host.data(), 0, W_proj_host.size() * sizeof(float));
        }
        if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
            std::cerr << " OK" << std::endl;
        }

        if (!is_warmup_run) {
            if (interactive) {
                std::cout << "__RESPONSE__" << std::endl;
            } else {
                std::cout << "[Response] " << std::flush;
            }
        }

        int32_t last_token = first_decode_token;
        bool sfa_active = false;
        std::string first_piece = model.token_to_piece(last_token);
        if (!is_warmup_run) {
            std::cout << first_piece << std::flush;
        }

        std::vector<int32_t> generated_tokens;
        generated_tokens.push_back(last_token);
        // max_generate: default 2048, overridable via DIFFKV_MAX_TOKENS env var or request
        int max_generate = 2048;
        if (is_warmup_run) {
            max_generate = 2;
        } else if (const char* env_mt = std::getenv("DIFFKV_MAX_TOKENS")) {
            max_generate = std::max(1, std::stoi(env_mt));
        }

        std::vector<int32_t> all_tokens = prompt_tokens;
        all_tokens.push_back(last_token);

        // Build initial SessionSRLState for all completed blocks in prefill using runtime_manager
        srl_state.vsl_active_candidates.clear();
        srl_state.vsl_consecutive_helpers = 0;
        srl_state.ordered_slot_ids.clear();
        srl_state.sink_blocks.clear();
        srl_state.inverted_index.clear();
        srl_state.chunk_graph = diffkv::ChunkGraph();
        srl_state.semantic_index = diffkv::SemanticIndex();

        auto & blocks_layer0 = runtime_manager.get_ingest_manager().get_blocks(0);
        std::vector<int32_t> compressed_slots;
        for (int i = 0; i < (int)blocks_layer0.size(); ++i) {
            if (blocks_layer0[i]->pool_idx != -1 &&
                (blocks_layer0[i]->state == BlockState::CompressedResident ||
                 blocks_layer0[i]->state == BlockState::CPUResident)) {
                compressed_slots.push_back(blocks_layer0[i]->pool_idx); // Physical slot ID
            }
        }
        int completed_blocks = compressed_slots.size();
        if (completed_blocks > 0) {
            std::vector<float> desc_matrix_host(completed_blocks * desc_dim);
            for (int j = 0; j < completed_blocks; ++j) {
                int slot_id = compressed_slots[j];
                ggml_backend_tensor_get(
                    runtime_manager.get_engines()[0]->get_desc_matrix(),
                    desc_matrix_host.data() + j * desc_dim,
                    slot_id * desc_dim * sizeof(float),
                    desc_dim * sizeof(float)
                );
            }

            srl_state = build_srl_state_from_blocks(
                desc_matrix_host.data(),
                compressed_slots.data(),
                completed_blocks,
                prompt_tokens.data(),
                L,
                runtime_manager.get_micro_block_size() + 1, // block_size (adaptive)
                stop_token_ids,
                6, // K_semantic
                2, // K_temporal
                0.15f, // overlap_threshold
                true, // add_first_as_sink
                true  // add_last_as_sink
            );

            std::unordered_set<int32_t> prime_slots(
                srl_state.chunk_graph.cluster_centers_tensor.begin(),
                srl_state.chunk_graph.cluster_centers_tensor.end()
            );
            srl_state.factual_store.build(
                k_activations,
                v_activations,
                prompt_tokens,
                W_proj_host.data(),
                desc_dim,
                head_dim,
                kv_heads,
                stop_token_ids,
                srl_state.ordered_slot_ids,
                runtime_manager.get_micro_block_size() + 1,
                srl_state.inverted_index,
                prime_slots,
                get_helper_token_ids_cpp(model),
                true // use_salience_parser
            );
            srl_state.setup_sas_and_eqa(prompt_tokens, stop_token_ids, [&](int32_t tid) {
                return model.token_to_piece(tid);
            });
        }

        int active_slot = runtime_manager.get_ingest_manager().get_blocks(0).size() - 1;
        if (active_slot < 0) {
            active_slot = 0;
        }
        int active_block_tokens = 0;
        if (!runtime_manager.get_ingest_manager().get_blocks(0).empty()) {
            active_block_tokens = runtime_manager.get_ingest_manager().get_blocks(0).back()->token_count();
            if (active_block_tokens > 0) {
                active_block_tokens--; // exclude anchor
            }
        }

        // ── Retrieval throttle: cache routed_blocks across N decode steps ──────
        // route_decode_slots() does lexical scoring + 2-hop graph traversal — calling
        // it every token is O(N_blocks × vocab) extra CPU work per token, severely
        // limiting TPS. Re-route every DIFFKV_RETRIEVAL_INTERVAL steps (default 8).
        int retrieval_interval = 8;
        if (const char* env_ri = std::getenv("DIFFKV_RETRIEVAL_INTERVAL")) {
            retrieval_interval = std::max(1, std::stoi(env_ri));
        }
        std::vector<int32_t> cached_routed_blocks;
        std::vector<int32_t> cached_physical_candidates;
        int last_retrieval_step = -retrieval_interval; // force retrieval on step 0
        int last_retrieval_active_slot = -1;

        std::vector<int> dense_start_positions(n_layers, 0);
        std::vector<int> total_dense_tokens(n_layers, 0);

        for (int l = 0; l < n_layers; ++l) {
            auto & b_list = runtime_manager.get_ingest_manager().get_blocks(l);
            int curr_token_idx = 0;
            bool found_first = false;

            std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
            std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);

            for (auto & block : b_list) {
                if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                    if (!found_first) {
                        dense_start_positions[l] = block->anchor_idx;
                        found_first = true;
                    }

                    std::memcpy(
                        active_k_dense[l].data() + curr_token_idx * F_test,
                        block->anchor_k.data(),
                        F_test * sizeof(float)
                    );
                    std::memcpy(
                        active_v_dense[l].data() + curr_token_idx * F_test,
                        block->anchor_v.data(),
                        F_test * sizeof(float)
                    );
                    curr_token_idx++;

                    if (!block->active_k.empty()) {
                        int active_len = block->active_k.size() / F_test;
                        std::memcpy(
                            active_k_dense[l].data() + curr_token_idx * F_test,
                            block->active_k.data(),
                            block->active_k.size() * sizeof(float)
                        );
                        std::memcpy(
                            active_v_dense[l].data() + curr_token_idx * F_test,
                            block->active_v.data(),
                            block->active_v.size() * sizeof(float)
                        );
                        curr_token_idx += active_len;
                    }
                }
            }
            total_dense_tokens[l] = curr_token_idx;
        }

        {
            auto & b_list = runtime_manager.get_ingest_manager().get_blocks(0);
            int curr_pos_idx = 0;
            std::fill(active_positions_dense.begin(), active_positions_dense.end(), 0);
            for (auto & block : b_list) {
                if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                    for (int32_t t_pos : block->token_indices) {
                        active_positions_dense[curr_pos_idx++] = t_pos;
                    }
                }
            }
            total_positions = curr_pos_idx;
        }

        if (!decode_use_sparse) {
            for (int l = 0; l < n_layers; ++l) {
                total_dense_tokens[l] = L;
                std::memcpy(active_k_dense[l].data(), k_activations[l].data(), L * F_test * sizeof(float));
                std::memcpy(active_v_dense[l].data(), v_activations[l].data(), L * F_test * sizeof(float));
            }
            for (int i = 0; i < L; ++i) {
                active_positions_dense[i] = i;
            }
            total_positions = L;
        }

        const char* env_td = std::getenv("DIFFKV_TIME_DECODE");
        bool time_decode = (env_td && std::string(env_td) == "1");

        for (int step = 0; step < max_generate; ++step) {
            auto t_step_start = std::chrono::high_resolution_clock::now();
            if (active_slot >= n_slots) {
                std::cerr << "\n[DiffKV Native] Warning: Context slot capacity reached during decode. Stopping generation." << std::endl;
                break;
            }
            int current_pos = L + step;

            bool step_use_sparse = (current_pos >= engage_threshold);
            bool rebuild_needed = (step_use_sparse != decode_use_sparse);
            if (rebuild_needed) {
                decode_use_sparse = step_use_sparse;
                ggml_free(decode_ctx);
                decode_ctx = ggml_init(decode_params);
                if (!decode_ctx) {
                    std::cerr << "\n[ERROR] Failed to re-initialize decode context!" << std::endl;
                    break;
                }
                
                input_token_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, 1);
                ggml_set_input(input_token_decode);
                position_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, 1);
                ggml_set_input(position_decode);
                W_proj_decode = ggml_new_tensor_2d(decode_ctx, GGML_TYPE_F32, head_dim, desc_dim);
                ggml_set_input(W_proj_decode);
                slots_mask_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_F32, n_slots);
                ggml_set_input(slots_mask_decode);
                host_slots_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, srl_k_host);
                ggml_set_input(host_slots_decode);
                dense_attn_mask_decode = ggml_new_tensor_2d(decode_ctx, GGML_TYPE_F16, engage_threshold + 1, 1);
                ggml_set_input(dense_attn_mask_decode);
                
                // Reuse persistent dense K/V input tensors allocated in dense_past_ctx
                
                decode_logits = nullptr;
                decode_selected_slots = nullptr;
                decode_concat_k = nullptr;
                decode_concat_v = nullptr;
                
                decode_graph = build_decode_graph(
                    decode_ctx, model, input_token_decode, position_decode, W_proj_decode,
                    kv_engines[0]->get_desc_matrix(), kv_engines[0]->get_anchors_K(),
                    slots_mask_decode, host_slots_decode,
                    srl_k_semantic, srl_k_keep,
                    userdata.data(), &decode_logits, &decode_selected_slots,
                    &decode_concat_k, &decode_concat_v,
                    decode_use_sparse, current_pos, engage_threshold,
                    dense_k_past_inputs.data(), dense_v_past_inputs.data(),
                    dense_attn_mask_decode,
                    native_dense_kr.data(), native_dense_v.data(), native_dense_mask, native_maxd,
                    native_dup_tri, native_half, native_dense_pos
                );
                ggml_set_output(decode_logits);
                if (decode_selected_slots) ggml_set_output(decode_selected_slots);
                if (decode_concat_k) ggml_set_output(decode_concat_k);
                if (decode_concat_v) ggml_set_output(decode_concat_v);

                bool realloc_ok;
                if (native_attn_on) {
                    if (native_decode_buf) ggml_backend_buffer_free(native_decode_buf);
                    native_decode_buf = ggml_backend_alloc_ctx_tensors(decode_ctx, backend);
                    realloc_ok = (native_decode_buf != nullptr);
                } else {
                    ggml_backend_sched_reset(sched);
                    if (decode_concat_k) ggml_backend_sched_set_tensor_backend(sched, decode_concat_k, backend);
                    if (decode_concat_v) ggml_backend_sched_set_tensor_backend(sched, decode_concat_v, backend);
                    for (int l = 0; l < n_layers; ++l) {
                        if (dense_k_past_inputs[l]) ggml_backend_sched_set_tensor_backend(sched, dense_k_past_inputs[l], backend);
                        if (dense_v_past_inputs[l]) ggml_backend_sched_set_tensor_backend(sched, dense_v_past_inputs[l], backend);
                    }
                    if (dense_attn_mask_decode) ggml_backend_sched_set_tensor_backend(sched, dense_attn_mask_decode, backend);
                    realloc_ok = ggml_backend_sched_alloc_graph(sched, decode_graph);
                }
                if (!realloc_ok) {
                    std::cerr << "Error: Decode graph reallocation failed" << std::endl;
                    break;
                }
                if (decode_use_sparse) {
                    ggml_backend_tensor_set(W_proj_decode, W_proj_host.data(), 0, W_proj_host.size() * sizeof(float));
                }
            }

            ggml_backend_tensor_set(input_token_decode, &last_token, 0, sizeof(last_token));
            ggml_backend_tensor_set(position_decode, &current_pos, 0, sizeof(current_pos));

            if (!decode_use_sparse) {
                for (int l = 0; l < n_layers; ++l) {
                    if (dense_k_past_inputs[l] && dense_v_past_inputs[l]) {
                        if (step == 0) {
                            ggml_backend_tensor_set(
                                dense_k_past_inputs[l],
                                active_k_dense_rotated[l].data(),
                                0,
                                current_pos * F_test * sizeof(float)
                            );
                            ggml_backend_tensor_set(
                                dense_v_past_inputs[l],
                                active_v_dense[l].data(),
                                0,
                                current_pos * F_test * sizeof(float)
                            );
                        } else {
                            int prev_idx = current_pos - 1;
                            ggml_backend_tensor_set(
                                dense_k_past_inputs[l],
                                active_k_dense_rotated[l].data() + prev_idx * F_test,
                                prev_idx * F_test * sizeof(float),
                                F_test * sizeof(float)
                            );
                            ggml_backend_tensor_set(
                                dense_v_past_inputs[l],
                                active_v_dense[l].data() + prev_idx * F_test,
                                prev_idx * F_test * sizeof(float),
                                F_test * sizeof(float)
                            );
                        }
                    }
                }
                std::vector<ggml_fp16_t> dense_mask_host(engage_threshold + 1);
                for (int i = 0; i < engage_threshold + 1; ++i) {
                    float val = (i < current_pos || i == engage_threshold) ? 0.0f : -1e10f;
                    dense_mask_host[i] = ggml_fp32_to_fp16(val);
                }
                ggml_backend_tensor_set(dense_attn_mask_decode, dense_mask_host.data(), 0, (engage_threshold + 1) * sizeof(ggml_fp16_t));
            }

            auto t_before_retrieval = std::chrono::high_resolution_clock::now();
            auto t_after_retrieval = t_before_retrieval;
            // ── Throttled retrieval ─────────────────────────────────────────
            // Only re-run route_decode_slots every `retrieval_interval` steps.
            // Between re-routes, reuse the cached block list — retrieval results
            // are stable across a few tokens and the scoring overhead is O(N×vocab).
            bool do_retrieval = (step - last_retrieval_step >= retrieval_interval) || (active_slot != last_retrieval_active_slot);
            if (do_retrieval) {
                cached_routed_blocks = runtime_manager.route_decode_slots(
                    current_pos,
                    all_tokens,
                    srl_state,
                    stop_token_ids,
                    srl_k_recency,
                    srl_k_lexical,
                    srl_k_graph,
                    srl_k_host,
                    active_slot
                );
                last_retrieval_step = step;
                last_retrieval_active_slot = active_slot;

                // Translate block indices to physical pool slot indices
                cached_physical_candidates = cached_routed_blocks;

                if (step == 0 && !is_warmup_run && std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
                    // Diagnostic: print retrieval results before first generated token
                    std::cerr << "\n[DiffKV DIAG] Decode step 0 retrieval:\n";
                    std::cerr << "  active_slot=" << active_slot << " total_blocks=" << runtime_manager.get_ingest_manager().get_blocks(0).size() << "\n";
                    std::cerr << "  routed blocks (logical): ";
                    for (int32_t b : cached_routed_blocks) std::cerr << b << " ";
                    std::cerr << "\n  physical pool slots: ";
                    for (int32_t s : cached_physical_candidates) std::cerr << s << " ";
                    std::cerr << "\n  srl_state.ordered_slot_ids size=" << srl_state.ordered_slot_ids.size() << "\n";
                    std::cerr << "  srl_state.sink_blocks: ";
                    for (int32_t s : srl_state.sink_blocks) std::cerr << s << " ";
                    std::cerr << std::endl;
                }
                t_after_retrieval = std::chrono::high_resolution_clock::now();
            }
            std::vector<int32_t> routed_blocks = cached_routed_blocks;
            std::vector<int32_t> physical_candidates = cached_physical_candidates;

            // Touch blocks to ensure CPU Resident blocks are reloaded to GPU
            runtime_manager.touch_active_slots(routed_blocks);

            if (decode_use_sparse) {
                ggml_backend_tensor_set(host_slots_decode, physical_candidates.data(), 0, srl_k_host * sizeof(int32_t));
            }

            std::vector<float> slots_mask_host(n_slots, -1e10f);
            int occupied_up_to = active_slot - 1;
            for (int i = 0; i <= occupied_up_to; ++i) {
                auto & blocks = runtime_manager.get_ingest_manager().get_blocks(0);
                if (i >= 0 && i < (int)blocks.size()) {
                    int slot_id = blocks[i]->pool_idx;
                    if (slot_id >= 0 && slot_id < n_slots) {
                        if (blocks[i]->state != BlockState::DenseResident && blocks[i]->state != BlockState::Compressing) {
                            slots_mask_host[slot_id] = 0.0f;
                        }
                    }
                }
            }
            if (decode_use_sparse) {
                ggml_backend_tensor_set(slots_mask_decode, slots_mask_host.data(), 0, n_slots * sizeof(float));
            }

            int current_k = std::max(0, std::min(srl_k_keep, active_slot));
            
            // Get physical active slot index
            int physical_active_slot = 0;
            auto & b_list = runtime_manager.get_ingest_manager().get_blocks(0);
            if (active_slot >= 0 && active_slot < (int)b_list.size()) {
                physical_active_slot = b_list[active_slot]->pool_idx;
            }

            for (int l = 0; l < n_layers; ++l) {
                userdata[l].K = current_k;
                userdata[l].active_k_dense = active_k_dense[l].data();
                userdata[l].active_v_dense = active_v_dense[l].data();
                userdata[l].active_positions_dense = active_positions_dense.data();
                userdata[l].active_block_tokens = total_dense_tokens[l];
                userdata[l].active_slot = (total_dense_tokens[l] > 0) ? dense_start_positions[l] : current_pos;
                userdata[l].current_pos = current_pos;  // for Metal dense-window RoPE
            }

            // Native sparse-attn: upload the past dense window (rotated K + raw V) and its
            // validity mask into the persistent graph inputs (current token handled in-graph).
            if (native_attn_on && decode_use_sparse) {
                const int cnt0 = std::min(total_dense_tokens[0], native_maxd);
                if (std::getenv("DIFFKV_DBG_SEL") && step == 0) {
                    double n0=0,nv=0,nvlast=0; for (int i=0;i<F_test;++i){ float x=active_k_dense[0][i]; n0+=(double)x*x; float y=active_v_dense[0][i]; nv+=(double)y*y; float z=active_v_dense[0][(cnt0-1)*F_test+i]; nvlast+=(double)z*z; }
                    std::cerr << "[DBG_UPLOAD] step0 cnt0=" << cnt0 << " |active_k[tok0]|=" << std::sqrt(n0)
                              << " |active_v[tok0]|=" << std::sqrt(nv) << " |active_v[last]|=" << std::sqrt(nvlast) << std::endl;
                }
                std::vector<float> dmask(native_maxd, -INFINITY);
                for (int t = 0; t < cnt0; ++t) dmask[t] = 0.0f;
                ggml_backend_tensor_set(native_dense_mask, dmask.data(), 0, (size_t)native_maxd * sizeof(float));
                // Positions for in-graph RoPE of the raw past-dense keys (clamp/pad to 0).
                std::vector<int32_t> pbuf(native_maxd, 0);
                for (int t = 0; t < cnt0; ++t) pbuf[t] = active_positions_dense[t];
                ggml_backend_tensor_set(native_dense_pos, pbuf.data(), 0, (size_t)native_maxd * sizeof(int32_t));
                std::vector<float> kbuf((size_t)native_maxd * F_test), vbuf((size_t)native_maxd * F_test);
                for (int l = 0; l < n_layers; ++l) {
                    const int c2 = std::min(total_dense_tokens[l], native_maxd);
                    std::fill(kbuf.begin(), kbuf.end(), 0.0f);
                    std::fill(vbuf.begin(), vbuf.end(), 0.0f);
                    if (c2 > 0) {
                        std::memcpy(kbuf.data(), active_k_dense[l].data(), (size_t)c2 * F_test * sizeof(float)); // RAW K (roped in-graph)
                        std::memcpy(vbuf.data(), active_v_dense[l].data(), (size_t)c2 * F_test * sizeof(float));
                    }
                    ggml_backend_tensor_set(native_dense_kr[l], kbuf.data(), 0, (size_t)native_maxd * F_test * sizeof(float));
                    ggml_backend_tensor_set(native_dense_v[l],  vbuf.data(), 0, (size_t)native_maxd * F_test * sizeof(float));
                }
                if (std::getenv("DIFFKV_DBG_SEL") && step == 0) {
                    std::vector<float> rb(F_test), rv(F_test);
                    ggml_backend_tensor_get(native_dense_kr[0], rb.data(), 0, F_test*sizeof(float));
                    ggml_backend_tensor_get(native_dense_v[0], rv.data(), 0, F_test*sizeof(float));
                    double n=0,nv=0; for(int i=0;i<F_test;++i){ n+=(double)rb[i]*rb[i]; nv+=(double)rv[i]*rv[i]; }
                    std::cerr << "[DBG_DEV] native_dense_kr[0] dev |tok0|=" << std::sqrt(n) << " native_dense_v[0] dev |tok0|=" << std::sqrt(nv) << std::endl;
                }
            }

            auto t_before_compute = std::chrono::high_resolution_clock::now();
            ggml_status decode_st = native_attn_on
                ? ggml_backend_graph_compute(backend, decode_graph)
                : ggml_backend_sched_graph_compute(sched, decode_graph);
            if (decode_st != GGML_STATUS_SUCCESS) {
                std::cerr << "Error: Decode graph compute failed at step " << step << std::endl;
                break;
            }
            auto t_after_compute = std::chrono::high_resolution_clock::now();

            // DEFINITIVE native-vs-CPU sparse check (run with DIFFKV_DBG_DENSEOFF=1 so native attn_out
            // is sparse-only). Feeds the SAME in-graph q_rope + selected_slots into execute_cpu_attention
            // (host pool) and diffs against the native L0 output. maxAbsDiff≈0 ⇒ inputs match (math proven);
            // large ⇒ a device/host input diverges for these exact slots.
            if (std::getenv("DIFFKV_DBG_CMP") && step < 12 && g_dbg_attn0 && g_dbg_qrope && g_dbg_sel0) {
                const auto & cfg = model.get_config();
                const char* clp = std::getenv("DIFFKV_DBG_CMP_LAYER"); int cmpL = clp ? atoi(clp) : 0;
                int nq=cfg.n_head, nkv=cfg.n_head_kv, D=cfg.n_embd/cfg.n_head, K=(int)g_dbg_sel0->ne[0];
                int rank=kv_engines[cmpL]->get_rank(); float scale=1.0f/std::sqrt((float)D), freq=cfg.rope_freq_base;
                std::vector<float> qh((size_t)nq*D); ggml_backend_tensor_get(g_dbg_qrope, qh.data(), 0, qh.size()*sizeof(float));
                std::vector<int32_t> sl(K); ggml_backend_tensor_get(g_dbg_sel0, sl.data(), 0, (size_t)K*sizeof(int32_t));
                std::vector<float> natv((size_t)nq*D); ggml_backend_tensor_get(g_dbg_attn0, natv.data(), 0, natv.size()*sizeof(float));
                std::vector<float> outS((size_t)nq*D,0.0f), lseS(nq,-1e30f);
                diffkv::execute_cpu_attention(qh.data(), sl.data(), outS.data(), lseS.data(), kv_engines[cmpL].get(),
                                              nq, nkv, rank, 64, K, D, scale, true, freq, true);
                // FULL ref = sparse ⊕ dense(active_k_dense[0] + current token) via 3-way LSE combine.
                int F=nkv*D, Td=total_dense_tokens[cmpL];
                // Match the real callback: ignore_c=true → DO NOT attend the current token here
                // (it's appended to active_k_dense after compute, for the NEXT step). DIFFKV_DBG_CMP_CUR
                // re-adds it to A/B the effect.
                bool add_cur = std::getenv("DIFFKV_DBG_CMP_CUR") != nullptr;
                int Tc = Td + (add_cur?1:0);
                std::vector<float> dk((size_t)Tc*F), dv((size_t)Tc*F); std::vector<int32_t> dp(Tc);
                std::memcpy(dk.data(), active_k_dense[cmpL].data(), (size_t)Td*F*sizeof(float));
                std::memcpy(dv.data(), active_v_dense[cmpL].data(), (size_t)Td*F*sizeof(float));
                for(int t=0;t<Td;++t)dp[t]=active_positions_dense[t];
                if (add_cur) {
                    std::vector<float> ck((size_t)F), cv((size_t)F);
                    ggml_backend_tensor_get(g_dbg_curk, ck.data(), 0, ck.size()*sizeof(float));
                    ggml_backend_tensor_get(g_dbg_curv, cv.data(), 0, cv.size()*sizeof(float));
                    std::memcpy(dk.data()+(size_t)Td*F, ck.data(), (size_t)F*sizeof(float));
                    std::memcpy(dv.data()+(size_t)Td*F, cv.data(), (size_t)F*sizeof(float));
                    dp[Td]=current_pos;
                }
                std::vector<float> outD((size_t)nq*D,0.0f), lseD(nq,-1e30f);
                diffkv::cpu_dense_attention(qh.data(), dk.data(), dv.data(), dp.data(), Tc, nq, nkv, D, scale, true, freq, 0, outD.data(), lseD.data());
                std::vector<float> cpuv((size_t)nq*D);
                for(int h=0;h<nq;++h){ double lmax=std::max(lseS[h],lseD[h]); double ws=(lseS[h]<=-1e20?0.0:std::exp(lseS[h]-lmax)), wd=(lseD[h]<=-1e20?0.0:std::exp(lseD[h]-lmax)); double den=std::max(ws+wd,1e-9); for(int d=0;d<D;++d) cpuv[h*D+d]=(float)((outS[h*D+d]*ws+outD[h*D+d]*wd)/den); }
                double md=0,nn=0,cn=0; int worst=0; for(int i=0;i<nq*D;++i){ double d=std::fabs((double)natv[i]-cpuv[i]); if(d>md){md=d;worst=i;} nn+=(double)natv[i]*natv[i]; cn+=(double)cpuv[i]*cpuv[i]; }
                std::cerr<<"[DBG_CMP] step="<<step<<" FULL L"<<cmpL<<" (K="<<K<<" Td="<<Td<<"): |native|="<<std::sqrt(nn)<<" |cpu|="<<std::sqrt(cn)<<" maxAbsDiff="<<md<<" @"<<worst<<" head"<<worst/D<<std::endl;
            }

            if (std::getenv("DIFFKV_DBG_SEL") && step == 0 && dbg_anc) {
                std::vector<float> a(ggml_nelements(dbg_anc));
                ggml_backend_tensor_get(dbg_anc, a.data(), 0, a.size()*sizeof(float));
                double mn=1e30,mx=-1e30,ss=0; for (float x : a){ mn=std::min(mn,(double)x); mx=std::max(mx,(double)x); ss+=(double)x*x; }
                std::cerr << "[DBG_ANC] L0 anchor scores: n=" << a.size() << " min=" << mn << " max=" << mx << " rms=" << std::sqrt(ss/a.size()) << " first8=";
                for (int i=0;i<8 && i<(int)a.size();++i) std::cerr << a[i] << " ";
                std::cerr << std::endl;
            }
            if (std::getenv("DIFFKV_DBG_SEL") && step == 0 && decode_selected_slots) {
                std::vector<int32_t> sel(decode_selected_slots->ne[0]);
                ggml_backend_tensor_get(decode_selected_slots, sel.data(), 0, sel.size()*sizeof(int32_t));
                // Reliably check DEVICE anchorK_rot norm per selected slot vs HOST host_anchors_K
                // (what execute_cpu_attention reads). A device-zero-but-host-nonzero slot = the bug.
                { ggml_tensor* aKr=kv_engines[0]->get_anchorK_rot(); const ggml_fp16_t* hAK=kv_engines[0]->get_host_anchors_K();
                  const int32_t* sls=kv_engines[0]->get_host_seq_lens(); int Fkv=kv_heads*head_dim;
                  std::cerr<<"[DBG_DEVSLOT] slot(seq:devAKr/hostAK): ";
                  for(int i=0;i<(int)sel.size() && i<10;++i){ int s=sel[i]; std::vector<float> d(Fkv); ggml_backend_tensor_get(aKr,d.data(),(size_t)s*Fkv*sizeof(float),Fkv*sizeof(float));
                    double nd=0,nh=0; for(int j=0;j<Fkv;++j){ float x=d[j]; nd+=(double)x*x; float y=ggml_fp16_to_fp32(hAK[(size_t)s*Fkv+j]); nh+=(double)y*y; }
                    std::cerr<<s<<"(sl"<<sls[s]<<":"<<std::sqrt(nd)<<"/"<<std::sqrt(nh)<<") "; }
                  std::cerr<<std::endl; }
                if (dbg_anc) {
                    // dkr is [D, MAXD]; dump norm of token 0 (column 0) and token 1.
                    std::vector<float> a(ggml_nelements(dbg_anc));
                    ggml_backend_tensor_get(dbg_anc, a.data(), 0, a.size()*sizeof(float));
                    // a is [D, nq]=[64,14]: head h = column h. norm per head.
                    const int D=head_dim, nq=a.size()/D; double tot=0;
                    std::cerr << "[DBG_KVOUT] per-head norms: ";
                    for(int h=0;h<nq;++h){ double nh=0; for(int d=0;d<D;++d){ double x=a[d+h*D]; nh+=x*x; tot+=x*x; } std::cerr<<std::sqrt(nh)<<" "; }
                    std::cerr << " | total="<<std::sqrt(tot)<<std::endl;
                }
            }

            std::vector<float> output_logits(n_vocab);
            ggml_backend_tensor_get(decode_logits, output_logits.data(), 0, n_vocab * sizeof(float));
            auto t_after_logits = std::chrono::high_resolution_clock::now();

            auto t_before_kv_get = std::chrono::high_resolution_clock::now();
            std::vector<float> concat_k_host(n_layers * F_test);
            std::vector<float> concat_v_host(n_layers * F_test);
            ggml_backend_tensor_get(decode_concat_k, concat_k_host.data(), 0, n_layers * F_test * sizeof(float));
            ggml_backend_tensor_get(decode_concat_v, concat_v_host.data(), 0, n_layers * F_test * sizeof(float));
            auto t_after_kv_get = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<float>> decode_k(n_layers, std::vector<float>(F_test));
            std::vector<std::vector<float>> decode_v(n_layers, std::vector<float>(F_test));
            for (int l = 0; l < n_layers; ++l) {
                std::memcpy(decode_k[l].data(), concat_k_host.data() + l * F_test, F_test * sizeof(float));
                std::memcpy(decode_v[l].data(), concat_v_host.data() + l * F_test, F_test * sizeof(float));
            }

            if (std::getenv("DIFFKV_DBG_SEL") && step == 0) {
                double nk=0,nv=0,nk0=0,nv0=0; for(int i=0;i<F_test;++i){ nk+=(double)decode_k[0][i]*decode_k[0][i]; nv+=(double)decode_v[0][i]*decode_v[0][i]; if(i<head_dim){nk0+=(double)decode_k[0][i]*decode_k[0][i]; nv0+=(double)decode_v[0][i]*decode_v[0][i];} }
                std::cerr << "[DBG_CURV] current token L0: |k|="<<std::sqrt(nk)<<" |v|="<<std::sqrt(nv)<<" |k[0..63]|="<<std::sqrt(nk0)<<" |v[0..63]|="<<std::sqrt(nv0)<<std::endl;
            }
            runtime_manager.ingest_decode(decode_k, decode_v, current_pos, all_tokens, &srl_state);

            {
                std::vector<float> k_avg(head_dim, 0.0f);
                for (int h = 0; h < kv_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        k_avg[d] += decode_k[0][h * head_dim + d];
                    }
                }
                for (int d = 0; d < head_dim; ++d) {
                    k_avg[d] /= kv_heads;
                }
                srl_state.recent_decode_keys.push_back(k_avg);
                if (srl_state.recent_decode_keys.size() > 512) {
                    srl_state.recent_decode_keys.erase(srl_state.recent_decode_keys.begin());
                }

                int32_t current_slot_id = 0;
                if (!srl_state.ordered_slot_ids.empty()) {
                    current_slot_id = srl_state.ordered_slot_ids.back();
                }
                srl_state.generated_token_slots.push_back(current_slot_id);
                srl_state.update_dynamic_anchors(stop_token_ids);
            }

            // Compute cos/sin for the current token position: current_pos
            int half_dim = head_dim / 2;
            std::vector<float> cos_tok(half_dim);
            std::vector<float> sin_tok(half_dim);
            for (int i = 0; i < half_dim; ++i) {
                float theta = (float)current_pos / std::pow(model.get_config().rope_freq_base, (float)(2 * i) / (float)head_dim);
                cos_tok[i] = std::cos(theta);
                sin_tok[i] = std::sin(theta);
            }

            // Manual append newly ingested token to host-side dense resident buffers
            for (int l = 0; l < n_layers; ++l) {
                int offset = total_dense_tokens[l] * F_test;
                if (offset + F_test <= (int)active_k_dense[l].size()) {
                    std::memcpy(active_k_dense[l].data() + offset, decode_k[l].data(), F_test * sizeof(float));
                    std::memcpy(active_v_dense[l].data() + offset, decode_v[l].data(), F_test * sizeof(float));

                    apply_rope_neox_cpu_fast(
                        decode_k[l].data(),
                        active_k_dense_rotated[l].data() + offset,
                        cos_tok.data(),
                        sin_tok.data(),
                        1, kv_heads, head_dim
                    );

                    total_dense_tokens[l]++;
                }
            }
            if (total_positions < (int)active_positions_dense.size()) {
                active_positions_dense[total_positions++] = current_pos;
            }
            auto t_step_end = std::chrono::high_resolution_clock::now();

            int32_t current_entity = srl_state.current_entity_id;
            const auto& entity_ids = srl_state.current_step_sequence_entity_ids;
            const auto& is_prime_list = srl_state.current_step_sequence_is_prime;

            // +7.0 factual token bias (raised from +3)
            if (!srl_state.current_step_factual_tokens.empty()) {
                if (current_entity != -1) {
                    std::unordered_set<int32_t> entity_factual_tokens;
                    for (size_t i = 0; i < srl_state.current_step_factual_sequences.size(); ++i) {
                        int32_t seq_eid = (i < entity_ids.size()) ? entity_ids[i] : -1;
                        bool seq_is_prime = (i < is_prime_list.size()) ? is_prime_list[i] : false;
                        if (seq_eid == -1 || seq_eid == current_entity || seq_is_prime) {
                            entity_factual_tokens.insert(srl_state.current_step_factual_sequences[i].begin(),
                                                         srl_state.current_step_factual_sequences[i].end());
                        }
                    }
                    for (int32_t tok_id : entity_factual_tokens) {
                        if (tok_id >= 0 && tok_id < n_vocab) {
                            output_logits[tok_id] += 7.0f;
                        }
                    }
                } else {
                    for (int32_t tok_id : srl_state.current_step_factual_tokens) {
                        if (tok_id >= 0 && tok_id < n_vocab) {
                            output_logits[tok_id] += 7.0f;
                        }
                    }
                }
            }

            // RC8 — generation-time binding validator.  While locked to an entity,
            // penalise tokens that belong exclusively to OTHER entities so the model
            // cannot emit "EP2 has codimension 3".  Also fires in comparison mode
            // (RC5 locks one entity per block) and escalates once retrieval is
            // confident.
            if (current_entity != -1 && !srl_state.current_step_factual_sequences.empty()) {
                std::unordered_set<int32_t> licensed, foreign;
                diffkv::compute_entity_token_license(
                    srl_state.current_step_factual_sequences,
                    entity_ids, is_prime_list, current_entity, licensed, foreign);
                float pen = (srl_state.current_step_max_similarity >= 0.70f) ? 12.0f : 4.0f;
                for (int32_t tok_id : foreign) {
                    if (tok_id >= 0 && tok_id < n_vocab) output_logits[tok_id] -= pen;
                }
            }

            // +7.0 VSL active-candidate boost: when VSL is tracking a suffix, the
            // exact next token is confirmed. Give it a decisive advantage.
            for (const auto& suffix : srl_state.vsl_active_candidates) {
                if (!suffix.empty() && suffix[0] >= 0 && suffix[0] < n_vocab) {
                    output_logits[suffix[0]] += 7.0f;
                }
            }

            // -3.5 anti-hallucination penalty: only at sim ≥ 0.55 — at 0.3/0.4 the
            // excluded set is almost the full domain vocabulary (every entry in a focused
            // document passes the lower bar) making the penalty meaningless and the model
            // unable to generate anything outside the retrieved-but-wrong sequences.
            if (srl_state.current_step_max_similarity >= 0.55f &&
                !srl_state.dual_entity_mode &&
                !srl_state.current_step_factual_tokens.empty()) {
                const auto& helper_ids_penalty = diffkv::get_helper_token_ids_cpp(model);
                for (int i = 0; i < n_vocab; ++i) {
                    if (srl_state.current_step_factual_tokens.count(i) == 0 &&
                        helper_ids_penalty.count(i) == 0) {
                        output_logits[i] -= 3.5f;
                    }
                }
            }

            // +10.0 transition bias — only when last_token is a CONTENT word (not a
            // helper word). Helper words appear at many positions across all sequences;
            // boosting their successors adds random noise and drives wrong continuations.
            if (last_token >= 0 && !srl_state.current_step_factual_sequences.empty()) {
                const auto& helper_ids_trans = diffkv::get_helper_token_ids_cpp(model);
                if (helper_ids_trans.count(last_token) == 0) {
                    std::unordered_set<int32_t> transition_candidates;
                    for (size_t i = 0; i < srl_state.current_step_factual_sequences.size(); ++i) {
                        int32_t seq_entity = (i < entity_ids.size()) ? entity_ids[i] : -1;
                        if (current_entity != -1 && seq_entity != -1 && seq_entity != current_entity) {
                            continue; // skip cross-entity transitions
                        }
                        const auto& seq = srl_state.current_step_factual_sequences[i];
                        if (seq.size() > 1) {
                            for (size_t idx = 0; idx < seq.size() - 1; ++idx) {
                                if (seq[idx] == last_token) {
                                    transition_candidates.insert(seq[idx + 1]);
                                }
                            }
                        }
                    }
                    for (int32_t tok_id : transition_candidates) {
                        if (tok_id >= 0 && tok_id < n_vocab) {
                            output_logits[tok_id] += 10.0f;
                        }
                    }
                }
            }
            t_after_logits = std::chrono::high_resolution_clock::now();

            float rep_penalty = repetition_penalty;
            auto is_alphanumeric_token = [&](int32_t tok_id) -> bool {
                if (tok_id < 0 || tok_id >= n_vocab) return false;
                return alnum_cache[tok_id] == 1;
            };

            std::unordered_set<int32_t> unique_penalized;
            int gen_start = std::max(0, (int)generated_tokens.size() - 64);
            for (size_t i = gen_start; i < generated_tokens.size(); ++i) {
                int32_t tok = generated_tokens[i];
                if (is_alphanumeric_token(tok)) {
                    unique_penalized.insert(tok);
                }
            }
            if (is_alphanumeric_token(last_token)) {
                unique_penalized.insert(last_token);
            }

            for (int32_t tok : unique_penalized) {
                if (tok >= 0 && tok < n_vocab) {
                    float& l = output_logits[tok];
                    l = (l > 0.0f) ? l / rep_penalty : l * rep_penalty;
                }
            }

            // F22 fix: drop factual sequences that are a strict PREFIX of another
            // surfaced sequence (the 20-token chunker can leave a fragment like
            // "84729" beside the full span "847291…"; the fragment lets the VSL
            // "complete" early and truncate). The longer sequence covers the shorter.
            {
                auto& seqs = srl_state.current_step_factual_sequences;
                if (seqs.size() > 1) {
                    std::vector<bool> drop(seqs.size(), false);
                    for (size_t a = 0; a < seqs.size(); ++a) {
                        for (size_t b = 0; b < seqs.size(); ++b) {
                            if (a == b || drop[b]) continue;
                            if (seqs[a].size() > seqs[b].size()) continue;
                            if (seqs[a].size() == seqs[b].size() && a < b) continue;
                            if (std::equal(seqs[a].begin(), seqs[a].end(), seqs[b].begin())) { drop[a] = true; break; }
                        }
                    }
                    std::vector<std::vector<int32_t>> kept;
                    for (size_t i = 0; i < seqs.size(); ++i) if (!drop[i]) kept.push_back(std::move(seqs[i]));
                    seqs.swap(kept);
                }
            }

            // SFA threshold raised 0.3→0.55: at 0.3 almost every entry in a topically
            // focused document matches, activating the VSL and merging all categories'
            // tokens into one set. At 0.55 only high-confidence specific retrieval
            // triggers the sequence constraint.
            sfa_active = (srl_state.current_step_max_similarity >= 0.55f &&
                          !srl_state.current_step_factual_sequences.empty());

            // LM-VSL (Logit Masking) — graduated by retrieval confidence.
            // sim 0.55–0.69 → soft (-7): model can escape if LM distribution is strong.
            // sim ≥ 0.70    → hard (-1e10): verbatim extraction — with sequence-start-only
            //   fallback in get_allowed_tokens_vsl_cpp, the model must enter factual sequences
            //   from their first token and advance in order, fixing entity binding failure.
            if (sfa_active && !srl_state.current_step_factual_sequences.empty()) {
                const auto& helper_ids = diffkv::get_helper_token_ids_cpp(model);
                const auto& structural_ids = diffkv::get_structural_helper_token_ids_cpp(model);
                auto allowed = diffkv::get_allowed_tokens_vsl_cpp(
                    srl_state, helper_ids, &structural_ids, /*sfa_active=*/true);
                float max_sim = srl_state.current_step_max_similarity;
                // F25: never mask a token that IS part of a surfaced factual sequence.
                // The allowed set is only helpers + sequence-START tokens, so a
                // mid-sequence content token (e.g. a digit of "847291" when the model
                // answers directly) would otherwise be hard-masked while helpers like
                // "." stay free → truncation. Exempting factual tokens keeps the VSL's
                // anti-hallucination intent without blocking verbatim factual content.
                const auto& fact_toks = srl_state.current_step_factual_tokens;
                for (int i = 0; i < n_vocab; ++i) {
                    if (allowed.count(i) == 0 && fact_toks.count(i) == 0) {
                        if (max_sim >= 0.70f) {
                            output_logits[i] = -1e10f;   // hard: verbatim
                        } else {
                            output_logits[i] -= 7.0f;    // soft: guided
                        }
                    }
                }
            }

            int32_t next_token = 0;
            if (interactive) {
                float effective_temperature = temperature;
                // Dynamic temperature threshold raised 0.3→0.55 to match tighter SFA bar.
                if (srl_state.current_step_max_similarity >= 0.55f) {
                    effective_temperature = temperature * (1.0f - srl_state.current_step_max_similarity * 0.95f);
                }
                next_token = sample_logits(output_logits, effective_temperature, top_p, sample_rng);
            } else {
                std::vector<std::pair<float, int>> logits_sorted;
                logits_sorted.reserve(n_vocab);
                for (int i = 0; i < n_vocab; ++i) {
                    logits_sorted.push_back({output_logits[i], i});
                }
                std::partial_sort(logits_sorted.begin(), logits_sorted.begin() + 5, logits_sorted.end(),
                                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first > b.first;
                });
                next_token = logits_sorted[0].second;

                std::cerr << "\n[Step " << step << " Top predictions]:\n";
                for (int i = 0; i < std::min(5, n_vocab); ++i) {
                    std::cerr << "  " << i << ": \"" << model.token_to_piece(logits_sorted[i].second) << "\" (id: " << logits_sorted[i].second << ", logit: " << logits_sorted[i].first << ")\n";
                }
            }

            if (model.is_eog_token(next_token) || next_token == model.token_eos()) {
                if (!interactive) {
                    std::cerr << " [EOS]" << std::endl;
                }
                break;
            }

            // Strict Factual Alignment (SFA) State Update and Loop Check
            if (sfa_active) {
                const auto& helper_ids = diffkv::get_helper_token_ids_cpp(model);
                diffkv::update_vsl_state_cpp(next_token, srl_state, helper_ids);
                
                if (srl_state.vsl_consecutive_helpers >= 16) {
                    std::string uncertainty_str = " [uncertain: details missing in source]";
                    std::vector<int32_t> uncertainty_toks = model.tokenize(uncertainty_str, false);
                    for (int32_t t : uncertainty_toks) {
                        if (!is_warmup_run) {
                            std::cout << model.token_to_piece(t);
                        }
                        generated_tokens.push_back(t);
                        all_tokens.push_back(t);
                    }
                    if (!is_warmup_run) {
                        std::cout << std::flush;
                    }
                    break;
                }
            }

            // Factual Early Stopping (Option 2 Extension)
            bool stop_generation = false;
            if (max_generate < 64 && srl_state.current_step_max_similarity >= 0.4f) {
                for (const auto& seq : srl_state.current_step_factual_sequences) {
                    if (seq.size() >= 5 && next_token == seq.back() && generated_tokens.size() >= seq.size() - 1) {
                        bool match = true;
                        size_t offset = generated_tokens.size() - (seq.size() - 1);
                        for (size_t k = 0; k < seq.size() - 1; ++k) {
                            if (generated_tokens[offset + k] != seq[k]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            stop_generation = true;
                            break;
                        }
                    }
                }
            }
            if (stop_generation) {
                if (!interactive) {
                    std::cerr << " [Factual Early Stop]" << std::endl;
                }
                break;
            }

            std::string piece = model.token_to_piece(next_token);
            if (!is_warmup_run) {
                std::cout << piece << std::flush;
            }

            srl_state.update_generated_tokens(next_token);
            generated_tokens.push_back(next_token);
            all_tokens.push_back(next_token);
            last_token = next_token;

            // ── Factual store query ──────────────────────────────────────────
            // Use layer-0 decode K as a proxy Q vector to find matching factual
            // spans. Results populate current_step_factual_tokens/sequences and
            // current_step_max_similarity, which drive logit biasing (+3/+4),
            // VSL masking, temperature reduction, and SFA on the NEXT decode step.
            // (One-step lag is unavoidable: Q is inside the GGML graph; K is the
            //  best available proxy extracted from the same token embedding.)
            if (!srl_state.factual_store.entries.empty()) {
                std::unordered_set<int32_t> active_slots_set(
                    cached_physical_candidates.begin(),
                    cached_physical_candidates.end()
                );
                const std::unordered_set<int32_t>* slot_filter =
                    active_slots_set.empty() ? nullptr : &active_slots_set;

                // ── Query Anchor Blending ─────────────────────────────────────
                // Store layer-0 decode-K from the first decode step as a stable
                // anchor. On subsequent steps blend 20% current + 80% anchor so
                // accumulated generated-token context cannot pull retrieval away
                // from the original question topic (mirrors Python 0.20/0.80 blend).
                std::vector<float> q_for_factual(F_test);
                if (srl_state.factual_anchor_q.empty()) {
                    srl_state.factual_anchor_q = decode_k[0];
                    q_for_factual = decode_k[0];

                    // ── Early Entity Binding (Component 4) ────────────────────
                    // Analyze query tokens against prime entries at the very start.
                    if (!srl_state.current_query_tokens.empty()) {
                        std::unordered_set<int32_t> query_toks(srl_state.current_query_tokens.begin(), srl_state.current_query_tokens.end());
                        struct PrimeMatch {
                            int32_t start_idx;
                            int overlap;
                        };
                        std::vector<PrimeMatch> prime_matches;
                        for (const auto& fe : srl_state.factual_store.entries) {
                            if (fe.is_prime) {
                                int overlap = 0;
                                for (int32_t t : fe.tokens) {
                                    if (query_toks.count(t)) overlap++;
                                }
                                if (overlap >= 1) {
                                    prime_matches.push_back({fe.start_idx, overlap});
                                }
                            }
                        }
                        if (prime_matches.size() == 1) {
                            srl_state.current_entity_id = prime_matches[0].start_idx;
                            srl_state.dual_entity_mode = false;
                        } else if (prime_matches.size() >= 2) {
                            std::sort(prime_matches.begin(), prime_matches.end(), [](const PrimeMatch& a, const PrimeMatch& b) {
                                return a.overlap > b.overlap;
                            });
                            srl_state.dual_entity_mode = true;
                            srl_state.dual_entity_ids = {prime_matches[0].start_idx, prime_matches[1].start_idx};
                            // RC5: sequence the comparison as per-entity blocks and
                            // lock to the first entity now (instead of leaving entity
                            // context open, which lets the two interleave).
                            srl_state.comparison_entities = srl_state.dual_entity_ids;
                            srl_state.comparison_active_idx = 0;
                            srl_state.comparison_covered.clear();
                            srl_state.current_entity_id = srl_state.comparison_entities[0];
                        }
                    }
                } else {
                    for (int qi = 0; qi < F_test; ++qi) {
                        q_for_factual[qi] = 0.20f * decode_k[0][qi]
                                          + 0.80f * srl_state.factual_anchor_q[qi];
                    }
                }

                // RC3 — tell the store which entities the query is about so it
                // ranks their spans above shared-vocabulary spans of other entities.
                std::unordered_set<int32_t> qbias;
                if (srl_state.dual_entity_mode && !srl_state.dual_entity_ids.empty()) {
                    qbias.insert(srl_state.dual_entity_ids.begin(), srl_state.dual_entity_ids.end());
                } else if (srl_state.current_entity_id != -1) {
                    qbias.insert(srl_state.current_entity_id);
                }
                const std::unordered_set<int32_t>* qbias_ptr = qbias.empty() ? nullptr : &qbias;

                auto fact_hits = srl_state.factual_store.query(
                    q_for_factual.data(),
                    kv_heads,
                    head_dim,
                    W_proj_host.data(),
                    desc_dim,
                    0.50f,
                    slot_filter,
                    qbias_ptr
                );

                srl_state.current_step_factual_tokens.clear();
                srl_state.current_step_factual_sequences.clear();
                srl_state.current_step_max_similarity = 0.0f;
                srl_state.current_step_sequence_entity_ids.clear();
                srl_state.current_step_sequence_is_prime.clear();
                srl_state.current_step_sequence_prefixes.clear();

                // Helper lambda to add a sequence if not already present
                auto add_seq = [&](const std::vector<int32_t>& seq) {
                    if (seq.empty()) return;
                    for (const auto& s : srl_state.current_step_factual_sequences)
                        if (s == seq) return;
                    for (int32_t t : seq) srl_state.current_step_factual_tokens.insert(t);
                    srl_state.current_step_factual_sequences.push_back(seq);
                };

                for (const auto& hit : fact_hits) {
                    for (int32_t tok_id : hit.tokens) {
                        srl_state.current_step_factual_tokens.insert(tok_id);
                    }
                    if (!hit.tokens.empty()) {
                        add_seq(hit.tokens);
                    }
                    if (hit.current_sim > srl_state.current_step_max_similarity) {
                        srl_state.current_step_max_similarity = hit.current_sim;
                    }
                    // RC1 — inject triple sequences from prime entries so the VSL
                    // can lock onto the complete (bridge + value) ordering, preventing
                    // freely hallucinated relational connectives.
                    if (hit.is_prime) {
                        for (const auto& triple_seq : hit.triple_sequences) {
                            add_seq(triple_seq);
                        }
                    }
                    // ── 1-hop neighbor injection ────────────────────────────────
                    for (size_t ni = 0; ni < hit.neighbors.size(); ++ni) {
                        int nb_idx = hit.neighbors[ni];
                        float nb_w  = hit.weights[ni];
                        if (nb_w >= 0.45f && nb_idx < (int)srl_state.factual_store.entries.size()) {
                            const auto& nb_e = srl_state.factual_store.entries[nb_idx];
                            add_seq(nb_e.tokens);
                            if (nb_e.is_prime) {
                                for (const auto& ts : nb_e.triple_sequences) add_seq(ts);
                            }
                            // ── 2-hop neighbor injection ────────────────────
                            for (size_t ni2 = 0; ni2 < nb_e.neighbors.size(); ++ni2) {
                                int nb2_idx = nb_e.neighbors[ni2];
                                float nb2_w  = nb_e.weights[ni2];
                                if (nb2_w >= 0.65f && nb2_idx < (int)srl_state.factual_store.entries.size()) {
                                    const auto& nb2_e = srl_state.factual_store.entries[nb2_idx];
                                    add_seq(nb2_e.tokens);
                                }
                            }
                        }
                    }
                }

                // ── Lexical Tripwire ──────────────────────────────────────────────
                // If the last generated token is high-IDF (≥2.5), inject any
                // factual entries that contain it and whose slot_ids overlap the
                // inverted-index occurrences for that token. This re-anchors the
                // VSL token set to the relevant fact spans when a content-bearing
                // token was just generated (mirrors Python Lexical Tripwire).
                if (!srl_state.recent_generated_tokens.empty()) {
                    int32_t last_tok = srl_state.recent_generated_tokens.back();
                    auto idf_it = srl_state.inverted_index.idf.find(last_tok);
                    if (idf_it != srl_state.inverted_index.idf.end() && idf_it->second >= 2.5f) {
                        auto occ_it = srl_state.inverted_index.occurrences.find(last_tok);
                        if (occ_it != srl_state.inverted_index.occurrences.end()) {
                            std::unordered_set<int32_t> occ_slots;
                            for (const auto& occ : occ_it->second) {
                                occ_slots.insert(std::get<0>(occ));
                            }
                            for (const auto& fe : srl_state.factual_store.entries) {
                                bool has_tok = false;
                                for (int32_t t : fe.tokens) {
                                    if (t == last_tok) { has_tok = true; break; }
                                }
                                bool has_slot = false;
                                for (int32_t s : fe.slot_ids) {
                                    if (occ_slots.count(s)) { has_slot = true; break; }
                                }
                                if (has_tok && has_slot) {
                                    bool already = false;
                                    for (const auto& s : srl_state.current_step_factual_sequences) {
                                        if (s == fe.tokens) { already = true; break; }
                                    }
                                    if (!already) {
                                        for (int32_t t : fe.tokens) srl_state.current_step_factual_tokens.insert(t);
                                        srl_state.current_step_factual_sequences.push_back(fe.tokens);
                                    }
                                    // 1-hop from tripwire entry
                                    for (size_t ni = 0; ni < fe.neighbors.size(); ++ni) {
                                        int nb_idx = fe.neighbors[ni];
                                        float nb_w  = (ni < fe.weights.size()) ? fe.weights[ni] : 0.0f;
                                        if (nb_w >= 0.45f && nb_idx < (int)srl_state.factual_store.entries.size()) {
                                            const auto& nb_e = srl_state.factual_store.entries[nb_idx];
                                            if (!nb_e.tokens.empty()) {
                                                bool alr = false;
                                                for (const auto& s : srl_state.current_step_factual_sequences) {
                                                    if (s == nb_e.tokens) { alr = true; break; }
                                                }
                                                if (!alr) {
                                                    for (int32_t t : nb_e.tokens) srl_state.current_step_factual_tokens.insert(t);
                                                    srl_state.current_step_factual_sequences.push_back(nb_e.tokens);
                                                }
                                            }
                                        }
                                    }
                                    break; // only inject from first matching entry
                                }
                            }
                        }
                    }
                }

                // ── Contrastive Category Anchor, Coherence Cap, & Entity-Subgraph Tagging ──────
                diffkv::process_and_tag_vsl_step(srl_state);
            }
            // ────────────────────────────────────────────────────────────────

            srl_state.update_query_segment(next_token);
            srl_state.save_step_state(L + generated_tokens.size());
            t_step_end = std::chrono::high_resolution_clock::now();

            if (time_decode) {
                double total_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
                double retrieval_ms = std::chrono::duration<double, std::milli>(t_after_retrieval - t_before_retrieval).count();
                double compute_ms = std::chrono::duration<double, std::milli>(t_after_compute - t_before_compute).count();
                double logits_ms = std::chrono::duration<double, std::milli>(t_after_logits - t_after_compute).count();
                double kv_get_ms = std::chrono::duration<double, std::milli>(t_after_kv_get - t_before_kv_get).count();
                double ingest_ms = std::chrono::duration<double, std::milli>(t_step_end - t_after_kv_get).count();
                double metal_wait_ms = diffkv::get_and_reset_accumulated_wait_ms();
                std::cerr << "[Timing Step " << step << "] Retrieval: " << retrieval_ms 
                          << "ms | Compute: " << compute_ms 
                          << "ms (Metal Wait: " << metal_wait_ms << "ms) | Logits: " << logits_ms 
                          << "ms | KV Get: " << kv_get_ms 
                          << "ms | Ingest: " << ingest_ms 
                          << "ms | Total: " << total_ms << "ms" << std::endl;
            }

            int old_active_slot = active_slot;
            active_slot = runtime_manager.get_ingest_manager().get_blocks(0).size() - 1;
            if (active_slot < 0) {
                active_slot = 0;
            }
            active_block_tokens = 0;
            if (!runtime_manager.get_ingest_manager().get_blocks(0).empty()) {
                active_block_tokens = runtime_manager.get_ingest_manager().get_blocks(0).back()->token_count();
                if (active_block_tokens > 0) {
                    active_block_tokens--; // exclude anchor
                }
            }


            // ── Dense buffer sync is handled inside the new-block guard below ──
            // Only sync when active_slot advances (a new block was sealed).
            // Removed the O(N_blocks × N_layers) per-token memcpy that was here.

            // ── Dense buffer sync: only rebuild when a new block was sealed ──
            // The dense buffer changes only when active_slot advances (a new block
            // was created). Rebuilding it every token causes O(N_blocks × N_layers)
            // memcpy per step — dominant CPU overhead when N_blocks is large.
            if (active_block_tokens == 0 && active_slot > old_active_slot) {
                int prev_slot = old_active_slot;
                auto & blocks = runtime_manager.get_ingest_manager().get_blocks(0);
                if (prev_slot >= 0 && prev_slot < (int)blocks.size()) {
                    auto & block = blocks[prev_slot];
                    
                    // Update descriptors (computes raw averages for Compressing blocks, SVD descriptors for Compressed blocks)
                    runtime_manager.update_descriptors(W_proj_host, desc_dim, head_dim);

                    std::vector<float> desc(desc_dim);
                    const float* host_desc = runtime_manager.get_engines()[0]->get_host_desc_matrix();
                    std::memcpy(desc.data(), host_desc + block->pool_idx * desc_dim, desc_dim * sizeof(float));

                    update_srl_from_compressed_block(
                        srl_state,
                        desc.data(),
                        block->pool_idx, // Physical slot ID
                        all_tokens.data() + block->anchor_idx, // Correct token IDs
                        block->token_count(),
                        block->anchor_idx,
                        stop_token_ids
                    );

                    // Rebuild chunk graph only over compressed slots (including those currently compressing)
                    auto & all_blocks = runtime_manager.get_ingest_manager().get_blocks(0);
                    std::vector<int32_t> cur_slots;
                    for (int i = 0; i < (int)all_blocks.size(); ++i) {
                        if (all_blocks[i]->pool_idx != -1 &&
                            (all_blocks[i]->state == BlockState::CompressedResident ||
                             all_blocks[i]->state == BlockState::CPUResident ||
                             all_blocks[i]->state == BlockState::Compressing)) {
                            cur_slots.push_back(all_blocks[i]->pool_idx); // Physical slot ID
                        }
                    }
                    int cur_N = cur_slots.size();
                    std::vector<float> cur_desc_matrix(cur_N * desc_dim);
                    for (int j = 0; j < cur_N; ++j) {
                        int slot_id = cur_slots[j];
                        std::memcpy(
                            cur_desc_matrix.data() + j * desc_dim,
                            host_desc + slot_id * desc_dim,
                            desc_dim * sizeof(float)
                        );
                    }

                    srl_state.chunk_graph = build_chunk_graph(
                        cur_desc_matrix.data(),
                        cur_slots.data(),
                        cur_N,
                        6, 2,
                        &srl_state.inverted_index,
                        0.15f
                    );
                }

                // When a block completes, also sync the dense buffer for ALL layers
                // (this is the correct place — before this guard it ran every token)
                for (int l = 0; l < n_layers; ++l) {
                    auto & b_list_l = runtime_manager.get_ingest_manager().get_blocks(l);
                    int curr_token_idx = 0;
                    bool found_first = false;

                    std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                    std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);

                    for (auto & block : b_list_l) {
                        if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                            if (!found_first) {
                                dense_start_positions[l] = block->anchor_idx;
                                found_first = true;
                            }
                            std::memcpy(active_k_dense[l].data() + curr_token_idx * F_test,
                                        block->anchor_k.data(), F_test * sizeof(float));
                            std::memcpy(active_v_dense[l].data() + curr_token_idx * F_test,
                                        block->anchor_v.data(), F_test * sizeof(float));
                            curr_token_idx++;
                            if (!block->active_k.empty()) {
                                int active_len = block->active_k.size() / F_test;
                                std::memcpy(active_k_dense[l].data() + curr_token_idx * F_test,
                                            block->active_k.data(), block->active_k.size() * sizeof(float));
                                std::memcpy(active_v_dense[l].data() + curr_token_idx * F_test,
                                            block->active_v.data(), block->active_v.size() * sizeof(float));
                                curr_token_idx += active_len;
                            }
                        }
                    }
                    total_dense_tokens[l] = curr_token_idx;
                }
                // Rebuild active_positions_dense
                {
                    auto & b_list = runtime_manager.get_ingest_manager().get_blocks(0);
                    int curr_pos_idx = 0;
                    std::fill(active_positions_dense.begin(), active_positions_dense.end(), 0);
                    for (auto & block : b_list) {
                        if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                            for (int32_t t_pos : block->token_indices) {
                                active_positions_dense[curr_pos_idx++] = t_pos;
                            }
                        }
                    }
                    total_positions = curr_pos_idx;
                }

                // Rebuild active_k_dense_rotated for all layers
                int num_dense = total_positions;
                if (num_dense > 0) {
                    int half_dim = head_dim / 2;
                    std::vector<float> cos_table_rebuild(num_dense * half_dim);
                    std::vector<float> sin_table_rebuild(num_dense * half_dim);
                    for (int t = 0; t < num_dense; ++t) {
                        int pos = active_positions_dense[t];
                        for (int i = 0; i < half_dim; ++i) {
                            float theta = (float)pos / std::pow(model.get_config().rope_freq_base, (float)(2 * i) / (float)head_dim);
                            cos_table_rebuild[t * half_dim + i] = std::cos(theta);
                            sin_table_rebuild[t * half_dim + i] = std::sin(theta);
                        }
                    }
                    for (int l = 0; l < n_layers; ++l) {
                        std::fill(active_k_dense_rotated[l].begin(), active_k_dense_rotated[l].end(), 0.0f);
                        apply_rope_neox_cpu_fast(
                            active_k_dense[l].data(),
                            active_k_dense_rotated[l].data(),
                            cos_table_rebuild.data(),
                            sin_table_rebuild.data(),
                            num_dense, kv_heads, head_dim
                        );
                    }
                } else {
                    for (int l = 0; l < n_layers; ++l) {
                        std::fill(active_k_dense_rotated[l].begin(), active_k_dense_rotated[l].end(), 0.0f);
                    }
                }
            }
        }
        if (!is_warmup_run) {
            std::cout << std::endl;
            if (interactive) {
                std::cout << "__FINISH__" << std::endl;
                // Emit KV cache size so gateway can skip re-prefilling on next turn
                std::cout << "__CACHED__:" << ((int)all_tokens.size()) << std::endl;
            }
        }

        // Call commit_turn to prune low-salience blocks and consolidate the SRL index
        try {
            runtime_manager.commit_turn(srl_state);
        } catch (const std::exception& e) {
            std::cerr << "[DiffKV Native] Error in commit_turn on turn boundary: " << e.what() << std::endl;
        }


        // Free decode context — must happen before next iteration rebuilds it
        ggml_free(decode_ctx);
        ggml_backend_buffer_free(dense_past_buffer);
        ggml_free(dense_past_ctx);

        if (is_warmup_run) {
            is_warmup_run = false;
            continue;
        }

        if (!interactive) {
            break;
        }

        // ── Update session KV prefix cache (ACTIVE_RUNTIME update_session_token_prefix) ──────
        // After this turn's generation, the KV pool contains all_tokens[0..L+gen-1].
        // On the next turn, the gateway will send __CACHED__:<N> with N = session_cached_len.
        // The binary will verify the first N tokens match and skip re-prefilling them.
        session_cached_len = (int)all_tokens.size(); // prompt + generated
        session_cached_token_ids = all_tokens;        // for prefix verification next turn
        if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
            std::cerr << "[DiffKV Native] Session prefix updated: " << session_cached_len
                      << " tokens now resident in KV pool." << std::endl;
        }

    }

    // Stop compressor and cleanup
    compressor.stop();

    std::cerr << "[DiffKV Native] Text generation completed successfully!" << std::endl;
    return 0;
}
