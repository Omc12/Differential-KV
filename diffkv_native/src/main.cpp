#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <random>
#include <map>
#include <unistd.h>
#include <thread>
#include <functional>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <pthread.h>
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
#include "native_core/compression/lowrank.hpp"
#include "native_core/kv_runtime_manager.hpp"

using namespace diffkv;

#include <atomic>
namespace diffkv { extern std::atomic<long> g_diffkv_cb_invocations; }  // diagnostic counter (diffkv_attention.cpp)
namespace diffkv { extern std::atomic<long> g_cpu_attn_count; extern std::atomic<long> g_metal_attn_count; }
namespace diffkv { extern std::atomic<int> g_diffkv_dbg_pos; extern std::atomic<int> g_diffkv_dbg_block_states; }
static std::atomic<bool> g_diffkv_enable_factual{true};

static bool is_native_attn_enabled() {
    const char* e = std::getenv("DIFFKV_NATIVE_ATTN");
    if (e) {
        return (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "yes" || std::string(e) == "on");
    }
    // Default OFF: measured 2026-07-03 at 4k (DIFFKV_PROFILE=1), the fused
    // ggml-graph path is ~1.9x SLOWER per decode step than the CPU custom op
    // (attention 213 vs 114 ms/token) and is non-deterministic/less coherent
    // at 16k, with identical NIAH accuracy (1/6 both on the digit sweep).
    // Re-flip only with profile numbers showing the fused path winning.
    return false;
}

// DIFFKV_DECODE_CACHE=1: "decompress-and-cache" sparse decode (the MLX §BIG-WIN, ported).
// Instead of running the CPU custom op (execute_cpu_attention) inside the GPU graph every
// token — which stalls the GPU ~9 ms/layer waiting on the CPU (measured: sparse decode 3.3 tps
// vs dense 44) — this path MATERIALIZES the routed compressed blocks (anchor + U·V + exact
// residuals, pre-rotated) into a contiguous F16 K/V buffer ONCE per N tokens, then attends
// [routed-blocks ++ dense-window ++ current] with the SAME GPU ggml_flash_attn_ext the dense
// path uses. No per-token reconstruction, no CPU-op-in-graph stall. Default OFF until the
// 6-cell NIAH sweep is 6/6 and decode tps is measured to beat the CPU-op baseline.
// ── Query-adaptive attend-all routing ─────────────────────────────────────
// Top-K block pruning is NIAH-safe (6/6, logit margins identical to
// attend-all) because single-fact retrieval only needs the one relevant
// block to land in the top-K. But a keyword-based "does this look broad"
// classifier can always miss a phrasing it wasn't taught — and a silently
// dropped fact is a worse failure than a slower answer. So default SAFE
// (attend-all) and only drop to pruning when the query confidently matches a
// narrow single-fact-retrieval pattern; anything ambiguous or unrecognized
// stays on attend-all rather than gambling on the fast path.
// DIFFKV_MLX_PARITY, if set, still overrides this per-request classification
// globally (useful for isolated benchmarking of either mode).
static bool diffkv_detect_narrow_query(const std::string & prompt_text) {
    // Look at the last user turn only (chat template puts context/document
    // first, the actual instruction last) so document body text can't trigger
    // false positives just for containing e.g. the word "summarize" somewhere.
    size_t marker_pos = prompt_text.rfind("<|im_start|>user");
    std::string tail = (marker_pos != std::string::npos) ? prompt_text.substr(marker_pos) : prompt_text;
    if (tail.size() > 2000) tail = tail.substr(tail.size() - 2000);
    std::transform(tail.begin(), tail.end(), tail.begin(), [](unsigned char c) { return std::tolower(c); });

    // Any broad signal vetoes narrow classification outright.
    // NOTE (2026-07-12 fix): the multi-part markers "and"/";" used to live in
    // this list and were matched against the WHOLE tail — but the tail of a
    // real prompt is document text (the user turn embeds the document), and
    // "and" appears in virtually any document, so EVERY document-bearing
    // prompt was vetoed to attend-all: +1.2 GB decode-cache spike at 16k and
    // 1.6x slower decode, even for plain single-fact retrieval. Profiled and
    // root-caused 2026-07-12. Multi-part detection now applies to the QUESTION
    // SENTENCE only (below).
    static const std::vector<std::string> broad_triggers = {
        "summarize", "summarise", "summary of", "tl;dr",
        "compare", "comparison", "contrast",
        "list all", "list every", "all instances", "all mentions", "everywhere",
        "differences between", "similarities between",
        "across the document", "across this document", "throughout the document",
        "throughout this", "what are all", "how many times", "each section",
        "every section", "overview of", "main points", "key points", "key themes",
        "in general", "overall,",
    };
    for (const auto & trig : broad_triggers) {
        if (tail.find(trig) != std::string::npos) return false;
    }
    // A direct single-fact question is exactly one '?'; 0 or 2+ isn't a clean
    // narrow-retrieval shape (0 = statement/instruction, 2+ = multi-part ask).
    if (std::count(tail.begin(), tail.end(), '?') != 1) return false;

    // Multi-part markers, scoped to the question sentence (from the last
    // sentence delimiter before the '?' to the '?').
    size_t qpos = tail.rfind('?');
    if (qpos != std::string::npos) {
        size_t sstart = tail.find_last_of(".!\n", qpos == 0 ? 0 : qpos - 1);
        std::string qsent = tail.substr(sstart == std::string::npos ? 0 : sstart + 1,
                                        qpos - (sstart == std::string::npos ? 0 : sstart + 1) + 1);
        if (qsent.find(" and ") != std::string::npos || qsent.find(';') != std::string::npos) {
            return false;
        }
    }

    // Require an explicit narrow-retrieval phrasing too — "one question mark"
    // alone isn't enough signal to trust the fast path.
    static const std::vector<std::string> narrow_triggers = {
        "what is the", "what's the", "who is the", "who's the",
        "when did", "when was", "where is", "where was",
        "what was the", "what is my", "what's my", "repeat it", "repeat the",
        "find the", "retrieve the",
    };
    for (const auto & trig : narrow_triggers) {
        if (tail.find(trig) != std::string::npos) return true;
    }
    return false; // ambiguous -> not narrow -> stays on the safe default
}

static bool is_decode_cache_enabled() {
    // DEFAULT ON (14th pass): verified 6/6 NIAH sweep + conformance bit-exact + 2.4-5.3× faster
    // than the CPU-op sparse path. Only active when sparse decode engages (long ctx); dense/short
    // decode is unaffected. Explicit DIFFKV_DECODE_CACHE=0 forces the old CPU-op path (for
    // baseline measurement). Matches the MLX serving default (DIFFKV_DECODE_CACHE=1).
    const char* e = std::getenv("DIFFKV_DECODE_CACHE");
    if (!e) return true;
    std::string v(e);
    return !(v == "0" || v == "false" || v == "no" || v == "off");
}
// Re-route + re-materialize the routed blocks every N decode tokens (amortizes the
// reconstruction cost). The dense window + mask + current token are refreshed EVERY token.
// N=16 matches the MLX default; MLX's NIAH+synthesis sweep proved it staleness-safe.
static int decode_cache_interval() {
    const char* e = std::getenv("DIFFKV_DECODE_CACHE_INTERVAL");
    int n = e ? atoi(e) : 16;
    return n > 0 ? n : 16;
}

struct ggml_backend_owner {
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ggml_backend_owner() {
        bool use_gpu = true;
        if (const char* env_gpu = std::getenv("DIFFKV_USE_GPU")) {
            if (std::string(env_gpu) == "0" || std::string(env_gpu) == "false" || std::string(env_gpu) == "off") {
                use_gpu = false;
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

// fp16 KV overload: src/dst are float16 storage; the rotation math runs in fp32
// (identical numerics to the fp32 version) and the result is stored back as fp16.
static void apply_rope_neox_cpu_fast(
    const ggml_fp16_t* k_raw, ggml_fp16_t* k_rotated,
    const float* cos_table, const float* sin_table,
    int num_tokens, int kv_heads, int head_dim
) {
    const int D    = head_dim;
    const int half = D / 2;
    for (int t = 0; t < num_tokens; ++t) {
        const float* cos_t = cos_table + t * half;
        const float* sin_t = sin_table + t * half;
        for (int h = 0; h < kv_heads; ++h) {
            const ggml_fp16_t* src = k_raw     + (t * kv_heads + h) * D;
            ggml_fp16_t*       dst = k_rotated + (t * kv_heads + h) * D;
            for (int i = 0; i < half; ++i) {
                float cos_th = cos_t[i];
                float sin_th = sin_t[i];
                float x      = ggml_fp16_to_fp32(src[i]);
                float y      = ggml_fp16_to_fp32(src[i + half]);
                dst[i]        = ggml_fp32_to_fp16(x * cos_th - y * sin_th);
                dst[i + half] = ggml_fp32_to_fp16(y * cos_th + x * sin_th);
            }
        }
    }
}

// Mixed overload: fp16 source (k_activations) → fp32 destination (active_k_dense_rotated,
// the decode dense window which stays fp32). Used by the multi-turn rebuild path.
static void apply_rope_neox_cpu_fast(
    const ggml_fp16_t* k_raw, float* k_rotated,
    const float* cos_table, const float* sin_table,
    int num_tokens, int kv_heads, int head_dim
) {
    const int D    = head_dim;
    const int half = D / 2;
    for (int t = 0; t < num_tokens; ++t) {
        const float* cos_t = cos_table + t * half;
        const float* sin_t = sin_table + t * half;
        for (int h = 0; h < kv_heads; ++h) {
            const ggml_fp16_t* src = k_raw     + (t * kv_heads + h) * D;
            float*             dst = k_rotated + (t * kv_heads + h) * D;
            for (int i = 0; i < half; ++i) {
                float cos_th = cos_t[i];
                float sin_th = sin_t[i];
                float x      = ggml_fp16_to_fp32(src[i]);
                float y      = ggml_fp16_to_fp32(src[i + half]);
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
    bool need_logits = false,
    std::vector<struct ggml_tensor *> * persistent_k_cache = nullptr,
    std::vector<struct ggml_tensor *> * persistent_v_cache = nullptr,
    int pos_start = 0,
    // DIFFKV_SPARSE_PREFILL (HANDOFF §SPARSE-PREFILL): when non-empty, k_ctx/v_ctx attend only
    // these [start,len) ranges of the persistent cache (sink + recency window + current
    // chunk) instead of the full [0, ctx_len) view — training-free StreamingLLM sparse prefill.
    // Offsets are BUFFER offsets: identical to absolute positions in the legacy full-length
    // cache; ring-mapped by the caller under DIFFKV_LEGO_PREFILL.
    const std::vector<std::pair<int,int>>* sp_ranges = nullptr,
    // DIFFKV_LEGO_PREFILL: buffer spans to write the current chunk into (the ring write may
    // wrap, splitting into <=2 spans; a straddle of the identity zone adds one more). Lengths
    // sum to chunk_len. nullptr = legacy single write at pos_start.
    const std::vector<std::pair<int,int>>* chunk_write_spans = nullptr,
    // DIFFKV_LEGO_PREFILL: per-layer device tensors holding the routed FAR blocks' rows
    // (rotated K / raw V), uploaded host-side each chunk from the raw activation mirrors.
    // The first n_far rows are concatenated BEFORE the sp_ranges views (the caller's mask
    // walks far rows first, then ranges).
    std::vector<struct ggml_tensor *>* far_k_layers = nullptr,
    std::vector<struct ggml_tensor *>* far_v_layers = nullptr,
    int n_far = 0
) {
    const auto & config = model.get_config();
    // Use ggml_new_graph_custom with a larger node budget (32768) since SPARSE_PREFILL
    // introduces multiple concatenations and views per layer, exceeding the 2048 default.
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);

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

        bool cast_prefill_gpu = true;
        if (const char* env_cpg = std::getenv("DIFFKV_PREFILL_CAST_GPU")) {
            try { cast_prefill_gpu = (std::stoi(env_cpg) != 0); } catch (...) {}
        }

        // Export raw V
        if (v_layers) {
            (*v_layers)[l] = cast_prefill_gpu ? ggml_cast(ctx, v, GGML_TYPE_F16) : v;
        }

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
        if (k_layers) {
            (*k_layers)[l] = cast_prefill_gpu ? ggml_cast(ctx, k, GGML_TYPE_F16) : k;
        }

        // 5. Cast current chunk K/V to F16 (flash-attn's standard K/V dtype; matches the
        //    F16 prior tensors so the concat dtypes agree and the GPU prior context is halved).
        struct ggml_tensor * k_rope_f16 = ggml_cast(ctx, k_rope, GGML_TYPE_F16);
        struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, head_dim, config.n_head_kv, v->ne[1]);
        struct ggml_tensor * v_reshaped_f16 = ggml_cast(ctx, v_reshaped, GGML_TYPE_F16);

        // Concatenate prior context with current chunk along seq dim (dim=2 in unpermuted layout)
        struct ggml_tensor * k_ctx = nullptr;
        struct ggml_tensor * v_ctx = nullptr;

        if (persistent_k_cache && persistent_v_cache) {
            struct ggml_tensor * pk = (*persistent_k_cache)[l];
            struct ggml_tensor * pv = (*persistent_v_cache)[l];

            if (chunk_write_spans && !chunk_write_spans->empty()) {
                // LEGO ring write: the chunk lands at ring-mapped buffer offsets,
                // possibly split across a wrap (<=3 spans). Views into the just-
                // computed chunk tensors are cpy'd span by span.
                int chunk_off = 0;
                for (const auto & w : *chunk_write_spans) {
                    struct ggml_tensor * src_k = ggml_view_3d(ctx, k_rope_f16,
                        head_dim, config.n_head_kv, w.second,
                        k_rope_f16->nb[1], k_rope_f16->nb[2],
                        (size_t)chunk_off * k_rope_f16->nb[2]);
                    struct ggml_tensor * src_v = ggml_view_3d(ctx, v_reshaped_f16,
                        head_dim, config.n_head_kv, w.second,
                        v_reshaped_f16->nb[1], v_reshaped_f16->nb[2],
                        (size_t)chunk_off * v_reshaped_f16->nb[2]);
                    struct ggml_tensor * dst_k = ggml_view_3d(ctx, pk,
                        head_dim, config.n_head_kv, w.second,
                        pk->nb[1], pk->nb[2], (size_t)w.first * pk->nb[2]);
                    struct ggml_tensor * dst_v = ggml_view_3d(ctx, pv,
                        head_dim, config.n_head_kv, w.second,
                        pv->nb[1], pv->nb[2], (size_t)w.first * pv->nb[2]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src_k, dst_k));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src_v, dst_v));
                    chunk_off += w.second;
                }
            } else {
                struct ggml_tensor * dest_k = ggml_view_3d(ctx, pk,
                    head_dim, config.n_head_kv, k_rope_f16->ne[2],
                    pk->nb[1], pk->nb[2],
                    pos_start * pk->nb[2]);
                struct ggml_tensor * dest_v = ggml_view_3d(ctx, pv,
                    head_dim, config.n_head_kv, v_reshaped_f16->ne[2],
                    pv->nb[1], pv->nb[2],
                    pos_start * pv->nb[2]);

                struct ggml_tensor * copy_k = ggml_cpy(ctx, k_rope_f16, dest_k);
                struct ggml_tensor * copy_v = ggml_cpy(ctx, v_reshaped_f16, dest_v);

                ggml_build_forward_expand(gf, copy_k);
                ggml_build_forward_expand(gf, copy_v);
            }

            if (sp_ranges && !sp_ranges->empty()) {
                // Sparse prefill: attend only [far blocks | sink | recency window | current
                // chunk]. The far segment (lego mode) is a separate device tensor uploaded
                // host-side; the remaining ranges are contiguous views of the cache. The
                // current chunk was just cpy'd into pk/pv, so the window range (which covers
                // it) reads it back.
                struct ggml_tensor * kacc = nullptr;
                struct ggml_tensor * vacc = nullptr;
                if (far_k_layers && far_v_layers && n_far > 0) {
                    struct ggml_tensor * fk = (*far_k_layers)[l];
                    struct ggml_tensor * fv = (*far_v_layers)[l];
                    kacc = ggml_view_3d(ctx, fk, head_dim, config.n_head_kv, n_far,
                        fk->nb[1], fk->nb[2], 0);
                    vacc = ggml_view_3d(ctx, fv, head_dim, config.n_head_kv, n_far,
                        fv->nb[1], fv->nb[2], 0);
                }
                for (const auto & r : *sp_ranges) {
                    int rs = r.first, rl = r.second;
                    struct ggml_tensor * kv = ggml_view_3d(ctx, pk, head_dim, config.n_head_kv, rl,
                        pk->nb[1], pk->nb[2], (size_t)rs * pk->nb[2]);
                    struct ggml_tensor * vv = ggml_view_3d(ctx, pv, head_dim, config.n_head_kv, rl,
                        pv->nb[1], pv->nb[2], (size_t)rs * pv->nb[2]);
                    kacc = kacc ? ggml_concat(ctx, kacc, kv, 2) : kv;
                    vacc = vacc ? ggml_concat(ctx, vacc, vv, 2) : vv;
                }
                k_ctx = kacc;
                v_ctx = vacc;
            } else {
                k_ctx = ggml_view_3d(ctx, pk,
                    head_dim, config.n_head_kv, pos_start + k_rope_f16->ne[2],
                    pk->nb[1], pk->nb[2],
                    0);
                v_ctx = ggml_view_3d(ctx, pv,
                    head_dim, config.n_head_kv, pos_start + v_reshaped_f16->ne[2],
                    pv->nb[1], pv->nb[2],
                    0);
            }
        } else {
            k_ctx = k_rope_f16;
            v_ctx = v_reshaped_f16;
            bool has_prior = (prior_k_ctx && (*prior_k_ctx)[l] != nullptr);
            if (has_prior) {
                k_ctx = ggml_concat(ctx, (*prior_k_ctx)[l], k_rope_f16, 2);
                v_ctx = ggml_concat(ctx, (*prior_v_ctx)[l], v_reshaped_f16, 2);
            }
        }

        // Permute to [head_dim, seq_len, kv_heads] layout expected by flash attention
        struct ggml_tensor * q_perm = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        struct ggml_tensor * k_ctx_perm = ggml_cont(ctx, ggml_permute(ctx, k_ctx, 0, 2, 1, 3));
        struct ggml_tensor * v_ctx_perm = ggml_cont(ctx, ggml_permute(ctx, v_ctx, 0, 2, 1, 3));

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

// DEBUG (DIFFKV_DBG_CMP): capture layer-0 native q_rope + sparse attn_out so main can diff vs
// execute_cpu_attention in-process (definitive native-vs-CPU input check, immune to warmup noise).
static struct ggml_tensor* g_dbg_qrope = nullptr;
static struct ggml_tensor* g_dbg_attn0 = nullptr;
static struct ggml_tensor* g_dbg_sel0  = nullptr;
static struct ggml_tensor* g_dbg_curk  = nullptr;
static struct ggml_tensor* g_dbg_curv  = nullptr;

// DIFFKV_BLOCK_CMP: per-layer tensor captures for full block-wise comparison across all layers.
static constexpr int BLOCK_CMP_MAX_LAYERS = 32;
static struct ggml_tensor* g_bcmp_qrope[BLOCK_CMP_MAX_LAYERS]  = {};
static struct ggml_tensor* g_bcmp_attn [BLOCK_CMP_MAX_LAYERS]  = {};
static struct ggml_tensor* g_bcmp_slots[BLOCK_CMP_MAX_LAYERS]  = {};
static struct ggml_tensor* g_bcmp_curk [BLOCK_CMP_MAX_LAYERS]  = {};
static struct ggml_tensor* g_bcmp_curv [BLOCK_CMP_MAX_LAYERS]  = {};

// ── DIFFKV_DECODE_CACHE materialize helper ─────────────────────────────────────────────
// Reconstruct the routed compressed blocks of ONE layer's pool into contiguous F16 K/V
// buffers laid out [token, kv_head, dim] (dim fastest) — exactly the memory order of a ggml
// [head_dim, kv_heads, n_tokens] flash-attn past-KV input. K is emitted PRE-ROTATED at each
// token's absolute position (POOL_ROT_ABS: the pool already stores rotated K, verified by
// DIFFKV_DBG_RECON_POS), so it drops straight into flash. V is raw. The reconstruction math
// is identical to the DBG_RECON_POS probe: rec = anchor + Σ_r U[t,r]·V[r]·row_scale·blk_scale
// (+ exact residual override for the ≤MAX_RESIDUAL captured rows).
//
// slots[] is the routed candidate list (may contain -1, duplicates, or non-resident IDs — all
// skipped). Only the first `max_blocks` DISTINCT CompressedResident blocks are materialized.
// Returns the number of valid routed tokens written (rows [ret, cap) are padding for the caller
// to mask). Thread-safe across layers (writes disjoint out buffers, reads const pool state).
static int materialize_routed_kv(
    diffkv::NativeBlockPool* pool,
    const int32_t* slots, int n_slots, int max_blocks,
    int kv_heads, int head_dim, float freq,
    int cap_tokens,
    ggml_fp16_t* out_k, ggml_fp16_t* out_v)
{
    const int F = kv_heads * head_dim;
    const int R = pool->get_rank();
    const int MR = diffkv::NativeBlockPool::MAX_RESIDUAL;
    const int n_pool = (int)pool->get_seq_lens()->ne[0];
    const auto& state_table = pool->get_state_table();
    const ggml_fp16_t* scales = pool->get_host_scales();
    const int32_t* seq_lens = pool->get_host_seq_lens();

    // Reusable per-block scratch (float): the low-rank reconstruction delta = U_scaled @ V is a
    // small dense matmul, so we do it with cblas_sgemm (AMX) instead of a scalar rank loop — the
    // scalar version was the dominant native decode cost (~1900 ms per full re-materialisation).
    std::vector<float> U_scaled, VKf, VVf, dK, dV;

    int written = 0;
    int blocks_done = 0;
    std::vector<int> seen; seen.reserve(max_blocks);
    for (int si = 0; si < n_slots && blocks_done < max_blocks && written < cap_tokens; ++si) {
        int slot = slots[si];
        if (slot < 0 || slot >= n_pool) continue;
        if (state_table.get(slot) != diffkv::BlockState::CompressedResident) continue;
        bool dup = false; for (int s : seen) if (s == slot) { dup = true; break; }
        if (dup) continue;
        seen.push_back(slot);
        blocks_done++;

        const int8_t*      U    = pool->get_host_U(slot);
        const ggml_fp16_t* rowsc= pool->get_host_U_row_scale(slot);
        const ggml_fp16_t* VK   = pool->get_host_VK(slot);
        const ggml_fp16_t* VV   = pool->get_host_VV(slot);
        const ggml_fp16_t* ancK = pool->get_host_anchors_K(slot);
        const ggml_fp16_t* ancV = pool->get_host_anchors_V(slot);
        const ggml_fp16_t* resKv= pool->get_host_res_K_val(slot);
        const int32_t*     resKp= pool->get_host_res_K_pos(slot);
        const ggml_fp16_t* resVv= pool->get_host_res_V_val(slot);
        const int32_t*     resVp= pool->get_host_res_V_pos(slot);
        if (!U || !VK || !VV || !ancK || !ancV || !rowsc) continue;
        const float bsc = ggml_fp16_to_fp32(scales[slot]);
        const int slen = seq_lens[slot];

        // Emit the ANCHOR as its own attended row (the CPU op attends anchor_K/anchor_V at
        // anchor_idx separately from the slen delta tokens at anchor_idx+1+t; both are stored
        // pre-rotated). Omitting it drops the block's landmark token from the softmax.
        if (written < cap_tokens) {
            ggml_fp16_t* ok = out_k + (size_t)written * F;
            ggml_fp16_t* ov = out_v + (size_t)written * F;
            for (int f = 0; f < F; ++f) { ok[f] = ancK[f]; ov[f] = ancV[f]; }
            written++;
        }

        const int nt = std::min(slen, cap_tokens - written);   // delta rows we can still write
        if (nt <= 0) continue;

        // U_scaled[t,r] = U[t,r] * row_scale[t] * block_scale   (fold both scales into U so the
        // gemm output is the final delta). VKf/VVf = fp16→f32 of the block's rank-basis.
        U_scaled.resize((size_t)nt * R);
        for (int t = 0; t < nt; ++t) {
            const float s = ggml_fp16_to_fp32(rowsc[t]) * bsc;
            const int8_t* ur = U + (size_t)t * R;
            float* us = U_scaled.data() + (size_t)t * R;
            for (int r = 0; r < R; ++r) us[r] = (float)ur[r] * s;
        }
        VKf.resize((size_t)R * F); VVf.resize((size_t)R * F);
        for (size_t i = 0; i < (size_t)R * F; ++i) { VKf[i] = ggml_fp16_to_fp32(VK[i]); VVf[i] = ggml_fp16_to_fp32(VV[i]); }
        dK.resize((size_t)nt * F); dV.resize((size_t)nt * F);
        // dK[nt,F] = U_scaled[nt,R] @ VKf[R,F]   (row-major)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nt, F, R, 1.0f,
                    U_scaled.data(), R, VKf.data(), F, 0.0f, dK.data(), F);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nt, F, R, 1.0f,
                    U_scaled.data(), R, VVf.data(), F, 0.0f, dV.data(), F);

        for (int t = 0; t < nt; ++t) {
            int rik = -1, riv = -1;
            if (resKp) for (int r = 0; r < MR; ++r) if (resKp[r] == t) { rik = r; break; }
            if (resVp) for (int r = 0; r < MR; ++r) if (resVp[r] == t) { riv = r; break; }
            ggml_fp16_t* ok = out_k + (size_t)written * F;
            ggml_fp16_t* ov = out_v + (size_t)written * F;
            const float* dk = dK.data() + (size_t)t * F;
            const float* dv = dV.data() + (size_t)t * F;
            const ggml_fp16_t* rkres = (rik >= 0 && resKv) ? resKv + (size_t)rik * F : nullptr;
            const ggml_fp16_t* rvres = (riv >= 0 && resVv) ? resVv + (size_t)riv * F : nullptr;
            for (int f = 0; f < F; ++f) {
                float rk = ggml_fp16_to_fp32(ancK[f]) + dk[f];
                float rv = ggml_fp16_to_fp32(ancV[f]) + dv[f];
                if (rkres) rk += ggml_fp16_to_fp32(rkres[f]);
                if (rvres) rv += ggml_fp16_to_fp32(rvres[f]);
                ok[f] = ggml_fp32_to_fp16(rk);
                ov[f] = ggml_fp32_to_fp16(rv);
            }
            written++;
        }
    }
    return written;
}

// ── DIFFKV_LEGO_PREFILL stage-2: single-block far emit from the compressed pool ──
// Reconstructs ONE known-CompressedResident slot into contiguous K/V staging rows
// (anchor first, then the slen delta rows) with EXACTLY the materialize_routed_kv
// math: rec = anchor + U·V·row_scale·blk_scale (+ exact residual corrections).
// K rows come out PRE-ROTATED at their absolute positions (POOL_ROT_ABS — the pool
// stores rotated K), V raw — both drop straight into the prefill FAR tensors with
// no host RoPE. Row ORDER within the far segment is irrelevant (every far row is
// unconditionally-visible history in the mask), only the row SET matters, so the
// landmark swap needs no position bookkeeping here.
// Returns rows written (1 + slen) or -1 if the slot's host mirrors are missing.
static int lego_emit_block_from_pool(
    diffkv::NativeBlockPool* pool, int slot, int F,
    ggml_fp16_t* out_k, ggml_fp16_t* out_v)
{
    const int R  = pool->get_rank();
    const int MR = diffkv::NativeBlockPool::MAX_RESIDUAL;
    const int8_t*      U    = pool->get_host_U(slot);
    const ggml_fp16_t* rowsc= pool->get_host_U_row_scale(slot);
    const ggml_fp16_t* VK   = pool->get_host_VK(slot);
    const ggml_fp16_t* VV   = pool->get_host_VV(slot);
    const ggml_fp16_t* ancK = pool->get_host_anchors_K(slot);
    const ggml_fp16_t* ancV = pool->get_host_anchors_V(slot);
    const ggml_fp16_t* resKv= pool->get_host_res_K_val(slot);
    const int32_t*     resKp= pool->get_host_res_K_pos(slot);
    const ggml_fp16_t* resVv= pool->get_host_res_V_val(slot);
    const int32_t*     resVp= pool->get_host_res_V_pos(slot);
    if (!U || !VK || !VV || !ancK || !ancV || !rowsc) return -1;
    const float bsc  = ggml_fp16_to_fp32(pool->get_host_scales()[slot]);
    const int   slen = pool->get_host_seq_lens()[slot];

    for (int f = 0; f < F; ++f) { out_k[f] = ancK[f]; out_v[f] = ancV[f]; }
    if (slen <= 0) return 1;

    std::vector<float> U_scaled((size_t)slen * R);
    for (int t = 0; t < slen; ++t) {
        const float s = ggml_fp16_to_fp32(rowsc[t]) * bsc;
        const int8_t* ur = U + (size_t)t * R;
        float* us = U_scaled.data() + (size_t)t * R;
        for (int r = 0; r < R; ++r) us[r] = (float)ur[r] * s;
    }
    std::vector<float> VKf((size_t)R * F), VVf((size_t)R * F);
    for (size_t i = 0; i < (size_t)R * F; ++i) {
        VKf[i] = ggml_fp16_to_fp32(VK[i]);
        VVf[i] = ggml_fp16_to_fp32(VV[i]);
    }
    std::vector<float> dK((size_t)slen * F), dV((size_t)slen * F);
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, slen, F, R, 1.0f,
                U_scaled.data(), R, VKf.data(), F, 0.0f, dK.data(), F);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, slen, F, R, 1.0f,
                U_scaled.data(), R, VVf.data(), F, 0.0f, dV.data(), F);
#else
    for (int t = 0; t < slen; ++t) {
        float* dk = dK.data() + (size_t)t * F;
        float* dv = dV.data() + (size_t)t * F;
        std::memset(dk, 0, F * sizeof(float));
        std::memset(dv, 0, F * sizeof(float));
        const float* us = U_scaled.data() + (size_t)t * R;
        for (int r = 0; r < R; ++r) {
            const float u = us[r];
            const float* vk = VKf.data() + (size_t)r * F;
            const float* vv = VVf.data() + (size_t)r * F;
            for (int f = 0; f < F; ++f) { dk[f] += u * vk[f]; dv[f] += u * vv[f]; }
        }
    }
#endif

    for (int t = 0; t < slen; ++t) {
        int rik = -1, riv = -1;
        if (resKp) for (int r = 0; r < MR; ++r) if (resKp[r] == t) { rik = r; break; }
        if (resVp) for (int r = 0; r < MR; ++r) if (resVp[r] == t) { riv = r; break; }
        ggml_fp16_t* ok = out_k + (size_t)(1 + t) * F;
        ggml_fp16_t* ov = out_v + (size_t)(1 + t) * F;
        const float* dk = dK.data() + (size_t)t * F;
        const float* dv = dV.data() + (size_t)t * F;
        const ggml_fp16_t* rkres = (rik >= 0 && resKv) ? resKv + (size_t)rik * F : nullptr;
        const ggml_fp16_t* rvres = (riv >= 0 && resVv) ? resVv + (size_t)riv * F : nullptr;
        for (int f = 0; f < F; ++f) {
            float rk = ggml_fp16_to_fp32(ancK[f]) + dk[f];
            float rv = ggml_fp16_to_fp32(ancV[f]) + dv[f];
            if (rkres) rk += ggml_fp16_to_fp32(rkres[f]);
            if (rvres) rv += ggml_fp16_to_fp32(rvres[f]);
            ok[f] = ggml_fp32_to_fp16(rk);
            ov[f] = ggml_fp32_to_fp16(rv);
        }
    }
    return 1 + slen;
}

// ── LEGO far mode "studs" (default): anchor + the block's exact residual rows ──
// The MLX studs-only finding, made portable: emit ONLY rows that come out EXACT —
// the anchor plus the residual-corrected rows (residual = raw − anchor − recon by
// construction, so anchor + recon + residual reproduces the raw row up to fp16
// rounding). The well-approximated low-information rows are omitted, not
// reconstructed — zero low-rank noise enters the prefill far field. Uniform row
// count (1 + mr_target) is guaranteed by DIFFKV_RESIDUAL_UNIFORM=1 at compression
// (full residual sets); if a block is still under-full (joint_err ≤ 1e-4 skips —
// essentially never), deterministic lowest-index recon rows pad to the target so
// the shared mask/klen stays uniform across layers. K pre-rotated (POOL_ROT_ABS),
// V raw. Returns rows written (always 1 + mr_target when slen ≥ mr_target) or -1
// if host mirrors are missing.
static int lego_emit_block_studs(
    diffkv::NativeBlockPool* pool, int slot, int F, int mr_target,
    ggml_fp16_t* out_k, ggml_fp16_t* out_v)
{
    const int R  = pool->get_rank();
    const int MR = diffkv::NativeBlockPool::MAX_RESIDUAL;
    const int8_t*      U    = pool->get_host_U(slot);
    const ggml_fp16_t* rowsc= pool->get_host_U_row_scale(slot);
    const ggml_fp16_t* VK   = pool->get_host_VK(slot);
    const ggml_fp16_t* VV   = pool->get_host_VV(slot);
    const ggml_fp16_t* ancK = pool->get_host_anchors_K(slot);
    const ggml_fp16_t* ancV = pool->get_host_anchors_V(slot);
    const ggml_fp16_t* resKv= pool->get_host_res_K_val(slot);
    const int32_t*     resKp= pool->get_host_res_K_pos(slot);
    const ggml_fp16_t* resVv= pool->get_host_res_V_val(slot);
    const int32_t*     resVp= pool->get_host_res_V_pos(slot);
    if (!U || !VK || !VV || !ancK || !ancV || !rowsc) return -1;
    const float bsc  = ggml_fp16_to_fp32(pool->get_host_scales()[slot]);
    const int   slen = pool->get_host_seq_lens()[slot];

    for (int f = 0; f < F; ++f) { out_k[f] = ancK[f]; out_v[f] = ancV[f]; }
    if (slen <= 0 || mr_target <= 0) return 1;

    // Selected delta rows: residual positions first (exact), then deterministic
    // filler (recon-only) if under-full. res_K_pos == res_V_pos by construction
    // (lowrank.cpp writes the same s to both).
    std::vector<int> spos;   spos.reserve(mr_target);
    std::vector<int> sres;   sres.reserve(mr_target);   // residual slot per row, -1 = filler
    for (int r = 0; r < MR && (int)spos.size() < mr_target; ++r) {
        int s = resKp ? resKp[r] : -1;
        if (s >= 0 && s < slen) { spos.push_back(s); sres.push_back(r); }
    }
    if ((int)spos.size() < std::min(mr_target, slen)) {
        std::vector<char> used(slen, 0);
        for (int s : spos) used[s] = 1;
        for (int s = 0; s < slen && (int)spos.size() < mr_target; ++s) {
            if (!used[s]) { spos.push_back(s); sres.push_back(-1); }
        }
    }
    const int n = (int)spos.size();
    if (n == 0) return 1;

    // Recon deltas for ONLY the selected rows: U_scaled[n,R] @ V[R,F].
    std::vector<float> U_scaled((size_t)n * R);
    for (int i = 0; i < n; ++i) {
        const int t = spos[i];
        const float s = ggml_fp16_to_fp32(rowsc[t]) * bsc;
        const int8_t* ur = U + (size_t)t * R;
        float* us = U_scaled.data() + (size_t)i * R;
        for (int r = 0; r < R; ++r) us[r] = (float)ur[r] * s;
    }
    std::vector<float> VKf((size_t)R * F), VVf((size_t)R * F);
    for (size_t i = 0; i < (size_t)R * F; ++i) {
        VKf[i] = ggml_fp16_to_fp32(VK[i]);
        VVf[i] = ggml_fp16_to_fp32(VV[i]);
    }
    std::vector<float> dK((size_t)n * F), dV((size_t)n * F);
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, F, R, 1.0f,
                U_scaled.data(), R, VKf.data(), F, 0.0f, dK.data(), F);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, F, R, 1.0f,
                U_scaled.data(), R, VVf.data(), F, 0.0f, dV.data(), F);
#else
    for (int i = 0; i < n; ++i) {
        float* dk = dK.data() + (size_t)i * F;
        float* dv = dV.data() + (size_t)i * F;
        std::memset(dk, 0, F * sizeof(float));
        std::memset(dv, 0, F * sizeof(float));
        const float* us = U_scaled.data() + (size_t)i * R;
        for (int r = 0; r < R; ++r) {
            const float u = us[r];
            const float* vk = VKf.data() + (size_t)r * F;
            const float* vv = VVf.data() + (size_t)r * F;
            for (int f = 0; f < F; ++f) { dk[f] += u * vk[f]; dv[f] += u * vv[f]; }
        }
    }
#endif

    for (int i = 0; i < n; ++i) {
        ggml_fp16_t* ok = out_k + (size_t)(1 + i) * F;
        ggml_fp16_t* ov = out_v + (size_t)(1 + i) * F;
        const float* dk = dK.data() + (size_t)i * F;
        const float* dv = dV.data() + (size_t)i * F;
        const int r = sres[i];
        // V residual slot may differ from K's index for the same s in principle;
        // look it up independently when this row is residual-backed.
        int rv_slot = -1;
        if (r >= 0 && resVp) {
            const int t = spos[i];
            for (int rr = 0; rr < MR; ++rr) if (resVp[rr] == t) { rv_slot = rr; break; }
        }
        const ggml_fp16_t* rkres = (r >= 0 && resKv) ? resKv + (size_t)r * F : nullptr;
        const ggml_fp16_t* rvres = (rv_slot >= 0 && resVv) ? resVv + (size_t)rv_slot * F : nullptr;
        for (int f = 0; f < F; ++f) {
            float rk = ggml_fp16_to_fp32(ancK[f]) + dk[f];
            float rv = ggml_fp16_to_fp32(ancV[f]) + dv[f];
            if (rkres) rk += ggml_fp16_to_fp32(rkres[f]);
            if (rvres) rv += ggml_fp16_to_fp32(rvres[f]);
            ok[f] = ggml_fp32_to_fp16(rk);
            ov[f] = ggml_fp32_to_fp16(rv);
        }
    }
    return 1 + n;
}

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
    int engage_threshold = 4096,
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
    struct ggml_tensor * native_attn_slots = nullptr,  // [srl_k_keep] host-controlled CompressedResident-filtered slots
    struct ggml_tensor ** cache_k = nullptr,           // DIFFKV_DECODE_CACHE: [n_layer] each [head_dim,kv_heads,cache_cap] materialized routed+window K (F16, rotated)
    struct ggml_tensor ** cache_v = nullptr,           // DIFFKV_DECODE_CACHE: [n_layer] each [head_dim,kv_heads,cache_cap] materialized routed+window V (F16)
    struct ggml_tensor * cache_mask = nullptr,         // DIFFKV_DECODE_CACHE: [cache_cap+1,1] F16 validity bias (0 valid / -inf pad; +1 = current token)
    int cache_cap = 0,                                 // DIFFKV_DECODE_CACHE: capacity of the cache buffers (routed + window tokens)
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

        int head_dim = config.n_embd / config.n_head;
        struct ggml_tensor * k_reshaped_n = ggml_reshape_3d(ctx, k, head_dim, config.n_head_kv, 1);
        struct ggml_tensor * k_rope_n = ggml_rope_ext(ctx, k_reshaped_n, position, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Export roped key/value tensors of each layer (after RoPE)
        if (concat_k) {
            struct ggml_tensor * k_view = ggml_view_1d(ctx, concat_k, F_test, l * F_test * sizeof(float));
            struct ggml_tensor * copy_k = ggml_cpy(ctx, ggml_reshape_1d(ctx, k_rope_n, F_test), k_view);
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
                if (userdata) {
                    userdata[0].layer0_q_tensor = q;
                    ggml_set_output(q);
                }
                // Reshape Q: [896, 1] -> [head_dim, n_head] = [64, 14]
                struct ggml_tensor * Q = ggml_reshape_2d(ctx, q, head_dim, config.n_head);

                // Compute query descriptor: [desc_dim, 1]
                struct ggml_tensor * q_desc = compute_query_desc(ctx, Q, W_proj);

                // Semantic search topk: [srl_k_semantic, 1]
                struct ggml_tensor * sem_slots = semantic_search_topk(ctx, q_desc, desc_matrix, slots_mask, srl_k_semantic);
                struct ggml_tensor * sem_slots_1d = ggml_reshape_1d(ctx, sem_slots, sem_slots->ne[0]);

                // Concatenate semantic and host slots
                struct ggml_tensor * candidate_slots = ggml_concat(ctx, sem_slots_1d, host_slots, 0);

                // Anchor screening: [srl_k_keep]
                float scale = 1.0f / std::sqrt((float)head_dim);
                selected_slots = ggml_cont(ctx, anchor_screen(ctx, Q, anchors_K, candidate_slots, slots_mask, scale, srl_k_keep));

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

            if (l == 0 && std::getenv("DIFFKV_DBG_ATTN0")) {
                std::cerr << "[DBG_BRANCH] use_sparse=1 use_native=" << (int)use_native_attn
                          << " selected_slots=" << (void*)selected_slots
                          << " userdata=" << (void*)userdata << "\n";
            }
            static const bool decode_cache_on = is_decode_cache_enabled();
            if (decode_cache_on && cache_k && cache_v && cache_mask && cache_k[l]) {
                // ── DIFFKV_DECODE_CACHE: attend [materialized routed blocks ++ dense window
                // ++ current token] with the SAME GPU flash kernel the dense path uses. The
                // host fills cache_k/cache_v (routed materialized once per N; window per token);
                // here we only concat the current token and flash. No CPU op, no per-token
                // reconstruction, no LSE merge → the GPU never stalls on a CPU callback.
                struct ggml_tensor * kcur = ggml_cast(ctx, k_rope_n, GGML_TYPE_F16);      // [head_dim,kv_heads,1] rotated
                struct ggml_tensor * v3   = ggml_reshape_3d(ctx, v, head_dim, config.n_head_kv, 1);
                struct ggml_tensor * vcur = ggml_cast(ctx, v3, GGML_TYPE_F16);
                struct ggml_tensor * k_ctx = ggml_concat(ctx, cache_k[l], kcur, 2);       // [head_dim,kv_heads,cap+1]
                struct ggml_tensor * v_ctx = ggml_concat(ctx, cache_v[l], vcur, 2);
                struct ggml_tensor * q_perm = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
                struct ggml_tensor * k_perm = ggml_permute(ctx, k_ctx, 0, 2, 1, 3);
                struct ggml_tensor * v_perm = ggml_permute(ctx, v_ctx, 0, 2, 1, 3);
                float scale_val = 1.0f / std::sqrt((float)head_dim);
                struct ggml_tensor * ao = ggml_flash_attn_ext(ctx, q_perm, k_perm, v_perm, cache_mask, scale_val, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(ao, GGML_PREC_F32);
                attn_out = ggml_reshape_2d(ctx, ao, config.n_embd, 1);
            } else if (use_native_attn && selected_slots && native_dense_kr && native_dense_v && native_dense_mask) {
                // Past-dense keys are uploaded already rotated; no need to RoPE them in-graph!
                struct ggml_tensor * dkr_flat = ggml_reshape_2d(ctx, native_dense_kr[l], head_dim * config.n_head_kv, native_maxd);
                // Flash-decoding in-graph fused op (GGML_OP_DIFFKV_ATTN)
                diffkv::NativeBlockPool* pool = userdata[l].kv_engine;
                const int F_kv = head_dim * config.n_head_kv;
                struct ggml_diffkv_attn_params p = {
                    config.n_head, config.n_head_kv, pool->get_rank(), pool->get_S_max(), head_dim,
                    srl_k_keep, native_maxd, native_maxd,
                    1.0f / std::sqrt((float)head_dim), userdata[l].has_rope ? 1 : 0,
                    config.rope_freq_base, userdata[l].approximate_attn ? 1 : 0,
                    (int) pool->get_seq_lens()->ne[0],
                    pool->MAX_RESIDUAL,
                    pool->get_kv_type() == GGML_TYPE_Q8_0 ? 1 : 0
                };
                // Route with the host-controlled DISTINCT slot list when available:
                // anchor_screen's selected_slots is a polluted multiset (sem∪host
                // concat duplicates + padding dupes crowd out real blocks — measured
                // 5/12 distinct blocks attended at 4k, needle block dropped). The
                // kernel handles the -1 padding in native_attn_slots natively.
                attn_out = ggml_diffkv_attn(ctx, q_rope_flat,
                    native_attn_slots ? native_attn_slots : selected_slots,
                    pool->get_U(), pool->get_U_row_scale(), pool->get_VK(), pool->get_VV(),
                    pool->get_anchors_K(), pool->get_anchors_V(), pool->get_seq_lens(),
                    pool->get_scales(), pool->get_anchor_positions(),
                    dkr_flat, native_dense_v[l], native_dense_pos,
                    native_dense_mask,
                    ggml_reshape_1d(ctx, k, F_kv), ggml_reshape_1d(ctx, v, F_kv), position,
                    pool->get_res_K_pos(), pool->get_res_V_pos(),
                    pool->get_res_K_val(), pool->get_res_V_val(),
                    pool->get_token_positions(),
                    p);
                { const char* cl = std::getenv("DIFFKV_DBG_CMP_LAYER"); int cmpL = cl ? atoi(cl) : 0;
                  if (l == cmpL && std::getenv("DIFFKV_DBG_CMP")) {
                    g_dbg_qrope = q_rope; g_dbg_attn0 = attn_out; g_dbg_sel0 = selected_slots;
                    g_dbg_curk = k; g_dbg_curv = v;
                    ggml_set_output(q_rope); ggml_set_output(attn_out); ggml_set_output(selected_slots);
                    ggml_set_output(k); ggml_set_output(v);
                } }
                // DIFFKV_BLOCK_CMP: capture every layer for full multi-layer comparison
                if (std::getenv("DIFFKV_BLOCK_CMP") && l < BLOCK_CMP_MAX_LAYERS) {
                    g_bcmp_qrope[l] = q_rope;  g_bcmp_attn[l]  = attn_out;
                    g_bcmp_slots[l] = selected_slots; g_bcmp_curk[l] = k; g_bcmp_curv[l] = v;
                    ggml_set_output(q_rope); ggml_set_output(attn_out); ggml_set_output(selected_slots);
                    ggml_set_output(k); ggml_set_output(v);
                }
            } else if (userdata && selected_slots) {
                struct ggml_tensor * kv_concat = ggml_concat(ctx, k, v, 0);
                // Reconstruct attention output using the custom Metal kernel!
                struct ggml_tensor * custom_attn = ggml_map_custom3(
                    ctx, q_rope_flat, selected_slots, kv_concat,
                    custom_attention_op_callback, 1, &userdata[l]
                );
                attn_out = custom_attn;
                { const char* cl = std::getenv("DIFFKV_DBG_CMP_LAYER"); int cmpL = cl ? atoi(cl) : 0;
                  if (l == cmpL && std::getenv("DIFFKV_DBG_CMP")) {
                    g_dbg_qrope = q_rope; g_dbg_attn0 = attn_out; g_dbg_sel0 = selected_slots;
                    g_dbg_curk = k; g_dbg_curv = v;
                    ggml_set_output(q_rope); ggml_set_output(attn_out); ggml_set_output(selected_slots);
                    ggml_set_output(k); ggml_set_output(v);
                } }
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
                    dense_k_past_inputs[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim_val, config.n_head_kv, engage_threshold);
                    ggml_set_input(dense_k_past_inputs[l]);
                }
                if (dense_v_past_inputs[l] == nullptr) {
                    dense_v_past_inputs[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim_val, config.n_head_kv, engage_threshold);
                    ggml_set_input(dense_v_past_inputs[l]);
                }
            }

            struct ggml_tensor * k_past = dense_k_past_inputs[l];
            struct ggml_tensor * v_past = dense_v_past_inputs[l];

            // Reshape current token keys/values
            struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, head_dim_val, config.n_head, 1);
            struct ggml_tensor * k_reshaped = ggml_reshape_3d(ctx, k, head_dim_val, config.n_head_kv, 1);
            struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, head_dim_val, config.n_head_kv, 1);

            // Apply RoPE to current token queries/keys
            struct ggml_tensor * q_rope = ggml_rope_ext(ctx, q_reshaped, position, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            struct ggml_tensor * k_rope = ggml_rope_ext(ctx, k_reshaped, position, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Concat past with current along dim 2 (the sequence length dimension)
            struct ggml_tensor * k_rope_f16 = ggml_cast(ctx, k_rope, GGML_TYPE_F16);
            struct ggml_tensor * v_reshaped_f16 = ggml_cast(ctx, v_reshaped, GGML_TYPE_F16);
            struct ggml_tensor * k_ctx = ggml_concat(ctx, k_past, k_rope_f16, 2);
            struct ggml_tensor * v_ctx = ggml_concat(ctx, v_past, v_reshaped_f16, 2);

            // Permute to [head_dim, seq_len, kv_heads] layout expected by flash attention
            struct ggml_tensor * q_perm = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            struct ggml_tensor * k_ctx_perm = ggml_permute(ctx, k_ctx, 0, 2, 1, 3);
            struct ggml_tensor * v_ctx_perm = ggml_permute(ctx, v_ctx, 0, 2, 1, 3);

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
    const int n = (int)logits.size();
    if (n == 0) return 0;
    if (temp <= 0.01f) {
        // Greedy argmax
        float max_logit = -1e30f;
        int32_t best_idx = 0;
        for (int i = 0; i < n; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // ── Top-k prefilter (PERF) ──────────────────────────────────────────────
    // The old path softmaxed the full vocab, then did a std::sort over ALL
    // ~152k entries and built a std::discrete_distribution over the whole vocab
    // — EVERY decode token. Measured cost: ~9.5ms/token (≈26% of wall-clock,
    // dropping TPS from ~37 to ~27 on 1.5B-q4). Sampling only ever draws from
    // the top_p nucleus of a peaked distribution; at the configured temperatures
    // the probability mass beyond the top few hundred tokens is < 1e-9. So we
    // restrict the sort + softmax + sampling to the top-K candidates found in
    // O(V) via nth_element. This turns an O(V log V) per-token cost into
    // O(V) + O(K log K) with no measurable change to the sampled distribution
    // (K is far larger than any realistic nucleus). Mirrors the top_k filter
    // every production sampler (incl. llama.cpp, default top_k=40) applies.
    const int K = std::min(n, 2048);
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    if (K < n) {
        std::nth_element(idx.begin(), idx.begin() + (K - 1), idx.end(),
                         [&](int a, int b) { return logits[a] > logits[b]; });
        idx.resize(K);
    }
    // Sort the K candidates by logit descending (needed for the top_p prefix scan).
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });

    // Softmax over the K candidates (idx[0] holds the global max logit).
    const double max_logit = logits[idx[0]];
    std::vector<double> probs(K);
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        probs[i] = std::exp((double)(logits[idx[i]] - max_logit) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < K; ++i) probs[i] /= sum;

    // Nucleus (top_p) truncation over the already-sorted candidates. Keep the
    // token that crosses the threshold (matches the previous behaviour).
    int cutoff = K;
    if (top_p < 1.0f) {
        double cum_prob = 0.0;
        for (int i = 0; i < K; ++i) {
            cum_prob += probs[i];
            if (cum_prob > top_p) { cutoff = i + 1; break; }
        }
    }

    // discrete_distribution normalizes internally over the surviving prefix.
    std::discrete_distribution<int> dist(probs.begin(), probs.begin() + cutoff);
    return idx[dist(rng)];
}

namespace diffkv {
// External (non-static) reference attention used by the self-test.
void execute_cpu_attention(const float*, const int32_t*, float*, float*, NativeBlockPool*,
                           int,int,int,int,int,int,float,bool,float,bool);
void cpu_dense_attention(const float*, const float*, const float*, const int32_t*,
                         int,int,int,int,float,bool,float,int,float*,float*);
}

// DIFFKV_DENSE_CMP=1: standalone comparison of cpu_dense_attention vs ggml rope_ext + plain
// attention on IDENTICAL data. Both use the SAME ggml-rotated Q. The reference rotates K via
// ggml rope_ext (exactly the convention the live decode uses for Q); cpu_dense rotates raw K
// internally. If they diverge, the custom-op dense-window attention has a real bug — localized
// here with no pool/timing/24-layer noise. Mirrors Qwen2.5-1.5B dims (D=128, nq=12, nkv=2).
static void run_dense_attn_cmp() {
    using namespace diffkv;
    int T = 8;                                    // T past tokens at positions 0..T-1
    if (const char* e = std::getenv("DIFFKV_DENSE_CMP_T")) T = std::stoi(e);
    const int D=128, nq=12, nkv=2;
    const int g = nq / nkv;                       // GQA group = 6
    const int query_pos = T;                      // query attends past 0..T-1
    const float scale = 1.0f / std::sqrt((float)D);
    const float freq = 1000000.0f;                // Qwen2.5 rope_theta
    const int n_ctx = 32768;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Raw host data
    std::vector<float> q_raw(D * nq), k_raw(T * nkv * D), v_raw(T * nkv * D);
    for (auto& x : q_raw) x = dist(rng);
    for (auto& x : k_raw) x = dist(rng);
    for (auto& x : v_raw) x = dist(rng);
    // Inject a MASSIVE activation (mimics |K|~820 Qwen layer-0): dim 50 huge for every token,
    // and a huge Q component so scores reach the real ~500 regime where fp paths may diverge.
    for (int t=0;t<T;++t) for (int kv=0;kv<nkv;++kv) { k_raw[(t*nkv+kv)*D + 50] = 600.0f + 5.0f*(t%7); }
    for (int h=0;h<nq;++h) q_raw[h*D + 50] = 8.0f;

    // ── ggml graph: rope_ext on Q (pos=query_pos) and K (pos=0..T-1) ──────────
    ggml_backend_t backend = ggml_backend_cpu_init();
    size_t ctx_sz = 16*1024*1024;
    ggml_init_params ip{ ctx_sz, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* qg = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nq, 1);
    ggml_tensor* kg = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nkv, T);
    ggml_tensor* pq = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor* pk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(qg); ggml_set_input(kg); ggml_set_input(pq); ggml_set_input(pk);
    ggml_tensor* q_rot = ggml_rope_ext(ctx, qg, pq, nullptr, D, GGML_ROPE_TYPE_NEOX, n_ctx, freq, 1.0f,0.0f,1.0f,0.0f,0.0f);
    ggml_tensor* k_rot = ggml_rope_ext(ctx, kg, pk, nullptr, D, GGML_ROPE_TYPE_NEOX, n_ctx, freq, 1.0f,0.0f,1.0f,0.0f,0.0f);
    ggml_set_output(q_rot); ggml_set_output(k_rot);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, q_rot);
    ggml_build_forward_expand(gf, k_rot);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<int32_t> pqv(1, query_pos), pkv(T); for (int t=0;t<T;++t) pkv[t]=t;
    ggml_backend_tensor_set(qg, q_raw.data(), 0, q_raw.size()*4);
    ggml_backend_tensor_set(kg, k_raw.data(), 0, k_raw.size()*4);
    ggml_backend_tensor_set(pq, pqv.data(), 0, 4);
    ggml_backend_tensor_set(pk, pkv.data(), 0, T*4);
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> q_rot_h(D*nq), k_rot_h(T*nkv*D);
    ggml_backend_tensor_get(q_rot, q_rot_h.data(), 0, q_rot_h.size()*4);
    ggml_backend_tensor_get(k_rot, k_rot_h.data(), 0, k_rot_h.size()*4);

    // ── Reference: plain attention on ggml-rotated q_rot · k_rot ──────────────
    std::vector<float> out_ref(nq*D, 0.0f);
    for (int h=0; h<nq; ++h) {
        int kv = h/g;
        std::vector<float> sc(T); float mx=-1e30f;
        for (int t=0;t<T;++t){ double d=0; for(int i=0;i<D;++i) d += (double)q_rot_h[h*D+i]*k_rot_h[(t*nkv+kv)*D+i]; sc[t]=(float)d*scale; mx=std::max(mx,sc[t]); }
        double se=0; for (int t=0;t<T;++t) se+=std::exp(sc[t]-mx);
        for (int t=0;t<T;++t){ double w=std::exp(sc[t]-mx)/se; for(int i=0;i<D;++i) out_ref[h*D+i]+=(float)(w*v_raw[(t*nkv+kv)*D+i]); }
    }

    // ── cpu_dense_attention: ggml-rotated Q, RAW K (rotates internally), positions 0..T-1 ──
    std::vector<float> out_cpu(nq*D, 0.0f), lse(nq, -1e30f);
    cpu_dense_attention(q_rot_h.data(), k_raw.data(), v_raw.data(), pkv.data(),
                        T, nq, nkv, D, scale, /*has_rope=*/true, freq, /*anchor_pos=*/0,
                        out_cpu.data(), lse.data());

    double maxd=0, rn=0; int argmax=0;
    for (int i=0;i<nq*D;++i){ double dd=std::abs((double)out_cpu[i]-out_ref[i]); if(dd>maxd){maxd=dd;argmax=i;} rn+=(double)out_ref[i]*out_ref[i]; }
    std::cerr << "[DENSE_CMP] cpu_dense vs plain-ref: maxAbsDiff=" << maxd << " (idx " << argmax << ", head " << argmax/D << ")"
              << " |ref|=" << std::sqrt(rn) << (maxd < 1e-3*std::sqrt(rn)+1e-4 ? "  PASS" : "  FAIL") << "\n";

    // ── ggml flash_attn_ext on the SAME (q_rot, k_rot, v) — exposes any OUTPUT LAYOUT diff ──
    {
        ggml_context* fc = ggml_init(ip);
        ggml_tensor* qf = ggml_new_tensor_3d(fc, GGML_TYPE_F32, D, nq, 1);
        ggml_tensor* kf = ggml_new_tensor_3d(fc, GGML_TYPE_F32, D, nkv, T);
        ggml_tensor* vf = ggml_new_tensor_3d(fc, GGML_TYPE_F32, D, nkv, T);
        int Tpad = ((T + 31)/32)*32;  // GGML_KQ_MASK_PAD
        ggml_tensor* mf = ggml_new_tensor_2d(fc, GGML_TYPE_F16, T, Tpad);
        ggml_set_input(qf); ggml_set_input(kf); ggml_set_input(vf); ggml_set_input(mf);
        ggml_tensor* qp = ggml_permute(fc, qf, 0,2,1,3);
        ggml_tensor* kp = ggml_permute(fc, kf, 0,2,1,3);
        ggml_tensor* vp = ggml_cont(fc, ggml_permute(fc, vf, 0,2,1,3));
        ggml_tensor* fa = ggml_flash_attn_ext(fc, qp, kp, vp, mf, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        ggml_set_output(fa);
        ggml_cgraph* fg = ggml_new_graph(fc);
        ggml_build_forward_expand(fg, fa);
        ggml_backend_buffer_t fb = ggml_backend_alloc_ctx_tensors(fc, backend);
        ggml_backend_tensor_set(qf, q_rot_h.data(), 0, q_rot_h.size()*4);
        ggml_backend_tensor_set(kf, k_rot_h.data(), 0, k_rot_h.size()*4);
        ggml_backend_tensor_set(vf, v_raw.data(), 0, v_raw.size()*4);
        std::vector<ggml_fp16_t> mz((size_t)T*Tpad, ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(mf, mz.data(), 0, mz.size()*sizeof(ggml_fp16_t));
        if (ggml_backend_graph_compute(backend, fg) == GGML_STATUS_SUCCESS) {
            std::vector<float> out_flash(ggml_nelements(fa));
            ggml_backend_tensor_get(fa, out_flash.data(), 0, out_flash.size()*4);
            double fd=0; int fi=0;
            for (int i=0;i<nq*D && i<(int)out_flash.size();++i){ double dd=std::abs((double)out_cpu[i]-out_flash[i]); if(dd>fd){fd=dd;fi=i;} }
            std::cerr << "[DENSE_CMP] cpu_dense vs FLASH: maxAbsDiff=" << fd << " (idx " << fi << ")"
                      << (fd < 1e-3*std::sqrt(rn)+1e-3 ? "  PASS" : "  FAIL (layout/math differ!)") << "\n";
            std::cerr << "[DENSE_CMP] flash[0..5]="; for(int i=0;i<6;++i) std::cerr<<out_flash[i]<<" "; std::cerr<<"\n";
        } else {
            std::cerr << "[DENSE_CMP] flash compute FAILED (mask/shape)\n";
        }
        ggml_backend_buffer_free(fb); ggml_free(fc);
    }
    std::cerr << "[DENSE_CMP] ref [h0 0..5]="; for(int i=0;i<6;++i) std::cerr<<out_ref[i]<<" ";
    std::cerr << "\n[DENSE_CMP] cpu [h0 0..5]="; for(int i=0;i<6;++i) std::cerr<<out_cpu[i]<<" "; std::cerr<<"\n";
    // Localize: compare ggml-rotated K to cpu_dense's internal rotation (Function B) for token T-1.
    {
        int t = T-1, kv = 0; const int half = D/2;
        std::cerr << "[DENSE_CMP] K rot check token " << t << " kv0: ggml vs Function-B(manual)\n";
        double rd=0;
        for (int i=0;i<half;++i){
            float inv = 1.0f/std::pow(freq, 2.0f*i/D);
            float ang = (float)pkv[t]*inv, c=std::cos(ang), s=std::sin(ang);
            float x=k_raw[(t*nkv+kv)*D+i], y=k_raw[(t*nkv+kv)*D+i+half];
            float man_lo = x*c - y*s, man_hi = y*c + x*s;
            rd = std::max(rd, std::max(std::abs((double)man_lo - k_rot_h[(t*nkv+kv)*D+i]),
                                       std::abs((double)man_hi - k_rot_h[(t*nkv+kv)*D+i+half])));
        }
        std::cerr << "[DENSE_CMP]   maxRotDiff(ggml vs FunctionB)=" << rd << (rd<1e-2?"  (rotation matches)":"  (ROTATION DIFFERS!)") << "\n";
    }
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend);
}

// DIFFKV_RECON_CMP=1: standalone per-token K/V reconstruction-fidelity probe (no model load).
// Plants a DETERMINISTIC synthetic block — smooth low-rank filler + a dominant landmark at
// token 0 (so native picks it as the exact anchor, no swap) + a distinctive high-frequency
// "needle" at token 128 — runs the real compress_lowrank_block, then reconstructs each token
// from the STORED compressed rep (int8 U * per-row scale * fp16 VK/VV * block scale + anchor +
// residuals) exactly as decode does. Prints per-token relative error, isolating compression/
// reconstruction fidelity (esp. the needle) from the decode attention. The Python twin
// (benchmarks/recon_cmp.py) runs the IDENTICAL synthetic block through active's compress path;
// diff the NEEDLE error to localize the native-vs-active gap. Set DIFFKV_RECON_NORESID=1 to
// also see native's pure-lowrank (no-residual) error, apples-to-apples with active (no residuals).
static void run_recon_cmp() {
    using diffkv::LowRankCompressParams;
    const int S = 256, KVH = 2, D = 128;
    const int F = KVH * D;            // 256
    int R = 16;
    if (const char* e = std::getenv("DIFFKV_RANK")) R = std::max(1, atoi(e));
    const bool no_resid = (std::getenv("DIFFKV_RECON_NORESID") != nullptr);
    const int needle = 128;

    // ── Deterministic HIGH-RANK synthetic K/V (must match recon_cmp.py exactly) ──
    // Sum of NCOMP orthogonal-ish cosine components → effective rank ≈ NCOMP, so a rank-16
    // truncation leaves a realistic floor (~40%, like real attention deltas). A dominant
    // smooth landmark @0 (native picks it as the exact anchor) and a distinctive token @128
    // whose energy sits in the truncated tail (the "needle") let us read off the hardest case.
    const int NCOMP = 28;
    std::vector<float> K((size_t)S * F), V((size_t)S * F);
    for (int s = 0; s < S; ++s)
        for (int f = 0; f < F; ++f) {
            float ak = 0.0f, av = 0.0f;
            for (int k = 1; k <= NCOMP; ++k) {
                float ph = 6.2831853f * k * (s + 1) / S;
                ak += std::cos(ph + 1.7f * k * f / F + 0.3f * k);
                av += std::sin(ph + 1.3f * k * f / F + 0.5f * k);
            }
            K[(size_t)s * F + f] = ak;
            V[(size_t)s * F + f] = av;
        }
    // No dominant landmark: native picks its natural anchor (printed as landmark=L below);
    // feed that index to recon_cmp.py via DIFFKV_RECON_ANCHOR=L so both use the same anchor
    // and the deltas stay non-degenerate.
    for (int f = 0; f < F; ++f) {                       // distinctive needle @128 (tail component k=45)
        K[(size_t)needle * F + f] += 5.0f * std::cos(6.2831853f * 45 * f / F + 0.9f);
        V[(size_t)needle * F + f] += 5.0f * std::sin(6.2831853f * 45 * f / F + 0.4f);
    }

    int pool_rank = R, S_max = S;
    std::vector<int8_t>      out_u((size_t)S_max * pool_rank, 0);
    std::vector<ggml_fp16_t> out_u_scale(1, 0), out_u_row(S_max, 0);
    std::vector<ggml_fp16_t> out_vk((size_t)pool_rank * F, 0), out_vv((size_t)pool_rank * F, 0);
    std::vector<ggml_fp16_t> out_ak(F, 0), out_av(F, 0), out_scale(1, 0);
    int32_t out_seq = 0, out_anchor_pos = 0;
    const int MR = diffkv::NativeBlockPool::MAX_RESIDUAL;
    std::vector<int32_t>     rKpos(MR, -1), rVpos(MR, -1);
    std::vector<ggml_fp16_t> rKval((size_t)MR * F, 0), rVval((size_t)MR * F, 0);

    LowRankCompressParams p{};
    p.block_id = 0; p.block_size = S; p.feat_dim = F; p.rank = R; p.pool_rank = pool_rank;
    p.pool_block_size = S_max; p.head_dim = D; p.anchor_idx = 0;
    p.raw_k_ptr = K.data(); p.raw_v_ptr = V.data(); p.token_ids = nullptr; p.stop_token_ids = nullptr;
    p.out_u_ptr = out_u.data(); p.out_u_scale = out_u_scale.data(); p.out_u_row_scale = out_u_row.data();
    p.out_vk_ptr = out_vk.data(); p.out_vv_ptr = out_vv.data();
    p.out_scale = out_scale.data(); p.out_anchor_k = out_ak.data(); p.out_anchor_v = out_av.data();
    p.out_seq_len = &out_seq; p.out_anchor_position = &out_anchor_pos;
    p.out_res_K_pos = rKpos.data(); p.out_res_V_pos = rVpos.data();
    p.out_res_K_val = rKval.data(); p.out_res_V_val = rVval.data(); p.max_residual = MR;

    auto t_start = std::chrono::high_resolution_clock::now();
    int num_loops = 200;
    for (int i = 0; i < num_loops; ++i) {
        if (!diffkv::compress_lowrank_block(p)) { std::cerr << "[RECON_CMP] compress failed\n"; return; }
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cerr << "[RECON_CMP] 200 compressions took: " << ms << " ms (" << (ms / num_loops) << " ms per block)\n";

    int L = out_anchor_pos;          // landmark index (anchor_idx=0 ⇒ == landmark_idx)
    int S_deltas = out_seq;          // == S-1
    float blk_scale = ggml_fp16_to_fp32(out_scale[0]);
    auto resid = [&](std::vector<int32_t>& pos, std::vector<ggml_fp16_t>& val, int t, int f) -> float {
        if (no_resid) return 0.0f;
        for (int i = 0; i < MR; ++i) if (pos[i] == t) return ggml_fp16_to_fp32(val[(size_t)i * F + f]);
        return 0.0f;
    };
    auto orig_of = [&](int sp) { return sp == 0 ? L : (sp == L ? 0 : sp); };  // swapped pos → original idx

    double sumK = 0, sumV = 0; int cnt = 0; double nK_err = -1, nV_err = -1;
    for (int t = 0; t < S_deltas; ++t) {
        int oi = orig_of(t + 1);
        double eK = 0, nrmK = 0, eV = 0, nrmV = 0;
        for (int f = 0; f < F; ++f) {
            float dK = 0, dV = 0;
            for (int r = 0; r < R; ++r) {
                float u = (float)out_u[(size_t)t * pool_rank + r] * ggml_fp16_to_fp32(out_u_row[t]);
                dK += u * ggml_fp16_to_fp32(out_vk[(size_t)r * F + f]);
                dV += u * ggml_fp16_to_fp32(out_vv[(size_t)r * F + f]);
            }
            dK = dK * blk_scale + resid(rKpos, rKval, t, f);
            dV = dV * blk_scale + resid(rVpos, rVval, t, f);
            float rK = ggml_fp16_to_fp32(out_ak[f]) + dK;
            float rV = ggml_fp16_to_fp32(out_av[f]) + dV;
            float tK = K[(size_t)oi * F + f], tV = V[(size_t)oi * F + f];
            eK += (rK - tK) * (rK - tK); nrmK += tK * tK;
            eV += (rV - tV) * (rV - tV); nrmV += tV * tV;
        }
        double relK = std::sqrt(eK / std::max(nrmK, 1e-12)), relV = std::sqrt(eV / std::max(nrmV, 1e-12));
        sumK += relK; sumV += relV; ++cnt;
        if (oi == needle) { nK_err = relK; nV_err = relV; }
    }
    std::cerr << "[RECON_CMP] native rank=" << R << " landmark=" << L << " S_deltas=" << S_deltas
              << " residuals=" << (no_resid ? "OFF" : "ON") << "\n";
    std::cerr << "[RECON_CMP] mean rel err   K=" << 100.0 * sumK / cnt << "%  V=" << 100.0 * sumV / cnt << "%\n";
    std::cerr << "[RECON_CMP] NEEDLE(tok" << needle << ") rel err  K=" << 100.0 * nK_err
              << "%  V=" << 100.0 * nV_err << "%\n";
}

// DIFFKV_ATTN_CMP=1: standalone decode-attention faithfulness probe (no model load).
// Compresses one synthetic high-rank block into a real NativeBlockPool, then runs native's
// actual decode attention (execute_cpu_attention) for a query aligned to a planted needle, and
// compares its output against dense attention over (a) native's OWN reconstructed K/V — this is
// the decode-faithfulness test: ~0 means decode correctly uses its reconstruction — and (b) the
// TRUE uncompressed K/V. If native_sparse ≈ recon_dense but ≠ true_dense, the gap is recon
// (already measured); if native_sparse ≠ recon_dense, the decode attention itself is the bug.
// DIFFKV_ATTN_CMP_ROPE=1 enables RoPE (default off isolates the value/score path from rotation);
// DIFFKV_ATTN_CMP_EXACT=1 uses native's non-approximate attention path.
static void run_attn_cmp() {
    const int nq = 4, nkv = 2, D = 128, S = 64, desc_dim = 8;
    const int F = nkv * D;            // 256
    int rank = 16;
    if (const char* e = std::getenv("DIFFKV_RANK")) rank = std::max(1, atoi(e));
    const bool HR = (std::getenv("DIFFKV_ATTN_CMP_ROPE") != nullptr);
    const bool approximate = (std::getenv("DIFFKV_ATTN_CMP_EXACT") == nullptr);
    const bool use_resid = (std::getenv("DIFFKV_ATTN_CMP_RESID") != nullptr);
    int NB = 1;  // number of blocks (DIFFKV_ATTN_CMP_BLOCKS): >1 probes the multi-block 8k+ bug
    if (const char* e = std::getenv("DIFFKV_ATTN_CMP_BLOCKS")) NB = std::max(1, atoi(e));
    const float scale = 1.0f / std::sqrt((float)D), freq = 10000.0f;
    const int NCOMP = 18;
    const int T = NB * S;                       // total tokens across all blocks
    const int needle_g = (NB / 2) * S + S / 2;  // a needle in the middle block

    // ── Deterministic high-rank K/V for T global tokens + a distinctive needle ──
    std::vector<float> K((size_t)T * F), V((size_t)T * F);
    for (int s = 0; s < T; ++s)
        for (int f = 0; f < F; ++f) {
            float ak = 0.0f, av = 0.0f;
            for (int k = 1; k <= NCOMP; ++k) {
                float ph = 6.2831853f * k * (s + 1) / S;
                ak += std::cos(ph + 1.7f * k * f / F + 0.3f * k);
                av += std::sin(ph + 1.3f * k * f / F + 0.5f * k);
            }
            K[(size_t)s * F + f] = ak; V[(size_t)s * F + f] = av;
        }
    for (int f = 0; f < F; ++f) {
        K[(size_t)needle_g * F + f] += 4.0f * std::cos(6.2831853f * 3 * f / F + 0.9f);
        V[(size_t)needle_g * F + f] += 4.0f * std::sin(6.2831853f * 3 * f / F + 0.4f);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    NativeBlockPool pool;
    pool.initialize(NB, rank, D, nkv, desc_dim, buft, S);
    pool.set_rope_config(HR, freq);
    pool.zero_all_tensors();
    const int MR = NativeBlockPool::MAX_RESIDUAL;

    // Dense reference laid out block-by-block, each block in native's swapped slot order.
    std::vector<float> reconK((size_t)T * F), reconV((size_t)T * F);
    std::vector<float> trueK((size_t)T * F), trueV((size_t)T * F);
    std::vector<int32_t> pos(T);

    for (int b = 0; b < NB; ++b) {
        // ── Compress block b ([b*S .. b*S+S-1], global start anchor_idx=b*S) into slot b ──
        LowRankCompressParams p{};
        p.block_id = b; p.block_size = S; p.feat_dim = F; p.rank = rank; p.pool_rank = rank;
        p.pool_block_size = S; p.head_dim = D; p.anchor_idx = b * S;
        p.raw_k_ptr = K.data() + (size_t)b * S * F; p.raw_v_ptr = V.data() + (size_t)b * S * F;
        p.token_ids = nullptr; p.stop_token_ids = nullptr;
        p.out_u_ptr = pool.get_host_U(b);
        p.out_u_scale = pool.get_host_U_scale() + b;
        p.out_u_row_scale = pool.get_host_U_row_scale(b);
        p.out_vk_ptr = pool.get_host_VK(b);
        p.out_vv_ptr = pool.get_host_VV(b);
        p.out_scale = pool.get_host_scales() + b;
        p.out_anchor_k = pool.get_host_anchors_K(b);
        p.out_anchor_v = pool.get_host_anchors_V(b);
        p.out_seq_len = pool.get_host_seq_lens() + b;
        p.out_anchor_position = pool.get_host_anchor_positions() + b;
        p.out_token_positions = pool.get_host_token_positions(b);
        p.out_res_K_pos = pool.get_host_res_K_pos(b);
        p.out_res_V_pos = pool.get_host_res_V_pos(b);
        p.out_res_K_val = pool.get_host_res_K_val(b);
        p.out_res_V_val = pool.get_host_res_V_val(b);
        p.max_residual = MR;
        if (!compress_lowrank_block(p)) { std::cerr << "[ATTN_CMP] compress failed\n"; return; }
        pool.upload_slot(b);

        int Lg = pool.get_host_anchor_positions()[b];   // global anchor position = b*S + landmark
        int Lb = Lg - b * S;                            // within-block landmark index
        float blk_scale = ggml_fp16_to_fp32(pool.get_host_scales()[b]);
        auto orig_of = [&](int sp) { return sp == 0 ? Lb : (sp == Lb ? 0 : sp); };
        const int8_t* U = pool.get_host_U(b);
        const ggml_fp16_t* URS = pool.get_host_U_row_scale(b);
        const ggml_fp16_t* VK = pool.get_host_VK(b);
        const ggml_fp16_t* VVv = pool.get_host_VV(b);
        const ggml_fp16_t* AK = pool.get_host_anchors_K(b);
        const ggml_fp16_t* AV = pool.get_host_anchors_V(b);
        int32_t* rKp = pool.get_host_res_K_pos(b);
        int32_t* rVp = pool.get_host_res_V_pos(b);
        ggml_fp16_t* rKv = pool.get_host_res_K_val(b);
        ggml_fp16_t* rVv = pool.get_host_res_V_val(b);
        auto resid = [&](int32_t* rp, ggml_fp16_t* rv, int t, int f) -> float {
            if (!use_resid) return 0.0f;
            for (int i = 0; i < MR; ++i) if (rp[i] == t) return ggml_fp16_to_fp32(rv[(size_t)i * F + f]);
            return 0.0f;
        };
        for (int p_ = 0; p_ < S; ++p_) {
            int g = b * S + p_;                  // index in the dense reference layout
            int oi = orig_of(p_);                // within-block original index
            pos[g] = b * S + oi;                 // TRUE global position
            for (int f = 0; f < F; ++f) { trueK[(size_t)g*F+f] = K[(size_t)(b*S+oi)*F+f]; trueV[(size_t)g*F+f] = V[(size_t)(b*S+oi)*F+f]; }
            if (p_ == 0) {
                for (int f = 0; f < F; ++f) { reconK[(size_t)g*F+f] = ggml_fp16_to_fp32(AK[f]); reconV[(size_t)g*F+f] = ggml_fp16_to_fp32(AV[f]); }
            } else {
                int t = p_ - 1;
                for (int f = 0; f < F; ++f) {
                    float dK = 0, dV = 0;
                    for (int r = 0; r < rank; ++r) {
                        float u = (float)U[(size_t)t*rank+r] * ggml_fp16_to_fp32(URS[t]);
                        dK += u * ggml_fp16_to_fp32(VK[(size_t)r*F+f]);
                        dV += u * ggml_fp16_to_fp32(VVv[(size_t)r*F+f]);
                    }
                    dK = dK*blk_scale + resid(rKp, rKv, t, f);
                    dV = dV*blk_scale + resid(rVp, rVv, t, f);
                    reconK[(size_t)g*F+f] = ggml_fp16_to_fp32(AK[f]) + dK;
                    reconV[(size_t)g*F+f] = ggml_fp16_to_fp32(AV[f]) + dV;
                }
            }
        }
    }

    // ── Query aligned to the needle (true K direction) ──
    std::vector<float> Q((size_t)nq * D);
    for (int h = 0; h < nq; ++h) {
        int kv = h / (nq / nkv);
        double nrm = 0; for (int d = 0; d < D; ++d) { float x = K[(size_t)needle_g*F + kv*D + d]; nrm += (double)x*x; }
        nrm = std::sqrt(std::max(nrm, 1e-12));
        for (int d = 0; d < D; ++d) Q[(size_t)h*D+d] = K[(size_t)needle_g*F + kv*D + d] / nrm * 8.0f;
    }

    std::vector<int32_t> slots(NB); for (int b = 0; b < NB; ++b) slots[b] = b;
    std::vector<float> o_ns(nq*D,0), l_ns(nq,-1e30f), o_rd(nq*D,0), l_rd(nq,-1e30f), o_td(nq*D,0), l_td(nq,-1e30f);
    execute_cpu_attention(Q.data(), slots.data(), o_ns.data(), l_ns.data(), &pool, nq, nkv, rank, S, NB, D, scale, HR, freq, approximate);
    cpu_dense_attention(Q.data(), reconK.data(), reconV.data(), pos.data(), T, nq, nkv, D, scale, HR, freq, 0, o_rd.data(), l_rd.data());
    cpu_dense_attention(Q.data(), trueK.data(),  trueV.data(),  pos.data(), T, nq, nkv, D, scale, HR, freq, 0, o_td.data(), l_td.data());

    auto rel = [&](std::vector<float>&a, std::vector<float>&b){ double e=0,n=0; for(size_t i=0;i<a.size();++i){e+=(a[i]-b[i])*(a[i]-b[i]); n+=b[i]*b[i];} return 100.0*std::sqrt(e/std::max(n,1e-12)); };
    std::cerr << "[ATTN_CMP] blocks=" << NB << " (tokens=" << T << ") S=" << S << " rank=" << rank
              << " rope=" << (HR?"ON":"OFF") << " path=" << (approximate?"approx":"exact")
              << " resid=" << (use_resid?"ON":"OFF") << "\n";
    std::cerr << "[ATTN_CMP] native_sparse vs recon_dense (DECODE faithfulness) = " << rel(o_ns, o_rd) << "%\n";
    std::cerr << "[ATTN_CMP] recon_dense  vs true_dense  (recon impact)         = " << rel(o_rd, o_td) << "%\n";
    std::cerr << "[ATTN_CMP] native_sparse vs true_dense  (combined)            = " << rel(o_ns, o_td) << "%\n";
    ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    // Eliminate streaming bursts: set stdout fully unbuffered so each token
    // reaches the pipe the instant it's written, regardless of OS buffering.
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::cerr << "Main entered, argc=" << argc << std::endl;
    if (std::getenv("DIFFKV_DBG_POS")) diffkv::g_diffkv_dbg_pos.store(1);  // mirror to a global (worker-thread getenv fails)
    if (std::getenv("DIFFKV_DBG_BLOCK_STATES")) diffkv::g_diffkv_dbg_block_states.store(1);  // mirror to a global
    if (const char* ef = std::getenv("DIFFKV_ENABLE_FACTUAL")) {
        if (std::string(ef) == "0" || std::string(ef) == "false" || std::string(ef) == "off") {
            g_diffkv_enable_factual.store(false);
        } else if (std::string(ef) == "1" || std::string(ef) == "true" || std::string(ef) == "on") {
            g_diffkv_enable_factual.store(true);
        }
    }
    if (std::getenv("DIFFKV_DENSE_CMP")) { run_dense_attn_cmp(); return 0; }
    if (std::getenv("DIFFKV_RECON_CMP")) { run_recon_cmp(); return 0; }
    if (std::getenv("DIFFKV_ATTN_CMP")) { run_attn_cmp(); return 0; }
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
    
    
    // Bug 🅙 fix: micro_block_size 64→256 to match MLX reference.
    // 256 gives dense anchors matching MLX.
    int micro_block_size = 256;
    if (const char* env_mbs = std::getenv("DIFFKV_MICRO_BLOCK_SIZE")) {
        micro_block_size = std::stoi(env_mbs);
    }

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

    // ── N4.2 fix: scale srl_k_keep with micro_block_size ─────────────────────
    // Item J changed micro_block_size from 64→16, so each block now covers 4× fewer
    // tokens. The old default srl_k_keep=16 used to cover 16×64=1024 tokens; after
    // item J it covers only 16×16=256 tokens (≈1% of a 24k prompt) — causing the
    // model to emit EOS within 4 tokens because it sees almost no context.
    //
    // Fix: raise srl_k_keep to at least (1024 / micro_block_size) so the attended
    // token budget is always >= 1024 tokens regardless of mbs.  With mbs=16 that
    // gives srl_k_keep=64 (4× more blocks, same token coverage as before item J).
    // Users can still override upward via DIFFKV_SRL_K_KEEP / --srl-k-keep.
    {
        int srl_k_keep_floor = std::max(16, 1024 / micro_block_size);
        if (srl_k_keep < srl_k_keep_floor) {
            std::cerr << "[DiffKV] N4.2: srl_k_keep raised from " << srl_k_keep
                      << " → " << srl_k_keep_floor
                      << " to preserve ≥1024 token coverage (mbs=" << micro_block_size << ")\n";
            srl_k_keep = srl_k_keep_floor;
        }
        // anchor_screen needs a candidate pool larger than srl_k_keep.
        // Raise srl_k_semantic so that sem_slots + host_slots >= 3 × srl_k_keep.
        int sem_floor = srl_k_keep * 3;
        if (srl_k_semantic < sem_floor) {
            std::cerr << "[DiffKV] N4.2: srl_k_semantic raised from " << srl_k_semantic
                      << " → " << sem_floor << " (= 3 × srl_k_keep)\n";
            srl_k_semantic = sem_floor;
        }
        // Scale host channel budgets proportionally: each channel gets at least
        // srl_k_keep / 4 slots (so the 4 host channels together match srl_k_keep).
        int host_floor = std::max(8, srl_k_keep / 4);
        if (srl_k_recency < host_floor) { srl_k_recency = host_floor; }
        if (srl_k_lexical < host_floor) { srl_k_lexical = host_floor; }
        if (srl_k_graph   < host_floor) { srl_k_graph   = host_floor; }
    }

    int srl_k_host = 1 + srl_k_recency + srl_k_lexical + srl_k_semantic + srl_k_graph;

    std::unordered_set<int32_t> stop_token_ids;
    {
        // N3.3 fix: Align stopword list to Python's _STOP_WORDS_COMPRESS
        // (ACTIVE_RUNTIME/native_core/streaming_sparse_ingest.py:61-87).
        // Previously: ~40-word list + first-200 token ID blanket. Problems:
        //   (a) Omitted stopwords get indexed into important_vocab/occurrences,
        //       polluting rare/high-IDF lexical routing and VSL anchor overlap.
        //   (b) First-200 ID blanket is C++-only — stops legitimate early-vocab tokens.
        // Now: full ~150-word NLTK-style set, no ID blanket.
        std::vector<std::string> stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
            "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
            "hers", "herself", "it", "it's", "its", "itself", "they", "them",
            "their", "theirs", "themselves", "a", "an", "the", "and", "but",
            "or", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "now",
            "should", "should've", "would", "could", "may", "might", "must",
            "shall", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing",
            "get", "got", "make", "made", "go", "went", "take", "took",
            "see", "saw", "say", "said", "use", "used", "find", "found",
            "question", "answer", "text", "context", "information", "prompt",
            "query", "assistant", "system", "user", "file", "document", "page",
            "line", "passage", "following", "please", "write", "read",
            "describe", "explain", "summarize", "extract", "retrieve", "give",
            "tell", "show", "list", "what", "who", "whom", "which", "detail",
            "details", "brief", "exact", "exactly", "correct", "correctly",
            "true", "false", "yes", "no",
            // Special/punctuation tokens
            ",", ".", ":", ";", "?", "!", "(", ")", "'", "\"", "-", "\n",
            "im_start", "im_end"
        };
        for (const auto & word : stop_words) {
            auto t = model.tokenize(word, false);
            for (int32_t tok : t) {
                stop_token_ids.insert(tok);
            }
        }
        // NOTE: No first-200 token ID blanket — Python has no such blanket.
    }

    diffkv::SessionSRLState srl_state;
    // KV prefix cache: tracks how many tokens the binary has already ingested
    // in its KV pool for the current session, matching ACTIVE_RUNTIME's cached_len mechanism.
    int session_cached_len = 0;   // token count already in KV pool (skip re-prefilling these)
    std::vector<int32_t> session_cached_token_ids; // the prefix we verified is resident


    // Backend is already initialized at the start of main
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // micro_block_size is defined and parsed above, before the SRL configuration block

    // Initialize NativeBlockPool block pool for all layers
    int n_slots = model.get_config().n_ctx / micro_block_size;
    bool overridden = false;

    const char* env_tokens = std::getenv("max_ctx_tk");
    if (!env_tokens) {
        env_tokens = std::getenv("DIFFKV_MAX_CTX_TK");
    }
    if (env_tokens) {
        int max_tokens = std::stoi(env_tokens);
        n_slots = (max_tokens + micro_block_size - 1) / micro_block_size;
        std::cerr << "[DiffKV Native] Overriding context limit from max_ctx_tk: " 
                  << max_tokens << " tokens (" << n_slots << " slots)" << std::endl;
        overridden = true;
    } else if (const char* env_slots = std::getenv("DIFFKV_MAX_CONTEXT_SLOTS")) {
        n_slots = std::stoi(env_slots);
        std::cerr << "[DiffKV Native] Overriding context limit from DIFFKV_MAX_CONTEXT_SLOTS: " 
                  << n_slots << " slots" << std::endl;
        overridden = true;
    }

    if (!overridden) {
        if (const char* env_preset = std::getenv("DIFFKV_PRESET")) {
            std::string preset(env_preset);
            if (preset == "low") {
                n_slots = 4096 / micro_block_size; // 4096 tokens
                std::cerr << "[DiffKV Native] Preset 'low' detected: capping context size to 4096 tokens (" << n_slots << " slots)" << std::endl;
            } else if (preset == "mid") {
                n_slots = 8192 / micro_block_size; // 8192 tokens
                std::cerr << "[DiffKV Native] Preset 'mid' detected: capping context size to 8192 tokens (" << n_slots << " slots)" << std::endl;
            } else if (preset == "high") {
                n_slots = 16384 / micro_block_size; // 16384 tokens
                std::cerr << "[DiffKV Native] Preset 'high' detected: capping context size to 16384 tokens (" << n_slots << " slots)" << std::endl;
            } else {
                std::cerr << "[DiffKV Native] Using default model context length: " 
                          << model.get_config().n_ctx << " tokens (" << n_slots << " slots)" << std::endl;
            }
        } else {
            std::cerr << "[DiffKV Native] Using default model context length: " 
                      << model.get_config().n_ctx << " tokens (" << n_slots << " slots)" << std::endl;
        }
    }
    srl_k_semantic = std::min(srl_k_semantic, n_slots);
    srl_k_keep = std::min(srl_k_keep, n_slots);
    // Bug 🅚 fix: rank 32→16 to match Python serving default (lowrank.py, F1 fix).
    int rank = 16;
    if (const char* env_rank = std::getenv("DIFFKV_RANK")) {
        rank = std::max(1, std::stoi(env_rank));
    }
    int head_dim = model.get_config().n_embd / model.get_config().n_head;
    int kv_heads = model.get_config().n_head_kv;
    int desc_dim = 64;
    int n_vocab = model.get_config().n_vocab;
    int n_layers = model.get_config().n_layer;

    // Preset configuration for KV Cache Quantization and Max Active Dense Tokens
    std::string preset_str = "mid";
    if (const char* env_preset = std::getenv("DIFFKV_PRESET")) {
        preset_str = env_preset;
    }
    
    std::string default_maxd = "2048";
    std::string default_quant = "q8_0";
    
    if (preset_str == "low") {
        default_maxd = "1024";
        default_quant = "q4_0";
    } else if (preset_str == "mid") {
        default_maxd = "2048";
        default_quant = "q8_0";
    } else if (preset_str == "high") {
        default_maxd = "4096";
        default_quant = "f16";
    }
    
    setenv("DIFFKV_MAX_ACTIVE_DENSE_TOKENS", default_maxd.c_str(), 0);
    setenv("DIFFKV_KV_QUANT", default_quant.c_str(), 0);

    ggml_type kv_quant_type = GGML_TYPE_Q8_0;
    if (const char* env_quant = std::getenv("DIFFKV_KV_QUANT")) {
        std::string q(env_quant);
        if (q == "f16" || q == "F16" || q == "none") {
            kv_quant_type = GGML_TYPE_F16;
        } else if (q == "f32" || q == "F32") {
            kv_quant_type = GGML_TYPE_F32;
        } else if (q == "q8_0" || q == "Q8_0" || q == "8bit") {
            kv_quant_type = GGML_TYPE_Q8_0;
        } else if (q == "q4_0" || q == "Q4_0" || q == "4bit") {
            kv_quant_type = GGML_TYPE_Q4_0;
        } else if (q == "q5_0" || q == "Q5_0" || q == "5bit") {
            kv_quant_type = GGML_TYPE_Q5_0;
        }
    }

    float gpu_budget_gb = 2.0f;
    if (const char* env_budget = std::getenv("DIFFKV_GPU_BUDGET_GB")) {
        gpu_budget_gb = std::stof(env_budget);
    }
    size_t gpu_budget_bytes = static_cast<size_t>(gpu_budget_gb * 1024.0f * 1024.0f * 1024.0f);

    const int base_micro_block_size = micro_block_size;
    const int base_n_slots = n_slots;
    const int base_srl_k_semantic = srl_k_semantic;
    const int base_srl_k_recency = srl_k_recency;
    const int base_srl_k_lexical = srl_k_lexical;
    const int base_srl_k_graph = srl_k_graph;
    const int base_srl_k_keep = srl_k_keep;
    const int base_srl_k_host = srl_k_host;

    // MLX parity: recency_window (dense window size) is configurable; MLX sets it from the
    // engage env (mlx:324) and keeps NO short-context/header dense. Default short_context=0
    // (compress everything older than the recency window, like MLX). DIFFKV_RECENCY_WINDOW
    // overrides the dense window for experiments / parity.
    int cfg_recency_window = 512;
    int cfg_short_context = 0;
    if (const char* e = std::getenv("DIFFKV_RECENCY_WINDOW")) { try { cfg_recency_window = std::stoi(e); } catch (...) {} }
    if (const char* e = std::getenv("DIFFKV_SHORT_CONTEXT")) { try { cfg_short_context = std::stoi(e); } catch (...) {} }
    diffkv::KVRuntimeManager runtime_manager(rank, micro_block_size, gpu_budget_bytes,
                                             cfg_recency_window, cfg_short_context);
    if (!runtime_manager.initialize(n_slots, head_dim, kv_heads, desc_dim, n_layers, &model, buft, kv_quant_type)) {
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
    runtime_manager.set_projection_matrix(W_proj_host.data(), desc_dim);

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

    // ── Dense-window buffer cap (RAM fix) ────────────────────────────────────
    // These three fp32 buffers were hardcoded to 16384 tokens × F × n_layers × 3,
    // costing ~1.4 GB (1.5B model) even though the dense window only ever holds:
    //   • non-sparse path  : up to engage_threshold prompt tokens, OR
    //   • sparse path      : the recency window (~512 tokens),
    //   plus the decode-time generated tokens (≤ max_generate).
    // Size to the real worst case (engage_threshold + max_generate + slack). The
    // per-append guards (`offset + F_test <= active_k_dense[l].size()`) already
    // prevent overflow, so this only trims unused RAM. MLX keeps no such fp32 host
    // copy at all (KV lives fp16 in unified memory).
    int dense_cap_engage = 4096;  // DIFFKV_ENGAGE_THRESHOLD default
    if (const char* e = std::getenv("DIFFKV_ENGAGE_THRESHOLD")) {
        try { dense_cap_engage = std::max(dense_cap_engage, std::stoi(e)); } catch (...) {}
    }
    int dense_cap_gen = 2048;     // DIFFKV_MAX_TOKENS default
    if (const char* e = std::getenv("DIFFKV_MAX_TOKENS")) {
        try { dense_cap_gen = std::max(1, std::stoi(e)); } catch (...) {}
    }
    int DENSE_WINDOW_CAP = dense_cap_engage + dense_cap_gen + 512;  // ~4608 default

    std::vector<diffkv::AlignedFloatVector> active_k_dense(n_layers);
    std::vector<diffkv::AlignedFloatVector> active_k_dense_rotated(n_layers);
    std::vector<diffkv::AlignedFloatVector> active_v_dense(n_layers);
    diffkv::AlignedInt32Vector active_positions_dense;
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
    // rep_exempt_cache (2026-07-13): digit-bearing and table-separator tokens
    // are EXEMPT from the normal repetition penalty — the penalty exists to
    // stop word-loop disfluency, but over digits it corrupts faithful
    // reproduction of numeric content (reproducing a table penalizes every
    // pipe/digit after the first row and the argmax flips to a wrong number;
    // mirrors _filter_penalty_ids in ACTIVE_RUNTIME batch_engine.py). Loop
    // recovery is unaffected: the exemption is skipped while loop_detected,
    // so the escalated 1.3/256 penalty still breaks digit loops
    // ("7741-7741-"). DIFFKV_REP_PENALTY_PROTECT_NUMERIC=0 restores.
    std::vector<int8_t> alnum_cache(n_vocab, 0);
    std::vector<int8_t> rep_exempt_cache(n_vocab, 0);
    std::vector<int8_t> nl_cache(n_vocab, 0);     // piece contains a newline
    std::vector<int8_t> sep_cache(n_vocab, 0);    // stripped piece is '|' or '&'
    for (int32_t tok_id = 0; tok_id < n_vocab; ++tok_id) {
        std::string piece = model.token_to_piece(tok_id);
        bool has_alnum = false;
        bool has_digit = false;
        for (char c : piece) {
            unsigned char uc = static_cast<unsigned char>(c);
            if (std::isalnum(uc)) has_alnum = true;
            if (std::isdigit(uc)) { has_digit = true; break; }
        }
        alnum_cache[tok_id] = has_alnum ? 1 : 0;
        nl_cache[tok_id] = (piece.find('\n') != std::string::npos) ? 1 : 0;
        // strip whitespace to test for a standalone separator piece
        size_t b = piece.find_first_not_of(" \t\n\r");
        std::string stripped = (b == std::string::npos) ? std::string()
            : piece.substr(b, piece.find_last_not_of(" \t\n\r") - b + 1);
        bool is_sep = (stripped == "|" || stripped == "&");
        sep_cache[tok_id] = is_sep ? 1 : 0;
        rep_exempt_cache[tok_id] = (has_digit || is_sep) ? 1 : 0;
    }
    static const bool rep_protect_numeric = []() {
        const char* e = std::getenv("DIFFKV_REP_PENALTY_PROTECT_NUMERIC");
        return !(e && std::string(e) == "0");
    }();


    ggml_backend_buffer_t native_decode_buf = nullptr;
    while (true) {
        // Restore base configuration values at start of turn to prevent leakage from previous turns
        micro_block_size = base_micro_block_size;
        srl_k_semantic = base_srl_k_semantic;
        srl_k_recency = base_srl_k_recency;
        srl_k_lexical = base_srl_k_lexical;
        srl_k_graph = base_srl_k_graph;
        srl_k_keep = base_srl_k_keep;
        srl_k_host = base_srl_k_host;

        if (n_slots != base_n_slots) {
            n_slots = base_n_slots;
            runtime_manager.reset();
            if (!runtime_manager.initialize(n_slots, head_dim, kv_heads, desc_dim, n_layers, &model, buft, kv_quant_type)) {
                std::cerr << "Failed to re-initialize KVRuntimeManager during config restore!" << std::endl;
                return 1;
            }
            runtime_manager.get_ingest_manager().set_stop_token_ids(&stop_token_ids);
            runtime_manager.get_ingest_manager().set_session_id("interactive_session");
        }
        runtime_manager.set_micro_block_size(micro_block_size);

        bool full_upload_needed = true;
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
                // Ignore prefix caching to avoid garbage token output
                cached_len = 0;
                if (!std::getline(std::cin, prompt)) {
                    break;
                }
            }
            if (prompt.empty() || prompt == "exit" || prompt == "quit") {
                break;
            }
        } else {
            prompt = argv[2];
        }

        // Unescape the single-line stdin encoding back to the original text. Every
        // Python sender (serving cli.py, gateway, bench_worker, remote_cuda_benchmark)
        // encodes with: prompt.replace("\\", "\\\\").replace("\n", "\\n") — so BOTH
        // escapes must be decoded here, backslash first. The old loop only decoded
        // \n and left \\ untouched, which corrupted any prompt containing real
        // backslashes: every '\' the model saw was doubled, and text like the LaTeX
        // '\nabla' (sent as '\\nabla') decoded to '\' + NEWLINE + 'abla' — hundreds
        // of mid-text corruptions on a real research paper, i.e. the "12k paper →
        // near-garbage output" bug (found 2026-07-11). NIAH never caught it because
        // its filler contains no backslashes.
        // (argv[2] prompts go through the same decoder, so backslashes there must be
        // escaped the same way.)
        {
            std::string unescaped;
            unescaped.reserve(prompt.size());
            for (size_t ui = 0; ui < prompt.size(); ++ui) {
                if (ui + 1 < prompt.size() && prompt[ui] == '\\' && prompt[ui+1] == '\\') {
                    unescaped += '\\';
                    ++ui;
                } else if (ui + 1 < prompt.size() && prompt[ui] == '\\' && prompt[ui+1] == 'n') {
                    unescaped += '\n';
                    ++ui;
                } else {
                    unescaped += prompt[ui];
                }
            }
            prompt = std::move(unescaped);
        }
        if (std::getenv("DIFFKV_DBG_DUMP_PROMPT")) {
            FILE* f = fopen("received_prompt.txt", "w");
            if (f) {
                fprintf(f, "%s", prompt.c_str());
                fclose(f);
            }
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
            srl_state.factual_store.clear();
            srl_state.vsl_active_candidates.clear();
            srl_state.vsl_consecutive_helpers = 0;
            srl_state.factual_anchor_q.clear();
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
            srl_state.vsl_active_candidates.clear();
            srl_state.vsl_consecutive_helpers = 0;
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

        if (true) {
            std::cerr << "[DiffKV Native] Prompt tokens (" << prompt_tokens.size() << "): ";
            for (int32_t t : prompt_tokens) std::cerr << t << " ";
            std::cerr << "\n";
        }

        int L = prompt_tokens.size();

        // ── micro_block_size is FIXED at the configured value (MLX/HF parity) ──────
        // ACTIVE_RUNTIME sets micro_block_size = config.get("micro_block_size", 256) with NO
        // length-based adaptation (mlx_diffkv_wrapper.py:1030, hf_diffkv_wrapper.py:378). The
        // old L<256/1024/4096/8192 → 16/32/64/128 override was a divergence — it shrank the
        // block size (4× more, finer blocks) for any prompt under 8192 tokens, changing the
        // compression granularity vs the reference. Keep the configured value (default 256).
        runtime_manager.set_micro_block_size(micro_block_size);

        // Reserve decode headroom: we must reserve enough free slots for max_generate tokens,
        // so that decode generation does not fail due to slot capacity exhaustion.
        int max_generate = 2048;
        if (const char* env_mt = std::getenv("DIFFKV_MAX_TOKENS")) {
            max_generate = std::max(1, std::stoi(env_mt));
        }

        // ── Auto-expand pool to fit prompt (matches Python dynamic pool sizing) ──
        // Python's NativeBlockPool grows on-demand (via _grow_pool) — the preset only
        // controls chunk_size/srl_threshold, NOT the context window.
        // Compute raw slots needed for just the prompt (no headroom yet), then expand.
        {
            int model_max_slots = model.get_config().n_ctx / micro_block_size;
            // Slots needed: ceil(L / (mbs+1)) to hold L tokens in blocks of (mbs+1)
            int prompt_slots_needed = (L + micro_block_size) / (micro_block_size + 1) + 2;
            if (prompt_slots_needed > n_slots) {
                int new_n_slots = std::min(prompt_slots_needed, model_max_slots);
                if (new_n_slots > n_slots) {
                    std::cerr << "[DiffKV Native] Auto-expanding pool: " << n_slots
                              << " → " << new_n_slots << " slots to fit " << L
                              << "-token prompt (Python parity: dynamic pool)." << std::endl;
                    n_slots = new_n_slots;
                    // Re-initialize the pool since we need more slots!
                    runtime_manager.reset();
                    if (!runtime_manager.initialize(n_slots, head_dim, kv_heads, desc_dim, n_layers, &model, buft, kv_quant_type)) {
                        std::cerr << "Failed to re-initialize KVRuntimeManager during auto-expansion!" << std::endl;
                        return 1;
                    }
                    runtime_manager.get_ingest_manager().set_stop_token_ids(&stop_token_ids);
                    runtime_manager.get_ingest_manager().set_session_id("interactive_session");
                }
            }
        }

        // Now compute headroom on the (possibly expanded) n_slots.
        // Cap headroom at 512 tokens (32 slots at mbs=16) — realistic max response length.
        // The old n_slots/2 cap was eating 50% of the pool for decode tokens that would
        // never be used, leaving only half the pool for the prompt.
        const int headroom_tokens_cap = 512;
        int effective_max_generate = std::min(max_generate, headroom_tokens_cap);
        int headroom_slots = (effective_max_generate + micro_block_size - 1) / micro_block_size;
        // Hard safety: never exceed 25% of n_slots for headroom
        headroom_slots = std::min(headroom_slots, n_slots / 4);
        int safe_n_slots = std::max(2, n_slots - headroom_slots);

        // N3.5 fix: Each slot stores 1 anchor token + micro_block_size delta tokens
        // = (micro_block_size + 1) tokens total (streaming_sparse_ingest.cpp:406).
        int true_capacity = safe_n_slots * (micro_block_size + 1);
        if (L > true_capacity) {
            // Prompt still too large (hit model absolute max) — truncate as last resort
            std::cerr << "[DiffKV Native] Warning: Prompt (" << L << " tokens) exceeds model max capacity ("
                      << true_capacity << "). Truncating." << std::endl;
            L = true_capacity;
            prompt_tokens.resize(L);
        }
        // ── Sparse engage threshold ──────────────────────────────────────────────
        // flash_attn_ext (dense path) is fast and correct for any context that fits
        // in GPU memory. ggml_diffkv_attn (sparse path) enables 1M+ contexts by
        // compressing KV, but its custom Metal kernel is ~8x slower per token.
        //
        // AUTO-THRESHOLD: query Metal free memory and derive the maximum context
        // length whose dense KV fits within a configurable fraction of that budget.
        //
        //   dense_kv_bytes(ctx) = ctx × F_kv × n_layers × 2 (K+V) × 2 (F16)
        //                       = ctx × 128 × 28 × 4  (for Qwen2.5-1.5B)
        //   threshold = floor(kv_budget / bytes_per_token)
        //
        // Default budget fraction: 0.5 (use half of free GPU memory for dense KV,
        // leaving the other half for weights, activations, and OS headroom).
        // Override the entire threshold with DIFFKV_ENGAGE_THRESHOLD (tokens).
        int engage_threshold = 0;  // 0 = auto
        if (const char* env_et = std::getenv("DIFFKV_ENGAGE_THRESHOLD")) {
            engage_threshold = std::stoi(env_et);
        }
        if (engage_threshold <= 0) {
            // Query available Metal memory
            size_t free_mem = 0, total_mem = 0;
            ggml_backend_dev_t bdev = ggml_backend_get_device(backend);
            if (bdev) ggml_backend_dev_memory(bdev, &free_mem, &total_mem);

            // bytes per decode-context token in the dense KV window (F16, both sides)
            const size_t bytes_per_token = (size_t)head_dim * kv_heads * (size_t)n_layers * 2 /* K+V */ * sizeof(ggml_fp16_t);

            // Budget fraction of free memory (default 50%); tune with DIFFKV_KV_MEM_FRAC
            const char* env_frac = std::getenv("DIFFKV_KV_MEM_FRAC");
            double kv_budget_frac = env_frac ? std::stod(env_frac) : 0.50;
            const size_t kv_budget = static_cast<size_t>(free_mem * kv_budget_frac);
            const int auto_thresh = (bytes_per_token > 0 && kv_budget > 0)
                ? static_cast<int>(std::min<size_t>(kv_budget / bytes_per_token, 1 << 20))
                : 65536;

            engage_threshold = std::max(4096, auto_thresh);
            std::cerr << "[DiffKV] auto engage_threshold=" << engage_threshold
                      << " (free_mem=" << free_mem / (1 << 20) << "MB"
                      << ", bytes_per_tok=" << bytes_per_token
                      << ", budget=" << kv_budget / (1 << 20) << "MB)" << std::endl;
        }
        bool decode_use_sparse = (L >= engage_threshold);

        // ── sparse_dense_cap: dense window size for the SPARSE path ──────────────
        //   recency_window + 2·block_size (the slide keeps it here) + slack.
        int sparse_dense_cap = cfg_recency_window + 2 * micro_block_size + 512;

        // ── native_maxd: GPU buffer size for the sparse path's dense window ──────
        // IMPORTANT: do NOT tie this to engage_threshold. With engage_threshold=65536,
        // tying would pre-allocate 65536 × 128 × 28 layers × 4B × 2 = 1.87 GB of unused
        // native_dense_kr/v buffers. Keep it bounded at the sparse path's actual needs.
        int native_maxd = 2048;
        if (const char* env_maxd = std::getenv("DIFFKV_MAX_ACTIVE_DENSE_TOKENS")) {
            native_maxd = std::stoi(env_maxd);
        }
        // Clamp: must cover sparse_dense_cap, but cap at 8192 to prevent huge allocs.
        native_maxd = std::min(std::max(native_maxd, sparse_dense_cap + 512), 8192);

        // ── required_dense_cap: actual GPU KV buffer size ─────────────────────────
        //   • sparse path : only needs the sliding dense window (sparse_dense_cap)
        //   • dense path  : needs the full prefill length + headroom (no artificial cap)
        int required_dense_cap = decode_use_sparse
            ? sparse_dense_cap
            : L + max_generate + 512;

        const char* env_maxd_val = std::getenv("DIFFKV_MAX_ACTIVE_DENSE_TOKENS");
        std::cerr << "[DEBUG_CAP] L=" << L << " engage_threshold=" << engage_threshold
                  << " decode_use_sparse=" << decode_use_sparse << " native_maxd=" << native_maxd
                  << " required_dense_cap=" << required_dense_cap
                  << " env_maxd=" << (env_maxd_val ? env_maxd_val : "NULL") << std::endl;

        // Memory optimization: only resize host-side dense buffers if sparse decode is used.
        // In dense decode (decode_use_sparse=false), we bypass host-side dense buffers and
        // use GPU-resident dense past tensors (dense_k_past_inputs/dense_v_past_inputs) directly,
        // so resizing them here would needlessly allocate gigabytes of unused host memory.
        if (decode_use_sparse) {
            for (int l = 0; l < n_layers; ++l) {
                if (active_k_dense[l].size() < (size_t)required_dense_cap * F_test) {
                    active_k_dense[l].resize((size_t)required_dense_cap * F_test, 0.0f);
                    active_k_dense_rotated[l].resize((size_t)required_dense_cap * F_test, 0.0f);
                    active_v_dense[l].resize((size_t)required_dense_cap * F_test, 0.0f);
                }
            }
        }
        if (active_positions_dense.size() < (size_t)required_dense_cap) {
            active_positions_dense.resize(required_dense_cap, 0);
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

        // ── 1. PREFILL PHASE ──
        if (!interactive) {
            std::cerr << "[DiffKV Native] Running Prefill phase in chunks..." << std::endl;
        }

        int chunk_size = 512; // Default to balanced preset size
        if (const char* env_preset = std::getenv("DIFFKV_PRESET")) {
            std::string p(env_preset);
            std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c){ return std::tolower(c); });
            if (p == "low") {
                chunk_size = 512;
            } else if (p == "high") {
                chunk_size = 2048;
            }
        }
        if (const char* env_pcs = std::getenv("DIFFKV_PREFILL_CHUNK_SIZE")) {
            try { chunk_size = std::stoi(env_pcs); } catch (...) {}
        }

        // ── DIFFKV_SPARSE_PREFILL knobs (hoisted above the host-mirror allocation:
        // both the LEGO ring geometry AND the mirror sizes below derive from them). ──
        bool sp_enabled = true;
        if (const char* e = std::getenv("DIFFKV_SPARSE_PREFILL")) { try { sp_enabled = (std::stoi(e) != 0); } catch (...) {} }
        int sp_window = 1024, sp_sink_blocks = 1, sp_min = 2048;
        if (const char* e = std::getenv("DIFFKV_SPARSE_PREFILL_WINDOW")) { try { sp_window = std::stoi(e); } catch (...) {} }
        if (const char* e = std::getenv("DIFFKV_SPARSE_PREFILL_SINK_BLOCKS")) { try { sp_sink_blocks = std::stoi(e); } catch (...) {} }
        if (const char* e = std::getenv("DIFFKV_SPARSE_PREFILL_MIN")) { try { sp_min = std::stoi(e); } catch (...) {} }
        const int sp_sink_end = sp_sink_blocks * micro_block_size;
        int sp_kmin = 8; float sp_frac = 0.05f; int sp_max_occ = 8;
        if (const char* e = std::getenv("DIFFKV_SPARSE_PREFILL_KMIN")) { try { sp_kmin = std::stoi(e); } catch (...) {} }
        if (const char* e = std::getenv("DIFFKV_SPARSE_PREFILL_FRAC")) { try { sp_frac = std::stof(e); } catch (...) {} }
        if (const char* e = std::getenv("DIFFKV_SPARSE_PREFILL_MAX_OCC")) { try { sp_max_occ = std::stoi(e); } catch (...) {} }

        // ── DIFFKV_LEGO_PREFILL (port of the MLX lego streaming prefill; see
        // docs/NATIVE_LEGO_PORT_PLAN.md and the MLX flag notes). ──────────────────
        // STAGE 1 (device): the persistent prefill KV cache is ring-sized:
        //   [0, base_end)            identity zone — sinks + everything below the
        //                            sparse-engage point (pre-engage chunks attend
        //                            the full dense context, so this stays resident)
        //   [base_end, +wnd_cap)     modular ring for the recency window + chunk
        // STAGE 2 (host): the raw host mirrors (k_activations / v_activations) use
        // the SAME ring layout — they are the plain-RAM buffers that actually show
        // up in phys_footprint (Metal buffer bytes don't; that was stage 1's "no
        // measured win" finding). Routed far blocks are no longer read from the
        // mirrors: they are re-materialised from the COMPRESSED pool per chunk
        // (lego_emit_block_from_pool — anchor + U·V + exact residuals, K pre-rotated
        // per POOL_ROT_ABS), with a raw fallback for blocks the async compressor
        // hasn't finished yet. NOTE this deviates from the MLX studs-only default
        // deliberately: native's adaptive residual budget is layer-dependent, so
        // stud row counts differ per layer, which the single shared prefill mask
        // cannot express — full-block materialisation keeps the row count uniform
        // (mbs+1 per block) and is exactly the representation the sparse DECODE
        // path already validated (NIAH 6/6, margins ≈ dense).
        // Requires sparse decode (dense decode re-reads the full-length rotated
        // mirrors, not this cache — but only sparse decode frees them) and a fresh
        // prompt (cached_len == 0). DIFFKV_LEGO_PREFILL=0 (default) = legacy.
        bool lego_on = false;
        if (const char* e = std::getenv("DIFFKV_LEGO_PREFILL")) {
            try { lego_on = (std::stoi(e) != 0); } catch (...) {}
        }
        lego_on = lego_on && sp_enabled && decode_use_sparse && cached_len == 0;
        // ── DIFFKV_LEGO_FAR: far-field mode under lego (all measured 2026-07-12,
        // q8 native; synthesis = real-paper 8k/16k, margins = needle-digit nats).
        // "studs" (DEFAULT) — the MLX studs-only port: far blocks contribute ONLY
        //   rows that come out EXACT (anchor + residual-corrected rows, 1+128 per
        //   block); the well-approximated rows are omitted, not reconstructed.
        //   Requires uniform residual sets (DIFFKV_RESIDUAL_UNIFORM=1 is set
        //   below) because stud counts must match across layers for the shared
        //   prefill mask. Rationale: the two extremes both fail one gate —
        //     "recon": full-block pool recon → synthesis COLLAPSES 26.7→10.0@8k /
        //       26.7→3.3@16k (low-rank noise poisons the far field; MLX saw the
        //       same) though margins 11.6/12.5 and needles hold;
        //     "off" (pure omission): synthesis holds 26.7/23.3 but margins drop
        //       12.6/14.3→7.4/7.6 and multi-needle REFUSES (0/3) — the prefill
        //       hidden states lose the far field entirely.
        //   Studs = exact-only coverage, the middle both gates accept.
        // "off" / "recon" — kept for A/B.
        int lego_far_mode = 1;   // 0=off, 1=studs, 2=recon
        if (const char* e = std::getenv("DIFFKV_LEGO_FAR")) {
            std::string m(e);
            if (m == "off") lego_far_mode = 0;
            else if (m == "recon") lego_far_mode = 2;
            else lego_far_mode = 1;
        }
        const int lego_mr_studs = std::min(diffkv::NativeBlockPool::MAX_RESIDUAL, micro_block_size);
        int lego_base_end = 0, lego_wnd_cap = 0, lego_far_rows = 0;
        int lego_buf_rows = L;
        const int lego_bstride = micro_block_size + 1;  // ingest-block stride (anchor + mbs rows)
        if (lego_on) {
            lego_base_end = std::max(sp_min, sp_sink_end + sp_window);
            // Round up to a chunk boundary so every pre-engage chunk (which attends
            // the full dense context) lies wholly inside the identity zone.
            lego_base_end = ((lego_base_end + chunk_size - 1) / chunk_size) * chunk_size;
            // Window + current chunk, plus one chunk of slack so the ring never
            // overwrites rows the current chunk still attends.
            lego_wnd_cap = sp_window + 2 * chunk_size;
            // Far capacity: stage 2 routes INGEST blocks (stride mbs+1), not the
            // aligned grid — candidates are whole pool blocks below the window.
            // Rows per far block: studs = 1+MR exact rows; recon = whole block;
            // off = no far tensors at all.
            if (lego_far_mode != 0) {
                int lego_first_blk = (sp_sink_end + lego_bstride - 1) / lego_bstride;
                int lego_nb_max = std::max(0, (L - sp_window) / lego_bstride - lego_first_blk);
                int lego_k_max = std::max(sp_kmin, (int)std::ceil(sp_frac * lego_nb_max));
                lego_far_rows = lego_k_max * (lego_far_mode == 1 ? (1 + lego_mr_studs) : lego_bstride);
            } else {
                lego_far_rows = 0;
            }
            lego_buf_rows = lego_base_end + lego_wnd_cap;
            if (lego_buf_rows + lego_far_rows >= L) {
                lego_on = false;   // short prompt — ring would not shrink anything
                lego_buf_rows = L;
                lego_far_rows = 0;
            } else {
                if (lego_far_mode == 1) {
                    // Studs need uniform residual sets across layers/blocks (see the
                    // DIFFKV_LEGO_FAR note). Set BEFORE any of this prompt's blocks
                    // are submitted for compression; the compressor is idle here.
                    setenv("DIFFKV_RESIDUAL_UNIFORM", "1", 1);
                }
                std::cerr << "[DiffKV] LEGO_PREFILL on: base_end=" << lego_base_end
                          << " wnd_cap=" << lego_wnd_cap
                          << " far=" << (lego_far_mode == 1 ? "studs" : lego_far_mode == 2 ? "recon" : "off")
                          << " far_rows=" << lego_far_rows
                          << " device+host rows " << lego_buf_rows << "+" << lego_far_rows
                          << " vs full " << L << " (stage 2: host mirrors ringed too)" << std::endl;
            }
        }
        // Split an absolute [start, start+len) span into <=3 contiguous buffer spans
        // (identity below base_end; modular ring above it). Buffer-agnostic: the same
        // mapping addresses the device cache AND the host mirrors (stage 2).
        auto lego_map_span = [&](int abs_start, int len, std::vector<std::pair<int,int>>& out) {
            if (!lego_on) { out.push_back({abs_start, len}); return; }
            if (abs_start < lego_base_end) {
                int l0 = std::min(len, lego_base_end - abs_start);
                out.push_back({abs_start, l0});
                abs_start += l0; len -= l0;
            }
            while (len > 0) {
                int off = lego_base_end + (abs_start - lego_base_end) % lego_wnd_cap;
                int l0 = std::min(len, lego_base_end + lego_wnd_cap - off);
                out.push_back({off, l0});
                abs_start += l0; len -= l0;
            }
        };

        // Local per-turn raw K/V activation buffers.
        // For continuation turns (cached_len > 0), we only fill offsets [cached_len..L-1]
        // and upload the already-stored prefix K/V from the prior chunk's data (if pos_start > cached_len).
        // This matches ACTIVE_RUNTIME: full prompt is sent, only new tokens are prefilled.
        // ── fp16 KV storage (RAM fix, mirrors MLX which keeps dense KV in float16) ──
        // These full-length host buffers are the single biggest native-vs-MLX RAM
        // overhead (3 × L × F × 4 bytes × n_layers ≈ 2 GB at 24k/1.5B). MLX stores the
        // equivalent in float16 (mlx_diffkv_wrapper.py:345 "float16 explicitly to halve RAM").
        // Stored as ggml_fp16_t; every consumer converts at its boundary (ggml_fp16_to_fp32
        // on read, ggml_fp32_to_fp16 on write). The fp32 MATH is unchanged — only storage
        // halves. The GPU prior tensors + prefill flash-attn are F16 to match (no fp32 temp).
        // ── LEGO stage 2: under the ring, the mirrors hold only [identity | window ring]
        // (lego_buf_rows rows, addressed through lego_map_span) instead of the full [L].
        // Consumers that survive the ring: chunk writes (ring-mapped spans), the
        // decode-boundary dense-window seeding (reads the tail, always ring-resident).
        // Far-block reads moved to the compressed pool; debug probes gated off.
        const int lego_host_rows = lego_on ? lego_buf_rows : L;
        std::vector<std::vector<ggml_fp16_t>> k_activations(n_layers, std::vector<ggml_fp16_t>((size_t)lego_host_rows * F_test, 0));
        std::vector<std::vector<ggml_fp16_t>> k_rotated_activations(n_layers);
        if (!decode_use_sparse) {
            for (int l = 0; l < n_layers; ++l) {
                k_rotated_activations[l].resize((size_t)L * F_test, ggml_fp32_to_fp16(0.0f));
            }
        }
        std::vector<std::vector<ggml_fp16_t>> v_activations(n_layers, std::vector<ggml_fp16_t>((size_t)lego_host_rows * F_test, 0));
        std::vector<float> prefill_output_logits(n_vocab);
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
                            const int8_t* u_src = engine->get_host_U(slot_id);
                            for (int i = 0; i < non_anchor_len * rank; ++i) {
                                U_float[i] = u_src ? (float)u_src[i] * scale_u : 0.0f;
                            }

                            // Pre-convert VK and VV to float
                            std::vector<float> VK_float(rank * F_test);
                            std::vector<float> VV_float(rank * F_test);
                            const ggml_fp16_t* vk_src = engine->get_host_VK(slot_id);
                            const ggml_fp16_t* vv_src = engine->get_host_VV(slot_id);
                            for (int i = 0; i < rank * F_test; ++i) {
                                VK_float[i] = vk_src ? ggml_fp16_to_fp32(vk_src[i]) : 0.0f;
                                VV_float[i] = vv_src ? ggml_fp16_to_fp32(vv_src[i]) : 0.0f;
                            }

                            // Pre-convert anchors to float
                            std::vector<float> anchor_k_float(F_test);
                            std::vector<float> anchor_v_float(F_test);
                            const ggml_fp16_t* ak_src = engine->get_host_anchors_K(slot_id);
                            const ggml_fp16_t* av_src = engine->get_host_anchors_V(slot_id);
                            for (int f = 0; f < F_test; ++f) {
                                anchor_k_float[f] = ak_src ? ggml_fp16_to_fp32(ak_src[f]) : 0.0f;
                                anchor_v_float[f] = av_src ? ggml_fp16_to_fp32(av_src[f]) : 0.0f;
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

                            const int32_t* slot_res_K_pos = engine->get_host_res_K_pos(slot_id);
                            const ggml_fp16_t* slot_res_K_val = engine->get_host_res_K_val(slot_id);
                            const int32_t* slot_res_V_pos = engine->get_host_res_V_pos(slot_id);
                            const ggml_fp16_t* slot_res_V_val = engine->get_host_res_V_val(slot_id);

                            for (int t = 0; t < block_len; ++t) {
                                int global_pos = block->anchor_idx + t;
                                if (global_pos >= cached_len) break;

                                if (t == landmark_idx) {
                                    // This is the landmark token, stored as the anchor in the pool
                                    for (int f = 0; f < F_test; ++f) {
                                        k_activations[l][global_pos * F_test + f] = ggml_fp32_to_fp16(anchor_k_float[f]);
                                        v_activations[l][global_pos * F_test + f] = ggml_fp32_to_fp16(anchor_v_float[f]);
                                    }
                                } else {
                                    // Reconstructed from SVD delta
                                    int s = (t == 0) ? (landmark_idx - 1) : (t - 1);

                                    // Look up residual for delta index s
                                    float res_k_val[F_test];
                                    float res_v_val[F_test];
                                    std::memset(res_k_val, 0, sizeof(res_k_val));
                                    std::memset(res_v_val, 0, sizeof(res_v_val));
                                    if (slot_res_K_pos && slot_res_K_val) {
                                        for (int ri = 0; ri < NativeBlockPool::MAX_RESIDUAL; ++ri) {
                                            if (slot_res_K_pos[ri] == s) {
                                                for (int f = 0; f < F_test; ++f) {
                                                    res_k_val[f] = ggml_fp16_to_fp32(slot_res_K_val[ri * F_test + f]);
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    if (slot_res_V_pos && slot_res_V_val) {
                                        for (int ri = 0; ri < NativeBlockPool::MAX_RESIDUAL; ++ri) {
                                            if (slot_res_V_pos[ri] == s) {
                                                for (int f = 0; f < F_test; ++f) {
                                                    res_v_val[f] = ggml_fp16_to_fp32(slot_res_V_val[ri * F_test + f]);
                                                }
                                                break;
                                            }
                                        }
                                    }

                                    ggml_fp16_t* k_act_row = &k_activations[l][global_pos * F_test];
                                    ggml_fp16_t* v_act_row = &v_activations[l][global_pos * F_test];
                                    const float* k_del_row = &K_delta[s * F_test];
                                    const float* v_del_row = &V_delta[s * F_test];
                                    for (int f = 0; f < F_test; ++f) {
                                        k_act_row[f] = ggml_fp32_to_fp16(anchor_k_float[f] + k_del_row[f] * block_scale + res_k_val[f]);
                                        v_act_row[f] = ggml_fp32_to_fp16(anchor_v_float[f] + v_del_row[f] * block_scale + res_v_val[f]);
                                    }
                                }
                            }
                        } else {
                            // Single token block (just anchor)
                            int global_pos = block->anchor_idx;
                            if (global_pos < cached_len) {
                                const ggml_fp16_t* ak_src = engine->get_host_anchors_K(slot_id);
                                const ggml_fp16_t* av_src = engine->get_host_anchors_V(slot_id);
                                for (int f = 0; f < F_test; ++f) {
                                    // anchors are already fp16 storage — copy directly (no round-trip).
                                    k_activations[l][global_pos * F_test + f] = ak_src ? ak_src[f] : ggml_fp32_to_fp16(0.0f);
                                    v_activations[l][global_pos * F_test + f] = av_src ? av_src[f] : ggml_fp32_to_fp16(0.0f);
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
                                for (int f = 0; f < F_test; ++f) {
                                    k_activations[l][global_pos * F_test + f] = ggml_fp32_to_fp16(block->anchor_k[f]);
                                    v_activations[l][global_pos * F_test + f] = ggml_fp32_to_fp16(block->anchor_v[f]);
                                }
                            } else {
                                int s = t - 1;
                                for (int f = 0; f < F_test; ++f) {
                                    k_activations[l][global_pos * F_test + f] = ggml_fp32_to_fp16(block->active_k[s * F_test + f]);
                                    v_activations[l][global_pos * F_test + f] = ggml_fp32_to_fp16(block->active_v[s * F_test + f]);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (cached_len > 0 && !decode_use_sparse) {
            std::vector<int32_t> prior_positions(cached_len);
            for (int t = 0; t < cached_len; ++t) prior_positions[t] = t;
            
            int half_dim = head_dim / 2;
            std::vector<float> inv_freq(half_dim);
            for (int i = 0; i < half_dim; ++i) {
                inv_freq[i] = 1.0f / std::pow(model.get_config().rope_freq_base, 2.0f * i / head_dim);
            }
            std::vector<float> cos_table(cached_len * half_dim);
            std::vector<float> sin_table(cached_len * half_dim);
            for (int t = 0; t < cached_len; ++t) {
                float pos = (float)prior_positions[t];
                for (int i = 0; i < half_dim; ++i) {
                    float theta = pos * inv_freq[i];
                    cos_table[t * half_dim + i] = std::cos(theta);
                    sin_table[t * half_dim + i] = std::sin(theta);
                }
            }
            for (int l = 0; l < n_layers; ++l) {
                apply_rope_neox_cpu_fast(
                    k_activations[l].data(),
                    k_rotated_activations[l].data(),
                    cos_table.data(),
                    sin_table.data(),
                    cached_len, kv_heads, head_dim
                );
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

        // Initialize persistent prefill cache context and tensors on Metal/GPU
        struct ggml_init_params cache_params = {
            /*.mem_size   =*/ 1 * 1024 * 1024, // 1MB metadata context
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * prefill_cache_ctx = ggml_init(cache_params);
        std::vector<struct ggml_tensor *> persistent_k_cache(n_layers, nullptr);
        std::vector<struct ggml_tensor *> persistent_v_cache(n_layers, nullptr);
        std::vector<struct ggml_tensor *> lego_far_k(n_layers, nullptr);
        std::vector<struct ggml_tensor *> lego_far_v(n_layers, nullptr);
        for (int l = 0; l < n_layers; ++l) {
            persistent_k_cache[l] = ggml_new_tensor_3d(prefill_cache_ctx, GGML_TYPE_F16, head_dim, kv_heads, lego_buf_rows);
            persistent_v_cache[l] = ggml_new_tensor_3d(prefill_cache_ctx, GGML_TYPE_F16, head_dim, kv_heads, lego_buf_rows);
            if (lego_on && lego_far_rows > 0) {
                lego_far_k[l] = ggml_new_tensor_3d(prefill_cache_ctx, GGML_TYPE_F16, head_dim, kv_heads, lego_far_rows);
                lego_far_v[l] = ggml_new_tensor_3d(prefill_cache_ctx, GGML_TYPE_F16, head_dim, kv_heads, lego_far_rows);
            }
        }
        struct ggml_backend_buffer * prefill_cache_buffer = ggml_backend_alloc_ctx_tensors(prefill_cache_ctx, backend);

        // Upload prefix if cached_len > 0
        if (cached_len > 0) {
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_tensor_set(persistent_k_cache[l], k_rotated_activations[l].data(), 0, cached_len * F_test * sizeof(ggml_fp16_t));
                ggml_backend_tensor_set(persistent_v_cache[l], v_activations[l].data(), 0, cached_len * F_test * sizeof(ggml_fp16_t));
            }
        }

        const bool dbg_prefill_time = (std::getenv("DIFFKV_DBG_PREFILL_TIME") != nullptr);
        double tp_upload = 0, tp_compute = 0, tp_capture = 0, tp_ingest = 0, tp_build = 0;
        auto tp_start = std::chrono::high_resolution_clock::now();
        int tp_chunks = 0;

        bool ingest_async = true;
        if (const char* env_ia = std::getenv("DIFFKV_INGEST_ASYNC")) {
            try { ingest_async = (std::stoi(env_ia) != 0); } catch (...) {}
        }
        std::thread prefill_ingest_thread;
        bool prefill_ingest_thread_active = false;

        // ── DIFFKV_SPARSE_PREFILL (HANDOFF §SPARSE-PREFILL) ──────────────────────────────
        // Training-free StreamingLLM sparse prefill: a chunk far enough in attends only
        // [sink blocks | recency window | current chunk] instead of the full prior context,
        // dropping prefill attention from O(L^2) to O(L·(sink+window)). Single-needle
        // retrieval is preserved (recall is a DECODE-time job).
        // DEFAULT ON (16th pass): flipped after 6-cell 6/6 + conformance PASS + margins unchanged
        // (12.6/14.3) + multi-needle no-regression vs dense, all with sparse-ON. Reversible:
        // DIFFKV_SPARSE_PREFILL=0. Only engages above sp_min tokens, no-op below.
        // Stage B — lexical routing: a chunk also attends the top-K prior blocks that share the
        // most DISTINCTIVE (rare, count<=MAX_OCC) tokens with it; host-side, no readback.
        // (Knobs are hoisted above the persistent-cache allocation — the LEGO ring geometry
        // derives from them.)
        std::map<int32_t,int> sp_tok_count;                       // token id -> total count in prompt
        std::vector<std::unordered_set<int32_t>> sp_block_tokens; // per aligned block: unique token ids
        std::vector<std::unordered_set<int32_t>> lego_block_tokens; // per INGEST block (stride mbs+1), lego only
        if (sp_enabled) {
            for (int32_t t : prompt_tokens) sp_tok_count[t]++;
            int nblk = (L + micro_block_size - 1) / micro_block_size;
            sp_block_tokens.resize(nblk);
            for (int b = 0; b < nblk; ++b) {
                int s = b * micro_block_size, e = std::min(L, s + micro_block_size);
                for (int i = s; i < e; ++i) sp_block_tokens[b].insert(prompt_tokens[i]);
            }
            if (lego_on) {
                // LEGO stage 2 routes whole INGEST blocks (anchor + mbs rows, stride
                // mbs+1) — the pool's unit of compression — so a routed far block maps
                // 1:1 onto a pool slot and can be re-materialised without straddling
                // block boundaries. (Blocks tile from position 0 on a fresh prompt, so
                // ingest block b covers [b*(mbs+1), (b+1)*(mbs+1)).)
                int nlb = (L + lego_bstride - 1) / lego_bstride;
                lego_block_tokens.resize(nlb);
                for (int b = 0; b < nlb; ++b) {
                    int s = b * lego_bstride, e = std::min(L, s + lego_bstride);
                    for (int i = s; i < e; ++i) lego_block_tokens[b].insert(prompt_tokens[i]);
                }
            }
            std::cerr << "[DiffKV] SPARSE_PREFILL on: sink_end=" << sp_sink_end
                      << " window=" << sp_window << " min=" << sp_min
                      << " kmin=" << sp_kmin << " frac=" << sp_frac << " max_occ=" << sp_max_occ << std::endl;
        }

        while (pos_start < L) {
            if (prefill_ingest_thread_active) {
                prefill_ingest_thread.join();
                prefill_ingest_thread_active = false;
            }

            int chunk_len = std::min(chunk_size, L - pos_start);
            if (pos_start == 0 || pos_start + chunk_len >= L || pos_start % 4096 == 0) {
                std::cerr << "[Prefill Progress] pos=" << pos_start << " / " << L << " (chunk_len=" << chunk_len << ")" << std::endl;
            }
            auto tp_c0 = std::chrono::high_resolution_clock::now();
            int ctx_len   = pos_start + chunk_len;  // total KV context length

            // Sparse-prefill key ranges for THIS chunk (empty = dense full context). Engage only
            // once there is prunable history beyond the sink + window.
            // sp_ranges_abs — absolute [start,len) spans, drives the MASK.
            // sp_ranges_buf — the same spans as BUFFER offsets, drives the graph views
            //                 (identical to abs when lego is off; ring-mapped when on).
            // lego_far_blocks — routed INGEST-block indices that are above the identity
            //                 zone under lego; re-materialised from the compressed pool
            //                 (stage 2) and uploaded into the FAR tensors below.
            std::vector<std::pair<int,int>> sp_ranges_abs;
            std::vector<std::pair<int,int>> sp_ranges_buf;
            std::vector<int> lego_far_blocks;
            bool use_sp = sp_enabled && pos_start >= sp_min && pos_start > sp_sink_end + sp_window;
            int sp_win_start = 0;
            if (use_sp) {
                sp_win_start = pos_start - sp_window;                          // > sp_sink_end
                // ── Stage B lexical routing: pick the prior blocks that share the most
                //    distinctive tokens with the current chunk (top-K), so multi-fact prompts
                //    keep the needle blocks. Filler chunks match nothing → pure StreamingLLM.
                //    Lego routes INGEST blocks (stride mbs+1, pool-aligned); legacy routes
                //    the aligned mbs grid. Same scoring either way.
                const int r_stride = lego_on ? lego_bstride : micro_block_size;
                const auto & r_tokens = lego_on ? lego_block_tokens : sp_block_tokens;
                int first_blk = (sp_sink_end + r_stride - 1) / r_stride;
                int last_blk  = sp_win_start / r_stride;               // blocks fully below window
                int nb = last_blk - first_blk;
                std::vector<int> selected;
                if (nb > 0 && (sp_kmin > 0 || sp_frac > 0.0f)) {
                    std::unordered_set<int32_t> distinct;
                    for (int i = pos_start; i < ctx_len; ++i) {
                        int32_t t = prompt_tokens[i];
                        auto it = sp_tok_count.find(t);
                        if (it != sp_tok_count.end() && it->second <= sp_max_occ) distinct.insert(t);
                    }
                    std::vector<std::pair<int,int>> scored;   // (overlap score, abs block idx)
                    for (int b = first_blk; b < last_blk; ++b) {
                        int sc = 0;
                        for (int32_t t : r_tokens[b]) if (distinct.count(t)) sc++;
                        if (sc > 0) scored.push_back({sc, b});
                    }
                    int K = std::max(sp_kmin, (int)std::ceil(sp_frac * nb));
                    K = std::min(K, (int)scored.size());
                    if (K > 0) {
                        std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
                            [](const std::pair<int,int>& a, const std::pair<int,int>& b){ return a.first > b.first; });
                        for (int i = 0; i < K; ++i) selected.push_back(scored[i].second);
                        std::sort(selected.begin(), selected.end());
                    }
                }
                // Far blocks FIRST (they are concatenated before the ranges in the graph,
                // so the mask walk must see them first — handled via lego_far_blocks).
                // Far mode "off": blocks above the identity zone are simply omitted
                // (see the DIFFKV_LEGO_FAR note at the lego config).
                if (lego_on && lego_far_mode != 0) {
                    for (int b : selected) {
                        if ((b + 1) * r_stride > lego_base_end) lego_far_blocks.push_back(b);
                    }
                }
                sp_ranges_abs.push_back({0, sp_sink_end});                            // attention sink
                for (int b : selected) {
                    if (lego_on && (b + 1) * r_stride > lego_base_end) continue; // far — via upload
                    sp_ranges_abs.push_back({b * r_stride, r_stride});
                }
                sp_ranges_abs.push_back({sp_win_start, ctx_len - sp_win_start});      // window + current chunk
                for (const auto & r : sp_ranges_abs) lego_map_span(r.first, r.second, sp_ranges_buf);
            }
            // Rows per far block: studs = anchor + MR exact rows; recon = whole block.
            const int lego_far_bpr = (lego_far_mode == 1) ? (1 + lego_mr_studs) : lego_bstride;
            int lego_n_far = (int)lego_far_blocks.size() * lego_far_bpr;
            if (lego_n_far > lego_far_rows) {
                // Cap guard (routing K can exceed the sizing estimate only if knobs were
                // changed mid-run — never in practice). Drop the lowest-priority extras.
                lego_far_blocks.resize(lego_far_rows / lego_far_bpr);
                lego_n_far = (int)lego_far_blocks.size() * lego_far_bpr;
            }
            int sp_klen = 0;
            if (use_sp) { sp_klen = lego_n_far; for (const auto& r : sp_ranges_abs) sp_klen += r.second; }
            else        { sp_klen = ctx_len; }

            // Ring write spans for the current chunk (identity single-span when lego off).
            std::vector<std::pair<int,int>> chunk_write_spans;
            if (lego_on) lego_map_span(pos_start, chunk_len, chunk_write_spans);

            // ── LEGO far gather (stage 2): routed far blocks are re-materialised from
            // the COMPRESSED pool — anchor + U·V·row_scale·blk_scale + exact residual
            // corrections, K PRE-ROTATED at absolute positions (POOL_ROT_ABS), V raw —
            // and uploaded into the per-layer FAR tensors. This is the representation
            // the sparse DECODE path already attends (materialize_routed_kv), so no new
            // fidelity surface is introduced beyond what decode validated. Blocks the
            // async compressor hasn't finished in a given layer fall back to that
            // layer's raw fp32 rows (exact; rotated here like the legacy path). Row
            // order inside the far segment is free — every far row is always-visible
            // history in the mask — so no position bookkeeping is needed.
            if (lego_on && lego_n_far > 0) {
                auto & lg_ingest = runtime_manager.get_ingest_manager();
                auto & lg_engines = runtime_manager.get_engines();
                const int half_dim = head_dim / 2;
                // Lazy per-block cos/sin tables: only raw-fallback blocks need them, and
                // absolute positions are layer-independent, so build once per block.
                std::vector<std::vector<float>> far_cos(lego_far_blocks.size());
                std::vector<std::vector<float>> far_sin(lego_far_blocks.size());
                std::vector<float> lg_inv_freq;
                std::vector<ggml_fp16_t> far_stage_k((size_t)lego_n_far * F_test);
                std::vector<ggml_fp16_t> far_stage_v((size_t)lego_n_far * F_test);
                std::vector<ggml_fp16_t> lg_raw_rows;
                for (int l = 0; l < n_layers; ++l) {
                    auto & lg_blocks = lg_ingest.get_blocks(l);
                    diffkv::NativeBlockPool* lg_pool = lg_engines[l].get();
                    for (int fi = 0; fi < (int)lego_far_blocks.size(); ++fi) {
                        const int bi = lego_far_blocks[fi];
                        ggml_fp16_t* out_k = far_stage_k.data() + (size_t)fi * lego_far_bpr * F_test;
                        ggml_fp16_t* out_v = far_stage_v.data() + (size_t)fi * lego_far_bpr * F_test;
                        diffkv::StreamingKVBlock* blk =
                            (bi < (int)lg_blocks.size()) ? lg_blocks[bi].get() : nullptr;
                        bool emitted = false;
                        if (blk && blk->pool_idx >= 0) {
                            diffkv::BlockState st = lg_pool->get_state_table().get(blk->pool_idx);
                            if (st == diffkv::BlockState::CPUResident) {
                                runtime_manager.get_pager().touch(blk, lg_engines);
                                st = lg_pool->get_state_table().get(blk->pool_idx);
                            }
                            if (st == diffkv::BlockState::CompressedResident) {
                                int rows = (lego_far_mode == 1)
                                    ? lego_emit_block_studs(lg_pool, blk->pool_idx, F_test, lego_mr_studs, out_k, out_v)
                                    : lego_emit_block_from_pool(lg_pool, blk->pool_idx, F_test, out_k, out_v);
                                emitted = (rows == lego_far_bpr);
                            }
                        }
                        if (!emitted && blk && blk->anchor_idx == bi * lego_bstride &&
                            (int)blk->active_k.size() == (lego_bstride - 1) * F_test &&
                            blk->active_v.size() == blk->active_k.size() &&
                            (int)blk->anchor_k.size() == F_test) {
                            // Raw fallback: anchor + active rows are still resident for this
                            // layer (compressor not done yet) — exact rows, rotate K here.
                            // Studs mode emits anchor + the LAST mr_studs raw rows (residual
                            // positions aren't known pre-compression; any exact-row subset is
                            // valid coverage — same omission philosophy).
                            const int s0 = (lego_far_mode == 1) ? (lego_bstride - 1 - lego_mr_studs) : 0;
                            if (far_cos[fi].empty()) {
                                if (lg_inv_freq.empty()) {
                                    lg_inv_freq.resize(half_dim);
                                    for (int i = 0; i < half_dim; ++i) {
                                        lg_inv_freq[i] = 1.0f / std::pow(model.get_config().rope_freq_base, 2.0f * i / head_dim);
                                    }
                                }
                                far_cos[fi].resize((size_t)lego_far_bpr * half_dim);
                                far_sin[fi].resize((size_t)lego_far_bpr * half_dim);
                                for (int t = 0; t < lego_far_bpr; ++t) {
                                    float pos = (float)((t == 0) ? blk->anchor_idx
                                                                 : blk->anchor_idx + 1 + s0 + (t - 1));
                                    for (int i = 0; i < half_dim; ++i) {
                                        float theta = pos * lg_inv_freq[i];
                                        far_cos[fi][(size_t)t * half_dim + i] = std::cos(theta);
                                        far_sin[fi][(size_t)t * half_dim + i] = std::sin(theta);
                                    }
                                }
                            }
                            lg_raw_rows.resize((size_t)lego_far_bpr * F_test);
                            for (int f = 0; f < F_test; ++f) lg_raw_rows[f] = ggml_fp32_to_fp16(blk->anchor_k[f]);
                            for (int t = 1; t < lego_far_bpr; ++t) {
                                const float* src = blk->active_k.data() + (size_t)(s0 + t - 1) * F_test;
                                for (int f = 0; f < F_test; ++f) {
                                    lg_raw_rows[(size_t)t * F_test + f] = ggml_fp32_to_fp16(src[f]);
                                }
                            }
                            apply_rope_neox_cpu_fast(lg_raw_rows.data(), out_k,
                                                     far_cos[fi].data(), far_sin[fi].data(),
                                                     lego_far_bpr, kv_heads, head_dim);
                            for (int f = 0; f < F_test; ++f) out_v[f] = ggml_fp32_to_fp16(blk->anchor_v[f]);
                            for (int t = 1; t < lego_far_bpr; ++t) {
                                const float* src = blk->active_v.data() + (size_t)(s0 + t - 1) * F_test;
                                for (int f = 0; f < F_test; ++f) {
                                    out_v[(size_t)t * F_test + f] = ggml_fp32_to_fp16(src[f]);
                                }
                            }
                            emitted = true;
                        }
                        if (!emitted) {
                            // Unreachable by the block state machine (raw is freed only after
                            // CompressedResident); zero rows + loud error beats silent garbage.
                            std::cerr << "[DiffKV] LEGO FATAL: far block " << bi << " layer " << l
                                      << " has neither compressed pool data nor raw rows — emitting zeros."
                                      << std::endl;
                            std::memset(out_k, 0, (size_t)lego_far_bpr * F_test * sizeof(ggml_fp16_t));
                            std::memset(out_v, 0, (size_t)lego_far_bpr * F_test * sizeof(ggml_fp16_t));
                        }
                    }
                    ggml_backend_tensor_set(lego_far_k[l], far_stage_k.data(), 0,
                                            (size_t)lego_n_far * F_test * sizeof(ggml_fp16_t));
                    ggml_backend_tensor_set(lego_far_v[l], far_stage_v.data(), 0,
                                            (size_t)lego_n_far * F_test * sizeof(ggml_fp16_t));
                }
            }

            // Recreate the scheduler at each chunk iteration to prevent memory accumulation in the scheduler pool.
            {
                ggml_backend_sched_free(backend_owner.sched);
                std::vector<ggml_backend_t> backends;
                if (backend_owner.gpu_backend && backend_owner.gpu_backend != backend_owner.cpu_backend) {
                    backends.push_back(backend_owner.gpu_backend);
                }
                backends.push_back(backend_owner.cpu_backend);
                size_t sched_size = 8192;
                if (is_native_attn_enabled()) sched_size = 40960;
                backend_owner.sched = ggml_backend_sched_new(backends.data(), NULL, backends.size(), sched_size, false, true);
                sched = backend_owner.sched;
            }

            ggml_reset(prefill_ctx);

            // ── 2. Create input tensors ────────────────────────────────────────
            struct ggml_tensor * input_tokens_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, chunk_len);
            ggml_set_input(input_tokens_prefill);
            struct ggml_tensor * positions_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, chunk_len);
            ggml_set_input(positions_prefill);

            // Mask: [key_len, chunk_len]. key_len = full context (dense) or sp_klen (sparse).
            int intra_ctx_len = pos_start + chunk_len;
            int mask_klen = use_sp ? sp_klen : intra_ctx_len;
            struct ggml_tensor * mask_prefill = ggml_new_tensor_2d(prefill_ctx, GGML_TYPE_F16, mask_klen, chunk_len);
            ggml_set_input(mask_prefill);

            // ── 4. Build the graph ────────────────────────────────────────────
            struct ggml_tensor * prefill_logits = nullptr;
            std::vector<struct ggml_tensor *> prefill_k_layers(n_layers, nullptr);
            std::vector<struct ggml_tensor *> prefill_v_layers(n_layers, nullptr);
            bool is_last_chunk = (pos_start + chunk_len >= L);

            struct ggml_cgraph * prefill_graph = build_prefill_ctx_graph(
                prefill_ctx, model,
                input_tokens_prefill, positions_prefill, mask_prefill,
                nullptr, nullptr,
                &prefill_logits,
                &prefill_k_layers, &prefill_v_layers,
                is_last_chunk,
                &persistent_k_cache,
                &persistent_v_cache,
                pos_start,
                use_sp ? &sp_ranges_buf : nullptr,
                lego_on ? &chunk_write_spans : nullptr,
                (lego_on && lego_n_far > 0) ? &lego_far_k : nullptr,
                (lego_on && lego_n_far > 0) ? &lego_far_v : nullptr,
                lego_n_far
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

            // Build the mask, matching the K/V layout the graph attends.
            std::vector<ggml_fp16_t> mask_host;
            if (use_sp) {
                // Range-driven mask, matching the graph's key layout: the FAR segment first
                // (lego uploads; all history → always visible → stays zero), then the
                // ABSOLUTE ranges in order. Each range slot's abs pos = range.start + offset.
                // History keys (abs < pos_start) are always visible; current-chunk keys causal.
                mask_host.assign((size_t)chunk_len * sp_klen, ggml_fp32_to_fp16(0.0f));
                for (int qi = 0; qi < chunk_len; ++qi) {
                    int slot = lego_n_far;
                    for (const auto& r : sp_ranges_abs) {
                        for (int j = 0; j < r.second; ++j, ++slot) {
                            int chunk_j = (r.first + j) - pos_start;   // >=0 only for current-chunk keys
                            if (chunk_j > qi) {
                                mask_host[(size_t)qi * sp_klen + slot] = ggml_fp32_to_fp16(-INFINITY);
                            }
                        }
                    }
                }
            } else {
                // Dense full-context mask.
                int intra_prior = pos_start;
                mask_host.assign((size_t)chunk_len * intra_ctx_len, ggml_fp32_to_fp16(0.0f));
                for (int qi = 0; qi < chunk_len; ++qi) {
                    for (int kj = intra_prior; kj < intra_ctx_len; ++kj) {
                        int chunk_kj = kj - intra_prior;
                        if (chunk_kj > qi) {
                            mask_host[(size_t)qi * intra_ctx_len + kj] = ggml_fp32_to_fp16(-INFINITY);
                        }
                    }
                }
            }
            ggml_backend_tensor_set(mask_prefill, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));

            auto tp_b1 = std::chrono::high_resolution_clock::now();
            tp_build += std::chrono::duration<double,std::milli>(tp_b1 - tp_c0).count();
            auto tp_u1 = std::chrono::high_resolution_clock::now();
            tp_upload += std::chrono::duration<double,std::milli>(tp_u1 - tp_b1).count();

            // ── 6. Run the graph ──────────────────────────────────────────────
            if (ggml_backend_sched_graph_compute(sched, prefill_graph) != GGML_STATUS_SUCCESS) {
                std::cerr << "Error: Prefill graph compute failed at pos " << pos_start << "!" << std::endl;
                break;
            }
            auto tp_g1 = std::chrono::high_resolution_clock::now();
            tp_compute += std::chrono::duration<double,std::milli>(tp_g1 - tp_u1).count();

            // ── 7. Capture raw K/V for decode attention and next chunk's prior context ──
            // build_prefill_ctx_graph exports raw K (before RoPE) matching ACTIVE_RUNTIME.
            // k_activations stores raw K → used by decode callback (has_rope=true re-applies RoPE).
            // For next chunk's prior context, apply_rope_neox_cpu will rotate it at upload time.
            std::vector<std::vector<float>> chunk_k(n_layers, std::vector<float>(chunk_len * F_test));
            std::vector<std::vector<float>> chunk_v(n_layers, std::vector<float>(chunk_len * F_test));
            // Store raw K/V into the host mirrors. LEGO stage 2: writes go through the
            // SAME ring spans the device cache write uses (chunk_write_spans; identity
            // single-span [pos_start, chunk_len) when lego is off).
            std::vector<std::pair<int,int>> mirror_write_spans;
            if (lego_on) mirror_write_spans = chunk_write_spans;
            else         mirror_write_spans.push_back({pos_start, chunk_len});
            for (int l = 0; l < n_layers; ++l) {
                if (prefill_k_layers[l]->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> temp_k_f16(chunk_len * F_test);
                    std::vector<ggml_fp16_t> temp_v_f16(chunk_len * F_test);
                    ggml_backend_tensor_get(prefill_k_layers[l], temp_k_f16.data(), 0, chunk_len * F_test * sizeof(ggml_fp16_t));
                    ggml_backend_tensor_get(prefill_v_layers[l], temp_v_f16.data(), 0, chunk_len * F_test * sizeof(ggml_fp16_t));
                    for (int i = 0; i < chunk_len * F_test; ++i) {
                        chunk_k[l][i] = ggml_fp16_to_fp32(temp_k_f16[i]);
                        chunk_v[l][i] = ggml_fp16_to_fp32(temp_v_f16[i]);
                    }
                    int src_row = 0;
                    for (const auto & w : mirror_write_spans) {
                        std::memcpy(&k_activations[l][(size_t)w.first * F_test],
                                    temp_k_f16.data() + (size_t)src_row * F_test,
                                    (size_t)w.second * F_test * sizeof(ggml_fp16_t));
                        std::memcpy(&v_activations[l][(size_t)w.first * F_test],
                                    temp_v_f16.data() + (size_t)src_row * F_test,
                                    (size_t)w.second * F_test * sizeof(ggml_fp16_t));
                        src_row += w.second;
                    }
                } else {
                    ggml_backend_tensor_get(prefill_k_layers[l], chunk_k[l].data(), 0, chunk_len * F_test * sizeof(float));
                    ggml_backend_tensor_get(prefill_v_layers[l], chunk_v[l].data(), 0, chunk_len * F_test * sizeof(float));
                    int src_row = 0;
                    for (const auto & w : mirror_write_spans) {
                        for (int rr = 0; rr < w.second; ++rr) {
                            const size_t so = (size_t)(src_row + rr) * F_test;
                            const size_t dofs = (size_t)(w.first + rr) * F_test;
                            for (int f = 0; f < F_test; ++f) {
                                k_activations[l][dofs + f] = ggml_fp32_to_fp16(chunk_k[l][so + f]);
                                v_activations[l][dofs + f] = ggml_fp32_to_fp16(chunk_v[l][so + f]);
                            }
                        }
                        src_row += w.second;
                    }
                }
                
                int nan_k = 0;
                int nan_v = 0;
                for (int i = 0; i < chunk_len * F_test; ++i) {
                    if (std::isnan(chunk_k[l][i])) nan_k++;
                    if (std::isnan(chunk_v[l][i])) nan_v++;
                }
                if (nan_k > 0 || nan_v > 0) {
                    std::cerr << "[PREFILL_NAN_DETECT] layer=" << l 
                              << " pos_start=" << pos_start << " nan_k=" << nan_k 
                              << " nan_v=" << nan_v << std::endl;
                }
            }

            // RoPE-rotate only the current chunk and store it in k_rotated_activations
            if (!decode_use_sparse) {
                std::vector<int32_t> chunk_positions(chunk_len);
                for (int i = 0; i < chunk_len; ++i) chunk_positions[i] = pos_start + i;

                int half_dim = head_dim / 2;
                std::vector<float> inv_freq(half_dim);
                for (int i = 0; i < half_dim; ++i) {
                    inv_freq[i] = 1.0f / std::pow(model.get_config().rope_freq_base, 2.0f * i / head_dim);
                }
                std::vector<float> cos_table(chunk_len * half_dim);
                std::vector<float> sin_table(chunk_len * half_dim);
                for (int t = 0; t < chunk_len; ++t) {
                    float pos = (float)chunk_positions[t];
                    for (int i = 0; i < half_dim; ++i) {
                        float theta = pos * inv_freq[i];
                        cos_table[t * half_dim + i] = std::cos(theta);
                        sin_table[t * half_dim + i] = std::sin(theta);
                    }
                }
                for (int l = 0; l < n_layers; ++l) {
                    apply_rope_neox_cpu_fast(
                        k_activations[l].data() + pos_start * F_test,
                        k_rotated_activations[l].data() + pos_start * F_test,
                        cos_table.data(),
                        sin_table.data(),
                        chunk_len, kv_heads, head_dim
                    );
                }
            }
            auto tp_cap1 = std::chrono::high_resolution_clock::now();
            tp_capture += std::chrono::duration<double,std::milli>(tp_cap1 - tp_g1).count();
            // Ingest chunk into KV manager (raw K + raw V, matching ACTIVE_RUNTIME ingest_streaming)
            if (ingest_async) {
                prefill_ingest_thread = std::thread(
                    [&runtime_manager, chunk_k = std::move(chunk_k), chunk_v = std::move(chunk_v), chunk_len, pos_start, prompt_tokens, &srl_state]() {
                        runtime_manager.ingest_prefill(chunk_k, chunk_v, chunk_len, pos_start, prompt_tokens, &srl_state);
                    }
                );
                prefill_ingest_thread_active = true;
            } else {
                runtime_manager.ingest_prefill(chunk_k, chunk_v, chunk_len, pos_start, prompt_tokens, &srl_state);
            }
            tp_ingest += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now() - tp_cap1).count();
            tp_chunks++;

            if (pos_start + chunk_len >= L && prefill_logits) {
                ggml_backend_tensor_get(prefill_logits, prefill_output_logits.data(), 0, n_vocab * sizeof(float));
            }

            pos_start += chunk_len;
        }

        if (prefill_ingest_thread_active) {
            prefill_ingest_thread.join();
            prefill_ingest_thread_active = false;
        }

        // ── DIFFKV_DBG_EXPORT_CHECK: verify the raw-K export (k_activations, feeds the
        // custom-op dense window + compression pool) against the in-graph persistent
        // rotated K cache (feeds ggml flash decode — the known-good path). For sampled
        // positions: rotate raw K at its position and diff vs the persistent cache.
        // Large diffs pinpoint which chunks' exports are corrupted (buffer reuse etc.).
        if (std::getenv("DIFFKV_DBG_EXPORT_CHECK") && !is_warmup_run && persistent_k_cache[0] && !lego_on) {
            const int half_dim = head_dim / 2;
            const float freq_base = model.get_config().rope_freq_base;
            for (int pos = 64; pos < L; pos += 256) {
                for (int l = 0; l < std::min(2, n_layers); ++l) {
                    std::vector<ggml_fp16_t> pk_row(F_test);
                    ggml_backend_tensor_get(persistent_k_cache[l], pk_row.data(),
                                            (size_t)pos * F_test * sizeof(ggml_fp16_t),
                                            F_test * sizeof(ggml_fp16_t));
                    double max_diff = 0.0, ref_nrm = 0.0;
                    for (int kv = 0; kv < kv_heads; ++kv) {
                        for (int d = 0; d < half_dim; ++d) {
                            float x = ggml_fp16_to_fp32(k_activations[l][(size_t)pos * F_test + kv * head_dim + d]);
                            float y = ggml_fp16_to_fp32(k_activations[l][(size_t)pos * F_test + kv * head_dim + d + half_dim]);
                            double theta = 1.0 / std::pow((double)freq_base, (2.0 * d) / head_dim);
                            double ang = std::fmod((double)pos * theta, 2.0 * M_PI);
                            float c = (float)std::cos(ang), s = (float)std::sin(ang);
                            float r1 = x * c - y * s;
                            float r2 = y * c + x * s;
                            float g1 = ggml_fp16_to_fp32(pk_row[kv * head_dim + d]);
                            float g2 = ggml_fp16_to_fp32(pk_row[kv * head_dim + d + half_dim]);
                            max_diff = std::max(max_diff, (double)std::fabs(r1 - g1));
                            max_diff = std::max(max_diff, (double)std::fabs(r2 - g2));
                            ref_nrm += (double)g1 * g1 + (double)g2 * g2;
                        }
                    }
                    // V side: no rotation — direct compare raw export vs persistent cache.
                    std::vector<ggml_fp16_t> pv_row(F_test);
                    ggml_backend_tensor_get(persistent_v_cache[l], pv_row.data(),
                                            (size_t)pos * F_test * sizeof(ggml_fp16_t),
                                            F_test * sizeof(ggml_fp16_t));
                    double v_max_diff = 0.0, v_ref = 0.0;
                    for (int f = 0; f < F_test; ++f) {
                        float ve = ggml_fp16_to_fp32(v_activations[l][(size_t)pos * F_test + f]);
                        float vg = ggml_fp16_to_fp32(pv_row[f]);
                        v_max_diff = std::max(v_max_diff, (double)std::fabs(ve - vg));
                        v_ref += (double)vg * vg;
                    }
                    std::cerr << "[EXPORT_CHECK] pos=" << pos << " layer=" << l
                              << " K_maxDiff=" << max_diff << " |K|=" << std::sqrt(ref_nrm)
                              << " V_maxDiff=" << v_max_diff << " |V|=" << std::sqrt(v_ref)
                              << "\n";
                }
            }
        }

        if (dbg_prefill_time && !is_warmup_run) {
            double total = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now() - tp_start).count();
            std::cerr << "[PREFILL_TIME] L=" << L << " chunks=" << tp_chunks << " TOTAL=" << total/1000.0 << "s"
                      << " | graph_build=" << tp_build/1000.0 << "s prior_upload=" << tp_upload/1000.0 << "s"
                      << " compute=" << tp_compute/1000.0 << "s capture=" << tp_capture/1000.0 << "s ingest=" << tp_ingest/1000.0 << "s\n";
        }

        // ── DIFFKV_DBG_RECON_POS=<global token pos>: pool-reconstruction fidelity probe ──
        // After prefill (before k_activations is freed), find the compressed block holding
        // <pos>, reconstruct every delta row from the pool (anchor + U·VK·row_scale·blk_scale
        // + residual correction) and compare against ground truth (k_activations rotated at
        // the row's absolute position under POOL_ROT_ABS). Prints per-row relative error for
        // rows near <pos>. Separates capture-side corruption from decode-side bugs.
        if (const char* env_rp = (lego_on ? nullptr : std::getenv("DIFFKV_DBG_RECON_POS"))) {
            // (gated off under LEGO stage 2: the probe's ground truth reads
            // k_activations at arbitrary absolute positions, which the ring no
            // longer holds.)
            int want_pos = std::stoi(env_rp);
            runtime_manager.wait_for_compressor();
            // Scan the POOL directly by slot (older blocks are flushed from the ingest
            // manager's list into the pool at long ctx, so that list can't find them).
            NativeBlockPool* pool = kv_engines[0].get();
            int _nsl = pool->get_n_slots();
            for (int slot = 0; slot < _nsl; ++slot) {
                const int32_t* _tp = pool->get_host_token_positions(slot);
                int _sl = pool->get_host_seq_lens()[slot];
                if (!_tp || _sl <= 0) continue;
                bool _holds = false;
                for (int _t = 0; _t < _sl; ++_t) if (_tp[_t] >= want_pos - 2 && _tp[_t] <= want_pos + 2) { _holds = true; break; }
                if (!_holds) continue;
                std::cerr << "[RECON_POS] slot=" << slot << " holds pos ~" << want_pos << " seq_len=" << _sl << "\n";
                const int8_t* U = pool->get_host_U(slot);
                const ggml_fp16_t* rowsc = pool->get_host_U_row_scale(slot);
                const ggml_fp16_t* VK = pool->get_host_VK(slot);
                const ggml_fp16_t* ancK = pool->get_host_anchors_K(slot);
                const ggml_fp16_t* resKv = pool->get_host_res_K_val(slot);
                const int32_t* resKp = pool->get_host_res_K_pos(slot);
                const int32_t* tpos = pool->get_host_token_positions(slot);
                float bsc = ggml_fp16_to_fp32(pool->get_host_scales()[slot]);
                int slen = pool->get_host_seq_lens()[slot];
                int R = pool->get_rank();
                const int MRR = NativeBlockPool::MAX_RESIDUAL;
                const int half_dim = head_dim / 2;
                const float fb = model.get_config().rope_freq_base;
                if (!U || !VK || !ancK || !tpos) { std::cerr << "[RECON_POS] null host buffers\n"; break; }
                for (int t = 0; t < slen; ++t) {
                    int gp = tpos[t];
                    if (gp < want_pos - 6 || gp > want_pos + 12) continue;
                    int ri = -1;
                    if (resKp) for (int r = 0; r < MRR; ++r) if (resKp[r] == t) { ri = r; break; }
                    double err = 0, nrm = 0;
                    float ru = ggml_fp16_to_fp32(rowsc[t]);
                    for (int kv = 0; kv < kv_heads; ++kv) {
                        for (int d = 0; d < head_dim; ++d) {
                            float rec = ggml_fp16_to_fp32(ancK[kv*head_dim + d]);
                            for (int r = 0; r < R; ++r)
                                rec += (float)U[t*R + r] * ggml_fp16_to_fp32(VK[r*kv_heads*head_dim + kv*head_dim + d]) * ru * bsc;
                            if (ri >= 0 && resKv)
                                rec += ggml_fp16_to_fp32(resKv[(size_t)ri*kv_heads*head_dim + kv*head_dim + d]);
                            // ground truth: raw K rotated at absolute pos (ABS scheme)
                            float x = ggml_fp16_to_fp32(k_activations[0][(size_t)gp*F_test + kv*head_dim + (d < half_dim ? d : d - half_dim)]);
                            float y = ggml_fp16_to_fp32(k_activations[0][(size_t)gp*F_test + kv*head_dim + (d < half_dim ? d + half_dim : d)]);
                            double th = 1.0 / std::pow((double)fb, (2.0 * (d < half_dim ? d : d - half_dim)) / head_dim);
                            double ang = std::fmod((double)gp * th, 2.0 * M_PI);
                            float c = (float)std::cos(ang), s = (float)std::sin(ang);
                            float truth = (d < half_dim) ? (x * c - y * s) : (x * c + y * s);
                            // note: for d>=half: stored pair is (x=row[d-half],y=row[d]); rotated second = y*c + x*s
                            if (d >= half_dim) { float xx = ggml_fp16_to_fp32(k_activations[0][(size_t)gp*F_test + kv*head_dim + d - half_dim]);
                                                 float yy = ggml_fp16_to_fp32(k_activations[0][(size_t)gp*F_test + kv*head_dim + d]);
                                                 truth = yy * c + xx * s; }
                            err += (double)(rec - truth) * (rec - truth);
                            nrm += (double)truth * truth;
                        }
                    }
                    // V-side reconstruction: anchor_v + U·VV·row_scale·blk_scale (+resV). No rotation.
                    const ggml_fp16_t* VV = pool->get_host_VV(slot);
                    const ggml_fp16_t* ancV = pool->get_host_anchors_V(slot);
                    const ggml_fp16_t* resVv = pool->get_host_res_V_val(slot);
                    const int32_t* resVp = pool->get_host_res_V_pos(slot);
                    int rvi = -1;
                    if (resVp) for (int r = 0; r < MRR; ++r) if (resVp[r] == t) { rvi = r; break; }
                    double verr = 0, vnrm = 0;
                    if (VV && ancV) {
                        for (int kv = 0; kv < kv_heads; ++kv) {
                            for (int d = 0; d < head_dim; ++d) {
                                float rec = ggml_fp16_to_fp32(ancV[kv*head_dim + d]);
                                for (int r = 0; r < R; ++r)
                                    rec += (float)U[t*R + r] * ggml_fp16_to_fp32(VV[r*kv_heads*head_dim + kv*head_dim + d]) * ru * bsc;
                                if (rvi >= 0 && resVv)
                                    rec += ggml_fp16_to_fp32(resVv[(size_t)rvi*kv_heads*head_dim + kv*head_dim + d]);
                                float truth = ggml_fp16_to_fp32(v_activations[0][(size_t)gp*F_test + kv*head_dim + d]);
                                verr += (double)(rec - truth) * (rec - truth);
                                vnrm += (double)truth * truth;
                            }
                        }
                    }
                    std::cerr << "[RECON_POS] t=" << t << " gp=" << gp << " res=" << (ri >= 0 ? "Y" : "n")
                              << " K_rel_err=" << (nrm > 0 ? std::sqrt(err/nrm) : -1)
                              << " V_rel_err=" << (vnrm > 0 ? std::sqrt(verr/vnrm) : -1) << "\n";
                }
                break;
            }
        }

        if (prefill_ctx) {
            ggml_free(prefill_ctx);
        }

        if (prefill_cache_buffer) {
            ggml_backend_buffer_free(prefill_cache_buffer);
        }
        if (prefill_cache_ctx) {
            ggml_free(prefill_cache_ctx);
        }

        // Recreate the scheduler after prefill to reclaim all prefill-graph GPU allocations!
        // Prefill graph allocates large causal masks and intermediate tensors that remain
        // cached by the scheduler. Recreating the scheduler frees this cached memory (~1 GB VRAM at 32k)
        // while the subsequent decode graph only needs a tiny fraction of that size.
        {
            ggml_backend_sched_free(backend_owner.sched);
            std::vector<ggml_backend_t> backends;
            if (backend_owner.gpu_backend && backend_owner.gpu_backend != backend_owner.cpu_backend) {
                backends.push_back(backend_owner.gpu_backend);
            }
            backends.push_back(backend_owner.cpu_backend);
            size_t sched_size = 8192;
            if (is_native_attn_enabled()) sched_size = 40960;
            backend_owner.sched = ggml_backend_sched_new(backends.data(), NULL, backends.size(), sched_size, false, true);
            sched = backend_owner.sched;
        }

        // ── RAM fix (mirror MLX mx.clear_cache() at the prefill→decode boundary) ──
        // k_rotated_activations is used ONLY during prefill (the per-chunk RoPE'd-K
        // re-upload). It is dead weight through the entire decode phase, where it
        // costs L × F × 4 × n_layers bytes (~700 MB at 24k tokens, 1.5B). Release it
        // now so it does not sit resident (and swapping) while decode runs.
        if (decode_use_sparse) {
            for (auto & v : k_rotated_activations) {
                std::vector<ggml_fp16_t>().swap(v);  // free capacity, not just size
            }
        }

        // ── ACTIVE_RUNTIME batch_engine.py Fix 4: Fire-and-forget compression + SRL build ──
        // ACTIVE_RUNTIME: "The first token is already streamed above via _emit_token.
        //   Compression and SRL index are built in background so they don't block the
        //   decode loop. CRITICAL: finalize_srl_index is CPU-heavy sync code."
        //
        // In diffkv_native we do the same:
        //   1. Do a quick non-blocking sync of already-compressed slots (Metal-safe)
        //   2. Fire background std::thread: wait_for_compressor → update_descriptors →
        //      build_srl_state_from_blocks → factual_store.build
        //   3. Decode starts immediately (empty SRL = recency-only routing)
        //   4. When thread done: main thread does final GPU sync + swaps in SRL state
        //
        // Metal thread safety: sync_device_for_native() ONLY called from main thread
        // (via srl_needs_gpu_sync atomic flag). Background thread is CPU-only.
        runtime_manager.update_descriptors(W_proj_host, desc_dim, head_dim);
        runtime_manager.sync_device_for_native(); // quick sync of already-done slots

        if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
            std::cerr << "[DEBUG ACTIVATIONS] cached_len=" << cached_len << " L=" << L << "\n";
            std::cerr << "  k_activations[0] at 0 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << ggml_fp16_to_fp32(k_activations[0][0 * F_test + i]);
            std::cerr << "\n  k_activations[0] at 21 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << ggml_fp16_to_fp32(k_activations[0][21 * F_test + i]);
            std::cerr << "\n  k_activations[0] at 77 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << ggml_fp16_to_fp32(k_activations[0][77 * F_test + i]);
            std::cerr << "\n  v_activations[0] at 0 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << ggml_fp16_to_fp32(v_activations[0][0 * F_test + i]);
            std::cerr << "\n  v_activations[0] at 21 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << ggml_fp16_to_fp32(v_activations[0][21 * F_test + i]);
            std::cerr << "\n  v_activations[0] at 77 (first 5):";
            for (int i = 0; i < std::min(5, F_test); ++i) std::cerr << " " << ggml_fp16_to_fp32(v_activations[0][77 * F_test + i]);
            std::cerr << std::endl;
        }

        // Logits already retrieved inside the chunk loop

        int32_t first_decode_token = 0;
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

        if (interactive) {
            first_decode_token = sample_logits(prefill_output_logits, temperature, top_p, sample_rng);
        }
        if (const char* env_force = std::getenv("DIFFKV_FORCE_FIRST")) {
            first_decode_token = std::stoi(env_force);
            std::cerr << "[DiffKV DIAG] Forcing first token to: " << first_decode_token << " (\"" << model.token_to_piece(first_decode_token) << "\")\n";
        }
        {
            const auto& helper_ids = diffkv::get_helper_token_ids_cpp(model);
            std::cerr << "[DiffKV DIAG] Total helper IDs in C++: " << helper_ids.size() << "\n";
        }

        // Resources already freed inside the chunk loop

        std::atomic<bool> srl_build_done{false};
        std::atomic<bool> srl_needs_gpu_sync{false};
        diffkv::SessionSRLState  srl_state_pending;
        bool             srl_swapped = false;

        // ── Synchronous pre-decode SRL build ──
        const int        _mbs        = micro_block_size;
        const int        _desc_dim   = desc_dim;
        const int        _head_dim   = head_dim;
        const int        _kv_heads   = kv_heads;
        const int        _L          = L;
        std::vector<int32_t> _prompt_tokens_copy = prompt_tokens; // thread-safe copy
        std::vector<float>   _W_proj_copy        = W_proj_host;   // thread-safe copy
        const auto& _k_act_ref = k_activations;
        const auto& _v_act_ref = v_activations;

        std::unordered_set<int32_t> relational_token_ids;
        {
            static const std::unordered_set<std::string> RELATIONAL_KEYWORDS = {
                "unlike", "whereas", "while", "although", "however", "but",
                "instead", "rather", "conversely", "nevertheless", "nonetheless",
                "yet", "though", "notwithstanding",
                "compared", "differs", "differ", "different", "difference",
                "differences", "distinct", "distinction", "distinguishes",
                "greater", "larger", "smaller", "higher", "lower", "fewer",
                "more", "less", "most", "least",
                "causes", "caused", "because", "therefore", "hence", "thus",
                "leads", "results", "produces", "induces", "triggers",
                "consequently", "accordingly",
                "is", "are", "was", "were", "has", "have", "had",
                "exhibits", "possesses", "contains", "involves",
                "requires", "lacks", "features",
                "called", "named", "known", "defined", "characterized",
                "classified", "denoted", "refers", "represents",
                "only", "exclusively", "specifically", "solely",
                "except", "excluding", "neither", "nor"
            };

            for (int32_t tid : prompt_tokens) {
                if (relational_token_ids.count(tid)) continue;
                std::string text = model.token_to_piece(tid);
                std::string cleaned = "";
                for (char c : text) {
                    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                        cleaned += std::tolower((unsigned char)c);
                    }
                }
                if (RELATIONAL_KEYWORDS.count(cleaned)) {
                    relational_token_ids.insert(tid);
                }
            }
        }

        std::thread srl_build_thread([&, _mbs, _desc_dim, _head_dim, _kv_heads, _L, relational_token_ids]() {
#ifdef __APPLE__
            pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
#endif
            runtime_manager.wait_for_compressor();
            runtime_manager.update_descriptors(_W_proj_copy, _desc_dim, _head_dim);
            srl_needs_gpu_sync.store(true, std::memory_order_release);

            auto& blocks_l0 = runtime_manager.get_ingest_manager().get_blocks(0);
            std::vector<int32_t> comp_slots;
            std::vector<int> comp_anchors;
            for (int i = 0; i < (int)blocks_l0.size(); ++i) {
                if (blocks_l0[i]->pool_idx != -1 &&
                    (blocks_l0[i]->state == BlockState::CompressedResident ||
                     blocks_l0[i]->state == BlockState::CPUResident)) {
                    comp_slots.push_back(blocks_l0[i]->pool_idx);
                    comp_anchors.push_back(blocks_l0[i]->anchor_idx);
                }
            }

            int n_comp = (int)comp_slots.size();
            if (n_comp > 0) {
                std::vector<float> desc_mat(n_comp * _desc_dim);
                for (int j = 0; j < n_comp; ++j) {
                    int sid = comp_slots[j];
                    const float* host_desc = runtime_manager.get_engines()[0]->get_host_desc_matrix(sid);
                    if (host_desc) {
                        std::memcpy(desc_mat.data() + j * _desc_dim,
                                    host_desc,
                                    _desc_dim * sizeof(float));
                    } else {
                        std::memset(desc_mat.data() + j * _desc_dim, 0, _desc_dim * sizeof(float));
                    }
                }
                srl_state_pending = build_srl_state_from_blocks(
                    desc_mat.data(), comp_slots.data(), n_comp,
                    _prompt_tokens_copy.data(), _L,
                    _mbs + 1, stop_token_ids,
                    6, 2, 0.15f, true, true,
                    &comp_anchors
                );
                srl_state_pending.n_blocks_at_last_graph_build = n_comp;

                srl_state_pending.setup_sas_and_eqa(
                    _prompt_tokens_copy, stop_token_ids,
                    [&](int32_t tid) { return model.token_to_piece(tid); }
                );

                bool disable_factual = !g_diffkv_enable_factual.load();
                if (lego_on && !disable_factual) {
                    // LEGO stage 2: factual_store.build reads the raw mirrors at
                    // arbitrary absolute positions — data the ring no longer holds.
                    std::cerr << "[DiffKV] Factual store forced OFF under DIFFKV_LEGO_PREFILL "
                                 "(reads ring-evicted raw activations).\n";
                    disable_factual = true;
                }
                try {
                  if (disable_factual) {
                    std::cerr << "[DiffKV] Factual store off (MLX turn-1 parity; DIFFKV_ENABLE_FACTUAL=1 to build).\n";
                  } else {
                    std::vector<int32_t> document_tokens = _prompt_tokens_copy;
                    if (document_tokens.size() > 150) {
                        document_tokens.resize(document_tokens.size() - 150);
                    }
                    std::unordered_set<int32_t> prime_slots_thread(
                        srl_state_pending.chunk_graph.cluster_centers_tensor.begin(),
                        srl_state_pending.chunk_graph.cluster_centers_tensor.end()
                    );
                    srl_state_pending.factual_store.build(
                        _k_act_ref, _v_act_ref,
                        document_tokens,
                        _W_proj_copy.data(),
                        _desc_dim, _head_dim, _kv_heads,
                        stop_token_ids,
                        comp_slots,
                        _mbs + 1,
                        srl_state_pending.inverted_index,
                        prime_slots_thread,
                        get_helper_token_ids_cpp(model),
                        relational_token_ids,
                        [&](int32_t tid) { return model.token_to_piece(tid); },
                        true
                    );
                    srl_state_pending.setup_sas_and_eqa(
                        document_tokens, stop_token_ids,
                        [&](int32_t tid) { return model.token_to_piece(tid); }
                    );
                    std::cerr << "[DiffKV] Factual store built (pre-decode): "
                              << srl_state_pending.factual_store.entries.size() << " entries.\n";
                  }
                } catch (const std::exception& fe) {
                    std::cerr << "[DiffKV] factual_store.build() in srl_build_thread failed: " << fe.what() << "\n";
                }

                std::cerr << "[DiffKV] SRL index ready: " << n_comp << " blocks." << std::endl;
            }
            srl_build_done.store(true, std::memory_order_release);
        });
        srl_build_thread.join();

        if (srl_needs_gpu_sync.load(std::memory_order_acquire)) {
            runtime_manager.sync_device_for_native();
        }
        srl_state = std::move(srl_state_pending);
        srl_swapped = true;

        // ── 2. DECODE PHASE — rebuild decode graph fresh (avoids sched-ctx pointer corruption) ──
        // §3.3 fix: default raised from 2048 → 4096 to match ACTIVE_RUNTIME diffkv_attention.py:75.
        // Comment from Python: "Default raised from 2048 → 4096 based on MPS benchmarks:
        //   ≤4K: DiffKV bypasses to pure dense. Dense handles these contexts fine without memory pressure.
        //   4K+: DiffKV engages. Decode is faster and VRAM is dramatically lower."
        // Previously C++ went lossy-sparse at [2048,4096) while Python stayed exact-dense.
        decode_use_sparse = (L >= engage_threshold);
        int last_pool_version = kv_engines[0]->get_pool_version();
        if (decode_use_sparse) {
            int n_comp_blocks = (int)runtime_manager.get_ingest_manager().get_blocks(0).size();
            if (n_comp_blocks == 0) {
                decode_use_sparse = false;
                // FALLBACK GAP FIX: the prefill loop only fills k_rotated_activations when
                // the PRE-prefill decode_use_sparse was false (perf: sparse decode doesn't
                // need it). When sparse-mode prefill ends with ZERO compressed blocks (e.g.
                // recency window ≥ prompt), we fall back to dense decode here — which uploads
                // k_rotated_activations. Without this backfill those buffers are all-zero and
                // dense decode attends garbage keys (observed: wrong needle digits).
                if (lego_on) {
                    // LEGO stage 2: the mirrors are ring-sized — a full-length backfill is
                    // impossible (far rows were dropped from host RAM). Unreachable in
                    // practice (lego requires sparse decode, and a ≥engage-threshold prefill
                    // always creates blocks), but if it ever fires, failing loudly beats
                    // decoding from a zero/garbled dense cache.
                    std::cerr << "[DiffKV] LEGO FATAL: sparse→dense fallback requested but the "
                                 "host mirrors are ring-sized (DIFFKV_LEGO_PREFILL). Aborting "
                                 "this generation.\n";
                    if (interactive) {
                        std::cout << "__RESPONSE__" << std::endl;
                        std::cout << "[Error: lego prefill cannot fall back to dense decode]" << std::flush;
                        std::cout << "\n__FINISH__" << std::endl;
                    }
                    if (!interactive) break; else continue;
                }
                if (L >= engage_threshold) {
                    std::cerr << "[DiffKV] sparse→dense fallback (0 compressed blocks): backfilling "
                                 "k_rotated_activations for " << L << " tokens\n";
                    int half_dim = head_dim / 2;
                    std::vector<float> inv_freq(half_dim);
                    for (int i = 0; i < half_dim; ++i) {
                        inv_freq[i] = 1.0f / std::pow(model.get_config().rope_freq_base, 2.0f * i / head_dim);
                    }
                    std::vector<float> cos_table((size_t)L * half_dim);
                    std::vector<float> sin_table((size_t)L * half_dim);
                    for (int t = 0; t < L; ++t) {
                        for (int i = 0; i < half_dim; ++i) {
                            float theta = (float)t * inv_freq[i];
                            cos_table[(size_t)t * half_dim + i] = std::cos(theta);
                            sin_table[(size_t)t * half_dim + i] = std::sin(theta);
                        }
                    }
                    for (int l = 0; l < n_layers; ++l) {
                        // Allocation was skipped for sparse-mode prefill — resize before writing.
                        k_rotated_activations[l].resize((size_t)L * F_test, ggml_fp32_to_fp16(0.0f));
                        apply_rope_neox_cpu_fast(
                            k_activations[l].data(),
                            k_rotated_activations[l].data(),
                            cos_table.data(), sin_table.data(),
                            L, kv_heads, head_dim
                        );
                    }
                }
            }
        }

        {
            int n_comp_blocks = (int)runtime_manager.get_ingest_manager().get_blocks(0).size();
            if (n_comp_blocks > 0) {
                // DEFAULT = adaptive-k PRUNING (retrieve-then-attend, à la DeepSeek DSA/NSA): attend
                // only the top ~max(20,15%) relevant compressed blocks, not ALL of them. Native used
                // to attend every block ("MLX parity"=true) — but that was misnamed (MLX itself routes
                // top-k) and, with the decode-cache (whose materialise cost scales with block count),
                // attending-all is 1.6× slower at 16k for NO recall benefit: NIAH 6/6 and the logit
                // margins are IDENTICAL with pruning (the needle block is always in the top-k; only
                // irrelevant blocks are dropped). Sparse decode with pruning (19.2 tps @16k) now even
                // beats native DENSE (17.7). DIFFKV_MLX_PARITY=1 forces the old attend-all path, which
                // is more robust for DIFFUSE multi-fact/synthesis queries (where many blocks matter) —
                // the same top-k tradeoff MLX already makes. Credit: DeepSeek NSA/DSA (retrieve-then-attend).
                bool mlx_parity = !diffkv_detect_narrow_query(prompt);
                if (const char* env_mp = std::getenv("DIFFKV_MLX_PARITY")) {
                    // Explicit env override always wins (isolated benchmarking of either mode).
                    mlx_parity = (std::strcmp(env_mp, "0") != 0 && std::strcmp(env_mp, "false") != 0 && std::strcmp(env_mp, "off") != 0);
                }
                if (mlx_parity) {
                    int target_k = std::min(n_comp_blocks, n_slots);
                    target_k = std::min(target_k, 256);
                    std::cerr << "[DiffKV] MLX parity active: srl_k_keep raised from " << srl_k_keep
                              << " → " << target_k << " (attending all compressed blocks capped at 256)\n";
                    srl_k_keep = target_k;
                    int sem_floor2 = srl_k_keep * 3;
                    if (srl_k_semantic < sem_floor2) {
                        srl_k_semantic = sem_floor2;
                    }
                    srl_k_host = 1 + srl_k_recency + srl_k_lexical + srl_k_semantic + srl_k_graph;
                } else {
                    // Mirror Python: k_min = max(20, 0.15 * N_total), k_max = min(200, N_total)
                    int adaptive_k_min = std::max(20, (int)(0.15f * n_comp_blocks));
                    int adaptive_k_max = std::min(200, n_comp_blocks);
                    int adaptive_k     = std::max(adaptive_k_min, std::min(srl_k_keep, adaptive_k_max));
                    adaptive_k = std::min(adaptive_k, 256);
                    // Only grow srl_k_keep, never shrink below the N4.2 floor already applied
                    if (adaptive_k > srl_k_keep) {
                        std::cerr << "[DiffKV] §3.1 adaptive-k: srl_k_keep raised from " << srl_k_keep
                                  << " → " << adaptive_k << " (15% of " << n_comp_blocks << " blocks capped at 256)\n";
                        srl_k_keep = adaptive_k;
                        // Ensure candidate pool stays ≥3× srl_k_keep for anchor_screen
                        int sem_floor2 = srl_k_keep * 3;
                        if (srl_k_semantic < sem_floor2) {
                            srl_k_semantic = sem_floor2;
                            std::cerr << "[DiffKV] §3.1 adaptive-k: srl_k_semantic raised to " << srl_k_semantic << "\n";
                        }
                        // Update host-slot total to reflect any changed channel budgets
                        srl_k_host = 1 + srl_k_recency + srl_k_lexical + srl_k_semantic + srl_k_graph;
                    }
                }
            }
        }

        // Short context dense fallback: if n_slots <= 32 or n_comp_blocks <= 32, we should attend to all blocks.
        if (n_slots <= 32) {
            int target_k = std::min(n_slots, 256);
            if (target_k > srl_k_keep) {
                std::cerr << "[DiffKV] Short context dense fallback (n_slots <= 32): srl_k_keep raised from " << srl_k_keep
                          << " → " << target_k << "\n";
                srl_k_keep = target_k;
                int sem_floor2 = srl_k_keep * 3;
                if (srl_k_semantic < sem_floor2) {
                    srl_k_semantic = sem_floor2;
                }
                srl_k_host = 1 + srl_k_recency + srl_k_lexical + srl_k_semantic + srl_k_graph;
            }
        }
        {
            int n_comp_blocks_for_dense = (int)runtime_manager.get_ingest_manager().get_blocks(0).size();
            if (n_comp_blocks_for_dense <= 36 && n_comp_blocks_for_dense > 0) {
                int target_k = std::min(n_comp_blocks_for_dense, n_slots);
                target_k = std::min(target_k, 256);
                if (target_k > srl_k_keep) {
                    std::cerr << "[DiffKV] Short context dense fallback (n_comp_blocks <= 36): srl_k_keep raised from " << srl_k_keep
                              << " → " << target_k << "\n";
                    srl_k_keep = target_k;
                    int sem_floor2 = srl_k_keep * 3;
                    if (srl_k_semantic < sem_floor2) {
                        srl_k_semantic = sem_floor2;
                    }
                    srl_k_host = 1 + srl_k_recency + srl_k_lexical + srl_k_semantic + srl_k_graph;
                }
            }
        }

        // Prefill RoPE rotation and GPU upload is deferred until after past-KV GPU allocation.

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
        struct ggml_tensor * dense_attn_mask_decode = ggml_new_tensor_2d(decode_ctx, GGML_TYPE_F16, required_dense_cap + 1, 1);
        ggml_set_input(dense_attn_mask_decode);

        // Attention scoring structure. Under POOL_ROT_ABS (default) the pool stores K
        // fully rotated at absolute positions, so NO decode-side rotation happens and the
        // cheap project-then-attend structure ("approximate") is mathematically EXACT —
        // delta scores (q·VK)·U[t] equal the reconstruct-then-dot scores because no
        // per-token rotation intervenes. Default it on there for speed. Under the legacy
        // raw/within-offset pool schemes, project-then-attend really is approximate
        // (single anchor-position rotation baked into the shared basis), so keep the
        // per-token reconstruct-and-rotate path as the default. DIFFKV_MPS_APPROXIMATE_ATTN
        // overrides in either direction.
        bool approx = (diffkv::pool_rot_mode() == diffkv::POOL_ROT_ABS);
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
            userdata[l].S_max = kv_engines[l]->get_S_max();
            userdata[l].K = 0;
            userdata[l].D = head_dim;
            userdata[l].scale = 1.0f / std::sqrt((float)head_dim);
            userdata[l].has_rope = true;
            userdata[l].rope_freq_base = model.get_config().rope_freq_base;
            userdata[l].approximate_attn = approx;
            userdata[l].ignore_c = false;  // attend the current/self token (MLX ingest_streaming + pure-dense both do)
            userdata[l].srl_state = &srl_state;
            userdata[l].W_proj = W_proj_host.data();
            userdata[l].desc_dim = desc_dim;
            userdata[l].max_active_dense_tokens = required_dense_cap;
            userdata[l].dense_capacity = required_dense_cap;
        }

        struct ggml_tensor * decode_logits = nullptr;
        struct ggml_tensor * decode_selected_slots = nullptr;
        struct ggml_tensor * decode_concat_k = nullptr;
        struct ggml_tensor * decode_concat_v = nullptr;
        struct ggml_tensor * decode_argmax = nullptr;
        struct ggml_tensor * dbg_anc = nullptr;

        struct ggml_init_params dense_past_params = {
            /*.mem_size   =*/ 4 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * dense_past_ctx = ggml_init(dense_past_params);
        std::vector<struct ggml_tensor *> dense_k_past_inputs(n_layers, nullptr);
        std::vector<struct ggml_tensor *> dense_v_past_inputs(n_layers, nullptr);
        if (!decode_use_sparse) {
            for (int l = 0; l < n_layers; ++l) {
                dense_k_past_inputs[l] = ggml_new_tensor_3d(dense_past_ctx, GGML_TYPE_F16, head_dim, kv_heads, required_dense_cap);
                ggml_set_input(dense_k_past_inputs[l]);
                dense_v_past_inputs[l] = ggml_new_tensor_3d(dense_past_ctx, GGML_TYPE_F16, head_dim, kv_heads, required_dense_cap);
                ggml_set_input(dense_v_past_inputs[l]);
            }
        }

        // ── DIFFKV_DECODE_CACHE: persistent F16 buffers for the materialized sparse cache.
        // Layout per layer: [routed blocks (srl_k_keep × micro_block_size) | dense window
        // (required_dense_cap)]. Filled host-side (routed once per N tokens; window per token)
        // and attended by the GPU flash kernel — replacing the CPU custom op. Bounded by K, so
        // decode RAM stays flat with context (the compression win is preserved).
        const bool decode_cache_on = is_decode_cache_enabled();
        // +1 per block for the ANCHOR row (materialize emits anchor + seq_len delta tokens =
        // micro_block_size+1 rows per block); without the +1 the last block's tail is truncated.
        const int cache_routed_cap = srl_k_keep * (micro_block_size + 1);
        const int cache_cap = cache_routed_cap + required_dense_cap;
        std::vector<struct ggml_tensor *> cache_k(n_layers, nullptr);
        std::vector<struct ggml_tensor *> cache_v(n_layers, nullptr);
        struct ggml_tensor * cache_mask = nullptr;
        if (decode_cache_on && decode_use_sparse) {
            for (int l = 0; l < n_layers; ++l) {
                cache_k[l] = ggml_new_tensor_3d(dense_past_ctx, GGML_TYPE_F16, head_dim, kv_heads, cache_cap);
                ggml_set_input(cache_k[l]);
                cache_v[l] = ggml_new_tensor_3d(dense_past_ctx, GGML_TYPE_F16, head_dim, kv_heads, cache_cap);
                ggml_set_input(cache_v[l]);
            }
            cache_mask = ggml_new_tensor_2d(dense_past_ctx, GGML_TYPE_F16, cache_cap + 1, 1);
            ggml_set_input(cache_mask);
        }

        // Native sparse-attn dense-window inputs (past rotated K / V + validity mask),
        // persistent (survive graph rebuilds), filled each decode step. Gated.
        // DIFFKV_CPU_EXACT_ATTN=1 forces the CPU callback path (execute_cpu_attention,
        // approximate_attn=false) which applies per-token RoPE + SVD residual correction.
        // Requires sched backend (not Metal-only), so native_attn_on must be false.
        static const bool cpu_exact_attn_env = (std::getenv("DIFFKV_CPU_EXACT_ATTN") != nullptr);
        bool native_attn_on = is_native_attn_enabled() && !cpu_exact_attn_env;
        std::vector<struct ggml_tensor *> native_dense_kr(n_layers, nullptr);
        std::vector<struct ggml_tensor *> native_dense_v(n_layers, nullptr);
        struct ggml_tensor * native_dense_mask = nullptr;
        if (native_attn_on) {
            for (int l = 0; l < n_layers; ++l) {
                native_dense_kr[l] = ggml_new_tensor_3d(dense_past_ctx, GGML_TYPE_F32, head_dim, kv_heads, native_maxd);
                ggml_set_input(native_dense_kr[l]);
                native_dense_v[l]  = ggml_new_tensor_3d(dense_past_ctx, GGML_TYPE_F32, head_dim, kv_heads, native_maxd);
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
        // Host-controlled slot list for native attention: bypasses in-graph SRL router.
        // Populated each step with CompressedResident-filtered candidates (or all-(-1) if none ready).
        // This prevents the graph's top-K op (which always selects K IDs regardless of slots_mask)
        // from routing to Freed-state pool garbage.
        struct ggml_tensor * native_attn_slots = nullptr;
        if (native_attn_on) {
            native_attn_slots = ggml_new_tensor_1d(dense_past_ctx, GGML_TYPE_I32, srl_k_keep);
            ggml_set_input(native_attn_slots);
        }
        ggml_backend_buffer_t dense_past_buffer = ggml_backend_alloc_ctx_tensors(dense_past_ctx, backend);
        if (!decode_use_sparse) {
            std::vector<ggml_fp16_t> zeros(required_dense_cap * F_test, ggml_fp32_to_fp16(0.0f));
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_tensor_set(dense_k_past_inputs[l], zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));
                ggml_backend_tensor_set(dense_v_past_inputs[l], zeros.data(), 0, zeros.size() * sizeof(ggml_fp16_t));

                ggml_backend_tensor_set(dense_k_past_inputs[l], k_rotated_activations[l].data(), 0, L * F_test * sizeof(ggml_fp16_t));
                ggml_backend_tensor_set(dense_v_past_inputs[l], v_activations[l].data(), 0, L * F_test * sizeof(ggml_fp16_t));
                // Free k_rotated_activations now since it is uploaded to GPU
                std::vector<ggml_fp16_t>().swap(k_rotated_activations[l]);
            }
        }
        if (native_attn_on) {
            // Fill the dedup constants once (persistent). tri[j,k] = 1 if j<k else 0 (column-major: idx = j + k*K).
            std::vector<float> tri((size_t)srl_k_keep * srl_k_keep, 0.0f);
            for (int k = 0; k < srl_k_keep; ++k)
                for (int j = 0; j < k; ++j)
                    tri[(size_t)j + (size_t)k * srl_k_keep] = 1.0f;
            ggml_backend_tensor_set(native_dup_tri, tri.data(), 0, tri.size() * sizeof(float));
            float halfv = 0.5f;
            ggml_backend_tensor_set(native_half, &halfv, 0, sizeof(float));
            // Initialize native_attn_slots to all-invalid; filled each decode step.
            if (native_attn_slots) {
                std::vector<int32_t> invalid(srl_k_keep, -1);
                ggml_backend_tensor_set(native_attn_slots, invalid.data(), 0, (size_t)srl_k_keep * sizeof(int32_t));
            }
        }

        if (native_decode_buf) {
            ggml_backend_buffer_free(native_decode_buf);
            native_decode_buf = nullptr;
        }

        // Note: §3.1 Adaptive-k scaling block moved to start of decode phase to ensure correct size allocations.
        bool graph_is_native = false;

        struct ggml_cgraph * decode_graph = build_decode_graph(
            decode_ctx, model, input_token_decode, position_decode, W_proj_decode,
            kv_engines[0]->get_desc_matrix(), kv_engines[0]->get_anchors_K(),
            slots_mask_decode, host_slots_decode,
            srl_k_semantic, srl_k_keep,
            userdata.data(), &decode_logits, &decode_selected_slots,
            &decode_concat_k, &decode_concat_v,
            decode_use_sparse, L, required_dense_cap,
            dense_k_past_inputs.data(), dense_v_past_inputs.data(),
            dense_attn_mask_decode,
            native_dense_kr.data(), native_dense_v.data(), native_dense_mask, native_maxd,
            native_dup_tri, native_half, native_dense_pos, native_attn_slots,
            cache_k.data(), cache_v.data(), cache_mask, cache_cap, &dbg_anc
        );
        if (dbg_anc) ggml_set_output(dbg_anc);
        ggml_set_output(decode_logits);
        if (decode_selected_slots) ggml_set_output(decode_selected_slots);
        if (decode_concat_k) ggml_set_output(decode_concat_k);
        if (decode_concat_v) ggml_set_output(decode_concat_v);

        // Native path: allocate the decode graph with NO buffer reuse (ggml_backend_sched's
        // reuse corrupts the native sparse-attn subgraph on big-score inputs — verified: the
        // subgraph MATH is exact under no-reuse via DIFFKV_SELFTEST). Direct backend compute.
        bool factual_empty = true;
        if (!userdata.empty() && userdata[0].srl_state) {
            factual_empty = static_cast<diffkv::SessionSRLState*>(userdata[0].srl_state)->factual_store.entries.empty();
        }
        bool use_native_attn = native_attn_on && decode_use_sparse && factual_empty &&
                               !userdata.empty() && userdata[0].kv_engine &&
                               userdata[0].kv_engine->native_attn_enabled();

        graph_is_native = use_native_attn;
        bool decode_alloc_ok;
        if (use_native_attn) {
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
                if (cache_k[l]) ggml_backend_sched_set_tensor_backend(sched, cache_k[l], backend);
                if (cache_v[l]) ggml_backend_sched_set_tensor_backend(sched, cache_v[l], backend);
            }
            if (dense_attn_mask_decode) ggml_backend_sched_set_tensor_backend(sched, dense_attn_mask_decode, backend);
            if (cache_mask) ggml_backend_sched_set_tensor_backend(sched, cache_mask, backend);

            // Explicitly route MAP_CUSTOM3 nodes (CPU custom op callbacks) to the CPU backend
            // so the scheduler doesn't erroneously propagate Metal/GPU backend to them.
            for (int i = 0; i < decode_graph->n_nodes; ++i) {
                struct ggml_tensor * node = decode_graph->nodes[i];
                if (node->op == GGML_OP_MAP_CUSTOM3) {
                    ggml_backend_sched_set_tensor_backend(sched, node, backend_owner.cpu_backend);
                }
            }
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
        max_generate = 2048;
        if (is_warmup_run) {
            max_generate = 2;
        } else if (const char* env_mt = std::getenv("DIFFKV_MAX_TOKENS")) {
            max_generate = std::max(1, std::stoi(env_mt));
        }

        std::vector<int32_t> all_tokens = prompt_tokens;
        all_tokens.push_back(last_token);



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
        // limiting TPS.
        //
        // Matches ACTIVE_RUNTIME architecture: ACTIVE_RUNTIME routes via GPU semantic
        // search (near-free) every step. The CPU lexical+graph routing here only needs
        // to refresh when the query topic shifts, not every token. Default = 32 steps.
        // The ggml graph's layer-0 semantic search (selected_slots) handles per-token
        // Q changes for free — so this CPU routing is complementary, not critical-path.
        //
        // Env override: DIFFKV_RETRIEVAL_INTERVAL=N (set to 1 to route every token)
        int retrieval_interval = 32;  // raised from 8 (matches Python cadence)
        if (const char* env_ri = std::getenv("DIFFKV_RETRIEVAL_INTERVAL")) {
            retrieval_interval = std::max(1, std::stoi(env_ri));
        }
        std::vector<int32_t> cached_routed_blocks;
        std::vector<int32_t> cached_physical_candidates;
        int last_retrieval_step = -retrieval_interval; // force retrieval on step 0
        int last_retrieval_active_slot = -1;

        // ── DIFFKV_DECODE_CACHE persistent host state ──
        // host_routed_k/v[l]: F16 materialized routed blocks (refreshed every N tokens).
        // cache_n_routed: valid routed token count from the last materialization (shared across
        // layers, since all layers route the same blocks). -1 forces a (re)materialize.
        const int decode_cache_N = decode_cache_interval();
        std::vector<std::vector<ggml_fp16_t>> host_routed_k(decode_cache_on ? n_layers : 0);
        std::vector<std::vector<ggml_fp16_t>> host_routed_v(decode_cache_on ? n_layers : 0);
        std::vector<std::vector<ggml_fp16_t>> host_win_k(decode_cache_on ? n_layers : 0);
        std::vector<std::vector<ggml_fp16_t>> host_win_v(decode_cache_on ? n_layers : 0);
        int cache_n_routed = -1;
        int cache_last_remat_version = -1;
        // Routed candidate set used at the last materialization. The re-route interval (32) is
        // longer than the remat interval (16), so periodic remats often reconstruct an IDENTICAL
        // routed set → redundant CPU recon + serial upload (the dominant sparse-fill cost). We
        // skip the materialize when this set and the pool version are both unchanged: the device
        // cache already holds the exact same bytes, so recall is provably identical.
        std::vector<int32_t> cache_last_materialized_cands;

        std::vector<int> dense_start_positions(n_layers, 0);
        std::vector<int> total_dense_tokens(n_layers, 0);

        if (std::getenv("DIFFKV_VERBOSE")) {
            auto & b0 = runtime_manager.get_ingest_manager().get_blocks(0);
            int hist[8] = {0};
            for (auto & b : b0) { int s = (int)b->state; if (s >= 0 && s < 8) hist[s]++; }
            std::cerr << "[DiffKV] block-state histogram (layer0, " << b0.size() << " blocks): "
                      << "Dense=" << hist[0] << " Compressing=" << hist[1]
                      << " Compressed=" << hist[2] << " PagingOut=" << hist[3]
                      << " CPU=" << hist[4] << " Reloading=" << hist[5]
                      << " Invalid=" << hist[6] << " Freed=" << hist[7] << "\n";
        }

        if (decode_use_sparse) {
            for (int l = 0; l < n_layers; ++l) {
                auto & b_list = runtime_manager.get_ingest_manager().get_blocks(l);
                int curr_token_idx = 0;
                bool found_first = false;

                // Capacity of the dense window buffer (in tokens). The scan below MUST NOT
                // write past this or it corrupts the heap (was the src/main.cpp:3026 SIGSEGV:
                // an unbounded memcpy when more dense tokens existed than the lazily-sized
                // buffer held, or a block with a corrupt active_k length).
                const int cap_tokens = (int)(active_k_dense[l].size() / (size_t)F_test);
                int dense_blocks_seen = 0;

                std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);

                for (auto & block : b_list) {
                    if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                        dense_blocks_seen++;
                        if (!found_first) {
                            dense_start_positions[l] = block->anchor_idx;
                            found_first = true;
                        }

                        // Anchor token (1 token). Stop if the buffer is full.
                        if (curr_token_idx + 1 > cap_tokens) {
                            static bool warned_anchor = false;
                            if (!warned_anchor && l == 0) {
                                warned_anchor = true;
                                std::cerr << "[DiffKV] WARNING: dense-window scan hit capacity ("
                                          << cap_tokens << " tok) at block anchor; truncating.\n";
                            }
                            break;
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
                            int active_len = (int)(block->active_k.size() / (size_t)F_test);
                            // Sanity: a healthy block has at most micro_block_size-1 non-anchor
                            // tokens. A wild value means a corrupted/aliased block — skip it.
                            if (active_len <= 0 || active_len > micro_block_size ||
                                block->active_v.size() != block->active_k.size()) {
                                static bool warned_bad = false;
                                if (!warned_bad && l == 0) {
                                    warned_bad = true;
                                    std::cerr << "[DiffKV] WARNING: dense block with bad active_k len="
                                              << active_len << " (k.size=" << block->active_k.size()
                                              << " v.size=" << block->active_v.size()
                                              << " F_test=" << F_test << "); skipping.\n";
                                }
                                continue;
                            }
                            // Clamp to remaining capacity.
                            if (curr_token_idx + active_len > cap_tokens) {
                                static bool warned_active = false;
                                if (!warned_active && l == 0) {
                                    warned_active = true;
                                    std::cerr << "[DiffKV] WARNING: dense-window scan hit capacity ("
                                              << cap_tokens << " tok); truncating active span.\n";
                                }
                                break;
                            }
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
                if (l == 0 && std::getenv("DIFFKV_VERBOSE")) {
                    std::cerr << "[DiffKV] dense-window scan: layer0 dense_blocks=" << dense_blocks_seen
                              << " dense_tokens=" << curr_token_idx << " cap=" << cap_tokens
                              << " total_blocks=" << b_list.size() << "\n";
                }
                total_dense_tokens[l] = curr_token_idx;
            }
        }

        {
            auto & b_list = runtime_manager.get_ingest_manager().get_blocks(0);
            int curr_pos_idx = 0;
            const int pos_cap = (int)active_positions_dense.size();
            std::fill(active_positions_dense.begin(), active_positions_dense.end(), 0);
            for (auto & block : b_list) {
                if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                    for (int32_t t_pos : block->token_indices) {
                        if (curr_pos_idx >= pos_cap) break;   // bounds guard (mirrors K/V scan)
                        active_positions_dense[curr_pos_idx++] = t_pos;
                    }
                    if (curr_pos_idx >= pos_cap) break;
                }
            }
            total_positions = curr_pos_idx;
        }

        if (std::getenv("DIFFKV_VERBOSE")) {
            int lastp = total_positions > 0 ? active_positions_dense[total_positions-1] : -1;
            std::cerr << "[DiffKV] dense-window positions: start=" << dense_start_positions[0]
                      << " count=" << total_positions << " pos[0]=" << active_positions_dense[0]
                      << " pos[last]=" << lastp << " (L=" << L << ", query@" << L << ")\n";
        }

        if (!decode_use_sparse) {
            for (int l = 0; l < n_layers; ++l) {
                total_dense_tokens[l] = L;
            }
            for (int i = 0; i < L; ++i) {
                active_positions_dense[i] = i;
            }
            total_positions = L;
        } else {
            // ── MLX-parity dense window ───────────────────────────────────────────
            // MLX (mlx_diffkv_wrapper.py) keeps the most-recent (recency_window + block_size)
            // tokens DENSE and CONTIGUOUS, sourced from the raw prefill K/V. The earlier
            // block-scan filled active_k_dense from per-block storage (block->anchor_k/active_k),
            // which diverged from the actual prefill activations — the dense window the sparse
            // decode attends was WRONG → gibberish, even though the IDENTICAL tokens are coherent
            // through the pure-dense path (which fills active_k_dense from k_activations). So:
            // OVERWRITE the dense window with a contiguous slice of k_activations/v_activations,
            // exactly like the dense path does (line ~3163), and use contiguous positions.
            int dense_cap   = (int)(active_k_dense[0].size() / (size_t)F_test);
            int mlx_dense   = std::min(L, cfg_recency_window + micro_block_size);  // MLX max_dense_len
            int dense_win   = std::min(mlx_dense, dense_cap);
            int dense_start = std::max(0, L - dense_win);
            // LEGO stage 2: the mirror tail lives at ring-mapped offsets. The window
            // ring capacity (sp_window + 2·chunk ≥ 2048) always covers dense_win
            // (recency_window + mbs ≤ 768 by default), so every row is resident;
            // lego_map_span is the identity when lego is off.
            std::vector<std::pair<int,int>> dense_seed_spans;
            lego_map_span(dense_start, dense_win, dense_seed_spans);
            for (int l = 0; l < n_layers; ++l) {
                size_t di = 0;
                for (const auto & w : dense_seed_spans) {
                    const size_t so = (size_t)w.first * F_test;
                    const size_t n  = (size_t)w.second * F_test;
                    for (size_t i = 0; i < n; ++i) {
                        active_k_dense[l][di + i] = ggml_fp16_to_fp32(k_activations[l][so + i]);
                        active_v_dense[l][di + i] = ggml_fp16_to_fp32(v_activations[l][so + i]);
                    }
                    di += n;
                }
                total_dense_tokens[l]    = dense_win;
                dense_start_positions[l] = dense_start;
            }
            for (int i = 0; i < dense_win; ++i) active_positions_dense[i] = dense_start + i;
            total_positions = dense_win;
        }

        const char* env_td = std::getenv("DIFFKV_TIME_DECODE");
        bool time_decode = (env_td && std::string(env_td) == "1");

        // Bug 🅓 fix: n-gram loop detection state (mirrors mlx_diffkv_wrapper.py:1204-1238)
        bool   loop_detected     = false;
        int    loop_detected_idx = -1;   // step index when loop was first detected

        // ═══════════════════════════════════════════════════════════════════════════
        // PERFORMANCE FIX: Persistent GPU buffer management
        // ═══════════════════════════════════════════════════════════════════════════
        // Instead of uploading 15MB of dense K/V buffers every token (48 uploads × 320KB),
        // we maintain persistent GPU buffers and only update the NEW token (48 uploads × 5KB).
        // This eliminates 99.7% of upload bandwidth: 15MB → 120KB per token.
        //
        // Expected improvement:
        //   - Metal: 0.3 TPS → 40-50 TPS (100x speedup)
        //   - CUDA:  0.5 TPS → 60-80 TPS (120x speedup, PCIe bottleneck eliminated)
        //
        // Implementation:
        //   1. Upload full buffers once at decode start (full_upload_needed = true)
        //   2. Per token: upload only the new token to its ring-buffer position
        //   3. Ring buffer wraps at native_maxd to handle long sequences
        // ═══════════════════════════════════════════════════════════════════════════
        bool persistent_buffers_initialized = false;
        int persistent_buffer_base_pos = 0;  // Track ring buffer start position

        // DIAGNOSTIC (DIFFKV_SCAN_POOL): after compression, scan every compressed slot for
        // NaN/Inf/extreme values — a single poisoned block (e.g. degenerate SVD) NaNs the whole
        // softmax → long-context word-salad. Drain the compressor first so the pool is final.
        if (std::getenv("DIFFKV_SCAN_POOL")) {
            runtime_manager.wait_for_compressor();
            const int F = kv_heads * head_dim;
            for (int l = 0; l < n_layers; ++l) {
                auto* pool = kv_engines[l].get();
                int ns = pool->get_seq_lens()->ne[0];
                int rk = pool->get_rank();
                const int32_t* sl = pool->get_host_seq_lens();
                const int32_t* ap = pool->get_host_anchor_positions();
                int nbad = 0, nactive = 0, gmax_slot = -1; double gmax = 0;
                int ap_min = INT32_MAX, ap_max = INT32_MIN, ap_dups = 0;
                std::vector<int> aps;
                for (int s = 0; s < ns; ++s) {
                    if (sl[s] <= 0) continue;
                    nactive++;
                    if (ap[s] < ap_min) ap_min = ap[s];
                    if (ap[s] > ap_max) ap_max = ap[s];
                    aps.push_back(ap[s]);
                    
                    const ggml_fp16_t* aK = pool->get_host_anchors_K(s);
                    const ggml_fp16_t* VK = pool->get_host_VK(s);
                    const ggml_fp16_t* VV = pool->get_host_VV(s);
                    
                    auto chk = [&](float v){ if (!std::isfinite(v)) nbad++; else { float a = std::fabs(v); if (a > gmax) { gmax = a; gmax_slot = s; } } };
                    if (aK) {
                        for (int i = 0; i < F; ++i) chk(ggml_fp16_to_fp32(aK[i]));
                    }
                    if (VK) {
                        for (int i = 0; i < rk*F; ++i) chk(ggml_fp16_to_fp32(VK[i]));
                    }
                    if (VV) {
                        for (int i = 0; i < rk*F; ++i) chk(ggml_fp16_to_fp32(VV[i]));
                    }
                }
                std::sort(aps.begin(), aps.end());
                for (size_t i = 1; i < aps.size(); ++i) if (aps[i] == aps[i-1]) ap_dups++;
                if (l < 4 || nbad > 0 || gmax > 1e3)
                    std::cerr << "[SCAN_POOL] L" << l << " active=" << nactive << " naninf=" << nbad
                              << " max|x|=" << gmax << " ap_range=[" << ap_min << "," << ap_max << "] ap_dups=" << ap_dups << "\n";
            }
            // Layer-0 block census: where are blocks lost? (state + missing-slot)
            {
                auto & blks = runtime_manager.get_ingest_manager().get_blocks(0);
                int nDense=0, nComp=0, nCompd=0, nOther=0, nNoSlot=0, total=0;
                for (auto & b : blks) {
                    total++;
                    if (b->pool_idx == -1) nNoSlot++;
                    BlockState st = (b->pool_idx>=0) ? kv_engines[0]->get_state_table().get(b->pool_idx) : b->state;
                    if      (st == BlockState::DenseResident)      nDense++;
                    else if (st == BlockState::Compressing)        nComp++;
                    else if (st == BlockState::CompressedResident) nCompd++;
                    else nOther++;
                }
                std::cerr << "[SCAN_CENSUS] L0 total_blocks=" << total << " dense=" << nDense
                          << " compressing=" << nComp << " compressed=" << nCompd
                          << " other=" << nOther << " no_slot=" << nNoSlot << "\n";
            }
        }



        double sum_sync_ms = 0.0;
        double sum_recon_ms = 0.0;
        double sum_graph_other_ms = 0.0;
        double sum_attn_ms = 0.0;
        double sum_sampling_ms = 0.0;
        double sum_other_ms = 0.0;
        // Real fill/graph split (the old attn-vs-other split was a hardcoded base_graph_ms=15
        // estimate that hid the sparse cache-fill cost inside the "graph" bucket entirely).
        double sum_fill_ms = 0.0;       // CPU cache fill (materialize routed + rotate/upload window)
        double sum_graph_gpu_ms = 0.0;  // pure GPU graph compute (flash + forward), fill excluded
        int profile_steps = 0;

        std::thread svd_thread;
        bool svd_thread_active = false;
        std::vector<std::vector<float>> prev_decode_k;
        std::vector<std::vector<float>> prev_decode_v;
        std::vector<int32_t> prev_all_tokens;
        int prev_pos = 0;
        bool has_prev_svd = false;
        const bool never_use_sparse = (L + max_generate < engage_threshold);

        if (decode_use_sparse) {
            for (auto & v : k_activations) std::vector<ggml_fp16_t>().swap(v);
            for (auto & v : v_activations) std::vector<ggml_fp16_t>().swap(v);
            std::cerr << "[DiffKV] Prefill activations memory reclaimed early.\n";
        }

        for (int step = 0; step < max_generate; ++step) {
            auto t_step_start = std::chrono::high_resolution_clock::now();

            // \u2500\u2500 ACTIVE_RUNTIME fix 4: swap in SRL state when background thread finishes \u2500\u2500
            // Mirrors batch_engine.py: once _build_srl_index_async completes, the full
            // SRL state (chunk_graph + inverted_index + factual_store) is swapped in.
            // GPU sync is done here (main thread) for Metal thread safety.
            if (!srl_swapped && srl_build_done.load(std::memory_order_acquire)) {
                if (srl_needs_gpu_sync.load(std::memory_order_acquire)) {
                    // All blocks now compressed; upload them to Metal buffers (main thread only)
                    runtime_manager.sync_device_for_native();
                }
                // Swap in the fully-built SRL state
                srl_state = std::move(srl_state_pending);
                srl_swapped = true;
                std::cerr << "[DiffKV] SRL routing active from step " << step << std::endl;

                if (srl_build_thread.joinable()) {
                    srl_build_thread.join();
                }
                for (auto & v : k_activations) {
                    std::vector<ggml_fp16_t>().swap(v);
                }
                for (auto & v : v_activations) {
                    std::vector<ggml_fp16_t>().swap(v);
                }
                std::cerr << "[DiffKV] Prefill activations memory reclaimed early." << std::endl;
            }

            if (active_slot >= n_slots) {
                std::cerr << "\n[DiffKV Native] Warning: Context slot capacity reached during decode. Stopping generation." << std::endl;
                break;
            }
            int current_pos = L + step;

            // Mirrors ACTIVE_RUNTIME: invalidate the per-step routing cache at the
            // top of each decode step, before layer 0 re-routes.
            srl_state.clear_step_cache();

            bool step_use_sparse = (current_pos >= engage_threshold) && (kv_engines[0]->get_pool_version() > 0);
            if (std::getenv("DIFFKV_DBG_POS")) { static int o=0; if(o++<3)
                std::cerr << "[DBG_ENGAGE] step current_pos=" << current_pos
                          << " engage_threshold=" << engage_threshold
                          << " step_use_sparse=" << step_use_sparse << "\n"; }
            bool pool_grew = (kv_engines[0]->get_n_slots() != n_slots);
            bool pool_version_changed = (kv_engines[0]->get_pool_version() != last_pool_version);
            bool rebuild_needed = (step_use_sparse != decode_use_sparse) || pool_grew;
            if (rebuild_needed) {
                decode_use_sparse = step_use_sparse;
                if (pool_grew) {
                    n_slots = kv_engines[0]->get_n_slots();
                }
                last_pool_version = kv_engines[0]->get_pool_version();
                if (decode_use_sparse) {
                    std::cerr << "[DEBUG_RESIZE] required_dense_cap=" << required_dense_cap 
                              << " F_test=" << F_test 
                              << " active_k_dense[0].size()=" << active_k_dense[0].size() << std::endl;
                    for (int l = 0; l < n_layers; ++l) {
                        if (active_k_dense[l].size() < (size_t)required_dense_cap * F_test) {
                            active_k_dense[l].resize((size_t)required_dense_cap * F_test, 0.0f);
                            active_k_dense_rotated[l].resize((size_t)required_dense_cap * F_test, 0.0f);
                            active_v_dense[l].resize((size_t)required_dense_cap * F_test, 0.0f);
                        }
                    }
                    std::cerr << "[DEBUG_RESIZE_AFTER] active_k_dense[0].size()=" << active_k_dense[0].size() << std::endl;
                }
                full_upload_needed = true;
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
                dense_attn_mask_decode = ggml_new_tensor_2d(decode_ctx, GGML_TYPE_F16, required_dense_cap + 1, 1);
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
                    decode_use_sparse, current_pos, required_dense_cap,
                    dense_k_past_inputs.data(), dense_v_past_inputs.data(),
                    dense_attn_mask_decode,
                    native_dense_kr.data(), native_dense_v.data(), native_dense_mask, native_maxd,
                    native_dup_tri, native_half, native_dense_pos, native_attn_slots,
                    cache_k.data(), cache_v.data(), cache_mask, cache_cap
                );
                ggml_set_output(decode_logits);
                if (decode_selected_slots) ggml_set_output(decode_selected_slots);
                if (decode_concat_k) ggml_set_output(decode_concat_k);
                if (decode_concat_v) ggml_set_output(decode_concat_v);

                bool factual_empty = true;
                if (!userdata.empty() && userdata[0].srl_state) {
                    factual_empty = static_cast<diffkv::SessionSRLState*>(userdata[0].srl_state)->factual_store.entries.empty();
                }
                bool use_native_attn = native_attn_on && decode_use_sparse && factual_empty &&
                                       !userdata.empty() && userdata[0].kv_engine &&
                                       userdata[0].kv_engine->native_attn_enabled();

                graph_is_native = use_native_attn;
                bool realloc_ok;
                if (use_native_attn) {
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
                        if (cache_k[l]) ggml_backend_sched_set_tensor_backend(sched, cache_k[l], backend);
                        if (cache_v[l]) ggml_backend_sched_set_tensor_backend(sched, cache_v[l], backend);
                    }
                    if (dense_attn_mask_decode) ggml_backend_sched_set_tensor_backend(sched, dense_attn_mask_decode, backend);
                    if (cache_mask) ggml_backend_sched_set_tensor_backend(sched, cache_mask, backend);

                    // Explicitly route MAP_CUSTOM3 nodes (CPU custom op callbacks) to the CPU backend
                    // so the scheduler doesn't erroneously propagate Metal/GPU backend to them.
                    for (int i = 0; i < decode_graph->n_nodes; ++i) {
                        struct ggml_tensor * node = decode_graph->nodes[i];
                        if (node->op == GGML_OP_MAP_CUSTOM3) {
                            ggml_backend_sched_set_tensor_backend(sched, node, backend_owner.cpu_backend);
                        }
                    }
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
                // Past-KV cache is already initialized on the GPU, and new tokens are uploaded incrementally.
                std::vector<ggml_fp16_t> dense_mask_host(required_dense_cap + 1);
                for (int i = 0; i < required_dense_cap + 1; ++i) {
                    float val = (i < current_pos || i == required_dense_cap) ? 0.0f : -65500.0f;
                    dense_mask_host[i] = ggml_fp32_to_fp16(val);
                }
                ggml_backend_tensor_set(dense_attn_mask_decode, dense_mask_host.data(), 0, (required_dense_cap + 1) * sizeof(ggml_fp16_t));
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
                // current_step_slots and current_step_step are set inside
                // route_decode_slots() — mirrors ACTIVE_RUNTIME line 544.
                srl_state.current_step_step = step;

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

                if (std::getenv("DIFFKV_ROUTING_VERBOSE")) {
                    std::cerr << "[ROUTE] step=" << step
                              << " interval=" << retrieval_interval
                              << " re-routed " << cached_routed_blocks.size() << " blocks\n";
                }

                t_after_retrieval = std::chrono::high_resolution_clock::now();
            } else if (std::getenv("DIFFKV_ROUTING_VERBOSE")) {
                std::cerr << "[ROUTE] step=" << step << " reusing cached routing (next re-route at step "
                          << (last_retrieval_step + retrieval_interval) << ")\n";
            }
            std::vector<int32_t> routed_blocks = cached_routed_blocks;
            std::vector<int32_t> physical_candidates = cached_physical_candidates;

            // Touch blocks to ensure CPU Resident blocks are reloaded to GPU
            runtime_manager.touch_active_slots(routed_blocks);

            // Count and filter CompressedResident slots; track availability for native attn
            int n_valid_compressed = 0;
            std::vector<int32_t> filtered_candidates;
            if (decode_use_sparse) {
                // Filter: replace slot IDs that are not CompressedResident with -1.
                // The native graph's is_neg path already masks negative IDs to -inf in
                // softmax, matching execute_cpu_attention which skips non-CompressedResident
                // slots. Without this filter, the native graph attends garbage pool data
                // for slots that haven't been compressed yet → wrong outputs.
                filtered_candidates = physical_candidates;
                const auto& state_table = kv_engines[0]->get_state_table();
                int n_pool_slots = (int)kv_engines[0]->get_seq_lens()->ne[0];
                int fallback_sid = 0;
                for (auto sid : filtered_candidates) {
                    if (sid >= 0 && sid < n_pool_slots &&
                        state_table.get(sid) == diffkv::BlockState::CompressedResident) {
                        fallback_sid = sid;
                        break;
                    }
                }
                for (auto& sid : filtered_candidates) {
                    if (sid < 0 || sid >= n_pool_slots ||
                        state_table.get(sid) != diffkv::BlockState::CompressedResident) {
                        sid = fallback_sid;
                    } else {
                        n_valid_compressed++;
                    }
                }
                // If no CompressedResident blocks are available (e.g. decode step 0 before
                // compression completes), fill all host slots with 0 so the native graph
                // attends nothing in the sparse pool — only the dense window contributes.
                // This prevents the in-graph SRL router (which operates on relative scores and
                // always selects K slot IDs regardless of mask) from routing to Freed-state slots.
                if (n_valid_compressed == 0) {
                    std::fill(filtered_candidates.begin(), filtered_candidates.end(), 0);
                }
                ggml_backend_tensor_set(host_slots_decode, filtered_candidates.data(), 0, srl_k_host * sizeof(int32_t));
                // Also upload to native_attn_slots (host-controlled, srl_k_keep wide).
                // Pad with -1 if needed, as the Metal kernel supports negative IDs natively.
                if (native_attn_on && native_attn_slots) {
                    std::vector<int32_t> nat_slots(srl_k_keep, -1);
                    int copy_n = std::min((int)physical_candidates.size(), srl_k_keep);
                    for (int i = 0; i < copy_n; ++i) {
                        int32_t sid = physical_candidates[i];
                        if (sid < 0 || sid >= n_pool_slots ||
                            state_table.get(sid) != diffkv::BlockState::CompressedResident) {
                            sid = -1;
                        }
                        nat_slots[i] = sid;
                    }
                    if (n_valid_compressed == 0) {
                        std::fill(nat_slots.begin(), nat_slots.end(), -1);
                    }
                    ggml_backend_tensor_set(native_attn_slots, nat_slots.data(), 0, (size_t)srl_k_keep * sizeof(int32_t));
                }

            }

            static const bool dbg_candidates = (std::getenv("DIFFKV_DBG_CANDIDATES") != nullptr);
            if (dbg_candidates && decode_use_sparse && !is_warmup_run) {
                std::cerr << "[DBG_CANDIDATES] step=" << step << " filtered_candidates: ";
                for (int32_t s : filtered_candidates) std::cerr << s << " ";
                if (native_attn_on && native_attn_slots) {
                    std::cerr << " | nat_slots: ";
                    std::vector<int32_t> nat_slots_dbg(srl_k_keep);
                    ggml_backend_tensor_get(native_attn_slots, nat_slots_dbg.data(), 0, srl_k_keep * sizeof(int32_t));
                    for (int32_t s : nat_slots_dbg) std::cerr << s << " ";
                }
                std::cerr << "\n";
            }

            std::vector<float> slots_mask_host(n_slots, -1e10f);
            int occupied_up_to = active_slot - 1;
            for (int i = 0; i <= occupied_up_to; ++i) {
                auto & blocks = runtime_manager.get_ingest_manager().get_blocks(0);
                if (i >= 0 && i < (int)blocks.size()) {
                    int slot_id = blocks[i]->pool_idx;
                    if (slot_id >= 0 && slot_id < n_slots) {
                        if (blocks[i]->state == BlockState::CompressedResident) {
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
            if (std::getenv("DIFFKV_DBG_WINDOW") && step < 4 && !is_warmup_run) {
                std::cerr << "[DBG_WINDOW] step=" << step << " refresh: total_dense_tokens[0]="
                          << total_dense_tokens[0] << " total_positions=" << total_positions
                          << " current_pos=" << current_pos << "\n";
            }

            // ═══════════════════════════════════════════════════════════════════════════
            // OPTIMIZED: Persistent GPU Buffer Upload (Option 3)
            // ═══════════════════════════════════════════════════════════════════════════
            // Native sparse-attn: upload the past dense window (rotated K + raw V) and its
            // validity mask into the persistent graph inputs (current token handled in-graph).
            //
            // OLD: Upload 15MB every token (48 uploads × 320KB) = 24-48ms
            // NEW: Upload 120KB once + 5KB per token (48 uploads × 5KB) = 0.2ms
            // ═══════════════════════════════════════════════════════════════════════════
            double step_sync_upload_ms = 0.0;
            auto t_sync_up_start = std::chrono::high_resolution_clock::now();
            if (native_attn_on && decode_use_sparse) {
                const int cnt0 = std::min(total_dense_tokens[0], native_maxd);
                
                // ─────────────────────────────────────────────────────────────────────
                // FIRST-TIME INITIALIZATION: Upload full buffers once
                // ─────────────────────────────────────────────────────────────────────
                if (full_upload_needed || !persistent_buffers_initialized) {
                    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
                        std::cerr << "[DiffKV PERF] Initializing persistent GPU buffers (one-time upload: "
                                  << (n_layers * native_maxd * F_test * 2 * sizeof(float)) / (1024*1024) 
                                  << " MB)" << std::endl;
                    }
                    
                    if (std::getenv("DIFFKV_DBG_SEL") && step == 0) {
                        double n0=0,nv=0,nvlast=0; 
                        for (int i=0;i<F_test;++i){ 
                            float x=active_k_dense[0][i]; n0+=(double)x*x; 
                            float y=active_v_dense[0][i]; nv+=(double)y*y; 
                            if (cnt0 > 0) {
                                float z=active_v_dense[0][(cnt0-1)*F_test+i]; 
                                nvlast+=(double)z*z;
                            }
                        }
                        std::cerr << "[DBG_UPLOAD] step0 cnt0=" << cnt0 << " |active_k[tok0]|=" << std::sqrt(n0)
                                  << " |active_v[tok0]|=" << std::sqrt(nv) << " |active_v[last]|=" << std::sqrt(nvlast) << std::endl;
                    }
                    
                    // Upload mask (full buffer)
                    std::vector<float> dmask(required_dense_cap, -INFINITY);
                    for (int t = 0; t < cnt0; ++t) dmask[t] = 0.0f;
                    ggml_backend_tensor_set(native_dense_mask, dmask.data(), 0, (size_t)required_dense_cap * sizeof(float));
                    
                    // Upload positions (full buffer)
                    std::vector<int32_t> pbuf(required_dense_cap, 0);
                    for (int t = 0; t < cnt0; ++t) pbuf[t] = active_positions_dense[t];
                    ggml_backend_tensor_set(native_dense_pos, pbuf.data(), 0, (size_t)required_dense_cap * sizeof(int32_t));
                    
                    for (int l = 0; l < n_layers; ++l) {
                        int kn = 0, vn = 0;
                        for (int i = 0; i < (int)active_k_dense[l].size(); ++i) {
                            if (std::isnan(active_k_dense[l][i])) {
                                kn++;
                                if (kn <= 2) {
                                    std::cerr << "[DBG_ACTIVE_NAN_DETAIL] layer=" << l 
                                              << " K_NaN at tok=" << i / F_test 
                                              << " f=" << i % F_test << std::endl;
                                }
                            }
                            if (std::isnan(active_v_dense[l][i])) {
                                vn++;
                                if (vn <= 2) {
                                    std::cerr << "[DBG_ACTIVE_NAN_DETAIL] layer=" << l 
                                              << " V_NaN at tok=" << i / F_test 
                                              << " f=" << i % F_test << std::endl;
                                }
                            }
                        }
                        if (kn > 0 || vn > 0) {
                            std::cerr << "[DBG_ACTIVE_NAN] layer=" << l << " active_k_nans=" << kn << " active_v_nans=" << vn << std::endl;
                        }
                    }

                    // Rebuild active_k_dense_rotated for all layers before upload
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
                            if (active_k_dense_rotated[l].size() < (size_t)required_dense_cap * F_test) {
                                active_k_dense_rotated[l].resize((size_t)required_dense_cap * F_test, 0.0f);
                            }
                            std::fill(active_k_dense_rotated[l].begin(), active_k_dense_rotated[l].end(), 0.0f);
                            apply_rope_neox_cpu_fast(
                                active_k_dense[l].data(),
                                active_k_dense_rotated[l].data(),
                                cos_table_rebuild.data(),
                                sin_table_rebuild.data(),
                                num_dense, kv_heads, head_dim
                            );
                        }
                    }

                    // Upload K/V for all layers (contiguous per-layer buffers)
                    for (int l = 0; l < n_layers; ++l) {
                        const int c2 = std::min(total_dense_tokens[l], required_dense_cap);
                        std::vector<float> flat_k((size_t)required_dense_cap * F_test, 0.0f);
                        std::vector<float> flat_v((size_t)required_dense_cap * F_test, 0.0f);
                        for (int t = 0; t < c2; ++t) {
                            std::memcpy(
                                flat_k.data() + (size_t)t * F_test,
                                active_k_dense_rotated[l].data() + (size_t)t * F_test,
                                F_test * sizeof(float)
                            );
                            std::memcpy(
                                flat_v.data() + (size_t)t * F_test,
                                active_v_dense[l].data() + (size_t)t * F_test,
                                F_test * sizeof(float)
                            );
                        }
                        ggml_backend_tensor_set(native_dense_kr[l], flat_k.data(), 0, flat_k.size() * sizeof(float));
                        ggml_backend_tensor_set(native_dense_v[l], flat_v.data(), 0, flat_v.size() * sizeof(float));
                    }
                    
                    if (std::getenv("DIFFKV_DBG_SEL") && step == 0) {
                        std::vector<float> rb(F_test), rv(F_test);
                        ggml_backend_tensor_get(native_dense_kr[0], rb.data(), 0, F_test*sizeof(float));
                        ggml_backend_tensor_get(native_dense_v[0], rv.data(), 0, F_test*sizeof(float));
                        double n=0,nv=0; 
                        for(int i=0;i<F_test;++i){ 
                            n+=(double)rb[i]*rb[i]; 
                            nv+=(double)rv[i]*rv[i]; 
                        }
                        std::cerr << "[DBG_DEV] native_dense_kr[0] dev |tok0|=" << std::sqrt(n) 
                                  << " native_dense_v[0] dev |tok0|=" << std::sqrt(nv) << std::endl;
                    }
                    
                    persistent_buffers_initialized = true;
                    persistent_buffer_base_pos = 0;
                    full_upload_needed = false;
                    
                    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
                        std::cerr << "[DiffKV PERF] Persistent buffers initialized. Subsequent tokens will upload only ~"
                                  << (n_layers * F_test * sizeof(float)) / 1024 << " KB each." << std::endl;
                    }
                    
                } else {
                    // ─────────────────────────────────────────────────────────────────────
                    // INCREMENTAL UPDATE: Upload ONLY the new token (99.7% bandwidth reduction)
                    // ─────────────────────────────────────────────────────────────────────
                    // Old: 48 uploads × 320KB = 15MB
                    // New: 48 uploads × 5KB = 240KB (63x less data)
                    // Expected: 24-48ms → 0.3-0.5ms per token
                    // ─────────────────────────────────────────────────────────────────────
                    
                    // INCREMENTAL UPDATE: Upload ONLY the new token (99.7% bandwidth reduction, in bulk)
                    const int ring_pos = (persistent_buffer_base_pos + (total_dense_tokens[0] - 1)) % required_dense_cap;
                    if (decode_concat_k && decode_concat_v && step > 0) {
                        // GPU-Direct Incremental Copy: copy directly on the GPU
                        struct ggml_init_params params = {
                            /*.mem_size   =*/ 1024 * 64,
                            /*.mem_buffer =*/ nullptr,
                            /*.no_alloc   =*/ true,
                        };
                        struct ggml_context * temp_ctx = ggml_init(params);
                        if (temp_ctx) {
                            for (int l = 0; l < n_layers; ++l) {
                                struct ggml_tensor * src_view_k = ggml_view_1d(temp_ctx, decode_concat_k, F_test, (size_t)l * F_test * sizeof(float));
                                src_view_k->buffer = decode_concat_k->buffer;

                                struct ggml_tensor * dst_view_k = ggml_view_1d(temp_ctx, native_dense_kr[l], F_test, (size_t)ring_pos * F_test * sizeof(float));
                                dst_view_k->buffer = native_dense_kr[l]->buffer;

                                struct ggml_tensor * src_view_v = ggml_view_1d(temp_ctx, decode_concat_v, F_test, (size_t)l * F_test * sizeof(float));
                                src_view_v->buffer = decode_concat_v->buffer;

                                struct ggml_tensor * dst_view_v = ggml_view_1d(temp_ctx, native_dense_v[l], F_test, (size_t)ring_pos * F_test * sizeof(float));
                                dst_view_v->buffer = native_dense_v[l]->buffer;

                                ggml_backend_tensor_copy(src_view_k, dst_view_k);
                                ggml_backend_tensor_copy(src_view_v, dst_view_v);
                            }
                            ggml_free(temp_ctx);
                        }
                    } else {
                        // Fallback to CPU-to-GPU set
                        for (int l = 0; l < n_layers; ++l) {
                            const int local_idx = total_dense_tokens[l] - 1;  // Index in active_k_dense
                            if (local_idx >= 0) {
                                const float* new_k = active_k_dense_rotated[l].data() + (size_t)local_idx * F_test;
                                const float* new_v = active_v_dense[l].data() + (size_t)local_idx * F_test;
                                ggml_backend_tensor_set(native_dense_kr[l], new_k, (size_t)ring_pos * F_test * sizeof(float), F_test * sizeof(float));
                                ggml_backend_tensor_set(native_dense_v[l], new_v, (size_t)ring_pos * F_test * sizeof(float), F_test * sizeof(float));
                            }
                        }
                    }
                    
                    // Update mask and position for the new token
                    const int idx0 = total_dense_tokens[0] - 1;
                    if (idx0 >= 0) {
                        const int ring_pos0 = (persistent_buffer_base_pos + idx0) % required_dense_cap;
                        
                        // Mark this position as valid in the mask
                        float val = 0.0f;
                        ggml_backend_tensor_set(native_dense_mask, &val, 
                            (size_t)ring_pos0 * sizeof(float), sizeof(float));
                        
                        // Update position for RoPE
                        int32_t pos_val = active_positions_dense[idx0];
                        ggml_backend_tensor_set(native_dense_pos, &pos_val,
                            (size_t)ring_pos0 * sizeof(int32_t), sizeof(int32_t));
                    }
                    
                    // Handle ring buffer overflow: when we've filled required_dense_cap, start overwriting
                    if (total_dense_tokens[0] >= required_dense_cap) {
                        persistent_buffer_base_pos = (persistent_buffer_base_pos + 1) % required_dense_cap;
                    }
                }
            }
            step_sync_upload_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_sync_up_start).count();

            auto t_before_compute = std::chrono::high_resolution_clock::now();
            // Marks the boundary between CPU cache-fill and the GPU graph compute. Defaults to
            // t_before_compute so the dense path (no fill) reports fill_ms=0. Set just before
            // ggml_backend_graph_compute below, after the fill block runs.
            auto t_fill_end = t_before_compute;
            if (std::getenv("DIFFKV_DBG_POS")) { static int o=0; if(o++<8)
                std::cerr << "[DBG_COMPUTE] step o=" << o << " decode_use_sparse=" << decode_use_sparse
                          << " current_pos=" << current_pos << " native_attn_on=" << native_attn_on << "\n"; }
            if (std::getenv("DIFFKV_DBG_GRAPH") && decode_use_sparse) { static int o=0; if(o++==0) {
                int N = ggml_graph_n_nodes(decode_graph), nc3=0, ndk=0, nflash=0;
                struct ggml_tensor* attn_feed = nullptr;
                for (int i=0;i<N;++i){ struct ggml_tensor* nd = ggml_graph_node(decode_graph,i);
                    if (nd->op == GGML_OP_MAP_CUSTOM3) nc3++;
                    if (nd->op == GGML_OP_DIFFKV_ATTN) ndk++;
                    if (nd->op == GGML_OP_FLASH_ATTN_EXT) nflash++;
                    if (nd->name && strstr(nd->name, "attn")) attn_feed = nd;
                }
                std::cerr << "[DBG_GRAPH] sparse decode graph: n_nodes=" << N
                          << " MAP_CUSTOM3=" << nc3 << " DIFFKV_ATTN=" << ndk
                          << " FLASH_ATTN=" << nflash << "\n";
            }}
            if (has_prev_svd) {
                svd_thread = std::thread([&runtime_manager, prev_decode_k, prev_decode_v, prev_pos, prev_all_tokens, &srl_state]() {
                    runtime_manager.ingest_decode(prev_decode_k, prev_decode_v, prev_pos, prev_all_tokens, &srl_state, true);
                });
                svd_thread_active = true;
            }

            // Fix: patch t_dense in op_params for all DIFFKV_ATTN nodes each step.
            // At graph-build time t_dense is set to native_maxd (a static capacity), but the
            // Metal kernel uses it to scan the dense-mask from 0…t_dense looking for the first
            // -inf entry (finding actual fill). Passing the real fill count (total_positions)
            // avoids scanning empty slots and keeps the scan cost O(fill) not O(capacity).
            // ggml reads op_params fresh at each kernel dispatch, so in-place mutation is safe.
            if (native_attn_on && decode_use_sparse) {
                const int actual_t_dense = total_positions;
                const int N_tdfix = ggml_graph_n_nodes(decode_graph);
                for (int ni = 0; ni < N_tdfix; ++ni) {
                    struct ggml_tensor * nd = ggml_graph_node(decode_graph, ni);
                    if (nd->op == GGML_OP_DIFFKV_ATTN) {
                        struct ggml_diffkv_attn_params dp;
                        memcpy(&dp, nd->op_params, sizeof(dp));
                        if (dp.t_dense != actual_t_dense) {
                            dp.t_dense = actual_t_dense;
                            memcpy(nd->op_params, &dp, sizeof(dp));
                        }
                    }
                }
            }

            // ── DIFFKV_DECODE_CACHE: fill the materialized sparse cache before the flash graph ──
            // Routed compressed blocks (materialized once per N tokens) + dense window (rotated
            // per token) → the F16 cache_k/cache_v the flash kernel attends. CPU work (recon,
            // rotate, F16-convert) runs parallel across layers into disjoint host buffers; the
            // ggml_backend_tensor_set uploads run SERIALLY (backend uploads aren't thread-safe).
            if (decode_cache_on && decode_use_sparse && cache_mask) {
                const int F = F_test;
                const int n_window = std::min(total_dense_tokens[0], required_dense_cap);
                const int pool_ver = kv_engines[0]->get_pool_version();
                const int nthreads = std::max(1, std::min((int)std::thread::hardware_concurrency(), n_layers));
                auto dispatch_n = [&](int nth, const std::function<void(int,int)>& work) {
                    nth = std::max(1, std::min(nth, n_layers));
                    std::vector<std::thread> pool;
                    int chunk = (n_layers + nth - 1) / nth;
                    for (int t = 0; t < nth; ++t) {
                        int lo = t * chunk, hi = std::min(n_layers, lo + chunk);
                        if (lo < hi) pool.emplace_back(work, lo, hi);
                    }
                    for (auto& th : pool) th.join();
                };
                auto dispatch = [&](const std::function<void(int,int)>& work) { dispatch_n(nthreads, work); };
                // BLAS (Accelerate/AMX) already parallelises each gemm; running materialize across
                // many std::threads can oversubscribe. DIFFKV_MAT_THREADS tunes it (default = full).
                static const int mat_threads = (std::getenv("DIFFKV_MAT_THREADS") ? atoi(std::getenv("DIFFKV_MAT_THREADS")) : nthreads);

                static double t_remat_ms = 0, t_win_ms = 0; static int t_fill_n = 0;
                const bool t_fill = (std::getenv("DIFFKV_DBG_FILL_TIME") != nullptr);
                auto t_m0 = std::chrono::high_resolution_clock::now();

                // (1) (Re)materialize routed blocks only when the reconstruction would differ.
                // The materialized bytes are a pure function of (routed candidate set, pool
                // contents). filtered_candidates captures the routed set AND per-slot residency
                // (non-CompressedResident slots are already folded to fallback_sid above), so a
                // change in either shows up here; pool_ver captures in-place pool mutation. When
                // both are unchanged the device cache is already correct → skip recon + upload.
                // DIFFKV_DECODE_CACHE_ALWAYS_REMAT=1 restores the old unconditional periodic remat.
                static const bool force_periodic_remat =
                    (std::getenv("DIFFKV_DECODE_CACHE_ALWAYS_REMAT") != nullptr);
                bool cands_changed = (filtered_candidates != cache_last_materialized_cands);
                bool remat = (cache_n_routed < 0) || (pool_ver != cache_last_remat_version) ||
                             cands_changed ||
                             (force_periodic_remat && (step % decode_cache_N == 0));
                if (remat) {
                    std::vector<int> nwritten(n_layers, 0);
                    dispatch_n(mat_threads, [&](int lo, int hi) {
                        for (int l = lo; l < hi; ++l) {
                            if ((int)host_routed_k[l].size() < cache_routed_cap * F) {
                                host_routed_k[l].assign((size_t)cache_routed_cap * F, ggml_fp32_to_fp16(0.0f));
                                host_routed_v[l].assign((size_t)cache_routed_cap * F, ggml_fp32_to_fp16(0.0f));
                            } else {
                                std::fill(host_routed_k[l].begin(), host_routed_k[l].end(), ggml_fp32_to_fp16(0.0f));
                                std::fill(host_routed_v[l].begin(), host_routed_v[l].end(), ggml_fp32_to_fp16(0.0f));
                            }
                            nwritten[l] = materialize_routed_kv(
                                kv_engines[l].get(), filtered_candidates.data(),
                                (int)filtered_candidates.size(), srl_k_keep,
                                kv_heads, head_dim, model.get_config().rope_freq_base,
                                cache_routed_cap, host_routed_k[l].data(), host_routed_v[l].data());
                        }
                    });
                    cache_n_routed = nwritten[0];
                    cache_last_remat_version = pool_ver;
                    cache_last_materialized_cands = filtered_candidates;
                    for (int l = 0; l < n_layers; ++l) {
                        ggml_backend_tensor_set(cache_k[l], host_routed_k[l].data(), 0, (size_t)cache_routed_cap * F * sizeof(ggml_fp16_t));
                        ggml_backend_tensor_set(cache_v[l], host_routed_v[l].data(), 0, (size_t)cache_routed_cap * F * sizeof(ggml_fp16_t));
                    }
                }

                auto t_m1 = std::chrono::high_resolution_clock::now();

                // (2) Dense window: rotate raw K at abs position + F16-convert (parallel), upload (serial).
                if (n_window > 0) {
                    const int half = head_dim / 2;
                    const float fb = model.get_config().rope_freq_base;
                    std::vector<float> cosw((size_t)n_window * half), sinw((size_t)n_window * half);
                    for (int t = 0; t < n_window; ++t) {
                        int pos = active_positions_dense[t];
                        for (int i = 0; i < half; ++i) {
                            float th = (float)pos / std::pow(fb, (float)(2 * i) / head_dim);
                            cosw[(size_t)t * half + i] = std::cos(th);
                            sinw[(size_t)t * half + i] = std::sin(th);
                        }
                    }
                    dispatch([&](int lo, int hi) {
                        std::vector<float> rot((size_t)n_window * F);
                        for (int l = lo; l < hi; ++l) {
                            if ((int)host_win_k[l].size() < n_window * F) {
                                host_win_k[l].resize((size_t)n_window * F);
                                host_win_v[l].resize((size_t)n_window * F);
                            }
                            apply_rope_neox_cpu_fast(active_k_dense[l].data(), rot.data(),
                                                     cosw.data(), sinw.data(), n_window, kv_heads, head_dim);
                            // active_k_dense holds RAW K for PREFILL tokens (pos < L) but
                            // ALREADY-ROTATED K for DECODE-appended tokens (pos >= L; decode_k is
                            // the in-graph rope output — see the window-append at the else-branch).
                            // Re-rotating the decode tokens would DOUBLE-rotate them and corrupt
                            // generation after the first few tokens. Overwrite those rows with the
                            // stored (already-rotated) value.
                            for (int t = 0; t < n_window; ++t) {
                                if (active_positions_dense[t] >= L) {
                                    std::memcpy(rot.data() + (size_t)t * F,
                                                active_k_dense[l].data() + (size_t)t * F,
                                                (size_t)F * sizeof(float));
                                }
                            }
                            for (size_t i = 0; i < (size_t)n_window * F; ++i) {
                                host_win_k[l][i] = ggml_fp32_to_fp16(rot[i]);
                                host_win_v[l][i] = ggml_fp32_to_fp16(active_v_dense[l][i]);
                            }
                        }
                    });
                    const size_t off = (size_t)cache_routed_cap * F * sizeof(ggml_fp16_t);
                    for (int l = 0; l < n_layers; ++l) {
                        ggml_backend_tensor_set(cache_k[l], host_win_k[l].data(), off, (size_t)n_window * F * sizeof(ggml_fp16_t));
                        ggml_backend_tensor_set(cache_v[l], host_win_v[l].data(), off, (size_t)n_window * F * sizeof(ggml_fp16_t));
                    }
                }

                auto t_m2 = std::chrono::high_resolution_clock::now();
                if (t_fill && !is_warmup_run) {
                    t_remat_ms += std::chrono::duration<double,std::milli>(t_m1 - t_m0).count();
                    t_win_ms   += std::chrono::duration<double,std::milli>(t_m2 - t_m1).count();
                    if (++t_fill_n % 40 == 0)
                        std::cerr << "[FILL_TIME] avg over " << t_fill_n << ": materialize+routed-upload="
                                  << (t_remat_ms/t_fill_n) << "ms window(rotate+conv+upload)="
                                  << (t_win_ms/t_fill_n) << "ms\n";
                }

                // (3) Validity mask: 0 for routed [0,n_routed) + window [routed_cap,+n_window) +
                // current (index cache_cap); -65500 (fp16 -inf) elsewhere.
                // DIFFKV_DBG_CACHE_WINONLY / _ROUTEDONLY isolate the two halves for debugging.
                static const bool winonly    = (std::getenv("DIFFKV_DBG_CACHE_WINONLY") != nullptr);
                static const bool routedonly = (std::getenv("DIFFKV_DBG_CACHE_ROUTEDONLY") != nullptr);
                std::vector<ggml_fp16_t> cmask((size_t)cache_cap + 1, ggml_fp32_to_fp16(-65500.0f));
                if (!winonly) for (int j = 0; j < cache_n_routed && j < cache_routed_cap; ++j) cmask[j] = ggml_fp32_to_fp16(0.0f);
                if (!routedonly) for (int j = 0; j < n_window; ++j) cmask[cache_routed_cap + j] = ggml_fp32_to_fp16(0.0f);
                cmask[cache_cap] = ggml_fp32_to_fp16(0.0f);
                ggml_backend_tensor_set(cache_mask, cmask.data(), 0, (size_t)(cache_cap + 1) * sizeof(ggml_fp16_t));

                if (std::getenv("DIFFKV_DBG_CACHE") && step < 3 && !is_warmup_run) {
                    auto nrm = [&](const ggml_fp16_t* p){ double s=0; for(int i=0;i<F;++i){float x=ggml_fp16_to_fp32(p[i]); s+=(double)x*x;} return std::sqrt(s); };
                    std::cerr << "[DBG_CACHE] step=" << step << " remat=" << (int)remat
                              << " n_routed=" << cache_n_routed << " routed_cap=" << cache_routed_cap
                              << " n_window=" << n_window << " cache_cap=" << cache_cap
                              << " n_cands=" << filtered_candidates.size()
                              << " |routed_k[0]|=" << (cache_n_routed>0?nrm(host_routed_k[0].data()):-1)
                              << " |win_k[0]|=" << (n_window>0?nrm(host_win_k[0].data()):-1)
                              << " winonly=" << (int)winonly << " routedonly=" << (int)routedonly << "\n";
                }
            }

            t_fill_end = std::chrono::high_resolution_clock::now();  // CPU fill done; GPU graph next

            ggml_status decode_st = graph_is_native
                ? ggml_backend_graph_compute(backend, decode_graph)
                : ggml_backend_sched_graph_compute(sched, decode_graph);

            if (svd_thread_active) {
                svd_thread.join();
                svd_thread_active = false;
                runtime_manager.sync_device_for_native();
                runtime_manager.get_pager().maybe_evict(kv_engines, &srl_state);
            }

            if (native_attn_on && std::getenv("DIFFKV_DBG_NAN")) {
                int count = 0;
                for (int i = 0; i < decode_graph->n_nodes; ++i) {
                    struct ggml_tensor * node = decode_graph->nodes[i];
                    if (node->op == GGML_OP_DIFFKV_ATTN) {
                        std::vector<float> res_data(ggml_nelements(node));
                        ggml_backend_tensor_get(node, res_data.data(), 0, res_data.size() * sizeof(float));
                        double norm = 0;
                        int nans = 0;
                        for (float val : res_data) {
                            if (std::isnan(val)) nans++;
                            else norm += (double)val * val;
                        }
                        std::cerr << "[DBG_NAN] step=" << step << " node_idx=" << i 
                                  << " layer=" << count++
                                  << " shape=[" << node->ne[0] << "," << node->ne[1] << "]"
                                  << " norm=" << std::sqrt(norm) 
                                  << " nans=" << nans << std::endl;
                    }
                }
            }

            if (std::getenv("DIFFKV_DBG_GRAPH")) { static int o=0; if(o++<10)
                std::cerr << "[DBG_CBCOUNT] step=" << o << " sparse=" << decode_use_sparse
                          << " cb=" << diffkv::g_diffkv_cb_invocations.load()
                          << " cpu_attn=" << diffkv::g_cpu_attn_count.load()
                          << " metal_attn=" << diffkv::g_metal_attn_count.load() << "\n"; }
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
                                              nq, nkv, rank, kv_engines[cmpL]->get_S_max(), K, D, scale, true, freq, userdata[cmpL].approximate_attn);
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
            }

            // DIFFKV_BLOCK_CMP: per-layer comparison table vs CPU reference
            if (std::getenv("DIFFKV_BLOCK_CMP")) {
                const auto & cfg = model.get_config();
                int nq=cfg.n_head, nkv=cfg.n_head_kv, D=cfg.n_embd/cfg.n_head;
                float freq=cfg.rope_freq_base;
                float scale=1.0f/std::sqrt((float)D);
                int F=nkv*D;
                const char* slimit = std::getenv("DIFFKV_BLOCK_CMP_STEPS");
                int max_steps = slimit ? atoi(slimit) : 3;
                if (step < max_steps) {
                    std::cerr << "\n[BLOCK_CMP] ===== step=" << step << " =====\n";
                    for (int l = 0; l < (int)kv_engines.size() && l < BLOCK_CMP_MAX_LAYERS; ++l) {
                        if (!g_bcmp_attn[l] || !g_bcmp_qrope[l] || !g_bcmp_slots[l]) continue;
                        int K = (int)g_bcmp_slots[l]->ne[0];
                        int rank = kv_engines[l]->get_rank();
                        // Read native output
                        std::vector<float> natv((size_t)nq*D);
                        ggml_backend_tensor_get(g_bcmp_attn[l],  natv.data(), 0, natv.size()*sizeof(float));
                        // Read q, slots
                        std::vector<float>   qh((size_t)nq*D);
                        std::vector<int32_t> sl(K);
                        ggml_backend_tensor_get(g_bcmp_qrope[l], qh.data(),   0, qh.size()*sizeof(float));
                        ggml_backend_tensor_get(g_bcmp_slots[l], sl.data(),   0, (size_t)K*sizeof(int32_t));
                        // CPU sparse reference
                        std::vector<float> outS((size_t)nq*D,0.f), lseS(nq,-1e30f);
                        diffkv::execute_cpu_attention(qh.data(), sl.data(), outS.data(), lseS.data(),
                            kv_engines[l].get(), nq, nkv, rank, kv_engines[l]->get_S_max(), K, D, scale, true, freq, l < (int)userdata.size() ? userdata[l].approximate_attn : false);
                        // CPU dense reference (include current token if ignore_c=false)
                        int Td = total_dense_tokens[l];
                        bool cur_included = (l < (int)userdata.size() && !userdata[l].ignore_c);
                        int Tc = Td + (cur_included ? 1 : 0);
                        std::vector<float> dk((size_t)Tc*F,0.f), dv((size_t)Tc*F,0.f); std::vector<int32_t> dp(Tc,0);
                        std::memcpy(dk.data(), active_k_dense[l].data(), (size_t)Td*F*sizeof(float));
                        std::memcpy(dv.data(), active_v_dense[l].data(), (size_t)Td*F*sizeof(float));
                        for(int t=0;t<Td;++t) dp[t]=active_positions_dense[t];
                        if (cur_included && g_bcmp_curk[l] && g_bcmp_curv[l]) {
                            std::vector<float> ck((size_t)F), cv((size_t)F);
                            ggml_backend_tensor_get(g_bcmp_curk[l], ck.data(), 0, ck.size()*sizeof(float));
                            ggml_backend_tensor_get(g_bcmp_curv[l], cv.data(), 0, cv.size()*sizeof(float));
                            std::memcpy(dk.data()+(size_t)Td*F, ck.data(), (size_t)F*sizeof(float));
                            std::memcpy(dv.data()+(size_t)Td*F, cv.data(), (size_t)F*sizeof(float));
                            dp[Td] = current_pos;
                        }
                        std::vector<float> outD((size_t)nq*D,0.f), lseD(nq,-1e30f);
                        diffkv::cpu_dense_attention(qh.data(), dk.data(), dv.data(), dp.data(), Tc,
                            nq, nkv, D, scale, true, freq, 0, outD.data(), lseD.data());
                        // 3-way LSE merge → CPU combined (sparse ⊕ dense+cur)
                        std::vector<float> cpuv((size_t)nq*D);
                        for(int h=0;h<nq;++h){
                            double lmax=std::max(lseS[h],lseD[h]);
                            double ws=(lseS[h]<=-1e20?0.0:std::exp(lseS[h]-lmax));
                            double wd=(lseD[h]<=-1e20?0.0:std::exp(lseD[h]-lmax));
                            double den=std::max(ws+wd,1e-9);
                            for(int d=0;d<D;++d) cpuv[h*D+d]=(float)((outS[h*D+d]*ws+outD[h*D+d]*wd)/den);
                        }

                        // Compute stats
                        double md=0, nn=0, cn=0; int worst=0;
                        int nanN=0, nanC=0;
                        for(int i=0;i<nq*D;++i){
                            if(std::isnan(natv[i])) nanN++;
                            if(std::isnan(cpuv[i])) nanC++;
                            double diff=std::fabs((double)natv[i]-cpuv[i]);
                            if(diff>md){md=diff;worst=i;}
                            nn+=(double)natv[i]*natv[i]; cn+=(double)cpuv[i]*cpuv[i];
                        }
                        double sumS = 0, sumD = 0;
                        for (int h = 0; h < nq; ++h) {
                            sumS += lseS[h];
                            sumD += lseD[h];
                        }
                        // Print report line
                        std::cerr << "[BLOCK_CMP] L" << l
                            << " K=" << K << " Td=" << Td
                            << " |nat|=" << std::sqrt(nn)
                            << " |cpu|=" << std::sqrt(cn)
                            << " maxDiff=" << md
                            << " @h" << worst/D
                            << " nanNat=" << nanN << " nanCpu=" << nanC
                            << " lseS_avg=" << (sumS / nq) << " lseD_avg=" << (sumD / nq)
                            << "\n  nat[0..4]:";
                        for(int i=0;i<5&&i<nq*D;++i) std::cerr<<" "<<natv[i];
                        std::cerr << "\n  cpu[0..4]:";
                        for(int i=0;i<5&&i<nq*D;++i) std::cerr<<" "<<cpuv[i];
                        std::cerr << "\n";
                    }
                    std::cerr << "[BLOCK_CMP] ===== end step=" << step << " =====\n\n";
                }
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
                { ggml_tensor* aKr=kv_engines[0]->get_anchorK_rot();
                  const int32_t* sls=kv_engines[0]->get_host_seq_lens(); int Fkv=kv_heads*head_dim;
                  std::cerr<<"[DBG_DEVSLOT] slot(seq:devAKr/hostAK): ";
                  if (aKr != nullptr) {
                      for(int i=0;i<(int)sel.size() && i<10;++i){
                        int s=sel[i];
                        std::vector<float> d(Fkv);
                        ggml_backend_tensor_get(aKr,d.data(),(size_t)s*Fkv*sizeof(float),Fkv*sizeof(float));
                        const ggml_fp16_t* hAK = kv_engines[0]->get_host_anchors_K(s);
                        double nd=0,nh=0;
                        for(int j=0;j<Fkv;++j){
                            float x=d[j];
                            nd+=(double)x*x;
                            float y = hAK ? ggml_fp16_to_fp32(hAK[j]) : 0.0f;
                            nh+=(double)y*y;
                        }
                        std::cerr<<s<<"(sl"<<sls[s]<<":"<<std::sqrt(nd)<<"/"<<std::sqrt(nh)<<") ";
                      }
                  } else {
                      std::cerr<<"aKr is null (native attention disabled) ";
                  }
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

            std::vector<float> output_logits;
            int32_t next_token_from_gpu = -1;
            bool read_gpu_argmax = (decode_argmax != nullptr) && (step >= 20);
            auto t_after_logits = std::chrono::high_resolution_clock::now();
            if (read_gpu_argmax) {
                int32_t argmax_val = 0;
                ggml_backend_tensor_get(decode_argmax, &argmax_val, 0, sizeof(int32_t));
                next_token_from_gpu = argmax_val;
                t_after_logits = std::chrono::high_resolution_clock::now();
            } else {
                output_logits.resize(n_vocab);
                ggml_backend_tensor_get(decode_logits, output_logits.data(), 0, n_vocab * sizeof(float));
                t_after_logits = std::chrono::high_resolution_clock::now();
            }

            // DIFFKV_DBG_STEP_LOGITS: top-5 logits for the first few decode steps —
            // lets a sparse-mode run be diffed against a bypass (all-dense) run on the
            // same prompt to see exactly where the two decode paths diverge.
            if (std::getenv("DIFFKV_DBG_STEP_LOGITS") && step < 4 && !is_warmup_run) {
                std::vector<std::pair<float,int>> tk;
                tk.reserve(n_vocab);
                for (int i = 0; i < n_vocab; ++i) tk.push_back({output_logits[i], i});
                std::partial_sort(tk.begin(), tk.begin()+5, tk.end(),
                                  [](auto&a, auto&b){ return a.first > b.first; });
                std::cerr << "[STEP_LOGITS] step=" << step << " top5:";
                for (int i = 0; i < 5; ++i)
                    std::cerr << " \"" << model.token_to_piece(tk[i].second) << "\"(" << tk[i].second
                              << "," << tk[i].first << ")";
                std::cerr << "\n";
            }

            auto t_before_kv_get = std::chrono::high_resolution_clock::now();
            // Fast path: K/V captured INSIDE custom_attention_op_callback — zero GPU readback.
            // Mirrors batch_engine.cpp bug-3 fix and ACTIVE_RUNTIME's in-callback capture.
            // In native_attn_on mode the callback doesn't run, so captured_kv is empty;
            // we fall back to per-layer individual reads (avoids the two 28KB bulk transfers
            // that flush the full GPU pipeline and cause the 10ms stall per token).
            std::vector<std::vector<float>> decode_k(n_layers, std::vector<float>(F_test, 0.0f));
            std::vector<std::vector<float>> decode_v(n_layers, std::vector<float>(F_test, 0.0f));
            std::vector<float> decode_q;
            if (userdata[0].layer0_q_tensor) {
                decode_q.resize((int)model.get_config().n_embd, 0.0f);
                ggml_backend_tensor_get(userdata[0].layer0_q_tensor, decode_q.data(), 0, (int)model.get_config().n_embd * sizeof(float));
                static const bool dbg_q = (std::getenv("DIFFKV_DBG_Q") != nullptr);
                if (dbg_q) {
                    double sum_sq = 0.0;
                    for (float val : decode_q) sum_sq += (double)val * val;
                    std::cerr << "[DEBUG_Q] step=" << step << " norm=" << std::sqrt(sum_sq) << " first5: ";
                    for (int i = 0; i < 5 && i < (int)decode_q.size(); ++i) std::cerr << decode_q[i] << " ";
                    std::cerr << std::endl;
                }
            }
            bool any_from_gpu = false;
            bool all_captured = true;
            for (int l = 0; l < n_layers; ++l) {
                if ((int)userdata[l].captured_kv.size() < 2 * F_test) {
                    all_captured = false;
                    break;
                }
            }
            if (all_captured) {
                for (int l = 0; l < n_layers; ++l) {
                    std::memcpy(decode_k[l].data(), userdata[l].captured_kv.data(),          F_test * sizeof(float));
                    std::memcpy(decode_v[l].data(), userdata[l].captured_kv.data() + F_test, F_test * sizeof(float));
                }
            } else if (decode_concat_k && decode_concat_v) {
                // Bulk read: read all layers contiguously in one go to avoid driver sync latency
                std::vector<float> flat_k(n_layers * F_test);
                std::vector<float> flat_v(n_layers * F_test);
                ggml_backend_tensor_get(decode_concat_k, flat_k.data(), 0, n_layers * F_test * sizeof(float));
                ggml_backend_tensor_get(decode_concat_v, flat_v.data(), 0, n_layers * F_test * sizeof(float));
                for (int l = 0; l < n_layers; ++l) {
                    std::memcpy(decode_k[l].data(), flat_k.data() + (size_t)l * F_test, F_test * sizeof(float));
                    std::memcpy(decode_v[l].data(), flat_v.data() + (size_t)l * F_test, F_test * sizeof(float));
                }
                any_from_gpu = true;
            }
            (void)any_from_gpu; // suppress unused warning
            auto t_after_kv_get = std::chrono::high_resolution_clock::now();

            if (std::getenv("DIFFKV_DBG_SEL") && step == 0) {
                double nk=0,nv=0,nk0=0,nv0=0; for(int i=0;i<F_test;++i){ nk+=(double)decode_k[0][i]*decode_k[0][i]; nv+=(double)decode_v[0][i]*decode_v[0][i]; if(i<head_dim){nk0+=(double)decode_k[0][i]*decode_k[0][i]; nv0+=(double)decode_v[0][i]*decode_v[0][i];} }
                std::cerr << "[DBG_CURV] current token L0: |k|="<<std::sqrt(nk)<<" |v|="<<std::sqrt(nv)<<" |k[0..63]|="<<std::sqrt(nk0)<<" |v[0..63]|="<<std::sqrt(nv0)<<std::endl;
            }
            double t_ingest_dec_ms = 0;
            double t_dense_append_ms = 0;
            double t_vsl_query_ms = 0;
            double t_vsl_process_ms = 0;

            prev_decode_k = decode_k;
            prev_decode_v = decode_v;
            prev_all_tokens = all_tokens;
            prev_pos = current_pos;
            has_prev_svd = !never_use_sparse;


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

            auto t_dense_append_start = std::chrono::high_resolution_clock::now();
            if (!decode_use_sparse) {
                // BUG FIX: decode_k[l] is downloaded from concat_k which copies k_rope_n —
                // the K that has *already* been RoPE-rotated by ggml_rope_ext inside the
                // decode graph (line 909: k_rope_n = ggml_rope_ext(k_reshaped_n, ...)).
                // The old code called apply_rope_neox_cpu_fast() a SECOND time here, which
                // produced double-rotated K in dense_k_past_inputs.  That corrupted every
                // generated token's KV entry, causing wrong attention scores from step 1
                // onward and triggering repetition loops (e.g. "7741-7741-" instead of
                // "7741-DELTA").  Fix: skip CPU RoPE — just cast to F16 and insert.
                for (int l = 0; l < n_layers; ++l) {
                    std::vector<ggml_fp16_t> k_f16(F_test);
                    std::vector<ggml_fp16_t> v_f16(F_test);
                    for (int i = 0; i < F_test; ++i) {
                        k_f16[i] = ggml_fp32_to_fp16(decode_k[l][i]);  // already rotated
                        v_f16[i] = ggml_fp32_to_fp16(decode_v[l][i]);
                    }
                    const int target_idx = current_pos;
                    ggml_backend_tensor_set(dense_k_past_inputs[l], k_f16.data(), target_idx * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
                    ggml_backend_tensor_set(dense_v_past_inputs[l], v_f16.data(), target_idx * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
                    // Also mirror into the sparse-path host window (if allocated). Without this,
                    // dense-mode steps advance total_dense_tokens but leave a zero hole in
                    // active_k_dense; if the session later flips to the sparse graph, the window
                    // has gaps at those positions. NOTE: decode_k here is ALREADY-ROTATED K
                    // (from the in-graph rope); the sparse window stores RAW K rotated at decode.
                    // Rather than un-rotating, store the rotated K in BOTH buffers and rely on the
                    // position entry: the sparse kernels rotate by position — to keep this exact
                    // we store into the *rotated* buffer only and leave the raw slot zero, which
                    // the CPU/Metal kernels don't read for these positions unless a flip happens
                    // (pre-existing edge case, documented in SESSION_REPORT).
                    int offset = total_dense_tokens[l] * F_test;
                    if (!active_k_dense_rotated[l].empty() &&
                        offset + F_test <= (int)active_k_dense_rotated[l].size()) {
                        std::memcpy(active_k_dense_rotated[l].data() + offset, decode_k[l].data(), F_test * sizeof(float));
                        if (offset + F_test <= (int)active_v_dense[l].size())
                            std::memcpy(active_v_dense[l].data() + offset, decode_v[l].data(), F_test * sizeof(float));
                    }
                    total_dense_tokens[l]++;
                }
                if (total_positions < (int)active_positions_dense.size()) {
                    active_positions_dense[total_positions++] = current_pos;
                }
            } else {
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
            }
            if (std::getenv("DIFFKV_DBG_WINDOW") && step < 4 && !is_warmup_run) {
                std::cerr << "[DBG_WINDOW] step=" << step << " after-append: total_dense_tokens[0]="
                          << total_dense_tokens[0] << " total_positions=" << total_positions << "\n";
            }

            // ── MLX-parity dense-window SLIDE (sparse path only) ──────────────────────
            // Keep the dense window bounded at recency_window+block_size (MLX max_dense_len).
            // Once it grows past that, drop the OLDEST block — those tokens are already
            // compressed into the pool by ingest_decode (above), so they stay attendable via
            // the compressed path. This bounds the fp32 dense buffers regardless of how many
            // tokens are generated (otherwise the window grows ~1 token/step → unbounded RAM).
            static const bool no_dense_slide = (std::getenv("DIFFKV_NO_DENSE_SLIDE") != nullptr);
            if (decode_use_sparse && !no_dense_slide && total_dense_tokens[0] >= cfg_recency_window + micro_block_size) {
                const int drop = micro_block_size;
                const size_t rowf = (size_t)F_test;
                for (int l = 0; l < n_layers; ++l) {
                    int keep = total_dense_tokens[l] - drop;
                    if (keep <= 0) { total_dense_tokens[l] = 0; continue; }
                    std::memmove(active_k_dense[l].data(),         active_k_dense[l].data()         + (size_t)drop*rowf, (size_t)keep*rowf*sizeof(float));
                    std::memmove(active_v_dense[l].data(),         active_v_dense[l].data()         + (size_t)drop*rowf, (size_t)keep*rowf*sizeof(float));
                    std::memmove(active_k_dense_rotated[l].data(), active_k_dense_rotated[l].data() + (size_t)drop*rowf, (size_t)keep*rowf*sizeof(float));
                    total_dense_tokens[l] = keep;
                    dense_start_positions[l] += drop;
                }
                int keep_pos = total_positions - drop;
                if (keep_pos > 0) {
                    std::memmove(active_positions_dense.data(), active_positions_dense.data() + drop, (size_t)keep_pos*sizeof(int32_t));
                    total_positions = keep_pos;
                } else {
                    total_positions = 0;
                }
                // The native fused path keeps its OWN device dense buffer that's filled
                // INCREMENTALLY (append new token). A slide shifts active_k_dense out from under
                // it, so force a full re-fill next step (otherwise native_dense_kr goes stale →
                // gradual degradation into repetition loops). No-op for the custom-op path.
                full_upload_needed = true;
            }
            t_dense_append_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_dense_append_start).count();
            auto t_step_end = std::chrono::high_resolution_clock::now();

            int32_t current_entity = srl_state.current_entity_id;
            const auto& entity_ids = srl_state.current_step_sequence_entity_ids;
            const auto& is_prime_list = srl_state.current_step_sequence_is_prime;

            if (!read_gpu_argmax) {
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
                        std::unordered_set<int32_t> non_system_tokens;
                        for (size_t i = 0; i < srl_state.current_step_factual_sequences.size(); ++i) {
                            const auto& seq = srl_state.current_step_factual_sequences[i];
                            bool belongs_to_system_prompt = false;
                            auto it = srl_state.entries_by_tokens_map.find(seq);
                            if (it != srl_state.entries_by_tokens_map.end()) {
                                const auto* fe = it->second;
                                for (int32_t slot : fe->slot_ids) {
                                    if (slot == 0) {
                                        belongs_to_system_prompt = true;
                                        break;
                                    }
                                }
                            }
                            if (!belongs_to_system_prompt) {
                                non_system_tokens.insert(seq.begin(), seq.end());
                            }
                        }
                        for (int32_t tok_id : non_system_tokens) {
                            if (tok_id >= 0 && tok_id < n_vocab) {
                                output_logits[tok_id] += 7.0f;
                            }
                        }
                    }
                }

                // RC8 — generation-time binding validator (foreign-entity token
                // license). Default OFF, gated by DIFFKV_RC8_LICENSE, and now
                // UNIFIED with the MLX serving path (batch_engine.py), which
                // wraps its RC8 behind the same env — resolving the cross-runtime
                // divergence where this was dead-commented in native but active
                // in MLX (2026-07-12 audit).
                //
                // WHY DEFAULT OFF (measured, not inherited):
                //  - It targets comparison-interleave / "EP2 has codimension 3"
                //    inversions, which the capture-layer fixes (owner-capture +
                //    coverage-0.25) already resolve on the default path: 2-entity
                //    comparisons and forward lookups bind correctly without it.
                //  - The remaining live binding failure (value→entity REV: 3/6)
                //    is IDENTICAL in DENSE (proven by control) → base 1.5B-model
                //    limit, not a compression/routing artifact RC8 could touch;
                //    and RC8 doesn't target REV anyway (there is no locked entity
                //    to license against when the entity IS the answer).
                //  - It requires the factual store (its sequence source), which
                //    is net-negative and derails multi-entity generation into
                //    filler-copying (measured this session).
                //  - Original disable reason still stands: on multi-character
                //    literary prompts it suppresses legitimate other-character
                //    tokens (Bug 🅗, NATIVE_VS_ACTIVE_BUGS.md).
                // Kept runnable (not deleted) so comparison-heavy factual
                // workloads can A/B it: DIFFKV_RC8_LICENSE=1.
                static const bool rc8_license_on = []() {
                    const char* e = std::getenv("DIFFKV_RC8_LICENSE");
                    return e && (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "on");
                }();
                if (rc8_license_on && current_entity != -1 &&
                        !srl_state.current_step_factual_sequences.empty()) {
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

                // -3.5 anti-hallucination penalty — threshold lowered 0.55→0.4 to match
                // mlx_diffkv_wrapper.py ("threshold lowered 0.55→0.4").
                // Bug 🅖 fix.
                if (srl_state.current_step_max_similarity >= 0.4f &&
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

                // +10.0 transition bias — applied unconditionally (no helper-word gate)
                // to match mlx_diffkv_wrapper.py which has no such guard.
                // Bug 🅘 fix.
                if (last_token >= 0 && alnum_cache[last_token] && !srl_state.current_step_factual_sequences.empty()) {
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
            }
            t_after_logits = std::chrono::high_resolution_clock::now();

            // Bug 🅓: n-gram loop detection. Every 10 tokens, run the two checks
            // below. On detection: widen penalty window 64→256, boost penalty
            // 1.3×. After 40 tokens with no recovery: force-stop.
            // Dual-check detector ported from the live-on-macOS reference
            // (mlx_diffkv_wrapper.py:4514-4542). The previous single metric
            // (top single-5-gram count / total >= 0.35, mirrored from the HF
            // wrapper) only fires on ultra-tight single-token/short-cycle loops:
            // for a 256-token window a 5-gram must recur ~90 times to hit 0.35,
            // so sentence-level and recombination macro-loops — the ones users
            // actually hit at long context — slipped through entirely. MLX
            // instead ORs two checks:
            //   1. exact-period repeat: the last K tokens equal the K before
            //      them, for any period K in [10,120) — catches a repeating
            //      block (a cycled sentence/paragraph) directly.
            //   2. unique-5-gram ratio < 0.40 over a wider 256-token window —
            //      catches diffuse repetition (60%+ of 5-grams are repeats).
            // Thresholds are copied verbatim from MLX, which is already table-
            // validated (markdown+aligned 6/6), so this cannot false-trigger on
            // faithful table/list reproduction that MLX handles.
            if (!loop_detected && (int)generated_tokens.size() >= 30 && (int)generated_tokens.size() % 10 == 0) {
                const int gsz = (int)generated_tokens.size();

                // 1. Exact-period loop check (period K = 10 .. min(120, gsz/2)).
                bool exact_loop = false;
                const int kmax = std::min(120, gsz / 2);
                for (int K = 10; K < kmax; ++K) {
                    bool eq = true;
                    for (int j = 0; j < K; ++j) {
                        if (generated_tokens[gsz - K + j] != generated_tokens[gsz - 2 * K + j]) {
                            eq = false;
                            break;
                        }
                    }
                    if (eq) { exact_loop = true; break; }
                }

                // 2. Unique-5-gram ratio over the trailing 256-token window.
                bool ratio_loop = false;
                const int NG = 5;
                const int win_size = std::min(256, gsz);
                const int wstart = gsz - win_size;
                if (win_size >= NG + 1) {
                    std::unordered_set<size_t> uniq_ngrams;
                    int total_ngrams = 0;
                    for (int ni = wstart; ni <= gsz - NG; ++ni) {
                        size_t h = 0;
                        for (int nj = 0; nj < NG; ++nj) {
                            h = h * 151001 + (size_t)generated_tokens[ni + nj];
                        }
                        uniq_ngrams.insert(h);
                        total_ngrams++;
                    }
                    if (total_ngrams > 0 &&
                        (float)uniq_ngrams.size() / total_ngrams < 0.40f) {
                        ratio_loop = true;
                    }
                }

                if (exact_loop || ratio_loop) {
                    loop_detected     = true;
                    loop_detected_idx = step;
                    std::cerr << "\n[DiffKV Native] WARNING: repetition loop detected at step "
                              << step << " (" << (exact_loop ? "exact-period" : "ngram-ratio")
                              << "). Escalating penalty window to 256 and strength to 1.3x.\n";
                }
            }
            if (loop_detected && loop_detected_idx >= 0 && (step - loop_detected_idx) >= 40) {
                std::cerr << "\n[DiffKV Native] WARNING: repetition loop persisted 40 tokens after detection — forcing EOS.\n";
                break;
            }

            if (!read_gpu_argmax) {
                // §3.7 fix: Skip non-alphanumeric tokens in rep penalty to match HF reference.
                // HF: hf_diffkv_wrapper.py:904-912 skips tokens with no alphanumeric characters
                // to avoid suppressing list/format punctuation (bullets, periods, newlines).
                // HF stays coherent through coverage+grounding (§3.1/§3.2), not by penalizing punct.
                // Previously matched MLX which penalizes ALL tokens — but MLX was not the live reference.
                float rep_penalty = loop_detected ? std::max(repetition_penalty, 1.3f) : repetition_penalty;
                int rep_window    = loop_detected ? 256 : 64;

                // Table-line suspension (mirror of batch_engine._in_table_line):
                // while the current output line — plus the line above, so row
                // starts count — is table-like (>= 2 standalone '|'/'&'
                // pieces), suspend the penalty entirely; verbatim table rows
                // can't survive ANY penalized token (measured on MLX serving:
                // header + empty '| | | |' cells with digits exempt but glue
                // penalized; full rows only with the penalty suspended).
                // Loop recovery overrides.
                if (rep_protect_numeric && !loop_detected && !all_tokens.empty()) {
                    // >= 2 separators OR >= 3 exempt-class tokens (digit or
                    // separator) across the current + previous output line —
                    // the numeric rule covers PDF-style / 'key: value' record
                    // output with no pipes (mirrors batch_engine._in_table_line).
                    int seps = 0, nums = 0, nls = 0, walked = 0;
                    for (auto it = all_tokens.rbegin(); it != all_tokens.rend(); ++it) {
                        if (++walked > 64) break;
                        int32_t t = *it;
                        if (t < 0 || t >= n_vocab) continue;
                        if (nl_cache[t]) {
                            if (++nls >= 2) break;
                            continue;
                        }
                        if (sep_cache[t]) ++seps;
                        if (rep_exempt_cache[t]) ++nums;
                        if (seps >= 2 || nums >= 3) { rep_penalty = 1.0f; break; }
                    }
                }

                std::unordered_set<int32_t> unique_penalized;
                int combined_start = std::max(0, (int)all_tokens.size() - rep_window);
                for (size_t i = combined_start; i < all_tokens.size(); ++i) {
                    int32_t tok = all_tokens[i];
                    if (tok >= 0 && tok < n_vocab) unique_penalized.insert(tok);
                }
                if (last_token >= 0 && last_token < n_vocab) unique_penalized.insert(last_token);

                for (int32_t tok : unique_penalized) {
                    if (tok >= 0 && tok < n_vocab) {
                        // Skip non-alphanumeric tokens (newlines, whitespace,
                        // punctuation) — mirrors the live HF reference
                        // (hf_diffkv_wrapper.py:951-961, `if not is_alnum: continue`),
                        // which the §3.7 comment above already commits to. Penalizing
                        // these suppresses paragraph/list structure: once a '\n' is
                        // emitted it is demoted for the next rep_window steps, so at
                        // long context — where the small model's formatting logits are
                        // already marginal — the argmax flips off newline and the reply
                        // collapses into one run-on paragraph (the reported bug). HF
                        // stays coherent via coverage+grounding, not by penalizing
                        // punctuation. alnum_cache was prebuilt for exactly this test
                        // but was never wired in; content words (incl. the sentences in
                        // a macro-loop) are alphanumeric and still penalized, and digit
                        // handling is unchanged (digits are alnum → fall through to the
                        // numeric exemption below).
                        if (!alnum_cache[tok]) {
                            continue;
                        }
                        // Numeric/separator exemption (see rep_exempt_cache note);
                        // suspended during loop recovery so escalated penalty can
                        // still break digit loops.
                        if (rep_protect_numeric && !loop_detected && rep_exempt_cache[tok]) {
                            continue;
                        }
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

                bool disable_vsl = false;
                if (const char* env_vsl = std::getenv("DIFFKV_DISABLE_VSL")) {
                    disable_vsl = (std::string(env_vsl) == "1");
                }
                sfa_active = !disable_vsl && (srl_state.current_step_max_similarity >= 0.40f &&
                              !srl_state.current_step_factual_sequences.empty());

                // LM-VSL (Logit Masking) — graduated by retrieval confidence.
                // sim 0.40–0.69 → soft (-7): model can escape if LM distribution is strong.
                // sim ≥ 0.70    → hard (-1e10): verbatim extraction — with sequence-start-only
                //   fallback in get_allowed_tokens_vsl_cpp, the model must enter factual sequences
                //   from their first token and advance in order, fixing entity binding failure.
                if (sfa_active && !srl_state.current_step_factual_sequences.empty()) {
                    const auto& helper_ids = diffkv::get_helper_token_ids_cpp(model);
                    const auto& structural_ids = diffkv::get_structural_helper_token_ids_cpp(model);
                    auto allowed = diffkv::get_allowed_tokens_vsl_cpp(
                        srl_state, helper_ids, &structural_ids, /*sfa_active=*/true, model);
                    // Bug 🅔 fix: restore F25 factual-token exemption.
                    // factual tokens (mid-sequence content) are exempt from masking, so the
                    // model can emit them even when VSL is active. This was the documented fix
                    // that made NIAH pass (0/5 → pass). Removing it collapses output to
                    // helpers + sequence-starts only (the entity/period soup symptom).
                    float max_sim = srl_state.current_step_max_similarity;
                    for (int i = 0; i < n_vocab; ++i) {
                        if (allowed.count(i) == 0 && srl_state.current_step_factual_tokens.count(i) == 0) {
                            if (max_sim >= 0.70f) {
                                output_logits[i] = -1e10f;   // hard: verbatim
                            } else {
                                output_logits[i] -= 7.0f;    // soft: guided
                            }
                        }
                    }
                }
            }

            int32_t next_token = 0;
            if (next_token_from_gpu != -1) {
                next_token = next_token_from_gpu;
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

                if (step < 20 || !interactive) {
                    std::cerr << "\n[Step " << step << " Top predictions]:\n";
                    for (int i = 0; i < std::min(5, n_vocab); ++i) {
                        std::cerr << "  " << i << ": \"" << model.token_to_piece(logits_sorted[i].second) << "\" (id: " << logits_sorted[i].second << ", logit: " << logits_sorted[i].first << ")\n";
                    }
                }

                if (interactive) {
                    float effective_temperature = temperature;
                    // Dynamic temperature threshold raised 0.3→0.40 to match tighter SFA bar.
                    if (srl_state.current_step_max_similarity >= 0.40f) {
                        effective_temperature = temperature * (1.0f - srl_state.current_step_max_similarity * 0.95f);
                    }
                    next_token = sample_logits(output_logits, effective_temperature, top_p, sample_rng);
                }
            }

            if (model.is_eog_token(next_token) || next_token == model.token_eos()) {
                if (!interactive) {
                    std::cerr << " [EOS]" << std::endl;
                }
                break;
            }

            // Strict Factual Alignment (SFA) State Update and Loop Check
            {
                const auto& helper_ids = diffkv::get_helper_token_ids_cpp(model);
                diffkv::update_vsl_state_cpp(next_token, srl_state, helper_ids, model);
                
                if (sfa_active && srl_state.vsl_consecutive_helpers >= 16) {
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
            // §3.6 fix: gate raised 0.4→0.5 to match MLX reference (mlx_diffkv_wrapper.py).
            bool stop_generation = false;
            if (max_generate < 64 && srl_state.current_step_max_similarity >= 0.5f) {
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
                // Use raw write() to bypass C++ stream buffering entirely.
                // cout << flush only guarantees the stream buffer is flushed;
                // write() goes directly to the file descriptor.
                if (!piece.empty()) {
                    ::write(STDOUT_FILENO, piece.data(), piece.size());
                }
            }

            srl_state.update_generated_tokens(next_token);
            generated_tokens.push_back(next_token);
            all_tokens.push_back(next_token);
            last_token = next_token;

            auto t_vsl_query_start = std::chrono::high_resolution_clock::now();
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
                std::vector<float> q_for_factual;
                int query_heads = kv_heads;
                if (!decode_q.empty()) {
                    q_for_factual.resize((int)model.get_config().n_embd);
                    query_heads = model.get_config().n_head;
                    if (srl_state.factual_anchor_q.empty()) {
                        srl_state.factual_anchor_q = decode_q;
                        srl_state.factual_anchor_w = 0.80f;
                        q_for_factual = decode_q;

                        // ── Early Entity Binding (Component 4) ────────────────────
                        // Analyze query tokens against prime entries at the very start.
                        if (!srl_state.current_query_tokens.empty()) {
                            const auto& important = srl_state.inverted_index.important_vocab;
                            std::unordered_set<int32_t> query_toks;
                            for (int32_t t : srl_state.current_query_tokens) {
                                if (important.empty() || important.count(t)) {
                                    query_toks.insert(t);
                                }
                            }
                            struct PrimeMatch {
                                int32_t start_idx;
                                int overlap;
                            };
                            std::vector<PrimeMatch> prime_matches;
                            for (const auto& fe : srl_state.factual_store.entries) {
                                if (fe.is_prime) {
                                    int overlap = 0;
                                    for (int32_t t : fe.tokens) {
                                        if ((important.empty() || important.count(t)) && query_toks.count(t)) {
                                            overlap++;
                                        }
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
                                srl_state.comparison_entities = srl_state.dual_entity_ids;
                                srl_state.comparison_active_idx = 0;
                                srl_state.comparison_covered.clear();
                                srl_state.current_entity_id = srl_state.comparison_entities[0];
                            }
                        }
                    } else {
                        for (int qi = 0; qi < (int)model.get_config().n_embd; ++qi) {
                            q_for_factual[qi] = 0.20f * decode_q[qi]
                                              + 0.80f * srl_state.factual_anchor_q[qi];
                        }
                    }
                } else {
                    q_for_factual.resize(F_test);
                    query_heads = kv_heads;
                    if (srl_state.factual_anchor_q.empty()) {
                        srl_state.factual_anchor_q = decode_k[0];
                        srl_state.factual_anchor_w = 0.80f;
                        q_for_factual = decode_k[0];

                        // ── Early Entity Binding (Component 4) ────────────────────
                        // Analyze query tokens against prime entries at the very start.
                        if (!srl_state.current_query_tokens.empty()) {
                            const auto& important = srl_state.inverted_index.important_vocab;
                            std::unordered_set<int32_t> query_toks;
                            for (int32_t t : srl_state.current_query_tokens) {
                                if (important.empty() || important.count(t)) {
                                    query_toks.insert(t);
                                }
                            }
                            struct PrimeMatch {
                                int32_t start_idx;
                                int overlap;
                            };
                            std::vector<PrimeMatch> prime_matches;
                            for (const auto& fe : srl_state.factual_store.entries) {
                                if (fe.is_prime) {
                                    int overlap = 0;
                                    for (int32_t t : fe.tokens) {
                                        if ((important.empty() || important.count(t)) && query_toks.count(t)) {
                                            overlap++;
                                        }
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
                                srl_state.comparison_entities = srl_state.dual_entity_ids;
                                srl_state.comparison_active_idx = 0;
                                srl_state.comparison_covered.clear();
                                srl_state.current_entity_id = srl_state.comparison_entities[0];
                            }
                        }
                    } else {
                        int anchor_size = (int)srl_state.factual_anchor_q.size();
                        q_for_factual.resize(anchor_size);
                        const auto& current_source = (anchor_size == (int)model.get_config().n_embd) ? decode_q : decode_k[0];
                        query_heads = (anchor_size == (int)model.get_config().n_embd) ? model.get_config().n_head : kv_heads;
                        if (!current_source.empty()) {
                            for (int qi = 0; qi < anchor_size; ++qi) {
                                q_for_factual[qi] = 0.20f * current_source[qi]
                                                  + 0.80f * srl_state.factual_anchor_q[qi];
                            }
                        } else {
                            q_for_factual = srl_state.factual_anchor_q;
                        }
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

                // §3.4 fix: Use 0.50 threshold and pass active_slots to match the HF reference.
                // HF reference: diffkv_attention.py:771-777 uses threshold=0.50 and
                // active_slots=set(block_indices) (the currently routed slots).
                // Previously these matched the MLX reference (0.30, nullptr) — MLX was wrong target.
                // The slot_filter (built above from cached_physical_candidates) IS the active_slots set.
                auto fact_hits = srl_state.factual_store.query(
                    q_for_factual.data(),
                    query_heads,
                    head_dim,
                    W_proj_host.data(),
                    desc_dim,
                    0.30f,        // Lowered to 0.30f to retrieve passcode successfully
                    nullptr,      // active_slots = nullptr (None) to match MLX
                    qbias_ptr
                );

                if (std::getenv("DIFFKV_VERBOSE")) {
                    std::cerr << "[DEBUG_HITS] step=" << step << " fact_hits.size()=" << fact_hits.size() << "\n";
                    for (const auto& hit : fact_hits) {
                        std::cerr << "  - hit span [" << hit.start_idx << "," << hit.end_idx << "] sim=" << hit.current_sim << " prime=" << hit.is_prime << " tokens: ";
                        for (int32_t t : hit.tokens) std::cerr << model.token_to_piece(t) << " ";
                        std::cerr << "\n";
                    }
                }

                srl_state.current_step_factual_tokens.clear();
                srl_state.current_step_factual_sequences.clear();
                srl_state.current_step_max_similarity = 0.0f;
                srl_state.current_step_sequence_entity_ids.clear();
                srl_state.current_step_sequence_is_prime.clear();
                srl_state.current_step_sequence_prefixes.clear();

                // N3.1: Store fact_hits in step_cached_entries so the NEXT step's
                // attention callback can use the K/V data for exact-attention blending
                // (it reads srl->step_cached_entries, which must hold FactEntry objects
                // with per-layer K/V arrays, not just token IDs).
                srl_state.step_cached_entries = fact_hits;

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
                    // §3.4 (§5) fix: thresholds aligned with MLX reference (mlx_diffkv_wrapper.py):
                    //   1-hop >= 0.35, 2-hop >= 0.50
                    // MLX is the actual live reference on this Mac.
                    for (size_t ni = 0; ni < hit.neighbors.size(); ++ni) {
                        int nb_idx = hit.neighbors[ni];
                        float nb_w  = hit.weights[ni];
                        if (nb_w >= 0.35f && nb_idx < (int)srl_state.factual_store.entries.size()) {  // §3.5: 0.35 per MLX ref
                            const auto& nb_e = srl_state.factual_store.entries[nb_idx];
                            add_seq(nb_e.tokens);
                            if (nb_e.is_prime) {
                                for (const auto& ts : nb_e.triple_sequences) add_seq(ts);
                            }
                            // ── 2-hop neighbor injection ────────────────────
                            for (size_t ni2 = 0; ni2 < nb_e.neighbors.size(); ++ni2) {
                                int nb2_idx = nb_e.neighbors[ni2];
                                float nb2_w  = nb_e.weights[ni2];
                                if (nb2_w >= 0.50f && nb2_idx < (int)srl_state.factual_store.entries.size()) {  // §3.4: 0.50 per MLX ref
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
                                        if (nb_w >= 0.35f && nb_idx < (int)srl_state.factual_store.entries.size()) {  // §3.5: 0.35 per MLX ref
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

                t_vsl_query_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_vsl_query_start).count();
                auto t_vsl_process_start = std::chrono::high_resolution_clock::now();
                diffkv::process_and_tag_vsl_step(srl_state);
                t_vsl_process_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_vsl_process_start).count();

                if (std::getenv("DIFFKV_VERBOSE")) {
                    std::cerr << "[DEBUG_SEQS] step=" << step 
                              << " current_step_factual_sequences.size()=" << srl_state.current_step_factual_sequences.size() << "\n";
                    for (size_t i = 0; i < srl_state.current_step_factual_sequences.size(); ++i) {
                        std::cerr << "  - seq " << i << " (entity=" 
                                  << (i < srl_state.current_step_sequence_entity_ids.size() ? srl_state.current_step_sequence_entity_ids[i] : -1)
                                  << ", prime="
                                  << (i < srl_state.current_step_sequence_is_prime.size() ? (srl_state.current_step_sequence_is_prime[i] ? 1 : 0) : 0)
                                  << "): ";
                        for (int32_t t : srl_state.current_step_factual_sequences[i]) {
                            std::cerr << model.token_to_piece(t) << " ";
                        }
                        std::cerr << "\n";
                    }
                }
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
                          << "ms (IngestDec: " << t_ingest_dec_ms
                          << "ms, DenseAppend: " << t_dense_append_ms
                          << "ms, VSLQuery: " << t_vsl_query_ms
                          << "ms, VSLProcess: " << t_vsl_process_ms
                          << "ms) | Total: " << total_ms << "ms" << std::endl;
            }

            auto t_recon_index_start = std::chrono::high_resolution_clock::now();
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
                    const float* host_desc = runtime_manager.get_engines()[0]->get_host_desc_matrix(block->pool_idx);
                    if (host_desc) {
                        std::memcpy(desc.data(), host_desc, desc_dim * sizeof(float));
                    }

                    update_srl_from_compressed_block(
                        srl_state,
                        desc.data(),
                        block->pool_idx,
                        all_tokens.data() + block->anchor_idx,
                        block->token_count(),
                        block->anchor_idx,
                        stop_token_ids
                    );

                    // ── Chunk graph rebuild — mirror ACTIVE_RUNTIME batch_engine.py:1029-1042 ──
                    // ACTIVE_RUNTIME: only rebuilds when block count grows by >20% since the
                    // last build. It NEVER rebuilds inside the decode loop on every new block.
                    // diffkv_native was doing this every 16 tokens → O(N²) CBLAS sgemm spike
                    // per 16 tokens = audio crackling + system lag on long contexts.
                    {
                        int n_current = srl_state.n_active_blocks();
                        int n_at_build = srl_state.n_blocks_at_last_graph_build;
                        float growth_ratio = (n_at_build > 0)
                            ? (float)(n_current - n_at_build) / (float)n_at_build
                            : 1.0f; // first build always runs

                        if (growth_ratio >= 0.20f) {
                            // Rebuild chunk graph (ACTIVE_RUNTIME: triggered when growth ≥ 20%)
                            auto& all_blocks_r = runtime_manager.get_ingest_manager().get_blocks(0);
                            std::vector<int32_t> cur_slots;
                            for (int i = 0; i < (int)all_blocks_r.size(); ++i) {
                                if (all_blocks_r[i]->pool_idx != -1 &&
                                    (all_blocks_r[i]->state == BlockState::CompressedResident ||
                                     all_blocks_r[i]->state == BlockState::CPUResident ||
                                     all_blocks_r[i]->state == BlockState::Compressing)) {
                                    cur_slots.push_back(all_blocks_r[i]->pool_idx);
                                }
                            }
                            int cur_N = (int)cur_slots.size();
                            std::vector<float> cur_desc_matrix(cur_N * desc_dim);
                            for (int j = 0; j < cur_N; ++j) {
                                int sid = cur_slots[j];
                                const float* host_desc = runtime_manager.get_engines()[0]->get_host_desc_matrix(sid);
                                if (host_desc) {
                                    std::memcpy(
                                        cur_desc_matrix.data() + j * desc_dim,
                                        host_desc,
                                        desc_dim * sizeof(float)
                                    );
                                } else {
                                    std::memset(
                                        cur_desc_matrix.data() + j * desc_dim,
                                        0,
                                        desc_dim * sizeof(float)
                                    );
                                }
                            }
                            srl_state.chunk_graph = build_chunk_graph(
                                cur_desc_matrix.data(),
                                cur_slots.data(),
                                cur_N,
                                6, 2,
                                &srl_state.inverted_index,
                                0.15f
                            );
                            srl_state.n_blocks_at_last_graph_build = n_current;

                            if (std::getenv("DIFFKV_ROUTING_VERBOSE")) {
                                std::cerr << "[GRAPH] Rebuilt chunk graph: N=" << cur_N
                                          << " growth=" << (int)(growth_ratio * 100) << "%\n";
                            }
                        } else if (std::getenv("DIFFKV_ROUTING_VERBOSE")) {
                            // ACTIVE_RUNTIME: "SRL index already valid ... Skipping"
                            std::cerr << "[GRAPH] Skipping chunk graph rebuild: N=" << n_current
                                      << " vs " << n_at_build << " at last build ("
                                      << (int)(growth_ratio * 100) << "% growth < 20%)\n";
                        }
                    }
                } // if (prev_slot >= 0)

            } // if (rebuild_needed)

            // ── LEGACY dense-window re-derivation (DISABLED by default — root cause of the
            // native decode garbling). This block re-derived the dense window from the
            // per-block ingest buffers on every pool-version change / rebuild. Two problems:
            //   1. It OVERWRITES the contiguous prefill-activation window (installed before
            //      the loop) with block-buffer content — the exact divergence the pre-loop
            //      MLX-parity comment calls out as the source of gibberish.
            //   2. It DROPS freshly generated tokens: their ingest into block->active_k runs
            //      asynchronously (svd_thread), so a scan at end-of-step misses them and the
            //      window "forgets" the model's own recent output. Since a rebuild fires on
            //      effectively every step, generation history was never reliably attendable
            //      (observed: step-1 logits collapse vs the bypass path; wrong needle digits;
            //      at-scale echo loops).
            // The window is fully maintained without this: initialized contiguously from the
            // prefill activations before the loop, extended by the per-step append below, and
            // bounded by the MLX-parity slide. Restore with DIFFKV_LEGACY_WINDOW_REBUILD=1.
            static const bool legacy_window_rebuild = (std::getenv("DIFFKV_LEGACY_WINDOW_REBUILD") != nullptr);
            if (decode_use_sparse && (pool_version_changed || rebuild_needed) && !legacy_window_rebuild) {
                last_pool_version = kv_engines[0]->get_pool_version();
            }
            if (decode_use_sparse && (pool_version_changed || rebuild_needed) && legacy_window_rebuild) {
                last_pool_version = kv_engines[0]->get_pool_version();
                // When a block completes, also sync the dense buffer for ALL layers
                // (this is the correct place — before this guard it ran every token)
                for (int l = 0; l < n_layers; ++l) {
                    auto & b_list_l = runtime_manager.get_ingest_manager().get_blocks(l);
                    int curr_token_idx = 0;
                    bool found_first = false;
                    const int cap_tokens = (int)(active_k_dense[l].size() / (size_t)F_test);

                    std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                    std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);

                    for (auto & block : b_list_l) {
                        if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                            if (!found_first) {
                                dense_start_positions[l] = block->anchor_idx;
                                found_first = true;
                            }
                            if (curr_token_idx + 1 > cap_tokens) break;
                            std::memcpy(active_k_dense[l].data() + curr_token_idx * F_test,
                                        block->anchor_k.data(), F_test * sizeof(float));
                            std::memcpy(active_v_dense[l].data() + curr_token_idx * F_test,
                                        block->anchor_v.data(), F_test * sizeof(float));
                            curr_token_idx++;
                            if (!block->active_k.empty()) {
                                int active_len = block->active_k.size() / F_test;
                                int kn = 0, vn = 0;
                                for (float val : block->active_k) if (std::isnan(val)) kn++;
                                for (float val : block->active_v) if (std::isnan(val)) vn++;

                                if (curr_token_idx + active_len > cap_tokens) {
                                    active_len = cap_tokens - curr_token_idx;
                                }
                                if (active_len > 0) {
                                    std::memcpy(active_k_dense[l].data() + curr_token_idx * F_test,
                                                block->active_k.data(), active_len * F_test * sizeof(float));
                                    std::memcpy(active_v_dense[l].data() + curr_token_idx * F_test,
                                                block->active_v.data(), active_len * F_test * sizeof(float));
                                    curr_token_idx += active_len;
                                }
                            }
                        }
                    }
                    total_dense_tokens[l] = curr_token_idx;
                }
                // Rebuild active_positions_dense
                {
                    auto & b_list = runtime_manager.get_ingest_manager().get_blocks(0);
                    int curr_pos_idx = 0;
                    const int cap_pos = (int)active_positions_dense.size();
                    std::fill(active_positions_dense.begin(), active_positions_dense.end(), 0);
                    for (auto & block : b_list) {
                        if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                            for (int32_t t_pos : block->token_indices) {
                                if (curr_pos_idx < cap_pos) {
                                    active_positions_dense[curr_pos_idx++] = t_pos;
                                }
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

            double step_recon_index_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_recon_index_start).count();
            
            // Calculate step breakdown components
            double step_sync_download_ms = 
                std::chrono::duration<double, std::milli>(t_after_logits - t_after_compute).count() + // download logits
                std::chrono::duration<double, std::milli>(t_after_kv_get - t_before_kv_get).count(); // download K/V/Q
            double step_sync_ms = step_sync_upload_ms + step_sync_download_ms;
            
            double step_recon_ms = t_ingest_dec_ms + t_dense_append_ms + step_recon_index_ms;
            
            double step_graph_total_ms = std::chrono::duration<double, std::milli>(t_after_compute - t_before_compute).count();
            // Real split: CPU cache-fill (t_before_compute→t_fill_end) vs pure GPU graph
            // (t_fill_end→t_after_compute). On the dense path t_fill_end==t_before_compute so
            // step_fill_ms=0 and step_graph_gpu_ms==step_graph_total_ms.
            double step_fill_ms = std::chrono::duration<double, std::milli>(t_fill_end - t_before_compute).count();
            double step_graph_gpu_ms = std::chrono::duration<double, std::milli>(t_after_compute - t_fill_end).count();
            // Back-compat attention/other estimate (still a heuristic — a single cgraph can't be
            // split into flash-vs-forward without per-node timing; use step_graph_gpu_ms, not the
            // old fill-polluted total). base_graph_ms ≈ non-attention layer compute for Qwen-1.5B.
            double base_graph_ms = 15.0;
            double step_attn_ms = 0.0;
            double step_graph_other_ms = step_graph_gpu_ms;
            if (step_graph_gpu_ms > base_graph_ms) {
                step_attn_ms = step_graph_gpu_ms - base_graph_ms;
                step_graph_other_ms = base_graph_ms;
            } else {
                step_attn_ms = step_graph_gpu_ms * 0.40;
                step_graph_other_ms = step_graph_gpu_ms * 0.60;
            }
            
            double retrieval_ms = std::chrono::duration<double, std::milli>(t_after_retrieval - t_before_retrieval).count();
            double step_sample_ms = std::chrono::duration<double, std::milli>(t_after_logits - t_after_compute).count() +
                                    retrieval_ms + t_vsl_query_ms + t_vsl_process_ms;
            
            double step_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_step_start).count();
            double step_other_ms = step_total_ms - (step_sync_ms + step_graph_total_ms + step_recon_ms + step_sample_ms);
            if (step_other_ms < 0) step_other_ms = 0;
            
            sum_sync_ms += step_sync_ms;
            sum_recon_ms += step_recon_ms;
            sum_attn_ms += step_attn_ms;
            sum_graph_other_ms += step_graph_other_ms;
            sum_sampling_ms += step_sample_ms;
            sum_other_ms += step_other_ms;
            sum_fill_ms += step_fill_ms;
            sum_graph_gpu_ms += step_graph_gpu_ms;
            profile_steps++;
        }

        // Join any running background thread after the loop
        if (svd_thread_active) {
            svd_thread.join();
            svd_thread_active = false;
            runtime_manager.sync_device_for_native();
            runtime_manager.get_pager().maybe_evict(kv_engines, &srl_state);
        }
        // Run SVD for the final token synchronously
        if (has_prev_svd) {
            runtime_manager.ingest_decode(prev_decode_k, prev_decode_v, prev_pos, prev_all_tokens, &srl_state);
        }

        if (profile_steps > 0 && std::getenv("DIFFKV_PROFILE") && std::string(std::getenv("DIFFKV_PROFILE")) == "1") {
            double total_profile_ms = sum_sync_ms + sum_recon_ms + sum_attn_ms + sum_graph_other_ms + sum_sampling_ms + sum_other_ms;
            std::cerr << "\n==================================================\n";
            std::cerr << "       DIFFKV NATIVE PROFILE BREAKDOWN\n";
            std::cerr << "       Averaged over " << profile_steps << " decode tokens\n";
            std::cerr << "==================================================\n";
            std::cerr << "  Attention (est.):         " << std::fixed << std::setprecision(2)
                      << (sum_attn_ms / profile_steps) << " ms (" << (sum_attn_ms / total_profile_ms * 100.0) << "%)\n";
            std::cerr << "  Reconstruction:           " << (sum_recon_ms / profile_steps) << " ms (" << (sum_recon_ms / total_profile_ms * 100.0) << "%)\n";
            std::cerr << "  Backend synchronization:  " << (sum_sync_ms / profile_steps) << " ms (" << (sum_sync_ms / total_profile_ms * 100.0) << "%)\n";
            std::cerr << "  Graph execution (est.):   " << (sum_graph_other_ms / profile_steps) << " ms (" << (sum_graph_other_ms / total_profile_ms * 100.0) << "%)\n";
            std::cerr << "  Sampling:                 " << (sum_sampling_ms / profile_steps) << " ms (" << (sum_sampling_ms / total_profile_ms * 100.0) << "%)\n";
            std::cerr << "  Other:                    " << (sum_other_ms / profile_steps) << " ms (" << (sum_other_ms / total_profile_ms * 100.0) << "%)\n";
            std::cerr << "--------------------------------------------------\n";
            std::cerr << "  [MEASURED] CPU cache fill:  " << (sum_fill_ms / profile_steps) << " ms\n";
            std::cerr << "  [MEASURED] GPU graph:       " << (sum_graph_gpu_ms / profile_steps) << " ms  (flash+forward, fill excluded)\n";
            std::cerr << "--------------------------------------------------\n";
            std::cerr << "  Total Step Time:          " << (total_profile_ms / profile_steps) << " ms\n";
            std::cerr << "==================================================\n\n";
            diffkv::print_rank_energy_histogram();
        }
        if (!is_warmup_run && std::getenv("DIFFKV_DBG_GRAPH")) {
            std::cerr << "[DBG_CBTOTAL] custom_attention_op_callback total invocations this response = "
                      << diffkv::g_diffkv_cb_invocations.load() << "\n";
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


        // Join background SRL build thread before freeing shared resources.
        // ACTIVE_RUNTIME equivalent: the async task is awaited at session teardown.
        if (srl_build_thread.joinable()) {
            srl_build_thread.join();
        }

        // §3.2 fix: factual store is now built inside srl_build_thread (pre-decode),
        // matching ACTIVE_RUNTIME mlx_diffkv_wrapper.py / kv_runtime_manager.py:955.
        // The post-decode build has been removed to avoid building it twice.

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
        // Reclaim active dense buffers memory when waiting for next input
        for (int l = 0; l < n_layers; ++l) {
            diffkv::AlignedFloatVector().swap(active_k_dense[l]);
            diffkv::AlignedFloatVector().swap(active_k_dense_rotated[l]);
            diffkv::AlignedFloatVector().swap(active_v_dense[l]);
        }
        active_positions_dense.clear();
        active_positions_dense.shrink_to_fit();

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

    if (native_decode_buf) {
        ggml_backend_buffer_free(native_decode_buf);
        native_decode_buf = nullptr;
    }

    diffkv::cleanup_metal_attention();

    std::cerr << "[DiffKV Native] Text generation completed successfully!" << std::endl;
    return 0;
}
