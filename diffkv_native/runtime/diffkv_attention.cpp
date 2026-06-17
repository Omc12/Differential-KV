#include "runtime/diffkv_attention.hpp"
#include "native_core/srl/attention_cache.hpp"
#include "native_core/srl/session_srl_state.hpp"
#include "native_core/srl/factual_store.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_set>

namespace diffkv {

// ── CPU Project-Then-Attend (reference / non-Apple fallback) ──────────────────
// Kept for correctness comparison and CUDA/CPU builds.
// On Apple the Metal kernel in diffkv_attention.mm is used instead.
void execute_cpu_attention(
    const float* Q,
    const int32_t* slots,
    float* cpu_output,
    float* lse_sparse,
    NativeBlockPool* kv_engine,
    int n_q_heads, int n_kv_heads, int rank, int S_max, int K, int D, float scale,
    bool has_rope, float rope_freq_base, bool approximate_attn
) {
    const float* Q_ptr = Q;
    const int8_t* U = kv_engine->get_host_U();
    const ggml_fp16_t* U_scale_arr = kv_engine->get_host_U_scale();
    const ggml_fp16_t* VK = kv_engine->get_host_VK();
    const ggml_fp16_t* VV = kv_engine->get_host_VV();
    const ggml_fp16_t* anchors_K = kv_engine->get_host_anchors_K();
    const ggml_fp16_t* anchors_V = kv_engine->get_host_anchors_V();
    const int32_t* seq_lens = kv_engine->get_host_seq_lens();
    const ggml_fp16_t* scales = kv_engine->get_host_scales();
    const int32_t* anchor_positions = kv_engine->get_host_anchor_positions();
    const int32_t* res_K_pos = kv_engine->get_host_res_K_pos();
    const int32_t* res_V_pos = kv_engine->get_host_res_V_pos();
    const ggml_fp16_t* res_K_val = kv_engine->get_host_res_K_val();
    const ggml_fp16_t* res_V_val = kv_engine->get_host_res_V_val();
    const int MR = NativeBlockPool::MAX_RESIDUAL;

    // Bug 2 fix: O(1) dedup via unordered_set.
    int n_slots = kv_engine->get_seq_lens()->ne[0];
    std::unordered_set<int32_t> seen_set;
    seen_set.reserve(K);
    std::vector<int32_t> unique_slots;
    unique_slots.reserve(K);
    for (int k = 0; k < K; ++k) {
        int32_t sid = slots[k];
        if (sid >= 0 && sid < n_slots && seen_set.insert(sid).second)
            unique_slots.push_back(sid);
    }
    const int32_t* active_slots = unique_slots.data();
    int active_K = (int)unique_slots.size();

    const int half_d = D / 2;
    const int g = n_q_heads / n_kv_heads;

    // Bug 9/1 fix: precomputed RoPE-rotated key buffers.
    const float* anc_k_rot  = kv_engine->get_host_anchorK_rot();
    const float* vk_rot_buf = kv_engine->get_host_VK_rot();
    const bool use_precomp_rot = (has_rope && anc_k_rot != nullptr && vk_rot_buf != nullptr);

    // ── theta_table: half_d std::pow calls, once per call ─────────────────────
    std::vector<float> theta_table;
    if (has_rope) {
        theta_table.resize(half_d);
        for (int i = 0; i < half_d; ++i)
            theta_table[i] = 1.0f / std::pow(rope_freq_base, (2.0f * i) / D);
    }

    // ── cos_step / sin_step: half_d cos+sin calls, computed ONCE per call ─────
    // Enables angular recurrence in the non-approx token loop:
    //   cos((pos+1)·θ_d) = cos(pos·θ_d)·cos_step[d] - sin(pos·θ_d)·sin_step[d]
    // → 4 multiplies per dim-pair per token step, zero transcendental calls.
    std::vector<float> cos_step(half_d, 1.0f), sin_step(half_d, 0.0f);
    if (has_rope && !approximate_attn) {
        for (int d = 0; d < half_d; ++d) {
            cos_step[d] = std::cos(theta_table[d]);
            sin_step[d] = std::sin(theta_table[d]);
        }
    }

    // ── Per-block anchor cos/sin: computed ONCE per block, OUTSIDE the head loop.
    // anchor_pos depends only on the block (not on the query head), so these values
    // are identical for all n_q_heads query heads. Computing once and reusing saves
    //   active_K × half_d × (n_q_heads - 1)   cos+sin calls per execute call.
    // They also serve as the initial rotation state for the non-approx recurrence.
    struct BlockRope { std::vector<float> ca, sa; };
    std::vector<BlockRope> blk_rope(active_K);
    if (has_rope) {
        for (int k = 0; k < active_K; ++k) {
            int ap = anchor_positions[active_slots[k]];
            blk_rope[k].ca.resize(half_d);
            blk_rope[k].sa.resize(half_d);
            for (int d = 0; d < half_d; ++d) {
                float angle = (float)ap * theta_table[d];  // multiply only — no pow
                blk_rope[k].ca[d] = std::cos(angle);
                blk_rope[k].sa[d] = std::sin(angle);
            }
        }
    }

    // ── Per-head attention ───────────────────────────────────────────────────
    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;

        float max_score = -1e30f;
        struct SlotInfo {
            float anchor_score;
            std::vector<float> q_proj;
            std::vector<float> token_scores;
        };
        std::vector<SlotInfo> slot_infos(active_K);

        for (int k = 0; k < active_K; ++k) {
            int slot_id   = active_slots[k];
            int slen      = seq_lens[slot_id];
            float scale_u = ggml_fp16_to_fp32(U_scale_arr[slot_id]);
            float blk_sc  = ggml_fp16_to_fp32(scales[slot_id]);

            // Reuse precomputed per-block anchor cos/sin — zero new trig per head.
            const float* ca = has_rope ? blk_rope[k].ca.data() : nullptr;
            const float* sa = has_rope ? blk_rope[k].sa.data() : nullptr;

            // ── 1. Anchor score ──────────────────────────────────────────────
            float score_anc = 0.0f;
            if (use_precomp_rot) {
                // Precomputed anchor-pos-rotated K — zero trig.
                const float* ak_r = anc_k_rot +
                    (size_t)slot_id * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d)
                    score_anc += Q_ptr[h * D + d] * ak_r[d];
            } else {
                const ggml_fp16_t* ak_base = anchors_K +
                    slot_id * n_kv_heads * D + kv_head * D;
                if (has_rope) {
                    // ca[d]/sa[d] precomputed — zero new trig.
                    for (int d = 0; d < half_d; ++d) {
                        float x = ggml_fp16_to_fp32(ak_base[d]);
                        float y = ggml_fp16_to_fp32(ak_base[d + half_d]);
                        score_anc += Q_ptr[h * D + d]          * (x * ca[d] - y * sa[d]);
                        score_anc += Q_ptr[h * D + d + half_d] * (y * ca[d] + x * sa[d]);
                    }
                } else {
                    for (int d = 0; d < D; ++d)
                        score_anc += Q_ptr[h * D + d] *
                                     ggml_fp16_to_fp32(ak_base[d]);
                }
            }
            slot_infos[k].anchor_score = score_anc;
            { float s = score_anc * scale; if (s > max_score) max_score = s; }

            if (std::getenv("DIFFKV_DBG_KANC") && h == 0 && k < 8)
                std::cerr << "[DBG_KANC] slot=" << slot_id
                          << " score_anc=" << score_anc << "\n";

            if (!approximate_attn) {
                // ── Non-approximate: full K reconstruction + angular recurrence ──
                //
                // Initialise cos_run at position (anchor_pos + 1) by advancing the
                // per-block ca/sa by one step with cos_step (4 mults, 0 trig).
                // Each subsequent token advances by one more position — still 4
                // multiplies per dim-pair, zero transcendental function calls.
                std::vector<float> cos_run(half_d), sin_run(half_d);
                if (has_rope) {
                    for (int d = 0; d < half_d; ++d) {
                        cos_run[d] = ca[d] * cos_step[d] - sa[d] * sin_step[d];
                        sin_run[d] = sa[d] * cos_step[d] + ca[d] * sin_step[d];
                    }
                }

                slot_infos[k].token_scores.resize(slen);
                int ak_off = slot_id * n_kv_heads * D + kv_head * D;

                for (int t = 0; t < slen; ++t) {
                    // cos_run[d] = cos((anchor_pos + t + 1)·θ_d) — exact, no trig
                    float score_t = 0.0f;
                    int u_off = slot_id * S_max * rank + t * rank;

                    for (int d = 0; d < half_d; ++d) {
                        int d2 = d + half_d;
                        // Reconstruct K pair: anchor + SVD delta
                        float raw_k1 = ggml_fp16_to_fp32(anchors_K[ak_off + d]);
                        float raw_k2 = ggml_fp16_to_fp32(anchors_K[ak_off + d2]);
                        float dk1 = 0.0f, dk2 = 0.0f;
                        int vk1 = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                        int vk2 = slot_id * rank * n_kv_heads * D + kv_head * D + d2;
                        for (int r = 0; r < rank; ++r) {
                            float ur = (float)U[u_off + r];
                            dk1 += ur * ggml_fp16_to_fp32(VK[vk1 + r * n_kv_heads * D]);
                            dk2 += ur * ggml_fp16_to_fp32(VK[vk2 + r * n_kv_heads * D]);
                        }
                        float k1 = raw_k1 + dk1 * scale_u * blk_sc;
                        float k2 = raw_k2 + dk2 * scale_u * blk_sc;

                        // Rotate using recurrence state — zero cos/sin calls.
                        float kr1, kr2;
                        if (has_rope) {
                            kr1 = k1 * cos_run[d] - k2 * sin_run[d];
                            kr2 = k2 * cos_run[d] + k1 * sin_run[d];
                        } else { kr1 = k1; kr2 = k2; }

                        score_t += Q_ptr[h * D + d]  * kr1
                                 + Q_ptr[h * D + d2] * kr2;
                    }

                    float t_score = score_t * scale;
                    slot_infos[k].token_scores[t] = t_score;
                    if (t_score > max_score) max_score = t_score;

                    // Advance recurrence to next token (4 mults × half_d, 0 trig)
                    if (has_rope && t + 1 < slen) {
                        for (int d = 0; d < half_d; ++d) {
                            float nc = cos_run[d] * cos_step[d] - sin_run[d] * sin_step[d];
                            float ns = sin_run[d] * cos_step[d] + cos_run[d] * sin_step[d];
                            cos_run[d] = nc;
                            sin_run[d] = ns;
                        }
                    }
                }

            } else {
                // ── Approximate path ─────────────────────────────────────────
                slot_infos[k].q_proj.resize(rank, 0.0f);
                if (use_precomp_rot) {
                    // VK already rotated at anchor_pos — zero trig.
                    for (int r = 0; r < rank; ++r) {
                        float proj = 0.0f;
                        const float* vkr = vk_rot_buf +
                            (size_t)slot_id * rank * n_kv_heads * D +
                            (size_t)r       * n_kv_heads * D + kv_head * D;
                        for (int d = 0; d < D; ++d)
                            proj += Q_ptr[h * D + d] * vkr[d];
                        slot_infos[k].q_proj[r] = proj;
                    }
                } else {
                    // Fallback: reuse per-block ca/sa at anchor_pos — zero new trig.
                    for (int r = 0; r < rank; ++r) {
                        float proj = 0.0f;
                        int bvk = slot_id * rank * n_kv_heads * D +
                                  r * n_kv_heads * D + kv_head * D;
                        if (has_rope) {
                            for (int d = 0; d < half_d; ++d) {
                                float x = ggml_fp16_to_fp32(VK[bvk + d]);
                                float y = ggml_fp16_to_fp32(VK[bvk + d + half_d]);
                                // ca[d]/sa[d] precomputed — zero new trig.
                                proj += Q_ptr[h * D + d]          * (x * ca[d] - y * sa[d]);
                                proj += Q_ptr[h * D + d + half_d] * (y * ca[d] + x * sa[d]);
                            }
                        } else {
                            for (int d = 0; d < D; ++d)
                                proj += Q_ptr[h * D + d] *
                                        ggml_fp16_to_fp32(VK[bvk + d]);
                        }
                        slot_infos[k].q_proj[r] = proj;
                    }
                }

                slot_infos[k].token_scores.resize(slen);
                for (int t = 0; t < slen; ++t) {
                    float delta = 0.0f;
                    int u_off = slot_id * S_max * rank + t * rank;
                    for (int r = 0; r < rank; ++r)
                        delta += slot_infos[k].q_proj[r] * (float)U[u_off + r];

                    // F9: residual-K correction — reuse ca/sa, zero new trig.
                    float res_score = 0.0f;
                    for (int ri = 0; ri < MR; ++ri) {
                        if (res_K_pos[(size_t)slot_id * MR + ri] != t) continue;
                        const ggml_fp16_t* rk = res_K_val +
                            ((size_t)slot_id * MR + ri) * n_kv_heads * D + kv_head * D;
                        if (has_rope) {
                            for (int d = 0; d < half_d; ++d) {
                                float x = ggml_fp16_to_fp32(rk[d]);
                                float y = ggml_fp16_to_fp32(rk[d + half_d]);
                                res_score += Q_ptr[h * D + d]          * (x * ca[d] - y * sa[d]);
                                res_score += Q_ptr[h * D + d + half_d] * (y * ca[d] + x * sa[d]);
                            }
                        } else {
                            for (int d = 0; d < D; ++d)
                                res_score += Q_ptr[h * D + d] *
                                             ggml_fp16_to_fp32(rk[d]);
                        }
                        break;
                    }
                    float t_score = (delta * scale_u * blk_sc + res_score + score_anc) * scale;
                    slot_infos[k].token_scores[t] = t_score;
                    if (t_score > max_score) max_score = t_score;
                }
            }
        }

        // Softmax denominator
        double sum_exp = 0.0;
        for (int k = 0; k < active_K; ++k) {
            sum_exp += std::exp(slot_infos[k].anchor_score * scale - max_score);
            for (float s : slot_infos[k].token_scores)
                sum_exp += std::exp(s - max_score);
        }
        lse_sparse[h] = max_score + std::log(std::max(sum_exp, 1e-9));

        // Value accumulation
        std::vector<double> accum(D, 0.0);
        for (int k = 0; k < active_K; ++k) {
            int slot_id   = active_slots[k];
            int slen      = seq_lens[slot_id];
            float blk_sc  = ggml_fp16_to_fp32(scales[slot_id]);
            float scale_u = ggml_fp16_to_fp32(U_scale_arr[slot_id]);

            double w_anc = std::exp(slot_infos[k].anchor_score * scale - max_score) / sum_exp;
            double sum_w = 0.0;
            std::vector<double> w_proj(rank, 0.0);
            std::vector<double> res_v_accum(D, 0.0);

            for (int t = 0; t < slen; ++t) {
                double w_t = std::exp(slot_infos[k].token_scores[t] - max_score) / sum_exp;
                sum_w += w_t;
                int u_off = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r)
                    w_proj[r] += w_t * (float)U[u_off + r] * scale_u;
                for (int ri = 0; ri < MR; ++ri) {
                    if (res_V_pos[(size_t)slot_id * MR + ri] != t) continue;
                    const ggml_fp16_t* rv = res_V_val +
                        ((size_t)slot_id * MR + ri) * n_kv_heads * D + kv_head * D;
                    for (int d = 0; d < D; ++d)
                        res_v_accum[d] += w_t * ggml_fp16_to_fp32(rv[d]);
                    break;
                }
            }

            double w_total = w_anc + sum_w;
            for (int d = 0; d < D; ++d) {
                accum[d] += w_total * ggml_fp16_to_fp32(
                    anchors_V[slot_id * n_kv_heads * D + kv_head * D + d]);
                double svd_v = 0.0;
                int bvv = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                for (int r = 0; r < rank; ++r)
                    svd_v += w_proj[r] *
                             ggml_fp16_to_fp32(VV[bvv + r * n_kv_heads * D]);
                accum[d] += svd_v * blk_sc + res_v_accum[d];
            }
        }
        for (int d = 0; d < D; ++d) cpu_output[h * D + d] = (float)accum[d];
    }
}

// ── CPU dense window attention helper (used by CPU fallback path only) ────────
void cpu_dense_attention(
    const float* Q_ptr,          // [n_q_heads, D]
    const float* active_k_dense, // [T, n_kv, D]
    const float* active_v_dense, // [T, n_kv, D]
    const int32_t* positions,    // [T] actual sequence positions (or nullptr)
    int T, int n_q_heads, int n_kv_heads, int D,
    float scale, bool has_rope, float rope_freq_base,
    int anchor_pos,              // fallback start position if positions == nullptr
    float* out_dense,            // [n_q_heads, D] — written
    float* lse_dense_out         // [n_q_heads]   — written
) {
    const int g = n_q_heads / n_kv_heads;
    const int half_d = D / 2;

    // theta table: half_d pow calls, once.
    std::vector<float> theta(half_d);
    for (int idx = 0; idx < half_d; ++idx)
        theta[idx] = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);

    // Pre-rotate K for each KV head using pair-wise RoPE (halves trig vs per-d loop).
    // Where positions are consecutive, angular recurrence eliminates per-token trig:
    //   cos_run advances by cos(theta[d]) each step — 4 mults, 0 cos/sin per token.
    std::vector<float> K_rot(T * n_kv_heads * D, 0.0f);

    // Detect consecutive positions for recurrence path.
    bool consecutive = (T > 0);
    if (consecutive && positions != nullptr) {
        int base_pos = positions[0];
        for (int t = 1; t < T; ++t) {
            if (positions[t] != base_pos + t) { consecutive = false; break; }
        }
    }

    for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
        if (!has_rope) {
            // No rotation — copy as-is.
            for (int t = 0; t < T; ++t) {
                int src = t * n_kv_heads * D + kv_head * D;
                int dst = (t * n_kv_heads + kv_head) * D;
                for (int d = 0; d < D; ++d)
                    K_rot[dst + d] = active_k_dense[src + d];
            }
            continue;
        }

        int pos0 = positions ? positions[0] : anchor_pos;

        if (consecutive) {
            // Angular recurrence path: half_d cos+sin for init, 4 mults/dim for steps.
            // Initialise cos_run at pos0, compute cos_step once.
            std::vector<float> cos_run(half_d), sin_run(half_d);
            std::vector<float> cos_stp(half_d), sin_stp(half_d);
            for (int d = 0; d < half_d; ++d) {
                float angle0 = (float)pos0 * theta[d];
                cos_run[d] = std::cos(angle0);
                sin_run[d] = std::sin(angle0);
                cos_stp[d] = std::cos(theta[d]);
                sin_stp[d] = std::sin(theta[d]);
            }
            for (int t = 0; t < T; ++t) {
                int src = t * n_kv_heads * D + kv_head * D;
                int dst = (t * n_kv_heads + kv_head) * D;
                for (int d = 0; d < half_d; ++d) {
                    int d2 = d + half_d;
                    float x = active_k_dense[src + d];
                    float y = active_k_dense[src + d2];
                    K_rot[dst + d]  = x * cos_run[d] - y * sin_run[d];
                    K_rot[dst + d2] = y * cos_run[d] + x * sin_run[d];
                }
                // Advance recurrence (0 trig)
                if (t + 1 < T) {
                    for (int d = 0; d < half_d; ++d) {
                        float nc = cos_run[d] * cos_stp[d] - sin_run[d] * sin_stp[d];
                        float ns = sin_run[d] * cos_stp[d] + cos_run[d] * sin_stp[d];
                        cos_run[d] = nc;
                        sin_run[d] = ns;
                    }
                }
            }
        } else {
            // Non-consecutive positions: pair-wise trig per token (halved vs old per-d loop).
            for (int t = 0; t < T; ++t) {
                int pos = positions ? positions[t] : (anchor_pos + t);
                int src = t * n_kv_heads * D + kv_head * D;
                int dst = (t * n_kv_heads + kv_head) * D;
                for (int d = 0; d < half_d; ++d) {
                    int d2 = d + half_d;
                    float x = active_k_dense[src + d];
                    float y = active_k_dense[src + d2];
                    float angle = (float)pos * theta[d];
                    float c = std::cos(angle), s = std::sin(angle);
                    K_rot[dst + d]  = x * c - y * s;
                    K_rot[dst + d2] = y * c + x * s;
                }
            }
        }
    }

    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;

        float max_s = -1e30f;
        std::vector<float> scores(T);
        for (int t = 0; t < T; ++t) {
            float dot = 0.0f;
            const float* k_t = K_rot.data() + (t * n_kv_heads + kv_head) * D;
            for (int d = 0; d < D; ++d) dot += Q_ptr[h * D + d] * k_t[d];
            scores[t] = dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }
        float sum_e = 0.0f;
        for (int t = 0; t < T; ++t) sum_e += std::exp(scores[t] - max_s);
        lse_dense_out[h] = max_s + std::log(std::max(sum_e, 1e-9f));

        if (std::getenv("DIFFKV_DBG_KDENSE") && h < 7)
            std::cerr << "[DBG_KDENSE] h" << h << " kv=" << kv_head
                      << " current(last)=" << (T > 0 ? scores[T - 1] : 0.0f)
                      << " max=" << max_s
                      << (h == 0 ? " T_dense=" + std::to_string(T) : "") << "\n";

        for (int d = 0; d < D; ++d) out_dense[h * D + d] = 0.0f;
        for (int t = 0; t < T; ++t) {
            float w = std::exp(scores[t] - lse_dense_out[h]);
            const float* v_t = active_v_dense + (t * n_kv_heads + kv_head) * D;
            for (int d = 0; d < D; ++d) out_dense[h * D + d] += w * v_t[d];
        }
    }
}

// ── Main GGML custom-op callback ──────────────────────────────────────────────
void custom_attention_op_callback(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,  // Q  [n_q_heads, D]
    const struct ggml_tensor * b,  // slot_indices [K]
    const struct ggml_tensor * c,  // kv_concat of current token [2, n_kv, D]
    int ith, int nth,
    void * userdata
) {
    if (ith != 0) return;

    const struct ggml_tensor* Q             = a;
    const struct ggml_tensor* slot_indices  = b;
    CustomAttnUserData* data = static_cast<CustomAttnUserData*>(userdata);

    const int n_q_heads = data->n_q_heads;
    const int n_kv_heads = data->n_kv_heads;
    const int D = data->D;
    const int F_kv = n_kv_heads * D;

    bool all_reused = false;
    bool cache_active = (get_global_attn_cache().threshold <= 1.0f);
    std::vector<float> q_host;
    if (cache_active) {
        q_host.resize(n_q_heads * D);
        ggml_backend_tensor_get(Q, q_host.data(), 0, n_q_heads * D * sizeof(float));

        std::vector<bool> reuse_mask(n_q_heads, false);
        std::vector<float> out_cached(n_q_heads * D, 0.0f);
        bool has_cache = get_global_attn_cache().check_and_update(
            data->session_id,
            data->layer_idx,
            q_host.data(),
            n_q_heads,
            D,
            reuse_mask,
            out_cached
        );
        all_reused = has_cache && std::all_of(reuse_mask.begin(), reuse_mask.end(), [](bool b) { return b; });
        if (all_reused) {
            std::memcpy(dst->data, out_cached.data(), n_q_heads * D * sizeof(float));
            return;
        }
    }

    // ── Step 1: Append current token K/V to the dense buffer ─────────────────
    // The dense buffer (active_k_dense / active_v_dense) is a flat CPU array
    // maintained by batch_engine.cpp. When ignore_c == false, we append the
    // current token's projected K/V here so the CPU/Metal kernel can see it.
    int T_dense = data->active_block_tokens;  // tokens already in dense buf

    if (c != nullptr && !data->ignore_c &&
        data->active_k_dense != nullptr && data->active_v_dense != nullptr) {

        // Read current token K/V from GGML tensor c = ggml_concat(k, v)
        std::vector<float> cur_kv(2 * F_kv, 0.0f);
        if (c->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(2 * F_kv);
            ggml_backend_tensor_get(c, tmp.data(), 0, 2 * F_kv * sizeof(ggml_fp16_t));
            for (int i = 0; i < 2 * F_kv; ++i) cur_kv[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            ggml_backend_tensor_get(c, cur_kv.data(), 0, 2 * F_kv * sizeof(float));
        }

        // Bug 3 fix: cache K/V in userdata so batch_engine.cpp can read them after
        // the graph without a second blocking ggml_backend_tensor_get per layer.
        data->captured_kv = cur_kv;

        // Dense buffer bounds guard (Issue 2 fix): only write if capacity allows.
        // dense_capacity is set by batch_engine from session->active_k_dense[l].size()/F_kv.
        // Without this guard, T_dense >= capacity causes a silent buffer overflow that
        // zeros-out the dense K/V window and corrupts every subsequent decode step.
        bool cap_ok = (data->dense_capacity <= 0) || (T_dense < data->dense_capacity);
        if (cap_ok) {
            float* k_dst = data->active_k_dense + (size_t)T_dense * F_kv;
            float* v_dst = data->active_v_dense + (size_t)T_dense * F_kv;
            std::memcpy(k_dst, cur_kv.data(),        F_kv * sizeof(float));
            std::memcpy(v_dst, cur_kv.data() + F_kv, F_kv * sizeof(float));

            // Record the actual sequence position for this token.
            if (data->active_positions_dense != nullptr)
                data->active_positions_dense[T_dense] = data->current_pos;
            T_dense++;
        } else {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr << "[DiffKV] WARNING: dense buffer full (T_dense=" << T_dense
                          << " >= dense_capacity=" << data->dense_capacity
                          << "). Skipping dense write to prevent corruption.\n";
            }
        }
    }

    // ── Step 2: Dispatch GPU or CPU attention ─────────────────────────────────
#ifdef __APPLE__
    const char* env_cpu = std::getenv("DIFFKV_FORCE_CPU_ATTN");
    bool force_cpu = (env_cpu && std::string(env_cpu) == "1");

    if (data->srl_state != nullptr) {
        SessionSRLState* srl = static_cast<SessionSRLState*>(data->srl_state);
        if (!srl->factual_store.entries.empty()) {
            force_cpu = true;
        }
    }

    // F9 (Metal completion): residual corrections are applied only on the CPU path.
    // The Metal kernel intentionally does NOT read residual buffers — porting them
    // (device tensors + upload + 4 buffers + shader edits) is unwarranted because
    // the Metal path is already bypassed whenever a factual store exists (i.e. for
    // ALL salient content, which is what residuals serve). To guarantee correctness
    // in the rare non-factual case, force CPU whenever a routed block actually has
    // residuals, so the exact corrections are never silently skipped.
    if (!force_cpu && data->kv_engine != nullptr && slot_indices && slot_indices->data && data->K > 0) {
        NativeBlockPool* pool = data->kv_engine;
        const int32_t* rkp = pool->get_host_res_K_pos();
        const int32_t* rvp = pool->get_host_res_V_pos();
        const int MR = NativeBlockPool::MAX_RESIDUAL;
        int n_slots = pool->get_seq_lens()->ne[0];
        const int32_t* slots_ptr = (const int32_t*)slot_indices->data;
        for (int k = 0; k < data->K && !force_cpu; ++k) {
            int s = slots_ptr[k];
            if (s < 0 || s >= n_slots) continue;
            if (rkp[(size_t)s * MR] != -1 || rvp[(size_t)s * MR] != -1) force_cpu = true;
        }
    }

    if (!force_cpu) {
        // Unified Metal path: one kernel dispatch handles sparse + dense.
        // Output is written directly into dst->data.
        std::vector<float> lse_dummy(n_q_heads, -1e30f);
        execute_metal_attention(
            dst, Q, (struct ggml_tensor*)slot_indices, data,
            lse_dummy.data(),
            data->active_k_dense,
            data->active_v_dense,
            data->active_positions_dense,
            T_dense
        );


        if (cache_active) {
            get_global_attn_cache().save(data->session_id, data->layer_idx, q_host.data(), n_q_heads, D, (const float*)dst->data);
        }
        return;
    }
#endif

    // ── CPU fallback (non-Apple builds or DIFFKV_FORCE_CPU_ATTN=1) ───────────
    const int K = data->K;
    const int rank = data->rank;
    const int S_max = data->S_max;
    const float scale = data->scale;
    const bool has_rope = data->has_rope;
    const float rope_freq_base = data->rope_freq_base;
    NativeBlockPool* kv_engine = data->kv_engine;

    // Sparse attention (CPU Project-Then-Attend)
    std::vector<float> out_sparse(n_q_heads * D, 0.0f);
    std::vector<float> lse_sparse(n_q_heads, -1e30f);
    if (K > 0 && slot_indices && slot_indices->data) {
        execute_cpu_attention(
            (const float*)Q->data,
            (const int32_t*)slot_indices->data,
            out_sparse.data(), lse_sparse.data(),
            kv_engine,
            n_q_heads, n_kv_heads, rank, S_max, K, D, scale,
            has_rope, rope_freq_base, data->approximate_attn
        );
    }

    // ── Factual Exact Store Attention ──
    // N3.1 fix: The factual store is queried ONCE per decode step by the decode loop
    // in main.cpp (after sampling, using the layer-0 key as proxy, threshold=0.30,
    // active_slots=None, with 1-hop/2-hop neighbor injection and process_and_tag_vsl_step).
    // The callback MUST NOT re-query here — doing so would:
    //   1. Overwrite current_step_factual_{tokens,sequences,max_similarity} written by the
    //      decode loop, discarding richer neighbor-injected + entity-tagged state.
    //   2. Leave current_step_sequence_{entity_ids,is_prime,prefixes} from the decode loop
    //      desynced vs current_step_factual_sequences from the callback (different lengths).
    //   3. Add a 2nd full factual query cost per decode step (descriptor projection + graph walk).
    // The callback's only job here is to READ step_cached_entries (populated by the decode
    // loop at the end of the PREVIOUS step) for K/V exact-attention blending.
    SessionSRLState* srl = (data->srl_state != nullptr) ? static_cast<SessionSRLState*>(data->srl_state) : nullptr;

    // Reference the per-step cached factual entries (populated by the decode loop at
    // end of the previous step); reused across all layers without re-querying.
    static const std::vector<FactEntry> kEmptyEntries;
    const std::vector<FactEntry>& matching_entries =
        (srl != nullptr) ? srl->step_cached_entries : kEmptyEntries;

    std::vector<float> out_facts(n_q_heads * D, 0.0f);
    std::vector<float> lse_facts(n_q_heads, -1e30f);

    std::vector<float> fact_k;
    std::vector<float> fact_v;
    std::vector<int> fact_positions;
    int F_test = n_kv_heads * D;

    for (const auto& entry : matching_entries) {
        int span_len = entry.end_idx - entry.start_idx;
        int offset = data->layer_idx * F_test * span_len;
        fact_k.insert(fact_k.end(), entry.K.begin() + offset, entry.K.begin() + offset + F_test * span_len);
        fact_v.insert(fact_v.end(), entry.V.begin() + offset, entry.V.begin() + offset + F_test * span_len);
        for (int p = entry.start_idx; p < entry.end_idx; ++p) {
            fact_positions.push_back(p);
        }
    }

    int total_fact_len = fact_positions.size();
    if (total_fact_len > 0) {
        int half_d = D / 2;
        std::vector<float> theta(half_d);
        for (int d = 0; d < half_d; ++d) {
            theta[d] = 1.0f / std::pow(rope_freq_base, (2.0f * d) / D);
        }

        std::vector<float> fact_k_rot(total_fact_len * F_test);
        for (int t = 0; t < total_fact_len; ++t) {
            int pos = fact_positions[t];
            for (int kh = 0; kh < n_kv_heads; ++kh) {
                int head_off = t * F_test + kh * D;
                for (int d = 0; d < D; ++d) {
                    float raw_k = fact_k[head_off + d];
                    if (has_rope) {
                        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        float raw_p = fact_k[head_off + partner];
                        float rot_c = (d < half_d) ? -raw_p : raw_p;
                        int idx = (d < half_d) ? d : (d - half_d);
                        float angle = pos * theta[idx];
                        fact_k_rot[head_off + d] = raw_k * std::cos(angle) + rot_c * std::sin(angle);
                    } else {
                        fact_k_rot[head_off + d] = raw_k;
                    }
                }
            }
        }

        const float* Q_ptr = (const float*)Q->data;
        const int g = n_q_heads / n_kv_heads;
        
        for (int h = 0; h < n_q_heads; ++h) {
            int kv_head = h / g;
            float max_s = -1e30f;
            std::vector<float> scores(total_fact_len);
            
            float fact_scale = scale / 0.12f;
            for (int t = 0; t < total_fact_len; ++t) {
                float dot = 0.0f;
                const float* k_t = fact_k_rot.data() + t * F_test + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    dot += Q_ptr[h * D + d] * k_t[d];
                }
                scores[t] = dot * fact_scale;
                if (scores[t] > max_s) max_s = scores[t];
            }
            
            float sum_e = 0.0f;
            for (int t = 0; t < total_fact_len; ++t) {
                sum_e += std::exp(scores[t] - max_s);
            }
            lse_facts[h] = max_s + std::log(std::max(sum_e, 1e-9f));
            
            for (int d = 0; d < D; ++d) out_facts[h * D + d] = 0.0f;
            for (int t = 0; t < total_fact_len; ++t) {
                float w = std::exp(scores[t] - lse_facts[h]);
                const float* v_t = fact_v.data() + t * F_test + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    out_facts[h * D + d] += w * v_t[d];
                }
            }
            // Apply Factual LSE Attention Boosting
            float max_sim = 0.0f;
            for (const auto& entry : matching_entries) {
                if (entry.current_sim > max_sim) {
                    max_sim = entry.current_sim;
                }
            }
            if (max_sim >= 0.4f) {
                float boost = 4.0f * (max_sim - 0.4f) / 0.6f;
                lse_facts[h] += boost;
            }
        }
    }

    // Dense window attention
    std::vector<float> out_dense(n_q_heads * D, 0.0f);
    std::vector<float> lse_dense(n_q_heads, -1e30f);
    if (T_dense > 0) {
        cpu_dense_attention(
            (const float*)Q->data,
            data->active_k_dense, data->active_v_dense,
            data->active_positions_dense,
            T_dense, n_q_heads, n_kv_heads, D,
            scale, has_rope, rope_freq_base,
            data->active_slot,
            out_dense.data(), lse_dense.data()
        );
    }

    // Three-way LSE combine
    std::vector<float> final_out(n_q_heads * D);
    for (int h = 0; h < n_q_heads; ++h) {
        float ld = lse_dense[h];
        float ls = lse_sparse[h];
        float lf = lse_facts[h];
        
        float lse_max = std::max({ld, ls, lf});
        if (lse_max <= -1e20f) {
            for (int d = 0; d < D; ++d) final_out[h * D + d] = 0.0f;
        } else {
            float wd = (std::isinf(ld) || ld <= -1e20f) ? 0.0f : std::exp(ld - lse_max);
            float ws = (std::isinf(ls) || ls <= -1e20f) ? 0.0f : std::exp(ls - lse_max);
            float wf = (std::isinf(lf) || lf <= -1e20f) ? 0.0f : std::exp(lf - lse_max);
            float denom = std::max(wd + ws + wf, 1e-9f);
            for (int d = 0; d < D; ++d) {
                final_out[h * D + d] = (out_dense[h * D + d] * wd +
                                        out_sparse[h * D + d] * ws +
                                        out_facts[h * D + d] * wf) / denom;
            }
        }
    }
    std::memcpy(dst->data, final_out.data(), n_q_heads * D * sizeof(float));
    if (std::getenv("DIFFKV_DBG_ATTN0") && data->layer_idx == 0) {
        double nrm=0, nf=0; for (int i=0;i<n_q_heads*D;++i){ nrm += (double)final_out[i]*final_out[i]; nf += (double)out_facts[i]*out_facts[i]; }
        int facts_active=0; for(int h=0;h<n_q_heads;++h) if(lse_facts[h] > -1e20f) facts_active++;
        std::cerr << "[DBG_ATTN0] CPU L0 attn_out norm=" << std::sqrt(nrm) << " |out_facts|=" << std::sqrt(nf)
                  << " facts_active_heads=" << facts_active << "/" << n_q_heads << " head0[0..5]=";
        for (int i=0;i<6;++i) std::cerr << final_out[i] << " ";
        std::cerr << std::endl;
    }
    if (cache_active) {
        get_global_attn_cache().save(data->session_id, data->layer_idx, q_host.data(), n_q_heads, D, (const float*)dst->data);
    }
}

} // namespace diffkv
