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
#include <atomic>

namespace diffkv {

// Unconditional invocation counter for custom_attention_op_callback (diagnostic: is the live
// sparse-attention custom op actually executed by the sched?). Read from main after decode.
std::atomic<long> g_diffkv_cb_invocations{0};
std::atomic<int>  g_diffkv_dbg_pos{0};   // mirror of DIFFKV_DBG_POS, set from main (worker-thread getenv fails)
std::atomic<long> g_cpu_attn_count{0};   // execute_cpu_attention entries (live path = CPU?)
std::atomic<long> g_metal_attn_count{0}; // execute_metal_attention entries (live path = Metal?)

// ── CPU Project-Then-Attend (reference / non-Apple fallback) ──────────────────
// On Apple, the Metal kernel handles the normal path; this function handles the
// CPU-forced fallback (factual_store non-empty, residuals present, etc.).
//
// KEY PERFORMANCE DESIGN:
//   VK/VV are stored as [slot, rank, kv_heads, D] in fp16. Accessing consecutive
//   'r' values for a fixed (slot, kv_head) jumps n_kv_heads×D×2 = 2048 bytes —
//   a cache miss per r iteration. Fix: precompute vk_local[rank, D] per
//   (block, kv_head transition) into a contiguous fp32 buffer (8 KB → fits L1).
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
    const ggml_fp16_t* U_scale_arr = kv_engine->get_host_U_scale();
    const int32_t* seq_lens = kv_engine->get_host_seq_lens();
    const ggml_fp16_t* scales = kv_engine->get_host_scales();
    const int32_t* anchor_positions = kv_engine->get_host_anchor_positions();
    const int MR = NativeBlockPool::MAX_RESIDUAL;

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

    g_cpu_attn_count.fetch_add(1, std::memory_order_relaxed);
    if (g_diffkv_dbg_pos.load(std::memory_order_relaxed)) {
        static int once = 0;
        if (once++ == 0) {
            std::cerr << "[DBG_POS] active_K=" << active_K << " S_max=" << S_max << "\n";
            for (int k = 0; k < active_K && k < 8; ++k) {
                int sid = active_slots[k];
                const int32_t* slot_token_positions = kv_engine->get_host_token_positions(sid);
                std::cerr << "[DBG_POS] slot " << sid << " anchor_pos=" << anchor_positions[sid]
                          << " seq_len=" << seq_lens[sid] << " tok_pos[0..4]=";
                for (int t = 0; t < seq_lens[sid] && t < 5; ++t)
                    std::cerr << (slot_token_positions ? slot_token_positions[t] : 0) << " ";
                std::cerr << "\n";
            }
        }
    }

    const int half_d = D / 2;
    const int g = n_q_heads / n_kv_heads;

    const float* vk_rot_buf  = kv_engine->get_host_VK_rot();
    const bool use_precomp_rot = (has_rope && vk_rot_buf != nullptr);

    std::vector<float> theta_table;
    if (has_rope) {
        theta_table.resize(half_d);
        for (int i = 0; i < half_d; ++i)
            theta_table[i] = 1.0f / std::pow(rope_freq_base, (2.0f * i) / D);
    }

    std::vector<float> cos_step(half_d, 1.0f), sin_step(half_d, 0.0f);
    if (has_rope && !approximate_attn) {
        for (int d = 0; d < half_d; ++d) {
            cos_step[d] = std::cos(theta_table[d]);
            sin_step[d] = std::sin(theta_table[d]);
        }
    }

    struct BlockRope { std::vector<float> ca, sa; };
    std::vector<BlockRope> blk_rope(active_K);
    if (has_rope) {
        for (int k = 0; k < active_K; ++k) {
            int ap = anchor_positions[active_slots[k]];
            blk_rope[k].ca.resize(half_d);
            blk_rope[k].sa.resize(half_d);
            for (int d = 0; d < half_d; ++d) {
                double angle = std::fmod((double)ap * (double)theta_table[d], 2.0 * M_PI);
                blk_rope[k].ca[d] = (float)std::cos(angle);
                blk_rope[k].sa[d] = (float)std::sin(angle);
            }
        }
    }

    int prev_kv_head = -1;
    std::vector<std::vector<float>> blk_vk_local(active_K);
    std::vector<std::vector<float>> blk_vv_local(active_K);
    std::vector<std::vector<float>> blk_anc_v(active_K);

    struct SlotInfo {
        float anchor_score;
        std::vector<float> q_proj;
        std::vector<float> token_scores;
    };
    std::vector<SlotInfo> slot_infos(active_K);

    std::vector<std::vector<float>> tok_cos(active_K), tok_sin(active_K);
    if (has_rope && !approximate_attn) {
        for (int k = 0; k < active_K; ++k) {
            int slot_id = active_slots[k];
            int slen = seq_lens[slot_id];
            tok_cos[k].resize((size_t)slen * half_d);
            tok_sin[k].resize((size_t)slen * half_d);
            const int32_t* slot_token_positions = kv_engine->get_host_token_positions(slot_id);
            for (int t = 0; t < slen; ++t) {
                double tpos = slot_token_positions ? (double)slot_token_positions[t] : 0.0;
                for (int d = 0; d < half_d; ++d) {
                    double ang = std::fmod(tpos * (double)theta_table[d], 2.0 * M_PI);
                    tok_cos[k][(size_t)t * half_d + d] = (float)std::cos(ang);
                    tok_sin[k][(size_t)t * half_d + d] = (float)std::sin(ang);
                }
            }
        }
    }

    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;

        if (kv_head != prev_kv_head) {
            prev_kv_head = kv_head;
            for (int k = 0; k < active_K; ++k) {
                int slot_id = active_slots[k];
                const ggml_fp16_t* slot_VK = kv_engine->get_host_VK(slot_id);
                const ggml_fp16_t* slot_VV = kv_engine->get_host_VV(slot_id);
                const ggml_fp16_t* slot_anchors_V = kv_engine->get_host_anchors_V(slot_id);
                int base_vk = kv_head * D;
                int base_vv = kv_head * D;
                const float* ca = has_rope ? blk_rope[k].ca.data() : nullptr;
                const float* sa = has_rope ? blk_rope[k].sa.data() : nullptr;

                blk_vk_local[k].resize(rank * D);
                blk_vv_local[k].resize(rank * D);
                blk_anc_v[k].resize(D);

                if (use_precomp_rot && approximate_attn) {
                    const float* base = vk_rot_buf +
                        (size_t)slot_id * rank * n_kv_heads * D + kv_head * D;
                    for (int r = 0; r < rank; ++r)
                        for (int d = 0; d < D; ++d)
                            blk_vk_local[k][r * D + d] = base[r * n_kv_heads * D + d];
                } else if (has_rope && approximate_attn) {
                    for (int r = 0; r < rank; ++r) {
                        int bvk = base_vk + r * n_kv_heads * D;
                        for (int d = 0; d < half_d; ++d) {
                            float x = slot_VK ? ggml_fp16_to_fp32(slot_VK[bvk + d]) : 0.0f;
                            float y = slot_VK ? ggml_fp16_to_fp32(slot_VK[bvk + d + half_d]) : 0.0f;
                            blk_vk_local[k][r * D + d]          = x * ca[d] - y * sa[d];
                            blk_vk_local[k][r * D + d + half_d] = y * ca[d] + x * sa[d];
                        }
                    }
                } else {
                    for (int r = 0; r < rank; ++r) {
                        int bvk = base_vk + r * n_kv_heads * D;
                        for (int d = 0; d < D; ++d)
                            blk_vk_local[k][r * D + d] = slot_VK ? ggml_fp16_to_fp32(slot_VK[bvk + d]) : 0.0f;
                    }
                }

                for (int r = 0; r < rank; ++r) {
                    int bvv = base_vv + r * n_kv_heads * D;
                    for (int d = 0; d < D; ++d)
                        blk_vv_local[k][r * D + d] = slot_VV ? ggml_fp16_to_fp32(slot_VV[bvv + d]) : 0.0f;
                }

                int av_base = kv_head * D;
                for (int d = 0; d < D; ++d)
                    blk_anc_v[k][d] = slot_anchors_V ? ggml_fp16_to_fp32(slot_anchors_V[av_base + d]) : 0.0f;
            }
        }

        float max_score = -1e30f;

        for (int k = 0; k < active_K; ++k) {
            int slot_id   = active_slots[k];
            int slen      = seq_lens[slot_id];
            const ggml_fp16_t* u_row_scale = kv_engine->get_host_U_row_scale(slot_id);
            float blk_sc  = ggml_fp16_to_fp32(scales[slot_id]);

            const float* ca = has_rope ? blk_rope[k].ca.data() : nullptr;
            const float* sa = has_rope ? blk_rope[k].sa.data() : nullptr;

            float score_anc = 0.0f;
            {
                const ggml_fp16_t* slot_anchors_K = kv_engine->get_host_anchors_K(slot_id);
                int ak_off = kv_head * D;
                if (has_rope) {
                    for (int d = 0; d < half_d; ++d) {
                        float x = slot_anchors_K ? ggml_fp16_to_fp32(slot_anchors_K[ak_off + d]) : 0.0f;
                        float y = slot_anchors_K ? ggml_fp16_to_fp32(slot_anchors_K[ak_off + d + half_d]) : 0.0f;
                        score_anc += Q_ptr[h * D + d]          * (x * ca[d] - y * sa[d]);
                        score_anc += Q_ptr[h * D + d + half_d] * (y * ca[d] + x * sa[d]);
                    }
                } else {
                    for (int d = 0; d < D; ++d)
                        score_anc += Q_ptr[h * D + d] *
                                     (slot_anchors_K ? ggml_fp16_to_fp32(slot_anchors_K[ak_off + d]) : 0.0f);
                }
            }
            slot_infos[k].anchor_score = score_anc;
            { float s = score_anc * scale; if (s > max_score) max_score = s; }

            if (approximate_attn) {
                slot_infos[k].q_proj.assign(rank, 0.0f);
                const float* vkl = blk_vk_local[k].data();
                for (int r = 0; r < rank; ++r) {
                    float proj = 0.0f;
                    const float* vkr = vkl + r * D;
                    for (int d = 0; d < D; ++d)
                        proj += Q_ptr[h * D + d] * vkr[d];
                    slot_infos[k].q_proj[r] = proj;
                }

                slot_infos[k].token_scores.resize(slen);
                const int8_t* slot_U = kv_engine->get_host_U(slot_id);
                const int32_t* slot_res_K_pos = kv_engine->get_host_res_K_pos(slot_id);
                const ggml_fp16_t* slot_res_K_val = kv_engine->get_host_res_K_val(slot_id);

                for (int t = 0; t < slen; ++t) {
                    float delta = 0.0f;
                    const int8_t* u_row = slot_U ? (slot_U + t * rank) : nullptr;
                    if (u_row) {
                        for (int r = 0; r < rank; ++r)
                            delta += slot_infos[k].q_proj[r] * (float)u_row[r];
                    }

                    float res_score = 0.0f;
                    if (slot_res_K_pos && slot_res_K_val) {
                        for (int ri = 0; ri < MR; ++ri) {
                            if (slot_res_K_pos[ri] != t) continue;
                            const ggml_fp16_t* rk = slot_res_K_val +
                                ri * n_kv_heads * D + kv_head * D;
                            if (has_rope) {
                                for (int d = 0; d < half_d; ++d) {
                                    float x = ggml_fp16_to_fp32(rk[d]);
                                    float y = ggml_fp16_to_fp32(rk[d + half_d]);
                                    res_score += Q_ptr[h * D + d]          * (x * ca[d] - y * sa[d]);
                                    res_score += Q_ptr[h * D + d + half_d] * (y * ca[d] + x * sa[d]);
                                }
                            } else {
                                for (int d = 0; d < D; ++d)
                                    res_score += Q_ptr[h * D + d] * ggml_fp16_to_fp32(rk[d]);
                            }
                            break;
                        }
                    }
                    float t_score = (delta * ggml_fp16_to_fp32(u_row_scale[t]) * blk_sc + res_score + score_anc) * scale;
                    slot_infos[k].token_scores[t] = t_score;
                    if (t_score > max_score) max_score = t_score;
                }
            } else {
                const ggml_fp16_t* slot_anchors_K = kv_engine->get_host_anchors_K(slot_id);
                int ak_off = kv_head * D;
                std::vector<float> anc_k_f(D);
                for (int d = 0; d < D; ++d)
                    anc_k_f[d] = slot_anchors_K ? ggml_fp16_to_fp32(slot_anchors_K[ak_off + d]) : 0.0f;

                slot_infos[k].token_scores.resize(slen);
                const int8_t* slot_U = kv_engine->get_host_U(slot_id);
                const int32_t* slot_res_K_pos = kv_engine->get_host_res_K_pos(slot_id);
                const ggml_fp16_t* slot_res_K_val = kv_engine->get_host_res_K_val(slot_id);
                const float* vkl = blk_vk_local[k].data();

                for (int t = 0; t < slen; ++t) {
                    float score_t = 0.0f;
                    const int8_t* u_row = slot_U ? (slot_U + t * rank) : nullptr;
                    const ggml_fp16_t* rk_res = nullptr;
                    if (slot_res_K_pos && slot_res_K_val) {
                        for (int ri = 0; ri < MR; ++ri) {
                            if (slot_res_K_pos[ri] == t) {
                                rk_res = slot_res_K_val + ri * n_kv_heads * D + kv_head * D;
                                break;
                            }
                        }
                    }
                    for (int d = 0; d < half_d; ++d) {
                        int d2 = d + half_d;
                        float dk1 = 0.0f, dk2 = 0.0f;
                        if (u_row) {
                            for (int r = 0; r < rank; ++r) {
                                float ur = (float)u_row[r];
                                dk1 += ur * vkl[r * D + d];
                                dk2 += ur * vkl[r * D + d2];
                            }
                        }
                        float ku = ggml_fp16_to_fp32(u_row_scale[t]);
                        float k1 = anc_k_f[d]  + dk1 * ku * blk_sc;
                        float k2 = anc_k_f[d2] + dk2 * ku * blk_sc;
                        if (rk_res) { k1 += ggml_fp16_to_fp32(rk_res[d]); k2 += ggml_fp16_to_fp32(rk_res[d2]); }
                        float kr1, kr2;
                        if (has_rope) {
                            float cr = tok_cos[k][(size_t)t * half_d + d];
                            float sr = tok_sin[k][(size_t)t * half_d + d];
                            kr1 = k1 * cr - k2 * sr;
                            kr2 = k2 * cr + k1 * sr;
                        } else { kr1 = k1; kr2 = k2; }
                        score_t += Q_ptr[h * D + d]  * kr1 + Q_ptr[h * D + d2] * kr2;
                    }
                    float t_score = score_t * scale;
                    slot_infos[k].token_scores[t] = t_score;
                    if (t_score > max_score) max_score = t_score;
                }
                slot_infos[k].q_proj.clear();
            }
        }

        double sum_exp = 0.0;
        for (int k = 0; k < active_K; ++k) {
            int slot_id = active_slots[k];
            int slen = seq_lens[slot_id];
            sum_exp += std::exp(slot_infos[k].anchor_score * scale - max_score);
            for (int t = 0; t < slen; ++t)
                sum_exp += std::exp(slot_infos[k].token_scores[t] - max_score);
        }
        lse_sparse[h] = max_score + std::log(std::max(sum_exp, 1e-9));

        std::vector<double> accum(D, 0.0);
        for (int k = 0; k < active_K; ++k) {
            int slot_id   = active_slots[k];
            int slen      = seq_lens[slot_id];
            float blk_sc  = ggml_fp16_to_fp32(scales[slot_id]);
            const ggml_fp16_t* u_row_scale = kv_engine->get_host_U_row_scale(slot_id);
            const float* vvl = blk_vv_local[k].data();
            const float* av  = blk_anc_v[k].data();

            double w_anc = std::exp(slot_infos[k].anchor_score * scale - max_score) / sum_exp;
            double sum_w = 0.0;
            std::vector<double> w_proj(rank, 0.0);
            std::vector<double> res_v_accum(D, 0.0);
            const int8_t* slot_U = kv_engine->get_host_U(slot_id);
            const int32_t* slot_res_V_pos = kv_engine->get_host_res_V_pos(slot_id);
            const ggml_fp16_t* slot_res_V_val = kv_engine->get_host_res_V_val(slot_id);

            for (int t = 0; t < slen; ++t) {
                double w_t = std::exp(slot_infos[k].token_scores[t] - max_score) / sum_exp;
                sum_w += w_t;
                const int8_t* u_row = slot_U ? (slot_U + t * rank) : nullptr;
                float ku = ggml_fp16_to_fp32(u_row_scale[t]);
                if (u_row) {
                    for (int r = 0; r < rank; ++r)
                        w_proj[r] += w_t * (float)u_row[r] * ku;
                }
                if (slot_res_V_pos && slot_res_V_val) {
                    for (int ri = 0; ri < MR; ++ri) {
                        if (slot_res_V_pos[ri] != t) continue;
                        const ggml_fp16_t* rv = slot_res_V_val +
                            ri * n_kv_heads * D + kv_head * D;
                        for (int d = 0; d < D; ++d)
                            res_v_accum[d] += w_t * ggml_fp16_to_fp32(rv[d]);
                        break;
                    }
                }
            }

            double w_total = w_anc + sum_w;
            std::vector<double> svd_v(D, 0.0);
            for (int r = 0; r < rank; ++r) {
                double wr = w_proj[r] * blk_sc;
                const float* vvr = vvl + r * D;
                for (int d = 0; d < D; ++d)
                    svd_v[d] += wr * vvr[d];
            }

            for (int d = 0; d < D; ++d)
                accum[d] += w_total * av[d] + svd_v[d] + res_v_accum[d];
        }
        for (int d = 0; d < D; ++d) cpu_output[h * D + d] = (float)accum[d];
    }
}

// ── CPU dense window attention helper ────────────────────────────────────────
void cpu_dense_attention(
    const float* Q_ptr,          // [n_q_heads, D]
    const float* active_k_dense, // [T, n_kv, D]
    const float* active_v_dense, // [T, n_kv, D]
    const int32_t* positions,    // [T] actual sequence positions (or nullptr)
    int T, int n_q_heads, int n_kv_heads, int D,
    float scale, bool has_rope, float rope_freq_base,
    int anchor_pos,
    float* out_dense,
    float* lse_dense_out
) {
    const int g = n_q_heads / n_kv_heads;
    const int half_d = D / 2;

    // theta table: half_d pow calls, once.
    std::vector<float> theta(half_d);
    for (int idx = 0; idx < half_d; ++idx)
        theta[idx] = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);

    // Detect whether positions are consecutive (enables zero-trig recurrence path).
    bool consecutive = (T > 0);
    if (consecutive && positions != nullptr) {
        int base_pos = positions[0];
        for (int t = 1; t < T; ++t) {
            if (positions[t] != base_pos + t) { consecutive = false; break; }
        }
    }
    if (std::getenv("DIFFKV_DENSE_DIRECT")) consecutive = false; // force per-token cos/sin (no recurrence)

    // Pre-rotate K for each KV head.
    std::vector<float> K_rot(T * n_kv_heads * D, 0.0f);

    for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
        if (!has_rope) {
            for (int t = 0; t < T; ++t) {
                int src = t * n_kv_heads * D + kv_head * D;
                int dst = (t * n_kv_heads + kv_head) * D;
                for (int d = 0; d < D; ++d) K_rot[dst + d] = active_k_dense[src + d];
            }
            continue;
        }

        int pos0 = positions ? positions[0] : anchor_pos;

        if (consecutive) {
            // Angular recurrence: half_d cos+sin for init, 4 mults/dim for steps.
            std::vector<float> cos_run(half_d), sin_run(half_d);
            std::vector<float> cos_stp(half_d), sin_stp(half_d);
            for (int d = 0; d < half_d; ++d) {
                double angle0 = std::fmod((double)pos0 * (double)theta[d], 2.0 * M_PI);
                cos_run[d] = (float)std::cos(angle0);
                sin_run[d] = (float)std::sin(angle0);
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
                if (t + 1 < T) {
                    for (int d = 0; d < half_d; ++d) {
                        float nc = cos_run[d] * cos_stp[d] - sin_run[d] * sin_stp[d];
                        float ns = sin_run[d] * cos_stp[d] + cos_run[d] * sin_stp[d];
                        cos_run[d] = nc; sin_run[d] = ns;
                    }
                }
            }
        } else {
            // Non-consecutive: pair-wise trig (halved vs old per-d loop).
            for (int t = 0; t < T; ++t) {
                int pos = positions ? positions[t] : (anchor_pos + t);
                int src = t * n_kv_heads * D + kv_head * D;
                int dst = (t * n_kv_heads + kv_head) * D;
                for (int d = 0; d < half_d; ++d) {
                    int d2 = d + half_d;
                    float x = active_k_dense[src + d];
                    float y = active_k_dense[src + d2];
                    double angle = std::fmod((double)pos * (double)theta[d], 2.0 * M_PI);
                    float c = (float)std::cos(angle), s = (float)std::sin(angle);
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
                      << " last=" << (T > 0 ? scores[T - 1] : 0.0f)
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
    g_diffkv_cb_invocations.fetch_add(1, std::memory_order_relaxed);  // unconditional: did the callback run?
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
            data->session_id, data->layer_idx, q_host.data(), n_q_heads, D,
            reuse_mask, out_cached
        );
        all_reused = has_cache && std::all_of(reuse_mask.begin(), reuse_mask.end(), [](bool b) { return b; });
        if (all_reused) {
            std::memcpy(dst->data, out_cached.data(), n_q_heads * D * sizeof(float));
            return;
        }
    }

    // ── Step 1: Append current token K/V to the dense buffer ─────────────────
    int T_dense = data->active_block_tokens;

    if (c != nullptr && !data->ignore_c &&
        data->active_k_dense != nullptr && data->active_v_dense != nullptr) {

        std::vector<float> cur_kv(2 * F_kv, 0.0f);
        if (c->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(2 * F_kv);
            ggml_backend_tensor_get(c, tmp.data(), 0, 2 * F_kv * sizeof(ggml_fp16_t));
            for (int i = 0; i < 2 * F_kv; ++i) cur_kv[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            ggml_backend_tensor_get(c, cur_kv.data(), 0, 2 * F_kv * sizeof(float));
        }

        // Bug 3 fix: cache K/V for batch_engine to read after graph without extra GPU readback.
        data->captured_kv = cur_kv;

        // Dense buffer bounds guard (Issue 2 fix): prevents overflow → corruption.
        bool cap_ok = (data->dense_capacity <= 0) || (T_dense < data->dense_capacity);
        if (cap_ok) {
            float* k_dst = data->active_k_dense + (size_t)T_dense * F_kv;
            float* v_dst = data->active_v_dense + (size_t)T_dense * F_kv;
            std::memcpy(k_dst, cur_kv.data(),        F_kv * sizeof(float));
            std::memcpy(v_dst, cur_kv.data() + F_kv, F_kv * sizeof(float));
            if (data->active_positions_dense != nullptr)
                data->active_positions_dense[T_dense] = data->current_pos;
            T_dense++;
        } else {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr << "[DiffKV] WARNING: dense buffer full (T_dense=" << T_dense
                          << " >= dense_capacity=" << data->dense_capacity
                          << "). Skipping dense write.\n";
            }
        }
    }

    // ── Step 2: Dispatch GPU or CPU attention ─────────────────────────────────
#ifdef __APPLE__
    const char* env_cpu = std::getenv("DIFFKV_FORCE_CPU_ATTN");
    bool force_cpu = (env_cpu && std::string(env_cpu) == "1");

    // §3.8: We no longer force CPU attention when step_cached_entries is non-empty,
    // as factual K/V attention injection has been removed (matching HF reference).

    // F9: force CPU if any selected block has residuals (Metal doesn't handle them).
    int actual_K = (slot_indices != nullptr) ? (int)slot_indices->ne[0] : 0;
    if (!force_cpu && data->kv_engine != nullptr && slot_indices && slot_indices->data && actual_K > 0) {
        NativeBlockPool* pool = data->kv_engine;
        int n_slots = pool->get_seq_lens()->ne[0];
        const int32_t* slots_ptr = (const int32_t*)slot_indices->data;
        for (int k = 0; k < actual_K && !force_cpu; ++k) {
            int s = slots_ptr[k];
            if (s < 0 || s >= n_slots) continue;
            const int32_t* rkp_s = pool->get_host_res_K_pos(s);
            const int32_t* rvp_s = pool->get_host_res_V_pos(s);
            if (rkp_s && rvp_s) {
                if (rkp_s[0] != -1 || rvp_s[0] != -1) {
                    force_cpu = true;
                }
            }
        }
    }

    if (!force_cpu) {
        std::vector<float> lse_dummy(n_q_heads, -1e30f);
        execute_metal_attention(
            dst, Q, (struct ggml_tensor*)slot_indices, data,
            lse_dummy.data(),
            data->active_k_dense, data->active_v_dense,
            data->active_positions_dense, T_dense
        );
        if (cache_active)
            get_global_attn_cache().save(data->session_id, data->layer_idx, q_host.data(), n_q_heads, D, (const float*)dst->data);
        return;
    }
#endif

    // ── CPU fallback ─────────────────────────────────────────────────────────
    const int K = (slot_indices != nullptr) ? (int)slot_indices->ne[0] : 0;
    const int rank = data->rank;
    const int S_max = data->S_max;
    const float scale = data->scale;
    const bool has_rope = data->has_rope;
    const float rope_freq_base = data->rope_freq_base;
    NativeBlockPool* kv_engine = data->kv_engine;

    // Sync Q from GPU (Metal backend) to host memory before CPU computation.
    // Q->data may not yet reflect completed GPU writes; ggml_backend_tensor_get
    // issues the correct backend-specific sync/copy.
    std::vector<float> q_cpu(n_q_heads * D);
    ggml_backend_tensor_get(Q, q_cpu.data(), 0, n_q_heads * D * sizeof(float));
    const float* Q_cpu = q_cpu.data();

    std::vector<float> out_sparse(n_q_heads * D, 0.0f);
    std::vector<float> lse_sparse(n_q_heads, -1e30f);
    if (K > 0 && slot_indices && slot_indices->data) {
        execute_cpu_attention(
            Q_cpu,
            (const int32_t*)slot_indices->data,
            out_sparse.data(), lse_sparse.data(),
            kv_engine,
            n_q_heads, n_kv_heads, rank, S_max, K, D, scale,
            has_rope, rope_freq_base, data->approximate_attn
        );
    }

    // Dense window attention
    std::vector<float> out_dense(n_q_heads * D, 0.0f);
    std::vector<float> lse_dense(n_q_heads, -1e30f);
    if (T_dense > 0) {
        cpu_dense_attention(
            Q_cpu,
            data->active_k_dense, data->active_v_dense,
            data->active_positions_dense,
            T_dense, n_q_heads, n_kv_heads, D,
            scale, has_rope, rope_freq_base,
            data->active_slot,
            out_dense.data(), lse_dense.data()
        );
    }
    if (std::getenv("DIFFKV_DBG_INPUTS") && data->layer_idx == 0) {
        static int s = 0;
        if (s < 2) { s++;
            auto nrm = [&](const float* p, int n){ double a=0; for(int i=0;i<n;++i) a+=(double)p[i]*p[i]; return std::sqrt(a); };
            fprintf(stderr, "[DBG_IN] L0 T_dense=%d has_rope=%d rope_base=%.1f scale=%.5f pos0=%d posLast=%d\n",
                    T_dense, (int)has_rope, rope_freq_base, scale,
                    data->active_positions_dense?data->active_positions_dense[0]:-1,
                    data->active_positions_dense?data->active_positions_dense[T_dense-1]:-1);
            fprintf(stderr, "[DBG_IN]  |Q|=%.3f Q[0..4]=%.3f %.3f %.3f %.3f\n", nrm(Q_cpu,n_q_heads*D), Q_cpu[0],Q_cpu[1],Q_cpu[2],Q_cpu[3]);
            fprintf(stderr, "[DBG_IN]  |K0|=%.3f K0[0..4]=%.3f %.3f %.3f %.3f  |Klast|=%.3f\n",
                    nrm(data->active_k_dense, D), data->active_k_dense[0],data->active_k_dense[1],data->active_k_dense[2],data->active_k_dense[3],
                    nrm(data->active_k_dense+(size_t)(T_dense-1)*n_kv_heads*D, D));
            fprintf(stderr, "[DBG_IN]  |out_dense|=%.3f out[0..4]=%.4f %.4f %.4f %.4f lse=%.2f\n",
                    nrm(out_dense.data(),D), out_dense[0],out_dense[1],out_dense[2],out_dense[3], lse_dense[0]);
        }
    }

    // Two-way LSE combine (sparse ⊕ dense)
    static const bool dbg_sparseoff = (std::getenv("DIFFKV_DBG_SPARSEOFF") != nullptr);
    static const bool dbg_denseoff  = (std::getenv("DIFFKV_DBG_DENSEOFF2") != nullptr);
    std::vector<float> final_out(n_q_heads * D);
    for (int h = 0; h < n_q_heads; ++h) {
        float ld = lse_dense[h], ls = lse_sparse[h];
        if (dbg_sparseoff) ls = -1e30f;   // attend dense window only
        if (dbg_denseoff)  ld = -1e30f;   // attend compressed blocks only
        float lse_max = std::max(ld, ls);
        if (lse_max <= -1e20f) {
            for (int d = 0; d < D; ++d) final_out[h * D + d] = 0.0f;
        } else {
            float wd = (std::isinf(ld) || ld <= -1e20f) ? 0.0f : std::exp(ld - lse_max);
            float ws = (std::isinf(ls) || ls <= -1e20f) ? 0.0f : std::exp(ls - lse_max);
            float denom = std::max(wd + ws, 1e-9f);
            for (int d = 0; d < D; ++d)
                final_out[h * D + d] = (out_dense[h * D + d] * wd +
                                        out_sparse[h * D + d] * ws) / denom;
        }
    }
    std::memcpy(dst->data, final_out.data(), n_q_heads * D * sizeof(float));

    if (std::getenv("DIFFKV_DBG_ATTN0") && data->layer_idx == 0) {
        static int dbg_step = 0;
        if (dbg_step < 4) {
            double ns=0, nd=0;
            for (int i=0;i<D;++i){ ns += (double)out_sparse[i]*out_sparse[i]; nd += (double)out_dense[i]*out_dense[i]; }
            std::cerr << "[DBG_ATTN0] step=" << dbg_step << " K=" << K << " T_dense=" << T_dense
                      << " | h0 lse_sparse=" << lse_sparse[0] << " lse_dense=" << lse_dense[0]
                      << " |out_sparse|=" << std::sqrt(ns) << " |out_dense|=" << std::sqrt(nd) << "\n";
            dbg_step++;
        }
    }
    if (cache_active)
        get_global_attn_cache().save(data->session_id, data->layer_idx, q_host.data(), n_q_heads, D, (const float*)dst->data);
}

} // namespace diffkv
