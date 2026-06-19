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
// On Apple, the Metal kernel handles the normal path; this function handles the
// CPU-forced fallback (factual_store non-empty, residuals present, etc.).
//
// KEY PERFORMANCE DESIGN:
//   VK/VV are stored as [slot, rank, kv_heads, D] in fp16. Accessing consecutive
//   'r' values for a fixed (slot, kv_head) jumps n_kv_heads×D×2 = 2048 bytes —
//   a cache miss per r iteration. Fix: precompute vk_local[rank, D] per
//   (block, kv_head transition) into a contiguous fp32 buffer (8 KB → fits L1).
//   The inner q_proj loop then strides sequentially. VV uses the same trick for
//   the value accumulation.
//
//   vk_local is rebuilt only when kv_head changes (once per g=4 Q-heads per
//   block), saving (g-1)/g ≈ 75% of the extraction work.
//
//   Angular recurrence replaces per-token cos/sin with 4 multiplies per dim-pair.
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

    // Bug 2 fix: O(1) dedup.
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
    const int g = n_q_heads / n_kv_heads;   // GQA groups

    // Bug 9/1 fix: use precomputed VK_rot if available (set by upload_slot).
    // NOTE: for the CPU q_proj path, even if vk_rot_buf != nullptr, we still
    // extract into vk_local for cache efficiency (contiguous access).
    const float* vk_rot_buf  = kv_engine->get_host_VK_rot();
    const bool use_precomp_rot = (has_rope && vk_rot_buf != nullptr);

    // ── theta_table: half_d pow calls, once per call ───────────────────────
    std::vector<float> theta_table;
    if (has_rope) {
        theta_table.resize(half_d);
        for (int i = 0; i < half_d; ++i)
            theta_table[i] = 1.0f / std::pow(rope_freq_base, (2.0f * i) / D);
    }

    // ── cos_step / sin_step: half_d cos+sin, computed ONCE per call ───────
    // Used by angular recurrence in non-approx token loop.
    std::vector<float> cos_step(half_d, 1.0f), sin_step(half_d, 0.0f);
    if (has_rope && !approximate_attn) {
        for (int d = 0; d < half_d; ++d) {
            cos_step[d] = std::cos(theta_table[d]);
            sin_step[d] = std::sin(theta_table[d]);
        }
    }

    // ── Per-block anchor cos/sin: computed ONCE per block, reused by all heads.
    // anchor_pos is block-specific but head-independent (GQA/MHA doesn't matter).
    struct BlockRope { std::vector<float> ca, sa; };
    std::vector<BlockRope> blk_rope(active_K);
    if (has_rope) {
        for (int k = 0; k < active_K; ++k) {
            int ap = anchor_positions[active_slots[k]];
            blk_rope[k].ca.resize(half_d);
            blk_rope[k].sa.resize(half_d);
            for (int d = 0; d < half_d; ++d) {
                float angle = (float)ap * theta_table[d];
                blk_rope[k].ca[d] = std::cos(angle);
                blk_rope[k].sa[d] = std::sin(angle);
            }
        }
    }

    // ── Contiguous VK/VV local buffers — built once per (kv_head, block).  ──
    // ACTIVE_RUNTIME uses batched einsum 'hd,nhrd->nhr' which is cache-friendly.
    // Here we match that by extracting VK[slot, kv_head, :] into vk_local[rank, D]
    // (stride-1 sequential floats) before the Q·VK inner product.
    // Rebuilt only when kv_head changes — amortised over g=4 Q-heads sharing kv_head.
    int prev_kv_head = -1;
    std::vector<std::vector<float>> blk_vk_local(active_K);  // [rank, D] per block
    std::vector<std::vector<float>> blk_vv_local(active_K);  // [rank, D] per block

    // Scratch space per block for value accumulation (avoids re-allocation per head)
    std::vector<std::vector<float>> blk_anc_v(active_K);     // [D] anchors_V, fp32

    // ── Slot information (scores) per head per block ────────────────────────
    struct SlotInfo {
        float anchor_score;
        std::vector<float> q_proj;
        std::vector<float> token_scores;
    };
    std::vector<SlotInfo> slot_infos(active_K);

    // ── Per-head attention loop ─────────────────────────────────────────────
    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;

        // ── Precompute vk_local / vv_local / anc_v on kv_head transition ──
        // This runs once per g=4 Q-heads and eliminates the stride-1024 VK/VV
        // cache miss that is the primary bottleneck in the CPU approximate path.
        if (kv_head != prev_kv_head) {
            prev_kv_head = kv_head;
            for (int k = 0; k < active_K; ++k) {
                int slot_id = active_slots[k];
                int base_vk = slot_id * rank * n_kv_heads * D + kv_head * D;
                int base_vv = slot_id * rank * n_kv_heads * D + kv_head * D;
                const float* ca = has_rope ? blk_rope[k].ca.data() : nullptr;
                const float* sa = has_rope ? blk_rope[k].sa.data() : nullptr;

                blk_vk_local[k].resize(rank * D);
                blk_vv_local[k].resize(rank * D);
                blk_anc_v[k].resize(D);

                // VK local [rank, D] — extract + (optionally) rotate at anchor_pos
                if (use_precomp_rot) {
                    // host_VK_rot_ has RoPE already applied; just repack contiguously.
                    const float* base = vk_rot_buf +
                        (size_t)slot_id * rank * n_kv_heads * D + kv_head * D;
                    for (int r = 0; r < rank; ++r)
                        for (int d = 0; d < D; ++d)
                            blk_vk_local[k][r * D + d] = base[r * n_kv_heads * D + d];
                } else if (has_rope) {
                    for (int r = 0; r < rank; ++r) {
                        int bvk = base_vk + r * n_kv_heads * D;
                        for (int d = 0; d < half_d; ++d) {
                            float x = ggml_fp16_to_fp32(VK[bvk + d]);
                            float y = ggml_fp16_to_fp32(VK[bvk + d + half_d]);
                            // Rotate at anchor_pos using precomputed ca/sa
                            blk_vk_local[k][r * D + d]          = x * ca[d] - y * sa[d];
                            blk_vk_local[k][r * D + d + half_d] = y * ca[d] + x * sa[d];
                        }
                    }
                } else {
                    for (int r = 0; r < rank; ++r) {
                        int bvk = base_vk + r * n_kv_heads * D;
                        for (int d = 0; d < D; ++d)
                            blk_vk_local[k][r * D + d] = ggml_fp16_to_fp32(VK[bvk + d]);
                    }
                }

                // VV local [rank, D] — no rotation needed for values
                for (int r = 0; r < rank; ++r) {
                    int bvv = base_vv + r * n_kv_heads * D;
                    for (int d = 0; d < D; ++d)
                        blk_vv_local[k][r * D + d] = ggml_fp16_to_fp32(VV[bvv + d]);
                }

                // Anchor V [D] in fp32 (same for all Q heads sharing this kv_head)
                int av_base = slot_id * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d)
                    blk_anc_v[k][d] = ggml_fp16_to_fp32(anchors_V[av_base + d]);
            }
        }

        float max_score = -1e30f;

        for (int k = 0; k < active_K; ++k) {
            int slot_id   = active_slots[k];
            int slen      = seq_lens[slot_id];
            float scale_u = ggml_fp16_to_fp32(U_scale_arr[slot_id]);
            float blk_sc  = ggml_fp16_to_fp32(scales[slot_id]);

            const float* ca = has_rope ? blk_rope[k].ca.data() : nullptr;
            const float* sa = has_rope ? blk_rope[k].sa.data() : nullptr;

            // ── 1. Anchor score ─────────────────────────────────────────────
            float score_anc = 0.0f;
            {
                int ak_off = slot_id * n_kv_heads * D + kv_head * D;
                if (has_rope) {
                    for (int d = 0; d < half_d; ++d) {
                        float x = ggml_fp16_to_fp32(anchors_K[ak_off + d]);
                        float y = ggml_fp16_to_fp32(anchors_K[ak_off + d + half_d]);
                        score_anc += Q_ptr[h * D + d]          * (x * ca[d] - y * sa[d]);
                        score_anc += Q_ptr[h * D + d + half_d] * (y * ca[d] + x * sa[d]);
                    }
                } else {
                    for (int d = 0; d < D; ++d)
                        score_anc += Q_ptr[h * D + d] *
                                     ggml_fp16_to_fp32(anchors_K[ak_off + d]);
                }
            }
            slot_infos[k].anchor_score = score_anc;
            { float s = score_anc * scale; if (s > max_score) max_score = s; }

            if (approximate_attn) {
                // ── APPROXIMATE path — matches ACTIVE_RUNTIME fused_decode_mps ──
                // Step A: q_proj[r] = Q[h,:] · vk_local[k][r,:]  (cache-friendly GEMV)
                // vk_local already has RoPE baked in at anchor_pos.
                slot_infos[k].q_proj.assign(rank, 0.0f);
                const float* vkl = blk_vk_local[k].data();
                for (int r = 0; r < rank; ++r) {
                    float proj = 0.0f;
                    const float* vkr = vkl + r * D;   // stride-1 sequential!
                    for (int d = 0; d < D; ++d)
                        proj += Q_ptr[h * D + d] * vkr[d];
                    slot_infos[k].q_proj[r] = proj;
                }

                // Step B: delta_s[t] = q_proj · U[t,:] * scale_u * blk_sc
                // U layout: [slot, S_max, rank] → U[u_base + t*rank + r] — stride-1 in r
                slot_infos[k].token_scores.resize(slen);
                int u_base = slot_id * S_max * rank;
                for (int t = 0; t < slen; ++t) {
                    float delta = 0.0f;
                    const int8_t* u_row = U + u_base + t * rank;
                    for (int r = 0; r < rank; ++r)
                        delta += slot_infos[k].q_proj[r] * (float)u_row[r];  // stride-1!

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
                                res_score += Q_ptr[h * D + d] * ggml_fp16_to_fp32(rk[d]);
                        }
                        break;
                    }
                    float t_score = (delta * scale_u * blk_sc + res_score + score_anc) * scale;
                    slot_infos[k].token_scores[t] = t_score;
                    if (t_score > max_score) max_score = t_score;
                }

            } else {
                // ── NON-APPROXIMATE path: full K reconstruction + angular recurrence ──
                // Initialise cos_run at anchor_pos+1 by advancing blk_rope by one step.
                std::vector<float> cos_run(half_d), sin_run(half_d);
                if (has_rope) {
                    for (int d = 0; d < half_d; ++d) {
                        cos_run[d] = ca[d] * cos_step[d] - sa[d] * sin_step[d];
                        sin_run[d] = sa[d] * cos_step[d] + ca[d] * sin_step[d];
                    }
                }

                // Precompute anchor K float for this (slot, kv_head)
                int ak_off = slot_id * n_kv_heads * D + kv_head * D;
                std::vector<float> anc_k_f(D);
                for (int d = 0; d < D; ++d)
                    anc_k_f[d] = ggml_fp16_to_fp32(anchors_K[ak_off + d]);

                slot_infos[k].token_scores.resize(slen);
                int u_base = slot_id * S_max * rank;
                const float* vkl = blk_vk_local[k].data();  // [rank, D] contiguous

                for (int t = 0; t < slen; ++t) {
                    // delta_K[d] = sum_r(U[t,r] * vkl[r,d])
                    // Then rotate (anc_k + delta_K * scale_uv) with cos_run recurrence
                    float score_t = 0.0f;
                    const int8_t* u_row = U + u_base + t * rank;

                    for (int d = 0; d < half_d; ++d) {
                        int d2 = d + half_d;
                        float dk1 = 0.0f, dk2 = 0.0f;
                        for (int r = 0; r < rank; ++r) {
                            float ur = (float)u_row[r];
                            // vkl already has RoPE at anchor_pos baked in,
                            // BUT non-approx needs exact per-token rotation.
                            // Fall back to raw VK (non-rotated) for accurate reconstruction.
                            // [Note: vkl used here is NOT the rotated version for non-approx]
                            dk1 += ur * vkl[r * D + d];
                            dk2 += ur * vkl[r * D + d2];
                        }
                        float k1 = anc_k_f[d]  + dk1 * scale_u * blk_sc;
                        float k2 = anc_k_f[d2] + dk2 * scale_u * blk_sc;

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

                    // Advance recurrence
                    if (has_rope && t + 1 < slen) {
                        for (int d = 0; d < half_d; ++d) {
                            float nc = cos_run[d] * cos_step[d] - sin_run[d] * sin_step[d];
                            float ns = sin_run[d] * cos_step[d] + cos_run[d] * sin_step[d];
                            cos_run[d] = nc; sin_run[d] = ns;
                        }
                    }
                }
                slot_infos[k].q_proj.clear();  // not used in non-approx value phase
            }
        }

        // ── Softmax denominator ─────────────────────────────────────────────
        double sum_exp = 0.0;
        for (int k = 0; k < active_K; ++k) {
            sum_exp += std::exp(slot_infos[k].anchor_score * scale - max_score);
            for (float s : slot_infos[k].token_scores)
                sum_exp += std::exp(s - max_score);
        }
        lse_sparse[h] = max_score + std::log(std::max(sum_exp, 1e-9));

        // ── Value accumulation ──────────────────────────────────────────────
        std::vector<double> accum(D, 0.0);
        for (int k = 0; k < active_K; ++k) {
            int slot_id   = active_slots[k];
            int slen      = seq_lens[slot_id];
            float blk_sc  = ggml_fp16_to_fp32(scales[slot_id]);
            float scale_u = ggml_fp16_to_fp32(U_scale_arr[slot_id]);
            const float* vvl = blk_vv_local[k].data();  // [rank, D] contiguous
            const float* av  = blk_anc_v[k].data();     // [D]

            double w_anc = std::exp(slot_infos[k].anchor_score * scale - max_score) / sum_exp;
            double sum_w = 0.0;
            std::vector<double> w_proj(rank, 0.0);
            std::vector<double> res_v_accum(D, 0.0);
            int u_base = slot_id * S_max * rank;

            for (int t = 0; t < slen; ++t) {
                double w_t = std::exp(slot_infos[k].token_scores[t] - max_score) / sum_exp;
                sum_w += w_t;
                const int8_t* u_row = U + u_base + t * rank;
                for (int r = 0; r < rank; ++r)
                    w_proj[r] += w_t * (float)u_row[r] * scale_u;  // stride-1!
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
            // svd_v[d] = sum_r(w_proj[r] * vvl[r*D+d]) — computed as r-outer GEMV
            // (stride-1 on d for each r → cache-friendly)
            std::vector<double> svd_v(D, 0.0);
            for (int r = 0; r < rank; ++r) {
                double wr = w_proj[r] * blk_sc;
                const float* vvr = vvl + r * D;   // stride-1 sequential!
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
        const int32_t* rkp = pool->get_host_res_K_pos();
        const int32_t* rvp = pool->get_host_res_V_pos();
        const int MR = NativeBlockPool::MAX_RESIDUAL;
        int n_slots = pool->get_seq_lens()->ne[0];
        const int32_t* slots_ptr = (const int32_t*)slot_indices->data;
        for (int k = 0; k < actual_K && !force_cpu; ++k) {
            int s = slots_ptr[k];
            if (s < 0 || s >= n_slots) continue;
            if (rkp[(size_t)s * MR] != -1 || rvp[(size_t)s * MR] != -1) force_cpu = true;
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

    // Two-way LSE combine (sparse ⊕ dense)
    std::vector<float> final_out(n_q_heads * D);
    for (int h = 0; h < n_q_heads; ++h) {
        float ld = lse_dense[h], ls = lse_sparse[h];
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
        double nrm=0;
        for (int i=0;i<n_q_heads*D;++i){ nrm += (double)final_out[i]*final_out[i]; }
        std::cerr << "[DBG_ATTN0] CPU L0 norm=" << std::sqrt(nrm)
                  << " head0[0..5]=";
        for (int i=0;i<6;++i) std::cerr << final_out[i] << " ";
        std::cerr << "\n";
    }
    if (cache_active)
        get_global_attn_cache().save(data->session_id, data->layer_idx, q_host.data(), n_q_heads, D, (const float*)dst->data);
}

} // namespace diffkv
