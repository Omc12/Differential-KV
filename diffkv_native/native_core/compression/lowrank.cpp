#include "native_core/compression/lowrank.hpp"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>

namespace diffkv {

#ifndef __APPLE__
static bool run_cpu_jacobi_svd(
    const float* A_input,
    int S, int F, int R,
    float* U_out,
    float* S_out,
    float* VT_out
) {
    if (S == 1) {
        float norm = 0.0f;
        for (int i = 0; i < F; ++i) {
            norm += A_input[i] * A_input[i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-8f) norm = 1e-8f;

        S_out[0] = norm;
        U_out[0] = 1.0f;
        for (int i = 0; i < F; ++i) {
            VT_out[i] = A_input[i] / norm;
        }
        return true;
    }

    std::vector<float> C(S * S, 0.0f);
    for (int i = 0; i < S; ++i) {
        for (int j = i; j < S; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < F; ++k) {
                sum += A_input[i * F + k] * A_input[j * F + k];
            }
            C[i * S + j] = sum;
            C[j * S + i] = sum;
        }
    }

    std::vector<float> U(S * S, 0.0f);
    for (int i = 0; i < S; ++i) U[i * S + i] = 1.0f;

    const int max_sweeps = 50;
    const float eps = 1e-7f;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float off_diag_norm = 0.0f;
        for (int i = 0; i < S; ++i) {
            for (int j = i + 1; j < S; ++j) {
                off_diag_norm += std::abs(C[i * S + j]);
            }
        }
        if (off_diag_norm < eps) break;

        for (int p = 0; p < S - 1; ++p) {
            for (int q = p + 1; q < S; ++q) {
                float app = C[p * S + p];
                float aqq = C[q * S + q];
                float apq = C[p * S + q];

                if (std::abs(apq) < 1e-15f) continue;

                float theta = (aqq - app) / (2.0f * apq);
                float t;
                if (std::abs(theta) < 1e-9f) {
                    t = 1.0f;
                } else {
                    float sign = (theta > 0.0f) ? 1.0f : -1.0f;
                    t = sign / (std::abs(theta) + std::sqrt(1.0f + theta * theta));
                }
                float c = 1.0f / std::sqrt(1.0f + t * t);
                float s = t * c;
                float tau = s / (1.0f + c);

                C[p * S + p] = app - t * apq;
                C[q * S + q] = aqq + t * apq;
                C[p * S + q] = 0.0f;
                C[q * S + p] = 0.0f;

                for (int r = 0; r < S; ++r) {
                    if (r != p && r != q) {
                        float arp = C[r * S + p];
                        float arq = C[r * S + q];
                        C[r * S + p] = c * arp - s * arq;
                        C[r * S + q] = s * arp + c * arq;
                        C[p * S + r] = C[r * S + p];
                        C[q * S + r] = C[r * S + q];
                    }
                }

                for (int r = 0; r < S; ++r) {
                    float urp = U[r * S + p];
                    float urq = U[r * S + q];
                    U[r * S + p] = c * urp - s * urq;
                    U[r * S + q] = s * urp + c * urq;
                }
            }
        }
    }

    std::vector<int> indices(S);
    std::vector<float> temp_S(S);
    for (int i = 0; i < S; ++i) {
        indices[i] = i;
        float val = C[i * S + i];
        temp_S[i] = (val > 0.0f) ? std::sqrt(val) : 0.0f;
    }

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return temp_S[a] > temp_S[b];
    });

    for (int r = 0; r < R; ++r) {
        S_out[r] = (r < S) ? temp_S[indices[r]] : 0.0f;
    }

    for (int s = 0; s < S; ++s) {
        for (int r = 0; r < R; ++r) {
            U_out[s * R + r] = (r < S) ? U[s * S + indices[r]] : 0.0f;
        }
    }

    for (int r = 0; r < R; ++r) {
        float s_val = S_out[r];
        float inv_s = (s_val > 1e-8f) ? (1.0f / s_val) : 0.0f;
        for (int f = 0; f < F; ++f) {
            float sum = 0.0f;
            if (r < S) {
                for (int s = 0; s < S; ++s) {
                    sum += U[s * S + indices[r]] * A_input[s * F + f];
                }
            }
            VT_out[r * F + f] = sum * inv_s;
        }
    }

    return true;
}
#endif

bool run_svd_driver(
    const float* A_input,
    int S, int F, int R,
    float* U_out,
    float* S_out,
    float* VT_out
) {
#ifdef __APPLE__
    if (S == 1) {
        float norm = 0.0f;
        for (int i = 0; i < F; ++i) {
            norm += A_input[i] * A_input[i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-8f) norm = 1e-8f;

        S_out[0] = norm;
        U_out[0] = 1.0f;
        for (int i = 0; i < F; ++i) {
            VT_out[i] = A_input[i] / norm;
        }
        return true;
    }

    int m = F;
    int n = S;
    int lda = m;
    int ldu = m;
    int min_dim = std::min(S, F);
    int ldvt = min_dim; 

    std::vector<float> a_copy(S * F);
    std::memcpy(a_copy.data(), A_input, S * F * sizeof(float));

    std::vector<float> s_temp(min_dim);
    std::vector<float> u_temp(m * min_dim); 
    std::vector<float> vt_temp(min_dim * n); 

    float work_query = 0.0f;
    int lwork = -1;
    std::vector<int> iwork(8 * min_dim);
    int info = 0;
    char jobz = 'S';

    sgesdd_(&jobz, &m, &n, a_copy.data(), &lda, s_temp.data(), u_temp.data(), &ldu, vt_temp.data(), &ldvt, &work_query, &lwork, iwork.data(), &info);

    bool sgesdd_ok = false;
    if (info == 0) {
        lwork = static_cast<int>(work_query);
        std::vector<float> work(lwork);
        sgesdd_(&jobz, &m, &n, a_copy.data(), &lda, s_temp.data(), u_temp.data(), &ldu, vt_temp.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
        if (info == 0) {
            sgesdd_ok = true;
        }
    }

    if (!sgesdd_ok) {
        std::memcpy(a_copy.data(), A_input, S * F * sizeof(float));
        char jobu = 'S';
        char jobvt = 'S';
        float work_query_svd = 0.0f;
        int lwork_svd = -1;
        sgesvd_(&jobu, &jobvt, &m, &n, a_copy.data(), &lda, s_temp.data(), u_temp.data(), &ldu, vt_temp.data(), &ldvt, &work_query_svd, &lwork_svd, &info);
        if (info == 0) {
            lwork_svd = static_cast<int>(work_query_svd);
            std::vector<float> work_svd(lwork_svd);
            std::memcpy(a_copy.data(), A_input, S * F * sizeof(float));
            sgesvd_(&jobu, &jobvt, &m, &n, a_copy.data(), &lda, s_temp.data(), u_temp.data(), &ldu, vt_temp.data(), &ldvt, work_svd.data(), &lwork_svd, &info);
        }
        if (info != 0) {
            std::cerr << "[SVD] Both sgesdd_ and sgesvd_ failed. info = " << info << std::endl;
            return false;
        }
    }

    std::memcpy(S_out, s_temp.data(), R * sizeof(float));

    for (int s = 0; s < S; ++s) {
        for (int r = 0; r < R; ++r) {
            U_out[s * R + r] = vt_temp[s * min_dim + r];
        }
    }

    std::memcpy(VT_out, u_temp.data(), R * F * sizeof(float));

    return true;
#else
    return run_cpu_jacobi_svd(A_input, S, F, R, U_out, S_out, VT_out);
#endif
}

bool compress_lowrank_block(const LowRankCompressParams& params) {
    int S_total = params.block_size;
    int F = params.feat_dim;
    int R = params.rank;
    int D = params.head_dim;
    int kv_heads = F / D;
    int joint_F = 2 * F;

    std::vector<float> scores(S_total, 0.0f);
    
    for (int s = 0; s < S_total; ++s) {
        float k_norms = 0.0f;
        for (int h = 0; h < kv_heads; ++h) {
            float sum_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                float val = params.raw_k_ptr[s * F + h * D + d];
                sum_sq += val * val;
            }
            k_norms += std::sqrt(sum_sq);
        }
        scores[s] = k_norms / kv_heads;
    }

    std::vector<float> K_f_normed(S_total * F);
    for (int s = 0; s < S_total; ++s) {
        float norm = 0.0f;
        for (int i = 0; i < F; ++i) {
            float val = params.raw_k_ptr[s * F + i];
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm < 1e-8f) norm = 1e-8f;
        for (int i = 0; i < F; ++i) {
            K_f_normed[s * F + i] = params.raw_k_ptr[s * F + i] / norm;
        }
    }

    std::vector<float> col_sums(F, 0.0f);
    for (int t = 0; t < S_total; ++t) {
        for (int i = 0; i < F; ++i) {
            col_sums[i] += K_f_normed[t * F + i];
        }
    }

    for (int s = 0; s < S_total; ++s) {
        float centrality = 0.0f;
        for (int i = 0; i < F; ++i) {
            centrality += K_f_normed[s * F + i] * col_sums[i];
        }
        scores[s] += centrality * 0.5f;
    }

    if (params.token_ids) {
        for (int s = 0; s < S_total; ++s) {
            int32_t tid = params.token_ids[s];
            bool is_stop = (params.stop_token_ids && params.stop_token_ids->count(tid) > 0);
            if (!is_stop) {
                scores[s] += 2.0f;
            }
            if (tid >= 48 && tid <= 57) {
                scores[s] += 3.0f;
            }
        }
    }

    int landmark_idx = 0;
    float max_score = -1e9f;
    for (int s = 0; s < S_total; ++s) {
        if (scores[s] > max_score) {
            max_score = scores[s];
            landmark_idx = s;
        }
    }

    std::vector<float> raw_k_swapped(S_total * F);
    std::vector<float> raw_v_swapped(S_total * F);
    std::memcpy(raw_k_swapped.data(), params.raw_k_ptr, S_total * F * sizeof(float));
    std::memcpy(raw_v_swapped.data(), params.raw_v_ptr, S_total * F * sizeof(float));
    if (landmark_idx > 0) {
        std::swap_ranges(raw_k_swapped.begin(), raw_k_swapped.begin() + F, raw_k_swapped.begin() + landmark_idx * F);
        std::swap_ranges(raw_v_swapped.begin(), raw_v_swapped.begin() + F, raw_v_swapped.begin() + landmark_idx * F);
    }

    if (params.out_anchor_k && params.out_anchor_v) {
        for (int i = 0; i < F; ++i) {
            params.out_anchor_k[i] = ggml_fp32_to_fp16(raw_k_swapped[i]);
            params.out_anchor_v[i] = ggml_fp32_to_fp16(raw_v_swapped[i]);
        }
    }

    int S_deltas = S_total - 1;
    int svd_dim = std::min(S_deltas, R);
    std::vector<float> K_V_joint(S_deltas * joint_F);
    for (int s = 0; s < S_deltas; ++s) {
        for (int i = 0; i < F; ++i) {
            K_V_joint[s * joint_F + i] = raw_k_swapped[(s + 1) * F + i] - raw_k_swapped[i];
            K_V_joint[s * joint_F + F + i] = raw_v_swapped[(s + 1) * F + i] - raw_v_swapped[i];
        }
    }

    // F9: keep the raw (pre-normalisation) delta to compute sparse residuals later.
    std::vector<float> raw_delta = K_V_joint;

    std::vector<float> token_norms(S_deltas);
    for (int s = 0; s < S_deltas; ++s) {
        float sum_sq = 0.0f;
        for (int j = 0; j < joint_F; ++j) {
            float val = K_V_joint[s * joint_F + j];
            sum_sq += val * val;
        }
        float norm = std::sqrt(sum_sq);
        if (norm < 1e-5f) norm = 1e-5f;
        token_norms[s] = norm;
        for (int j = 0; j < joint_F; ++j) {
            K_V_joint[s * joint_F + j] /= norm;
        }
    }

    float scale = 0.0f;
    for (float val : K_V_joint) {
        float abs_val = std::abs(val);
        if (abs_val > scale) {
            scale = abs_val;
        }
    }
    if (scale < 1e-9f) {
        scale = 1e-9f;
    }

    for (float& val : K_V_joint) {
        val /= scale;
    }

    std::vector<float> U_joint(S_deltas * svd_dim);
    std::vector<float> S_joint(svd_dim);
    std::vector<float> VT_joint(svd_dim * joint_F);

    bool ok = run_svd_driver(K_V_joint.data(), S_deltas, joint_F, svd_dim, U_joint.data(), S_joint.data(), VT_joint.data());
    if (!ok) {
        return false;
    }

    // Energy-preserving dynamic rank selection
    float total_energy = 0.0f;
    for (int r = 0; r < svd_dim; ++r) {
        total_energy += S_joint[r] * S_joint[r];
    }

    int k_dynamic = svd_dim;
    if (total_energy > 1e-9f) {
        float cum_energy = 0.0f;
        float threshold = 0.999f * total_energy;
        for (int r = 0; r < svd_dim; ++r) {
            cum_energy += S_joint[r] * S_joint[r];
            if (cum_energy >= threshold) {
                k_dynamic = std::max(4, std::min(r + 1, R));
                break;
            }
        }
    }

    std::vector<float> U_scaled(S_deltas * R, 0.0f);
    float max_abs = 0.0f;
    for (int s = 0; s < S_deltas; ++s) {
        for (int r = 0; r < svd_dim; ++r) {
            float val = 0.0f;
            if (r < k_dynamic) {
                val = U_joint[s * svd_dim + r] * S_joint[r] * token_norms[s];
            }
            U_scaled[s * R + r] = val;
            if (std::abs(val) > max_abs) {
                max_abs = std::abs(val);
            }
        }
    }

    float scale_u = std::max(max_abs / 127.0f, 1e-5f);
    int S_max = params.pool_block_size; 
    int pool_rank = params.pool_rank > 0 ? params.pool_rank : R;
    std::memset(params.out_u_ptr, 0, S_max * pool_rank * sizeof(int8_t));
    for (int s = 0; s < S_deltas; ++s) {
        for (int r = 0; r < R; ++r) {
            float val = U_scaled[s * R + r] / scale_u;
            int8_t val_i8 = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(val))));
            params.out_u_ptr[s * pool_rank + r] = val_i8;
        }
    }

    *params.out_u_scale = ggml_fp32_to_fp16(scale_u);

    std::memset(params.out_vk_ptr, 0, pool_rank * F * sizeof(ggml_fp16_t));
    std::memset(params.out_vv_ptr, 0, pool_rank * F * sizeof(ggml_fp16_t));
    for (int r = 0; r < svd_dim; ++r) {
        if (r < k_dynamic) {
            for (int f = 0; f < F; ++f) {
                params.out_vk_ptr[r * F + f] = ggml_fp32_to_fp16(VT_joint[r * joint_F + f]);
                params.out_vv_ptr[r * F + f] = ggml_fp32_to_fp16(VT_joint[r * joint_F + F + f]);
            }
        }
    }

    *params.out_scale = ggml_fp32_to_fp16(scale);
    if (params.out_seq_len) {
        *params.out_seq_len = S_deltas;
    }
    if (params.out_anchor_position) {
        *params.out_anchor_position = params.anchor_idx + landmark_idx;
    }

    // ── DBG: reconstruction-error decomposition (DIFFKV_DBG_COMPRESS_ERR=1) ─────
    // Compares the block reconstruction vs the original delta three ways:
    //   floor   = fp32 U · fp32 VT             (irreducible rank-k truncation loss)
    //   fp16_U  = fp16 U · fp16 VT             (what MLX stores)
    //   int8_U  = int8 U·scale_u · fp16 VT     (what C++ stores)
    // The int8-vs-fp16 GAP tells us how much fixing U→fp16 would actually buy.
    if (std::getenv("DIFFKV_DBG_COMPRESS_ERR")) {
        static double e_floor=0, e_fp16=0, e_int8=0, nrm=0; static long nb=0, ntok=0;
        for (int s = 0; s < S_deltas; ++s) {
            for (int f = 0; f < joint_F; ++f) {
                double rf=0, r16=0, r8=0;
                for (int r = 0; r < k_dynamic && r < svd_dim; ++r) {
                    double us = U_scaled[s * R + r];
                    double vt = VT_joint[r * joint_F + f];
                    double vt16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16((float)vt));
                    double us16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16((float)us));
                    int iv = (int)std::round(us / scale_u);
                    iv = std::max(-127, std::min(127, iv));
                    double us8 = (double)iv * scale_u;
                    rf  += us  * vt;
                    r16 += us16 * vt16;
                    r8  += us8  * vt16;
                }
                rf *= scale; r16 *= scale; r8 *= scale;
                double raw = raw_delta[s * joint_F + f];
                e_floor += (raw-rf)*(raw-rf);
                e_fp16  += (raw-r16)*(raw-r16);
                e_int8  += (raw-r8)*(raw-r8);
                nrm     += raw*raw;
            }
        }
        nb++; ntok += S_deltas;
        if (nb % 20 == 0 && nrm > 0) {
            double rfl=100*std::sqrt(e_floor/nrm), rf16=100*std::sqrt(e_fp16/nrm), ri8=100*std::sqrt(e_int8/nrm);
            std::cerr << "[COMPRESS_ERR] blocks=" << nb << " toks=" << ntok
                      << " rel_recon_err: floor(rank" << R << ")=" << rfl << "%  fp16_U=" << rf16
                      << "%  int8_U=" << ri8 << "%  | int8_penalty_over_fp16=" << (ri8-rf16) << "%\n";
        }
    }

    // ── F9: Post-SVD sparse residual storage ──────────────────────────────────
    // Mirrors ACTIVE_RUNTIME lowrank.py: for the top max_residual_frac (15%) tokens
    // by relative reconstruction error (> error_threshold 0.08), store the EXACT
    // residual delta = raw_delta - low_rank_recon. The decode kernel adds these back
    // so high-error tokens (e.g. digits) are recovered exactly even when compressed.
    if (params.out_res_K_pos && params.out_res_V_pos &&
        params.out_res_K_val && params.out_res_V_val && S_deltas > 0) {
        const int MR = params.max_residual;
        const float ERR_THRESH = 0.08f;
        for (int i = 0; i < MR; ++i) { params.out_res_K_pos[i] = -1; params.out_res_V_pos[i] = -1; }
        std::memset(params.out_res_K_val, 0, (size_t)MR * F * sizeof(ggml_fp16_t));
        std::memset(params.out_res_V_val, 0, (size_t)MR * F * sizeof(ggml_fp16_t));

        std::vector<float> resid((size_t)S_deltas * joint_F);
        std::vector<float> rel_K(S_deltas), rel_V(S_deltas), aerr_K(S_deltas), aerr_V(S_deltas);
        for (int s = 0; s < S_deltas; ++s) {
            float eK = 0.0f, eV = 0.0f, nK = 0.0f, nV = 0.0f;
            for (int f = 0; f < joint_F; ++f) {
                float recon = 0.0f;
                for (int r = 0; r < svd_dim; ++r) recon += U_scaled[s * R + r] * VT_joint[r * joint_F + f];
                recon *= scale;
                float res = raw_delta[s * joint_F + f] - recon;
                resid[(size_t)s * joint_F + f] = res;
                float raw = raw_delta[s * joint_F + f];
                if (f < F) { eK += res * res; nK += raw * raw; }
                else       { eV += res * res; nV += raw * raw; }
            }
            aerr_K[s] = std::sqrt(eK); aerr_V[s] = std::sqrt(eV);
            rel_K[s] = aerr_K[s] / std::max(std::sqrt(nK), 1e-8f);
            rel_V[s] = aerr_V[s] / std::max(std::sqrt(nV), 1e-8f);
        }
        int n_max = std::min((int)(S_deltas * 0.15f), MR);
        auto select = [&](const std::vector<float>& rel, const std::vector<float>& aerr,
                          int32_t* pos_out, ggml_fp16_t* val_out, int half_off) {
            std::vector<int> idx(S_deltas);
            for (int i = 0; i < S_deltas; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return rel[a] > rel[b]; });
            int written = 0;
            for (int ii = 0; ii < S_deltas && written < n_max; ++ii) {
                int s = idx[ii];
                if (rel[s] <= ERR_THRESH) break;      // sorted desc → rest are smaller
                if (aerr[s] <= 1e-4f) continue;
                pos_out[written] = s;                  // block-local delta index = decode token index
                for (int f = 0; f < F; ++f)
                    val_out[(size_t)written * F + f] = ggml_fp32_to_fp16(resid[(size_t)s * joint_F + half_off + f]);
                written++;
            }
        };
        if (n_max > 0) {
            select(rel_K, aerr_K, params.out_res_K_pos, params.out_res_K_val, 0);
            select(rel_V, aerr_V, params.out_res_V_pos, params.out_res_V_val, F);
        }
    }

    return true;
}

} // namespace diffkv
