#include "native_core/compression/lowrank.hpp"
#include "native_core/srl/chunk_descriptor.hpp"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <random>

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

static bool run_lapack_svd(
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

    int r_to_copy = std::min(R, min_dim);
    std::memset(S_out, 0, R * sizeof(float));
    std::memcpy(S_out, s_temp.data(), r_to_copy * sizeof(float));

    std::memset(U_out, 0, S * R * sizeof(float));
    for (int s = 0; s < S; ++s) {
        for (int r = 0; r < r_to_copy; ++r) {
            U_out[s * R + r] = vt_temp[s * min_dim + r];
        }
    }

    std::memset(VT_out, 0, R * F * sizeof(float));
    for (int r = 0; r < r_to_copy; ++r) {
        std::memcpy(VT_out + r * F, u_temp.data() + r * m, F * sizeof(float));
    }

    return true;
#else
    return run_cpu_jacobi_svd(A_input, S, F, R, U_out, S_out, VT_out);
#endif
}

static bool run_randomized_svd(
    const float* A_input,
    int S, int F, int R,
    float* U_out,
    float* S_out,
    float* VT_out
) {
    int n_oversamples = 5;
    int r_proj = std::min({R + n_oversamples, S, F});
    if (r_proj < 1) {
        return false;
    }

    // Generate random matrix Omega of shape F x r_proj
    std::vector<float> Omega(F * r_proj);
    std::mt19937 gen(42); // fixed seed
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < F * r_proj; ++i) {
        Omega[i] = dist(gen);
    }

    // Compute Y = A_input * Omega (shape S x r_proj)
    std::vector<float> Y(S * r_proj, 0.0f);
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                S, r_proj, F,
                1.0f, A_input, F,
                Omega.data(), r_proj,
                0.0f, Y.data(), r_proj);
#else
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < r_proj; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < F; ++k) {
                sum += A_input[i * F + k] * Omega[k * r_proj + j];
            }
            Y[i * r_proj + j] = sum;
        }
    }
#endif

    // One power iteration:
    // Z = A_input^T * Y (shape F x r_proj)
    std::vector<float> Z(F * r_proj, 0.0f);
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                F, r_proj, S,
                1.0f, A_input, F,
                Y.data(), r_proj,
                0.0f, Z.data(), r_proj);
#else
    for (int i = 0; i < F; ++i) {
        for (int j = 0; j < r_proj; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < S; ++k) {
                sum += A_input[k * F + i] * Y[k * r_proj + j];
            }
            Z[i * r_proj + j] = sum;
        }
    }
#endif

    // Y = A_input * Z (shape S x r_proj)
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                S, r_proj, F,
                1.0f, A_input, F,
                Z.data(), r_proj,
                0.0f, Y.data(), r_proj);
#else
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < r_proj; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < F; ++k) {
                sum += A_input[i * F + k] * Z[k * r_proj + j];
            }
            Y[i * r_proj + j] = sum;
        }
    }
#endif

    // QR decomposition of Y (shape S x r_proj) using Modified Gram-Schmidt
    std::vector<float> Q(S * r_proj);
    std::memcpy(Q.data(), Y.data(), S * r_proj * sizeof(float));
    for (int k = 0; k < r_proj; ++k) {
        double norm = 0.0;
        for (int i = 0; i < S; ++i) {
            float val = Q[i * r_proj + k];
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm < 1e-4f) {
            for (int i = 0; i < S; ++i) {
                Q[i * r_proj + k] = 0.0f;
            }
        } else {
            float inv_norm = 1.0f / norm;
            for (int i = 0; i < S; ++i) {
                Q[i * r_proj + k] *= inv_norm;
            }
            for (int j = k + 1; j < r_proj; ++j) {
                double dot = 0.0;
                for (int i = 0; i < S; ++i) {
                    dot += Q[i * r_proj + k] * Q[i * r_proj + j];
                }
                for (int i = 0; i < S; ++i) {
                    Q[i * r_proj + j] -= dot * Q[i * r_proj + k];
                }
            }
        }
    }

    // Compute B = Q^T * A_input (shape r_proj x F)
    std::vector<float> B(r_proj * F, 0.0f);
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                r_proj, F, S,
                1.0f, Q.data(), r_proj,
                A_input, F,
                0.0f, B.data(), F);
#else
    for (int i = 0; i < r_proj; ++i) {
        for (int j = 0; j < F; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < S; ++k) {
                sum += Q[k * r_proj + i] * A_input[k * F + j];
            }
            B[i * F + j] = sum;
        }
    }
#endif

    // Compute SVD of the tiny matrix B (shape r_proj x F)
    std::vector<float> U_b(r_proj * R, 0.0f);
    std::vector<float> S_joint(R, 0.0f);
    std::vector<float> VT_joint(R * F, 0.0f);
    if (!run_lapack_svd(B.data(), r_proj, F, R, U_b.data(), S_joint.data(), VT_joint.data())) {
        return false;
    }

    // Compute U_out = Q * U_b (shape S x R)
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                S, R, r_proj,
                1.0f, Q.data(), r_proj,
                U_b.data(), R,
                0.0f, U_out, R);
#else
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < R; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < r_proj; ++k) {
                sum += Q[i * r_proj + k] * U_b[k * R + j];
            }
            U_out[i * R + j] = sum;
        }
    }
#endif

    // Copy singular values and right singular vectors
    std::memcpy(S_out, S_joint.data(), R * sizeof(float));
    std::memcpy(VT_out, VT_joint.data(), R * F * sizeof(float));

    return true;
}

bool run_svd_driver(
    const float* A_input,
    int S, int F, int R,
    float* U_out,
    float* S_out,
    float* VT_out,
    bool force_lapack
) {
    static const bool use_rand_svd = []() {
        if (const char* env = std::getenv("DIFFKV_RAND_SVD")) {
            std::string s(env);
            return s != "0" && s != "false" && s != "off";
        }
        return true;
    }();

    if (use_rand_svd && !force_lapack) {
        return run_randomized_svd(A_input, S, F, R, U_out, S_out, VT_out);
    } else {
        return run_lapack_svd(A_input, S, F, R, U_out, S_out, VT_out);
    }
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

    bool ok = run_svd_driver(K_V_joint.data(), S_deltas, joint_F, svd_dim, U_joint.data(), S_joint.data(), VT_joint.data(), params.force_lapack);
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

    // ── Per-token-row int8 quantization (needle-recall correctness fix) ──────────
    // A single block-wide scale (max_abs/127) is set by the dominant token row, so a
    // low-norm token (e.g. a planted needle whose energy is small vs the block max)
    // has its U components rounded to ~0 in int8 and can no longer be reconstructed
    // ("omega" instead of "OMEGA-7741-DELTA"). Give each token row its own scale so
    // every token keeps full int8 resolution. When out_u_row_scale is present the
    // sparse-attention reader dequantizes U[t,r] with scale_row[t] (not the block
    // scalar). Falls back to the legacy single scale when the per-row buffer is null.
    if (params.out_u_row_scale) {
        for (int t = 0; t < S_max; ++t) params.out_u_row_scale[t] = ggml_fp32_to_fp16(0.0f);
        for (int s = 0; s < S_deltas; ++s) {
            float row_max = 0.0f;
            for (int r = 0; r < R; ++r) {
                float a = std::abs(U_scaled[s * R + r]);
                if (a > row_max) row_max = a;
            }
            float scale_row = std::max(row_max / 127.0f, 1e-5f);
            for (int r = 0; r < R; ++r) {
                float val = U_scaled[s * R + r] / scale_row;
                int8_t val_i8 = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(val))));
                params.out_u_ptr[s * pool_rank + r] = val_i8;
            }
            params.out_u_row_scale[s] = ggml_fp32_to_fp16(scale_row);
        }
    } else {
        for (int s = 0; s < S_deltas; ++s) {
            for (int r = 0; r < R; ++r) {
                float val = U_scaled[s * R + r] / scale_u;
                int8_t val_i8 = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(val))));
                params.out_u_ptr[s * pool_rank + r] = val_i8;
            }
        }
    }

    // Single block scale kept for descriptor computation + legacy fused paths.
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
    // True global sequence position of each delta token, for correct decode RoPE. Deltas are
    // swapped positions 1..S_total-1 (delta index t ↔ swapped pos t+1); the landmark swap put
    // original token `landmark_idx` at swapped 0 and original 0 at swapped landmark_idx. So the
    // within-block original index of delta t is (t+1==landmark_idx ? 0 : t+1); global pos adds
    // the block start (anchor_idx). The decode rotates each token by THIS, not the block anchor.
    if (params.out_token_positions) {
        for (int t = 0; t < S_deltas; ++t) {
            int sp = t + 1;                                  // swapped position
            int within = (sp == landmark_idx) ? 0 : sp;      // original within-block index
            params.out_token_positions[t] = params.anchor_idx + within;
        }
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
            // Per-row int8 scale (matches what's actually stored after the fix).
            double row_max = 0.0;
            for (int r = 0; r < R; ++r) { double a = std::abs(U_scaled[s*R+r]); if (a>row_max) row_max=a; }
            double scale_row = std::max(row_max/127.0, 1e-5);
            for (int f = 0; f < joint_F; ++f) {
                double rf=0, r16=0, r8=0;
                for (int r = 0; r < k_dynamic && r < svd_dim; ++r) {
                    double us = U_scaled[s * R + r];
                    double vt = VT_joint[r * joint_F + f];
                    double vt16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16((float)vt));
                    double us16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16((float)us));
                    int iv = (int)std::round(us / scale_row);
                    iv = std::max(-127, std::min(127, iv));
                    double us8 = (double)iv * scale_row;
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
        if ((nb <= 3 || nb % 20 == 0) && nrm > 0) {
            double rfl=100*std::sqrt(e_floor/nrm), rf16=100*std::sqrt(e_fp16/nrm), ri8=100*std::sqrt(e_int8/nrm);
            int tp0 = params.out_token_positions ? params.out_token_positions[0] : -1;
            int tpl = (params.out_token_positions && S_deltas>0) ? params.out_token_positions[S_deltas-1] : -1;
            std::cerr << "[COMPRESS_ERR] blocks=" << nb << " toks=" << ntok
                      << " anchor_idx=" << params.anchor_idx << " tok_pos[0]=" << tp0 << " tok_pos[last]=" << tpl
                      << " rel_recon_err: floor(rank" << R << ")=" << rfl << "%  fp16_U=" << rf16
                      << "%  int8_U=" << ri8 << "%  | int8_penalty_over_fp16=" << (ri8-rf16) << "%" << std::endl;
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
        const float ERR_THRESH = 0.001f;
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

        std::vector<float> joint_err(S_deltas);
        for (int s = 0; s < S_deltas; ++s) {
            joint_err[s] = std::sqrt(aerr_K[s] * aerr_K[s] + aerr_V[s] * aerr_V[s]);
        }

        int n_max = std::min(S_deltas, MR);
        std::vector<int> idx(S_deltas);
        for (int i = 0; i < S_deltas; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return joint_err[a] > joint_err[b]; });

        int written = 0;
        for (int ii = 0; ii < S_deltas && written < n_max; ++ii) {
            int s = idx[ii];
            if (joint_err[s] <= 1e-4f) continue;
            params.out_res_K_pos[written] = s;
            params.out_res_V_pos[written] = s;
            for (int f = 0; f < F; ++f) {
                params.out_res_K_val[(size_t)written * F + f] = ggml_fp32_to_fp16(resid[(size_t)s * joint_F + f]);
                params.out_res_V_val[(size_t)written * F + f] = ggml_fp32_to_fp16(resid[(size_t)s * joint_F + F + f]);
            }
            written++;
        }

        if (written > 0) {
            // Print residuals debug info
            if (params.token_ids) {
                std::cerr << "[DEBUG_RESIDUALS] block_id=" << params.block_id 
                          << " S_deltas=" << S_deltas << " n_max=" << n_max << " written=" << written << " top5_pos_err: ";
                for (int i = 0; i < std::min(written, 5); ++i) {
                    int s = params.out_res_K_pos[i];
                    std::cerr << s << "(" << joint_err[s] << ", tok=" << params.token_ids[s + 1] << ") ";
                }
                std::cerr << std::endl;
            }
        }
    }

    if (params.out_desc && params.W_proj && params.desc_dim > 0) {
        std::vector<ggml_fp16_t> desc_f16(params.desc_dim);
        compute_descriptor(
            (const uint16_t*)params.out_anchor_k,
            params.out_u_ptr,
            ggml_fp16_to_fp32(*params.out_u_scale),
            (const uint16_t*)params.out_vk_ptr,
            params.W_proj,
            kv_heads,
            D,
            S_deltas,
            R,
            (uint16_t*)desc_f16.data()
        );
        for (int r = 0; r < params.desc_dim; ++r) {
            params.out_desc[r] = ggml_fp16_to_fp32(desc_f16[r]);
        }
    }

    return true;
}

} // namespace diffkv
