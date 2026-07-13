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
#include <unordered_map>
#include <mutex>
#include <iomanip>

namespace diffkv {

static std::mutex g_hist_mutex;
static double g_cum_energy[17] = {0.0};
static double g_total_energy = 0.0;
static long long g_svd_count = 0;

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

    // ── V-side rebalancing for the joint K|V SVD ────────────────────────────────
    // K rows carry enormous norms (Qwen "massive activations": |K| ~ 1100 at layer 0)
    // while |V| ~ 5. In the JOINT SVD the V half contributes <1% of the energy, so the
    // rank-k basis barely represents V at all — measured V reconstruction error was
    // 24-73% per token (vs ~1% for K) even though the joint floor looked fine. Attention
    // ROUTES correctly (K good) but READS garbage (V bad) → exact-token recall (digits,
    // codes) fails for any token that didn't win a residual slot. Fix: scale the V half
    // up to K's RMS before the SVD so the basis serves both sides, and bake the inverse
    // into the stored VV basis (decode math unchanged). DIFFKV_V_SCALE=0 disables.
    float v_gain = 1.0f;
    {
        static const bool v_scale_on = []() {
            const char* e = std::getenv("DIFFKV_V_SCALE");
            return !(e && std::string(e) == "0");
        }();
        if (v_scale_on) {
            double eK = 0.0, eV = 0.0;
            for (int s = 0; s < S_deltas; ++s) {
                for (int f = 0; f < F; ++f) {
                    float kd = K_V_joint[(size_t)s * joint_F + f];
                    float vd = K_V_joint[(size_t)s * joint_F + F + f];
                    eK += (double)kd * kd;
                    eV += (double)vd * vd;
                }
            }
            if (eV > 1e-12 && eK > 1e-12) {
                v_gain = (float)std::sqrt(eK / eV);
                v_gain = std::min(std::max(v_gain, 1.0f), 10000.0f);
                for (int s = 0; s < S_deltas; ++s)
                    for (int f = 0; f < F; ++f)
                        K_V_joint[(size_t)s * joint_F + F + f] *= v_gain;
            }
        }
    }

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

    if (total_energy > 1e-9f) {
        std::lock_guard<std::mutex> lock(g_hist_mutex);
        float cum = 0.0f;
        for (int r = 0; r < 16; ++r) {
            if (r < svd_dim) {
                cum += S_joint[r] * S_joint[r];
            }
            g_cum_energy[r] += cum;
        }
        g_total_energy += total_energy;
        g_svd_count++;
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
    const float inv_v_gain = 1.0f / v_gain;
    for (int r = 0; r < svd_dim; ++r) {
        if (r < k_dynamic) {
            for (int f = 0; f < F; ++f) {
                params.out_vk_ptr[r * F + f] = ggml_fp32_to_fp16(VT_joint[r * joint_F + f]);
                // The SVD ran on v_gain-scaled V; baking 1/v_gain here makes decode's
                // U·VV reconstruction produce raw-space V with no kernel changes.
                params.out_vv_ptr[r * F + f] = ggml_fp32_to_fp16(VT_joint[r * joint_F + F + f] * inv_v_gain);
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
        // Reconstruct EXACTLY as decode dequantizes — int8 U row (out_u_ptr) ×
        // fp16 per-row scale × fp16 VK/VV (inv_v_gain already baked into VV) ×
        // fp16 block scale — so the stored residual correction makes the row
        // bit-exact (up to the fp16 rounding of the correction itself) at read
        // time. The previous float-U/float-VT reconstruction left the int8+fp16
        // quantization error INSIDE the corrected rows: measured K_rel_err ~3e-4
        // on needle rows at 4k/8k — the leading D7 digit-corruption suspect.
        // All pool buffers referenced here are written above (lines ~672-725).
        const float bsc_deq = ggml_fp16_to_fp32(*params.out_scale);
        const float ku_blk_deq = ggml_fp16_to_fp32(*params.out_u_scale);
        for (int s = 0; s < S_deltas; ++s) {
            float eK = 0.0f, eV = 0.0f, nK = 0.0f, nV = 0.0f;
            const float ku_deq = params.out_u_row_scale
                ? ggml_fp16_to_fp32(params.out_u_row_scale[s]) : ku_blk_deq;
            for (int f = 0; f < joint_F; ++f) {
                float recon = 0.0f;
                for (int r = 0; r < R; ++r) {
                    float u_deq = (float)params.out_u_ptr[(size_t)s * pool_rank + r];
                    float v_deq = (f < F)
                        ? ggml_fp16_to_fp32(params.out_vk_ptr[r * F + f])
                        : ggml_fp16_to_fp32(params.out_vv_ptr[r * F + (f - F)]);
                    recon += u_deq * v_deq;
                }
                recon *= ku_deq * bsc_deq;
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

        // OPT-A: Adaptive residual budget — C++ port.
        // Classifies each block's complexity using the median of rel_K and rel_V.
        float median_err_K = 0.0f;
        float median_err_V = 0.0f;
        if (S_deltas > 0) {
            std::vector<float> temp_K = rel_K;
            std::vector<float> temp_V = rel_V;
            size_t mid = S_deltas / 2;
            std::nth_element(temp_K.begin(), temp_K.begin() + mid, temp_K.end());
            median_err_K = temp_K[mid];
            std::nth_element(temp_V.begin(), temp_V.begin() + mid, temp_V.end());
            median_err_V = temp_V[mid];
        }
        float max_median_err = std::max(median_err_K, median_err_V);
        int adaptive_MR = MR;
        // DIFFKV_RESIDUAL_UNIFORM=1 (set by the native LEGO prefill, studs far
        // mode): skip the adaptive cap so every full prefill block carries the
        // full MAX_RESIDUAL set. Stud row counts must be uniform across layers
        // and blocks (the shared prefill mask cannot express per-layer counts;
        // the adaptive cap keys off per-LAYER error medians). Matches MLX, whose
        // prefill blocks always carry full residual sets. Pool residual tensors
        // are pre-allocated at MAX_RESIDUAL per slot, so this costs no memory.
        const char* res_uni_env = std::getenv("DIFFKV_RESIDUAL_UNIFORM");
        const bool res_uniform = (res_uni_env && res_uni_env[0] == '1');
        if (!res_uniform) {
            if (max_median_err < 0.05f) {
                // Easy block (prose filler): cap at 8 residuals
                adaptive_MR = std::min(8, adaptive_MR);
            } else if (max_median_err < 0.15f) {
                // Medium block: cap at 16 residuals
                adaptive_MR = std::min(16, adaptive_MR);
            }
        }

        std::vector<float> joint_err(S_deltas);
        for (int s = 0; s < S_deltas; ++s) {
            // Rank residual candidates in the BALANCED space: an unscaled absolute error
            // is dominated by K's huge norms, so a token whose VALUE is 50% wrong but
            // whose key happens to reconstruct well ranks below ordinary prose and never
            // wins a residual slot (measured: needle digits lost while 64 slots went to
            // rows with slightly higher K-absolute error). Weight the V error by v_gain
            // (the K/V RMS ratio) so both sides compete on equal footing.
            float ev_bal = aerr_V[s] * v_gain;
            joint_err[s] = std::sqrt(aerr_K[s] * aerr_K[s] + ev_bal * ev_bal);
        }

        // ── Stride-stratified residual coverage (APPENDED, not score-boosted) ───
        // Coverage rows give the block a positional scaffold through filler —
        // measured: they fix compressed enumeration-order transposition (binding
        // list-all 3/6→6/6 at frac 0.5). CRITICAL ordering constraint learned the
        // hard way (multi-needle 3/3→0/3 when coverage was a +1e12 score bonus):
        // the residual arrays DOUBLE as the block's ROUTING signature — decode
        // relevance reads the FIRST route_residuals rows — so coverage rows must
        // sit AFTER the ranked (distinctive) rows, never ahead of them. The cols
        // are collected here and appended in the selection phase below; the quota
        // is a fraction of the FINAL block budget (n_max, post-floor), matching
        // the MLX wrapper.
        // Default 0.25 (native): the measured Pareto point. Sweep 2026-07-12,
        // q8, binding list-all / multi-needle 16k: frac 0 = 3/6 + 3/3 (order
        // transposed), 0.125 = 4/6 + 3/3, 0.25 = 5/6 + 3/3, 0.5 = 6/6 + 0/3
        // (needle SUFFIX displacement — "OMEGA-7741-BETA"). 0.25 also restores
        // the owner-capture synthesis/margin costs to baseline (26.7/26.7,
        // margins 12.48/14.26) with NIAH 6/6. MLX keeps default 0 — its
        // enumeration was already 6/6 without the scaffold; flip only with
        // fresh measurements there.
        static const float cov_frac = []() {
            const char* e = std::getenv("DIFFKV_RESIDUAL_COVERAGE_FRAC");
            if (e) { try { return std::stof(e); } catch (...) {} }
            return 0.25f;
        }();
        std::vector<int> cov_cols;
        if (cov_frac > 0.0f && adaptive_MR > 0) {
            int n_cov = std::min(MR, std::max(1, (int)std::round(cov_frac * MR)));
            for (int i = 0; i < n_cov; ++i) {
                float val = 0.0f;
                if (n_cov > 1) {
                    val = (float)i * (S_deltas - 1) / (n_cov - 1);
                }
                int col = (int)std::round(val);
                if (col >= 0 && col < S_deltas &&
                    std::find(cov_cols.begin(), cov_cols.end(), col) == cov_cols.end()) {
                    cov_cols.push_back(col);
                }
            }
        }

        // ── Content-aware residual capture ──────────────────────────────────────
        // Reconstruction error CANNOT identify recall-critical tokens: the rank-16 V
        // reconstruction is 25-70% wrong for MOST tokens, so a buried passcode digit
        // ranks no higher than prose and never wins a slot — while emitting that digit
        // exactly requires ITS value to be exact. High-information token CLASSES
        // (digits, letter-like code fragments) get a multiplicative rank boost so the
        // exact-recall-critical rows claim residual slots first. Uses the tokenizer ids
        // actually produced by Qwen ('0'..'9' = 15..24; the older landmark boost checked
        // ids 48-57, which are NOT digits in this vocab). DIFFKV_RESIDUAL_TOKEN_BOOST=0
        // disables; value tunable (default 8×).
        int lr_boosted_rows = 0;   // rows carrying a boost after the window pass (feeds the budget floor)
        int lr_table_rows = 0;     // rows on table-like lines (drives the coverage skip below)
        if (params.token_ids) {
            static const float tok_boost = []() {
                const char* e = std::getenv("DIFFKV_RESIDUAL_TOKEN_BOOST");
                if (e) { try { return std::stof(e); } catch (...) {} }
                return 8.0f;
            }();
            if (tok_boost > 1.0f) {
                std::unordered_map<int32_t, int> token_counts;
                if (params.session_token_ids && params.session_len > 0) {
                    for (int i = 0; i < params.session_len; ++i) {
                        token_counts[params.session_token_ids[i]]++;
                    }
                }

                std::vector<std::string> tok_strs(S_deltas);
                for (int s = 0; s < S_deltas; ++s) {
                    int32_t tid = params.token_ids[s + 1];
                    if (params.token_to_piece_fn) {
                        tok_strs[s] = params.token_to_piece_fn(tid);
                    } else {
                        tok_strs[s] = "";
                    }
                }

                auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
                auto is_upper_char = [](char c) { return c >= 'A' && c <= 'Z'; };
                auto is_lower_char = [](char c) { return c >= 'a' && c <= 'z'; };
                auto is_alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };

                auto is_title_case = [&](const std::string& str) {
                    if (str.empty()) return false;
                    if (!is_upper_char(str[0])) return false;
                    for (size_t i = 1; i < str.length(); ++i) {
                        if (is_alpha(str[i]) && !is_lower_char(str[i])) {
                            return false;
                        }
                    }
                    return true;
                };

                auto is_alpha_str = [&](const std::string& str) {
                    if (str.empty()) return false;
                    for (char c : str) {
                        if (!is_alpha(c)) return false;
                    }
                    return true;
                };

                std::vector<bool> is_core(S_deltas, false);
                std::vector<bool> is_prose(S_deltas, false);

                for (int s = 0; s < S_deltas; ++s) {
                    std::string raw_str = tok_strs[s];
                    std::string clean_str = "";
                    for (char c : raw_str) {
                        if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                            clean_str += c;
                        }
                    }

                    if (clean_str.empty()) {
                        is_prose[s] = true;
                        continue;
                    }

                    bool has_digit = false;
                    for (char c : clean_str) {
                        if (is_digit(c)) {
                            has_digit = true;
                            break;
                        }
                    }

                    bool is_upper_word = true;
                    bool has_alpha = false;
                    for (char c : clean_str) {
                        if (is_alpha(c)) {
                            has_alpha = true;
                            if (!is_upper_char(c)) {
                                is_upper_word = false;
                            }
                        }
                    }
                    is_upper_word = is_upper_word && has_alpha && clean_str.length() >= 2;

                    is_core[s] = has_digit || is_upper_word || (clean_str == "-") || (clean_str == "_");

                    bool prose = false;
                    if (clean_str == "." || clean_str == "," || clean_str == ";" || clean_str == "?" ||
                        clean_str == "!" || clean_str == ":" || clean_str == "\"" || clean_str == "'" ||
                        clean_str == "(" || clean_str == ")" || clean_str == "[" || clean_str == "]" ||
                        clean_str == "{" || clean_str == "}") {
                        prose = true;
                    } else if (is_alpha_str(clean_str)) {
                        bool is_lower_word = true;
                        for (char c : clean_str) {
                            if (is_alpha(c) && !is_lower_char(c)) {
                                is_lower_word = false;
                                break;
                            }
                        }
                        if (is_lower_word || (is_title_case(clean_str) && clean_str.length() > 1)) {
                            prose = true;
                        }
                    }
                    is_prose[s] = prose;
                }

                bool has_strs = false;
                for (int s = 0; s < S_deltas; ++s) {
                    if (!tok_strs[s].empty()) {
                        has_strs = true;
                        break;
                    }
                }

                if (!has_strs) {
                    for (int s = 0; s < S_deltas; ++s) {
                        int32_t tid = params.token_ids[s + 1];
                        bool is_code = (tid >= 15 && tid <= 24) || (tid >= 32 && tid <= 57) || tid == 12;
                        is_core[s] = is_code;
                        is_prose[s] = !is_code;
                    }
                }

                std::vector<std::vector<int>> segment_indices;
                bool in_segment = false;
                for (int i = 0; i < S_deltas; ++i) {
                    if (!is_prose[i]) {
                        if (!in_segment) {
                            in_segment = true;
                            segment_indices.push_back({i});
                        } else {
                            segment_indices.back().push_back(i);
                        }
                    } else {
                        in_segment = false;
                    }
                }

                std::vector<float> boost_multipliers(S_deltas, 1.0f);
                for (const auto& seg : segment_indices) {
                    bool contains_core = false;
                    for (int idx : seg) {
                        if (is_core[idx]) {
                            contains_core = true;
                            break;
                        }
                    }
                    if (contains_core) {
                        for (int idx : seg) {
                            int32_t tid = params.token_ids[idx + 1];
                            int count = 1;
                            auto it = token_counts.find(tid);
                            if (it != token_counts.end()) {
                                count = it->second;
                            }
                            float idf = std::log(static_cast<float>(std::max(params.session_len, 2)) / (count + 0.1f));
                            float rarity_weight = std::max(1.0f, std::min(idf, 6.0f));
                            boost_multipliers[idx] = tok_boost * (rarity_weight / 2.0f);
                        }
                    }
                }

                // ── Owner capture (relational locality; port of the MLX
                // _apply_owner_capture, DIFFKV_RESIDUAL_OWNER_CAPTURE default ON).
                // A fact's exact residuals preserve its VALUE (digits are is_core)
                // but not its OWNER — entity names are title-case → is_prose →
                // never boosted, so they survive only as rank-r recon and decode
                // as corrupted neighbors ("Okazaki"→"Okinawa") while the value is
                // emitted exactly (MLX binding probe 2026-07-12: compressed
                // list-all 1/6 → 6/6 with this fix; value→entity corruption
                // eliminated). For each core segment, walk LEFT up to OWNER_DIST
                // tokens to the nearest capitalized non-function word, expand to
                // the full surface run (subword continuations right, multi-word
                // name left), and boost those rows with the same idf weight.
                {
                    static const bool owner_on = []() {
                        const char* e = std::getenv("DIFFKV_RESIDUAL_OWNER_CAPTURE");
                        return !(e && std::string(e) == "0");
                    }();
                    static const int owner_dist = []() {
                        const char* e = std::getenv("DIFFKV_RESIDUAL_OWNER_DIST");
                        if (e) { try { return std::stoi(e); } catch (...) {} }
                        return 12;
                    }();
                    static const std::unordered_set<std::string> owner_stop = {
                        "the", "a", "an", "this", "that", "these", "those", "its",
                        "their", "his", "her", "our", "your", "my", "it", "he",
                        "she", "they", "we", "in", "on", "at", "of", "for", "and",
                        "but", "or", "if", "as", "by", "with", "from", "to", "is",
                        "are", "was", "were", "there", "here",
                    };
                    auto lower_of = [](std::string s) {
                        for (auto& c : s) c = std::tolower(static_cast<unsigned char>(c));
                        return s;
                    };
                    auto strip_of = [](const std::string& s) {
                        size_t b = s.find_first_not_of(" \t\n\r");
                        if (b == std::string::npos) return std::string();
                        size_t e = s.find_last_not_of(" \t\n\r");
                        return s.substr(b, e - b + 1);
                    };
                    // Uppercase check accepts ASCII A-Z plus the Latin-1
                    // supplement capitals (À-Þ, UTF-8 lead 0xC3 + 0x80-0x9E) so
                    // accented entity names behave like the Python isupper()
                    // in the MLX reference (parity; full Unicode is out of scope).
                    auto is_upper_start = [](const std::string& s) {
                        if (s.empty()) return false;
                        unsigned char c0 = (unsigned char)s[0];
                        if (c0 >= 'A' && c0 <= 'Z') return true;
                        if (c0 == 0xC3 && s.size() >= 2) {
                            unsigned char c1 = (unsigned char)s[1];
                            return c1 >= 0x80 && c1 <= 0x9E && c1 != 0x97; // À-Þ minus ×
                        }
                        return false;
                    };
                    if (owner_on) {
                        for (const auto& seg : segment_indices) {
                            bool has_core = false;
                            for (int i2 : seg) if (is_core[i2]) { has_core = true; break; }
                            if (!has_core || seg.empty()) continue;
                            int run_end = -1;
                            int j = seg[0] - 1, steps = 0;
                            while (j >= 0 && steps < owner_dist) {
                                std::string sc = strip_of(tok_strs[j]);
                                if (is_upper_start(sc) &&
                                    owner_stop.find(lower_of(sc)) == owner_stop.end()) {
                                    run_end = j;
                                    break;
                                }
                                --j; ++steps;
                            }
                            if (run_end < 0) continue;
                            // right: subword continuations (no leading space, alphabetic)
                            auto is_alpha_str2 = [](const std::string& s) {
                                if (s.empty()) return false;
                                for (char c : s) {
                                    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) return false;
                                }
                                return true;
                            };
                            int k_hi = run_end;
                            while (k_hi + 1 < S_deltas) {
                                const std::string& nxt = tok_strs[k_hi + 1];
                                if (!nxt.empty() && nxt[0] != ' ' && nxt[0] != '\n' &&
                                    is_alpha_str2(strip_of(nxt))) {
                                    ++k_hi;
                                } else break;
                            }
                            // left: preceding capitalized words of a multi-word name
                            int k_lo = run_end;
                            while (k_lo - 1 >= 0) {
                                const std::string& prv = tok_strs[k_lo - 1];
                                std::string pc = strip_of(prv);
                                if (!prv.empty() && (prv[0] == ' ' || prv[0] == '\n') &&
                                    is_upper_start(pc) &&
                                    owner_stop.find(lower_of(pc)) == owner_stop.end()) {
                                    --k_lo;
                                } else break;
                            }
                            for (int i2 = k_lo; i2 <= k_hi && i2 < S_deltas; ++i2) {
                                if (boost_multipliers[i2] > 1.0f) continue;
                                int32_t tid = params.token_ids[i2 + 1];
                                int count = 1;
                                auto it = token_counts.find(tid);
                                if (it != token_counts.end()) count = it->second;
                                float idf = std::log(static_cast<float>(std::max(params.session_len, 2)) / (count + 0.1f));
                                float rarity_weight = std::max(1.0f, std::min(idf, 6.0f));
                                boost_multipliers[i2] = tok_boost * (rarity_weight / 2.0f);
                            }
                        }
                    }
                }

                // ── Table capture (structured-data lines; port of the MLX
                // _apply_table_capture, DIFFKV_RESIDUAL_TABLE_CAPTURE).
                // Tables break both capture rules at once: (1) the digit cells
                // are all is_core, so a block holding a table plus technical
                // filler carries MORE boosted rows than MAX_RESIDUAL slots
                // (measured 2026-07-13: 181 boosted / 128 slots on a NAT-style
                // table straddling a block boundary in paper text) and the
                // err-ranked cut drops table fragments structure-blind — the
                // decoder reassembles a plausible-but-wrong table (values
                // migrate rows); (2) header/unit cells ('Kernel', 'imgs',
                // '/sec') are prose/core-less → never boosted → rank-r smear —
                // real units come back fabricated ('G/s'). Fix: every token on
                // a table-like LINE (>=2 '|' or '&' separators among >=3
                // tokens) gets the core boost × DIFFKV_RESIDUAL_TABLE_PRIORITY
                // (default 4) so err×boost keeps whole table lines ahead of
                // ordinary boosted segments; the regular low-error rows
                // (separator dashes, pipes) degrade first under saturation.
                {
                    static const bool table_on = []() {
                        // Default ON (measured 2026-07-13: MLX 16k straddled-table
                        // list-all 3/6 → 6/6 == dense with recall gates green;
                        // native A/B in benchmarks/table_probe_native.py).
                        const char* e = std::getenv("DIFFKV_RESIDUAL_TABLE_CAPTURE");
                        return !(e && std::string(e) == "0");
                    }();
                    static const float table_priority = []() {
                        const char* e = std::getenv("DIFFKV_RESIDUAL_TABLE_PRIORITY");
                        if (e) { try { return std::stof(e); } catch (...) {} }
                        return 4.0f;
                    }();
                    if (table_on) {
                        auto strip_ws = [](const std::string& s) {
                            size_t b = s.find_first_not_of(" \t\n\r");
                            if (b == std::string::npos) return std::string();
                            size_t e = s.find_last_not_of(" \t\n\r");
                            return s.substr(b, e - b + 1);
                        };
                        auto has_digit_c = [](const std::string& s) {
                            for (char c : s) if (c >= '0' && c <= '9') return true;
                            return false;
                        };
                        // Split the block into lines once; two rules per line
                        // (mirrors the MLX _detect_table_rows — keep in sync):
                        // 1. SEPARATOR rule: >= 2 standalone '|'/'&' tokens +
                        //    shape guard (line-initial separator, LaTeX \\
                        //    terminator, or density >= 1/12 with >= 3 seps) —
                        //    the guard rejects prose with inline |x−y| math.
                        // 2. COLUMNAR rule (PDF copy-paste tables have NO
                        //    pipes; without it rows over-compress and decode
                        //    collapses onto one high-salience fragment):
                        //    >= 2 CONSECUTIVE lines of 3..48 tokens ending in
                        //    a digit-bearing token, >= 2 digit tokens, not
                        //    prose-dominated; the header line above joins.
                        std::vector<std::vector<int>> tbl_lines;
                        {
                            std::vector<int> cur;
                            for (int i = 0; i < S_deltas; ++i) {
                                cur.push_back(i);
                                if (tok_strs[i].find('\n') != std::string::npos) {
                                    tbl_lines.push_back(cur);
                                    cur.clear();
                                }
                            }
                            if (!cur.empty()) tbl_lines.push_back(cur);
                        }
                        auto sep_rule = [&](const std::vector<int>& ln) {
                            if ((int)ln.size() < 3) return false;
                            int seps = 0;
                            for (int i2 : ln) {
                                std::string st = strip_ws(tok_strs[i2]);
                                if (st == "|" || st == "&") ++seps;
                            }
                            if (seps < 2) return false;
                            std::string first = strip_ws(tok_strs[ln[0]]);
                            bool starts_sep = !first.empty() && (first[0] == '|' || first[0] == '&');
                            bool latex_term = tok_strs[ln.back()].find("\\\\") != std::string::npos;
                            return starts_sep || latex_term ||
                                   (seps >= 3 && seps * 12 >= (int)ln.size());
                        };
                        // Tightened candidate: real table rows are SHORT.
                        // References ("[3] D. Achlioptas, ..., 2001.") are
                        // 25-45 tokens, prose-heavy, and also end in digits —
                        // marking them STEALS residual slots from a real
                        // table sharing the block (measured on MLX: markdown
                        // probe 6/6 -> 1/6 before this tightening). <= 20
                        // tokens qualifies outright; 21..48 only when digit
                        // tokens outnumber prose words.
                        auto columnar_cand = [&](const std::vector<int>& ln) {
                            if ((int)ln.size() < 3 || (int)ln.size() > 48) return false;
                            int n_digit = 0, n_prose = 0;
                            for (int i2 : ln) {
                                std::string sc = strip_ws(tok_strs[i2]);
                                if (has_digit_c(sc)) { ++n_digit; continue; }
                                bool alpha_lower = sc.size() >= 3;
                                for (char c : sc) {
                                    if (!(c >= 'a' && c <= 'z')) { alpha_lower = false; break; }
                                }
                                if (alpha_lower) ++n_prose;
                            }
                            if (n_digit < 2) return false;
                            if ((int)ln.size() > 20 && n_digit < n_prose) return false;
                            for (auto it2 = ln.rbegin(); it2 != ln.rend(); ++it2) {
                                std::string sc = strip_ws(tok_strs[*it2]);
                                bool has_alnum2 = false;
                                for (char c : sc) {
                                    if (std::isalnum(static_cast<unsigned char>(c))) { has_alnum2 = true; break; }
                                }
                                if (sc.empty() || !has_alnum2) continue;
                                return has_digit_c(sc);
                            }
                            return false;
                        };
                        // Tiered boost: separator-rule lines (explicit
                        // tables) at full table_priority; columnar lines at
                        // half — under saturation the explicit table always
                        // outranks incidental numeric lines in its block.
                        auto boost_line = [&](const std::vector<int>& ln, float tier) {
                            for (int i2 : ln) {
                                int32_t tid = params.token_ids[i2 + 1];
                                int count = 1;
                                auto it = token_counts.find(tid);
                                if (it != token_counts.end()) count = it->second;
                                float idf = std::log(static_cast<float>(std::max(params.session_len, 2)) / (count + 0.1f));
                                float rarity_weight = std::max(1.0f, std::min(idf, 6.0f));
                                float b = tok_boost * (rarity_weight / 2.0f) * table_priority * tier;
                                boost_multipliers[i2] = std::max(boost_multipliers[i2], b);
                                ++lr_table_rows;
                            }
                        };
                        std::vector<char> cand(tbl_lines.size(), 0);
                        for (size_t li = 0; li < tbl_lines.size(); ++li) {
                            cand[li] = columnar_cand(tbl_lines[li]) ? 1 : 0;
                        }
                        // fired tier per line: 0 none, 2 separator, 1 columnar
                        std::vector<char> fired(tbl_lines.size(), 0);
                        for (size_t li = 0; li < tbl_lines.size(); ++li) {
                            if (sep_rule(tbl_lines[li])) {
                                fired[li] = 2;
                                boost_line(tbl_lines[li], 1.0f);
                            } else if (cand[li] &&
                                       ((li > 0 && cand[li - 1]) ||
                                        (li + 1 < tbl_lines.size() && cand[li + 1]))) {
                                fired[li] = 1;
                                boost_line(tbl_lines[li], 0.5f);
                                if (li > 0 && !cand[li - 1] &&
                                    (int)tbl_lines[li - 1].size() <= 48) {
                                    boost_line(tbl_lines[li - 1], 0.5f);   // header row
                                }
                            }
                        }
                        // CAPTION capture — COLUMNAR runs only. An aligned
                        // table's rows are anonymous numbers; the caption
                        // ("Table 4 reports ...") is its only identity anchor
                        // (without it: one high-salience value reused for
                        // every row at 16k). Separator tables are
                        // self-identifying, and capturing their captions was
                        // measured NET-NEGATIVE on MLX (two exact captions →
                        // the decoder fused both tables into a chimera).
                        // DIFFKV_RESIDUAL_TABLE_CAPTION=0 disables.
                        static const bool caption_on = []() {
                            const char* e = std::getenv("DIFFKV_RESIDUAL_TABLE_CAPTION");
                            return !(e && std::string(e) == "0");
                        }();
                        if (caption_on) {
                            auto lower_c = [](char c) {
                                return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
                            };
                            auto is_caption = [&](const std::vector<int>& ln) {
                                std::string text;
                                for (int i2 : ln) text += tok_strs[i2];
                                for (auto& c : text) c = lower_c(c);
                                for (const char* m : {"table", "tab."}) {
                                    size_t j = text.find(m);
                                    if (j == std::string::npos) continue;
                                    size_t hi = std::min(text.size(), j + strlen(m) + 4);
                                    for (size_t k = j; k < hi; ++k) {
                                        if (text[k] >= '0' && text[k] <= '9') return true;
                                    }
                                }
                                return false;
                            };
                            for (size_t li = 0; li < tbl_lines.size(); ++li) {
                                if (fired[li] != 1 || (li > 0 && fired[li - 1])) continue;
                                for (size_t up = 1; up <= 3 && up <= li; ++up) {
                                    if (is_caption(tbl_lines[li - up])) {
                                        boost_line(tbl_lines[li - up], 0.5f);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                // Apply window boost (Phase 2: contiguous runs)
                std::vector<float> final_boosts = boost_multipliers;
                const int W = 2;
                for (int i = 0; i < S_deltas; ++i) {
                    if (boost_multipliers[i] > 1.0f) {
                        int start_j = std::max(0, i - W);
                        int end_j = std::min(S_deltas - 1, i + W);
                        for (int j = start_j; j <= end_j; ++j) {
                            final_boosts[j] = std::max(final_boosts[j], boost_multipliers[i]);
                        }
                    }
                }
                boost_multipliers = final_boosts;

                for (int s = 0; s < S_deltas; ++s) {
                    joint_err[s] *= boost_multipliers[s];
                    if (boost_multipliers[s] > 1.0f) lr_boosted_rows++;
                }
            }
        }

        // Saturated table block: a block whose table lines + other boosted rows
        // outnumber the slots left after the coverage quota must spend EVERY
        // slot on ranked rows — the table IS the block's content, a stride
        // scaffold through it is redundant, and each coverage slot evicts one
        // exact table cell. (Coverage keeps its role in prose blocks; this
        // only fires when table rows were actually marked.)
        if (lr_table_rows > 0 && !cov_cols.empty() &&
            lr_boosted_rows + (int)cov_cols.size() > MR) {
            cov_cols.clear();
        }

        // Budget floor for boosted rows: a fact block's boosted set (value +
        // owner + window glue) exceeds the easy-block cap of 8 — without this
        // floor the adaptive budget silently evicts the owner the capture just
        // fought for (mirrors the MLX wrapper's floor). Coverage-aware: the
        // stride-stratified coverage quota takes cov_frac of the budget with a
        // +1e12 bonus, so the ranked (boosted) rows only get the remaining
        // (1-cov_frac) share — divide the requirement through, or enabling
        // coverage would silently evict the very rows the boosts protect.
        // Margin env: DIFFKV_RESIDUAL_FLOOR_MARGIN (default 4).
        static const int floor_margin = []() {
            const char* e = std::getenv("DIFFKV_RESIDUAL_FLOOR_MARGIN");
            if (e) { try { return std::stoi(e); } catch (...) {} }
            return 4;
        }();
        if (lr_boosted_rows > 0) {
            int need = lr_boosted_rows + floor_margin;
            if (cov_frac > 0.0f && cov_frac < 1.0f) {
                need = (int)std::ceil((float)need / (1.0f - cov_frac));
            }
            adaptive_MR = std::max(adaptive_MR, std::min(MR, need));
        }
        int n_max = std::min(S_deltas, adaptive_MR);
        std::vector<int> idx(S_deltas);
        for (int i = 0; i < S_deltas; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return joint_err[a] > joint_err[b]; });

        // Two-phase selection: ranked (distinctive) rows FIRST — they are the
        // routing signature — then the coverage scaffold appended in whatever
        // budget remains (see the coverage note above).
        int n_cov_eff = 0;
        if (!cov_cols.empty() && n_max > 0) {
            n_cov_eff = std::min((int)cov_cols.size(),
                                 std::max(1, (int)std::round(cov_frac * n_max)));
        }
        const int n_ranked = std::max(0, n_max - n_cov_eff);
        std::vector<char> res_taken(S_deltas, 0);
        int written = 0;
        for (int ii = 0; ii < S_deltas && written < n_ranked; ++ii) {
            int s = idx[ii];
            if (joint_err[s] <= 1e-4f) continue;
            res_taken[s] = 1;
            params.out_res_K_pos[written] = s;
            params.out_res_V_pos[written] = s;
            for (int f = 0; f < F; ++f) {
                params.out_res_K_val[(size_t)written * F + f] = ggml_fp32_to_fp16(resid[(size_t)s * joint_F + f]);
                params.out_res_V_val[(size_t)written * F + f] = ggml_fp32_to_fp16(resid[(size_t)s * joint_F + F + f]);
            }
            written++;
        }
        for (size_t ci = 0; ci < cov_cols.size() && written < n_max; ++ci) {
            int s = cov_cols[ci];
            if (s < 0 || s >= S_deltas || res_taken[s]) continue;
            if (joint_err[s] <= 1e-4f) continue;   // already effectively exact
            res_taken[s] = 1;
            params.out_res_K_pos[written] = s;
            params.out_res_V_pos[written] = s;
            for (int f = 0; f < F; ++f) {
                params.out_res_K_val[(size_t)written * F + f] = ggml_fp32_to_fp16(resid[(size_t)s * joint_F + f]);
                params.out_res_V_val[(size_t)written * F + f] = ggml_fp32_to_fp16(resid[(size_t)s * joint_F + F + f]);
            }
            written++;
        }

        static const bool dbg_residuals = (std::getenv("DIFFKV_DBG_RESIDUALS") != nullptr);
        if (dbg_residuals && written > 0) {
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

void print_rank_energy_histogram() {
    std::lock_guard<std::mutex> lock(g_hist_mutex);
    if (g_svd_count == 0) {
        std::cerr << "\n[Rank Energy Profile] No SVD computations recorded.\n";
        return;
    }
    std::cerr << "\n==================================================\n";
    std::cerr << "       SVD RANK-ENERGY HISTOGRAM & PROFILE\n";
    std::cerr << "       Averaged over " << g_svd_count << " block compressions\n";
    std::cerr << "==================================================\n";
    for (int r = 0; r < 16; ++r) {
        double avg_pct = (g_cum_energy[r] / g_total_energy) * 100.0;
        std::cerr << "  Rank " << std::setw(2) << (r + 1) << ":  "
                  << std::fixed << std::setprecision(4) << avg_pct << "%\n";
    }
    std::cerr << "==================================================\n\n";
}

} // namespace diffkv
