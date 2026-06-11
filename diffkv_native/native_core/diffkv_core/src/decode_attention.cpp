#include "native_core/diffkv_core/include/decode_attention.hpp"
#include "ggml.h"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

inline void cblas_sgemv(
    enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
    int M, int N, float alpha, const float *A, int lda,
    const float *X, int incX, float beta, float *Y, int incY
) {
    if (order != CblasRowMajor) return;
    if (trans == CblasNoTrans) {
        for (int i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += A[i * lda + j] * X[j * incX];
            }
            Y[i * incY] = alpha * sum + beta * Y[i * incY];
        }
    } else if (trans == CblasTrans) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < M; ++i) {
                sum += A[i * lda + j] * X[i * incX];
            }
            Y[j * incY] = alpha * sum + beta * Y[j * incY];
        }
    }
}
#endif
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

namespace diffkv {

void decode_attention(
    const float*    Q,               // [H_q, D]
    const int8_t*   U_pool,          // [N_pool, S_max, R]
    const float*    U_scale_pool,    // [N_pool]
    const uint16_t* VK_pool,         // [N_pool, R, kv_heads, D]
    const uint16_t* VV_pool,         // [N_pool, R, kv_heads, D]
    const uint16_t* anchors_K,       // [N_pool, kv_heads, D]
    const uint16_t* anchors_V,       // [N_pool, kv_heads, D]
    const int32_t*  seq_lens,        // [N_pool]
    const int32_t*  slot_indices,    // [K_active]
    int K_active,
    int N_pool, int S_max, int R,
    int H_q, int kv_heads, int D,
    float scale,
    float* out                      // [H_q, D]
) {
    (void)N_pool;
    if (K_active == 0) {
        std::memset(out, 0, H_q * D * sizeof(float));
        return;
    }

    const int g = H_q / kv_heads;

    // Local buffers to avoid allocations inside the head loop
    std::vector<float> VK_local(R * D);
    std::vector<float> VV_local(R * D);
    std::vector<float> U_local; // dynamic size based on S_k
    std::vector<float> q_proj(R);
    std::vector<float> s_delta_vec;
    std::vector<float> w_token_vec;
    std::vector<float> W_proj(R);
    std::vector<float> V_svd(D);
    std::vector<float> V_anc(D);

    for (int h = 0; h < H_q; ++h) {
        int kv_head = h / g;
        const float* q_h = Q + h * D;

        float m_h = -1e30f;
        float d_h = 0.0f;
        float* o_h = out + h * D;
        std::memset(o_h, 0, D * sizeof(float));

        for (int k = 0; k < K_active; ++k) {
            int slot_id = slot_indices[k];
            int S_k = seq_lens[slot_id];
            if (S_k < 0) S_k = 0;

            float scale_u = U_scale_pool[slot_id];

            // 1. Compute Anchor score (unrotated/rotated as Q is already rotated)
            float score_anc = 0.0f;
            for (int d = 0; d < D; ++d) {
                float ak_val = ggml_fp16_to_fp32(anchors_K[slot_id * kv_heads * D + kv_head * D + d]);
                score_anc += q_h[d] * ak_val;
            }
            float s_anc_scaled = score_anc * scale;

            // 2. Query projection into SVD subspace
            // Dequantize VK_pool slice for this block and kv_head
            for (int r = 0; r < R; ++r) {
                int offset = slot_id * R * kv_heads * D + r * kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    VK_local[r * D + d] = ggml_fp16_to_fp32(VK_pool[offset + d]);
                }
            }

            // q_proj = VK_local * q_h
            cblas_sgemv(CblasRowMajor, CblasNoTrans, R, D, 1.0f, VK_local.data(), D, q_h, 1, 0.0f, q_proj.data(), 1);

            // 3. Delta scores for tokens in block k
            s_delta_vec.resize(S_k);
            if (S_k > 0) {
                U_local.resize(S_k * R);
                for (int t = 0; t < S_k; ++t) {
                    int u_offset = slot_id * S_max * R + t * R;
                    for (int r = 0; r < R; ++r) {
                        U_local[t * R + r] = static_cast<float>(U_pool[u_offset + r]) * scale_u;
                    }
                }

                // s_delta_vec = U_local * q_proj
                cblas_sgemv(CblasRowMajor, CblasNoTrans, S_k, R, 1.0f, U_local.data(), R, q_proj.data(), 1, 0.0f, s_delta_vec.data(), 1);
            }

            // Local max score in block k
            float M_local = s_anc_scaled;
            for (int t = 0; t < S_k; ++t) {
                float t_score = (s_delta_vec[t] + score_anc) * scale;
                if (t_score > M_local) {
                    M_local = t_score;
                }
            }

            // Exponentials
            float E_anc = std::exp(s_anc_scaled - M_local);
            float E_sum = E_anc;
            w_token_vec.resize(S_k);
            for (int t = 0; t < S_k; ++t) {
                float t_score = (s_delta_vec[t] + score_anc) * scale;
                float E_t = std::exp(t_score - M_local);
                E_sum += E_t;
                w_token_vec[t] = E_t; // stored normalized by M_local, not E_sum yet
            }

            // Online softmax update stats
            float m_new = std::max(m_h, M_local);
            float alpha = std::exp(m_h - m_new);
            float beta = std::exp(M_local - m_new);
            float d_new = d_h * alpha + E_sum * beta;

            // 4. Value contribution from block k
            // Dequantize VV_pool slice
            for (int r = 0; r < R; ++r) {
                int offset = slot_id * R * kv_heads * D + r * kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    VV_local[r * D + d] = ggml_fp16_to_fp32(VV_pool[offset + d]);
                }
            }

            // Dequantize anchors_V
            for (int d = 0; d < D; ++d) {
                V_anc[d] = ggml_fp16_to_fp32(anchors_V[slot_id * kv_heads * D + kv_head * D + d]);
            }

            // W_proj = U_local^T * w_token_vec
            std::memset(W_proj.data(), 0, R * sizeof(float));
            if (S_k > 0) {
                cblas_sgemv(CblasRowMajor, CblasTrans, S_k, R, 1.0f, U_local.data(), R, w_token_vec.data(), 1, 0.0f, W_proj.data(), 1);
            }

            // V_svd = VV_local^T * W_proj
            std::memset(V_svd.data(), 0, D * sizeof(float));
            cblas_sgemv(CblasRowMajor, CblasTrans, R, D, 1.0f, VV_local.data(), D, W_proj.data(), 1, 0.0f, V_svd.data(), 1);

            // Update output vector
            float w_total_anc = E_anc;
            for (int t = 0; t < S_k; ++t) {
                w_total_anc += w_token_vec[t];
            }

            for (int d = 0; d < D; ++d) {
                float V_local_d = w_total_anc * V_anc[d] + V_svd[d];
                o_h[d] = o_h[d] * alpha + V_local_d * beta;
            }

            m_h = m_new;
            d_h = d_new;
        }

        // Final normalization
        if (d_h > 0.0f) {
            float inv_d = 1.0f / d_h;
            for (int d = 0; d < D; ++d) {
                o_h[d] *= inv_d;
            }
        }
    }
}

} // namespace diffkv
