#include "native_core/diffkv_core/include/srl_router.hpp"
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
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace diffkv {

void compute_query_desc(
    const float* Q,        // [H_q, D]
    int          H_q,
    int          D,
    const float* W_proj,   // [desc_dim, D]
    int          desc_dim,
    float*       out_desc  // [desc_dim]
) {
    if (H_q <= 0 || D <= 0 || desc_dim <= 0) return;

    // 1. Mean-pool over all query heads: q_mean = sum(Q[h,:]) / H_q
    std::vector<float> q_mean(D, 0.0f);
    for (int h = 0; h < H_q; ++h) {
        for (int d = 0; d < D; ++d) {
            q_mean[d] += Q[h * D + d];
        }
    }
    for (int d = 0; d < D; ++d) {
        q_mean[d] /= H_q;
    }

    // 2. Project into descriptor space: desc = W_proj @ q_mean
    cblas_sgemv(CblasRowMajor, CblasNoTrans, desc_dim, D, 1.0f, W_proj, D, q_mean.data(), 1, 0.0f, out_desc, 1);

    // 3. L2-normalize to unit sphere
    float norm = 0.0f;
    for (int i = 0; i < desc_dim; ++i) {
        norm += out_desc[i] * out_desc[i];
    }
    norm = std::sqrt(norm);
    if (norm < 1e-8f) norm = 1e-8f;
    
    float inv_norm = 1.0f / norm;
    for (int i = 0; i < desc_dim; ++i) {
        out_desc[i] *= inv_norm;
    }
}

void semantic_search_topk(
    const float*   q_desc,
    const float*   desc_matrix,
    int            N,
    int            desc_dim,
    int            k,
    int32_t*       out_indices,
    float*         out_scores
) {
    if (N <= 0 || desc_dim <= 0 || k <= 0) return;

    // 1. Dot product over all descriptors
    std::vector<float> scores(N, 0.0f);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, desc_dim, 1.0f, desc_matrix, desc_dim, q_desc, 1, 0.0f, scores.data(), 1);

    int k_clamped = std::min(k, N);

    // 2. Top-k sorting
    std::vector<int32_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k_clamped, idx.end(),
                      [&](int32_t a, int32_t b) {
                          return scores[a] > scores[b];
                      });

    for (int i = 0; i < k_clamped; ++i) {
        out_indices[i] = idx[i];
        out_scores[i] = scores[idx[i]];
    }
}

void anchor_screen(
    const float*   Q,
    int            H_q,
    int            D,
    const float*   anchors_K,
    int            N_pool,
    int            kv_heads,
    const int32_t* candidate_slots,
    int            M,
    float          scale,
    int            k_keep,
    int32_t*       out_slots
) {
    if (M <= 0 || k_keep <= 0) return;

    if (M <= k_keep) {
        std::copy(candidate_slots, candidate_slots + M, out_slots);
        return;
    }

    // 1. Mean-pool query over all heads -> q_mean
    std::vector<float> q_mean(D, 0.0f);
    for (int h = 0; h < H_q; ++h) {
        for (int d = 0; d < D; ++d) {
            q_mean[d] += Q[h * D + d];
        }
    }
    for (int d = 0; d < D; ++d) {
        q_mean[d] /= H_q;
    }

    // 2. Gather anchor keys for candidate slots and average over kv_heads
    std::vector<float> anc_flat(M * D, 0.0f);
    for (int m = 0; m < M; ++m) {
        int slot_id = candidate_slots[m];
        if (slot_id < 0 || slot_id >= N_pool) continue;
        for (int h = 0; h < kv_heads; ++h) {
            int base = slot_id * kv_heads * D + h * D;
            for (int d = 0; d < D; ++d) {
                anc_flat[m * D + d] += anchors_K[base + d];
            }
        }
        for (int d = 0; d < D; ++d) {
            anc_flat[m * D + d] /= kv_heads;
        }
    }

    // 3. Dot product with mean query
    std::vector<float> scores(M, 0.0f);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, D, scale, anc_flat.data(), D, q_mean.data(), 1, 0.0f, scores.data(), 1);

    // 4. Top-k sorting
    int k_clamped = std::min(k_keep, M);
    std::vector<int32_t> idx(M);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k_clamped, idx.end(),
                      [&](int32_t a, int32_t b) {
                          return scores[a] > scores[b];
                      });

    for (int i = 0; i < k_clamped; ++i) {
        out_slots[i] = candidate_slots[idx[i]];
    }
}

} // namespace diffkv
