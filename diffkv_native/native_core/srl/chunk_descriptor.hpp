// chunk_descriptor.hpp
// Translation of chunk_descriptor.py to C++17.
// Computes a 64-dim semantic fingerprint for compressed KV blocks.
// Uses macOS Accelerate framework (cblas_sgemv / cblas_sgemm) for BLAS ops.

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

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

namespace diffkv {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int DESC_DIM = 64;

// ---------------------------------------------------------------------------
// IEEE 754 half-precision (float16) utilities
// Bit-cast only — no hardware FP16 needed.
// ---------------------------------------------------------------------------

/// Convert IEEE 754 single-precision float to half-precision bits (uint16_t).
inline uint16_t float_to_f16(float v) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));

    uint32_t sign     = (bits >> 31) & 0x1u;
    uint32_t exponent = (bits >> 23) & 0xFFu;
    uint32_t mantissa =  bits        & 0x7FFFFFu;

    // Handle special values
    if (exponent == 0xFF) {  // Inf / NaN
        uint16_t m16 = mantissa ? 0x0200u : 0u;  // NaN preserves, Inf -> 0 mantissa
        return static_cast<uint16_t>((sign << 15) | 0x7C00u | m16);
    }

    int32_t new_exp = static_cast<int32_t>(exponent) - 127 + 15;

    if (new_exp >= 31) {  // Overflow -> Inf
        return static_cast<uint16_t>((sign << 15) | 0x7C00u);
    }
    if (new_exp <= 0) {  // Underflow -> subnormal or zero
        if (new_exp < -10) {
            return static_cast<uint16_t>(sign << 15);
        }
        mantissa = (mantissa | 0x800000u) >> (1 - new_exp);
        // Round-to-nearest
        if (mantissa & 0x1000u) mantissa += 0x2000u;
        return static_cast<uint16_t>((sign << 15) | (mantissa >> 13));
    }

    // Round-to-nearest
    uint32_t m = mantissa + 0x00001000u;  // round bit
    if (m & 0x800000u) { m = 0; ++new_exp; }
    if (new_exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00u);

    return static_cast<uint16_t>((sign << 15)
                                | (static_cast<uint32_t>(new_exp) << 10)
                                | (m >> 13));
}

/// Convert half-precision bits (uint16_t) to single-precision float.
inline float f16_to_float(uint16_t h) noexcept {
    uint32_t sign     = (h >> 15) & 0x1u;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa =  h        & 0x3FFu;

    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign << 31;
        } else {
            // Subnormal
            exponent = 1;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; --exponent; }
            mantissa &= 0x3FFu;
            bits = (sign << 31) | ((exponent - 1 + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline void l2_normalize(float* v, int n) noexcept {
    float sq = 0.0f;
    for (int i = 0; i < n; ++i) sq += v[i] * v[i];
    float norm = std::sqrt(sq) + 1e-8f;
    float inv = 1.0f / norm;
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

// ---------------------------------------------------------------------------
// compute_descriptor
//
// Computes a DESC_DIM-dimensional semantic fingerprint for one compressed
// KV block and writes it as float16 values into out_desc.
//
// Algorithm (mirrors chunk_descriptor.py):
//   1. anchor_mean  = mean of anchor_K over kv_heads dim     -> [head_dim]
//   2. U_f32        = U_int8 * U_scale  (dequantize)          -> [seq_len, rank]
//   3. mean_u       = column mean of U_f32                    -> [rank]
//   4. vk_mean      = mean of V_K over kv_heads dim           -> [rank, head_dim]
//   5. delta_centroid = mean_u @ vk_mean                      -> [head_dim]
//   6. centroid     = anchor_mean + delta_centroid            -> [head_dim]
//   7. desc         = W_proj @ centroid                       -> [DESC_DIM]
//   8. L2-normalise desc, store as float16
// ---------------------------------------------------------------------------
inline void compute_descriptor(
    const uint16_t* anchor_K,  // [kv_heads, head_dim] as float16
    const int8_t*   U_int8,    // [seq_len, rank]
    float           U_scale,
    const uint16_t* V_K,       // [rank, kv_heads, head_dim] as float16
    const float*    W_proj,    // [DESC_DIM, head_dim]
    int kv_heads,
    int head_dim,
    int seq_len,
    int rank,
    uint16_t*       out_desc   // [DESC_DIM] float16 output
) {
    // ------------------------------------------------------------------
    // Step 1: anchor_mean = mean(anchor_K, axis=0)  -> [head_dim]
    // anchor_K layout: [kv_heads][head_dim]
    // ------------------------------------------------------------------
    std::vector<float> anchor_mean(head_dim, 0.0f);
    for (int h = 0; h < kv_heads; ++h) {
        for (int d = 0; d < head_dim; ++d) {
            anchor_mean[d] += f16_to_float(anchor_K[h * head_dim + d]);
        }
    }
    float inv_kv = 1.0f / static_cast<float>(kv_heads);
    for (int d = 0; d < head_dim; ++d) anchor_mean[d] *= inv_kv;

    // ------------------------------------------------------------------
    // Step 2+3: U_f32 = U_int8 * U_scale;  mean_u = mean(U_f32, axis=0)
    // U_int8 layout: [seq_len][rank]
    // mean_u -> [rank]
    // ------------------------------------------------------------------
    std::vector<float> mean_u(rank, 0.0f);
    float inv_seq = 1.0f / static_cast<float>(seq_len > 0 ? seq_len : 1);
    for (int s = 0; s < seq_len; ++s) {
        for (int r = 0; r < rank; ++r) {
            mean_u[r] += static_cast<float>(U_int8[s * rank + r]) * U_scale;
        }
    }
    for (int r = 0; r < rank; ++r) mean_u[r] *= inv_seq;

    // ------------------------------------------------------------------
    // Step 4: vk_mean = mean(V_K, axis=1)  -> [rank, head_dim]
    // V_K layout: [rank][kv_heads][head_dim]
    // ------------------------------------------------------------------
    std::vector<float> vk_mean(rank * head_dim, 0.0f);
    for (int r = 0; r < rank; ++r) {
        for (int h = 0; h < kv_heads; ++h) {
            const uint16_t* row = V_K + (r * kv_heads + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                vk_mean[r * head_dim + d] += f16_to_float(row[d]);
            }
        }
        for (int d = 0; d < head_dim; ++d) vk_mean[r * head_dim + d] *= inv_kv;
    }

    // ------------------------------------------------------------------
    // Step 5: delta_centroid = mean_u @ vk_mean  -> [head_dim]
    // BLAS: y = alpha * A * x + beta * y
    //   A = vk_mean [rank x head_dim] in row-major  (CblasTrans needed for col-major)
    //   x = mean_u [rank]
    //   y = delta_centroid [head_dim]
    //
    // cblas_sgemv(CblasRowMajor, CblasTrans, M, N, alpha, A, lda, x, incx, beta, y, incy)
    //   with M=rank, N=head_dim computes: y = alpha * A^T * x + beta * y
    //   which gives [head_dim] = [rank,head_dim]^T * [rank]  ✓
    // ------------------------------------------------------------------
    std::vector<float> delta_centroid(head_dim, 0.0f);
    cblas_sgemv(CblasRowMajor, CblasTrans,
                rank, head_dim,
                1.0f, vk_mean.data(), head_dim,
                mean_u.data(), 1,
                0.0f, delta_centroid.data(), 1);

    // ------------------------------------------------------------------
    // Step 6: centroid = anchor_mean + delta_centroid  -> [head_dim]
    // ------------------------------------------------------------------
    std::vector<float> centroid(head_dim);
    for (int d = 0; d < head_dim; ++d)
        centroid[d] = anchor_mean[d] + delta_centroid[d];

    // ------------------------------------------------------------------
    // Step 7: desc = W_proj @ centroid  -> [DESC_DIM]
    // W_proj: [DESC_DIM, head_dim] row-major
    // cblas_sgemv: y = A * x, A is [DESC_DIM x head_dim]
    // ------------------------------------------------------------------
    std::vector<float> desc(DESC_DIM, 0.0f);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                DESC_DIM, head_dim,
                1.0f, W_proj, head_dim,
                centroid.data(), 1,
                0.0f, desc.data(), 1);

    // ------------------------------------------------------------------
    // Step 8: L2-normalize and convert to float16
    // ------------------------------------------------------------------
    l2_normalize(desc.data(), DESC_DIM);
    for (int i = 0; i < DESC_DIM; ++i) {
        out_desc[i] = float_to_f16(desc[i]);
    }
}

// ---------------------------------------------------------------------------
// compute_query_descriptor
//
// Projects pooled query vector through W_proj to produce a DESC_DIM float32
// descriptor (L2-normalized).
//
//   q_mean = mean of Q over q_heads    -> [head_dim]
//   desc   = W_proj @ q_mean           -> [DESC_DIM]
//   L2-normalize desc
// ---------------------------------------------------------------------------
inline void compute_query_descriptor(
    const float* Q,        // [q_heads, head_dim] float32
    const float* W_proj,   // [DESC_DIM, head_dim] float32
    int q_heads,
    int head_dim,
    float* out_desc        // [DESC_DIM] float32 output
) {
    // Pool over heads -> q_mean [head_dim]
    std::vector<float> q_mean(head_dim, 0.0f);
    float inv_h = 1.0f / static_cast<float>(q_heads > 0 ? q_heads : 1);
    for (int h = 0; h < q_heads; ++h) {
        const float* row = Q + h * head_dim;
        for (int d = 0; d < head_dim; ++d)
            q_mean[d] += row[d];
    }
    for (int d = 0; d < head_dim; ++d) q_mean[d] *= inv_h;

    // Project: desc = W_proj @ q_mean
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                DESC_DIM, head_dim,
                1.0f, W_proj, head_dim,
                q_mean.data(), 1,
                0.0f, out_desc, 1);

    // L2 normalize
    l2_normalize(out_desc, DESC_DIM);
}

} // namespace diffkv
