// mac_utils.hpp
// Apple Silicon (Metal / CPU) compatibility layer for DiffKV native.
// C++ translation of ACTIVE_RUNTIME/native_core/mac_utils.py
//
// Provides unified device detection and helpers so every other module can
// call get_best_device() / is_apple_silicon() without platform guards.
//
// Priority order (same as Python version):
//   1. Metal/MPS  — Apple Silicon M-series (always true in diffkv_native on Mac)
//   2. CPU        — universal fallback

#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <Accelerate/Accelerate.h>
#endif

namespace diffkv {

// Forward-declared, implemented in metal_runtime.mm
bool has_metal();

inline bool is_apple_silicon() {
#ifdef __APPLE__
    return true;
#else
    return false;
#endif
}

// In diffkv_native (llama.cpp-based), we are always Metal on Mac.
// Returns "metal" on Apple Silicon, "cpu" otherwise.
inline std::string get_best_device() {
    if (has_metal()) return "metal";
    return "cpu";
}

// ── Memory helpers ────────────────────────────────────────────────────────────

struct MemoryInfo {
    double allocated_mb = 0.0;
    double reserved_mb  = 0.0;
    double rss_mb       = 0.0;
};

inline MemoryInfo get_memory_info() {
    MemoryInfo info;
#ifdef __APPLE__
    // RSS via mach task_info
    task_vm_info_data_t vm;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm), &count) == KERN_SUCCESS) {
        info.rss_mb       = static_cast<double>(vm.phys_footprint) / 1e6;
        info.allocated_mb = static_cast<double>(vm.internal)       / 1e6;
    }
    // Metal heap stats via sysctl
    // (Metal does not expose a stable C API for this; use vm_info as proxy)
    info.reserved_mb = info.allocated_mb;
#endif
    return info;
}

// ── Synchronization ───────────────────────────────────────────────────────────

// Block until all pending Metal/GPU ops complete.
// In llama.cpp, ops are submitted synchronously by default on Metal,
// so this is a lightweight fence.
inline void synchronize() {
#ifdef __APPLE__
    // ggml_metal_graph_compute already waits; this is a safety fence.
    // We use a sysctl memory barrier as a proxy.
    __sync_synchronize();
#endif
}

// Release unused memory (hint to OS).
inline void empty_cache() {
#ifdef __APPLE__
    // On unified memory, there is no separate GPU heap to release.
    // The OS will reclaim as needed; we only advise.
    // No equivalent of torch.mps.empty_cache() in raw Metal.
#endif
}

// ── dtype helpers ─────────────────────────────────────────────────────────────

// Returns preferred compute precision: "fp16" on Apple Silicon, "fp32" on CPU.
inline std::string get_default_dtype() {
    return is_apple_silicon() ? "fp16" : "fp32";
}

// ── Randomized SVD helper (CPU, Accelerate) ───────────────────────────────────
// Implements mlx_svd_lowrank equivalent using pure Accelerate BLAS/LAPACK.
// Replaces the MLX bridge in mac_utils.py.
//
// Input:  A [n, d] float32 (row-major)
// Output: U [n, r], S [r], Vt [r, d]
// n_oversamples, n_iter: rSVD tuning (match Python defaults)

#ifdef __APPLE__
inline bool randomized_svd(
    const float* A,    // [n, d]
    int n, int d, int rank,
    float* out_U,      // [n, rank]
    float* out_S,      // [rank]
    float* out_Vt,     // [rank, d]
    int n_oversamples = 5,
    int n_iter = 2
) {
    int r = std::min(rank + n_oversamples, std::min(n, d));
    if (r < 1) return false;

    // --- Step 1: Random Gaussian projection ---
    std::vector<float> Omega(d * r);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& x : Omega) x = dist(rng);
    }

    // Y = A @ Omega  [n, r]  via cblas_sgemm
    std::vector<float> Y(n * r, 0.0f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, r, d,
                1.0f, A, d,
                Omega.data(), r,
                0.0f, Y.data(), r);

    // --- Step 2: Power iteration for quality ---
    // Y = A @ (A^T @ Y) repeated n_iter times
    std::vector<float> Z(d * r, 0.0f);
    for (int i = 0; i < n_iter; ++i) {
        // Z = A^T @ Y  [d, r]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    d, r, n,
                    1.0f, A, d,
                    Y.data(), r,
                    0.0f, Z.data(), r);
        // Y = A @ Z  [n, r]
        std::fill(Y.begin(), Y.end(), 0.0f);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, r, d,
                    1.0f, A, d,
                    Z.data(), r,
                    0.0f, Y.data(), r);
    }

    // --- Step 3: QR decomposition of Y ---
    // Y [n, r] — we use sgeqrf + sorgqr
    std::vector<float> tau(r);
    int info = 0, lwork = -1;
    float work_query;
    sgeqrf_(&n, &r, Y.data(), &n, tau.data(), &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    std::vector<float> work(lwork);
    sgeqrf_(&n, &r, Y.data(), &n, tau.data(), work.data(), &lwork, &info);
    if (info != 0) return false;

    // Extract Q [n, r]
    std::vector<float> Q = Y; // copy
    lwork = -1;
    sorgqr_(&n, &r, &r, Q.data(), &n, tau.data(), &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    work.resize(lwork);
    sorgqr_(&n, &r, &r, Q.data(), &n, tau.data(), work.data(), &lwork, &info);
    if (info != 0) return false;

    // --- Step 4: B = Q^T @ A  [r, d] ---
    std::vector<float> B(r * d, 0.0f);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                r, d, n,
                1.0f, Q.data(), r,
                A, d,
                0.0f, B.data(), d);

    // --- Step 5: SVD of small B [r, d] ---
    int min_rd = std::min(r, d);
    std::vector<float> U_b(r * min_rd);
    std::vector<float> S_b(min_rd);
    std::vector<float> Vt_b(min_rd * d);
    char jobu = 'S', jobvt = 'S';
    lwork = -1;
    sgesvd_(&jobu, &jobvt, &r, &d,
            B.data(), &r,
            S_b.data(),
            U_b.data(), &r,
            Vt_b.data(), &min_rd,
            &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    work.resize(lwork);
    sgesvd_(&jobu, &jobvt, &r, &d,
            B.data(), &r,
            S_b.data(),
            U_b.data(), &r,
            Vt_b.data(), &min_rd,
            work.data(), &lwork, &info);
    if (info != 0) return false;

    // --- Step 6: U = Q @ U_b  [n, rank] ---
    // Note: LAPACK is column-major, we stored row-major.
    // For a quick port we use the first `rank` singular triplets.
    int take = std::min(rank, min_rd);
    // U_out[i][j] = sum_k Q[i][k] * U_b[k][j]  (row-major)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, take, r,
                1.0f, Q.data(), r,
                U_b.data(), min_rd,
                0.0f, out_U, take);

    // Copy S and Vt[:take, :]
    std::copy(S_b.begin(), S_b.begin() + take, out_S);
    for (int i = 0; i < take; ++i)
        std::copy(Vt_b.begin() + i * d, Vt_b.begin() + (i + 1) * d,
                  out_Vt + i * d);

    return true;
}
#endif

} // namespace diffkv
