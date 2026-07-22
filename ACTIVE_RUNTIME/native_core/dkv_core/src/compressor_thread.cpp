// dkv_core/src/compressor_thread.cpp
// Implementation of DKVCompressorThread.
// Uses cuSOLVER for SVD — all GPU math, no Python involvement.
//
// process_job() runs cusolverDnSgesvd (truncated column extraction) for both
// K and V delta matrices on the GPU.  The slab_u_ptr / slab_v_ptr destinations
// are pre-allocated slab pools; we write top-rank columns/rows directly.

#include "compressor_thread.hpp"
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <cstdio>

namespace dkv {

void DKVCompressorThread::start() {
    if (running_.load()) return;
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&DKVCompressorThread::worker_loop, this);
}

void DKVCompressorThread::stop() {
    running_.store(false, std::memory_order_release);
    if (worker_.joinable())
        worker_.join();
}

bool DKVCompressorThread::submit(const CompressJob& job) {
    if (!queue_.push(job)) {
        queue_overflows_.fetch_add(1, std::memory_order_relaxed);
        return false; // Queue full — block stays DenseResident, no stall
    }
    return true;
}

void DKVCompressorThread::worker_loop() {
    // Create per-thread cuSOLVER handle — avoids multi-thread handle contention.
    cusolverDnHandle_t cusolver_handle = nullptr;
    cusolverStatus_t cs = cusolverDnCreate(&cusolver_handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "[DKV] cusolverDnCreate failed: %d — compressor thread exiting.\n", cs);
        return;
    }

    while (running_.load(std::memory_order_acquire)) {
        auto job_opt = queue_.pop();
        if (!job_opt.has_value()) {
            // No jobs — yield briefly to avoid busy-spinning at 100% CPU
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }
        process_job(*job_opt, cusolver_handle);
        jobs_processed_.fetch_add(1, std::memory_order_relaxed);
    }

    cusolverDnDestroy(cusolver_handle);
}

// ── SVD helper: computes truncated SVD via cusolverDnSgesvd and writes ──────
// top `rank` singular components to the destination slab pointers.
//
// Input:  A [m, n] row-major float32 on GPU (m = block_size, n = feat_dim)
// Output: U_out [rank, m] row-major (first `rank` right singular vectors of A)
//         Vt_out [n, rank] row-major (first `rank` left singular vectors of A)
//
// cuSOLVER uses column-major (Fortran) convention. Passing row-major [m, n] as
// column-major treats it as [m, n]^T = [n, m], so we call gesvd on A^T [n, m].
// A^T = U_cm[n,k] * S * Vt_cm[k,m] in column-major
//   →  A = (Vt_cm^T)[m,k] * S * (U_cm^T)[k,n]  in row-major
// So: right singular vectors of A (V) = U_cm columns
//     left  singular vectors of A (U) = Vt_cm rows (transposed)
//
// We write top `rank` components into the slab: U is [m, rank] (block → rank)
// and Vt is [rank, n] (rank → feature). Pool reader expects [m, rank] U and
// [rank, n] Vt in fp32.
static bool run_svd_and_write_slab(
    cusolverDnHandle_t handle,
    float* A_gpu,    // [m, n] row-major GPU pointer (not modified)
    float* U_out,    // [m, rank] row-major destination (slab U segment)
    float* Vt_out,   // [rank, n] row-major destination (slab Vt segment)
    int m, int n, int rank
) {
    const int k_out = std::min(m, n);
    // Clamp rank to the actual SVD rank
    if (rank > k_out) rank = k_out;

    // Query workspace size — treats A_gpu as [n, m] column-major (= A^T)
    int lwork = 0;
    cusolverStatus_t cs = cusolverDnSgesvd_bufferSize(handle, n, m, &lwork);
    if (cs != CUSOLVER_STATUS_SUCCESS) return false;

    // Allocate temp GPU buffers
    float* d_work = nullptr;
    float* d_S    = nullptr;
    float* d_U    = nullptr;   // [n, k_out] column-major (left sv's of A^T = V of A)
    float* d_Vt   = nullptr;   // [k_out, m] column-major (right sv's of A^T = U^T of A)
    int*   d_info = nullptr;

    if (cudaMalloc(&d_work, sizeof(float) * lwork) != cudaSuccess) goto cleanup_fail;
    if (cudaMalloc(&d_S,    sizeof(float) * k_out)  != cudaSuccess) goto cleanup_fail;
    if (cudaMalloc(&d_U,    sizeof(float) * n * k_out) != cudaSuccess) goto cleanup_fail;
    if (cudaMalloc(&d_Vt,   sizeof(float) * k_out * m) != cudaSuccess) goto cleanup_fail;
    if (cudaMalloc(&d_info, sizeof(int))             != cudaSuccess) goto cleanup_fail;

    // Run SVD: A^T [n, m] col-major = d_U * diag(d_S) * d_Vt
    cs = cusolverDnSgesvd(
        handle,
        'S',    // 'S' = thin U — only first k_out columns; saves memory vs 'A'
        'S',    // 'S' = thin Vt — only first k_out rows
        n, m,               // dimensions of A^T
        A_gpu,  n,          // A^T device ptr, leading dim = n (col-major rows)
        d_S,                // singular values [k_out]
        d_U,    n,          // U of A^T: [n, k_out] col-major
        d_Vt,   k_out,      // Vt of A^T: [k_out, m] col-major
        d_work, lwork,
        /*rwork=*/nullptr,
        d_info
    );
    if (cs != CUSOLVER_STATUS_SUCCESS) goto cleanup_fail;

    // U_out [m, rank] row-major = (d_Vt [k_out, m] col-major)^T restricted to rank rows.
    // In col-major, d_Vt[k_out, m] stores its rows contiguously IF we read it as
    // row-major [m, k_out]. So d_Vt viewed as row-major [m, k_out]: U_out = d_Vt_rowmajor[:, :rank].
    // Simplest: copy the entire [k_out, m] col-major block then Python side can slice to rank.
    // Since the slab always allocates for max_rank, we copy min(rank, k_out) * m floats.
    // NOTE: col-major [k_out, m] viewed as row-major is [m, k_out] — strided copy needed for
    // truncation. For now copy full [k_out * m] and let the consumer (pool reader) slice to rank.
    // This is safe because slab_u_ptr allocation is always [block_size, max_rank] = [m, rank].
    // We copy exactly rank*m floats from d_Vt (first rank rows in col-major = first rank*m floats
    // because Fortran col-major [k_out, m] starts with row 0 for all m columns at stride k_out).
    // Correct approach for row-major output: use cudaMemcpy2D to extract rank rows.
    if (cudaMemcpy2D(
            U_out,              // dst: [rank, m] row-major (stride m floats per row)
            sizeof(float) * m,  // dst pitch (bytes per row)
            d_Vt,               // src: col-major [k_out, m] — column 0 starts at d_Vt[0]
            sizeof(float) * k_out, // src pitch (bytes per col-major column = k_out floats)
            sizeof(float) * rank,  // width: rank floats per destination row (first rank rows of Vt)
            m,                  // height: m columns of A^T = m rows in Vt
            cudaMemcpyDeviceToDevice
        ) != cudaSuccess) goto cleanup_fail;

    // Vt_out [rank, n] row-major = (d_U [n, k_out] col-major)^T restricted to rank cols.
    // Similarly: col-major [n, k_out] viewed as row-major is [k_out, n].
    // We want first `rank` rows of [k_out, n] row-major = first `rank` cols of d_U col-major.
    if (cudaMemcpy2D(
            Vt_out,             // dst: [rank, n] row-major
            sizeof(float) * n,  // dst pitch
            d_U,                // src: col-major [n, k_out]
            sizeof(float) * n,  // src pitch (col-major col stride = n floats)
            sizeof(float) * n,  // width: full n features per row
            rank,               // height: rank rows
            cudaMemcpyDeviceToDevice
        ) != cudaSuccess) goto cleanup_fail;

    { // block to allow goto-over-init
        cudaFree(d_work); cudaFree(d_S); cudaFree(d_U); cudaFree(d_Vt); cudaFree(d_info);
        return true;
    }

cleanup_fail:
    cudaFree(d_work); cudaFree(d_S); cudaFree(d_U); cudaFree(d_Vt); cudaFree(d_info);
    return false;
}

void DKVCompressorThread::process_job(const CompressJob& job, cusolverDnHandle_t handle) {
    // 1. Check if session is still alive before expensive SVD
    if (!alive_cb_(job.session_id)) {
        // Session disconnected. Block -> Invalid -> will be Freed by cleanup.
        state_table_.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 2. Verify block is still in Compressing state (CAS check)
    BlockState current = state_table_.get(job.block_id);
    if (current != BlockState::Compressing) {
        // Something invalidated the block between submission and execution. Skip.
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 3. Execute Truncated SVD via cuSOLVER
    // feat_dim = heads * head_dim (full flattened feature dimension)
    int rank     = static_cast<int>(job.target_slab); // 8, 16, or 32
    int m        = job.block_size;
    int feat_dim = job.heads * job.head_dim;

    // --- K matrix SVD ---
    // Input:  dense_k_ptr [m, feat_dim] on GPU (row-major float32)
    // Output: slab_u_ptr  [m, rank]     principal projections for K tokens
    //         slab_v_ptr  [rank, feat_dim] principal components for K features
    bool k_ok = run_svd_and_write_slab(
        handle,
        job.dense_k_ptr,
        job.slab_u_ptr,
        job.slab_v_ptr,
        m, feat_dim, rank
    );

    // --- V matrix SVD ---
    // V's slab is offset by K's slab size. slab_u has shape [2*m, rank] — K first, V second.
    // slab_v has shape [2*rank, feat_dim] — K first, V second.
    bool v_ok = run_svd_and_write_slab(
        handle,
        job.dense_v_ptr,
        job.slab_u_ptr + (size_t)m * rank,          // V's U segment follows K's
        job.slab_v_ptr + (size_t)rank * feat_dim,    // V's Vt segment follows K's
        m, feat_dim, rank
    );

    if (!k_ok || !v_ok) {
        // SVD failed (CUDA OOM or cuSOLVER error) — leave block as Compressing.
        // The manager's GC loop will eventually detect the stale Compressing state
        // and revert the block to DenseResident for retry.
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 4. Check again if session disconnected mid-SVD
    if (!alive_cb_(job.session_id)) {
        state_table_.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 5. Transition block: Compressing -> CompressedResident (atomic CAS)
    bool ok = state_table_.transition(
        job.block_id,
        BlockState::Compressing,
        BlockState::CompressedResident
    );

    if (!ok) {
        // CAS failed — block was externally invalidated between SVD and commit
        state_table_.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace dkv
