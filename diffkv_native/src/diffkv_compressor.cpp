#include "diffkv_compressor.hpp"
#include <Accelerate/Accelerate.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

namespace diffkv {

// Helper function to run SVD using macOS Accelerate LAPACK (sgesdd_)
// A_input is row-major of shape [S, F].
// Since LAPACK is column-major, it views A_input as B = A_input^T of shape [F, S] (column-major).
// LAPACK sgesdd computes B = U_b * S_b * V_b^T.
// VT_out (row-major [R, F]) corresponds to U_b^T (first R rows of column-major [F, S] or similar).
// U_out (row-major [S, R]) corresponds to V_b (first R columns of [S, S]).
static bool run_accelerate_svd(
    const float* A_input,
    int S, int F, int R,
    float* U_out,
    float* S_out,
    float* VT_out
) {
    if (S == 1) {
        // Analytical SVD for a single row vector
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
    int ldvt = S; // Since jobz = "S", LAPACK computes min(M, N) = S singular vectors.
                  // V_b^T (VT) is S x S, so ldvt is S.

    // Copy input matrix because LAPACK sgesdd overwrites its input
    std::vector<float> a_copy(S * F);
    std::memcpy(a_copy.data(), A_input, S * F * sizeof(float));

    int min_dim = std::min(S, F);
    std::vector<float> s_temp(min_dim);
    std::vector<float> u_temp(m * min_dim); // shape [F, S] column-major
    std::vector<float> vt_temp(min_dim * n); // shape [S, S] column-major

    // 1. Try divide-and-conquer SVD (sgesdd_)
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

    // 2. Fallback to standard SVD (sgesvd_) if sgesdd_ fails
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

    // 1. Copy top R singular values
    std::memcpy(S_out, s_temp.data(), R * sizeof(float));

    // 2. Extract U_out (shape [S, R] row-major) from vt_temp (shape [S, S] column-major, where U_a = V_b).
    // vt_temp (V_b^T) is stored column-major. So columns of vt_temp correspond to rows of V_b.
    // Index (r, s) in vt_temp is at s * ldvt + r = s * S + r.
    // This is identical to U_out[s * R + r] in memory!
    for (int s = 0; s < S; ++s) {
        for (int r = 0; r < R; ++r) {
            U_out[s * R + r] = vt_temp[s * S + r];
        }
    }

    // 3. Extract VT_out (shape [R, F] row-major) from u_temp (shape [F, min_dim] column-major truncated to [F, R] column-major).
    // u_temp (U_b) has shape [F, S] column-major.
    // The first R columns are stored contiguously as F * R floats.
    // Index (f, r) in column-major is at r * F + f.
    // This is identical to VT_out[r * F + f] in memory!
    std::memcpy(VT_out, u_temp.data(), R * F * sizeof(float));

    return true;
}

DiffKVCompressor::DiffKVCompressor(DiffKVBlockStateTable& state_table, std::function<bool(int)> alive_cb)
    : state_table_(state_table), alive_cb_(alive_cb) {}

DiffKVCompressor::~DiffKVCompressor() {
    stop();
}

bool DiffKVCompressor::start() {
    if (running_.load(std::memory_order_acquire)) return true;
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&DiffKVCompressor::worker_loop, this);
    return true;
}

void DiffKVCompressor::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);
    queue_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool DiffKVCompressor::submit(const CompressJob& job) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (queue_.size() >= MAX_QUEUE_SIZE) {
        queue_overflows_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    queue_.push(job);
    queue_cv_.notify_one();
    return true;
}

void DiffKVCompressor::compress_sync(const CompressJob& job) {
    process_job(job);
}

void DiffKVCompressor::worker_loop() {
    while (running_.load(std::memory_order_acquire)) {
        CompressJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !queue_.empty() || !running_.load(std::memory_order_acquire);
            });
            if (!running_.load(std::memory_order_acquire) && queue_.empty()) {
                break;
            }
            job = queue_.front();
            queue_.pop();
        }
        process_job(job);
        jobs_processed_.fetch_add(1, std::memory_order_relaxed);
    }
}

void DiffKVCompressor::process_job(const CompressJob& job) {
    DiffKVBlockStateTable& active_table = job.state_table ? *job.state_table : state_table_;

    // 1. Check if session is alive
    if (alive_cb_ && !alive_cb_(job.session_id)) {
        active_table.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 2. Check if block is in Compressing state
    BlockState current = active_table.get(job.block_id);
    if (current != BlockState::Compressing) {
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    int S = job.block_size;
    int F = job.feat_dim;
    int R = job.rank;
    int joint_F = 2 * F;

    int svd_dim = std::min(S, R);

    // 3. Construct the joint matrix [S, 2 * F] (no padding with zero rows!)
    std::vector<float> K_V_joint(S * joint_F);
    for (int s = 0; s < S; ++s) {
        std::memcpy(K_V_joint.data() + s * joint_F, job.dense_k_ptr + s * F, F * sizeof(float));
        std::memcpy(K_V_joint.data() + s * joint_F + F, job.dense_v_ptr + s * F, F * sizeof(float));
    }

    // Compute input max absolute value (scale)
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

    // Normalize input matrix
    for (float& val : K_V_joint) {
        val /= scale;
    }

    // Allocate temporary buffers for SVD outputs of the joint matrix
    std::vector<float> U_joint(S * svd_dim);
    std::vector<float> S_joint(svd_dim);
    std::vector<float> VT_joint(svd_dim * joint_F);

    // 4. Perform SVD for the joint matrix of actual size S
    bool ok = run_accelerate_svd(K_V_joint.data(), S, joint_F, svd_dim, U_joint.data(), S_joint.data(), VT_joint.data());

    if (!ok) {
        std::cerr << "[Compressor] Error: SVD failed for block " << job.block_id << std::endl;
        active_table.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 5. Absorb singular values into U_joint: U_scaled = U_joint * S_joint
    // We pad columns of U_scaled to size R
    std::vector<float> U_scaled(S * R, 0.0f);
    float max_abs = 0.0f;
    for (int s = 0; s < S; ++s) {
        for (int r = 0; r < svd_dim; ++r) {
            float val = U_joint[s * svd_dim + r] * S_joint[r];
            U_scaled[s * R + r] = val;
            if (std::abs(val) > max_abs) {
                max_abs = std::abs(val);
            }
        }
    }

    // 6. Quantize U_scaled to int8_t and write to job.out_u_ptr
    float scale_u = std::max(max_abs / 127.0f, 1e-5f);
    int S_max = 64; // pad to S_max
    std::memset(job.out_u_ptr, 0, S_max * R * sizeof(int8_t));
    for (int s = 0; s < S; ++s) {
        for (int r = 0; r < R; ++r) {
            float val = U_scaled[s * R + r] / scale_u;
            int8_t val_i8 = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(val))));
            job.out_u_ptr[s * R + r] = val_i8;
        }
    }

    // 7. Write scale_u to out_u_scale (FP16)
    *job.out_u_scale = ggml_fp32_to_fp16(scale_u);

    // 8. Split VT_joint into VK and VV and write to out_vk_ptr / out_vv_ptr in FP16
    // If svd_dim < R, the remaining rows of out_vk_ptr and out_vv_ptr must be padded with 0.0f
    std::memset(job.out_vk_ptr, 0, R * F * sizeof(ggml_fp16_t));
    std::memset(job.out_vv_ptr, 0, R * F * sizeof(ggml_fp16_t));
    for (int r = 0; r < svd_dim; ++r) {
        for (int f = 0; f < F; ++f) {
            job.out_vk_ptr[r * F + f] = ggml_fp32_to_fp16(VT_joint[r * joint_F + f]);
            job.out_vv_ptr[r * F + f] = ggml_fp32_to_fp16(VT_joint[r * joint_F + F + f]);
        }
    }

    // 9. Write scale to out_scale (FP16)
    *job.out_scale = ggml_fp32_to_fp16(scale);

    // 10. Check session alive again
    if (alive_cb_ && !alive_cb_(job.session_id)) {
        active_table.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 11. Atomic transition: Compressing -> CompressedResident
    bool trans_ok = active_table.transition(
        job.block_id,
        BlockState::Compressing,
        BlockState::CompressedResident
    );

    if (!trans_ok) {
        active_table.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace diffkv
