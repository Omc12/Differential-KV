#include "native_core/dkv_core/include/compressor_cpu.hpp"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cmath>
#include <vector>
#include <algorithm>

namespace dkv {
// Simple, self-contained Jacobi SVD fallback for non-Apple CPU target
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
} // namespace dkv
#endif
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace dkv {

void DKVCompressorThreadCPU::start() {
    if (running_.load(std::memory_order_acquire)) return;
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&DKVCompressorThreadCPU::worker_loop, this);
}

void DKVCompressorThreadCPU::stop() {
    running_.store(false, std::memory_order_release);
    if (worker_.joinable())
        worker_.join();
    queue_.clear();
}

bool DKVCompressorThreadCPU::submit(const CompressJobCPU& job) {
    if (!queue_.push(job)) {
        queue_overflows_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void DKVCompressorThreadCPU::worker_loop() {
    while (running_.load(std::memory_order_acquire)) {
        auto job_opt = queue_.pop();
        if (!job_opt.has_value()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }
        process_job(*job_opt);
        jobs_processed_.fetch_add(1, std::memory_order_relaxed);
    }
}

void DKVCompressorThreadCPU::process_job(const CompressJobCPU& job) {
    // 1. Check session alive
    if (!alive_cb_(job.session_id)) {
        state_table_.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 2. Verify block state
    BlockState current = state_table_.get(job.block_id);
    if (current != BlockState::Compressing) {
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const int S = job.block_size;
    const int F = job.feat_dim;    // kv_heads * head_dim
    const int R = std::min(job.rank, std::min(S, F));
    const int K_min = std::min(S, F);

    if (R > 0 && K_min > 0) {
#ifdef __APPLE__
        // 3. Transpose dense_k_ptr [S, F] (row-major) to [F, S] (column-major)
        std::vector<float> A_transposed(S * F);
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < F; ++j) {
                A_transposed[j * S + i] = job.dense_k_ptr[i * F + j];
            }
        }

        // 4. Call LAPACK sgesvd_ in column-major
        char jobu = 'S';
        char jobvt = 'S';
        int m = S;
        int n = F;
        int lda = S;
        int ldu = S;
        int ldvt = K_min;

        std::vector<float> s_vals(K_min);
        std::vector<float> u_col_major(S * K_min);
        std::vector<float> vt_col_major(K_min * F);
        int info = 0;
        int lwork = -1;
        float work_query = 0.0f;

        sgesvd_(&jobu, &jobvt, &m, &n, A_transposed.data(), &lda, s_vals.data(),
                u_col_major.data(), &ldu, vt_col_major.data(), &ldvt,
                &work_query, &lwork, &info);

        lwork = static_cast<int>(work_query);
        std::vector<float> work(lwork);

        sgesvd_(&jobu, &jobvt, &m, &n, A_transposed.data(), &lda, s_vals.data(),
                u_col_major.data(), &ldu, vt_col_major.data(), &ldvt,
                work.data(), &lwork, &info);

        if (info == 0) {
            // 5. Scale U and copy to out_u_ptr [S, R] (row-major)
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < R; ++j) {
                    job.out_u_ptr[i * R + j] = u_col_major[j * S + i] * s_vals[j];
                }
            }

            // 6. Copy Vt to out_vk_ptr [R, F] (row-major)
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < F; ++j) {
                    job.out_vk_ptr[i * F + j] = vt_col_major[j * K_min + i];
                }
            }

            // 7. Compute scale factor (max abs of U_scaled)
            float max_abs = 0.0f;
            for (int i = 0; i < S * R; ++i) {
                float val = std::abs(job.out_u_ptr[i]);
                if (val > max_abs) max_abs = val;
            }
            *job.out_scale = (max_abs > 0.0f) ? max_abs : 1.0f;
        } else {
            // Fail safe fallback
            std::memset(job.out_u_ptr, 0, S * R * sizeof(float));
            std::memset(job.out_vk_ptr, 0, R * F * sizeof(float));
            *job.out_scale = 1.0f;
        }

        // 8. Optional: V matrix SVD
        if (job.dense_v_ptr != nullptr && job.out_vv_ptr != nullptr) {
            std::vector<float> V_transposed(S * F);
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < F; ++j) {
                    V_transposed[j * S + i] = job.dense_v_ptr[i * F + j];
                }
            }

            std::vector<float> sv_vals(K_min);
            std::vector<float> uv_col_major(S * K_min);
            std::vector<float> vtv_col_major(K_min * F);
            info = 0;
            lwork = -1;
            work_query = 0.0f;

            sgesvd_(&jobu, &jobvt, &m, &n, V_transposed.data(), &lda, sv_vals.data(),
                    uv_col_major.data(), &ldu, vtv_col_major.data(), &ldvt,
                    &work_query, &lwork, &info);

            lwork = static_cast<int>(work_query);
            work.resize(lwork);

            sgesvd_(&jobu, &jobvt, &m, &n, V_transposed.data(), &lda, sv_vals.data(),
                    uv_col_major.data(), &ldu, vtv_col_major.data(), &ldvt,
                    work.data(), &lwork, &info);

            if (info == 0) {
                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < F; ++j) {
                        job.out_vv_ptr[i * F + j] = vtv_col_major[j * K_min + i];
                    }
                }
            } else {
                std::memset(job.out_vv_ptr, 0, R * F * sizeof(float));
            }
        }
#else
        // Non-Apple fallback: use Jacobi SVD directly on row-major inputs
        std::vector<float> s_vals(K_min, 0.0f);
        bool svd_k_ok = run_cpu_jacobi_svd(job.dense_k_ptr, S, F, R, job.out_u_ptr, s_vals.data(), job.out_vk_ptr);
        if (svd_k_ok) {
            // Compute scale factor (max abs of U_scaled)
            float max_abs = 0.0f;
            for (int i = 0; i < S * R; ++i) {
                // Scale U by singular values (absorb scale)
                int r_idx = i % R;
                job.out_u_ptr[i] *= s_vals[r_idx];
                float val = std::abs(job.out_u_ptr[i]);
                if (val > max_abs) max_abs = val;
            }
            *job.out_scale = (max_abs > 0.0f) ? max_abs : 1.0f;
        } else {
            std::memset(job.out_u_ptr, 0, S * R * sizeof(float));
            std::memset(job.out_vk_ptr, 0, R * F * sizeof(float));
            *job.out_scale = 1.0f;
        }

        if (job.dense_v_ptr != nullptr && job.out_vv_ptr != nullptr) {
            std::vector<float> sv_vals(K_min, 0.0f);
            std::vector<float> u_v_dummy(S * R, 0.0f);
            bool svd_v_ok = run_cpu_jacobi_svd(job.dense_v_ptr, S, F, R, u_v_dummy.data(), sv_vals.data(), job.out_vv_ptr);
            if (!svd_v_ok) {
                std::memset(job.out_vv_ptr, 0, R * F * sizeof(float));
            }
        }
#endif
    } else {
        // Edge case: empty block or rank-0
        *job.out_scale = 1.0f;
    }

    // 9. Check session alive after SVD
    if (!alive_cb_(job.session_id)) {
        state_table_.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 10. Transition block: Compressing -> CompressedResident
    bool ok = state_table_.transition(
        job.block_id,
        BlockState::Compressing,
        BlockState::CompressedResident
    );

    if (!ok) {
        state_table_.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace dkv
