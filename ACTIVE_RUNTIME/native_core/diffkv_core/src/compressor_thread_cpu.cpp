// diffkv_core/src/compressor_thread_cpu.cpp
// CPU SVD compressor for Apple Silicon (Mac).
//
// Uses torch::linalg::svd (ATen C++ API) with cpu tensors.
// On Apple Silicon this dispatches to Accelerate's AMX-backed LAPACK.
// Zero Metal/MPS API calls — safe from any thread.
//
// Why ATen SVD instead of raw LAPACKE?
//   macOS Accelerate exposes Fortran-ABI sgesdd_ (column-major), not the
//   LAPACKE row-major wrapper. Using ATen's linalg::svd lets us stay
//   row-major, avoid Fortran ABI complexity, and still get AMX acceleration
//   because PyTorch's MKL/LAPACK backend on Mac uses Accelerate under the hood
//   for CPU tensors.
//
//   Critically: torch::linalg::svd on a CPU tensor never touches Metal/MPS.
//   The ATen dispatcher selects the CPU kernel path because device == kCPU.

#include "compressor_cpu.hpp"
#include <torch/extension.h>
#include <ATen/ops/_linalg_svd.h>  // at::_linalg_svd — available without torch::linalg namespace
#include <chrono>
#include <cstring>
#include <algorithm>

namespace diffkv {

// ── Worker lifecycle ──────────────────────────────────────────────────────────

void DiffKVCompressorThreadCPU::start() {
    if (running_.load(std::memory_order_acquire)) return;
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&DiffKVCompressorThreadCPU::worker_loop, this);
}

void DiffKVCompressorThreadCPU::stop() {
    running_.store(false, std::memory_order_release);
    if (worker_.joinable())
        worker_.join();
}

bool DiffKVCompressorThreadCPU::submit(const CompressJobCPU& job) {
    if (!queue_.push(job)) {
        queue_overflows_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

// ── Worker loop ───────────────────────────────────────────────────────────────

void DiffKVCompressorThreadCPU::worker_loop() {
    // This thread only uses ATen CPU ops and unified memory writes.
    // No CUDA context, no Metal command encoder, no MPS interaction.
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

// ── Core SVD processing using ATen linalg::svd ────────────────────────────────

void DiffKVCompressorThreadCPU::process_job(const CompressJobCPU& job) {
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

    // 3. Build ATen CPU tensor wrapping the dense_k_ptr (zero-copy view)
    //    dense_k_ptr: float32, [S, F], already in unified memory
    auto options_cpu_f32 = torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(torch::kCPU);

    // Use from_blob — no data copy, just a view
    auto K_mat = torch::from_blob(
        job.dense_k_ptr,
        {S, F},
        options_cpu_f32
    );

    // 4. at::_linalg_svd on CPU → dispatches to Accelerate on Apple Silicon.
    //    Returns (U, S, Vh) where U [S, k], S [k], Vh [k, F] for full_matrices=false.
    auto [U_full, S_vals, Vh_full] = at::_linalg_svd(K_mat, /*full_matrices=*/false, /*compute_uv=*/true);

    // 5. Truncate to rank R
    auto U_trunc  = U_full.slice(1, 0, R).contiguous();   // [S, R]
    auto S_trunc  = S_vals.slice(0, 0, R);                // [R]
    auto Vh_trunc = Vh_full.slice(0, 0, R).contiguous();  // [R, F]

    // 6. Absorb singular values into U: U_scaled = U * S (for quantization later)
    auto U_scaled = U_trunc * S_trunc.unsqueeze(0);       // [S, R]

    // 7. Write U_scaled → out_u_ptr [S, R] float32
    auto U_cpu = U_scaled.to(torch::kFloat32).contiguous();
    std::memcpy(job.out_u_ptr, U_cpu.data_ptr<float>(), S * R * sizeof(float));

    // 8. Write VK (Vh_trunc) → out_vk_ptr [R, F] float32
    auto VK_cpu = Vh_trunc.to(torch::kFloat32);
    std::memcpy(job.out_vk_ptr, VK_cpu.data_ptr<float>(), R * F * sizeof(float));

    // 9. Compute scale factor: max abs of U_scaled (for INT8 quantization headroom)
    float max_abs = U_cpu.abs().max().item<float>();
    *job.out_scale = (max_abs > 0.0f) ? max_abs : 1.0f;

    // 10. Optional: SVD for V (separate right-singular vectors for VV)
    if (job.dense_v_ptr != nullptr && job.out_vv_ptr != nullptr) {
        auto V_mat = torch::from_blob(
            job.dense_v_ptr,
            {S, F},
            options_cpu_f32
        );

        auto [UV, SV, VhV] = at::_linalg_svd(V_mat, /*full_matrices=*/false, /*compute_uv=*/true);
        auto VhV_trunc = VhV.slice(0, 0, R).contiguous();  // [R, F]
        auto VV_cpu    = VhV_trunc.to(torch::kFloat32);
        std::memcpy(job.out_vv_ptr, VV_cpu.data_ptr<float>(), R * F * sizeof(float));
    }

    // 11. Check session alive after expensive SVD
    if (!alive_cb_(job.session_id)) {
        state_table_.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 12. Atomic state transition: Compressing → CompressedResident
    //     acq_rel ordering: CPU writes are visible to GPU after this completes.
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

} // namespace diffkv
