// diffkv_core/include/compressor_cpu.hpp
// CPU-only SVD compressor for Apple Silicon (Mac).
//
// On MPS (Metal Performance Shaders), the background compressor must NOT call
// any MPS/Metal operations from a worker thread, because PyTorch's MPS backend
// uses a single global Metal command queue. Concurrent submissions from the
// Python main thread (attention forward) and a worker thread (SVD) cause command
// encoder validation crashes.
//
// This implementation uses Accelerate's LAPACK (LAPACKE_sgesdd) instead.
// On Apple Silicon, LAPACK is AMX-accelerated and runs on the CPU's matrix
// coprocessor, not the GPU. Because all memory on M-series chips is unified,
// the compressed result is immediately visible to the GPU without any transfer.
//
// Thread safety:
//   The worker thread uses ONLY CPU math and writes to unified memory.
//   The main thread (Metal/MPS) reads from the same unified memory region.
//   This is safe on Apple Silicon because:
//     - CPU writes to unified memory are visible to GPU after the CPU write
//       completes (no explicit DMA transfer needed).
//     - The compressor writes to a separate slot that the GPU reads AFTER
//       DiffKVBlockStateTable confirms CompressedResident state.
//     - The state table uses std::atomic with acq_rel ordering, which acts as
//       the memory barrier between the CPU write and the GPU read.

#pragma once
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include "spsc_ring_buffer.hpp"
#include "block_state.hpp"

namespace diffkv {

struct CompressJobCPU {
    uint32_t  block_id;
    uint64_t  session_id;
    float*    dense_k_ptr;    // CPU float32 pointer to raw K block [S, kv_heads*D]
    float*    dense_v_ptr;    // CPU float32 pointer to raw V block [S, kv_heads*D]
    float*    out_u_ptr;      // Destination: U matrix [S, rank]  (unified memory)
    float*    out_vk_ptr;     // Destination: VK matrix [rank, kv_heads*D]
    float*    out_vv_ptr;     // Destination: VV matrix [rank, kv_heads*D]
    float*    out_scale;      // Destination: scale factor (scalar)
    int       block_size;     // S = number of tokens in block
    int       feat_dim;       // kv_heads * head_dim
    int       rank;           // SVD truncation rank
};

using CPUSessionAliveCallback = std::function<bool(uint64_t session_id)>;

// ── DiffKVCompressorThreadCPU ─────────────────────────────────────────────────
// Drop-in replacement for DiffKVCompressorThread on Mac.
// Uses Accelerate LAPACK (LAPACKE_sgesdd) for truncated SVD.
// Zero Metal/MPS interaction.
class DiffKVCompressorThreadCPU {
public:
    static constexpr size_t QUEUE_CAPACITY = 4096; // Power of 2

    explicit DiffKVCompressorThreadCPU(
        DiffKVBlockStateTable& state_table,
        CPUSessionAliveCallback alive_cb
    ) : state_table_(state_table),
        alive_cb_(alive_cb),
        running_(false),
        jobs_processed_(0),
        jobs_dropped_(0),
        queue_overflows_(0) {}

    ~DiffKVCompressorThreadCPU() { stop(); }

    void start();
    void stop();

    // Non-blocking submission. Returns false on queue overflow.
    bool submit(const CompressJobCPU& job);

    // Metrics (thread-safe atomic reads)
    uint64_t jobs_processed()  const { return jobs_processed_.load(std::memory_order_relaxed); }
    uint64_t jobs_dropped()    const { return jobs_dropped_.load(std::memory_order_relaxed); }
    uint64_t queue_overflows() const { return queue_overflows_.load(std::memory_order_relaxed); }
    size_t   queue_depth()     const { return queue_.size(); }

private:
    void worker_loop();
    void process_job(const CompressJobCPU& job);

    DiffKVBlockStateTable&                         state_table_;
    CPUSessionAliveCallback                        alive_cb_;
    SPSCRingBuffer<CompressJobCPU, QUEUE_CAPACITY> queue_;
    std::thread                                    worker_;
    std::atomic<bool>                              running_;
    std::atomic<uint64_t>                          jobs_processed_;
    std::atomic<uint64_t>                          jobs_dropped_;
    std::atomic<uint64_t>                          queue_overflows_;
};

} // namespace diffkv
