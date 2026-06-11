// diffkv_core/include/compressor_thread.hpp
// Native OS thread for asynchronous KV block compression via cuSOLVER SVD.
// Completely outside Python GIL.

#pragma once
#include <thread>
#include <atomic>
#include <functional>
#include <cstdint>
#include "native_core/diffkv_core/include/spsc_ring_buffer.hpp"
#include "native_core/diffkv_core/include/block_state.hpp"

namespace diffkv {

// Slab tiers — must match Python-side SlabAllocator constants
enum class SlabTier : uint8_t { Rank8 = 8, Rank16 = 16, Rank32 = 32 };

struct CompressJob {
    uint32_t  block_id;
    uint64_t  session_id;
    float*    dense_k_ptr;  // Raw GPU pointer to dense K block [block_size, heads, head_dim]
    float*    dense_v_ptr;  // Raw GPU pointer to dense V block
    float*    slab_u_ptr;   // Destination: slab U matrix pointer
    float*    slab_v_ptr;   // Destination: slab V matrix pointer
    int       block_size;
    int       heads;
    int       head_dim;
    SlabTier  target_slab;
};

// Signature for the "is session alive?" callback — implemented in Python/C++ bridge
using SessionAliveCallback = std::function<bool(uint64_t session_id)>;

class DiffKVCompressorThread {
public:
    static constexpr size_t QUEUE_CAPACITY = 4096; // Power of 2

    explicit DiffKVCompressorThread(
        DiffKVBlockStateTable& state_table,
        SessionAliveCallback   alive_cb
    ) : state_table_(state_table),
        alive_cb_(alive_cb),
        running_(false),
        jobs_processed_(0),
        jobs_dropped_(0),
        queue_overflows_(0) {}

    ~DiffKVCompressorThread() { stop(); }

    void start();
    void stop();

    // Non-blocking job submission. Returns false on queue overflow.
    // In that case, the block remains DenseResident — no decode stall.
    bool submit(const CompressJob& job);

    // Metrics (thread-safe reads)
    uint64_t jobs_processed()  const { return jobs_processed_.load(std::memory_order_relaxed); }
    uint64_t jobs_dropped()    const { return jobs_dropped_.load(std::memory_order_relaxed); }
    uint64_t queue_overflows() const { return queue_overflows_.load(std::memory_order_relaxed); }
    size_t   queue_depth()     const { return queue_.size(); }

private:
    void worker_loop();
    void process_job(const CompressJob& job);

    DiffKVBlockStateTable&                    state_table_;
    SessionAliveCallback                      alive_cb_;
    SPSCRingBuffer<CompressJob, QUEUE_CAPACITY> queue_;
    std::thread                               worker_;
    std::atomic<bool>                         running_;
    std::atomic<uint64_t>                     jobs_processed_;
    std::atomic<uint64_t>                     jobs_dropped_;
    std::atomic<uint64_t>                     queue_overflows_;
};

} // namespace diffkv
