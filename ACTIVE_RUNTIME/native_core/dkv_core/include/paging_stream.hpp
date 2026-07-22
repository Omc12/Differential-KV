// dkv_core/include/paging_stream.hpp
// CUDA-stream-based async paging subsystem.
// All H2D/D2H transfers run on a dedicated stream with event synchronization.
// Zero Python involvement — no tensor.to(), no GIL.

#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include "block_state.hpp"

namespace dkv {

struct PagingJob {
    uint32_t  block_id;
    uint64_t  session_id;
    void*     gpu_ptr;       // Source or destination GPU slab pointer
    void*     cpu_ptr;       // Source or destination pinned CPU pointer
    size_t    byte_size;
    bool      is_reload;     // true = H2D (reload), false = D2H (eviction)
};

class DKVPagingStream {
public:
    DKVPagingStream(DKVBlockStateTable& state_table, int cuda_device = 0);
    ~DKVPagingStream();

    // Issue async D2H (eviction). Non-blocking. Returns false if block not in CompressedResident.
    bool issue_eviction(const PagingJob& job);

    // Issue async H2D (reload). Non-blocking. Returns false if block not in CPUResident.
    bool issue_reload(const PagingJob& job);

    // Wait for a specific block's pending transfer to complete (graph replay safety).
    // Inserts cudaStreamWaitEvent into compute_stream — does NOT block the CPU.
    void sync_to_compute_stream(uint32_t block_id, cudaStream_t compute_stream);

    // Poll for completed transfers, update state machine.
    // Called by the engine scheduler between batch steps — NOT in the hot decode path.
    void poll_completions();

    uint64_t evictions_completed()  const { return evictions_completed_; }
    uint64_t reloads_completed()    const { return reloads_completed_; }
    uint64_t cancelled_transfers()  const { return cancelled_transfers_; }

private:
    struct PendingTransfer {
        PagingJob    job;
        cudaEvent_t  completion_event;
    };

    DKVBlockStateTable&               state_table_;
    cudaStream_t                         paging_stream_;
    std::unordered_map<uint32_t, PendingTransfer> pending_;
    std::mutex                           pending_mutex_; // Only protects the map, not the stream
    uint64_t                             evictions_completed_{0};
    uint64_t                             reloads_completed_{0};
    uint64_t                             cancelled_transfers_{0};
};

} // namespace dkv
