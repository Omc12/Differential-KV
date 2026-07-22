// dkv_core/include/paging_stream.hpp
// Async paging subsystem supporting both CUDA and CPU/Metal backends.

#pragma once
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include "native_core/dkv_core/include/block_state.hpp"

#if defined(GGML_USE_CUDA) || defined(USE_CUDA)
#include <cuda_runtime.h>
typedef cudaStream_t paging_stream_t;
typedef cudaEvent_t paging_event_t;
#else
typedef void* paging_stream_t;
typedef void* paging_event_t;
#endif

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
#if defined(GGML_USE_CUDA) || defined(USE_CUDA)
    void sync_to_compute_stream(uint32_t block_id, cudaStream_t compute_stream);
#else
    void sync_to_compute_stream(uint32_t block_id, void* compute_stream);
#endif

    // Poll for completed transfers, update state machine.
    // Called by the engine scheduler between batch steps — NOT in the hot decode path.
    void poll_completions();

    uint64_t evictions_completed()  const { return evictions_completed_; }
    uint64_t reloads_completed()    const { return reloads_completed_; }
    uint64_t cancelled_transfers()  const { return cancelled_transfers_; }

private:
    struct PendingTransfer {
        PagingJob       job;
        paging_event_t  completion_event;
    };

    DKVBlockStateTable&               state_table_;
    paging_stream_t                      paging_stream_;
    std::unordered_map<uint32_t, PendingTransfer> pending_;
    std::mutex                           pending_mutex_; // Only protects the map, not the stream
    uint64_t                             evictions_completed_{0};
    uint64_t                             reloads_completed_{0};
    uint64_t                             cancelled_transfers_{0};
};

} // namespace dkv
