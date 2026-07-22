// dkv_core/src/paging_stream.cpp
// Fallback implementation of paging stream for non-CUDA systems.

#include "native_core/dkv_core/include/paging_stream.hpp"
#include <cstring>

namespace dkv {

#if !defined(GGML_USE_CUDA) && !defined(USE_CUDA)

DKVPagingStream::DKVPagingStream(DKVBlockStateTable& state_table, int cuda_device)
    : state_table_(state_table), paging_stream_(nullptr) {
    (void)cuda_device;
}

DKVPagingStream::~DKVPagingStream() {
    // No async resources to release
}

bool DKVPagingStream::issue_eviction(const PagingJob& job) {
    if (!state_table_.transition(job.block_id, BlockState::CompressedResident, BlockState::PagingOut))
        return false;

    // Sync copy fallback: since there's no background CUDA stream, we perform standard sync copy.
    // On macOS CPU/Metal, Unified Memory allows fast host access, or we memcpy.
    std::memcpy(job.cpu_ptr, job.gpu_ptr, job.byte_size);

    // Complete the transition immediately
    state_table_.transition(job.block_id, BlockState::PagingOut, BlockState::CPUResident);
    evictions_completed_++;
    return true;
}

bool DKVPagingStream::issue_reload(const PagingJob& job) {
    if (!state_table_.transition(job.block_id, BlockState::CPUResident, BlockState::Reloading))
        return false;

    std::memcpy(job.gpu_ptr, job.cpu_ptr, job.byte_size);

    state_table_.transition(job.block_id, BlockState::Reloading, BlockState::CompressedResident);
    reloads_completed_++;
    return true;
}

void DKVPagingStream::sync_to_compute_stream(uint32_t block_id, void* compute_stream) {
    (void)block_id;
    (void)compute_stream;
    // No-op on unified memory/sync copies
}

void DKVPagingStream::poll_completions() {
    // No-op because all copies are done synchronously on the issue call
}

#endif

} // namespace dkv
