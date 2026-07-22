// dkv_core/src/paging_stream.cu
// CUDA implementation of async H2D/D2H paging transfers.

#include "native_core/dkv_core/include/paging_stream.hpp"
#include <stdexcept>
#include <string>

namespace dkv {

#if defined(GGML_USE_CUDA) || defined(USE_CUDA)

DKVPagingStream::DKVPagingStream(DKVBlockStateTable& state_table, int cuda_device)
    : state_table_(state_table) {
    cudaSetDevice(cuda_device);
    cudaError_t err = cudaStreamCreateWithFlags(&paging_stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(err));
}

DKVPagingStream::~DKVPagingStream() {
    cudaStreamSynchronize(paging_stream_);
    cudaStreamDestroy(paging_stream_);
    std::lock_guard<std::mutex> lock(pending_mutex_);
    for (auto& [id, pt] : pending_)
        cudaEventDestroy(pt.completion_event);
}

bool DKVPagingStream::issue_eviction(const PagingJob& job) {
    if (!state_table_.transition(job.block_id, BlockState::CompressedResident, BlockState::PagingOut))
        return false;

    cudaEvent_t evt;
    cudaEventCreate(&evt);

    cudaMemcpyAsync(job.cpu_ptr, job.gpu_ptr, job.byte_size,
                    cudaMemcpyDeviceToHost, paging_stream_);
    cudaEventRecord(evt, paging_stream_);

    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_[job.block_id] = { job, evt };
    return true;
}

bool DKVPagingStream::issue_reload(const PagingJob& job) {
    if (!state_table_.transition(job.block_id, BlockState::CPUResident, BlockState::Reloading))
        return false;

    cudaEvent_t evt;
    cudaEventCreate(&evt);

    cudaMemcpyAsync(job.gpu_ptr, job.cpu_ptr, job.byte_size,
                    cudaMemcpyHostToDevice, paging_stream_);
    cudaEventRecord(evt, paging_stream_);

    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_[job.block_id] = { job, evt };
    return true;
}

void DKVPagingStream::sync_to_compute_stream(uint32_t block_id, cudaStream_t compute_stream) {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_.find(block_id);
    if (it == pending_.end()) return;
    cudaStreamWaitEvent(compute_stream, it->second.completion_event, 0);
}

void DKVPagingStream::poll_completions() {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    for (auto it = pending_.begin(); it != pending_.end(); ) {
        const auto& [block_id, pt] = *it;
        cudaError_t status = cudaEventQuery(pt.completion_event);

        if (status == cudaSuccess) {
            BlockState next = pt.job.is_reload
                ? BlockState::CompressedResident
                : BlockState::CPUResident;
            BlockState expected = pt.job.is_reload
                ? BlockState::Reloading
                : BlockState::PagingOut;

            bool ok = state_table_.transition(block_id, expected, next);
            if (!ok) state_table_.force_invalidate(block_id);

            if (pt.job.is_reload) reloads_completed_++;
            else evictions_completed_++;

            cudaEventDestroy(pt.completion_event);
            it = pending_.erase(it);
        } else if (status == cudaErrorNotReady) {
            ++it;
        } else {
            state_table_.force_invalidate(block_id);
            cancelled_transfers_++;
            cudaEventDestroy(pt.completion_event);
            it = pending_.erase(it);
        }
    }
}

#endif

} // namespace dkv
