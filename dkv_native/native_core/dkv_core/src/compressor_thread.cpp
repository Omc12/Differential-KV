#include "native_core/dkv_core/include/compressor_thread.hpp"

namespace dkv {

void DKVCompressorThread::start() {
    // No-op on macOS GPU compressor
}

void DKVCompressorThread::stop() {
    // No-op on macOS GPU compressor
}

bool DKVCompressorThread::submit(const CompressJob& job) {
    (void)job;
    // GPU-side compressor is a stub on macOS, return false
    return false;
}

void DKVCompressorThread::worker_loop() {
    // No-op on macOS GPU compressor
}

void DKVCompressorThread::process_job(const CompressJob& job) {
    (void)job;
    // No-op on macOS GPU compressor
}

} // namespace dkv
