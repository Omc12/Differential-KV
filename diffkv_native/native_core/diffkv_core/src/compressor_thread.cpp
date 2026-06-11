#include "native_core/diffkv_core/include/compressor_thread.hpp"

namespace diffkv {

void DiffKVCompressorThread::start() {
    // No-op on macOS GPU compressor
}

void DiffKVCompressorThread::stop() {
    // No-op on macOS GPU compressor
}

bool DiffKVCompressorThread::submit(const CompressJob& job) {
    (void)job;
    // GPU-side compressor is a stub on macOS, return false
    return false;
}

void DiffKVCompressorThread::worker_loop() {
    // No-op on macOS GPU compressor
}

void DiffKVCompressorThread::process_job(const CompressJob& job) {
    (void)job;
    // No-op on macOS GPU compressor
}

} // namespace diffkv
