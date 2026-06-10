#pragma once

#include "diffkv_kv_engine.hpp"
#include "ggml.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>

namespace diffkv {

struct CompressJob {
    int session_id;
    int block_id;
    int block_size;  // S
    int feat_dim;    // F
    int rank;        // R
    
    // Host-accessible unified pointers (float32 inputs)
    const float* dense_k_ptr;
    const float* dense_v_ptr;
    
    // Outputs in block pool
    int8_t* out_u_ptr;         // [S_max, rank] destination in pool U
    ggml_fp16_t* out_u_scale;   // [1] destination in pool U_scale
    ggml_fp16_t* out_vk_ptr;    // [rank, kv_heads * head_dim] destination in pool VK
    ggml_fp16_t* out_vv_ptr;    // [rank, kv_heads * head_dim] destination in pool VV
    ggml_fp16_t* out_scale;     // [1] destination in pool scales
    DiffKVBlockStateTable* state_table = nullptr;
};

class DiffKVCompressor {
public:
    DiffKVCompressor(DiffKVBlockStateTable& state_table, std::function<bool(int)> alive_cb = nullptr);
    ~DiffKVCompressor();

    bool start();
    void stop();
    bool submit(const CompressJob& job);
    void compress_sync(const CompressJob& job);

    uint64_t get_jobs_processed() const { return jobs_processed_.load(std::memory_order_relaxed); }
    uint64_t get_jobs_dropped() const { return jobs_dropped_.load(std::memory_order_relaxed); }
    uint64_t get_queue_overflows() const { return queue_overflows_.load(std::memory_order_relaxed); }

private:
    void worker_loop();
    void process_job(const CompressJob& job);

    DiffKVBlockStateTable& state_table_;
    std::function<bool(int)> alive_cb_;

    std::atomic<bool> running_{false};
    std::thread worker_;

    // Thread-safe queue
    std::queue<CompressJob> queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    static constexpr size_t MAX_QUEUE_SIZE = 128;

    std::atomic<uint64_t> jobs_processed_{0};
    std::atomic<uint64_t> jobs_dropped_{0};
    std::atomic<uint64_t> queue_overflows_{0};
};

} // namespace diffkv
