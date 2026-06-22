#pragma once

#include "runtime/native_block_pool.hpp"
#include "ggml.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>
#include <vector>
#include <unordered_set>

namespace diffkv {

struct CompressJob {
    std::string session_id;
    int block_id;
    int block_size;  // S (total tokens in block, typically 64)
    int feat_dim;    // F
    int rank;        // R
    int pool_rank = 0; // The stride/capacity of the pool for this slot
    int pool_block_size = 64; // S_max of the pool slot
    int head_dim;    // D
    int anchor_idx;  // Sequence position of block start
    
    // Host-accessible unified pointers (float32 inputs)
    const float* raw_k_ptr;
    const float* raw_v_ptr;
    
    // Token sequence and stop-word details for landmark scoring
    const int32_t* token_ids = nullptr;
    const std::unordered_set<int32_t>* stop_token_ids = nullptr;
    
    // Outputs in block pool
    int8_t* out_u_ptr;         // [S_max, rank] destination in pool U
    ggml_fp16_t* out_u_scale;   // [1] destination in pool U_scale
    ggml_fp16_t* out_u_row_scale = nullptr; // [S_max] per-token int8 scale in pool U_row_scale
    ggml_fp16_t* out_vk_ptr;    // [rank, kv_heads * head_dim] destination in pool VK
    ggml_fp16_t* out_vv_ptr;    // [rank, kv_heads * head_dim] destination in pool VV
    ggml_fp16_t* out_scale;     // [1] destination in pool scales
    ggml_fp16_t* out_anchor_k;  // [kv_heads * head_dim] destination in pool anchors_K
    ggml_fp16_t* out_anchor_v;  // [kv_heads * head_dim] destination in pool anchors_V
    int32_t* out_seq_len;
    int32_t* out_anchor_position; // destination in pool anchor_positions
    // F9 sparse-residual output destinations (pool host buffers; nullptr = skip).
    int32_t* out_res_K_pos = nullptr;
    int32_t* out_res_V_pos = nullptr;
    ggml_fp16_t* out_res_K_val = nullptr;
    ggml_fp16_t* out_res_V_val = nullptr;
    int max_residual = 8;
    DiffKVBlockStateTable* state_table = nullptr;
};

class AsyncCompressor {
public:
    AsyncCompressor(DiffKVBlockStateTable& state_table, std::function<bool(const std::string&)> alive_cb = nullptr);
    ~AsyncCompressor();

    bool start();
    void stop();
    bool submit(const CompressJob& job);
    void compress_sync(const CompressJob& job);

    uint64_t get_jobs_processed() const { return jobs_processed_.load(std::memory_order_relaxed); }
    uint64_t get_jobs_dropped() const { return jobs_dropped_.load(std::memory_order_relaxed); }
    uint64_t get_queue_overflows() const { return queue_overflows_.load(std::memory_order_relaxed); }
    uint64_t get_jobs_submitted() const { return jobs_submitted_.load(std::memory_order_relaxed); }
    // Block until every submitted job has been fully processed (processed+dropped == submitted).
    // Deterministic drain (no snapshot/deadline race): the caller must have finished submitting.
    void wait_until_idle(uint64_t timeout_ms = 120000);

private:
    void worker_loop();
    void process_job(const CompressJob& job);

    DiffKVBlockStateTable& state_table_;
    std::function<bool(const std::string&)> alive_cb_;

    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;

    // Thread-safe queue
    std::queue<CompressJob> queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    // RECONSTRUCTION FIX (F21): match ACTIVE_RUNTIME async_compressor.py max_queue=32768
    // (was 16384). Overflow is already handled gracefully (submit() returns false → the
    // ingest caller reverts the block to DenseResident), so this only reduces drop frequency
    // under burst; cost is negligible (~100 B/job).
    static constexpr size_t MAX_QUEUE_SIZE = 32768;

    std::atomic<uint64_t> jobs_processed_{0};
    std::atomic<uint64_t> jobs_dropped_{0};
    std::atomic<uint64_t> queue_overflows_{0};
    std::atomic<uint64_t> jobs_submitted_{0};
};

} // namespace diffkv
