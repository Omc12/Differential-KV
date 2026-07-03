#include "native_core/compression/async_compressor.hpp"
#include "native_core/compression/lowrank.hpp"
#include <iostream>
#ifdef __APPLE__
#  include <pthread.h>
#else
#  include <pthread.h>
#  include <sys/resource.h>
#endif

namespace diffkv {

extern std::atomic<int> g_diffkv_dbg_pos;

AsyncCompressor::AsyncCompressor(DiffKVBlockStateTable& state_table, std::function<bool(const std::string&)> alive_cb)
    : state_table_(state_table), alive_cb_(alive_cb) {}

AsyncCompressor::~AsyncCompressor() {
    stop();
}

bool AsyncCompressor::start() {
    if (running_.load(std::memory_order_acquire)) return true;
    running_.store(true, std::memory_order_release);
    // Default to a few workers: long-context prefill submits ~80 SVD jobs and the decode now
    // DRAINS them (wait_until_idle) before generating, so a single background thread serialized
    // them into a ~150s stall. A handful of UTILITY-QoS workers drains ~80 blocks in a few s
    // without starving the decode. Override with DIFFKV_COMPRESSOR_THREADS.
    int num_threads = std::max(2, std::min(6, (int)(std::thread::hardware_concurrency() / 2)));
    if (const char* env_nt = std::getenv("DIFFKV_COMPRESSOR_THREADS")) {
        num_threads = std::max(1, std::stoi(env_nt));
    }
    workers_.clear();
    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&AsyncCompressor::worker_loop, this);
    }
    return true;
}

void AsyncCompressor::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);
    queue_cv_.notify_all();
    for (auto & worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::queue<CompressJob> empty;
        std::swap(queue_, empty);
    }
}

bool AsyncCompressor::submit(const CompressJob& job) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (queue_.size() >= MAX_QUEUE_SIZE) {
        queue_overflows_.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "[Compressor] WARNING: Job queue overflow! SVD job dropped for block " << job.block_id << std::endl;
        return false;
    }
    queue_.push(job);
    jobs_submitted_.fetch_add(1, std::memory_order_relaxed);
    queue_cv_.notify_one();
    return true;
}

void AsyncCompressor::compress_sync(const CompressJob& job) {
    process_job(job);
}

void AsyncCompressor::wait_until_idle(uint64_t timeout_ms) {
    // Block until every job pushed so far has been dequeued + handled (worker increments
    // jobs_processed_ after each process_job, including early-returns). The caller must have
    // finished submitting before calling this. Sleeping yields CPU to the background-QoS
    // workers so they actually run. The timeout is a safety net against a worker crash.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (jobs_processed_.load(std::memory_order_acquire) <
           jobs_submitted_.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::cerr << "[Compressor] wait_until_idle TIMEOUT: processed="
                      << jobs_processed_.load() << " submitted=" << jobs_submitted_.load() << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void AsyncCompressor::worker_loop() {
    // ── Background / utility QoS so decode + audio are never starved ──────────
    // ACTIVE_RUNTIME: compression runs in a ThreadPoolExecutor that yields to
    // the asyncio event loop. Here we achieve the same by setting the thread
    // to the lowest scheduler class the OS offers.
#ifdef __APPLE__
    // QOS_CLASS_UTILITY: runs concurrently on performance cores but below the DEFAULT-QoS decode
    // thread. BACKGROUND was effectively unschedulable while the decode loop waits to DRAIN the
    // queue (the OS parks BACKGROUND on efficiency cores even when P-cores are idle) → ~150s
    // stalls. UTILITY drains fast without preempting decode/Metal.
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#else
    // Linux: set nice +10 (lower priority, but not the absolute floor) for this thread
    setpriority(PRIO_PROCESS, 0, 10);
#endif

    while (running_.load(std::memory_order_acquire)) {
        CompressJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !queue_.empty() || !running_.load(std::memory_order_acquire);
            });
            if (!running_.load(std::memory_order_acquire) && queue_.empty()) {
                break;
            }
            job = queue_.front();
            queue_.pop();
        }
        process_job(job);
        jobs_processed_.fetch_add(1, std::memory_order_relaxed);
    }
}

void AsyncCompressor::process_job(const CompressJob& job) {
    DiffKVBlockStateTable& active_table = job.state_table ? *job.state_table : state_table_;

    // 1. Check if session is alive
    if (alive_cb_ && !alive_cb_(job.session_id)) {
        active_table.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 2. Check if block is in Compressing state
    BlockState current = active_table.get(job.block_id);
    if (current != BlockState::Compressing) {
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 3. Delegate to lowrank block compressor logic
    LowRankCompressParams params;
    params.block_id = job.block_id;
    params.block_size = job.block_size;
    params.feat_dim = job.feat_dim;
    params.rank = job.rank;
    params.pool_rank = job.pool_rank;
    params.pool_block_size = job.pool_block_size;
    params.head_dim = job.head_dim;
    params.anchor_idx = job.anchor_idx;
    params.raw_k_ptr = job.raw_k_ptr;
    params.raw_v_ptr = job.raw_v_ptr;
    params.token_ids = job.token_ids;
    params.stop_token_ids = job.stop_token_ids;
    params.token_to_piece_fn = job.token_to_piece_fn;
    params.session_token_ids = job.session_token_ids;
    params.session_len = job.session_len;
    params.out_u_ptr = job.out_u_ptr;
    params.out_u_scale = job.out_u_scale;
    params.out_u_row_scale = job.out_u_row_scale;
    params.out_vk_ptr = job.out_vk_ptr;
    params.out_vv_ptr = job.out_vv_ptr;
    params.out_scale = job.out_scale;
    params.out_anchor_k = job.out_anchor_k;
    params.out_anchor_v = job.out_anchor_v;
    params.out_seq_len = job.out_seq_len;
    params.out_anchor_position = job.out_anchor_position;
    params.out_token_positions = job.out_token_positions;
    params.out_res_K_pos = job.out_res_K_pos;
    params.out_res_V_pos = job.out_res_V_pos;
    params.out_res_K_val = job.out_res_K_val;
    params.out_res_V_val = job.out_res_V_val;
    params.max_residual = job.max_residual;
    params.W_proj = job.W_proj;
    params.desc_dim = job.desc_dim;
    params.out_desc = job.out_desc;
    params.force_lapack = job.force_lapack;

    bool ok = compress_lowrank_block(params);

    if (!ok) {
        std::cerr << "[Compressor] Error: SVD failed for block " << job.block_id << std::endl;
        active_table.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 4. Check session alive again
    if (alive_cb_ && !alive_cb_(job.session_id)) {
        active_table.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 5. Atomic transition: Compressing -> CompressedResident
    bool trans_ok = active_table.transition(
        job.block_id,
        BlockState::Compressing,
        BlockState::CompressedResident
    );

    if (trans_ok) {
        if (g_diffkv_dbg_pos.load(std::memory_order_relaxed)) {
            std::cerr << "[DBG_TRANS] Session " << job.session_id << " anchor_idx=" << job.anchor_idx << " pool_idx=" << job.block_id << " state=CompressedResident\n";
        }
    } else {
        active_table.force_invalidate(job.block_id);
        jobs_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace diffkv
