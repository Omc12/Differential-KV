// native_decode_scheduler.hpp
// RCO-N Phase 41.1: Native Decode Scheduler
// Low-overhead C++ scheduling primitives for persistent decode batches.
// Provides pybind11 bindings for Python integration.

#pragma once
#include <vector>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <string>
#include <functional>
#include <optional>

namespace dkv {

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
using DurationMs = std::chrono::duration<double, std::milli>;

// -------------------------------------------------------------------------
// Slot represents one active decode session in the persistent batch
// -------------------------------------------------------------------------
struct DecodeSlot {
    std::string session_id;
    std::string request_id;
    int tokens_generated{0};
    int max_tokens{128};
    int priority{0};
    bool finished{false};
    TimePoint admitted_ts;
    TimePoint last_token_ts;
    double queue_wait_ms{0.0};

    explicit DecodeSlot(const std::string& sid, const std::string& rid,
                        int max_tok = 128, int prio = 0)
        : session_id(sid), request_id(rid), max_tokens(max_tok), priority(prio),
          admitted_ts(Clock::now()), last_token_ts(Clock::now()) {}

    bool is_expired(double idle_threshold_ms = 30000.0) const {
        auto now = Clock::now();
        double idle = DurationMs(now - last_token_ts).count();
        return idle > idle_threshold_ms;
    }

    double slot_age_ms() const {
        return DurationMs(Clock::now() - admitted_ts).count();
    }
};

// -------------------------------------------------------------------------
// Scheduling Statistics (read by Python side)
// -------------------------------------------------------------------------
struct SchedulerStats {
    uint64_t total_batch_steps{0};
    uint64_t total_tokens_scheduled{0};
    uint64_t slot_fills{0};
    uint64_t slot_evictions{0};
    uint64_t starvation_events{0};
    double   avg_batch_size{0.0};
    double   avg_starvation_gap_ms{0.0};
    double   max_starvation_gap_ms{0.0};
    double   scheduler_overhead_us{0.0};    // avg microseconds per schedule() call
    uint64_t admission_queue_depth{0};
    uint64_t active_slots{0};
};

// -------------------------------------------------------------------------
// NativeDecodeScheduler
// -------------------------------------------------------------------------
class NativeDecodeScheduler {
public:
    explicit NativeDecodeScheduler(int max_batch_size = 32,
                                    double starvation_threshold_ms = 1.5);
    ~NativeDecodeScheduler() = default;

    // Session management
    void admit(const std::string& session_id, const std::string& request_id,
               int max_tokens = 128, int priority = 0);
    void complete(const std::string& session_id);
    void cancel(const std::string& session_id);

    // Batch preparation (hot path — must be very fast)
    std::vector<std::string> prepare_batch();

    // Token recording
    void record_token(const std::string& session_id, int count = 1);

    // Step lifecycle
    void step_begin();
    void step_end(const std::vector<std::string>& batch);

    // Statistics
    SchedulerStats get_stats() const;
    std::string    get_stats_json() const;

    // Configuration
    int    max_batch_size() const { return max_batch_size_; }
    void   set_max_batch_size(int n) { max_batch_size_ = n; }

private:
    void evict_finished_locked_();
    void fill_from_admission_locked_();
    void record_starvation_gap_(double gap_ms);

    mutable std::mutex mtx_;

    int    max_batch_size_;
    double starvation_threshold_ms_;

    // Persistent slot array
    std::unordered_map<std::string, DecodeSlot> slots_;

    // Admission queue (priority ordered)
    struct AdmissionEntry {
        int priority;
        TimePoint enqueue_ts;
        DecodeSlot slot;
        bool operator<(const AdmissionEntry& o) const {
            return priority < o.priority; // max-heap by priority
        }
    };
    std::priority_queue<AdmissionEntry> admission_queue_;

    // Step tracking
    TimePoint last_step_end_ts_;
    bool      first_step_{true};

    // Statistics accumulators
    uint64_t total_batch_steps_{0};
    uint64_t total_tokens_scheduled_{0};
    uint64_t slot_fills_{0};
    uint64_t slot_evictions_{0};
    uint64_t starvation_events_{0};
    double   starvation_gap_sum_ms_{0.0};
    double   max_starvation_gap_ms_{0.0};
    double   batch_size_sum_{0.0};
    double   scheduler_overhead_sum_us_{0.0};
};

} // namespace dkv
