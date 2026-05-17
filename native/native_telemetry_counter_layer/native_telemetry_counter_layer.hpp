// native_telemetry_counter_layer.hpp
// RCO-N Phase 41.1: Native Telemetry Counter Layer
// Lock-free atomic counters replacing Python high-frequency telemetry wakeups.

#pragma once
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <chrono>
#include <cstdint>

namespace diffkv {

// -------------------------------------------------------------------------
// Atomic counter group — one per category
// All increments are lock-free (std::atomic with relaxed ordering for perf)
// -------------------------------------------------------------------------
struct TelemetryCounters {
    // GPU counters
    std::atomic<uint64_t> gpu_kernels_dispatched{0};
    std::atomic<uint64_t> gpu_starvation_events{0};
    std::atomic<uint64_t> gpu_sync_stalls{0};
    std::atomic<double>   gpu_total_stall_ms{0.0};

    // Scheduler counters
    std::atomic<uint64_t> scheduler_steps{0};
    std::atomic<uint64_t> scheduler_admissions{0};
    std::atomic<uint64_t> scheduler_evictions{0};
    std::atomic<uint64_t> scheduler_starvation_events{0};

    // Queue counters
    std::atomic<uint64_t> queue_enqueues{0};
    std::atomic<uint64_t> queue_dequeues{0};
    std::atomic<uint64_t> queue_cancellations{0};
    std::atomic<uint64_t> queue_reconnects{0};
    std::atomic<uint64_t> queue_reconnects_coalesced{0};

    // Token/occupancy counters
    std::atomic<uint64_t> tokens_generated{0};
    std::atomic<uint64_t> governance_fires{0};
    std::atomic<uint64_t> governance_skips{0};
    std::atomic<uint64_t> dense_fallbacks{0};
    std::atomic<uint64_t> partial_repairs{0};
    std::atomic<uint64_t> fusion_calls{0};

    // Suppression (counts telemetry calls that were batched/suppressed)
    std::atomic<uint64_t> telemetry_calls_suppressed{0};
    std::atomic<uint64_t> telemetry_calls_emitted{0};

    // Padding to avoid false sharing between counter groups
    char _pad[64 - sizeof(std::atomic<uint64_t>) % 64];

    TelemetryCounters() = default;
    TelemetryCounters(const TelemetryCounters&) = delete;
    TelemetryCounters& operator=(const TelemetryCounters&) = delete;
};

// -------------------------------------------------------------------------
// NativeTelemetryCounterLayer
// -------------------------------------------------------------------------
class NativeTelemetryCounterLayer {
public:
    NativeTelemetryCounterLayer();

    // ---- Lock-free increment operations (HOT PATH) ----
    void gpu_kernel_dispatched()            noexcept;
    void gpu_starvation_event()             noexcept;
    void gpu_sync_stall(double ms)          noexcept;
    void scheduler_step()                   noexcept;
    void scheduler_admission()              noexcept;
    void scheduler_eviction()               noexcept;
    void queue_enqueue()                    noexcept;
    void queue_dequeue()                    noexcept;
    void queue_cancel()                     noexcept;
    void queue_reconnect(bool coalesced)    noexcept;
    void token_generated(uint64_t n = 1)   noexcept;
    void governance_fired()                 noexcept;
    void governance_skipped()               noexcept;
    void dense_fallback()                   noexcept;
    void partial_repair()                   noexcept;
    void fusion_call()                      noexcept;
    void telemetry_suppressed()             noexcept;
    void telemetry_emitted()                noexcept;

    // ---- Snapshot reads (can be called at lower frequency) ----
    std::string get_snapshot_json() const;

    // ---- Derived metrics ----
    double governance_collapse_ratio() const noexcept;
    double queue_reconnect_coalesce_ratio() const noexcept;
    double telemetry_suppression_ratio() const noexcept;

    // ---- Reset (for fresh window) ----
    void reset_counters() noexcept;

private:
    TelemetryCounters counters_;
    mutable std::mutex snapshot_mtx_;  // Only for JSON serialization
    std::chrono::steady_clock::time_point created_ts_;
};

} // namespace diffkv
