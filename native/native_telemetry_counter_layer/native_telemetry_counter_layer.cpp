// native_telemetry_counter_layer.cpp
// RCO-N Phase 41.1

#include "native_telemetry_counter_layer.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace diffkv {

NativeTelemetryCounterLayer::NativeTelemetryCounterLayer()
    : created_ts_(std::chrono::steady_clock::now()) {}

void NativeTelemetryCounterLayer::gpu_kernel_dispatched() noexcept {
    counters_.gpu_kernels_dispatched.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::gpu_starvation_event() noexcept {
    counters_.gpu_starvation_events.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::gpu_sync_stall(double ms) noexcept {
    counters_.gpu_sync_stalls.fetch_add(1, std::memory_order_relaxed);
    // Approximate atomic double add (not truly atomic but acceptable for telemetry)
    auto old = counters_.gpu_total_stall_ms.load(std::memory_order_relaxed);
    counters_.gpu_total_stall_ms.store(old + ms, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::scheduler_step() noexcept {
    counters_.scheduler_steps.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::scheduler_admission() noexcept {
    counters_.scheduler_admissions.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::scheduler_eviction() noexcept {
    counters_.scheduler_evictions.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::queue_enqueue() noexcept {
    counters_.queue_enqueues.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::queue_dequeue() noexcept {
    counters_.queue_dequeues.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::queue_cancel() noexcept {
    counters_.queue_cancellations.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::queue_reconnect(bool coalesced) noexcept {
    counters_.queue_reconnects.fetch_add(1, std::memory_order_relaxed);
    if (coalesced)
        counters_.queue_reconnects_coalesced.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::token_generated(uint64_t n) noexcept {
    counters_.tokens_generated.fetch_add(n, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::governance_fired() noexcept {
    counters_.governance_fires.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::governance_skipped() noexcept {
    counters_.governance_skips.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::dense_fallback() noexcept {
    counters_.dense_fallbacks.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::partial_repair() noexcept {
    counters_.partial_repairs.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::fusion_call() noexcept {
    counters_.fusion_calls.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::telemetry_suppressed() noexcept {
    counters_.telemetry_calls_suppressed.fetch_add(1, std::memory_order_relaxed);
}
void NativeTelemetryCounterLayer::telemetry_emitted() noexcept {
    counters_.telemetry_calls_emitted.fetch_add(1, std::memory_order_relaxed);
}

double NativeTelemetryCounterLayer::governance_collapse_ratio() const noexcept {
    uint64_t fires = counters_.governance_fires.load(std::memory_order_relaxed);
    uint64_t skips = counters_.governance_skips.load(std::memory_order_relaxed);
    uint64_t total = fires + skips;
    return total > 0 ? static_cast<double>(skips) / total : 0.0;
}

double NativeTelemetryCounterLayer::queue_reconnect_coalesce_ratio() const noexcept {
    uint64_t total = counters_.queue_reconnects.load(std::memory_order_relaxed);
    uint64_t coalesced = counters_.queue_reconnects_coalesced.load(std::memory_order_relaxed);
    return total > 0 ? static_cast<double>(coalesced) / total : 0.0;
}

double NativeTelemetryCounterLayer::telemetry_suppression_ratio() const noexcept {
    uint64_t suppressed = counters_.telemetry_calls_suppressed.load(std::memory_order_relaxed);
    uint64_t emitted    = counters_.telemetry_calls_emitted.load(std::memory_order_relaxed);
    uint64_t total      = suppressed + emitted;
    return total > 0 ? static_cast<double>(suppressed) / total : 0.0;
}

void NativeTelemetryCounterLayer::reset_counters() noexcept {
    counters_.gpu_kernels_dispatched.store(0, std::memory_order_relaxed);
    counters_.gpu_starvation_events.store(0, std::memory_order_relaxed);
    counters_.gpu_sync_stalls.store(0, std::memory_order_relaxed);
    counters_.gpu_total_stall_ms.store(0.0, std::memory_order_relaxed);
    counters_.scheduler_steps.store(0, std::memory_order_relaxed);
    counters_.scheduler_admissions.store(0, std::memory_order_relaxed);
    counters_.scheduler_evictions.store(0, std::memory_order_relaxed);
    counters_.queue_enqueues.store(0, std::memory_order_relaxed);
    counters_.queue_dequeues.store(0, std::memory_order_relaxed);
    counters_.queue_cancellations.store(0, std::memory_order_relaxed);
    counters_.queue_reconnects.store(0, std::memory_order_relaxed);
    counters_.queue_reconnects_coalesced.store(0, std::memory_order_relaxed);
    counters_.tokens_generated.store(0, std::memory_order_relaxed);
    counters_.governance_fires.store(0, std::memory_order_relaxed);
    counters_.governance_skips.store(0, std::memory_order_relaxed);
    counters_.dense_fallbacks.store(0, std::memory_order_relaxed);
    counters_.partial_repairs.store(0, std::memory_order_relaxed);
    counters_.fusion_calls.store(0, std::memory_order_relaxed);
    counters_.telemetry_calls_suppressed.store(0, std::memory_order_relaxed);
    counters_.telemetry_calls_emitted.store(0, std::memory_order_relaxed);
}

std::string NativeTelemetryCounterLayer::get_snapshot_json() const {
    // Relaxed reads — telemetry doesn't need perfect consistency
    auto r = [](auto& a) { return a.load(std::memory_order_relaxed); };
    double gov_collapse = governance_collapse_ratio();
    double telem_suppress = telemetry_suppression_ratio();
    double reconnect_coalesce = queue_reconnect_coalesce_ratio();

    auto elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - created_ts_).count();

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "{"
        << "\"elapsed_ms\":"               << elapsed_ms              << ","
        << "\"gpu_kernels_dispatched\":"   << r(counters_.gpu_kernels_dispatched) << ","
        << "\"gpu_starvation_events\":"    << r(counters_.gpu_starvation_events) << ","
        << "\"gpu_sync_stalls\":"          << r(counters_.gpu_sync_stalls) << ","
        << "\"gpu_total_stall_ms\":"       << r(counters_.gpu_total_stall_ms) << ","
        << "\"scheduler_steps\":"          << r(counters_.scheduler_steps) << ","
        << "\"scheduler_admissions\":"     << r(counters_.scheduler_admissions) << ","
        << "\"scheduler_evictions\":"      << r(counters_.scheduler_evictions) << ","
        << "\"queue_enqueues\":"           << r(counters_.queue_enqueues) << ","
        << "\"queue_dequeues\":"           << r(counters_.queue_dequeues) << ","
        << "\"queue_cancellations\":"      << r(counters_.queue_cancellations) << ","
        << "\"queue_reconnects\":"         << r(counters_.queue_reconnects) << ","
        << "\"queue_reconnects_coalesced\":"<< r(counters_.queue_reconnects_coalesced) << ","
        << "\"tokens_generated\":"         << r(counters_.tokens_generated) << ","
        << "\"governance_fires\":"         << r(counters_.governance_fires) << ","
        << "\"governance_skips\":"         << r(counters_.governance_skips) << ","
        << "\"dense_fallbacks\":"          << r(counters_.dense_fallbacks) << ","
        << "\"partial_repairs\":"          << r(counters_.partial_repairs) << ","
        << "\"fusion_calls\":"             << r(counters_.fusion_calls) << ","
        << "\"telemetry_suppressed\":"     << r(counters_.telemetry_calls_suppressed) << ","
        << "\"telemetry_emitted\":"        << r(counters_.telemetry_calls_emitted) << ","
        << "\"governance_collapse_ratio\":" << gov_collapse << ","
        << "\"telemetry_suppress_ratio\":"  << telem_suppress << ","
        << "\"reconnect_coalesce_ratio\":"  << reconnect_coalesce
        << "}";
    return oss.str();
}

} // namespace diffkv
