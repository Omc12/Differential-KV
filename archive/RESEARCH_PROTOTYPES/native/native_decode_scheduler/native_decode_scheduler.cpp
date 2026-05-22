// native_decode_scheduler.cpp
// RCO-N Phase 41.1: Native Decode Scheduler Implementation

#include "native_decode_scheduler.hpp"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace diffkv {

NativeDecodeScheduler::NativeDecodeScheduler(int max_batch_size,
                                               double starvation_threshold_ms)
    : max_batch_size_(max_batch_size),
      starvation_threshold_ms_(starvation_threshold_ms),
      last_step_end_ts_(Clock::now()),
      first_step_(true) {}

void NativeDecodeScheduler::admit(const std::string& session_id,
                                   const std::string& request_id,
                                   int max_tokens, int priority) {
    std::lock_guard<std::mutex> lock(mtx_);
    AdmissionEntry entry{
        priority,
        Clock::now(),
        DecodeSlot(session_id, request_id, max_tokens, priority)
    };
    admission_queue_.push(std::move(entry));
}

void NativeDecodeScheduler::complete(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = slots_.find(session_id);
    if (it != slots_.end()) {
        it->second.finished = true;
    }
}

void NativeDecodeScheduler::cancel(const std::string& session_id) {
    complete(session_id); // Cancellation = mark finished
}

std::vector<std::string> NativeDecodeScheduler::prepare_batch() {
    auto t0 = Clock::now();
    std::lock_guard<std::mutex> lock(mtx_);

    evict_finished_locked_();
    fill_from_admission_locked_();

    std::vector<std::string> batch;
    batch.reserve(slots_.size());
    for (auto& [sid, slot] : slots_) {
        if (!slot.finished) {
            batch.push_back(sid);
        }
    }

    total_batch_steps_++;
    batch_size_sum_ += static_cast<double>(batch.size());

    double overhead_us = DurationMs(Clock::now() - t0).count() * 1000.0;
    scheduler_overhead_sum_us_ += overhead_us;

    return batch;
}

void NativeDecodeScheduler::record_token(const std::string& session_id, int count) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = slots_.find(session_id);
    if (it != slots_.end()) {
        it->second.tokens_generated += count;
        it->second.last_token_ts = Clock::now();
        if (it->second.tokens_generated >= it->second.max_tokens) {
            it->second.finished = true;
        }
        total_tokens_scheduled_ += count;
    }
}

void NativeDecodeScheduler::step_begin() {
    // Measure starvation gap since last step end
    if (!first_step_) {
        auto now = Clock::now();
        double gap_ms = DurationMs(now - last_step_end_ts_).count();
        if (gap_ms > starvation_threshold_ms_) {
            record_starvation_gap_(gap_ms);
        }
    }
    first_step_ = false;
}

void NativeDecodeScheduler::step_end(const std::vector<std::string>& /*batch*/) {
    last_step_end_ts_ = Clock::now();
}

void NativeDecodeScheduler::evict_finished_locked_() {
    std::vector<std::string> to_evict;
    for (auto& [sid, slot] : slots_) {
        if (slot.finished || slot.is_expired()) {
            to_evict.push_back(sid);
        }
    }
    for (const auto& sid : to_evict) {
        slots_.erase(sid);
        slot_evictions_++;
    }
}

void NativeDecodeScheduler::fill_from_admission_locked_() {
    while (!admission_queue_.empty() &&
           static_cast<int>(slots_.size()) < max_batch_size_) {
        auto entry = admission_queue_.top();
        admission_queue_.pop();
        // Compute queue wait time
        double wait_ms = DurationMs(Clock::now() - entry.enqueue_ts).count();
        entry.slot.queue_wait_ms = wait_ms;
        slots_.emplace(entry.slot.session_id, std::move(entry.slot));
        slot_fills_++;
    }
}

void NativeDecodeScheduler::record_starvation_gap_(double gap_ms) {
    starvation_events_++;
    starvation_gap_sum_ms_ += gap_ms;
    if (gap_ms > max_starvation_gap_ms_) {
        max_starvation_gap_ms_ = gap_ms;
    }
}

SchedulerStats NativeDecodeScheduler::get_stats() const {
    std::lock_guard<std::mutex> lock(mtx_);
    SchedulerStats s;
    s.total_batch_steps       = total_batch_steps_;
    s.total_tokens_scheduled  = total_tokens_scheduled_;
    s.slot_fills              = slot_fills_;
    s.slot_evictions          = slot_evictions_;
    s.starvation_events       = starvation_events_;
    s.active_slots            = slots_.size();
    s.admission_queue_depth   = admission_queue_.size();
    s.avg_batch_size = total_batch_steps_ > 0
        ? batch_size_sum_ / static_cast<double>(total_batch_steps_) : 0.0;
    s.avg_starvation_gap_ms = starvation_events_ > 0
        ? starvation_gap_sum_ms_ / static_cast<double>(starvation_events_) : 0.0;
    s.max_starvation_gap_ms   = max_starvation_gap_ms_;
    s.scheduler_overhead_us = total_batch_steps_ > 0
        ? scheduler_overhead_sum_us_ / static_cast<double>(total_batch_steps_) : 0.0;
    return s;
}

std::string NativeDecodeScheduler::get_stats_json() const {
    auto s = get_stats();
    std::ostringstream oss;
    oss << "{"
        << "\"total_batch_steps\":" << s.total_batch_steps << ","
        << "\"total_tokens_scheduled\":" << s.total_tokens_scheduled << ","
        << "\"slot_fills\":" << s.slot_fills << ","
        << "\"slot_evictions\":" << s.slot_evictions << ","
        << "\"starvation_events\":" << s.starvation_events << ","
        << "\"active_slots\":" << s.active_slots << ","
        << "\"admission_queue_depth\":" << s.admission_queue_depth << ","
        << "\"avg_batch_size\":" << s.avg_batch_size << ","
        << "\"avg_starvation_gap_ms\":" << s.avg_starvation_gap_ms << ","
        << "\"max_starvation_gap_ms\":" << s.max_starvation_gap_ms << ","
        << "\"scheduler_overhead_us\":" << s.scheduler_overhead_us
        << "}";
    return oss.str();
}

} // namespace diffkv
