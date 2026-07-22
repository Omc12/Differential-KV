// native_sparse_metadata_engine.cpp
// RCO-N Phase 41.1

#include "native_sparse_metadata_engine.hpp"
#include <sstream>
#include <iomanip>

namespace dkv {

NativeSparseMetadataEngine::NativeSparseMetadataEngine(int max_sessions)
    : max_sessions_(max_sessions) {
    entries_.reserve(max_sessions);
}

void NativeSparseMetadataEngine::create_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    SparseMetadataEntry e{};
    e.sparse_ratio       = 1.0f;  // Start fully sparse
    e.confidence_score   = 1.0f;
    e.continuity_score   = 1.0f;
    e.zone_stability     = 1.0f;
    e.zone_id            = 1;     // sparse-safe zone
    e.flags              = 0x01;  // sparse_safe=true
    entries_[session_id] = e;
}

void NativeSparseMetadataEngine::remove_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    entries_.erase(session_id);
}

bool NativeSparseMetadataEngine::has_session(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return entries_.count(session_id) > 0;
}

void NativeSparseMetadataEngine::update(
    const std::string& session_id,
    float sparse_ratio, float confidence, float continuity,
    uint8_t zone_id, uint8_t repair_type,
    bool sparse_safe, bool repair_pending_flag, bool degraded) {

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(session_id);
    if (it == entries_.end()) {
        create_session(session_id);
        it = entries_.find(session_id);
    }
    auto& e = it->second;
    e.sparse_ratio     = sparse_ratio;
    e.confidence_score = confidence;
    e.continuity_score = continuity;
    e.zone_id          = zone_id;
    e.repair_type      = repair_type;
    e.tokens_since_fusion = 0;
    set_sparse_safe(e, sparse_safe);
    set_repair_pending(e, repair_pending_flag);
    set_degraded(e, degraded);
    e.routing_version++;
    total_updates_++;
}

bool NativeSparseMetadataEngine::is_sparse_safe(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(session_id);
    total_fast_reads_++;
    if (it == entries_.end()) {
        return true; // Default: assume sparse-safe for new sessions
    }
    bool safe = dkv::is_sparse_safe(it->second);
    if (safe) sparse_safe_hits_++;
    return safe;
}

float NativeSparseMetadataEngine::get_confidence(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(session_id);
    return it != entries_.end() ? it->second.confidence_score : 1.0f;
}

float NativeSparseMetadataEngine::get_sparse_ratio(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(session_id);
    return it != entries_.end() ? it->second.sparse_ratio : 1.0f;
}

void NativeSparseMetadataEngine::record_token(const std::string& session_id, uint32_t count) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(session_id);
    if (it != entries_.end()) {
        it->second.tokens_since_fusion += count;
        it->second.total_tokens += count;
    }
}

void NativeSparseMetadataEngine::record_fusion(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(session_id);
    if (it != entries_.end()) {
        it->second.tokens_since_fusion = 0;
    }
}

std::vector<std::string> NativeSparseMetadataEngine::get_sessions_below_confidence(float threshold) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::string> result;
    for (const auto& [sid, e] : entries_) {
        if (e.confidence_score < threshold) {
            result.push_back(sid);
        }
    }
    return result;
}

std::vector<std::string> NativeSparseMetadataEngine::get_sessions_needing_repair() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<std::string> result;
    for (const auto& [sid, e] : entries_) {
        if (repair_pending(e)) {
            result.push_back(sid);
            repair_triggered_++;
        }
    }
    return result;
}

size_t NativeSparseMetadataEngine::session_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return entries_.size();
}

std::string NativeSparseMetadataEngine::get_stats_json() const {
    std::lock_guard<std::mutex> lock(mtx_);
    double safe_hit_rate = total_fast_reads_ > 0
        ? static_cast<double>(sparse_safe_hits_) / total_fast_reads_ : 0.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "{"
        << "\"session_count\":" << entries_.size() << ","
        << "\"total_updates\":" << total_updates_ << ","
        << "\"total_fast_reads\":" << total_fast_reads_ << ","
        << "\"sparse_safe_hit_rate\":" << safe_hit_rate << ","
        << "\"repair_triggered\":" << repair_triggered_ << ","
        << "\"entry_size_bytes\":" << sizeof(SparseMetadataEntry)
        << "}";
    return oss.str();
}

} // namespace dkv
