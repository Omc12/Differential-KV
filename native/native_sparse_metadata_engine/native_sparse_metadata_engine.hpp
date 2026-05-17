// native_sparse_metadata_engine.hpp
// RCO-N Phase 41.1: Native Sparse Metadata Engine
// Compact C++ storage for sparse routing, confidence, and zoning metadata.

#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <chrono>

namespace diffkv {

// -------------------------------------------------------------------------
// Compact sparse metadata layout (64 bytes per session — cache-friendly)
// -------------------------------------------------------------------------
#pragma pack(push, 1)
struct SparseMetadataEntry {
    float    sparse_ratio;          // 0.0=dense, 1.0=fully sparse
    float    confidence_score;      // 0.0–1.0
    float    continuity_score;      // 0.0–1.0
    float    zone_stability;        // 0.0–1.0
    uint16_t active_layers;         // bitmask of sparse-active layers (up to 16)
    uint16_t repair_layers;         // bitmask of layers needing repair
    uint8_t  zone_id;               // 0=dense, 1=sparse-safe, 2=hybrid
    uint8_t  repair_type;           // 0=none, 1=head, 2=layer, 3=window, 4=full
    uint8_t  routing_version;       // Monotonic routing epoch
    uint8_t  flags;                 // bit0=sparse_safe, bit1=repair_pending, bit2=degraded
    uint32_t tokens_since_fusion;   // Tokens since last governance fusion
    uint32_t total_tokens;          // Lifetime tokens for this session
    uint8_t  _pad[16];              // Padding to 48 bytes (cache-line friendly)
};
#pragma pack(pop)

static_assert(sizeof(SparseMetadataEntry) == 48, "SparseMetadataEntry must be 48 bytes");

// -------------------------------------------------------------------------
// Helper flag accessors
// -------------------------------------------------------------------------
inline bool is_sparse_safe(const SparseMetadataEntry& e) { return (e.flags & 0x01) != 0; }
inline bool repair_pending(const SparseMetadataEntry& e) { return (e.flags & 0x02) != 0; }
inline bool is_degraded(const SparseMetadataEntry& e)    { return (e.flags & 0x04) != 0; }

inline void set_sparse_safe(SparseMetadataEntry& e, bool v) {
    e.flags = v ? (e.flags | 0x01) : (e.flags & ~0x01);
}
inline void set_repair_pending(SparseMetadataEntry& e, bool v) {
    e.flags = v ? (e.flags | 0x02) : (e.flags & ~0x02);
}
inline void set_degraded(SparseMetadataEntry& e, bool v) {
    e.flags = v ? (e.flags | 0x04) : (e.flags & ~0x04);
}

// -------------------------------------------------------------------------
// NativeSparseMetadataEngine
// -------------------------------------------------------------------------
class NativeSparseMetadataEngine {
public:
    explicit NativeSparseMetadataEngine(int max_sessions = 256);
    ~NativeSparseMetadataEngine() = default;

    // Session management
    void create_session(const std::string& session_id);
    void remove_session(const std::string& session_id);
    bool has_session(const std::string& session_id) const;

    // Metadata updates (called by governance fusion layer)
    void update(const std::string& session_id,
                float sparse_ratio, float confidence, float continuity,
                uint8_t zone_id, uint8_t repair_type,
                bool sparse_safe, bool repair_pending_flag, bool degraded);

    // Fast-path read (hot path — minimal locking)
    bool is_sparse_safe(const std::string& session_id) const;
    float get_confidence(const std::string& session_id) const;
    float get_sparse_ratio(const std::string& session_id) const;

    // Token bookkeeping
    void record_token(const std::string& session_id, uint32_t count = 1);
    void record_fusion(const std::string& session_id);

    // Batch query (for governance fusion)
    std::vector<std::string> get_sessions_below_confidence(float threshold) const;
    std::vector<std::string> get_sessions_needing_repair() const;

    // Statistics
    size_t session_count() const;
    std::string get_stats_json() const;

private:
    mutable std::mutex mtx_;
    int max_sessions_;
    std::unordered_map<std::string, SparseMetadataEntry> entries_;

    // Stats
    mutable uint64_t total_updates_{0};
    mutable uint64_t total_fast_reads_{0};
    mutable uint64_t sparse_safe_hits_{0};
    mutable uint64_t repair_triggered_{0};
};

} // namespace diffkv
