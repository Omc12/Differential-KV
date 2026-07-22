// attention_cache.hpp
// Translation of attention_cache.py to C++17.
// Per-head cosine-similarity cache that allows reuse of attention outputs
// across decode steps when query vectors change negligibly.

#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>

namespace dkv {

// ---------------------------------------------------------------------------
// AttentionScoreCache
//
// Caches attention outputs to allow reuse when queries barely change.
//
// Key:  (session_id, layer_idx)
// Value: (cached_q [H_q * D], cached_attn_out [H_q * D])
//
// check_and_update:
//   For each query head h, compute cosine similarity between current q[h]
//   and cached q_prev[h].  If sim[h] >= threshold, reuse_mask[h] = true.
//   Returns true when a cached entry exists (even if no heads are reused).
//
// save:
//   Store q and attn_out for the given (session_id, layer_idx).
// ---------------------------------------------------------------------------
struct AttentionScoreCache {
    float threshold = 2.0f;  // cosine-similarity threshold for reuse (>1 means "never reuse" by default — set to e.g. 0.999 to activate)

    AttentionScoreCache() {
        if (const char* env_t = std::getenv("DKV_ATTN_CACHE_THRESHOLD")) {
            try {
                threshold = std::stof(env_t);
            } catch (...) {
                threshold = 2.0f;
            }
        }
    }

    // session_id -> layer_idx -> (q_prev, attn_out_prev)
    std::unordered_map<std::string,
        std::unordered_map<int,
            std::pair<std::vector<float>,   // q_prev    [H_q * D]
                      std::vector<float>>>> // attn_out  [H_q * D]
    cache;

    // -----------------------------------------------------------------
    // clear_session: remove all cached layers for a session
    // -----------------------------------------------------------------
    void clear_session(const std::string& session_id) {
        cache.erase(session_id);
    }

    // -----------------------------------------------------------------
    // clear_all
    // -----------------------------------------------------------------
    void clear_all() {
        cache.clear();
    }

    // -----------------------------------------------------------------
    // check_and_update
    //
    // Compare current query q against the cached query for
    // (session_id, layer_idx).
    //
    // Populates:
    //   reuse_mask[h] = true  if cosine_sim(q[h], q_prev[h]) >= threshold
    //   out_cached: copy of cached attn_out (full [H_q * D])
    //
    // Returns true if a cache entry existed (regardless of reuse_mask).
    // -----------------------------------------------------------------
    bool check_and_update(
        const std::string& session_id,
        int                layer_idx,
        const float*       q,           // [H_q * D]
        int                H_q,
        int                D,
        std::vector<bool>& reuse_mask,  // [H_q] output
        std::vector<float>& out_cached  // [H_q * D] output
    ) {
        reuse_mask.assign(H_q, false);
        out_cached.assign(H_q * D, 0.0f);

        auto sess_it = cache.find(session_id);
        if (sess_it == cache.end()) return false;

        auto layer_it = sess_it->second.find(layer_idx);
        if (layer_it == sess_it->second.end()) return false;

        const std::vector<float>& q_prev     = layer_it->second.first;
        const std::vector<float>& attn_prev  = layer_it->second.second;

        if (static_cast<int>(q_prev.size()) != H_q * D) return false;

        // Copy cached attn_out
        out_cached = attn_prev;

        // Compute per-head cosine similarity
        for (int h = 0; h < H_q; ++h) {
            const float* qh      = q      + h * D;
            const float* qh_prev = q_prev.data() + h * D;

            float dot  = 0.0f;
            float sq_a = 0.0f;
            float sq_b = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot  += qh[d] * qh_prev[d];
                sq_a += qh[d] * qh[d];
                sq_b += qh_prev[d] * qh_prev[d];
            }
            float denom = (std::sqrt(sq_a) + 1e-8f) * (std::sqrt(sq_b) + 1e-8f);
            float sim   = dot / denom;

            if (sim >= threshold) {
                reuse_mask[h] = true;
            }
        }

        return true;
    }

    // -----------------------------------------------------------------
    // save
    // Store current q and attn_out for future reuse.
    // -----------------------------------------------------------------
    void save(
        const std::string& session_id,
        int                layer_idx,
        const float*       q,        // [H_q * D]
        int                H_q,
        int                D,
        const float*       attn_out  // [H_q * D]
    ) {
        auto& entry = cache[session_id][layer_idx];
        entry.first.assign(q, q + H_q * D);
        entry.second.assign(attn_out, attn_out + H_q * D);
    }

    // -----------------------------------------------------------------
    // has_entry: quick check without triggering full reuse logic
    // -----------------------------------------------------------------
    bool has_entry(const std::string& session_id, int layer_idx) const {
        auto sess_it = cache.find(session_id);
        if (sess_it == cache.end()) return false;
        return sess_it->second.count(layer_idx) > 0;
    }

    // -----------------------------------------------------------------
    // memory_bytes: approximate memory usage
    // -----------------------------------------------------------------
    size_t memory_bytes() const {
        size_t total = 0;
        for (const auto& sess : cache) {
            for (const auto& layer : sess.second) {
                total += layer.second.first.size()  * sizeof(float);
                total += layer.second.second.size() * sizeof(float);
            }
        }
        return total;
    }
};

// Global singleton (optional — caller may prefer to instantiate directly)
inline AttentionScoreCache& get_global_attn_cache() {
    static AttentionScoreCache inst;
    return inst;
}

} // namespace dkv
