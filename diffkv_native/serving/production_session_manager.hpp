#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <chrono>
#include "native_core/kv_runtime_manager.hpp"
#include "native_core/paging/paged_kv_store.hpp"
#include "native_core/srl/session_srl_state.hpp"

namespace diffkv {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ProductionSession {
    std::string session_id;
    double created_at = 0.0;
    double last_accessed = 0.0;
    std::unordered_map<std::string, std::string> config;
    std::string status = "active";
    std::vector<ChatMessage> history;
    std::vector<int32_t> token_ids;
    
    // Evicted/resident KV blocks data
    std::vector<std::vector<std::unique_ptr<StreamingKVBlock>>> layers_blocks;
    std::unordered_map<std::string, PageEntry> pager_entries;
    PagedKVStore::Stats pager_stats;
    
    // SRL Routing state
    SessionSRLState srl_state;
    bool has_srl_state = false;

    // Running execution state for serving
    std::vector<std::vector<float>> active_k_dense;
    std::vector<std::vector<float>> active_v_dense;
    std::vector<std::vector<int32_t>> seq_lens_by_layer;
    std::vector<int32_t> active_positions_dense;
    std::vector<int32_t> last_turn_token_prefix;
    int active_slot = 0;
    int active_block_tokens = 0;
    std::map<int, std::vector<float>> persistent_k_dense;
    std::map<int, std::vector<float>> persistent_v_dense;
    InvertedIndex inverted_index;
    int micro_block_size = 64;
};

class SharedPrefixManager {
public:
    SharedPrefixManager(KVRuntimeManager* kv_manager = nullptr);
    ~SharedPrefixManager();

    struct PrefixData {
        std::vector<int> pool_indices;
        std::vector<int> anchor_indices;
        int ref_count = 0;
    };

    void register_session_prefix(
        const std::string& session_id,
        const std::vector<int32_t>& prefix_tokens,
        const std::vector<int>& pool_indices,
        const std::vector<int>& anchor_indices
    );

    void release_session_prefixes(const std::string& session_id);

    PrefixData* lookup_prefix(const std::vector<int32_t>& prefix_tokens);

private:
    KVRuntimeManager* kv_manager_;
    
    // Hash function for token vector to use as map key
    struct VectorHash {
        size_t operator()(const std::vector<int32_t>& vec) const {
            size_t seed = vec.size();
            for (auto x : vec) {
                seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    std::unordered_map<std::vector<int32_t>, PrefixData, VectorHash> shared_prefixes_;
    std::unordered_map<std::string, std::vector<std::vector<int32_t>>> session_prefixes_;
    std::mutex mutex_;
};

class ProductionSessionManager {
public:
    ProductionSessionManager(
        const std::string& storage_path = "./session_checkpoints",
        int max_resident_sessions = -1,
        KVRuntimeManager* kv_manager = nullptr
    );
    ~ProductionSessionManager();

    std::string create_session(const std::unordered_map<std::string, std::string>& config = {});
    
    std::shared_ptr<ProductionSession> get_session(const std::string& session_id);
    
    void save_session(const std::string& session_id);
    
    void delete_session(const std::string& session_id);
    
    void cleanup_idle_sessions(int idle_timeout_seconds = 3600);
    
    std::vector<std::string> list_sessions();

    // Conversation history management
    std::vector<ChatMessage> get_history(const std::string& session_id);
    void append_message(const std::string& session_id, const std::string& role, const std::string& content);
    void clear_history(const std::string& session_id);

    // Swap active session in and out of the KVRuntimeManager
    void ensure_residency(const std::string& session_id);
    void evict_from_vram(const std::string& session_id);
    void load_into_vram(const std::string& session_id);

    std::string get_active_session_id() const {
        return current_active_session_id_;
    }

    std::mutex& get_mutex() { return mutex_; }
    const std::unordered_map<std::string, std::shared_ptr<ProductionSession>>& get_active_sessions() const { return active_sessions_; }
    std::string create_session_with_id(const std::string& session_id, const std::unordered_map<std::string, std::string>& config = {});

private:
    std::string create_session_with_id_locked(const std::string& session_id, const std::unordered_map<std::string, std::string>& config);
    std::shared_ptr<ProductionSession> load_session_from_disk(const std::string& session_id);
    
    void serialize_session(const std::shared_ptr<ProductionSession>& session);
    std::shared_ptr<ProductionSession> deserialize_session(const std::string& session_id);

    double get_current_time() const;

    std::string storage_path_;
    int max_resident_sessions_;
    KVRuntimeManager* kv_manager_;
    
    std::unordered_map<std::string, std::shared_ptr<ProductionSession>> active_sessions_;
    std::vector<std::string> resident_sessions_; // LRU list
    std::string current_active_session_id_;      // session currently loaded in KVRuntimeManager
    
    std::mutex mutex_;
};

} // namespace diffkv
