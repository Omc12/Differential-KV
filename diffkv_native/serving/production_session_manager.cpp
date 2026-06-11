#include "serving/production_session_manager.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)
#else
#include <unistd.h>
#endif

namespace diffkv {

static std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 4; i++) ss << dis(gen);
    ss << "-4"; // UUID version 4
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    ss << dis2(gen);
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 12; i++) ss << dis(gen);
    return ss.str();
}

static bool directory_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

static bool create_directory(const std::string& path) {
    if (directory_exists(path)) return true;
    return mkdir(path.c_str(), 0777) == 0;
}

// Helper serialization functions
template<typename T>
static void write_val(std::ostream& os, const T& val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template<typename T>
static void read_val(std::istream& is, T& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
}

template<typename T>
static void write_vec(std::ostream& os, const std::vector<T>& vec) {
    size_t size = vec.size();
    write_val(os, size);
    if (size > 0) {
        os.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
    }
}

template<typename T>
static void read_vec(std::istream& is, std::vector<T>& vec) {
    size_t size = 0;
    read_val(is, size);
    vec.resize(size);
    if (size > 0) {
        is.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
    }
}

static void serialize_metadata(const std::shared_ptr<ProductionSession>& session, std::ostream& os) {
    os << "{\n";
    os << "  \"session_id\": \"" << session->session_id << "\",\n";
    os << "  \"created_at\": " << std::fixed << std::setprecision(6) << session->created_at << ",\n";
    os << "  \"last_accessed\": " << session->last_accessed << ",\n";
    os << "  \"status\": \"" << session->status << "\",\n";
    os << "  \"config\": {\n";
    bool first = true;
    for (auto const& [key, val] : session->config) {
        if (!first) os << ",\n";
        os << "    \"" << key << "\": \"" << val << "\"";
        first = false;
    }
    os << "\n  },\n";
    os << "  \"token_ids\": [";
    for (size_t i = 0; i < session->token_ids.size(); ++i) {
        if (i > 0) os << ", ";
        os << session->token_ids[i];
    }
    os << "],\n";
    os << "  \"history\": [\n";
    for (size_t i = 0; i < session->history.size(); ++i) {
        if (i > 0) os << ",\n";
        os << "    {\n";
        std::string role = session->history[i].role;
        std::string content = session->history[i].content;
        auto escape = [](const std::string& s) {
            std::string res;
            for (char c : s) {
                if (c == '"') res += "\\\"";
                else if (c == '\\') res += "\\\\";
                else if (c == '\n') res += "\\n";
                else if (c == '\r') res += "\\r";
                else if (c == '\t') res += "\\t";
                else res += c;
            }
            return res;
        };
        os << "      \"role\": \"" << escape(role) << "\",\n";
        os << "      \"content\": \"" << escape(content) << "\"\n";
        os << "    }";
    }
    os << "\n  ]\n";
    os << "}\n";
}

static std::shared_ptr<ProductionSession> parse_metadata(std::istream& is) {
    auto session = std::make_shared<ProductionSession>();
    std::string line;
    bool in_config = false;
    bool in_history = false;
    ChatMessage current_msg;
    
    auto unescape = [](const std::string& s) {
        std::string res;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char next = s[i+1];
                if (next == 'n') res += '\n';
                else if (next == 'r') res += '\r';
                else if (next == 't') res += '\t';
                else if (next == '"') res += '"';
                else if (next == '\\') res += '\\';
                else res += next;
                i++;
            } else {
                res += s[i];
            }
        }
        return res;
    };

    while (std::getline(is, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;

        if (line == "}" || line == "},") {
            in_config = false;
            continue;
        }
        if (line == "]" || line == "],") {
            in_history = false;
            continue;
        }

        if (line.rfind("\"config\": {", 0) == 0) {
            in_config = true;
            continue;
        }
        if (line.rfind("\"history\": [", 0) == 0) {
            in_history = true;
            continue;
        }
        if (line.rfind("\"token_ids\": [", 0) == 0) {
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != std::string::npos && end != std::string::npos && end > start + 1) {
                std::string vals = line.substr(start + 1, end - start - 1);
                std::stringstream ss(vals);
                std::string val;
                while (std::getline(ss, val, ',')) {
                    val.erase(0, val.find_first_not_of(" \t"));
                    val.erase(val.find_last_not_of(" \t") + 1);
                    if (!val.empty()) {
                        session->token_ids.push_back(std::stoi(val));
                    }
                }
            }
            continue;
        }

        if (in_config) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string key = line.substr(0, colon);
                std::string val = line.substr(colon + 1);
                key.erase(0, key.find_first_not_of("\" \t"));
                key.erase(key.find_last_not_of("\" \t") + 1);
                val.erase(0, val.find_first_not_of("\" \t"));
                val.erase(val.find_last_not_of("\", \t") + 1);
                session->config[key] = val;
            }
            continue;
        }

        if (in_history) {
            if (line == "{" || line == "{,") {
                current_msg = ChatMessage();
                continue;
            }
            if (line == "}" || line == "},") {
                session->history.push_back(current_msg);
                continue;
            }
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string key = line.substr(0, colon);
                std::string val = line.substr(colon + 1);
                key.erase(0, key.find_first_not_of("\" \t"));
                key.erase(key.find_last_not_of("\" \t") + 1);
                val.erase(0, val.find_first_not_of("\" \t"));
                val.erase(val.find_last_not_of("\", \t") + 1);
                if (key == "role") {
                    current_msg.role = unescape(val);
                } else if (key == "content") {
                    current_msg.content = unescape(val);
                }
            }
            continue;
        }

        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            key.erase(0, key.find_first_not_of("\" \t"));
            key.erase(key.find_last_not_of("\" \t") + 1);
            val.erase(0, val.find_first_not_of("\" \t"));
            val.erase(val.find_last_not_of("\", \t") + 1);
            if (key == "session_id") {
                session->session_id = val;
            } else if (key == "created_at") {
                session->created_at = std::stod(val);
            } else if (key == "last_accessed") {
                session->last_accessed = std::stod(val);
            } else if (key == "status") {
                session->status = val;
            }
        }
    }
    return session;
}

static void serialize_binary(const std::shared_ptr<ProductionSession>& session, std::ostream& os) {
    size_t num_layers = session->layers_blocks.size();
    write_val(os, num_layers);
    
    for (size_t l = 0; l < num_layers; ++l) {
        size_t num_blocks = session->layers_blocks[l].size();
        write_val(os, num_blocks);
        for (size_t b = 0; b < num_blocks; ++b) {
            const auto& block = session->layers_blocks[l][b];
            write_val(os, block->anchor_idx);
            write_val(os, block->micro_block_size);
            write_val(os, block->pool_idx);
            
            uint8_t state_val = static_cast<uint8_t>(block->state);
            write_val(os, state_val);
            write_val(os, block->is_outlier);
            write_val(os, block->skip_compression);
            
            write_vec(os, block->token_indices);
            write_vec(os, block->active_k);
            write_vec(os, block->active_v);
            write_vec(os, block->anchor_k);
            write_vec(os, block->anchor_v);
            write_vec(os, block->svd_k);
            write_vec(os, block->svd_v);
        }
    }
    
    size_t num_entries = session->pager_entries.size();
    write_val(os, num_entries);
    for (const auto& [key, entry] : session->pager_entries) {
        size_t key_len = key.size();
        write_val(os, key_len);
        os.write(key.data(), key_len);
        
        int layer_idx = -1;
        int block_idx = -1;
        if (entry.block_ref) {
            for (size_t l = 0; l < num_layers; ++l) {
                for (size_t b = 0; b < session->layers_blocks[l].size(); ++b) {
                    if (session->layers_blocks[l][b].get() == entry.block_ref) {
                        layer_idx = static_cast<int>(l);
                        block_idx = static_cast<int>(b);
                        break;
                    }
                }
                if (layer_idx != -1) break;
            }
        }
        write_val(os, layer_idx);
        write_val(os, block_idx);
        
        uint8_t residency_val = static_cast<uint8_t>(entry.residency);
        write_val(os, residency_val);
        write_val(os, entry.last_access);
        write_val(os, entry.vram_bytes);
        
        size_t cpu_data_size = entry.layers_cpu_data.size();
        write_val(os, cpu_data_size);
        for (size_t l = 0; l < cpu_data_size; ++l) {
            const auto& data = entry.layers_cpu_data[l];
            write_vec(os, data.U);
            write_val(os, data.U_scale);
            write_vec(os, data.VK);
            write_vec(os, data.VV);
            write_vec(os, data.anchors_K);
            write_vec(os, data.anchors_V);
            write_val(os, data.seq_len);
            write_val(os, data.scale);
            write_vec(os, data.desc_matrix);
            write_val(os, data.anchor_position);
        }
    }
}

static void deserialize_binary(const std::shared_ptr<ProductionSession>& session, std::istream& is) {
    size_t num_layers = 0;
    read_val(is, num_layers);
    session->layers_blocks.resize(num_layers);
    
    for (size_t l = 0; l < num_layers; ++l) {
        size_t num_blocks = 0;
        read_val(is, num_blocks);
        session->layers_blocks[l].resize(num_blocks);
        for (size_t b = 0; b < num_blocks; ++b) {
            auto block = std::make_unique<StreamingKVBlock>();
            read_val(is, block->anchor_idx);
            read_val(is, block->micro_block_size);
            read_val(is, block->pool_idx);
            
            uint8_t state_val = 0;
            read_val(is, state_val);
            block->state = static_cast<BlockState>(state_val);
            read_val(is, block->is_outlier);
            read_val(is, block->skip_compression);
            
            read_vec(is, block->token_indices);
            read_vec(is, block->active_k);
            read_vec(is, block->active_v);
            read_vec(is, block->anchor_k);
            read_vec(is, block->anchor_v);
            read_vec(is, block->svd_k);
            read_vec(is, block->svd_v);
            
            session->layers_blocks[l][b] = std::move(block);
        }
    }
    
    size_t num_entries = 0;
    read_val(is, num_entries);
    for (size_t i = 0; i < num_entries; ++i) {
        size_t key_len = 0;
        read_val(is, key_len);
        std::string key(key_len, '\0');
        is.read(&key[0], key_len);
        
        int layer_idx = -1;
        int block_idx = -1;
        read_val(is, layer_idx);
        read_val(is, block_idx);
        
        PageEntry entry;
        if (layer_idx >= 0 && layer_idx < static_cast<int>(num_layers) &&
            block_idx >= 0 && block_idx < static_cast<int>(session->layers_blocks[layer_idx].size())) {
            entry.block_ref = session->layers_blocks[layer_idx][block_idx].get();
        } else {
            entry.block_ref = nullptr;
        }
        
        uint8_t residency_val = 0;
        read_val(is, residency_val);
        entry.residency = static_cast<BlockState>(residency_val);
        read_val(is, entry.last_access);
        read_val(is, entry.vram_bytes);
        
        size_t cpu_data_size = 0;
        read_val(is, cpu_data_size);
        entry.layers_cpu_data.resize(cpu_data_size);
        for (size_t l = 0; l < cpu_data_size; ++l) {
            auto& data = entry.layers_cpu_data[l];
            read_vec(is, data.U);
            read_val(is, data.U_scale);
            read_vec(is, data.VK);
            read_vec(is, data.VV);
            read_vec(is, data.anchors_K);
            read_vec(is, data.anchors_V);
            read_val(is, data.seq_len);
            read_val(is, data.scale);
            read_vec(is, data.desc_matrix);
            read_val(is, data.anchor_position);
        }
        
        session->pager_entries[key] = entry;
    }
}

// -----------------------------------------------------------------------------
// SharedPrefixManager implementation
// -----------------------------------------------------------------------------

SharedPrefixManager::SharedPrefixManager(KVRuntimeManager* kv_manager)
    : kv_manager_(kv_manager) {}

SharedPrefixManager::~SharedPrefixManager() {}

void SharedPrefixManager::register_session_prefix(
    const std::string& session_id,
    const std::vector<int32_t>& prefix_tokens,
    const std::vector<int>& pool_indices,
    const std::vector<int>& anchor_indices
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shared_prefixes_.find(prefix_tokens) == shared_prefixes_.end()) {
        PrefixData data;
        data.pool_indices = pool_indices;
        data.anchor_indices = anchor_indices;
        data.ref_count = 0;
        shared_prefixes_[prefix_tokens] = data;
    }
    shared_prefixes_[prefix_tokens].ref_count++;
    session_prefixes_[session_id].push_back(prefix_tokens);
}

void SharedPrefixManager::release_session_prefixes(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = session_prefixes_.find(session_id);
    if (it == session_prefixes_.end()) return;
    
    for (const auto& prefix_key : it->second) {
        auto shared_it = shared_prefixes_.find(prefix_key);
        if (shared_it != shared_prefixes_.end()) {
            shared_it->second.ref_count--;
            if (shared_it->second.ref_count <= 0) {
                if (kv_manager_) {
                    auto& engines = kv_manager_->get_engines();
                    for (int pool_idx : shared_it->second.pool_indices) {
                        for (auto& engine : engines) {
                            try {
                                engine->free_slot(pool_idx);
                            } catch (...) {}
                        }
                    }
                }
                shared_prefixes_.erase(shared_it);
            }
        }
    }
    session_prefixes_.erase(it);
}

SharedPrefixManager::PrefixData* SharedPrefixManager::lookup_prefix(const std::vector<int32_t>& prefix_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = shared_prefixes_.find(prefix_tokens);
    if (it != shared_prefixes_.end()) {
        return &it->second;
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// ProductionSessionManager implementation
// -----------------------------------------------------------------------------

ProductionSessionManager::ProductionSessionManager(
    const std::string& storage_path,
    int max_resident_sessions,
    KVRuntimeManager* kv_manager
) : storage_path_(storage_path),
    max_resident_sessions_(max_resident_sessions),
    kv_manager_(kv_manager) {
    
    if (max_resident_sessions_ <= 0) {
        const char* env_max = std::getenv("DIFFKV_MAX_SESSIONS");
        if (env_max) {
            max_resident_sessions_ = std::stoi(env_max);
        } else {
            max_resident_sessions_ = 4;
        }
    }
    create_directory(storage_path_);
}

ProductionSessionManager::~ProductionSessionManager() {}

double ProductionSessionManager::get_current_time() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

std::string ProductionSessionManager::create_session(const std::unordered_map<std::string, std::string>& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string session_id = generate_uuid();
    return create_session_with_id_locked(session_id, config);
}

std::string ProductionSessionManager::create_session_with_id(const std::string& session_id, const std::unordered_map<std::string, std::string>& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    return create_session_with_id_locked(session_id, config);
}

std::string ProductionSessionManager::create_session_with_id_locked(const std::string& session_id, const std::unordered_map<std::string, std::string>& config) {
    auto session = std::make_shared<ProductionSession>();
    session->session_id = session_id;
    session->created_at = get_current_time();
    session->last_accessed = get_current_time();
    session->config = config;
    session->status = "active";
    
    active_sessions_[session_id] = session;
    
    // Ensure metadata vectors are initialized to num_layers if we have a kv_manager
    if (kv_manager_) {
        int n_layers = kv_manager_->get_engines().size();
        session->layers_blocks.resize(n_layers);
    }

    // Call internal ensure residency (assumes mutex_ is held)
    if (std::find(resident_sessions_.begin(), resident_sessions_.end(), session_id) == resident_sessions_.end()) {
        if ((int)resident_sessions_.size() >= max_resident_sessions_) {
            std::string evicted_id = resident_sessions_.front();
            resident_sessions_.erase(resident_sessions_.begin());
            
            // Inline evict
            bool was_active = (current_active_session_id_ == evicted_id);
            if (was_active) {
                // Swap active out
                if (kv_manager_) {
                    auto prev_session = active_sessions_[current_active_session_id_];
                    if (prev_session) {
                        kv_manager_->get_ingest_manager().swap_blocks(prev_session->layers_blocks);
                        kv_manager_->get_pager().swap_state(prev_session->pager_entries, prev_session->pager_stats);
                    }
                }
                current_active_session_id_ = "";
            }
            
            auto evicted_sess = active_sessions_[evicted_id];
            if (evicted_sess && kv_manager_) {
                kv_manager_->get_ingest_manager().swap_blocks(evicted_sess->layers_blocks);
                kv_manager_->get_pager().swap_state(evicted_sess->pager_entries, evicted_sess->pager_stats);
                kv_manager_->get_pager().evict_all(kv_manager_->get_engines());
                kv_manager_->get_ingest_manager().swap_blocks(evicted_sess->layers_blocks);
                kv_manager_->get_pager().swap_state(evicted_sess->pager_entries, evicted_sess->pager_stats);
            }
            std::cout << "[PSM] Evicted session " << evicted_id << " to CPU RAM." << std::endl;
        }
        
        resident_sessions_.push_back(session_id);
        
        // Inline load
        bool was_active = (current_active_session_id_ == session_id);
        if (was_active) {
            if (kv_manager_) {
                auto prev_session = active_sessions_[current_active_session_id_];
                if (prev_session) {
                    kv_manager_->get_ingest_manager().swap_blocks(prev_session->layers_blocks);
                    kv_manager_->get_pager().swap_state(prev_session->pager_entries, prev_session->pager_stats);
                }
            }
            current_active_session_id_ = "";
        }
        
        if (kv_manager_) {
            kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
            kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
            kv_manager_->get_pager().reload_all(kv_manager_->get_engines());
            kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
            kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
        }
        std::cout << "[PSM] Loaded session " << session_id << " to VRAM." << std::endl;
    } else {
        auto it = std::find(resident_sessions_.begin(), resident_sessions_.end(), session_id);
        if (it != resident_sessions_.end() && resident_sessions_.back() != session_id) {
            resident_sessions_.erase(it);
            resident_sessions_.push_back(session_id);
        }
    }
    
    return session_id;
}

std::shared_ptr<ProductionSession> ProductionSessionManager::get_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_sessions_.find(session_id);
    if (it != active_sessions_.end()) {
        it->second->last_accessed = get_current_time();
        
        // Inline ensure residency
        if (std::find(resident_sessions_.begin(), resident_sessions_.end(), session_id) == resident_sessions_.end()) {
            if ((int)resident_sessions_.size() >= max_resident_sessions_) {
                std::string evicted_id = resident_sessions_.front();
                resident_sessions_.erase(resident_sessions_.begin());
                
                // Evict evicted_id
                bool was_active = (current_active_session_id_ == evicted_id);
                if (was_active) {
                    if (kv_manager_) {
                        auto prev_session = active_sessions_[current_active_session_id_];
                        if (prev_session) {
                            kv_manager_->get_ingest_manager().swap_blocks(prev_session->layers_blocks);
                            kv_manager_->get_pager().swap_state(prev_session->pager_entries, prev_session->pager_stats);
                        }
                    }
                    current_active_session_id_ = "";
                }
                
                auto evicted_sess = active_sessions_[evicted_id];
                if (evicted_sess && kv_manager_) {
                    kv_manager_->get_ingest_manager().swap_blocks(evicted_sess->layers_blocks);
                    kv_manager_->get_pager().swap_state(evicted_sess->pager_entries, evicted_sess->pager_stats);
                    kv_manager_->get_pager().evict_all(kv_manager_->get_engines());
                    kv_manager_->get_ingest_manager().swap_blocks(evicted_sess->layers_blocks);
                    kv_manager_->get_pager().swap_state(evicted_sess->pager_entries, evicted_sess->pager_stats);
                }
                std::cout << "[PSM] Evicted session " << evicted_id << " to CPU RAM." << std::endl;
            }
            
            resident_sessions_.push_back(session_id);
            
            // Load session_id into vram
            if (kv_manager_) {
                kv_manager_->get_ingest_manager().swap_blocks(it->second->layers_blocks);
                kv_manager_->get_pager().swap_state(it->second->pager_entries, it->second->pager_stats);
                kv_manager_->get_pager().reload_all(kv_manager_->get_engines());
                kv_manager_->get_ingest_manager().swap_blocks(it->second->layers_blocks);
                kv_manager_->get_pager().swap_state(it->second->pager_entries, it->second->pager_stats);
            }
            std::cout << "[PSM] Loaded session " << session_id << " to VRAM." << std::endl;
        } else {
            auto res_it = std::find(resident_sessions_.begin(), resident_sessions_.end(), session_id);
            if (res_it != resident_sessions_.end() && resident_sessions_.back() != session_id) {
                resident_sessions_.erase(res_it);
                resident_sessions_.push_back(session_id);
            }
        }
        
        return it->second;
    }
    
    // Try loading from disk
    return load_session_from_disk(session_id);
}

void ProductionSessionManager::ensure_residency(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) return;
    
    it->second->last_accessed = get_current_time();
    
    if (std::find(resident_sessions_.begin(), resident_sessions_.end(), session_id) == resident_sessions_.end()) {
        if ((int)resident_sessions_.size() >= max_resident_sessions_) {
            std::string evicted_id = resident_sessions_.front();
            resident_sessions_.erase(resident_sessions_.begin());
            
            bool was_active = (current_active_session_id_ == evicted_id);
            if (was_active) {
                if (kv_manager_) {
                    auto prev_session = active_sessions_[current_active_session_id_];
                    if (prev_session) {
                        kv_manager_->get_ingest_manager().swap_blocks(prev_session->layers_blocks);
                        kv_manager_->get_pager().swap_state(prev_session->pager_entries, prev_session->pager_stats);
                    }
                }
                current_active_session_id_ = "";
            }
            
            auto evicted_sess = active_sessions_[evicted_id];
            if (evicted_sess && kv_manager_) {
                kv_manager_->get_ingest_manager().swap_blocks(evicted_sess->layers_blocks);
                kv_manager_->get_pager().swap_state(evicted_sess->pager_entries, evicted_sess->pager_stats);
                kv_manager_->get_pager().evict_all(kv_manager_->get_engines());
                kv_manager_->get_ingest_manager().swap_blocks(evicted_sess->layers_blocks);
                kv_manager_->get_pager().swap_state(evicted_sess->pager_entries, evicted_sess->pager_stats);
            }
            std::cout << "[PSM] Evicted session " << evicted_id << " to CPU RAM." << std::endl;
        }
        
        resident_sessions_.push_back(session_id);
        
        if (kv_manager_) {
            kv_manager_->get_ingest_manager().swap_blocks(it->second->layers_blocks);
            kv_manager_->get_pager().swap_state(it->second->pager_entries, it->second->pager_stats);
            kv_manager_->get_pager().reload_all(kv_manager_->get_engines());
            kv_manager_->get_ingest_manager().swap_blocks(it->second->layers_blocks);
            kv_manager_->get_pager().swap_state(it->second->pager_entries, it->second->pager_stats);
        }
        std::cout << "[PSM] Loaded session " << session_id << " to VRAM." << std::endl;
    } else {
        auto res_it = std::find(resident_sessions_.begin(), resident_sessions_.end(), session_id);
        if (res_it != resident_sessions_.end() && resident_sessions_.back() != session_id) {
            resident_sessions_.erase(res_it);
            resident_sessions_.push_back(session_id);
        }
    }

    // Now swap it in as the current active session in the KV manager!
    if (current_active_session_id_ != session_id) {
        if (!current_active_session_id_.empty() && kv_manager_) {
            auto prev_session = active_sessions_[current_active_session_id_];
            if (prev_session) {
                kv_manager_->get_ingest_manager().swap_blocks(prev_session->layers_blocks);
                kv_manager_->get_pager().swap_state(prev_session->pager_entries, prev_session->pager_stats);
            }
        }
        
        if (kv_manager_) {
            kv_manager_->get_ingest_manager().swap_blocks(it->second->layers_blocks);
            kv_manager_->get_pager().swap_state(it->second->pager_entries, it->second->pager_stats);
        }
        current_active_session_id_ = session_id;
    }
}

void ProductionSessionManager::evict_from_vram(const std::string& session_id) {
    if (!kv_manager_) return;
    auto session = active_sessions_[session_id];
    if (!session) return;
    
    bool was_active = (current_active_session_id_ == session_id);
    if (was_active) {
        // Swap out
        kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
        kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
        current_active_session_id_ = "";
    }
    
    // Temporarily swap this session's state into the manager to run evict_all
    kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
    kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
    
    // Run evict_all (this frees GPU slots and copies data to CPU)
    kv_manager_->get_pager().evict_all(kv_manager_->get_engines());
    
    // Swap back
    kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
    kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
    
    std::cout << "[PSM] Evicted session " << session_id << " to CPU RAM." << std::endl;
}

void ProductionSessionManager::load_into_vram(const std::string& session_id) {
    if (!kv_manager_) return;
    auto session = active_sessions_[session_id];
    if (!session) return;
    
    bool was_active = (current_active_session_id_ == session_id);
    if (was_active) {
        kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
        kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
        current_active_session_id_ = "";
    }
    
    // Temporarily swap in
    kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
    kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
    
    // Reload all blocks back to GPU
    kv_manager_->get_pager().reload_all(kv_manager_->get_engines());
    
    // Swap back
    kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
    kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
    
    std::cout << "[PSM] Loaded session " << session_id << " back to VRAM." << std::endl;
}

void ProductionSessionManager::save_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_sessions_.find(session_id);
    if (it == active_sessions_.end()) {
        throw std::runtime_error("Session " + session_id + " not found.");
    }
    
    // If it is the current active session, we need to temporarily swap its blocks and pager state back into the session struct
    // so that the serialized copy is up to date!
    bool was_active = (current_active_session_id_ == session_id);
    if (was_active && kv_manager_) {
        kv_manager_->get_ingest_manager().swap_blocks(it->second->layers_blocks);
        kv_manager_->get_pager().swap_state(it->second->pager_entries, it->second->pager_stats);
    }
    
    serialize_session(it->second);
    
    // Swap back if it was active
    if (was_active && kv_manager_) {
        kv_manager_->get_ingest_manager().swap_blocks(it->second->layers_blocks);
        kv_manager_->get_pager().swap_state(it->second->pager_entries, it->second->pager_stats);
    }
}

std::shared_ptr<ProductionSession> ProductionSessionManager::load_session_from_disk(const std::string& session_id) {
    std::string meta_path = storage_path_ + "/" + session_id + "_meta.json";
    std::ifstream f(meta_path);
    if (!f.is_open()) return nullptr;
    
    auto session = deserialize_session(session_id);
    if (session) {
        active_sessions_[session_id] = session;
        
        // Ensure residency and load it back
        if (std::find(resident_sessions_.begin(), resident_sessions_.end(), session_id) == resident_sessions_.end()) {
            if ((int)resident_sessions_.size() >= max_resident_sessions_) {
                std::string evicted_id = resident_sessions_.front();
                resident_sessions_.erase(resident_sessions_.begin());
                
                bool was_active = (current_active_session_id_ == evicted_id);
                if (was_active) {
                    if (kv_manager_) {
                        auto prev_session = active_sessions_[current_active_session_id_];
                        if (prev_session) {
                            kv_manager_->get_ingest_manager().swap_blocks(prev_session->layers_blocks);
                            kv_manager_->get_pager().swap_state(prev_session->pager_entries, prev_session->pager_stats);
                        }
                    }
                    current_active_session_id_ = "";
                }
                
                auto evicted_sess = active_sessions_[evicted_id];
                if (evicted_sess && kv_manager_) {
                    kv_manager_->get_ingest_manager().swap_blocks(evicted_sess->layers_blocks);
                    kv_manager_->get_pager().swap_state(evicted_sess->pager_entries, evicted_sess->pager_stats);
                    kv_manager_->get_pager().evict_all(kv_manager_->get_engines());
                    kv_manager_->get_ingest_manager().swap_blocks(evicted_sess->layers_blocks);
                    kv_manager_->get_pager().swap_state(evicted_sess->pager_entries, evicted_sess->pager_stats);
                }
                std::cout << "[PSM] Evicted session " << evicted_id << " to CPU RAM." << std::endl;
            }
            
            resident_sessions_.push_back(session_id);
            
            if (kv_manager_) {
                kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
                kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
                kv_manager_->get_pager().reload_all(kv_manager_->get_engines());
                kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
                kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
            }
            std::cout << "[PSM] Loaded session " << session_id << " to VRAM." << std::endl;
        }
    }
    return session;
}

void ProductionSessionManager::delete_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clear history
    clear_history(session_id);
    
    // Remove from active list
    auto it = active_sessions_.find(session_id);
    if (it != active_sessions_.end()) {
        auto session = it->second;
        
        // Remove from resident list
        auto res_it = std::find(resident_sessions_.begin(), resident_sessions_.end(), session_id);
        if (res_it != resident_sessions_.end()) {
            resident_sessions_.erase(res_it);
        }
        
        bool was_active = (current_active_session_id_ == session_id);
        if (was_active) {
            // Swap out active state to clear it
            if (kv_manager_) {
                kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
                kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
            }
            current_active_session_id_ = "";
        }
        
        // Free its physical slots in the engine
        if (kv_manager_) {
            // Temporarily swap this session's state into the manager to run evict_all/free
            kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
            kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
            
            // Evicting all will free their slot IDs in the engines
            kv_manager_->get_pager().evict_all(kv_manager_->get_engines());
            
            // Clear the actual block list
            kv_manager_->get_ingest_manager().clear();
            kv_manager_->get_pager().clear();
            
            // Swap back the now empty/freed states
            kv_manager_->get_ingest_manager().swap_blocks(session->layers_blocks);
            kv_manager_->get_pager().swap_state(session->pager_entries, session->pager_stats);
        }
        
        active_sessions_.erase(it);
    }
    
    // Delete files from disk
    std::string pt_path = storage_path_ + "/" + session_id + ".bin";
    std::string meta_path = storage_path_ + "/" + session_id + "_meta.json";
    std::remove(pt_path.c_str());
    std::remove(meta_path.c_str());
}

void ProductionSessionManager::cleanup_idle_sessions(int idle_timeout_seconds) {
    double current_time = get_current_time();
    std::vector<std::string> to_delete;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [sid, session] : active_sessions_) {
            if (current_time - session->last_accessed > idle_timeout_seconds) {
                to_delete.push_back(sid);
            }
        }
    }
    
    for (const auto& sid : to_delete) {
        std::cout << "[PSM] Cleaning up idle session " << sid << " due to timeout." << std::endl;
        delete_session(sid);
    }
}

std::vector<std::string> ProductionSessionManager::list_sessions() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> sids;
    for (const auto& [sid, _] : active_sessions_) {
        sids.push_back(sid);
    }
    return sids;
}

// Conversation history management
std::vector<ChatMessage> ProductionSessionManager::get_history(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_sessions_.find(session_id);
    if (it != active_sessions_.end()) {
        return it->second->history;
    }
    return {};
}

void ProductionSessionManager::append_message(const std::string& session_id, const std::string& role, const std::string& content) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_sessions_.find(session_id);
    if (it != active_sessions_.end()) {
        it->second->history.push_back({role, content});
        it->second->last_accessed = get_current_time();
    }
}

void ProductionSessionManager::clear_history(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_sessions_.find(session_id);
    if (it != active_sessions_.end()) {
        it->second->history.clear();
        it->second->last_accessed = get_current_time();
    }
}

void ProductionSessionManager::serialize_session(const std::shared_ptr<ProductionSession>& session) {
    std::string meta_path = storage_path_ + "/" + session->session_id + "_meta.json";
    std::string bin_path = storage_path_ + "/" + session->session_id + ".bin";
    
    // Save metadata
    std::ofstream meta_file(meta_path);
    if (meta_file.is_open()) {
        serialize_metadata(session, meta_file);
    }
    
    // Save sparse state
    std::ofstream bin_file(bin_path, std::ios::binary);
    if (bin_file.is_open()) {
        serialize_binary(session, bin_file);
    }
}

std::shared_ptr<ProductionSession> ProductionSessionManager::deserialize_session(const std::string& session_id) {
    std::string meta_path = storage_path_ + "/" + session_id + "_meta.json";
    std::string bin_path = storage_path_ + "/" + session_id + ".bin";
    
    std::ifstream meta_file(meta_path);
    if (!meta_file.is_open()) return nullptr;
    
    auto session = parse_metadata(meta_file);
    if (!session) return nullptr;
    
    std::ifstream bin_file(bin_path, std::ios::binary);
    if (bin_file.is_open()) {
        deserialize_binary(session, bin_file);
    }
    return session;
}

} // namespace diffkv
