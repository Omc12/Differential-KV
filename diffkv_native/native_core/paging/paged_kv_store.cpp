#include "native_core/paging/paged_kv_store.hpp"
#include "native_core/streaming_sparse_ingest.hpp"
#include "native_core/srl/session_srl_state.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace diffkv {

PagedKVStore::PagedKVStore(size_t gpu_budget_bytes)
    : gpu_budget_bytes_(gpu_budget_bytes) {
    stats_.current_gpu_bytes = 0;
}

PagedKVStore::~PagedKVStore() {
    clear();
}

double PagedKVStore::get_current_time() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

size_t PagedKVStore::estimate_slot_vram(int rank, int head_dim, int kv_heads, int desc_dim) const {
    size_t total = 0;
    const int S_max = 64;
    total += rank * S_max * sizeof(int8_t);                   // U
    total += sizeof(ggml_fp16_t);                             // U_scale
    total += head_dim * kv_heads * rank * sizeof(ggml_fp16_t); // VK
    total += head_dim * kv_heads * rank * sizeof(ggml_fp16_t); // VV
    total += head_dim * kv_heads * sizeof(ggml_fp16_t);       // anchors_K
    total += head_dim * kv_heads * sizeof(ggml_fp16_t);       // anchors_V
    total += sizeof(int32_t);                                 // seq_lens
    total += sizeof(ggml_fp16_t);                             // scales
    total += desc_dim * sizeof(float);                        // desc_matrix
    total += sizeof(int32_t);                                 // anchor_positions
    return total;
}

void PagedKVStore::register_block(StreamingKVBlock* block, const std::vector<std::unique_ptr<NativeBlockPool>>& engines) {
    if (!block || block->pool_idx == -1 || engines.empty()) return;
    std::string key = std::to_string(block->anchor_idx);
    
    // Query parameters from first layer engine
    int rank = engines[0]->get_U()->ne[0];
    int head_dim = engines[0]->get_VK()->ne[0];
    int kv_heads = engines[0]->get_VK()->ne[1];
    int desc_dim = engines[0]->get_desc_matrix()->ne[0];

    // Total VRAM across all layers
    size_t single_layer_vram = estimate_slot_vram(rank, head_dim, kv_heads, desc_dim);
    size_t total_vram = single_layer_vram * engines.size();

    std::lock_guard<std::mutex> guard(lock_);
    if (entries_.find(key) != entries_.end()) {
        return;
    }

    PageEntry entry;
    entry.block_ref = block;
    entry.residency = BlockState::CompressedResident;
    entry.last_access = get_current_time();
    entry.vram_bytes = total_vram;
    entry.layers_cpu_data.resize(engines.size());
    
    entries_[key] = entry;
    stats_.current_gpu_bytes += total_vram;
}

void PagedKVStore::touch(StreamingKVBlock* block, const std::vector<std::unique_ptr<NativeBlockPool>>& engines) {
    if (!block) return;
    std::string key = std::to_string(block->anchor_idx);
    
    std::lock_guard<std::mutex> guard(lock_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return;
    }

    it->second.last_access = get_current_time();
    if (it->second.residency == BlockState::CPUResident) {
        reload_block(it->second, engines);
    }
}

void PagedKVStore::maybe_evict(const std::vector<std::unique_ptr<NativeBlockPool>>& engines, const SessionSRLState* srl_state) {
    std::lock_guard<std::mutex> guard(lock_);
    
    while (stats_.current_gpu_bytes > gpu_budget_bytes_) {
        // Find coldest GPU resident block
        std::string coldest_key = "";
        double oldest_time = 1e30;
        
        for (auto & pair : entries_) {
            if (pair.second.residency == BlockState::CompressedResident && pair.second.vram_bytes > 0) {
                double comp_time = pair.second.last_access;
                if (srl_state != nullptr) {
                    int slot_id = -1;
                    if (pair.second.block_ref != nullptr) {
                        slot_id = pair.second.block_ref->pool_idx;
                    }
                    if (slot_id != -1) {
                        float strength = 1.0f;
                        auto it = srl_state->slot_activation_strength.find(slot_id);
                        if (it != srl_state->slot_activation_strength.end()) {
                            strength = it->second;
                        }
                        // Boost time by (strength - 1.0) * 300.0 (5 minutes of virtual activity per unit strength)
                        double boost = (strength - 1.0) * 300.0;
                        comp_time += boost;
                    }
                }

                if (comp_time < oldest_time) {
                    oldest_time = comp_time;
                    coldest_key = pair.first;
                }
            }
        }

        if (coldest_key.empty()) {
            break;
        }

        auto & entry = entries_[coldest_key];
        evict_block(entry, engines);
    }
}

void PagedKVStore::evict_block(PageEntry& entry, const std::vector<std::unique_ptr<NativeBlockPool>>& engines) {
    StreamingKVBlock* block = entry.block_ref;
    if (!block || block->pool_idx == -1 || engines.empty()) return;
    int slot_id = block->pool_idx;

    int rank = engines[0]->get_U()->ne[0];
    int head_dim = engines[0]->get_VK()->ne[0];
    int kv_heads = engines[0]->get_VK()->ne[1];
    int desc_dim = engines[0]->get_desc_matrix()->ne[0];

    // Transition and copy all layers
    for (int l = 0; l < (int)engines.size(); ++l) {
        auto & engine = engines[l];
        engine->get_state_table().transition(slot_id, BlockState::CompressedResident, BlockState::PagingOut);

        PagedSlotData & cpu = entry.layers_cpu_data[l];
        cpu.U.resize(rank * 64);
        cpu.VK.resize(head_dim * kv_heads * rank);
        cpu.VV.resize(head_dim * kv_heads * rank);
        cpu.anchors_K.resize(head_dim * kv_heads);
        cpu.anchors_V.resize(head_dim * kv_heads);
        cpu.desc_matrix.resize(desc_dim);

        ggml_backend_tensor_get(engine->get_U(), cpu.U.data(), slot_id * engine->get_U()->nb[2], cpu.U.size() * sizeof(int8_t));
        ggml_backend_tensor_get(engine->get_U_scale(), &cpu.U_scale, slot_id * engine->get_U_scale()->nb[0], sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(engine->get_VK(), cpu.VK.data(), slot_id * engine->get_VK()->nb[3], cpu.VK.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(engine->get_VV(), cpu.VV.data(), slot_id * engine->get_VV()->nb[3], cpu.VV.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(engine->get_anchors_K(), cpu.anchors_K.data(), slot_id * engine->get_anchors_K()->nb[2], cpu.anchors_K.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(engine->get_anchors_V(), cpu.anchors_V.data(), slot_id * engine->get_anchors_V()->nb[2], cpu.anchors_V.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(engine->get_seq_lens(), &cpu.seq_len, slot_id * engine->get_seq_lens()->nb[0], sizeof(int32_t));
        ggml_backend_tensor_get(engine->get_scales(), &cpu.scale, slot_id * engine->get_scales()->nb[0], sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(engine->get_desc_matrix(), cpu.desc_matrix.data(), slot_id * engine->get_desc_matrix()->nb[1], cpu.desc_matrix.size() * sizeof(float));
        ggml_backend_tensor_get(engine->get_anchor_positions(), &cpu.anchor_position, slot_id * engine->get_anchor_positions()->nb[0], sizeof(int32_t));

        engine->get_state_table().transition(slot_id, BlockState::PagingOut, BlockState::CPUResident);
        engine->free_slot(slot_id);
    }

    stats_.evictions++;
    stats_.bytes_paged_out += entry.vram_bytes;
    stats_.current_gpu_bytes -= entry.vram_bytes;
    
    block->pool_idx = -1;
    block->state = BlockState::CPUResident;
    block->device_synced = false;
    entry.residency = BlockState::CPUResident;
}

void PagedKVStore::reload_block(PageEntry& entry, const std::vector<std::unique_ptr<NativeBlockPool>>& engines) {
    StreamingKVBlock* block = entry.block_ref;
    if (!block || block->pool_idx != -1 || engines.empty()) return;

    // Allocate physical slot ID aligned across all layers
    int slot_id = engines[0]->allocate_slot();
    if (slot_id == -1) {
        std::cerr << "[PagedKVStore] Error: Failed to allocate physical slot for reload!" << std::endl;
        return;
    }

    // Allocate that same slot ID in all other engines
    for (size_t l = 1; l < engines.size(); ++l) {
        int check_id = engines[l]->allocate_slot();
        if (check_id != slot_id) {
            std::cerr << "[PagedKVStore] FATAL: Slot allocation desynchronized across layers! Got: " 
                      << check_id << ", expected: " << slot_id << std::endl;
        }
    }

    block->pool_idx = slot_id;
    block->state = BlockState::CPUResident;

    for (int l = 0; l < (int)engines.size(); ++l) {
        auto & engine = engines[l];
        engine->get_state_table().transition(slot_id, BlockState::CPUResident, BlockState::Reloading);

        PagedSlotData & cpu = entry.layers_cpu_data[l];
        int rank = engine->get_U()->ne[0];
        int head_dim = engine->get_VK()->ne[0];
        int kv_heads = engine->get_VK()->ne[1];

        // Update host mirrors (CUDA compatible)
        std::memcpy(engine->get_host_U() + slot_id * rank * 64, cpu.U.data(), cpu.U.size() * sizeof(int8_t));
        *(engine->get_host_U_scale() + slot_id) = cpu.U_scale;
        std::memcpy(engine->get_host_VK() + slot_id * head_dim * kv_heads * rank, cpu.VK.data(), cpu.VK.size() * sizeof(ggml_fp16_t));
        std::memcpy(engine->get_host_VV() + slot_id * head_dim * kv_heads * rank, cpu.VV.data(), cpu.VV.size() * sizeof(ggml_fp16_t));
        std::memcpy(engine->get_host_anchors_K() + slot_id * head_dim * kv_heads, cpu.anchors_K.data(), cpu.anchors_K.size() * sizeof(ggml_fp16_t));
        std::memcpy(engine->get_host_anchors_V() + slot_id * head_dim * kv_heads, cpu.anchors_V.data(), cpu.anchors_V.size() * sizeof(ggml_fp16_t));
        *(engine->get_host_seq_lens() + slot_id) = cpu.seq_len;
        *(engine->get_host_scales() + slot_id) = cpu.scale;
        *(engine->get_host_anchor_positions() + slot_id) = cpu.anchor_position;

        ggml_backend_tensor_set(engine->get_U(), cpu.U.data(), slot_id * engine->get_U()->nb[2], cpu.U.size() * sizeof(int8_t));
        ggml_backend_tensor_set(engine->get_U_scale(), &cpu.U_scale, slot_id * engine->get_U_scale()->nb[0], sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine->get_VK(), cpu.VK.data(), slot_id * engine->get_VK()->nb[3], cpu.VK.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine->get_VV(), cpu.VV.data(), slot_id * engine->get_VV()->nb[3], cpu.VV.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine->get_anchors_K(), cpu.anchors_K.data(), slot_id * engine->get_anchors_K()->nb[2], cpu.anchors_K.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine->get_anchors_V(), cpu.anchors_V.data(), slot_id * engine->get_anchors_V()->nb[2], cpu.anchors_V.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine->get_seq_lens(), &cpu.seq_len, slot_id * engine->get_seq_lens()->nb[0], sizeof(int32_t));
        ggml_backend_tensor_set(engine->get_scales(), &cpu.scale, slot_id * engine->get_scales()->nb[0], sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine->get_desc_matrix(), cpu.desc_matrix.data(), slot_id * engine->get_desc_matrix()->nb[1], cpu.desc_matrix.size() * sizeof(float));
        ggml_backend_tensor_set(engine->get_anchor_positions(), &cpu.anchor_position, slot_id * engine->get_anchor_positions()->nb[0], sizeof(int32_t));

        engine->get_state_table().transition(slot_id, BlockState::Reloading, BlockState::CompressedResident);
    }

    stats_.reloads++;
    stats_.bytes_paged_in += entry.vram_bytes;
    stats_.current_gpu_bytes += entry.vram_bytes;
    block->state = BlockState::CompressedResident;
    block->device_synced = false;
    entry.residency = BlockState::CompressedResident;
}

void PagedKVStore::clear() {
    std::lock_guard<std::mutex> guard(lock_);
    entries_.clear();
    stats_ = Stats();
}

PagedKVStore::Stats PagedKVStore::get_stats() const {
    std::lock_guard<std::mutex> guard(lock_);
    return stats_;
}

std::string PagedKVStore::summary() const {
    std::lock_guard<std::mutex> guard(lock_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "{gpu_resident_mb: " << (stats_.current_gpu_bytes / 1e6)
        << ", total_evictions: " << stats_.evictions
        << ", total_reloads: " << stats_.reloads
        << ", bytes_paged_out_mb: " << (stats_.bytes_paged_out / 1e6)
        << ", bytes_paged_in_mb: " << (stats_.bytes_paged_in / 1e6)
        << ", tracked_blocks: " << entries_.size() << "}";
    return oss.str();
}

void PagedKVStore::evict_all(const std::vector<std::unique_ptr<NativeBlockPool>>& engines) {
    std::lock_guard<std::mutex> guard(lock_);
    for (auto & pair : entries_) {
        if (pair.second.residency == BlockState::CompressedResident) {
            evict_block(pair.second, engines);
        }
    }
}

void PagedKVStore::reload_all(const std::vector<std::unique_ptr<NativeBlockPool>>& engines) {
    std::lock_guard<std::mutex> guard(lock_);
    for (auto & pair : entries_) {
        if (pair.second.residency == BlockState::CPUResident) {
            reload_block(pair.second, engines);
        }
    }
}

void PagedKVStore::swap_state(std::unordered_map<std::string, PageEntry>& entries, Stats& stats) {
    std::lock_guard<std::mutex> guard(lock_);
    std::swap(entries_, entries);
    std::swap(stats_, stats);
}

} // namespace diffkv
