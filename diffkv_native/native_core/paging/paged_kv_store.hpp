#pragma once

#include "runtime/native_block_pool.hpp"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <string>
#include <memory>
#include <chrono>

namespace diffkv {

struct StreamingKVBlock; // Forward declaration

struct PagedSlotData {
    std::vector<int8_t> U;
    ggml_fp16_t U_scale;
    std::vector<ggml_fp16_t> VK;
    std::vector<ggml_fp16_t> VV;
    std::vector<ggml_fp16_t> anchors_K;
    std::vector<ggml_fp16_t> anchors_V;
    int32_t seq_len;
    ggml_fp16_t scale;
    std::vector<float> desc_matrix;
    int32_t anchor_position;
};

struct PageEntry {
    StreamingKVBlock* block_ref = nullptr;
    BlockState residency; // e.g. CompressedResident or CPUResident
    double last_access;   // System time in seconds
    size_t vram_bytes;
    std::vector<PagedSlotData> layers_cpu_data; // size = n_layers
};

struct SessionSRLState;

class PagedKVStore {
public:
    struct Stats {
        uint64_t evictions = 0;
        uint64_t reloads = 0;
        uint64_t bytes_paged_out = 0;
        uint64_t bytes_paged_in = 0;
        uint64_t current_gpu_bytes = 0;
    };

    PagedKVStore(size_t gpu_budget_bytes);
    ~PagedKVStore();

    void register_block(StreamingKVBlock* block, const std::vector<std::unique_ptr<NativeBlockPool>>& engines);
    void touch(StreamingKVBlock* block, const std::vector<std::unique_ptr<NativeBlockPool>>& engines);
    void maybe_evict(const std::vector<std::unique_ptr<NativeBlockPool>>& engines, const SessionSRLState* srl_state = nullptr);
    void clear();
    void evict_all(const std::vector<std::unique_ptr<NativeBlockPool>>& engines);
    void reload_all(const std::vector<std::unique_ptr<NativeBlockPool>>& engines);
    void swap_state(std::unordered_map<std::string, PageEntry>& entries, Stats& stats);

    Stats get_stats() const;
    std::string summary() const;

private:
    void evict_block(PageEntry& entry, const std::vector<std::unique_ptr<NativeBlockPool>>& engines);
    void reload_block(PageEntry& entry, const std::vector<std::unique_ptr<NativeBlockPool>>& engines);
    size_t estimate_slot_vram(int rank, int head_dim, int kv_heads, int desc_dim) const;
    double get_current_time() const;

    size_t gpu_budget_bytes_;
    std::unordered_map<std::string, PageEntry> entries_; // key: "anchor_idx"
    mutable std::mutex lock_;
    Stats stats_;
};

} // namespace diffkv
