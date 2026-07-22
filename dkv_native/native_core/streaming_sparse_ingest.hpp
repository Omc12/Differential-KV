#pragma once

#include "runtime/native_block_pool.hpp"
#include "runtime/dkv_model.hpp"
#include "native_core/compression/async_compressor.hpp"
#include <vector>
#include <unordered_set>
#include <string>
#include <memory>
#include <mutex>

namespace dkv {

class PagedKVStore;
struct SessionSRLState;

struct StreamingKVBlock {
    int anchor_idx;
    int micro_block_size = 256;
    int pool_idx = -1; // slot index in NativeBlockPool
    BlockState state = BlockState::DenseResident;
    bool is_outlier = false;
    bool skip_compression = false;
    bool skip_compression_evaluated = false; // regex result cached: never re-run should_skip_compression
    bool device_synced = false; // native-attn: host→device pool tensors pushed (incl. VK_rot etc.)
    std::vector<int32_t> token_indices;
    
    // Active dense cache on host
    std::vector<float> active_k; // [micro_block_size * F_test]
    std::vector<float> active_v; // [micro_block_size * F_test]
    
    // Anchor token (1 token dense, irreducible)
    std::vector<float> anchor_k; // [F_test]
    std::vector<float> anchor_v; // [F_test]

    // Contiguous buffer constructed for SVD compressor
    std::vector<float> svd_k; // [(active_block_tokens + 1) * F_test]
    std::vector<float> svd_v; // [(active_block_tokens + 1) * F_test]

    int token_count() const {
        return token_indices.size();
    }
};

class StreamingSparseIngestManager {
public:
    StreamingSparseIngestManager(
        int micro_block_size = 256,
        int recency_window = 512,
        int short_context_threshold = 256,
        bool protect_block_zero = true
    );
    ~StreamingSparseIngestManager();

    void initialize(int n_layers, const DKVModel* model);
    void clear();
    void register_prefill_tokens(const std::vector<int32_t>& token_ids);
    
    // Truncates/rolls back the sequence to target_len.
    void rollback(int target_len, std::vector<std::unique_ptr<NativeBlockPool>>& engines);

    // Ingest a chunk of prefill tokens.
    // engage_threshold: the SAME resolved sparse-engage threshold the caller's
    // own decode-time logic uses (main.cpp's cached_engage_threshold, or
    // batch_engine.cpp's own local computation). Bug found + fixed 2026-07-14:
    // this used to read DKV_ENGAGE_THRESHOLD itself with its own independent
    // hardcoded default (4096), completely disconnected from main.cpp's
    // decode-time default (8192) — a document between the two defaulted to
    // DENSE at decode time but still had its blocks silently COMPRESSED during
    // prefill (bypass_dkv false here, decode_use_sparse false there), so
    // "dense decode" was actually attending lossy low-rank reconstructions for
    // large parts of the context instead of the true exact values. That ~0.001-
    // level reconstruction noise was enough to flip a hard VSL similarity
    // threshold (>=0.40f) and produce a completely different generation.
    // Passing the value explicitly makes the two decisions provably identical.
    void ingest_chunk(
        int layer_idx,
        const float* k_chunk, // [chunk_len * F_test]
        const float* v_chunk,
        int chunk_len,
        int position_start,
        const std::vector<int32_t>& token_ids,
        std::vector<std::unique_ptr<NativeBlockPool>>& engines,
        AsyncCompressor& compressor,
        int rank,
        int engage_threshold,
        PagedKVStore* pager = nullptr,
        SessionSRLState* srl_state = nullptr
    );

    std::vector<std::unique_ptr<StreamingKVBlock>>& get_blocks(int layer_idx) {
        return layers_blocks_[layer_idx];
    }

    const std::vector<std::unique_ptr<StreamingKVBlock>>& get_blocks(int layer_idx) const {
        return layers_blocks_[layer_idx];
    }

    void set_query_words(const std::unordered_set<std::string>& words) {
        query_words_ = words;
    }

    bool should_skip_compression(int anchor_idx, const std::vector<int32_t>& block_tokens) const;
    bool should_boost_compression_rank(int anchor_idx, const std::vector<int32_t>& block_tokens) const;

    struct Stats {
        uint64_t total_blocks_created = 0;
        uint64_t total_compressed = 0;
        uint64_t peak_dense_tokens = 0;
    };

    Stats get_stats() const { return stats_; }

    void swap_blocks(std::vector<std::vector<std::unique_ptr<StreamingKVBlock>>>& other_blocks) {
        std::swap(layers_blocks_, other_blocks);
    }

    void swap_stats(Stats& other_stats) {
        std::swap(stats_, other_stats);
    }

    void set_micro_block_size(int size) {
        micro_block_size_ = size;
    }

    void set_stop_token_ids(const std::unordered_set<int32_t>* stop_token_ids) {
        stop_token_ids_ = stop_token_ids;
    }

    void set_session_id(const std::string& session_id) {
        active_session_id_ = session_id;
    }

    void set_projection_matrix(const float* W_proj, int desc_dim) {
        W_proj_ = W_proj;
        desc_dim_ = desc_dim;
    }

    const std::vector<int32_t>& get_session_token_ids() const { return session_token_ids_; }
    const std::unordered_set<int32_t>* get_stop_token_ids() const { return stop_token_ids_; }

private:
    int next_anchor_idx(int layer_idx) const;
    void submit_block_for_compression(
        int layer_idx,
        int block_idx,
        std::vector<std::unique_ptr<NativeBlockPool>>& engines,
        AsyncCompressor& compressor,
        int rank
    );

    const float* W_proj_ = nullptr;
    int desc_dim_ = 0;

    int micro_block_size_;
    int recency_window_;
    int short_context_threshold_;
    bool protect_block_zero_;
    const DKVModel* model_ = nullptr;
    int n_layers_ = 0;

    std::vector<std::vector<std::unique_ptr<StreamingKVBlock>>> layers_blocks_;
    std::unordered_set<std::string> stopwords_;
    std::unordered_set<std::string> query_words_;
    Stats stats_;
    std::vector<int32_t> session_token_ids_;
    std::vector<size_t> last_compression_scan_idx_;
    const std::unordered_set<int32_t>* stop_token_ids_ = nullptr;
    std::string active_session_id_;
};

} // namespace dkv
