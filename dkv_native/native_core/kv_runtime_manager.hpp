#pragma once

#include "runtime/native_block_pool.hpp"
#include "runtime/dkv_model.hpp"
#include "native_core/compression/async_compressor.hpp"
#include "native_core/paging/paged_kv_store.hpp"
#include "native_core/streaming_sparse_ingest.hpp"
#include "native_core/srl/session_srl_state.hpp" // For SessionSRLState
#include <vector>
#include <unordered_set>
#include <memory>
#include <string>

namespace dkv {

class KVRuntimeManager {
public:
    KVRuntimeManager(
        int base_rank,
        int micro_block_size,
        size_t gpu_budget_bytes,
        int recency_window = 512,
        int short_context_threshold = 256
    );
    ~KVRuntimeManager();

    bool initialize(
        int n_slots,
        int head_dim,
        int kv_heads,
        int desc_dim,
        int n_layers,
        const DKVModel* model,
        ggml_backend_buffer_type_t buft,
        ggml_type kv_quant_type = GGML_TYPE_Q8_0
    );

    void reset();
    void register_prefill_tokens(const std::vector<int32_t>& token_ids);

    // Ingest a chunk of prefill tokens for all layers.
    // engage_threshold: the caller's OWN already-resolved sparse-engage
    // threshold — must be the exact value driving that caller's decode-time
    // decode_use_sparse decision. See streaming_sparse_ingest.hpp's ingest_chunk
    // for why this can no longer be independently re-derived here.
    void ingest_prefill(
        const std::vector<std::vector<float>>& k_layers, // [n_layers][chunk_len * F_test]
        const std::vector<std::vector<float>>& v_layers, // [n_layers][chunk_len * F_test]
        int chunk_len,
        int position_start,
        const std::vector<int32_t>& token_ids,
        int engage_threshold,
        SessionSRLState* srl_state = nullptr
    );

    // Ingest a single decode token's key/value for all layers.
    // engage_threshold: same requirement as ingest_prefill above.
    void ingest_decode(
        const std::vector<std::vector<float>>& k_layers, // [n_layers][F_test]
        const std::vector<std::vector<float>>& v_layers, // [n_layers][F_test]
        int current_pos,
        const std::vector<int32_t>& token_ids,
        int engage_threshold,
        SessionSRLState* srl_state = nullptr,
        bool defer_device_sync = false
    );

    // Dynamic per-layer adaptive SVD rank schedule
    int get_layer_rank(int layer_idx) const;

    // SRL Routing slot selection.
    // high_quality: when true, expand the candidate pool with the dynamic 2-hop
    //   chunk-graph propagation + anchor-neighborhood expansion (best synthesis
    //   fidelity, but the candidate pool balloons on uniform docs). When false
    //   (fast bounded-K default), skip the graph — recency + lexical only — so
    //   the pool stays small and materialization is cheap. Gated by
    //   DKV_HIGH_QUALITY_ROUTING (see src/main.cpp).
    std::vector<int32_t> route_decode_slots(
        int current_pos,
        const std::vector<int32_t>& token_ids,
        SessionSRLState& srl_state,
        const std::unordered_set<int32_t>& stop_token_ids,
        int srl_k_recency,
        int srl_k_lexical,
        int srl_k_graph,
        int srl_k_host,
        int active_slot,
        bool high_quality = true
    ) const;

    // Update semantic descriptor matrix
    void update_descriptors(const std::vector<float>& W_proj_host, int desc_dim, int head_dim);

    // Set projection matrix for SVD jobs
    void set_projection_matrix(const float* W_proj, int desc_dim);

    // Wait for all submitted SVD jobs to complete
    void wait_for_compressor();

    // Touch slot indexes to trigger reloading from CPU if needed
    void touch_active_slots(const std::vector<int32_t>& active_slots);

    // Prune low-salience blocks and consolidate index
    void commit_turn(SessionSRLState& srl_state);

    // Native attn: push host→device pool tensors (incl. VK_rot/anchorK_rot/valid_mask/U_f16)
    // for any compressed-but-unsynced slot. Async SVD only writes host mirrors, so the native
    // ggml subgraph (which reads device tensors) needs this on the main thread before compute.
    void sync_device_for_native();

    void set_micro_block_size(int size);
    int get_micro_block_size() const { return micro_block_size_; }

    // Getters
    std::vector<std::unique_ptr<NativeBlockPool>>& get_engines() { return engines_; }
    PagedKVStore& get_pager() { return *pager_; }
    StreamingSparseIngestManager& get_ingest_manager() { return *ingest_manager_; }
    AsyncCompressor& get_compressor() { return *compressor_; }

private:
    int base_rank_;
    int micro_block_size_;
    size_t gpu_budget_bytes_;
    int recency_window_;
    int short_context_threshold_;
    int n_layers_ = 0;
    const DKVModel* model_ = nullptr;

    std::vector<std::unique_ptr<NativeBlockPool>> engines_;
    std::unique_ptr<PagedKVStore> pager_;
    std::unique_ptr<StreamingSparseIngestManager> ingest_manager_;
    std::unique_ptr<AsyncCompressor> compressor_;
};

} // namespace dkv
