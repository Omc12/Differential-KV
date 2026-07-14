#include "native_core/kv_runtime_manager.hpp"
#include "native_core/srl/chunk_descriptor.hpp"
#include "native_core/srl/query_router.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace diffkv {

// Helper to extract words from string
static std::unordered_set<std::string> extract_words(const std::string& text, const std::unordered_set<std::string>& stopwords) {
    std::unordered_set<std::string> words;
    std::string current = "";
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += std::tolower(static_cast<unsigned char>(c));
        } else {
            if (!current.empty()) {
                if (stopwords.find(current) == stopwords.end()) {
                    words.insert(current);
                }
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        if (stopwords.find(current) == stopwords.end()) {
            words.insert(current);
        }
    }
    return words;
}

static void deallocate_synced_raw_activations(StreamingSparseIngestManager& ingest_mgr,
                                             const std::vector<std::unique_ptr<NativeBlockPool>>& engines,
                                             int n_layers) {
    for (int l = 0; l < n_layers; ++l) {
        auto & blocks = ingest_mgr.get_blocks(l);
        for (auto & block : blocks) {
            if (block->pool_idx != -1) {
                BlockState st = engines[l]->get_state_table().get(block->pool_idx);
                if ((st == BlockState::CompressedResident || st == BlockState::CPUResident || st == BlockState::PagingOut || st == BlockState::Reloading) && block->device_synced) {
                    if (!block->active_k.empty() || !block->svd_k.empty()) {
                        block->active_k.clear();
                        block->active_k.shrink_to_fit();
                        block->active_v.clear();
                        block->active_v.shrink_to_fit();
                        block->svd_k.clear();
                        block->svd_k.shrink_to_fit();
                        block->svd_v.clear();
                        block->svd_v.shrink_to_fit();
                    }
                }
            }
        }
    }
}

KVRuntimeManager::KVRuntimeManager(
    int base_rank,
    int micro_block_size,
    size_t gpu_budget_bytes,
    int recency_window,
    int short_context_threshold
) : base_rank_(base_rank),
    micro_block_size_(micro_block_size),
    gpu_budget_bytes_(gpu_budget_bytes),
    recency_window_(recency_window),
    short_context_threshold_(short_context_threshold) {}

KVRuntimeManager::~KVRuntimeManager() {
    reset();
}

bool KVRuntimeManager::initialize(
    int n_slots,
    int head_dim,
    int kv_heads,
    int desc_dim,
    int n_layers,
    const DiffKVModel* model,
    ggml_backend_buffer_type_t buft,
    ggml_type kv_quant_type
) {
    n_layers_ = n_layers;
    model_ = model;

    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
        std::cerr << "[KVRuntimeManager] Initializing for " << n_layers << " layers..." << std::endl;
    }

    engines_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        engines_[l] = std::make_unique<NativeBlockPool>();
        int layer_rank = get_layer_rank(l);
        int pool_rank = layer_rank * 2; // 2x layer_rank for precision boost headroom
        if (!engines_[l]->initialize(n_slots, pool_rank, head_dim, kv_heads, desc_dim, buft, micro_block_size_, kv_quant_type)) {
            std::cerr << "[KVRuntimeManager] Error: Failed to initialize KVEngine for layer " << l << std::endl;
            return false;
        }
    }

    pager_ = std::make_unique<PagedKVStore>(gpu_budget_bytes_);
    
    ingest_manager_ = std::make_unique<StreamingSparseIngestManager>(
        micro_block_size_,
        recency_window_,
        short_context_threshold_,
        false // protect block zero
    );
    ingest_manager_->initialize(n_layers, model);

    compressor_ = std::make_unique<AsyncCompressor>(engines_[0]->get_state_table());
    if (!compressor_->start()) {
        std::cerr << "[KVRuntimeManager] Error: Failed to start compressor!" << std::endl;
        return false;
    }

    return true;
}

void KVRuntimeManager::reset() {
    if (compressor_) {
        compressor_->stop();
    }
    if (pager_) {
        pager_->clear();
    }
    if (ingest_manager_) {
        ingest_manager_->clear();
    }
    for (auto & engine : engines_) {
        if (engine) {
            engine->reset_slots();
            engine->zero_all_tensors();
            engine->get_state_table().clear();
        }
    }
    if (compressor_) {
        compressor_->start();
    }
}

void KVRuntimeManager::register_prefill_tokens(const std::vector<int32_t>& token_ids) {
    if (ingest_manager_) {
        ingest_manager_->register_prefill_tokens(token_ids);
    }
}

int KVRuntimeManager::get_layer_rank(int layer_idx) const {
    double ratio = (double)layer_idx / std::max(n_layers_, 1);
    // Rank schedule fixed 2026-07-14. This used to taper rank down for later
    // layers (0.75x at ratio>=0.50, 0.50x at ratio>=0.79) — a native-only
    // design; MLX (the reference this codebase mirrors) uses a single
    // uniform rank for EVERY layer, no depth-based reduction at all.
    //
    // Root-caused via a new per-block reconstruction-error probe
    // (DIFFKV_DBG_RECON_ERR, native_core/compression/lowrank.cpp) run on a
    // real table-heavy document: median relative reconstruction error was
    // flat (~0.41-0.50) through most of the network, then jumped sharply
    // (~0.55-0.60 avg, up to 0.75+ max) exactly in the last ~21% of layers —
    // precisely the tier this schedule had halved the rank for. Those are
    // also the layers closest to the LM head, where reconstruction error has
    // the least room to be corrected before it reaches the output logits —
    // this was a direct, measured contributor to native's garbled/incoherent
    // output on dense technical (table-heavy) documents under sparse decode.
    //
    // Restoring uniform rank flattened the error back to the ~0.41-0.50
    // baseline in those layers and measurably fixed the garbled-table failure
    // on the document that exposed it (coherent, on-topic, accurate output
    // afterward). Verified cost: ~15% slower decode at 32k (37.4 -> 31.9 tps)
    // from the extra reconstruction compute in the previously-reduced tiers;
    // peak RSS was NOT meaningfully affected (~3.4GB either way, within
    // run-to-run noise) — pool_rank is already sized at 2x layer_rank
    // regardless, so most of the allocation was already paid for. NIAH 6/6,
    // multi-fact 3/3, table reproduction byte-exact all held. Net: a real but
    // modest speed cost for fixing a real coherence bug.
    if (ratio < 0.15) {
        // Boosted schedule for early layers (disabled by default, check env)
        if (const char* boost_env = std::getenv("DIFFKV_EARLY_LAYER_RANK_BOOST")) {
            if (std::string(boost_env) == "1") {
                return std::min(2 * base_rank_, 64);
            }
        }
    }
    return base_rank_;
}

void KVRuntimeManager::ingest_prefill(
    const std::vector<std::vector<float>>& k_layers,
    const std::vector<std::vector<float>>& v_layers,
    int chunk_len,
    int position_start,
    const std::vector<int32_t>& token_ids,
    int engage_threshold,
    SessionSRLState* srl_state
) {
    if (position_start == 0) {
        register_prefill_tokens(token_ids);
    }

    // 1. Extract query words from the latest 128 tokens of the prompt/prefill
    if (model_) {
        int prefill_len = token_ids.size();
        int query_start = std::max(0, prefill_len - 128);
        std::vector<int32_t> query_tokens(token_ids.begin() + query_start, token_ids.begin() + prefill_len);
        std::string query_text = model_->detokenize(query_tokens);
        
        std::unordered_set<std::string> stopwords = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
            "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
            "hers", "herself", "it", "it's", "its", "itself", "they", "them",
            "their", "theirs", "themselves", "a", "an", "the", "and", "but",
            "or", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "now",
            "should", "should've", "would", "could", "may", "might", "must",
            "shall", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing",
            "get", "got", "make", "made", "go", "went", "take", "took",
            "see", "saw", "say", "said", "use", "used", "find", "found",
            "question", "answer", "text", "context", "information", "prompt",
            "query", "assistant", "system", "user", "file", "document", "page",
            "line", "passage", "following", "please", "write", "read",
            "describe", "explain", "summarize", "extract", "retrieve", "give",
            "tell", "show", "list", "what", "who", "whom", "which", "detail",
            "details", "brief", "exact", "exactly", "correct", "correctly",
            "true", "false", "yes", "no"
        };
        std::unordered_set<std::string> q_words = extract_words(query_text, stopwords);
        ingest_manager_->set_query_words(q_words);
    }

    // 2. Ingest chunk for all layers
    for (int l = 0; l < n_layers_; ++l) {
        int r = get_layer_rank(l);
        ingest_manager_->ingest_chunk(
            l,
            k_layers[l].data(),
            v_layers[l].data(),
            chunk_len,
            position_start,
            token_ids,
            engines_,
            *compressor_,
            r,
            engage_threshold,
            pager_.get(),
            srl_state
        );
    }
    
    // Sync block states in case any block transitioned (though mostly they will stay Compressing until wait_for_compressor)
    for (int l = 0; l < n_layers_; ++l) {
        auto & blocks = ingest_manager_->get_blocks(l);
        for (auto & block : blocks) {
            if (block->state == BlockState::Compressing && block->pool_idx != -1) {
                BlockState current_state = engines_[l]->get_state_table().get(block->pool_idx);
                if (current_state != BlockState::Compressing) {
                    block->state = current_state;
                    if (l == 0 && current_state == BlockState::CompressedResident) {
                        pager_->register_block(block.get(), engines_);
                    }
                    if (current_state == BlockState::CompressedResident) {
                        block->active_k.clear();
                        block->active_k.shrink_to_fit();
                        block->active_v.clear();
                        block->active_v.shrink_to_fit();
                        block->svd_k.clear();
                        block->svd_k.shrink_to_fit();
                        block->svd_v.clear();
                        block->svd_v.shrink_to_fit();
                    }
                }
            }
        }
    }
    
    // 3. Evict excess resident memory under budget
    pager_->maybe_evict(engines_, srl_state);
}

void KVRuntimeManager::ingest_decode(
    const std::vector<std::vector<float>>& k_layers,
    const std::vector<std::vector<float>>& v_layers,
    int current_pos,
    const std::vector<int32_t>& token_ids,
    int engage_threshold,
    SessionSRLState* srl_state,
    bool defer_device_sync
) {
    // 1. Sync states of any compressing blocks from background thread to host blocks
    for (int l = 0; l < n_layers_; ++l) {
        auto & blocks = ingest_manager_->get_blocks(l);
        for (auto & block : blocks) {
            if (block->state == BlockState::Compressing && block->pool_idx != -1) {
                BlockState current_state = engines_[l]->get_state_table().get(block->pool_idx);
                if (current_state != BlockState::Compressing) {
                    block->state = current_state;
                }
            }
        }
    }

    // 2. Ingest token
    for (int l = 0; l < n_layers_; ++l) {
        int r = get_layer_rank(l);
        ingest_manager_->ingest_chunk(
            l,
            k_layers[l].data(),
            v_layers[l].data(),
            1, // chunk_len = 1 for decode
            current_pos,
            token_ids,
            engines_,
            *compressor_,
            r,
            engage_threshold,
            pager_.get(),
            srl_state
        );
    }
    
    // 3. Sync states again in case any block transitioned during ingest_chunk
    for (int l = 0; l < n_layers_; ++l) {
        auto & blocks = ingest_manager_->get_blocks(l);
        for (auto & block : blocks) {
            if (block->state == BlockState::Compressing && block->pool_idx != -1) {
                BlockState current_state = engines_[l]->get_state_table().get(block->pool_idx);
                if (current_state != BlockState::Compressing) {
                    block->state = current_state;
                    if (current_state == BlockState::CompressedResident) {
                        block->device_synced = false;
                        if (l == 0) {
                            pager_->register_block(block.get(), engines_);
                        }
                    }
                }
            }
        }
    }
    
    // 3b. Native attn: push host→device for compressed-but-unsynced slots (main thread).
    if (!defer_device_sync) {
        sync_device_for_native();
        pager_->maybe_evict(engines_, srl_state);
    }
}

std::vector<int32_t> KVRuntimeManager::route_decode_slots(
    int current_pos,
    const std::vector<int32_t>& token_ids,
    SessionSRLState& srl_state,
    const std::unordered_set<int32_t>& stop_token_ids,
    int srl_k_recency,
    int srl_k_lexical,
    int srl_k_graph,
    int srl_k_host,
    int active_slot,
    bool high_quality
) const {
    std::vector<int32_t> host_candidates;
    std::unordered_set<int32_t> seen;

    // 0. Always include sink blocks (first/last blocks, critical for attention sinks)
    for (int32_t sink : srl_state.sink_blocks) {
        if (sink >= 0) {
            host_candidates.push_back(sink);
            seen.insert(sink);
        }
    }

    const auto& ord = srl_state.ordered_slot_ids;
    int n_ord = static_cast<int>(ord.size());

    if (n_ord <= 36) {
        for (int i = 0; i < n_ord; ++i) {
            int32_t slot = ord[i];
            if (slot >= 0 && !seen.count(slot)) {
                host_candidates.push_back(slot);
                seen.insert(slot);
            }
        }
    } else {
        // N4.3 fix: scale channel budgets with context length like Python's adaptive_k.
        int eff_recency = std::max(srl_k_recency, std::min(n_ord / 10, srl_k_recency * 4));
        int eff_lexical = std::max(srl_k_lexical, std::min(n_ord / 8,  srl_k_lexical * 4));
        int eff_graph   = std::max(srl_k_graph,   std::min(n_ord / 8,  srl_k_graph * 4));

        // 1. Recency window: latest eff_recency slots from ordered slot list
        int take_r = std::min(eff_recency, n_ord);
        for (int i = n_ord - take_r; i < n_ord; ++i) {
            int32_t slot = ord[i];
            if (slot >= 0) {
                if (!seen.count(slot)) {
                    host_candidates.push_back(slot);
                    seen.insert(slot);
                }
            }
        }

        // 2. Lexical search slots
        // Search a wider window of recent history (up to last 128 tokens) for keywords
        int query_start = std::max(0, current_pos - 128);
        std::vector<int> query_tokens;
        for (int i = query_start; i < current_pos; ++i) {
            if (i < (int)token_ids.size()) {
                query_tokens.push_back(token_ids[i]);
            }
        }
        float decay_factor = 1.0f;
        if (const char* env = std::getenv("DIFFKV_SRL_DECAY_FACTOR")) {
            decay_factor = std::stof(env);
        }
        auto lex_scored = score_lexical_slots(srl_state.inverted_index, query_tokens, decay_factor);
        if (current_pos == 8192) {
            std::cerr << "[DEBUG_LEX] occurrences.size()=" << srl_state.inverted_index.occurrences.size() << "\n";
            std::cerr << "[DEBUG_LEX] query_tokens.size()=" << query_tokens.size() << "\n";
            std::cerr << "[DEBUG_LEX] Lexical scored blocks count=" << lex_scored.size() << "\n";
            for (size_t i = 0; i < std::min((size_t)15, lex_scored.size()); ++i) {
                std::cerr << "  slot=" << lex_scored[i].first << " score=" << lex_scored[i].second << "\n";
            }
        }
        std::vector<int32_t> lexical_slots;
        for (int i = 0; i < std::min(eff_lexical, (int)lex_scored.size()); ++i) {
            int32_t slot = lex_scored[i].first;
            if (slot >= 0) {
                lexical_slots.push_back(slot);
                if (!seen.count(slot)) {
                    host_candidates.push_back(slot);
                    seen.insert(slot);
                }
            }
        }

        // 2.5. Semantic proxy-query relevance — ALWAYS ON, including fast mode.
        //
        // Every other channel here is TOKEN-level: recency is positional, lexical
        // is literal token-overlap against the last 128 tokens. Neither can surface
        // a block whose CONTENT is relevant but shares no VOCABULARY with what has
        // recently been said — and because the lexical query window includes the
        // model's OWN generated tokens, once generation drifts even slightly the
        // query keeps reinforcing the drift instead of correcting it. Measured
        // (2026-07-13): a technical paper's content was dropped ENTIRELY in favor
        // of unrelated trailing filler text, in BOTH fast pruning mode and the old
        // attend-all+graph HQ mode — so the missing channel, not the pruning
        // policy, was the actual gap. MLX's decode router has no lexical stage at
        // all: every block is scored by direct Q·anchor_K relevance every step.
        //
        // Give native the same signal, cheaply: dot each block's anchor K
        // (averaged across kv_heads to match the proxy query's shape) against a
        // ONE-STEP-LAGGED proxy query vector. This reuses the exact "K as Q proxy"
        // idiom already established for the factual store (srl_state.
        // recent_decode_keys; main.cpp ~6185-6194 — "one-step lag is unavoidable:
        // the true Q lives inside the GGML graph, K is the best available proxy
        // extracted from the same token embedding"). Cost is O(n_active_blocks *
        // head_dim) — a few hundred blocks x ~128 floats, negligible next to the
        // lexical/recency work already done every retrieval interval.
        if (const char* dbg = std::getenv("DIFFKV_DBG_SEMROUTE")) {
            std::cerr << "[SEMROUTE] n_ord=" << n_ord
                      << " recent_decode_keys.size()=" << srl_state.recent_decode_keys.size()
                      << " engines_.empty()=" << engines_.empty() << "\n";
        }
        if (!srl_state.recent_decode_keys.empty() && !engines_.empty()) {
            const std::vector<float>& q_proxy = srl_state.recent_decode_keys.back();
            NativeBlockPool* pool0 = engines_[0].get();
            const int hd  = pool0->get_head_dim();
            const int kvh = pool0->get_kv_heads();
            if (std::getenv("DIFFKV_DBG_SEMROUTE")) {
                std::cerr << "[SEMROUTE] hd=" << hd << " kvh=" << kvh << " q_proxy.size()=" << q_proxy.size() << "\n";
            }
            if (hd > 0 && kvh > 0 && (int)q_proxy.size() == hd) {
                auto& state_table = pool0->get_state_table();
                std::vector<std::pair<float, int32_t>> sem_scored;
                sem_scored.reserve(n_ord);
                for (int i = 0; i < n_ord; ++i) {
                    int32_t slot = ord[i];
                    if (slot < 0) continue;
                    if (state_table.get(slot) != BlockState::CompressedResident) continue;
                    const ggml_fp16_t* anc = pool0->get_host_anchors_K(slot);
                    if (!anc) continue;
                    float score = 0.0f;
                    for (int h = 0; h < kvh; ++h) {
                        const ggml_fp16_t* row = anc + (size_t)h * hd;
                        for (int d = 0; d < hd; ++d) {
                            score += q_proxy[d] * ggml_fp16_to_fp32(row[d]);
                        }
                    }
                    sem_scored.push_back({score / (float)kvh, slot});
                }
                // Same budget as the lexical channel — both are cheap first-stage
                // retrieval channels, just scored on different signals.
                int take_sem = std::min(eff_lexical, (int)sem_scored.size());
                if (take_sem > 0) {
                    std::partial_sort(sem_scored.begin(), sem_scored.begin() + take_sem, sem_scored.end(),
                                      [](const auto& a, const auto& b) { return a.first > b.first; });
                    int added = 0;
                    for (int i = 0; i < take_sem; ++i) {
                        int32_t slot = sem_scored[i].second;
                        if (!seen.count(slot)) {
                            host_candidates.push_back(slot);
                            seen.insert(slot);
                            ++added;
                        }
                    }
                    if (std::getenv("DIFFKV_DBG_SEMROUTE")) {
                        std::cerr << "[SEMROUTE] sem_scored.size()=" << sem_scored.size()
                                  << " take_sem=" << take_sem << " newly_added=" << added
                                  << " host_candidates.size()=" << host_candidates.size() << "\n";
                    }
                }
            }
        }

        // Sections 3–4.5 (chunk-graph 2-hop + anchor-neighborhood expansion) are
        // the "dynamic graph routing" — best synthesis fidelity but they balloon
        // the candidate pool on uniform docs. HIGH-QUALITY ONLY. In fast bounded-K
        // mode the pool stays at sink + recency + lexical, so materialization is
        // cheap and query-independent (mirrors MLX's decode router).
        if (high_quality) {
        // 3. Chunk Graph Adjacency / 2-hop neighborhood expansion
        const ChunkGraph& g = srl_state.chunk_graph;
        int N = g.N;
        if (N > 0 && N == srl_state.n_active_blocks()) {
            std::vector<float> seed_scores(N, 0.0f);
            std::unordered_set<int32_t> seed_set;

            // Populate seed activations from lexical match scores
            for (const auto& pair : lex_scored) {
                int32_t slot = pair.first;
                auto it = std::find(ord.begin(), ord.end(), slot);
                if (it != ord.end()) {
                    int idx = std::distance(ord.begin(), it);
                    if (idx >= 0 && idx < N) {
                        seed_scores[idx] = pair.second;
                        seed_set.insert(slot);
                    }
                }
            }

            // pointwise decay/retention
            std::vector<float> retention(N, srl_state.graph_hop_decay);

            // 1-hop propagation
            std::vector<float> A1 = graph_propagate(g, seed_scores, retention, srl_state.graph_hop_decay);
            // 2-hop propagation
            std::vector<float> A2 = graph_propagate(g, A1, retention, srl_state.graph_hop_decay);

            std::vector<std::pair<float, int32_t>> gscore_slots;
            for (int i = 0; i < N; ++i) {
                int32_t slot = ord[i];
                if (seed_set.count(slot)) continue;
                float gs = A1[i] + A2[i];
                if (gs > 0.0f && slot >= 0) {
                    gscore_slots.push_back({gs, slot});
                }
            }

            int take_g = std::min(eff_graph, (int)gscore_slots.size());
            if (take_g > 0) {
                std::partial_sort(gscore_slots.begin(), gscore_slots.begin() + take_g, gscore_slots.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                for (int i = 0; i < take_g; ++i) {
                    int32_t slot = gscore_slots[i].second;
                    if (!seen.count(slot)) {
                        host_candidates.push_back(slot);
                        seen.insert(slot);
                    }
                }
            }
        }

        // 4. Dynamic routing anchors expansion
        if (!srl_state.dynamic_anchors.empty()) {
            std::unordered_set<int32_t> da_set(srl_state.dynamic_anchors.begin(), srl_state.dynamic_anchors.end());
            std::vector<int32_t> expanded_da = srl_state.expand_neighborhood(da_set);
            for (int32_t slot : expanded_da) {
                if (slot >= 0) {
                    if (!seen.count(slot)) {
                        host_candidates.push_back(slot);
                        seen.insert(slot);
                    }
                }
            }
        }

        // 4.5 Prompt routing anchors expansion
        if (!srl_state.prompt_anchors.empty()) {
            int b_size = srl_state.inverted_index.block_size;
            std::unordered_set<int32_t> pa_slots;
            for (int idx : srl_state.prompt_anchors) {
                int block_idx = idx / b_size;
                if (block_idx >= 0 && block_idx < static_cast<int>(srl_state.ordered_slot_ids.size())) {
                    pa_slots.insert(srl_state.ordered_slot_ids[block_idx]);
                }
            }
            if (!pa_slots.empty()) {
                std::vector<int32_t> expanded_pa = srl_state.expand_neighborhood(pa_slots);
                for (int32_t slot : expanded_pa) {
                    if (slot >= 0) {
                        if (!seen.count(slot)) {
                            host_candidates.push_back(slot);
                            seen.insert(slot);
                        }
                    }
                }
            }
        }
        } // if (high_quality) — end dynamic graph routing (sections 3–4.5)
    }

    // 5. Structured Attention Segmenting filtering
    int curr_seg = srl_state.current_query_segment_id;
    if (curr_seg != 0 && !srl_state.segment_ids.empty()) {
        std::unordered_set<int32_t> sink_set(srl_state.sink_blocks.begin(), srl_state.sink_blocks.end());
        std::vector<int32_t> filtered;
        for (int32_t slot : host_candidates) {
            if (sink_set.count(slot)) {
                filtered.push_back(slot);
                continue;
            }
            auto it = srl_state.segment_ids.find(slot);
            if (it != srl_state.segment_ids.end()) {
                int seg_id = it->second;
                if (seg_id == 0 || seg_id == curr_seg) {
                    filtered.push_back(slot);
                }
            } else {
                filtered.push_back(slot);
            }
        }
        host_candidates = filtered;
    }

    if (std::getenv("DIFFKV_DBG_SEMROUTE")) {
        std::cerr << "[SEMROUTE] pre-cap host_candidates.size()=" << host_candidates.size()
                  << " srl_k_host=" << srl_k_host << "\n";
    }
    // Pad with 0 up to srl_k_host
    while (host_candidates.size() < (size_t)srl_k_host) {
        host_candidates.push_back(0);
    }

    // Cap at srl_k_host
    if (host_candidates.size() > (size_t)srl_k_host) {
        host_candidates.resize(srl_k_host);
    }

    // Update slot reinforcement/activation strength for the routed slots
    std::unordered_set<int32_t> selected_set(host_candidates.begin(), host_candidates.end());
    float alpha_boost = 0.05f;
    float decay_rate = 0.99f;
    for (int32_t slot : selected_set) {
        if (srl_state.slot_activation_strength.find(slot) == srl_state.slot_activation_strength.end()) {
            srl_state.slot_activation_strength[slot] = 1.0f;
        }
        srl_state.slot_activation_strength[slot] += alpha_boost;
    }
    for (auto& pair : srl_state.slot_activation_strength) {
        if (selected_set.count(pair.first) == 0) {
            pair.second *= decay_rate;
            if (pair.second < 1.0f) {
                pair.second = 1.0f;
            }
        }
    }

    // ── Write result back to srl_state (mirrors ACTIVE_RUNTIME) ──────────────
    // ACTIVE_RUNTIME/runtime/diffkv_attention.py:544:
    //   srl_state.current_step_slots = selected_slots  ← set once at layer 0
    //   selected_slots = srl_state.current_step_slots  ← reused layers 1-N
    // Here we do the same: any caller that throttles routing can read the last
    // result from srl_state.current_step_slots instead of re-running routing.
    srl_state.current_step_slots = host_candidates;
    srl_state.current_step_count++;

    if (std::getenv("DIFFKV_ROUTING_VERBOSE")) {
        std::cerr << "[ROUTE] pos=" << current_pos
                  << " n_blocks=" << srl_state.n_active_blocks()
                  << " routed=" << host_candidates.size()
                  << " call#" << srl_state.current_step_count << "\n";
    }

    return host_candidates;
}

void KVRuntimeManager::wait_for_compressor() {
    // Block until every submitted SVD job is fully processed.
    // After this returns, ALL blocks are CompressedResident in the state_table —
    // so the old pending_slots loop was dead code (pending_slots was always empty
    // because get_state_table().get() was no longer Compressing for any slot).
    // The real upload was accidentally skipped: device_synced=true was set without
    // calling upload_slot → GPU pool tensors remained uninitialized → sparse Metal op
    // read zeros → word-salad output at contexts where compression finishes early (e.g. 4k).
    if (compressor_) compressor_->wait_until_idle();

    // Upload ALL valid slots to GPU now that the compressor is idle.
    // (Removed the old pending_slots filter — every block needs to be pushed.)
    int n_slots = engines_[0]->get_n_slots();
    for (int pool_idx = 0; pool_idx < n_slots; ++pool_idx) {
        // Upload if ANY layer's state is non-freed/non-invalid (has compressed data)
        bool needs_upload = false;
        for (int l = 0; l < n_layers_; ++l) {
            BlockState st = engines_[l]->get_state_table().get(pool_idx);
            if (st != BlockState::Freed && st != BlockState::Invalid) {
                needs_upload = true;
                break;
            }
        }
        if (needs_upload) {
            for (int l = 0; l < n_layers_; ++l) {
                engines_[l]->upload_slot(pool_idx);
            }
        }
    }

    // Sync block->state from state_table and mark as device_synced.
    for (int l = 0; l < n_layers_; ++l) {
        auto & blocks = ingest_manager_->get_blocks(l);
        for (auto & block : blocks) {
            if (block->state == BlockState::Compressing && block->pool_idx != -1) {
                BlockState current_state = engines_[l]->get_state_table().get(block->pool_idx);
                if (current_state != BlockState::Compressing) {
                    block->state = current_state;
                    if (current_state == BlockState::CompressedResident) {
                        block->device_synced = true;
                        if (l == 0) {
                            pager_->register_block(block.get(), engines_);
                        }
                    }
                }
            }
        }
    }
    deallocate_synced_raw_activations(*ingest_manager_, engines_, n_layers_);
}



void KVRuntimeManager::touch_active_slots(const std::vector<int32_t>& active_slots) {
    if (active_slots.empty()) return;
    auto & blocks = ingest_manager_->get_blocks(0);
    for (int32_t slot_id : active_slots) {
        for (auto & block : blocks) {
            if (block->pool_idx == slot_id) {
                pager_->touch(block.get(), engines_);
                break;
            }
        }
    }
}

void KVRuntimeManager::update_descriptors(const std::vector<float>& W_proj_host, int desc_dim, int head_dim) {
    set_projection_matrix(W_proj_host.data(), desc_dim);
    auto & blocks = ingest_manager_->get_blocks(0);
    int F_test = engines_[0]->get_head_dim() * engines_[0]->get_kv_heads();
    int kv_heads = engines_[0]->get_kv_heads();
    
    for (size_t b = 0; b < blocks.size(); ++b) {
        auto & block = blocks[b];
        if (block->pool_idx == -1) continue; // paged out
        
        int slot_id = block->pool_idx;
        
        if (block->state == BlockState::CompressedResident || block->state == BlockState::CPUResident) {
            auto & engine = engines_[0];
            if (engine->get_host_U(slot_id) == nullptr) {
                // If skip_lowrank is true, the descriptor is already computed by the SVD compressor
                // and stored in host_desc_matrix_. No need to re-compute.
                continue;
            }
            int rank = engine->get_rank();
            
            std::vector<ggml_fp16_t> desc_f16(desc_dim);
            compute_descriptor(
                (const uint16_t*)engine->get_host_anchors_K(slot_id),
                engine->get_host_U(slot_id),
                ggml_fp16_to_fp32(engine->get_host_U_scale()[slot_id]),
                (const uint16_t*)engine->get_host_VK(slot_id),
                W_proj_host.data(),
                kv_heads,
                head_dim,
                block->token_count() - 1, // S_deltas = seq_len - 1
                rank,
                (uint16_t*)desc_f16.data()
            );
            
            std::vector<float> desc(desc_dim);
            for (int r = 0; r < desc_dim; ++r) {
                desc[r] = ggml_fp16_to_fp32(desc_f16[r]);
            }
            std::copy(desc.begin(), desc.end(), engine->get_host_desc_matrix(slot_id));
            ggml_backend_tensor_set(engine->get_desc_matrix(), desc.data(), slot_id * desc_dim * sizeof(float), desc_dim * sizeof(float));
        } else {
            std::vector<float> avg_k(F_test, 0.0f);
            int S_total = block->token_count();
            if (!block->svd_k.empty()) {
                for (int i = 0; i < F_test; ++i) {
                    for (int t = 0; t < S_total; ++t) {
                        avg_k[i] += block->svd_k[t * F_test + i];
                    }
                    avg_k[i] /= S_total;
                }
            } else {
                for (int i = 0; i < F_test; ++i) {
                    avg_k[i] += block->anchor_k[i];
                    for (size_t t = 0; t < block->active_k.size() / F_test; ++t) {
                        avg_k[i] += block->active_k[t * F_test + i];
                    }
                    avg_k[i] /= S_total;
                }
            }
            
            std::vector<float> desc(desc_dim, 0.0f);
            for (int r = 0; r < desc_dim; ++r) {
                float sum = 0.0f;
                for (int c = 0; c < head_dim; ++c) {
                    sum += avg_k[c] * W_proj_host[r * head_dim + c];
                }
                desc[r] = sum;
            }
            float sum_sq = 0.0f;
            for (float val : desc) sum_sq += val * val;
            float norm = std::sqrt(sum_sq) + 1e-8f;
            for (float & val : desc) val /= norm;
            
            std::copy(desc.begin(), desc.end(), engines_[0]->get_host_desc_matrix(slot_id));
            ggml_backend_tensor_set(engines_[0]->get_desc_matrix(), desc.data(), slot_id * desc_dim * sizeof(float), desc_dim * sizeof(float));
        }
    }
}

void KVRuntimeManager::set_projection_matrix(const float* W_proj, int desc_dim) {
    if (ingest_manager_) {
        ingest_manager_->set_projection_matrix(W_proj, desc_dim);
    }
}

void KVRuntimeManager::set_micro_block_size(int size) {
    micro_block_size_ = size;
    if (ingest_manager_) {
        ingest_manager_->set_micro_block_size(size);
    }
}

void KVRuntimeManager::sync_device_for_native() {
    // Async SVD writes only the HOST pool mirrors + flips block state; upload_slot does the
    // device push AND computes the native VK_rot/anchorK_rot/valid_mask/U_f16. The custom op
    // reads host buffers (immune), but the native ggml subgraph reads the device tensors, so
    // we must push them on the main thread before the decode graph runs.
    if (engines_.empty() || !engines_[0]->native_attn_enabled()) return;
    
    static const bool sync_all = std::getenv("DIFFKV_SYNC_ALL") != nullptr;
    if (sync_all) {
        int ns = engines_[0]->get_seq_lens()->ne[0];
        for (int l = 0; l < n_layers_; ++l)
            for (int s = 0; s < ns; ++s) engines_[l]->upload_slot(s);
        return;
    }

    for (int l = 0; l < n_layers_; ++l) {
        int ns = engines_[l]->get_seq_lens()->ne[0];
        
        // Build map of currently active slots and a slot-to-block lookup array
        std::vector<StreamingKVBlock*> slot_to_block(ns, nullptr);
        std::vector<bool> active_slots(ns, false);
        for (auto & block : ingest_manager_->get_blocks(l)) {
            if (block->pool_idx != -1 && block->pool_idx < ns) {
                slot_to_block[block->pool_idx] = block.get();
                BlockState st = engines_[l]->get_state_table().get(block->pool_idx);
                // Any block holding valid KV data must have its DEVICE tensors (anchorK_rot/
                // VK_rot/U_f16/valid_mask) uploaded for the native subgraph — not just
                // CompressedResident/DenseResident. Routed slots in Compressing/CPUResident/
                // Reloading were left with device=0 (host valid), so the native saw zero K for
                // them → glitches that DIFFKV_SYNC_ALL masked. Include all non-empty states.
                if (st != BlockState::Freed && st != BlockState::Invalid) {
                    active_slots[block->pool_idx] = true;
                }
            }
        }
        
        // For each slot, synchronize state
        for (int s = 0; s < ns; ++s) {
            if (active_slots[s]) {
                StreamingKVBlock* block = slot_to_block[s];
                if (block && !block->device_synced) {
                    engines_[l]->upload_slot(s);
                    block->device_synced = true;
                }
            } else {
                // If it is NOT occupied by any active block, we must ensure
                // that the device knows it is empty (i.e. seq_len = 0, valid_mask = -inf).
                if (engines_[l]->slot_device_has_data(s)) {
                    engines_[l]->upload_slot(s);
                }
            }
        }
    }
    deallocate_synced_raw_activations(*ingest_manager_, engines_, n_layers_);
}

void KVRuntimeManager::commit_turn(SessionSRLState& srl_state) {
    if (srl_state.n_active_blocks() == 0) return;

    // Identify which blocks to keep
    std::unordered_set<int32_t> keep_slots;
    
    // Add sink blocks
    for (int32_t sink : srl_state.sink_blocks) {
        keep_slots.insert(sink);
    }
    
    // Add cluster centers / landmarks
    for (int32_t center : srl_state.chunk_graph.cluster_centers_tensor) {
        keep_slots.insert(center);
    }
    for (int32_t landmark : srl_state.chunk_graph.parent_landmarks) {
        keep_slots.insert(landmark);
    }
    
    // Add factual exact store slots
    for (const auto& entry : srl_state.factual_store.entries) {
        for (int32_t slot : entry.slot_ids) {
            keep_slots.insert(slot);
        }
    }
    
    // Scan active blocks (anchor_idx >= srl_state.cached_len)
    std::unordered_set<int32_t> active_slots;
    auto & blocks_layer0 = ingest_manager_->get_blocks(0);
    for (const auto& b : blocks_layer0) {
        if (b->pool_idx != -1 && b->anchor_idx >= srl_state.cached_len) {
            active_slots.insert(b->pool_idx);
            
            // Check for high-IDF tokens in the block
            if (!srl_state.inverted_index.idf.empty()) {
                for (int32_t tok : b->token_indices) {
                    auto it = srl_state.inverted_index.idf.find(tok);
                    if (it != srl_state.inverted_index.idf.end() && it->second >= 2.5f) {
                        keep_slots.insert(b->pool_idx);
                        break;
                    }
                }
            }
        }
    }
    
    // Identify slots to prune
    std::unordered_set<int32_t> pruned_slots;
    for (int32_t slot : active_slots) {
        if (keep_slots.count(slot) == 0) {
            pruned_slots.insert(slot);
        }
    }
    
    if (pruned_slots.empty()) {
        srl_state.cached_len = static_cast<int>(ingest_manager_->get_session_token_ids().size());
        return;
    }
    
    // Free slots in engines and remove blocks from ingest manager
    for (int32_t slot : pruned_slots) {
        // Clear from slot reinforcement map
        srl_state.slot_activation_strength.erase(slot);
    }
    
    for (int l = 0; l < n_layers_; ++l) {
        auto & blocks = ingest_manager_->get_blocks(l);
        std::vector<std::unique_ptr<StreamingKVBlock>> kept;
        for (auto & block : blocks) {
            if (block->pool_idx != -1 && pruned_slots.count(block->pool_idx)) {
                engines_[l]->free_slot(block->pool_idx);
                continue;
            }
            kept.push_back(std::move(block));
        }
        blocks = std::move(kept);
    }
    
    // Save reinforcement strengths before rebuilding
    std::unordered_map<int32_t, float> strengths = srl_state.slot_activation_strength;
    
    // Update cached_len to current sequence length
    int prior_cached_len = srl_state.cached_len;
    const auto& all_tokens = ingest_manager_->get_session_token_ids();
    srl_state.cached_len = static_cast<int>(all_tokens.size());

    // Populate current_query_tokens (query tokens since last prefill)
    srl_state.current_query_tokens.clear();
    if (prior_cached_len == 0) {
        int query_start = std::max(0, (int)all_tokens.size() - 128);
        srl_state.current_query_tokens.assign(all_tokens.begin() + query_start, all_tokens.end());
    } else if (prior_cached_len < (int)all_tokens.size()) {
        srl_state.current_query_tokens.assign(all_tokens.begin() + prior_cached_len, all_tokens.end());
    }
    
    // Rebuild the index
    auto & blocks_l0 = ingest_manager_->get_blocks(0);
    std::vector<int32_t> compressed_slots;
    std::vector<int> compressed_anchors;
    for (const auto& block : blocks_l0) {
        if (block->pool_idx != -1 &&
            (block->state == BlockState::CompressedResident ||
             block->state == BlockState::CPUResident ||
             block->state == BlockState::Compressing ||
             block->state == BlockState::DenseResident)) {
            compressed_slots.push_back(block->pool_idx);
            compressed_anchors.push_back(block->anchor_idx);
        }
    }
    int completed_blocks = compressed_slots.size();
    if (completed_blocks > 0) {
        int desc_dim = engines_[0]->get_desc_matrix()->ne[0];
        std::vector<float> desc_matrix_host(completed_blocks * desc_dim);
        for (int j = 0; j < completed_blocks; ++j) {
            int slot_id = compressed_slots[j];
            ggml_backend_tensor_get(
                engines_[0]->get_desc_matrix(),
                desc_matrix_host.data() + j * desc_dim,
                slot_id * desc_dim * sizeof(float),
                desc_dim * sizeof(float)
            );
        }

        const auto& all_tokens = ingest_manager_->get_session_token_ids();
        std::unordered_set<int> stop_tokens_int;
        const auto* stop_ptr = ingest_manager_->get_stop_token_ids();
        if (stop_ptr) {
            stop_tokens_int.insert(stop_ptr->begin(), stop_ptr->end());
        }

        srl_state = build_srl_state_from_blocks(
            desc_matrix_host.data(),
            compressed_slots.data(),
            completed_blocks,
            all_tokens.data(),
            all_tokens.size(),
            micro_block_size_ + 1, // block_size
            stop_tokens_int,
            6, // K_semantic
            2, // K_temporal
            0.15f, // overlap_threshold
            true, // add_first_as_sink
            true,  // add_last_as_sink
            &compressed_anchors,
            srl_state.cached_len
        );
    }
    
    // Restore reinforcement strengths for kept slots
    for (const auto& pair : strengths) {
        if (std::find(srl_state.ordered_slot_ids.begin(), srl_state.ordered_slot_ids.end(), pair.first) != srl_state.ordered_slot_ids.end()) {
            srl_state.slot_activation_strength[pair.first] = pair.second;
        }
    }
}

} // namespace diffkv
