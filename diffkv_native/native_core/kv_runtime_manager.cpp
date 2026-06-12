#include "native_core/kv_runtime_manager.hpp"
#include "native_core/srl/chunk_descriptor.hpp"
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
    ggml_backend_buffer_type_t buft
) {
    n_layers_ = n_layers;
    model_ = model;

    if (std::getenv("DIFFKV_VERBOSE") && std::string(std::getenv("DIFFKV_VERBOSE")) == "1") {
        std::cerr << "[KVRuntimeManager] Initializing for " << n_layers << " layers..." << std::endl;
    }

    engines_.resize(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        engines_[l] = std::make_unique<NativeBlockPool>();
        if (!engines_[l]->initialize(n_slots, base_rank_, head_dim, kv_heads, desc_dim, buft)) {
            std::cerr << "[KVRuntimeManager] Error: Failed to initialize KVEngine for layer " << l << std::endl;
            return false;
        }
    }

    pager_ = std::make_unique<PagedKVStore>(gpu_budget_bytes_);
    
    ingest_manager_ = std::make_unique<StreamingSparseIngestManager>(
        micro_block_size_,
        recency_window_,
        short_context_threshold_,
        true // protect block zero
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

int KVRuntimeManager::get_layer_rank(int layer_idx) const {
    double ratio = (double)layer_idx / std::max(n_layers_, 1);
    if (ratio < 0.15) {
        // Boosted schedule for early layers (disabled by default, check env)
        if (const char* boost_env = std::getenv("DIFFKV_EARLY_LAYER_RANK_BOOST")) {
            if (std::string(boost_env) == "1") {
                return std::min(2 * base_rank_, 64);
            }
        }
        return base_rank_;
    } else if (ratio < 0.50) {
        return base_rank_;
    } else if (ratio < 0.79) {
        return std::max(6, (int)std::round(0.75 * base_rank_));
    } else {
        return std::max(8, (int)std::round(0.50 * base_rank_));
    }
}

void KVRuntimeManager::ingest_prefill(
    const std::vector<std::vector<float>>& k_layers,
    const std::vector<std::vector<float>>& v_layers,
    int chunk_len,
    int position_start,
    const std::vector<int32_t>& token_ids
) {
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
            pager_.get()
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
                }
            }
        }
    }

    // Register newly compressed blocks with the pager
    auto & blocks = ingest_manager_->get_blocks(0);
    for (auto & block : blocks) {
        if (block->state == BlockState::CompressedResident) {
            pager_->register_block(block.get(), engines_);
        }
    }
    
    // 3. Evict excess resident memory under budget
    pager_->maybe_evict(engines_);
}

void KVRuntimeManager::ingest_decode(
    const std::vector<std::vector<float>>& k_layers,
    const std::vector<std::vector<float>>& v_layers,
    int current_pos,
    const std::vector<int32_t>& token_ids
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
            pager_.get()
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
                }
            }
        }
    }
    
    // 4. Register any new compressed blocks
    auto & blocks = ingest_manager_->get_blocks(0);
    for (auto & block : blocks) {
        if (block->state == BlockState::CompressedResident) {
            pager_->register_block(block.get(), engines_);
        }
    }
    pager_->maybe_evict(engines_);
}

std::vector<int32_t> KVRuntimeManager::route_decode_slots(
    int current_pos,
    const std::vector<int32_t>& token_ids,
    const SessionSRLState& srl_state,
    const std::unordered_set<int32_t>& stop_token_ids,
    int srl_k_recency,
    int srl_k_lexical,
    int srl_k_graph,
    int srl_k_host,
    int active_slot
) const {
    std::vector<int32_t> host_candidates;
    std::unordered_set<int32_t> seen;

    // 0. Always include sink blocks (first/last blocks, critical for attention sinks)
    for (int32_t sink : srl_state.sink_blocks) {
        if (sink >= 0 && sink < active_slot) {
            host_candidates.push_back(sink);
            seen.insert(sink);
        }
    }

    const auto& ord = srl_state.ordered_slot_ids;
    int n_ord = static_cast<int>(ord.size());

    // 1. Recency window: latest srl_k_recency slots from ordered slot list
    int take_r = std::min(srl_k_recency, n_ord);
    for (int i = n_ord - take_r; i < n_ord; ++i) {
        int32_t slot = ord[i];
        if (slot >= 0 && slot < active_slot) {
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

    auto lex_scored = score_lexical_slots(srl_state.inverted_index, query_tokens, 0.999f);
    std::vector<int32_t> lexical_slots;
    for (int i = 0; i < std::min(srl_k_lexical, (int)lex_scored.size()); ++i) {
        int32_t slot = lex_scored[i].first;
        if (slot >= 0 && slot < active_slot) {
            lexical_slots.push_back(slot);
            if (!seen.count(slot)) {
                host_candidates.push_back(slot);
                seen.insert(slot);
            }
        }
    }

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
            if (gs > 0.0f && slot >= 0 && slot < active_slot) {
                gscore_slots.push_back({gs, slot});
            }
        }

        int take_g = std::min(srl_k_graph, (int)gscore_slots.size());
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

    // Pad with 0 up to srl_k_host
    while (host_candidates.size() < (size_t)srl_k_host) {
        host_candidates.push_back(0);
    }

    // Cap at srl_k_host
    if (host_candidates.size() > (size_t)srl_k_host) {
        host_candidates.resize(srl_k_host);
    }

    return host_candidates;
}

void KVRuntimeManager::wait_for_compressor() {
    // Gather all pool slots that are currently in Compressing state in any layer
    int n_slots = engines_[0]->get_U()->ne[2];
    std::vector<int> pending_slots;
    for (int pool_idx = 0; pool_idx < n_slots; ++pool_idx) {
        bool compressing = false;
        for (int l = 0; l < n_layers_; ++l) {
            if (engines_[l]->get_state_table().get(pool_idx) == BlockState::Compressing) {
                compressing = true;
                break;
            }
        }
        if (compressing) {
            pending_slots.push_back(pool_idx);
        }
    }

    // Second pass: wait for all pending slots with a global 5-second timeout
    // (prevents hanging on pathological cases like SVD thread crash)
    auto global_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (int pool_idx : pending_slots) {
        for (int l = 0; l < n_layers_; ++l) {
            while (std::chrono::steady_clock::now() < global_deadline) {
                BlockState st = engines_[l]->get_state_table().get(pool_idx);
                if (st != BlockState::Compressing) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            engines_[l]->upload_slot(pool_idx);
        }
    }

    // Sync block states across all layers
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

    // Register newly compressed blocks with the pager
    auto & blocks = ingest_manager_->get_blocks(0);
    for (auto & block : blocks) {
        if (block->state == BlockState::CompressedResident) {
            pager_->register_block(block.get(), engines_);
        }
    }
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
    auto & blocks = ingest_manager_->get_blocks(0);
    int F_test = engines_[0]->get_VK()->ne[0] * engines_[0]->get_VK()->ne[1];
    int kv_heads = engines_[0]->get_VK()->ne[1];
    
    for (size_t b = 0; b < blocks.size(); ++b) {
        auto & block = blocks[b];
        if (block->pool_idx == -1) continue; // paged out
        
        int slot_id = block->pool_idx;
        
        if (block->state == BlockState::CompressedResident || block->state == BlockState::CPUResident) {
            auto & engine = engines_[0];
            int rank = engine->get_U()->ne[0];
            int S_max = 64; // Block size
            
            std::vector<ggml_fp16_t> desc_f16(desc_dim);
            compute_descriptor(
                (const uint16_t*)engine->get_host_anchors_K() + slot_id * F_test,
                engine->get_host_U() + slot_id * S_max * rank,
                ggml_fp16_to_fp32(engine->get_host_U_scale()[slot_id]),
                (const uint16_t*)engine->get_host_VK() + slot_id * rank * F_test,
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
            
            ggml_backend_tensor_set(engines_[0]->get_desc_matrix(), desc.data(), slot_id * desc_dim * sizeof(float), desc_dim * sizeof(float));
        }
    }
}

void KVRuntimeManager::set_micro_block_size(int size) {
    micro_block_size_ = size;
    if (ingest_manager_) {
        ingest_manager_->set_micro_block_size(size);
    }
}

} // namespace diffkv
