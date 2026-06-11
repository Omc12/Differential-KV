#include "native_core/kv_runtime_manager.hpp"
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

    std::cerr << "[KVRuntimeManager] Initializing for " << n_layers << " layers..." << std::endl;

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
    const InvertedIndex& inverted_index,
    const std::unordered_set<int32_t>& stop_token_ids,
    int srl_k_recency,
    int srl_k_lexical,
    int srl_k_graph,
    int srl_k_host,
    int active_slot
) const {
    std::vector<int32_t> host_candidates;
    std::unordered_set<int32_t> seen;

    // 1. Recency window: latest srl_k_recency slots
    for (int i = 0; i < srl_k_recency; ++i) {
        int slot = active_slot - 1 - i;
        if (slot >= 0) {
            host_candidates.push_back(slot);
            seen.insert(slot);
        }
    }

    // 2. Lexical search slots
    int query_start = std::max(0, current_pos - 3);
    std::vector<int32_t> query_tokens;
    for (int i = query_start; i < current_pos; ++i) {
        if (i < (int)token_ids.size()) {
            query_tokens.push_back(token_ids[i]);
        }
    }

    std::vector<int32_t> lexical_slots = const_cast<InvertedIndex&>(inverted_index).search(query_tokens, srl_k_lexical);
    for (int32_t slot : lexical_slots) {
        if (slot >= 0 && slot < active_slot) {
            if (!seen.count(slot)) {
                host_candidates.push_back(slot);
                seen.insert(slot);
            }
        }
    }

    // 3. Graph adjacency slots
    std::vector<int32_t> graph_slots;
    for (int32_t seed : lexical_slots) {
        if (seed - 1 >= 0 && seed - 1 < active_slot) {
            graph_slots.push_back(seed - 1);
        }
        if (seed + 1 >= 0 && seed + 1 < active_slot) {
            graph_slots.push_back(seed + 1);
        }
    }

    int graph_added = 0;
    for (int32_t slot : graph_slots) {
        if (graph_added >= srl_k_graph) break;
        if (!seen.count(slot)) {
            host_candidates.push_back(slot);
            seen.insert(slot);
            graph_added++;
        }
    }

    // Pad with 0 up to srl_k_host
    while (host_candidates.size() < (size_t)srl_k_host) {
        host_candidates.push_back(0);
    }

    return host_candidates;
}

void KVRuntimeManager::wait_for_compressor() {
    for (int l = 0; l < n_layers_; ++l) {
        auto & blocks = ingest_manager_->get_blocks(l);
        for (auto & block : blocks) {
            if (block->state == BlockState::Compressing && block->pool_idx != -1) {
                auto start_time = std::chrono::steady_clock::now();
                while (true) {
                    BlockState current_state = engines_[l]->get_state_table().get(block->pool_idx);
                    if (current_state != BlockState::Compressing) {
                        block->state = current_state;
                        break;
                    }
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time
                    ).count();
                    if (elapsed > 60000) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
    for (int32_t block_idx : active_slots) {
        if (block_idx >= 0 && block_idx < (int)blocks.size()) {
            auto & block = blocks[block_idx];
            pager_->touch(block.get(), engines_);
        }
    }
}

void KVRuntimeManager::update_descriptors(const std::vector<float>& W_proj_host, int desc_dim, int head_dim) {
    auto & blocks = ingest_manager_->get_blocks(0);
    int F_test = engines_[0]->get_VK()->ne[0] * engines_[0]->get_VK()->ne[1];
    
    for (size_t b = 0; b < blocks.size(); ++b) {
        auto & block = blocks[b];
        if (block->pool_idx == -1) continue; // paged out
        
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
        
        ggml_backend_tensor_set(engines_[0]->get_desc_matrix(), desc.data(), block->pool_idx * desc_dim * sizeof(float), desc_dim * sizeof(float));
    }
}

} // namespace diffkv
