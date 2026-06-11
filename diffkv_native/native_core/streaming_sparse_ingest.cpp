#include "native_core/streaming_sparse_ingest.hpp"
#include "native_core/paging/paged_kv_store.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace diffkv {

static std::vector<uint32_t> decode_utf8(const std::string& str) {
    std::vector<uint32_t> codepoints;
    size_t i = 0;
    while (i < str.size()) {
        uint8_t c = str[i];
        if (c <= 0x7F) {
            codepoints.push_back(c);
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 < str.size()) {
                codepoints.push_back(((c & 0x1F) << 6) | (str[i+1] & 0x3F));
                i += 2;
            } else { i++; }
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 < str.size()) {
                codepoints.push_back(((c & 0x0F) << 12) | ((str[i+1] & 0x3F) << 6) | (str[i+2] & 0x3F));
                i += 3;
            } else { i++; }
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 < str.size()) {
                codepoints.push_back(((c & 0x07) << 18) | ((str[i+1] & 0x3F) << 12) | ((str[i+2] & 0x3F) << 6) | (str[i+3] & 0x3F));
                i += 4;
            } else { i++; }
        } else {
            i++;
        }
    }
    return codepoints;
}

static bool check_sci_notation(const std::vector<uint32_t>& codepoints) {
    int state = 0;
    for (uint32_t cp : codepoints) {
        bool is_digit = (cp >= 0x30 && cp <= 0x39);
        bool is_e = (cp == 'e' || cp == 'E');
        bool is_sign = (cp == '+' || cp == '-');
        bool is_dot = (cp == '.');
        
        if (state == 0) {
            if (is_digit) state = 1;
        } else if (state == 1) {
            if (is_digit) {}
            else if (is_dot) state = 2;
            else if (is_e) state = 4;
            else state = 0;
        } else if (state == 2) {
            if (is_digit) state = 3;
            else if (is_e) state = 4;
            else state = 0;
        } else if (state == 3) {
            if (is_digit) {}
            else if (is_e) state = 4;
            else state = 0;
        } else if (state == 4) {
            if (is_sign) state = 5;
            else if (is_digit) state = 6;
            else state = 0;
        } else if (state == 5) {
            if (is_digit) state = 6;
            else state = 0;
        } else if (state == 6) {
            if (is_digit) {}
            else if (is_e) state = 4;
            else if (is_dot) state = 2;
            else state = 0;
        }
        
        if (state == 6) {
            return true;
        }
    }
    return false;
}

static bool check_unicode_math(const std::vector<uint32_t>& codepoints) {
    for (uint32_t cp : codepoints) {
        if (cp == 0x221A || cp == 0x2211 || cp == 0x222B || cp == 0x2202 ||
            cp == 0x03C0 || cp == 0x03A0 || cp == 0x03A3 || cp == 0x221E ||
            cp == 0x2264 || cp == 0x2265 || cp == 0x2260 || cp == 0x00B1 ||
            cp == 0x00F7 || cp == 0x00D7) {
            return true;
        }
    }
    return false;
}

static bool check_long_digits(const std::vector<uint32_t>& codepoints) {
    int consecutive = 0;
    for (uint32_t cp : codepoints) {
        if (cp >= 0x30 && cp <= 0x39) {
            consecutive++;
            if (consecutive >= 5) return true;
        } else {
            consecutive = 0;
        }
    }
    return false;
}

static bool check_short_digits(const std::vector<uint32_t>& codepoints) {
    int consecutive = 0;
    for (uint32_t cp : codepoints) {
        if (cp >= 0x30 && cp <= 0x39) {
            consecutive++;
            if (consecutive >= 2) return true;
        } else {
            consecutive = 0;
        }
    }
    return false;
}

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

StreamingSparseIngestManager::StreamingSparseIngestManager(
    int micro_block_size,
    int recency_window,
    int short_context_threshold,
    bool protect_block_zero
) : micro_block_size_(micro_block_size),
    recency_window_(recency_window),
    short_context_threshold_(short_context_threshold),
    protect_block_zero_(protect_block_zero) {
    
    stopwords_ = {
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
}

StreamingSparseIngestManager::~StreamingSparseIngestManager() {
    clear();
}

void StreamingSparseIngestManager::initialize(int n_layers, const DiffKVModel* model) {
    model_ = model;
    n_layers_ = n_layers;
    layers_blocks_.resize(n_layers);
    clear();
}

void StreamingSparseIngestManager::clear() {
    for (auto & blocks : layers_blocks_) {
        blocks.clear();
    }
    query_words_.clear();
    stats_ = Stats();
    session_token_ids_.clear();
}

void StreamingSparseIngestManager::rollback(int target_len, std::vector<std::unique_ptr<NativeBlockPool>>& engines) {
    if (target_len < (int)session_token_ids_.size()) {
        session_token_ids_.resize(target_len);
    }
    for (int l = 0; l < n_layers_; ++l) {
        auto & blocks = layers_blocks_[l];
        std::vector<std::unique_ptr<StreamingKVBlock>> kept;
        for (auto & block : blocks) {
            if (block->anchor_idx >= target_len) {
                if (block->pool_idx != -1) {
                    engines[l]->free_slot(block->pool_idx);
                }
                continue;
            }
            int block_tokens_count = block->token_indices.size();
            if (block->anchor_idx + block_tokens_count > target_len) {
                // Truncate block
                int keep = target_len - block->anchor_idx;
                block->token_indices.resize(keep);
                int keep_active = keep - 1;
                
                int F_test = engines[l]->get_VK()->ne[0] * engines[l]->get_VK()->ne[1];
                if (keep_active > 0) {
                    block->active_k.resize(keep_active * F_test);
                    block->active_v.resize(keep_active * F_test);
                } else {
                    block->active_k.clear();
                    block->active_v.clear();
                }
                block->svd_k.clear();
                block->svd_v.clear();
                
                if (block->pool_idx != -1) {
                    // Update sequence length in engine
                    int32_t slen = keep_active;
                    ggml_backend_tensor_set(engines[l]->get_seq_lens(), &slen, block->pool_idx * sizeof(int32_t), sizeof(int32_t));
                }
            }
            kept.push_back(std::move(block));
        }
        blocks = std::move(kept);
    }
}

bool StreamingSparseIngestManager::should_skip_compression(int anchor_idx, const std::vector<int32_t>& block_tokens) const {
    if (!model_) return false;
    
    try {
        std::string block_text = model_->detokenize(block_tokens);
        std::vector<uint32_t> codepoints = decode_utf8(block_text);
        
        // Rule 1: Long digits
        if (check_long_digits(codepoints)) {
            return true;
        }
        
        // Rule 2: Scientific notation
        if (check_sci_notation(codepoints)) {
            return true;
        }
        
        // Rule 3: Unicode math symbols
        if (check_unicode_math(codepoints)) {
            return true;
        }
        
        // Rule 4: Short digits with query word overlap
        if (check_short_digits(codepoints)) {
            if (!query_words_.empty()) {
                std::unordered_set<std::string> block_words = extract_words(block_text, stopwords_);
                for (const auto & w : block_words) {
                    if (query_words_.find(w) != query_words_.end()) {
                        return true;
                    }
                }
            }
        }
    } catch (...) {}
    
    return false;
}

int StreamingSparseIngestManager::next_anchor_idx(int layer_idx) const {
    if (layers_blocks_[layer_idx].empty()) {
        return 0;
    }
    auto & last = layers_blocks_[layer_idx].back();
    return last->anchor_idx + last->token_count();
}

void StreamingSparseIngestManager::ingest_chunk(
    int layer_idx,
    const float* k_chunk,
    const float* v_chunk,
    int chunk_len,
    int position_start,
    const std::vector<int32_t>& token_ids,
    std::vector<std::unique_ptr<NativeBlockPool>>& engines,
    AsyncCompressor& compressor,
    int rank,
    PagedKVStore* pager
) {
    // Append newly ingested chunk of token IDs to session_token_ids_ on layer 0.
    if (layer_idx == 0) {
        if (position_start + chunk_len > (int)session_token_ids_.size()) {
            session_token_ids_.resize(position_start + chunk_len);
        }
        for (int t = 0; t < chunk_len; ++t) {
            // token_ids is the full prompt vector; use position_start as offset
            session_token_ids_[position_start + t] = token_ids[position_start + t];
        }
    }

    auto & blocks = layers_blocks_[layer_idx];
    int F_test = engines[layer_idx]->get_VK()->ne[0] * engines[layer_idx]->get_VK()->ne[1];
    
    for (int t = 0; t < chunk_len; ++t) {
        if (blocks.empty() || blocks.back()->token_count() == 1 + micro_block_size_) {
            // Allocate a new slot index
            int slot_id = engines[layer_idx]->allocate_slot();
            if (slot_id == -1 && pager) {
                pager->maybe_evict(engines);
                slot_id = engines[layer_idx]->allocate_slot();
            }
            
            auto new_block = std::make_unique<StreamingKVBlock>();
            new_block->anchor_idx = position_start + t;
            new_block->micro_block_size = micro_block_size_;
            new_block->pool_idx = slot_id;
            new_block->state = BlockState::DenseResident;
            
            // Extract anchor token
            new_block->anchor_k.resize(F_test);
            new_block->anchor_v.resize(F_test);
            std::memcpy(new_block->anchor_k.data(), k_chunk + t * F_test, F_test * sizeof(float));
            std::memcpy(new_block->anchor_v.data(), v_chunk + t * F_test, F_test * sizeof(float));
            
            if (slot_id != -1) {
                std::vector<ggml_fp16_t> k_fp16(F_test);
                std::vector<ggml_fp16_t> v_fp16(F_test);
                for (int i = 0; i < F_test; ++i) {
                    k_fp16[i] = ggml_fp32_to_fp16(new_block->anchor_k[i]);
                    v_fp16[i] = ggml_fp32_to_fp16(new_block->anchor_v[i]);
                }
                ggml_backend_tensor_set(engines[layer_idx]->get_anchors_K(), k_fp16.data(), slot_id * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
                ggml_backend_tensor_set(engines[layer_idx]->get_anchors_V(), v_fp16.data(), slot_id * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
                
                int32_t anchor_pos = new_block->anchor_idx;
                ggml_backend_tensor_set(engines[layer_idx]->get_anchor_positions(), &anchor_pos, slot_id * sizeof(int32_t), sizeof(int32_t));
                engines[layer_idx]->get_state_table().transition(slot_id, BlockState::Freed, BlockState::DenseResident);
            }
            
            new_block->token_indices.push_back(position_start + t);
            blocks.push_back(std::move(new_block));
            stats_.total_blocks_created++;
        } else {
            // Append token to the current active block
            auto & block = blocks.back();
            int active_offset = block->active_k.size();
            block->active_k.resize(active_offset + F_test);
            block->active_v.resize(active_offset + F_test);
            std::memcpy(block->active_k.data() + active_offset, k_chunk + t * F_test, F_test * sizeof(float));
            std::memcpy(block->active_v.data() + active_offset, v_chunk + t * F_test, F_test * sizeof(float));
            block->token_indices.push_back(position_start + t);
        }
    }
    
    // Scan and submit blocks that have fallen out of the recency window
    int current_seq_len = position_start + chunk_len;
    bool immediate_prefill = true;
    if (const char* env_p = std::getenv("DIFFKV_IMMEDIATE_PREFILL_COMPRESS")) {
        if (std::string(env_p) == "0") {
            immediate_prefill = false;
        }
    }

    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        auto & b = blocks[idx];
        if (b->state == BlockState::DenseResident && b->token_count() == 1 + micro_block_size_ && !b->skip_compression) {
            bool skip = false;
            if (b->anchor_idx == 0 && protect_block_zero_) {
                skip = true;
            } else if (b->anchor_idx + b->token_count() < short_context_threshold_) {
                skip = true;
            } else {
                if (b->anchor_idx + b->token_count() <= (int)session_token_ids_.size()) {
                    std::vector<int32_t> block_toks(
                        session_token_ids_.begin() + b->anchor_idx,
                        session_token_ids_.begin() + b->anchor_idx + b->token_count()
                    );
                    skip = should_skip_compression(b->anchor_idx, block_toks);
                } else {
                    skip = true;
                }
            }

            if (skip) {
                b->skip_compression = true;
            } else {
                bool should_compress = false;
                if (chunk_len > 1 && immediate_prefill) {
                    should_compress = true;
                } else if (b->anchor_idx + b->token_count() < current_seq_len - recency_window_) {
                    should_compress = true;
                }

                if (should_compress) {
                    submit_block_for_compression(layer_idx, idx, engines, compressor, rank);
                }
            }
        }
    }

    // Recount dense tokens
    uint64_t current_dense = 0;
    for (auto & b : blocks) {
        if (b->state == BlockState::DenseResident) {
            current_dense += b->token_count();
        } else {
            current_dense += 1; // anchor is always dense
        }
    }
    stats_.peak_dense_tokens = std::max(stats_.peak_dense_tokens, current_dense);
}

void StreamingSparseIngestManager::submit_block_for_compression(
    int layer_idx,
    int block_idx,
    std::vector<std::unique_ptr<NativeBlockPool>>& engines,
    AsyncCompressor& compressor,
    int rank
) {
    auto & block = layers_blocks_[layer_idx][block_idx];
    if (block->state != BlockState::DenseResident || block->pool_idx == -1) {
        return;
    }
    
    int F_test = engines[layer_idx]->get_VK()->ne[0] * engines[layer_idx]->get_VK()->ne[1];
    int head_dim = engines[layer_idx]->get_VK()->ne[0];
    int slot_id = block->pool_idx;
    
    // Construct contiguous svd_k and svd_v
    int S_total = block->token_count();
    block->svd_k.resize(S_total * F_test);
    block->svd_v.resize(S_total * F_test);
    std::memcpy(block->svd_k.data(), block->anchor_k.data(), F_test * sizeof(float));
    std::memcpy(block->svd_v.data(), block->anchor_v.data(), F_test * sizeof(float));
    std::memcpy(block->svd_k.data() + F_test, block->active_k.data(), (S_total - 1) * F_test * sizeof(float));
    std::memcpy(block->svd_v.data() + F_test, block->active_v.data(), (S_total - 1) * F_test * sizeof(float));
    
    // Transition state in table
    engines[layer_idx]->get_state_table().transition(slot_id, BlockState::DenseResident, BlockState::Compressing);
    block->state = BlockState::Compressing;
    
    CompressJob job;
    job.session_id = 42; // standard session ID
    job.block_id = slot_id;
    job.block_size = S_total;
    job.feat_dim = F_test;
    job.rank = rank;
    job.head_dim = head_dim;
    job.raw_k_ptr = block->svd_k.data();
    job.raw_v_ptr = block->svd_v.data();
    
    // Outputs in block pool
    job.out_u_ptr = reinterpret_cast<int8_t*>(engines[layer_idx]->get_U()->data) + slot_id * 64 * rank;
    job.out_u_scale = reinterpret_cast<ggml_fp16_t*>(engines[layer_idx]->get_U_scale()->data) + slot_id;
    job.out_vk_ptr = reinterpret_cast<ggml_fp16_t*>(engines[layer_idx]->get_VK()->data) + slot_id * rank * F_test;
    job.out_vv_ptr = reinterpret_cast<ggml_fp16_t*>(engines[layer_idx]->get_VV()->data) + slot_id * rank * F_test;
    job.out_scale = reinterpret_cast<ggml_fp16_t*>(engines[layer_idx]->get_scales()->data) + slot_id;
    job.out_anchor_k = reinterpret_cast<ggml_fp16_t*>(engines[layer_idx]->get_anchors_K()->data) + slot_id * F_test;
    job.out_anchor_v = reinterpret_cast<ggml_fp16_t*>(engines[layer_idx]->get_anchors_V()->data) + slot_id * F_test;
    job.state_table = &engines[layer_idx]->get_state_table();
    
    bool async_svd = true;
    if (const char* env_async = std::getenv("DIFFKV_ASYNC_SVD")) {
        std::string s(env_async);
        if (s == "0" || s == "false" || s == "off") {
            async_svd = false;
        }
    } else {
#ifdef __APPLE__
        async_svd = false;
#endif
    }

    if (async_svd) {
        bool submitted = compressor.submit(job);
        if (!submitted) {
            engines[layer_idx]->get_state_table().transition(slot_id, BlockState::Compressing, BlockState::DenseResident);
            block->state = BlockState::DenseResident;
        } else {
            stats_.total_compressed++;
        }
    } else {
        compressor.compress_sync(job);
        block->state = engines[layer_idx]->get_state_table().get(slot_id);
        stats_.total_compressed++;
    }
}

} // namespace diffkv
