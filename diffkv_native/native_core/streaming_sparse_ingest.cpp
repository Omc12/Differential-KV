#include "native_core/streaming_sparse_ingest.hpp"
#include "native_core/paging/paged_kv_store.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <regex>

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
    last_compression_scan_idx_.assign(n_layers, 0);
    clear();
}

void StreamingSparseIngestManager::clear() {
    for (auto & blocks : layers_blocks_) {
        blocks.clear();
    }
    query_words_.clear();
    stats_ = Stats();
    session_token_ids_.clear();
    last_compression_scan_idx_.assign(n_layers_, 0);  // reset scan pointer on session clear
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
                
                int F_test = engines[l]->get_head_dim() * engines[l]->get_kv_heads();
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
                    engines[l]->get_host_seq_lens()[block->pool_idx] = slen;
                    ggml_backend_tensor_set(engines[l]->get_seq_lens(), &slen, block->pool_idx * sizeof(int32_t), sizeof(int32_t));
                    block->device_synced = false;
                }
            }
            kept.push_back(std::move(block));
        }
        blocks = std::move(kept);
        if (last_compression_scan_idx_[l] > blocks.size()) {
            last_compression_scan_idx_[l] = blocks.size();
        }
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

        // RECONSTRUCTION FIX (F14): Rules 3b-3f mirror ACTIVE_RUNTIME
        // streaming_sparse_ingest.py:98-112 (_RE_LATEX_MATH / _RE_ASCII_EQUATION /
        // _RE_DEFINITIONS / _RE_CLAIMS / _RE_ACRONYMS) verbatim. Compiled once (static).
        // Patterns are ASCII so they run on block_text directly.
        static const std::regex re_latex(
            R"(\$\$|\\\[|\\\(|\\begin\{(?:equation|align|gather|math|displaymath)\}|\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|sum|int|prod|partial|nabla|hbar|infty|approx|neq|le|ge|times|div|cdot|sqrt|frac)\b|_\{[^\}]+\}|\^[^\}]+\})");
        static const std::regex re_ascii_eq(
            R"(\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[-+]?[a-zA-Z0-9_.\(\)\+\-\*\/]+)");
        static const std::regex re_definitions(
            R"(\b(?:is|are|we)\s+(?:defined|referred|called|known)\s+(?:as|by)\b|\brefers?\s+to\b|\b(?:denotes?|stands\s+for|represents?)\b|\bwe\s+define\b|\b(?:let\s+us|let)\s+define\b)",
            std::regex::icase);
        static const std::regex re_claims(
            R"(\b(?:theorem|lemma|proposition|corollary|conjecture|hypothesis|proof)\s+\d+(?:\.\d+)*\b|\bour\s+main\s+contribution\b|\bwe\s+(?:prove|show|demonstrate|argue|conclude|find)\s+that\b|\bour\s+(?:results|analysis)\s+show\b)",
            std::regex::icase);
        static const std::regex re_acronyms(R"(\b[A-Z]{2,}\b)");

        // Rule 3b: LaTeX math formula block
        if (std::regex_search(block_text, re_latex)) {
            return true;
        }
        // Rule 3c: ASCII equation statement
        if (std::regex_search(block_text, re_ascii_eq)) {
            return true;
        }
        // Rule 3d: Verbatim definitions
        if (std::regex_search(block_text, re_definitions)) {
            return true;
        }
        // Rule 3e: Formal claims / theorems
        if (std::regex_search(block_text, re_claims)) {
            return true;
        }
        // Rule 3f: Acronym density (>= 3 distinct uppercase acronyms)
        {
            std::unordered_set<std::string> acronyms;
            for (std::sregex_iterator it(block_text.begin(), block_text.end(), re_acronyms), end;
                 it != end; ++it) {
                acronyms.insert(it->str());
            }
            if (acronyms.size() >= 3) {
                return true;
            }
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

bool StreamingSparseIngestManager::should_boost_compression_rank(int anchor_idx, const std::vector<int32_t>& block_tokens) const {
    if (!model_) return false;
    
    try {
        std::string block_text = model_->detokenize(block_tokens);
        
        // 1. Any digits
        for (char c : block_text) {
            if (c >= '0' && c <= '9') {
                return true;
            }
        }
        
        // 2. Math formula markers: +, -, *, /, =, or LaTeX markers
        static const std::regex re_math_boost(
            R"([\+\-\*\/=]|\$\$|\\\[|\\\(|\\begin\{|\\alpha|\\beta|\\gamma|\\delta|\\sum|\\int|\\frac|\\sqrt|_\{|\^)");
        if (std::regex_search(block_text, re_math_boost)) {
            return true;
        }
        
        // 3. Key definition keywords
        static const std::regex re_definitions_boost(
            R"(\b(?:is|are|we)\s+(?:defined|referred|called|known)\s+(?:as|by)\b|\brefers?\s+to\b|\b(?:denotes?|stands\s+for|represents?)\b|\bwe\s+define\b|\b(?:let\s+us|let)\s+define\b)",
            std::regex::icase);
        if (std::regex_search(block_text, re_definitions_boost)) {
            return true;
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

// Content-based compression skip (long-digit / equation / acronym heuristic) is OFF
// by default to match ACTIVE_RUNTIME (MLX), which compresses every block leaving the
// recency window unconditionally. Set DIFFKV_SKIP_COMPRESSION_HEURISTIC=1 to restore
// it (factual/NIAH exact-retrieval path).
static bool skip_compression_heuristic_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("DIFFKV_SKIP_COMPRESSION_HEURISTIC");
        return e && (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "on");
    }();
    return enabled;
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
    PagedKVStore* pager,
    SessionSRLState* srl_state
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
    int F_test = engines[layer_idx]->get_head_dim() * engines[layer_idx]->get_kv_heads();
    
    int min_slot = -1;
    int max_slot = -1;

    for (int t = 0; t < chunk_len; ++t) {
        if (blocks.empty() || blocks.back()->token_count() == 1 + micro_block_size_) {
            // Allocate a new slot index
            int slot_id = engines[layer_idx]->allocate_slot();
            if (slot_id == -1 && pager) {
                pager->maybe_evict(engines, srl_state);
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
                if (min_slot == -1 || slot_id < min_slot) min_slot = slot_id;
                if (max_slot == -1 || slot_id > max_slot) max_slot = slot_id;

                std::vector<ggml_fp16_t> k_fp16(F_test);
                std::vector<ggml_fp16_t> v_fp16(F_test);
                for (int i = 0; i < F_test; ++i) {
                    k_fp16[i] = ggml_fp32_to_fp16(new_block->anchor_k[i]);
                    v_fp16[i] = ggml_fp32_to_fp16(new_block->anchor_v[i]);
                }
                
                // Keep host-side mirrors in sync
                std::copy(k_fp16.begin(), k_fp16.end(), engines[layer_idx]->get_host_anchors_K(slot_id));
                std::copy(v_fp16.begin(), v_fp16.end(), engines[layer_idx]->get_host_anchors_V(slot_id));
                int32_t anchor_pos = new_block->anchor_idx;
                engines[layer_idx]->get_host_anchor_positions()[slot_id] = anchor_pos;

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
    
    // Scan and submit blocks that have fallen out of the recency window.
    // Start from last_compression_scan_idx_[layer_idx] to avoid re-scanning
    // already-processed blocks (O(N) → amortized O(1) per block).
    int current_seq_len = position_start + chunk_len;
    bool immediate_prefill = false;
    if (const char* env_p = std::getenv("DIFFKV_IMMEDIATE_PREFILL_COMPRESS")) {
        if (std::string(env_p) == "1") {
            immediate_prefill = true;
        }
    }

    // RECONSTRUCTION FIX (F13): this MUST match the decode-side engage_threshold
    // (main.cpp uses 2048 for the same DIFFKV_ENGAGE_THRESHOLD env var). Previously
    // this defaulted to 4096 while decode defaulted to 2048 — so prompts in [2048,4096)
    // switched to the SPARSE decode path (L>=2048) but had ZERO compressed blocks
    // (bypass kept everything dense), causing ~4x dense-KV RAM vs ACTIVE_RUNTIME AND
    // sparse routing over an empty pool. ACTIVE_RUNTIME has no such global bypass — it
    // compresses blocks during prefill regardless of total length.
    int engage_threshold = 4096;
    if (const char* env_et = std::getenv("DIFFKV_ENGAGE_THRESHOLD")) {
        engage_threshold = std::stoi(env_et);
    }
    bool bypass_diffkv = ((int)token_ids.size() < engage_threshold);

    size_t scan_start = last_compression_scan_idx_[layer_idx];
    size_t scan_end   = layers_blocks_[layer_idx].size();
    auto & scan_blocks = layers_blocks_[layer_idx];

    for (size_t idx = scan_start; idx < scan_end; ++idx) {
        auto & b = scan_blocks[idx];
        if (b->state == BlockState::DenseResident && b->token_count() == 1 + micro_block_size_ && !b->skip_compression) {
            bool skip = false;
            if (bypass_diffkv) {
                skip = true;
            } else if (b->anchor_idx == 0 && protect_block_zero_) {
                skip = true;
            } else if (b->anchor_idx + b->token_count() < short_context_threshold_) {
                skip = true;
            } else if (skip_compression_heuristic_enabled() && !b->skip_compression_evaluated) {
                // CONTENT-SKIP HEURISTIC — DISABLED BY DEFAULT (MLX parity).
                //
                // ACTIVE_RUNTIME/mlx_diffkv_wrapper.py compresses every block that falls
                // out of the recency window unconditionally (_compress_eligible_blocks →
                // _flush_oldest_block, no content test). The old content heuristic below
                // (long-digit / sci-notation / unicode-math / LaTeX / ascii-equation /
                // definitions / claims / acronym-density) permanently pinned huge fractions
                // of technical prose DENSE — on a 13k-token technical prompt it kept 31/51
                // blocks uncompressed, which (a) bloats dense-KV RAM and (b) leaves empty
                // pool slots that the MLX-parity "attend all blocks" decode then attends as
                // garbage → gibberish output. Default off matches MLX; re-enable for the
                // factual/NIAH exact-retrieval path via DIFFKV_SKIP_COMPRESSION_HEURISTIC=1.
                //
                // should_skip_compression runs 6 std::regex searches — expensive.
                // Cache the result: once evaluated, never re-run (block text is immutable).
                if (b->anchor_idx + b->token_count() <= (int)session_token_ids_.size()) {
                    std::vector<int32_t> block_toks(
                        session_token_ids_.begin() + b->anchor_idx,
                        session_token_ids_.begin() + b->anchor_idx + b->token_count()
                    );
                    skip = should_skip_compression(b->anchor_idx, block_toks);
                } else {
                    skip = true;
                }
                b->skip_compression_evaluated = true; // cache: never re-run regex for this block
            }
            // If skip_compression_evaluated && !skip_compression: block should be compressed,
            // just not ready yet (recency window). Fall through to the submit logic below.

            if (skip) {
                b->skip_compression = true;
                // Advance scan pointer past this permanently-skipped block
                if (idx == last_compression_scan_idx_[layer_idx]) {
                    last_compression_scan_idx_[layer_idx]++;
                }
            } else {
                bool should_compress = false;
                if (chunk_len > 1 && immediate_prefill) {
                    should_compress = true;
                } else if (b->anchor_idx + b->token_count() < current_seq_len - recency_window_) {
                    should_compress = true;
                }

                if (should_compress) {
                    submit_block_for_compression(layer_idx, idx, engines, compressor, rank);
                    // Advance scan pointer once this block is submitted (won't need re-scanning)
                    if (idx == last_compression_scan_idx_[layer_idx]) {
                        last_compression_scan_idx_[layer_idx]++;
                    }
                }
            }
        } else if (b->state != BlockState::DenseResident && idx == last_compression_scan_idx_[layer_idx]) {
            // Block already compressed or skipped — advance pointer
            last_compression_scan_idx_[layer_idx]++;
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

    if (min_slot != -1 && max_slot != -1) {
        // Upload anchors K and V slot-by-slot (since dynamic host pools are non-contiguous)
        for (int slot_id = min_slot; slot_id <= max_slot; ++slot_id) {
            const ggml_fp16_t* slot_ak = engines[layer_idx]->get_host_anchors_K(slot_id);
            const ggml_fp16_t* slot_av = engines[layer_idx]->get_host_anchors_V(slot_id);
            if (slot_ak) {
                ggml_backend_tensor_set(
                    engines[layer_idx]->get_anchors_K(),
                    slot_ak,
                    slot_id * F_test * sizeof(ggml_fp16_t),
                    F_test * sizeof(ggml_fp16_t)
                );
            }
            if (slot_av) {
                ggml_backend_tensor_set(
                    engines[layer_idx]->get_anchors_V(),
                    slot_av,
                    slot_id * F_test * sizeof(ggml_fp16_t),
                    F_test * sizeof(ggml_fp16_t)
                );
            }
        }
        // Batch upload anchor positions to GPU (this is still flat/contiguous on host)
        int count = max_slot - min_slot + 1;
        ggml_backend_tensor_set(
            engines[layer_idx]->get_anchor_positions(),
            engines[layer_idx]->get_host_anchor_positions() + min_slot,
            min_slot * sizeof(int32_t),
            count * sizeof(int32_t)
        );
    }
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
    
    int F_test = engines[layer_idx]->get_head_dim() * engines[layer_idx]->get_kv_heads();
    int head_dim = engines[layer_idx]->get_head_dim();
    int slot_id = block->pool_idx;
    
    // Construct contiguous svd_k and svd_v
    int S_total = block->token_count();
    block->svd_k.resize(S_total * F_test);
    block->svd_v.resize(S_total * F_test);
    std::memcpy(block->svd_k.data(), block->anchor_k.data(), F_test * sizeof(float));
    std::memcpy(block->svd_v.data(), block->anchor_v.data(), F_test * sizeof(float));
    std::memcpy(block->svd_k.data() + F_test, block->active_k.data(), (S_total - 1) * F_test * sizeof(float));
    std::memcpy(block->svd_v.data() + F_test, block->active_v.data(), (S_total - 1) * F_test * sizeof(float));

    // ── PER-TOKEN ROTATION FIX (gated: DIFFKV_ROTATE_AT_INGEST) ────────────────────────────────
    // The decode rotates the WHOLE reconstructed block at the anchor position (anchor_idx),
    // collapsing all within-block positions onto the anchor → at long context the compressed
    // attention is too lossy → gibberish. RoPE rotations compose additively, so pre-rotate each
    // token here at its WITHIN-BLOCK offset t (K only; V is not RoPE'd). Combined with the decode's
    // anchor-position rotation the net is rotate(K_t, t + anchor_idx) = the true absolute position
    // for every token — matching MLX's per-token ingest rotation. Baked in BEFORE the landmark
    // swap, so it also corrects the landmark's position. Only valid when the decode rotates
    // (has_rope); otherwise it would not compose to the absolute position, so skip.
    // DEFAULT ON (fixes long-context compressed-attn gibberish); DIFFKV_NO_ROTATE_AT_INGEST to disable.
    static const bool rotate_at_ingest = (std::getenv("DIFFKV_NO_ROTATE_AT_INGEST") == nullptr);
    if (rotate_at_ingest && engines[layer_idx]->get_has_rope()) {
        const int D = head_dim;
        const int nkv = (D > 0) ? (F_test / D) : 1;
        const int half_d = D / 2;
        const float freq = engines[layer_idx]->get_rope_freq_base();
        for (int t = 0; t < S_total; ++t) {
            for (int kv = 0; kv < nkv; ++kv) {
                float* vec = block->svd_k.data() + (size_t)t * F_test + (size_t)kv * D;
                for (int d = 0; d < half_d; ++d) {
                    float theta = 1.0f / std::pow(freq, (2.0f * d) / D);
                    float angle = (float)t * theta;
                    float cos_a = std::cos(angle), sin_a = std::sin(angle);
                    float x = vec[d], y = vec[d + half_d];
                    vec[d]         = x * cos_a - y * sin_a;
                    vec[d + half_d]= y * cos_a + x * sin_a;
                }
            }
        }
    }
    
    // Transition state in table
    engines[layer_idx]->get_state_table().transition(slot_id, BlockState::DenseResident, BlockState::Compressing);
    block->state = BlockState::Compressing;

    // Check if block qualifies for rank boosting (1.5x) to prevent Precision Loss
    bool boost = false;
    if (block->anchor_idx + block->token_count() <= (int)session_token_ids_.size()) {
        std::vector<int32_t> block_toks(
            session_token_ids_.begin() + block->anchor_idx,
            session_token_ids_.begin() + block->anchor_idx + block->token_count()
        );
        boost = should_boost_compression_rank(block->anchor_idx, block_toks);
    }

    int pool_rank = engines[layer_idx]->get_rank();
    int svd_rank = rank;
    if (boost) {
        svd_rank = (int)std::ceil(rank * 1.5f);
        int seq_len = S_total - 1;
        if (svd_rank > seq_len) {
            svd_rank = seq_len;
        }
        if (svd_rank > pool_rank) {
            svd_rank = pool_rank;
        }
    }
    
    CompressJob job;
    job.session_id = active_session_id_.empty() ? "42" : active_session_id_;
    job.block_id = slot_id;
    job.block_size = S_total;
    job.feat_dim = F_test;
    job.rank = svd_rank;
    job.pool_rank = pool_rank;
    // Bug 10 fix (write-side): pool_block_size and out_u_ptr must use the pool's fixed
    // S_max, not micro_block_size_ which may have been updated by set_micro_block_size().
    // Using micro_block_size_ here would write U data at a stride that doesn't match the
    // pool tensor layout, corrupting every subsequent U read from execute_cpu_attention.
    const int pool_s_max = engines[layer_idx]->get_S_max();
    job.pool_block_size = pool_s_max;
    job.head_dim = head_dim;
    job.anchor_idx = block->anchor_idx;
    job.raw_k_ptr = block->svd_k.data();
    job.raw_v_ptr = block->svd_v.data();
    job.token_ids = session_token_ids_.data() + block->anchor_idx;
    job.stop_token_ids = stop_token_ids_;

    if (engines[layer_idx]->get_host_U(slot_id) == nullptr) {
        // Skip low-rank path: allocate temporary local buffers for this job
        job.u_buf.resize(pool_s_max * pool_rank, 0);
        job.u_scale_buf.resize(1, ggml_fp32_to_fp16(0.0f));
        job.u_row_scale_buf.resize(pool_s_max, ggml_fp32_to_fp16(0.0f));
        job.vk_buf.resize(pool_rank * F_test, ggml_fp32_to_fp16(0.0f));
        job.vv_buf.resize(pool_rank * F_test, ggml_fp32_to_fp16(0.0f));
        
        job.out_u_ptr = job.u_buf.data();
        job.out_u_scale = job.u_scale_buf.data();
        job.out_u_row_scale = job.u_row_scale_buf.data();
        job.out_vk_ptr = job.vk_buf.data();
        job.out_vv_ptr = job.vv_buf.data();
        
        const int MR = NativeBlockPool::MAX_RESIDUAL;
        job.res_K_pos_buf.resize(MR, -1);
        job.res_V_pos_buf.resize(MR, -1);
        job.res_K_val_buf.resize(MR * F_test, ggml_fp32_to_fp16(0.0f));
        job.res_V_val_buf.resize(MR * F_test, ggml_fp32_to_fp16(0.0f));
        
        job.out_res_K_pos = job.res_K_pos_buf.data();
        job.out_res_V_pos = job.res_V_pos_buf.data();
        job.out_res_K_val = job.res_K_val_buf.data();
        job.out_res_V_val = job.res_V_val_buf.data();
    } else {
        job.out_u_ptr = engines[layer_idx]->get_host_U(slot_id);
        job.out_u_scale = engines[layer_idx]->get_host_U_scale() + slot_id;
        job.out_u_row_scale = engines[layer_idx]->get_host_U_row_scale(slot_id);
        job.out_vk_ptr = engines[layer_idx]->get_host_VK(slot_id);
        job.out_vv_ptr = engines[layer_idx]->get_host_VV(slot_id);
        
        job.out_res_K_pos = engines[layer_idx]->get_host_res_K_pos(slot_id);
        job.out_res_V_pos = engines[layer_idx]->get_host_res_V_pos(slot_id);
        job.out_res_K_val = engines[layer_idx]->get_host_res_K_val(slot_id);
        job.out_res_V_val = engines[layer_idx]->get_host_res_V_val(slot_id);
    }
    
    // Outputs in block pool host mirrors (CUDA compatible)
    job.out_scale = engines[layer_idx]->get_host_scales() + slot_id;
    job.out_anchor_k = engines[layer_idx]->get_host_anchors_K(slot_id);
    job.out_anchor_v = engines[layer_idx]->get_host_anchors_V(slot_id);
    job.out_seq_len = engines[layer_idx]->get_host_seq_lens() + slot_id;
    job.out_anchor_position = engines[layer_idx]->get_host_anchor_positions() + slot_id;
    job.out_token_positions = engines[layer_idx]->get_host_token_positions(slot_id);
    
    job.max_residual = NativeBlockPool::MAX_RESIDUAL;
    job.W_proj = W_proj_;
    job.desc_dim = desc_dim_;
    job.out_desc = engines[layer_idx]->get_host_desc_matrix(slot_id);
    job.state_table = &engines[layer_idx]->get_state_table();
    
    bool async_svd = true;
    if (const char* env_async = std::getenv("DIFFKV_ASYNC_SVD")) {
        std::string s(env_async);
        if (s == "0" || s == "false" || s == "off") {
            async_svd = false;
        }
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
        engines[layer_idx]->upload_slot(slot_id);
        block->device_synced = true;
        stats_.total_compressed++;
    }
}

} // namespace diffkv
