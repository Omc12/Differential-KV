// SKO Phase 41.3: Native Sparse CUDA Extension Layer

#pragma once
#include <vector>
#include <cstdint>

namespace diffkv {

class NativeSparseCudaExtension {
public:
    NativeSparseCudaExtension() = default;
    ~NativeSparseCudaExtension() = default;

    // CUDA sparse metadata helpers
    void pack_gpu_metadata(const std::vector<float>& confidence_scores, std::vector<uint8_t>& out_buffer);

    // Sparse attention traversal primitives
    int traverse_sparse_attention_blocks(int total_blocks, const std::vector<uint8_t>& block_mask);

    // Sparse block indexing primitives
    std::vector<int> index_sparse_blocks(const std::vector<float>& scores, float threshold);
};

} // namespace diffkv
