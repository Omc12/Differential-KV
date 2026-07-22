// SKO Phase 41.3: Native Sparse CUDA Extension Layer

#include "native_sparse_cuda_extension.hpp"

namespace dkv {

void NativeSparseCudaExtension::pack_gpu_metadata(const std::vector<float>& confidence_scores, std::vector<uint8_t>& out_buffer) {
    out_buffer.clear();
    out_buffer.reserve(confidence_scores.size());
    for (float score : confidence_scores) {
        out_buffer.push_back(score >= 0.5f ? 1 : 0);
    }
}

int NativeSparseCudaExtension::traverse_sparse_attention_blocks(int total_blocks, const std::vector<uint8_t>& block_mask) {
    int traversed = 0;
    for (int i = 0; i < total_blocks && i < block_mask.size(); ++i) {
        if (block_mask[i] == 1) {
            traversed++;
        }
    }
    return traversed;
}

std::vector<int> NativeSparseCudaExtension::index_sparse_blocks(const std::vector<float>& scores, float threshold) {
    std::vector<int> indices;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] >= threshold) {
            indices.push_back(i);
        }
    }
    return indices;
}

} // namespace dkv
