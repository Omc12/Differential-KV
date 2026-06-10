#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace diffkv {

enum class BlockState : uint8_t {
    DenseResident      = 0,
    Compressing        = 1,
    CompressedResident = 2,
    PagingOut          = 3,
    CPUResident        = 4,
    Reloading          = 5,
    Invalid            = 6,
    Freed              = 7,
};

inline const char* state_name(BlockState s) {
    switch (s) {
        case BlockState::DenseResident:      return "DenseResident";
        case BlockState::Compressing:        return "Compressing";
        case BlockState::CompressedResident: return "CompressedResident";
        case BlockState::PagingOut:          return "PagingOut";
        case BlockState::CPUResident:        return "CPUResident";
        case BlockState::Reloading:          return "Reloading";
        case BlockState::Invalid:            return "Invalid";
        case BlockState::Freed:              return "Freed";
        default:                             return "Unknown";
    }
}

inline bool is_legal_transition(BlockState from, BlockState to) {
    switch (from) {
        case BlockState::DenseResident:
            return to == BlockState::Compressing;
        case BlockState::Compressing:
            return to == BlockState::CompressedResident || to == BlockState::Invalid;
        case BlockState::CompressedResident:
            return to == BlockState::PagingOut || to == BlockState::Freed || to == BlockState::DenseResident;
        case BlockState::PagingOut:
            return to == BlockState::CPUResident || to == BlockState::Invalid;
        case BlockState::CPUResident:
            return to == BlockState::Reloading;
        case BlockState::Reloading:
            return to == BlockState::CompressedResident || to == BlockState::Invalid;
        case BlockState::Invalid:
            return to == BlockState::Freed;
        case BlockState::Freed:
            return to == BlockState::DenseResident;
        default:
            return false;
    }
}

class DiffKVBlockStateTable {
public:
    static constexpr size_t MAX_BLOCKS = 65536;

    DiffKVBlockStateTable() {
        for (size_t i = 0; i < MAX_BLOCKS; ++i)
            states_[i].store(BlockState::Freed, std::memory_order_relaxed);
    }

    bool transition(uint32_t block_id, BlockState expected, BlockState desired) {
        if (block_id >= MAX_BLOCKS)
            throw std::out_of_range("block_id out of range");

        if (!is_legal_transition(expected, desired)) {
            throw std::logic_error(
                std::string("Illegal block transition: ") +
                state_name(expected) + " -> " + state_name(desired)
            );
        }
        return states_[block_id].compare_exchange_strong(
            expected, desired,
            std::memory_order_acq_rel,
            std::memory_order_acquire
        );
    }

    void force_invalidate(uint32_t block_id) {
        if (block_id >= MAX_BLOCKS) return;
        BlockState current = states_[block_id].load(std::memory_order_acquire);
        if (current != BlockState::Freed)
            states_[block_id].store(BlockState::Invalid, std::memory_order_release);
    }

    BlockState get(uint32_t block_id) const {
        if (block_id >= MAX_BLOCKS)
            throw std::out_of_range("block_id out of range");
        return states_[block_id].load(std::memory_order_acquire);
    }

    bool are_replay_safe(const uint32_t* block_ids, size_t count) const {
        for (size_t i = 0; i < count; ++i) {
            auto s = states_[block_ids[i]].load(std::memory_order_acquire);
            if (s != BlockState::CompressedResident && s != BlockState::DenseResident)
                return false;
        }
        return true;
    }

private:
    std::atomic<BlockState> states_[MAX_BLOCKS];
};

class DiffKVKVEngine {
public:
    DiffKVKVEngine();
    ~DiffKVKVEngine();

    bool initialize(int n_slots, int rank, int head_dim, int kv_heads, int desc_dim, ggml_backend_buffer_type_t buft);

    // Getters for GGML tensors
    struct ggml_tensor * get_U() { return U_; }
    struct ggml_tensor * get_U_scale() { return U_scale_; }
    struct ggml_tensor * get_VK() { return VK_; }
    struct ggml_tensor * get_VV() { return VV_; }
    struct ggml_tensor * get_anchors_K() { return anchors_K_; }
    struct ggml_tensor * get_anchors_V() { return anchors_V_; }
    struct ggml_tensor * get_seq_lens() { return seq_lens_; }
    struct ggml_tensor * get_scales() { return scales_; }
    struct ggml_tensor * get_desc_matrix() { return desc_matrix_; }
    struct ggml_tensor * get_anchor_positions() { return anchor_positions_; }

    DiffKVBlockStateTable & get_state_table() { return state_table_; }

private:
    int n_slots_ = 0;
    int rank_ = 0;
    int head_dim_ = 0;
    int kv_heads_ = 0;
    int desc_dim_ = 0;

    struct ggml_context * pool_ctx_ = nullptr;
    struct ggml_backend_buffer * pool_buffer_ = nullptr;

    // Pool Tensors
    struct ggml_tensor * U_ = nullptr;
    struct ggml_tensor * U_scale_ = nullptr;
    struct ggml_tensor * VK_ = nullptr;
    struct ggml_tensor * VV_ = nullptr;
    struct ggml_tensor * anchors_K_ = nullptr;
    struct ggml_tensor * anchors_V_ = nullptr;
    struct ggml_tensor * seq_lens_ = nullptr;
    struct ggml_tensor * scales_ = nullptr;
    struct ggml_tensor * desc_matrix_ = nullptr;
    struct ggml_tensor * anchor_positions_ = nullptr;  // [n_slots] int32: actual sequence position of each block's anchor

    DiffKVBlockStateTable state_table_;
};

} // namespace diffkv
