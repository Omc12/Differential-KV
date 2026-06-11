// diffkv_core/include/block_state.hpp
// Authoritative atomic block state machine for Differential KV.
// All state transitions are enforced at the C++ level — Python never owns synchronization.

#pragma once
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>

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

// Returns true if (from -> to) is a legal transition.
inline bool is_legal_transition(BlockState from, BlockState to) {
    switch (from) {
        case BlockState::DenseResident:
            return to == BlockState::Compressing;
        case BlockState::Compressing:
            return to == BlockState::CompressedResident || to == BlockState::Invalid;
        case BlockState::CompressedResident:
            return to == BlockState::PagingOut || to == BlockState::Freed;
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

// A table of atomic block states.
// Max 65536 blocks. Designed for lock-free reads and CAS-based transitions.
class DiffKVBlockStateTable {
public:
    static constexpr size_t MAX_BLOCKS = 65536;

    DiffKVBlockStateTable() {
        for (size_t i = 0; i < MAX_BLOCKS; ++i)
            states_[i].store(BlockState::Freed, std::memory_order_relaxed);
    }

    void clear() {
        for (size_t i = 0; i < MAX_BLOCKS; ++i)
            states_[i].store(BlockState::Freed, std::memory_order_relaxed);
    }

    // Try to transition block_id from expected to desired.
    // Throws on illegal transition. Returns false on CAS failure (retry required).
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

    // Forceful invalidation — for session disconnect. Always legal from any non-Freed state.
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

    // Graph replay safety: returns true if ALL blocks in the list are safe for decode.
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

} // namespace diffkv
