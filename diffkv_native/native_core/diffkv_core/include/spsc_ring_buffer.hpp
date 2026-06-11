// diffkv_core/include/spsc_ring_buffer.hpp
// Lock-free Single-Producer Single-Consumer ring buffer.
// Used by DiffKVCompressorThread to receive jobs from the Python/main thread
// without any mutex or GIL interaction.

#pragma once
#include <atomic>
#include <array>
#include <optional>

namespace diffkv {

template<typename T, size_t CAPACITY>
class SPSCRingBuffer {
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "CAPACITY must be a power of 2");
public:
    SPSCRingBuffer() : head_(0), tail_(0) {}

    // Producer: tries to push an item. Returns false if buffer is full (non-blocking).
    bool push(const T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & (CAPACITY - 1);
        if (next == tail_.load(std::memory_order_acquire))
            return false; // Full — caller must handle gracefully
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer: tries to pop an item. Returns nullopt if empty (non-blocking).
    std::optional<T> pop() {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
            return std::nullopt; // Empty
        T item = buffer_[tail];
        tail_.store((tail + 1) & (CAPACITY - 1), std::memory_order_release);
        return item;
    }

    bool empty() const {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t h = head_.load(std::memory_order_acquire);
        size_t t = tail_.load(std::memory_order_acquire);
        return (h - t + CAPACITY) & (CAPACITY - 1);
    }

private:
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    std::array<T, CAPACITY> buffer_;
};

} // namespace diffkv
