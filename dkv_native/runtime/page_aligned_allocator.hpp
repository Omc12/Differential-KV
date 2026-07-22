#pragma once

#include <cstdlib>
#include <new>
#include <stddef.h>

namespace dkv {

template <typename T>
struct PageAlignedAllocator {
    using value_type = T;
    PageAlignedAllocator() noexcept {}
    template <typename U> PageAlignedAllocator(const PageAlignedAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(n * sizeof(T), 4096);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, 4096, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) noexcept {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

template <typename T, typename U>
bool operator==(const PageAlignedAllocator<T>&, const PageAlignedAllocator<U>&) { return true; }

template <typename T, typename U>
bool operator!=(const PageAlignedAllocator<T>&, const PageAlignedAllocator<U>&) { return false; }

} // namespace dkv
