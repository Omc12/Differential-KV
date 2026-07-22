// dkv_native/include/diagnostics.hpp
// Translation of diagnostics.py → C++17
// Runtime diagnostics: VRAM/memory logging, TPS counter, pool stats.
// On macOS the "VRAM" concept maps to unified memory reported via sysctl/vm_stat.
// CUDA sync-debug helpers are no-ops on non-CUDA builds.

#pragma once

#include <string>
#include <cstdio>
#include <chrono>
#include <cstring>

#ifdef __APPLE__
#  include <sys/types.h>
#  include <sys/sysctl.h>
#  include <mach/mach.h>
#  include <mach/mach_host.h>
#  include <mach/vm_statistics.h>
#endif

namespace dkv {

// ---------------------------------------------------------------------------
// VramStats — returned by log_vram()
// ---------------------------------------------------------------------------
struct VramStats {
    float allocated_gb = 0.0f;  // bytes actively used
    float reserved_gb  = 0.0f;  // bytes held by allocator (may equal allocated on Metal)
};

// ---------------------------------------------------------------------------
// log_vram
// ---------------------------------------------------------------------------
// On macOS, queries Mach kernel for memory pressure stats and prints them.
// "allocated" = wired + active pages; "reserved" = total physical RAM used by process.
// On non-Apple platforms prints zeros (CUDA path not implemented here).
// ---------------------------------------------------------------------------
inline VramStats log_vram(const std::string& label = "") {
    VramStats stats;

#ifdef __APPLE__
    // ---- System-wide physical memory usage via host_statistics64 ----
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(mach_host_self(),
                                         HOST_VM_INFO64,
                                         reinterpret_cast<host_info64_t>(&vm_stat),
                                         &count);

    // Page size
    vm_size_t page_size = 0;
    host_page_size(mach_host_self(), &page_size);

    if (kr == KERN_SUCCESS && page_size > 0) {
        // "Allocated" ≈ wired + active (pages in real use)
        uint64_t active_bytes  = static_cast<uint64_t>(vm_stat.active_count)   * page_size;
        uint64_t wired_bytes   = static_cast<uint64_t>(vm_stat.wire_count)      * page_size;
        // "Reserved" ≈ active + inactive (held by OS / allocator)
        uint64_t inactive_bytes= static_cast<uint64_t>(vm_stat.inactive_count) * page_size;

        stats.allocated_gb = static_cast<float>(active_bytes + wired_bytes)
                             / (1024.0f * 1024.0f * 1024.0f);
        stats.reserved_gb  = static_cast<float>(active_bytes + wired_bytes + inactive_bytes)
                             / (1024.0f * 1024.0f * 1024.0f);
    }

    std::fprintf(stderr,
        "[DKV diagnostics] %s%smemory  allocated=%.3f GB  reserved=%.3f GB"
        "  (Apple Unified Memory via Mach)\n",
        label.empty() ? "" : label.c_str(),
        label.empty() ? "" : "  ",
        stats.allocated_gb, stats.reserved_gb);
#else
    std::fprintf(stderr,
        "[DKV diagnostics] %s%smemory  allocated=%.3f GB  reserved=%.3f GB"
        "  (non-macOS stub)\n",
        label.empty() ? "" : label.c_str(),
        label.empty() ? "" : "  ",
        stats.allocated_gb, stats.reserved_gb);
#endif

    return stats;
}

// ---------------------------------------------------------------------------
// log_block_states
// ---------------------------------------------------------------------------
// Prints a message about KV block states for a given session.
// Full introspection requires access to DKVBlockStateTable; the pointer
// is accepted as void* so this header remains dependency-free.
// Callers with a real manager can cast and inspect; here we provide the
// diagnostic message boundary.
// ---------------------------------------------------------------------------
inline void log_block_states(void* kv_manager_ptr, const std::string& session_id) {
    if (!kv_manager_ptr) {
        std::fprintf(stderr,
            "[DKV diagnostics] log_block_states: session='%s'"
            "  kv_manager_ptr=null — block state info unavailable without manager access.\n",
            session_id.c_str());
        return;
    }
    // When a concrete manager type is available the caller should cast and
    // query directly. This stub confirms the call was received.
    std::fprintf(stderr,
        "[DKV diagnostics] log_block_states: session='%s'"
        "  manager=%p — override this call with a typed accessor for full state info.\n",
        session_id.c_str(), kv_manager_ptr);
}

// ---------------------------------------------------------------------------
// TpsCounter — tokens-per-second / steps-per-second counter
// ---------------------------------------------------------------------------
class TpsCounter {
public:
    int steps = 0;

    TpsCounter() {
        reset();
    }

    /// Record one decode step.
    void tick() {
        ++steps;
    }

    /// Returns steps / elapsed_seconds.  Returns 0.0 if called immediately after reset.
    double rate() const {
        using namespace std::chrono;
        auto now = steady_clock::now();
        double elapsed = duration<double>(now - start_).count();
        if (elapsed < 1e-9) return 0.0;
        return static_cast<double>(steps) / elapsed;
    }

    /// Reset timer and step count.
    void reset() {
        steps  = 0;
        start_ = std::chrono::steady_clock::now();
    }

    /// Print current rate to stderr.
    void print(const std::string& label = "") const {
        std::fprintf(stderr,
            "[DKV diagnostics] TpsCounter %s%s steps=%d  rate=%.2f steps/s\n",
            label.empty() ? "" : label.c_str(),
            label.empty() ? "" : ":",
            steps, rate());
    }

private:
    std::chrono::steady_clock::time_point start_;
};

// ---------------------------------------------------------------------------
// log_pool_stats
// ---------------------------------------------------------------------------
// Placeholder — a real implementation would inspect an allocator pool.
// ---------------------------------------------------------------------------
inline void log_pool_stats(const std::string& label = "") {
    std::fprintf(stderr,
        "[DKV diagnostics] log_pool_stats%s%s — pool stats unavailable (stub).\n",
        label.empty() ? "" : "  label=",
        label.empty() ? "" : label.c_str());
}

// ---------------------------------------------------------------------------
// CUDA sync-debug helpers — no-ops on macOS / non-CUDA builds
// ---------------------------------------------------------------------------
inline void enable_cuda_sync_debug() {
#ifdef __APPLE__
    std::fprintf(stderr,
        "[DKV diagnostics] enable_cuda_sync_debug: no-op on macOS (no CUDA).\n");
#else
    // On CUDA builds, set CUDA_LAUNCH_BLOCKING=1 via environment or
    // cudaDeviceSetFlags(cudaDeviceScheduleBlockingSync) here.
    std::fprintf(stderr,
        "[DKV diagnostics] enable_cuda_sync_debug: stub — implement CUDA path.\n");
#endif
}

inline void disable_cuda_sync_debug() {
#ifdef __APPLE__
    std::fprintf(stderr,
        "[DKV diagnostics] disable_cuda_sync_debug: no-op on macOS (no CUDA).\n");
#else
    std::fprintf(stderr,
        "[DKV diagnostics] disable_cuda_sync_debug: stub — implement CUDA path.\n");
#endif
}

} // namespace dkv
