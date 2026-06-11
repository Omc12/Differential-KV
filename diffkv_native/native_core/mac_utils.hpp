// mac_utils.hpp
// Apple Silicon (Metal / CPU) compatibility layer for DiffKV native.
// C++ translation of ACTIVE_RUNTIME/native_core/mac_utils.py
//
// Provides unified device detection and helpers so every other module can
// call get_best_device() / is_apple_silicon() without platform guards.
//
// Priority order (same as Python version):
//   1. Metal/MPS  — Apple Silicon M-series (always true in diffkv_native on Mac)
//   2. CPU        — universal fallback

#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <Accelerate/Accelerate.h>
#endif

namespace diffkv {

// Forward-declared, implemented in metal_runtime.mm
bool has_metal();

inline bool is_apple_silicon() {
#ifdef __APPLE__
    return true;
#else
    return false;
#endif
}

// In diffkv_native (llama.cpp-based), we are always Metal on Mac.
// Returns "metal" on Apple Silicon, "cpu" otherwise.
inline std::string get_best_device() {
    if (has_metal()) return "metal";
    return "cpu";
}

// ── Memory helpers ────────────────────────────────────────────────────────────

struct MemoryInfo {
    double allocated_mb = 0.0;
    double reserved_mb  = 0.0;
    double rss_mb       = 0.0;
};

inline MemoryInfo get_memory_info() {
    MemoryInfo info;
#ifdef __APPLE__
    // RSS via mach task_info
    task_vm_info_data_t vm;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm), &count) == KERN_SUCCESS) {
        info.rss_mb       = static_cast<double>(vm.phys_footprint) / 1e6;
        info.allocated_mb = static_cast<double>(vm.internal)       / 1e6;
    }
    // Metal heap stats via sysctl
    // (Metal does not expose a stable C API for this; use vm_info as proxy)
    info.reserved_mb = info.allocated_mb;
#endif
    return info;
}

// ── Synchronization ───────────────────────────────────────────────────────────

// Block until all pending Metal/GPU ops complete.
// In llama.cpp, ops are submitted synchronously by default on Metal,
// so this is a lightweight fence.
inline void synchronize() {
#ifdef __APPLE__
    // ggml_metal_graph_compute already waits; this is a safety fence.
    // We use a sysctl memory barrier as a proxy.
    __sync_synchronize();
#endif
}

// Release unused memory (hint to OS).
inline void empty_cache() {
#ifdef __APPLE__
    // On unified memory, there is no separate GPU heap to release.
    // The OS will reclaim as needed; we only advise.
    // No equivalent of torch.mps.empty_cache() in raw Metal.
#endif
}

// ── dtype helpers ─────────────────────────────────────────────────────────────

// Returns preferred compute precision: "fp16" on Apple Silicon, "fp32" on CPU.
inline std::string get_default_dtype() {
    return is_apple_silicon() ? "fp16" : "fp32";
}



} // namespace diffkv
