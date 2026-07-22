// native_core/graph_runtime/static_decode_graph.cpp
// Implementation of StaticDecodeGraphRunner

#include "native_core/graph_runtime/static_decode_graph.hpp"
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace dkv {

StaticDecodeGraphRunner::StaticDecodeGraphRunner()
    : captured_(false)
    , captured_fn_(nullptr)
{}

bool StaticDecodeGraphRunner::is_captured() const {
    return captured_;
}

void StaticDecodeGraphRunner::capture(std::function<void()> fn, int num_warmup) {
    if (!fn) {
        throw std::invalid_argument(
            "StaticDecodeGraphRunner::capture: fn must not be null");
    }

#ifdef __APPLE__
    // Metal / MPS path — warmup runs to trigger shader compilation
    std::fprintf(stderr,
        "[StaticDecodeGraphRunner] macOS: running %d warmup iteration(s)...\n",
        num_warmup);

    for (int i = 0; i < num_warmup; ++i) {
        fn();
    }

    std::fprintf(stderr,
        "[StaticDecodeGraphRunner] macOS: warmup complete, graph marked as captured"
        " (Metal does not support explicit graph capture; fn stored for direct replay).\n");
#else
    // CUDA path placeholder
    std::fprintf(stderr,
        "[StaticDecodeGraphRunner] non-macOS: CUDA graph capture stub — "
        "running %d warmup iteration(s), then storing fn for direct replay.\n",
        num_warmup);

    for (int i = 0; i < num_warmup; ++i) {
        fn();
    }
#endif

    captured_fn_ = std::move(fn);
    captured_    = true;
}

void StaticDecodeGraphRunner::replay(std::function<void()> fn) {
    if (!captured_) {
        std::fprintf(stderr,
            "[StaticDecodeGraphRunner] replay called before capture — "
            "falling back to direct fn() call.\n");
        if (fn) fn();
        return;
    }

#ifdef __APPLE__
    // Direct replay using stored fn
    if (captured_fn_) {
        captured_fn_();
    } else if (fn) {
        fn(); // fallback if stored fn was somehow lost
    }
#else
    // CUDA path placeholder: would invoke cudaGraphLaunch
    if (captured_fn_) {
        captured_fn_();
    } else if (fn) {
        fn();
    }
#endif
}

void StaticDecodeGraphRunner::invalidate() {
    if (captured_) {
        std::fprintf(stderr,
            "[StaticDecodeGraphRunner] graph invalidated (prefill / shape change).\n");
    }
    captured_    = false;
    captured_fn_ = nullptr;
}

} // namespace dkv
