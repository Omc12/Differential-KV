// dkv_native/include/static_decode_graph.hpp
// Translation of static_decode_graph.py → C++17
//
// Metal Graph capture equivalent of CUDA Graph for decode hot-path on Apple Silicon.
// On macOS (Metal/MPS), we skip actual graph capture but provide the same API.
// On CUDA builds, this would capture a CUDA Graph; on Metal this is a no-op wrapper.
//
// The class stores the captured callable and replays it directly, giving callers
// a uniform interface regardless of backend.

#pragma once

#include <functional>

namespace dkv {

class StaticDecodeGraphRunner {
public:
    // Constructor
    StaticDecodeGraphRunner();

    // Check if graph has been captured
    bool is_captured() const;

    // Warmup + capture
    void capture(std::function<void()> fn, int num_warmup = 3);

    // Replay captured graph
    void replay(std::function<void()> fn = nullptr);

    // Invalidate captured graph (forces recapture on next step)
    void invalidate();

private:
    bool                  captured_;
    std::function<void()> captured_fn_;
};

} // namespace dkv
