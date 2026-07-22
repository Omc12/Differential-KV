#pragma once

// ACTIVE_RUNTIME/native_core/sparse_decode/triton_fused_decode.py is a CUDA Triton kernel.
// On macOS M-series GPUs, this is replaced by the custom Metal shader:
// native_core/dkv_core/metal/dkv_decode.metal
// which is loaded and compiled dynamically in the Metal runtime.
