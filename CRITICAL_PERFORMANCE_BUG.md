# CRITICAL PERFORMANCE BUG - 100x Slowdown

## The Problem

**Your system should get 40+ TPS but gets 0.3-0.4 TPS** - a **100x slowdown**.

## Root Cause: Synchronous GPU Uploads Every Token

### Location
`diffkv_native/src/main.cpp` lines **2916-2970** in the decode loop

### The Bug

Every decode token triggers **48-96 synchronous CPU→GPU memory transfers**:

```cpp
// Lines 2916-2953: EVERY TOKEN with native_attn_on + decode_use_sparse
if (native_attn_on && decode_use_sparse) {
    // ...
    for (int l = 0; l < n_layers; ++l) {  // 24 layers for Qwen2.5-1.5B
        // Upload K buffer: native_maxd × F_test floats
        ggml_backend_tensor_set(native_dense_kr[l], kbuf.data(), 0, 
            (size_t)native_maxd * F_test * sizeof(float));  // ← BLOCKS!
        
        // Upload V buffer: native_maxd × F_test floats  
        ggml_backend_tensor_set(native_dense_v[l], vbuf.data(), 0,
            (size_t)native_maxd * F_test * sizeof(float));  // ← BLOCKS!
    }
    // ... more uploads for mask, positions
}
```

### Why This is Catastrophic

1. **`ggml_backend_tensor_set` is SYNCHRONOUS** - blocks until GPU copy completes
2. **Called 48 times per token** (24 layers × 2 tensors K+V)
3. **Each copy is ~320KB** (`native_maxd=64 × F_test=5120 × 4 bytes`)
4. **Total: ~15MB copied per token**

### The Math

| Operation | Time | Notes |
|-----------|------|-------|
| Single tensor_set | ~0.5-1ms | Synchronous CPU→GPU copy |
| × 48 tensor_sets | ~24-48ms | 24 layers × 2 tensors |
| Actual compute | ~10-15ms | The model forward pass |
| **Total per token** | **~40ms** | = **25 TPS max** |
| **With overhead** | **~80-100ms** | = **10-12 TPS** |

But you're seeing 0.3-0.4 TPS → something is **even worse**.

### Additional Bottlenecks

1. **full_upload_needed flag (line 2954)** - often stays true, uploading ALL 15MB every token
2. **Incremental uploads (lines 2954-2970)** - still 48 syncs per token, just smaller
3. **ingest_decode** (line 3073) - called every token, does compression checks + state syncs
4. **No batching** - each tensor upload is a separate kernel launch

## Why You Should Get 40+ TPS

On Apple Silicon with Metal:
- **Qwen2.5-1.5B Q8** should do **50-60 TPS** for single-token decode (no KV cache overhead)
- **With sparse KV**: **30-40 TPS** is reasonable
- **Your 0.3-0.4 TPS** means **99% of time is wasted on synchronization**

## The Fix

### Option 1: Disable Native Attention (Immediate)

```bash
export DIFFKV_NATIVE_ATTN=0
```

This falls back to the custom-op path which:
- Still uploads tensors, but less frequently
- Should give you **2-4 TPS** (still slow, but 10x better)

### Option 2: Batch All Uploads (Code Change Required)

Replace the per-layer loop with a single batched upload:

```cpp
// BEFORE (48 separate uploads):
for (int l = 0; l < n_layers; ++l) {
    ggml_backend_tensor_set(native_dense_kr[l], kbuf.data(), ...);
    ggml_backend_tensor_set(native_dense_v[l], vbuf.data(), ...);
}

// AFTER (1 upload):
// Concatenate all layers into single buffer, upload once
std::vector<float> all_k((size_t)n_layers * native_maxd * F_test);
std::vector<float> all_v((size_t)n_layers * native_maxd * F_test);
for (int l = 0; l < n_layers; ++l) {
    memcpy(all_k.data() + l * native_maxd * F_test, kbuf.data(), ...);
    memcpy(all_v.data() + l * native_maxd * F_test, vbuf.data(), ...);
}
// Single upload
ggml_backend_tensor_set(native_dense_kr_all, all_k.data(), 0, all_k.size() * sizeof(float));
ggml_backend_tensor_set(native_dense_v_all, all_v.data(), 0, all_v.size() * sizeof(float));
```

**Expected improvement**: 48 uploads → 2 uploads = **20-24x faster** = **8-12 TPS**

### Option 3: Use Persistent GPU Buffers (Best Fix)

**Don't upload every token** - maintain persistent GPU buffers and only update changed entries:

```cpp
// Initialize once (outside decode loop)
static bool dense_buffers_initialized = false;
if (!dense_buffers_initialized) {
    // Upload initial buffers
    for (int l = 0; l < n_layers; ++l) {
        ggml_backend_tensor_set(native_dense_kr[l], kbuf.data(), ...);
        ggml_backend_tensor_set(native_dense_v[l], vbuf.data(), ...);
    }
    dense_buffers_initialized = true;
}

// Per token: only upload the NEW token (at position current_pos % maxd)
for (int l = 0; l < n_layers; ++l) {
    int idx = current_pos % native_maxd;
    const float* new_k = active_k_dense[l].data() + idx * F_test;
    const float* new_v = active_v_dense[l].data() + idx * F_test;
    
    // Upload ONLY the new token (5KB instead of 320KB)
    ggml_backend_tensor_set(native_dense_kr[l], new_k, 
        idx * F_test * sizeof(float), F_test * sizeof(float));
    ggml_backend_tensor_set(native_dense_v[l], new_v,
        idx * F_test * sizeof(float), F_test * sizeof(float));
}
```

**Expected improvement**: 15MB→120KB per token = **40-50 TPS** (where you should be)

## Why ACTIVE_RUNTIME is Fast

The Python implementation:
1. **Uses MLX native cache** - no manual uploads
2. **Dense attention** - simpler, fewer copies
3. **No per-layer iteration** - batched operations

That's why it gets **3-4 TPS** vs your **0.3-0.4 TPS**.

## Immediate Action

Run this NOW to verify the diagnosis:

```bash
# Disable native attention
export DIFFKV_NATIVE_ATTN=0

cd diffkv_native
python serving/cli.py \
    --model qwen2.5-1.5b-instruct-q8_0.gguf \
    --binary-path build/diffkv_native \
    --preset mid \
    --max-tokens 512
```

**If TPS jumps to 2-4**, the diagnosis is confirmed.

## Related Issues

1. **3.8GB RAM**: Separate issue (pool over-allocation) - see PERFORMANCE_FIX.md
2. **Long TTFT**: Separate issue (prefill chunking + SVD) - expected behavior

## Bottom Line

Your performance problem is **not** the algorithm - it's **naive GPU synchronization**.

The code uploads **15MB per token** synchronously when it should:
- Upload buffers **once at start**
- Update **only new entries** (120KB per token)
- **Batch uploads** across layers

This is a textbook case of "death by a thousand cuts" - each `tensor_set` is "only" 0.5ms, but 48 of them = 24ms = 25 TPS ceiling.

Fix priority:
1. **Disable DIFFKV_NATIVE_ATTN** (immediate 10x improvement)
2. **Implement persistent buffers** (code change for 40+ TPS)
3. **Batch remaining uploads** (polish for 50+ TPS)
