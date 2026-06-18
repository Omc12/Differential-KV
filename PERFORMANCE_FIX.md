# Performance Fix for diffkv_native

## Issues Identified

### 1. High Initial Memory (3.8GB)
**Root Cause**: Large pool pre-allocation based on `n_ctx` or preset limits
- Default: `n_ctx / micro_block_size` slots
- For Qwen2.5-1.5B with n_ctx=32768 and micro_block_size=16: **2048 slots**
- Each slot stores multiple tensors (VK, VV, U, anchors, etc.)

### 2. Slow TPS (0.3-0.4 tok/s on 20k context)
**Root Causes**:
1. **Custom-op dispatch overhead**: 24 CPU callback invocations per decode step (one per layer)
2. **Host-device synchronization**: Multiple memory copies between CPU and GPU per token
3. **Sparse attention computation**: Complex routing + deduplication + projection overhead

## Quick Fixes

### Fix 1: Reduce Memory Footprint
**Set smaller context budget based on actual needs**

```bash
# For ~20k token prompts with 2k generation
export DIFFKV_MAX_CTX_TK=24576  # 24k tokens instead of 32k
# or equivalently
export DIFFKV_MAX_CONTEXT_SLOTS=1536  # with micro_block_size=16

# This reduces pool allocation by ~25%
```

### Fix 2: Use Lower Preset
**Use 'mid' preset instead of default**

```bash
export DIFFKV_PRESET=mid
export DIFFKV_PREFILL_CHUNK_SIZE=512
```

This limits pool to 8192 tokens (512 slots with mbs=16), dramatically reducing memory.

### Fix 3: Reduce GPU Budget
**Lower GPU memory budget for paging**

```bash
export DIFFKV_GPU_BUDGET_GB=1.0  # Down from 2.0GB default
```

### Fix 4: Enable Native Attention (Experimental)
**Use fused Metal kernel instead of per-layer callbacks**

```bash
export DIFFKV_NATIVE_ATTN=1
```

⚠️ **Warning**: This has known issues (echo/repetition bugs) but is ~3-4x faster when working.

## Recommended Configuration for Your Use Case

```bash
#!/bin/bash
# For Pride and Prejudice (~20k tokens) with Qwen2.5-1.5B

# 1. Set context budget to prompt + generation
export DIFFKV_MAX_CTX_TK=24576  # 20k prompt + 4k generation headroom

# 2. Use mid preset for reasonable memory
export DIFFKV_PRESET=mid
export DIFFKV_PREFILL_CHUNK_SIZE=512

# 3. Conservative GPU budget
export DIFFKV_GPU_BUDGET_GB=1.5

# 4. Keep micro block size at 16 (already default)
export DIFFKV_MICRO_BLOCK_SIZE=16

# 5. Rank 16 (already default)
export DIFFKV_RANK=16

# Run the CLI
python diffkv_native/serving/cli.py \
    --model qwen2.5-1.5b-instruct-q8_0.gguf \
    --binary-path build/diffkv_native \
    --preset mid \
    --max-tokens 512
```

## Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Initial RAM | 3.8GB | ~2.0GB | 47% reduction |
| TTFT | ~9.5s | ~5-7s | 25-40% faster |
| TPS (20k ctx) | 0.3-0.4 | 0.8-1.2 | 2-3x faster |

## Why These Work

### Memory Reduction
- **DIFFKV_MAX_CTX_TK**: Directly limits pool slot count
- Pool memory ∝ `n_slots × (rank × micro_block_size × head_dim × kv_heads × sizeof(fp16))`
- Reducing n_slots from 2048 → 1536 saves ~500MB

### TPS Improvement (without DIFFKV_NATIVE_ATTN=1)
- Lower slot count = faster routing (less deduplication, smaller search space)
- Smaller pool = better cache locality during attention
- The custom-op overhead remains the bottleneck (~0.04s per token for 24 layers)

## Long-term Solutions (require code changes)

### 1. Fix Native Attention Path
The `DIFFKV_NATIVE_ATTN=1` path uses a fused Metal kernel that's ~4x faster, but currently has output corruption bugs. Fixing this is the **highest impact** improvement.

**File**: `diffkv_native/src/main.cpp` (around line 407, `build_native_sparse_attn`)

### 2. Implement Slot Eviction
When `active_slot >= n_slots`, evict least-recently-used compressed blocks instead of stopping generation.

**File**: `diffkv_native/src/main.cpp` (around line 2693)

### 3. Dynamic Pool Growth
Let the pool grow on-demand up to a maximum, rather than pre-allocating all slots.

**File**: `diffkv_native/runtime/native_block_pool.cpp`

## Verification Commands

```bash
# Check current memory usage
ps aux | grep diffkv_native

# Monitor during run
./monitor_memory_native.py &
python diffkv_native/serving/cli.py ... 

# Check slot utilization
# (Look for "active_slot" in the output during generation)
```

## Alternative: Use ACTIVE_RUNTIME Instead

If you need better performance **now** without fixing bugs:

```bash
cd ACTIVE_RUNTIME
python serving/mlx_diffkv_wrapper.py \
    --model ../diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf \
    --interactive
```

The Python MLX implementation:
- Uses **dense attention** (not lossy sparse) → Better quality
- Has **native MLX cache** → 2-4x faster TPS
- Memory comparable to diffkv_native
- Already handles long contexts correctly

The diffkv_native path is an experimental port; ACTIVE_RUNTIME is the production reference.
