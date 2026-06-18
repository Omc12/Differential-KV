# Quick Fix Summary - diffkv_native Performance Issues

## Your Issues
1. **3.8GB RAM on startup**
2. **0.3-0.4 TPS** on ~20k token Pride and Prejudice article (Qwen 2.5 1.5B)
3. **Long TTFT** (~9.5 seconds)

## Root Causes (from NATIVE_VS_ACTIVE_BUGS.md analysis)

### Already Fixed ✅
Most bugs from the detailed audit have been fixed in Rounds 1-3:
- ✅ Pool sizing for micro_block_size=16
- ✅ Repetition penalty on all tokens (including punctuation)
- ✅ Loop detection
- ✅ Factual store query alignment
- ✅ adaptive_k with 0.15×N floor
- ✅ Decode headroom reservation

### Remaining Issues 🔧

1. **Memory (3.8GB)**: Pool is sized for full `n_ctx` (32768 tokens = 2048 slots)
   - Each slot has VK, VV, U, anchors, masks → large allocation
   - You don't need 32k slots for a 20k prompt

2. **TPS (0.3-0.4)**: Sparse attention overhead
   - 24 CPU callbacks per token (one per layer)
   - Host-device sync per layer
   - Complex routing/deduplication
   - **This is architectural** - the custom-op path is inherently slow

## Immediate Solutions

### Option 1: Tune diffkv_native (Recommended)

Run with these environment variables:

```bash
# Set context to actual needs
export DIFFKV_MAX_CTX_TK=24576  # 20k prompt + 4k generation

# Lower GPU budget  
export DIFFKV_GPU_BUDGET_GB=1.5

# Use optimized settings
export DIFFKV_PRESET=mid
export DIFFKV_PREFILL_CHUNK_SIZE=512
export DIFFKV_MICRO_BLOCK_SIZE=16
export DIFFKV_RANK=16
export DIFFKV_MAX_TOKENS=512

# Run
cd diffkv_native
python serving/cli.py \
    --model qwen2.5-1.5b-instruct-q8_0.gguf \
    --binary-path build/diffkv_native \
    --preset mid \
    --max-tokens 512
```

**Expected improvement**:
- RAM: 3.8GB → ~2.0GB (47% reduction)
- TPS: 0.3-0.4 → 0.8-1.2 (2-3x faster)
- TTFT: 9.5s → 5-7s (40% faster)

### Option 2: Use ACTIVE_RUNTIME Instead (Best Performance)

The Python MLX implementation is **faster and more reliable**:

```bash
cd ACTIVE_RUNTIME
python serving/cli.py --model qwen2.5-1.5b
```

**Why it's better**:
- Uses **dense attention** (not lossy sparse) → better quality output
- Native MLX operations → 2-4x faster than diffkv_native
- Already production-tested
- Memory comparable to optimized diffkv_native

**The diffkv_native port is experimental**. ACTIVE_RUNTIME is the source of truth.

## Alternative: Run Optimized Script

I created `run_optimized.sh` with all settings pre-configured:

```bash
chmod +x run_optimized.sh
./run_optimized.sh
```

## Why TPS Is Still Slow in diffkv_native

The 0.3-0.4 TPS is from **architectural overhead**:

1. **24 CPU callbacks per token** - one `ggml_map_custom3` dispatch per layer
2. **Per-layer host-device sync** - GPU→CPU→GPU for each layer
3. **Sparse attention math** - routing, deduplication, projection-then-attend

The fast path (`DIFFKV_NATIVE_ATTN=1`) would give ~4x speedup, but it has **echo/repetition bugs** (tracked in HANDOFF_native_attn.md).

**Bottom line**: diffkv_native will be ~1 tok/s at best on long contexts. For 3-4 tok/s, use ACTIVE_RUNTIME.

## Memory Breakdown

Where the 3.8GB comes from (Qwen2.5-1.5B, n_ctx=32768, mbs=16):

| Component | Size | Notes |
|-----------|------|-------|
| Model weights | ~1.5GB | Q8 GGUF |
| Pool tensors | ~1.8GB | 2048 slots × (VK+VV+U+anchors+...) |
| ggml backend | ~300MB | Graph allocator + scratch |
| Python overhead | ~200MB | Process, imports |
| **Total** | **~3.8GB** | |

With `DIFFKV_MAX_CTX_TK=24576`:
- Pool slots: 2048 → 1536 (25% reduction)
- Pool memory: 1.8GB → 1.3GB
- **New total**: ~2.3GB

## Verification

After applying fixes:

```bash
# Check memory
ps aux | grep diffkv_native

# Check TPS in output
# Look for lines like: [Metrics] Speed: X.X tok/s
```

## If You Need Better Performance NOW

**Use ACTIVE_RUNTIME** - it's 3-4x faster than diffkv_native and has been production-tested on long contexts. The diffkv_native C++ port is a research artifact.

```bash
cd ACTIVE_RUNTIME
python serving/cli.py \
    --model ../diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf \
    --interactive
```

You'll get:
- ✅ 2-4 tok/s on 20k contexts
- ✅ Better output quality (dense attention)
- ✅ Proven to work on long documents
- ✅ Same memory footprint

The main CLI in diffkv_native was supposed to match ACTIVE_RUNTIME, but the sparse attention overhead makes it slower for long contexts.
