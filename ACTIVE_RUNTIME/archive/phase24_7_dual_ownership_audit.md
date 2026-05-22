# Phase 24.7 — Dual Ownership Audit

## Objective
Determine whether the runtime stores canonical dense KV alongside Differential KV's compressed sparse KV.

## Audit Findings

We traced the execution of a 25,000 token prompt to measure KV overlap.

1. **Bytes owned by legacy dense KV cache:** 0 bytes.
   - The Hugging Face `past_key_values` is explicitly bypassed in `ContinuousBatchEngine._step()`.
   - The `diffkv_attention.py` monkey-patch explicitly intercepts KV storage and routes it to `kv_manager.ingest_streaming()`, returning `None` to the HF native cache mechanisms.

2. **Bytes owned by DiffKV slabs:** ~340 MB (for 25K tokens, Qwen-0.5B).
   - `25,000 tokens / 16 (micro_block_size) ≈ 1562 blocks`
   - `1562 blocks * 24 layers * (rank 8 U+V + 1 anchor) ≈ 340 MB`

3. **Overlap Percentage:** 0%
4. **Duplicate Residency:** None.

## The "Hidden" 16 GB Allocation
If there is no dual ownership, why does a 25K prompt try to allocate 16.3 GB of VRAM?
Because standard PyTorch eagerly evaluates the attention matrix `[batch_size, num_heads, seq_len, seq_len]`. 
For a 25K sequence with 14 heads in fp16, this single transient activation tensor requires **exactly 16.3 GiB**. 

## Verdict
There is **NO dual ownership**. The KV is truly sparse. The VRAM explosion on large contexts is caused by O(N^2) eager attention activations, completely unrelated to KV cache storage.
