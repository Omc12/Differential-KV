# Phase 24.8 — Full Prefill Execution Trace

## Current Execution Path

When a user submits a 25,000 token prompt, it enters the `ContinuousBatchEngine`. The prefill step invokes `model(input_ids, use_cache=True)`.

Inside `diffkv_attention.py` (Phase 24.5 architecture), the execution proceeds as follows:

### 1. KV State Ingestion
```python
kv_manager.ingest_streaming(sid, captured_layer_idx, curr_k, curr_v)
```
- The entire 25K `curr_k` and `curr_v` sequence is handed to `StreamingSparseIngestManager`.
- It breaks the 25K sequence into micro-blocks of 16 tokens.
- These micro-blocks are queued to the `AsyncCompressor`.
- **Memory impact:** This is efficient. The compressor processes in the background. No giant tensors are materialized here beyond the original K/V projections.

### 2. Historical KV Reconstruction
```python
past_k, past_v = kv_manager.get_kv(sid, captured_layer_idx)
```
- Because this is the first turn, `get_kv()` returns `None`.
- For multi-turn, this rebuilds the compressed slabs back into dense representations using `TritonDiffKV.reconstruct_lowrank`.

### 3. Eager Attention Materialization (The Bottleneck)
```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
```
- `query_states` and `key_states` are `[1, 14, 25000, 64]`.
- The `.transpose(2, 3)` yields `[1, 14, 64, 25000]`.
- The `matmul` executes, resulting in `[1, 14, 25000, 25000]`.
- **Memory impact:** This is where the 16.3 GB allocation occurs. This tensor is fully materialized in High Bandwidth Memory (HBM).

### 4. Attention Masking and Softmax
```python
causal_mask = torch.triu(...)
attn_weights = attn_weights + causal_mask
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
```
- Masking requires creating a `[25000, 25000]` boolean/float mask.
- Softmax requires traversing the entire 16.3 GB tensor again, reading it into SRAM, exponentiating, and writing it back to HBM.

### 5. Output Computation
```python
attn_output = torch.matmul(attn_weights, value_states)
```
- The 16.3 GB tensor is multiplied by `value_states` `[1, 14, 25000, 64]`.
- Result is `[1, 14, 25000, 64]` (the normal output shape).

## Diagnosis of the Bottleneck

- **Activation Ownership:** Standard PyTorch eagerly owns intermediate activation tensors (the scores before softmax, the mask, the scores after softmax).
- **Peak SRAM vs VRAM:** The eager execution moves the giant tensor back and forth between SRAM (where ALUs compute) and VRAM (HBM). FlashAttention/SDPA fuses the matmul + mask + softmax + matmul into a single SRAM-resident pass, preventing HBM materialization.
- **Logits Materialization:** After all 24 layers compute attention, the final hidden state is projected to vocab size `[1, 25000, 151936]`, generating another massive tensor.

## Action Plan
We must replace Step 3, 4, and 5 with an SRAM-resident chunked attention execution (SDPA/FlashAttention) augmented with Sparse Anchor Routing to prevent the `[seq x seq]` allocation while preserving sparse retrieval.
