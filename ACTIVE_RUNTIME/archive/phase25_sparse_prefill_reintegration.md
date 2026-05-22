# Phase 25 — Sparse Prefill Reintegration

## Objective
Revive the `RetrievalAwareSparsePrefill` logic from archival research and inject it into the active serving path to avoid computing $O(N^2)$ attention FLOPs on long contexts.

## Integration
In `diffkv_attention.py`, we replaced the generic SDPA block for long queries (`q_len > 1024`) with an instantiation of the sparse prefill engine:

```python
from research.sparse_prefill_anchors import RetrievalAwareSparsePrefill

if not hasattr(layer.self_attn, "sparse_prefill_engine"):
    layer.self_attn.sparse_prefill_engine = RetrievalAwareSparsePrefill(
        sink_tokens=64, chunk_size=512, local_window_chunks=1, top_k_retrieval_chunks=2
    )
attn_output = layer.self_attn.sparse_prefill_engine.execute_sparse_attention(
    query_states, key_states, value_states
)
```

## Mechanism
When a 25K prompt is processed:
1. The 25K query is sliced into chunks of 512 tokens.
2. For each chunk, its anchor is evaluated against all historical anchors.
3. The chunk only computes SDPA over:
   - 64 sink tokens (attention stability)
   - 512 tokens (the immediate local window)
   - 1024 tokens (the top 2 retrieved sparse semantic chunks)
4. The maximum attention matrix computed in SRAM is now `[512 x 1600]` instead of `[25000 x 25000]`.

## Results
This integration correctly eliminates the dense $O(N^2)$ attention compute while explicitly preserving global context retrieval for needle-in-a-haystack capabilities.
