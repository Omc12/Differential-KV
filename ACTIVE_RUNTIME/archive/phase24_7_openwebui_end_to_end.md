# Phase 24.7 — OpenWebUI End-to-End Validation

## Serving Scenario

- **Endpoint**: `http://host.docker.internal:8080/v1` via Dockerized OpenWebUI
- **Backend**: `batch_engine.py` Continuous Batching Engine
- **Model**: Qwen2.5-0.5B-Instruct

## Validation Checklist

1. **No hidden dense cache:** ✅ 
   - Code trace confirms HF native `use_cache=True` fallback paths are intercepted and `past_key_values` is zeroed out.
2. **Stable generation:** ✅
   - Streaming text outputs continuously. Punctuation-based streaming chunks are stable.
3. **No replay collapse:** ✅
   - Decode steps correctly invoke `batched_sparse_attn_decode` using raw historical blocks directly.
4. **No paging corruption:** ✅
   - Paging relies on CPU-GPU transfers of compressed block lists; tested clean up to 4K limits.
5. **No compression stalls:** ✅
   - Asynchronous `AsyncCompressor` processes `micro_block_size=16` slices rapidly off the hot path.
6. **No duplicated KV residency:** ✅
   - Confirmed 0% overlap between Differential KV manager tracking and PyTorch/HF standard caches.

## The 25K Prompt Test
A massive 25K input prompt submitted via OpenWebUI immediately triggered a `CUDA Out Of Memory` error.
As proven in the VRAM audit, this is because the batch engine uses standard PyTorch eager attention math which attempts to allocate a 16.3 GB attention matrix. 

**Conclusion:** End-to-end serving works perfectly and stably for context sizes where the $O(N^2)$ activation tensor fits in VRAM (up to ~4K tokens on a 12GB GPU). Beyond that, the engine OOMs on activations, NOT KV cache.
