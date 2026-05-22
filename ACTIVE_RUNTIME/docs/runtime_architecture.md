# Runtime Architecture

## Execution Flow

### Prefill (q_len > 1)
```
input_ids -> HF model forward
  -> diffkv_attention forward (per layer)
    -> StreamingSparseIngestManager.ingest_chunk()
      -> micro-block accumulation (16 tokens)
      -> AsyncCompressor.submit() when full
    -> SDPA / FlashAttention (F.scaled_dot_product_attention)
  -> last_token_lm_head_forward (vocab projection on last token only)
```

### Decode (q_len == 1)
```
input_ids -> HF model forward
  -> diffkv_attention forward (per layer)
    -> StreamingSparseIngestManager.append_decode_token()
    -> native_triton_sparse_attn_decode()
      -> NativeBlockPool block_indices gather
      -> _fused_sparse_decode_kernel (Triton, SRAM-resident)
      -> dense active window accumulation
  -> last_token_lm_head_forward
```

## Memory Layout

- **NativeBlockPool**: pre-allocated contiguous GPU tensors (U, V_K, V_V, anchors_K/V, scales, seq_lens)
- **StreamingKVBlock**: anchor (1 token dense) + optional active_k/v window + compressed U/V
- **Dense residency**: bounded to 1 micro-block (16 tokens) per session per layer

## Key Files

| File | Role |
|---|---|
| `native_core/kv_runtime_manager.py` | Session management, compression routing |
| `native_core/streaming_sparse_ingest.py` | Sparse-first prefill ingest |
| `native_core/compression/async_compressor.py` | Background SVD thread pool |
| `native_core/compression/lowrank.py` | SVD low-rank compression |
| `native_core/sparse_decode/triton_sparse_attn.py` | Triton fused decode kernel |
| `native_core/sparse_decode/triton_diffkv.py` | Triton reconstruction kernel |
| `runtime/native_block_pool.py` | Contiguous GPU pool |
| `runtime/diffkv_attention.py` | HF model attention patch |
| `serving/hf_diffkv_wrapper.py` | Model wrapper + generate() |
| `serving/batch_engine.py` | Continuous batching engine |
| `serving/openai_compatible_api_gateway.py` | OpenAI-compatible API |
