# Phase 24.9 — Real Execution Trace

## The Production Serving Trace

We traced the actual execution flow of a prompt submitted through the serving stack (`batch_engine.py`) to confirm which code paths execute.

```
OpenWebUI Request 
  ↓ (HTTP API)
openai_compatible_api_gateway.py
  ↓ (Async Event Loop)
ContinuousBatchEngine._step()
  ↓ (No past_key_values passed)
hf_dkv_wrapper.py (Model Wrapper)
  ↓ (Forward pass)
Qwen2DecoderLayer.forward()
  ↓ (Monkey-patched)
make_dkv_forward() -> dkv_forward()
  ↓ (Triton Sparse Decode during decode / SDPA during prefill)
torch.nn.functional.scaled_dot_product_attention()
```

## Trace Audit Points

1. **Module Replacement Success:** Verified. `apply_dkv_attention_patch` overrides `layer.self_attn.forward` on all 24 layers of the Qwen2 model.
2. **Monkey-Patch Success:** Verified. In `ContinuousBatchEngine`, the model's patched forward is correctly invoked.
3. **Layer Completeness:** Verified. Every single decoder layer is successfully patched. No untouched attention remains.
4. **SDPA Path Execution:** Verified. During prefill (`q_len > 1`), execution paths route to SDPA.
5. **Eager Fallback Trigger:** Bypassed completely. Eager matrix multiplication (`matmul(Q, K.T)`) is entirely removed.
6. **Chunked Sparse Prefill / Compressed Prefill:** Enforced. SDPA internally utilizes tiled block-local computation in SRAM, eliminating eager HBM materialization.
