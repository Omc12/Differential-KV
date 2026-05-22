# Phase 24.6 -- OpenWebUI Serving Path Verification

## Objective
Confirm that actual OpenWebUI requests route through the `StreamingSparseIngestManager` and the `batched_sparse_attn_decode` kernel, avoiding legacy dense fallbacks.

---

## The Execution Path

### 1. Request Ingress
- OpenWebUI sends an HTTP request to `http://host.docker.internal:8080/v1/chat/completions`.
- FastAPI routes it to `create_chat_completion` in `openai_compatible_api_gateway.py`.
- The request is enqueued into the `ContinuousBatchEngine`.

### 2. Prefill Phase
- `ContinuousBatchEngine._step()` pulls the request.
- It calls `self.wrapper.model(input_ids=..., position_ids=..., use_cache=True)` with **NO `past_key_values`**.
- This correctly routes through the `apply_diffkv_attention_patch` because Hugging Face native cache is not provided.
- Inside `diffkv_attention.py`, the code hits the Phase 24.5 prefill branch:
  ```python
  kv_manager.ingest_streaming(sid, layer, curr_k, curr_v)
  ```
- **Conclusion**: Prefill correctly uses streaming sparse ingest.

### 3. Decode Phase
- `ContinuousBatchEngine._step()` executes the decode step.
- It again calls `self.wrapper.model` with **NO `past_key_values`**.
- Inside `diffkv_attention.py`, the code hits the decode branch (`q_len == 1`):
  ```python
  kv_manager.ingest_streaming(sid, layer, curr_k, curr_v)
  blocks = kv_manager.get_streaming_blocks(sid, layer)
  # ... calls fused_sparse_attention_decode()
  ```
- **Conclusion**: Decode correctly uses streaming ingest for the new token and sparse batched attention.

---

## The `generate()` Bypass (Legacy Code)

There is a method `DiffKVHFWrapper.generate()` in `hf_diffkv_wrapper.py`.
```python
outputs = self.model(
    input_ids=input_ids, past_key_values=past_kv, use_cache=True
)
```
- This method passes `past_key_values` explicitly.
- When `past_key_values` is provided, the Hugging Face native attention uses it, bypassing the DiffKV monkey-patch.
- **However**, this `generate()` method is **never called** by the `ContinuousBatchEngine`. It is a legacy method from earlier prototyping.

## Verification Result
- OpenWebUI serving via the batch engine **does** use the full Differential KV Phase 24.5 streaming sparse ingest path.
- Legacy dense pathways are effectively dead code in production serving.
