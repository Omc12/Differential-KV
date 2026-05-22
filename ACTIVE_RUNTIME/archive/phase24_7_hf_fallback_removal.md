# Phase 24.7 — HF Fallback Cache Removal

## Audit of Fallback Paths

We audited the production serving paths to ensure no Hugging Face (HF) native `past_key_values` fallback paths silently allocate dense KV.

### 1. `batch_engine.py` (Production Serving)
```python
# PREFILL
out = self.wrapper.model(
    input_ids=input_ids, 
    position_ids=position_ids,
    use_cache=True # <-- Does not pass past_key_values
)

# DECODE
out = self.wrapper.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    use_cache=True # <-- Does not pass past_key_values
)
```
**Status: CLEAN.** The batch engine never passes `past_key_values` into the model, forcing the DiffKV patch to handle 100% of KV state.

### 2. `hf_diffkv_wrapper.py` (Legacy `generate()`)
```python
outputs = self.model(
    input_ids=input_ids, past_key_values=past_kv, use_cache=True
)
```
**Status: DIRTY (but unused).** This method explicitly passes and updates `past_key_values`, silently accumulating a dense HF cache. However, this method is never invoked by the `openai_compatible_api_gateway.py` or the `ContinuousBatchEngine`. It is dead legacy code.

### 3. `diffkv_attention.py` (The Patch)
```python
outputs = (attn_output,)
if output_attentions:
    outputs += (attn_weights,)
if use_cache:
    outputs += (None,) # <-- Forces HF cache to remain empty
return outputs
```
**Status: CLEAN.** The patch actively destroys the HF KV return path by yielding `None` for `past_key_values`.

## Conclusion
The production path is fully severed from Hugging Face's dense `past_key_values` mechanics.
