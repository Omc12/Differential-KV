# Phase 25 — Last-Token Logits Optimization

## The Wasteful Baseline
Standard autoregressive LLMs during a prefill forward pass evaluate the `lm_head` across the entire sequence. For a 25,000 token prompt on Qwen2.5-0.5B (vocab size 151,936), the resulting tensor is `[1, 25000, 151936]`. In fp16, this single tensor consumes **~7.6 GB** of VRAM, entirely dominating the memory footprint even after attention optimization.

## The Patch
Since we only generate the *next* word, we only need the logits for the final token in the sequence.
We successfully monkey-patched `model.lm_head.forward` directly inside the Differential KV attention interceptor logic:

```python
original_lm_head_forward = model.lm_head.forward
def last_token_lm_head_forward(hidden_states):
    if hidden_states.shape[1] > 1:
        # Only project the last token
        return original_lm_head_forward(hidden_states[:, -1:, :])
    return original_lm_head_forward(hidden_states)
model.lm_head.forward = last_token_lm_head_forward
```

## Validation & Results
We ran the live 25K prompt test.
- **Before Patch:** Peak VRAM was **9,288.2 MB**.
- **After Patch:** Peak VRAM plummeted to **2,562.1 MB**.

**Impact:** A direct saving of ~6.7 GB of VRAM. The system no longer allocates any `[seq_len, vocab_size]` tensors, effectively rendering the final logits projection $O(1)$ with respect to sequence length during prefill generation.
