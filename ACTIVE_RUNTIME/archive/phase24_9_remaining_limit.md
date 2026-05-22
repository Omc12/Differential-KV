# Phase 24.9 — The True Remaining Limit

## Context
With the O(N^2) HBM attention matrix completely solved, we can easily run 25K+ context sessions. However, the system peak VRAM is still dominated by another massive token-length dependent tensor.

## The Dominant Limit: Logits Tensor Memory

During our successful 25K token test, the total VRAM reached **9.28 GB**.
Out of that, **7.6 GB** was allocated by a single tensor:
```
[1, 25000, 151936] (logits)
```

In autoregressive generation, we only need the logits for the *last* token to predict the next word. However, standard Hugging Face model forward passes evaluate the logits for *every* input token during prefill.

### How to Solve It
We can project only the last hidden state to vocab size in the model's final LM head during prefill:
```python
# Hidden states: [1, seq_len, hidden_size]
last_hidden_state = hidden_states[:, -1:, :]
logits = lm_head(last_hidden_state)
```
This simple patch reduces logits memory from **7.6 GB to 300 KB** for a 25K prompt, immediately slashing peak prefill VRAM to **< 1.7 GB** total.

**THE REMAINING LIMIT:**
Logits Tensor Memory is the true dominant remaining limit for context scaling.
