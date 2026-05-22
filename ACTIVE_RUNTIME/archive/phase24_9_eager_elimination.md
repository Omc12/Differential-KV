# Phase 24.9 — Eager Attention Elimination

## Hard-Disabling Eager Fallback

To prevent any possibility of the system silently falling back to un-patched, eager attention paths, we enforced strict measures in `diffkv_attention.py`:

1. **Eliminated `torch.matmul(Q, K.T)` entirely:** The code has been physically modified to replace the GQA dense path with a direct call to `torch.nn.functional.scaled_dot_product_attention`.
2. **Explicit Mask Handling:** 
   - If custom `attention_mask` is provided, we route to SDPA with `attn_mask=attention_mask` and `is_causal=False`.
   - If no mask is provided and `q_len > 1` (prefill), we route to SDPA with `is_causal=True`.
3. **No Silent Fallbacks:** There are no conditional `try-except` blocks that would fallback to manual eager math in case of SDPA failure. If SDPA fails, the engine fails loudly and immediately, guaranteeing 100% compliant, memory-bounded execution.
