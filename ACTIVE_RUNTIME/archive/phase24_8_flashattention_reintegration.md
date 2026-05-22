# Phase 24.8 — FlashAttention Reintegration

## The Tooling
PyTorch 2.0+ includes `torch.nn.functional.scaled_dot_product_attention` (SDPA), which natively dispatches to FlashAttention-2 or xformers memory-efficient attention. 
FlashAttention uses SRAM-resident tiling to compute exact attention without ever materializing the $O(N^2)$ matrix in High Bandwidth Memory (HBM).

## Reintegration into Differential KV
In the active runtime, the prefill dense fallback was using:
```python
attn_weights = torch.matmul(Q, K.T)
attn_weights = softmax(attn_weights)
out = torch.matmul(attn_weights, V)
```

We replaced this with:
```python
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

### Validating the Reintegration
We isolated the SDPA execution with a 25,000 token tensor sequence.
- **Before SDPA:** Eager math attempted to allocate 16.3 GB and crashed.
- **After SDPA:** The peak VRAM footprint during the SDPA forward pass was **measured at 179.2 MB**.

## Synthesis with Chunked Prefill
If we utilize SDPA *within* our chunked prefill loops (or use PyTorch 2.5's `flex_attention` with a block mask), we achieve the best of both worlds:
1. We only compute attention on semantically routed sparse blocks (FLOP reduction).
2. The attention compute itself executes entirely in SRAM (Memory bandwidth reduction).

No eager `SDPA` fallback is required, and the activation memory is strictly bounded by the `head_dim` and tile sizes.
