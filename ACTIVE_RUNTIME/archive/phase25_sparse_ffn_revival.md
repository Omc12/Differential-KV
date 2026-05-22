# Phase 25 — Sparse FFN Revival

## Context
During early phases, we explored `sparse_mlp.py` and `tiered_ffn.py` to compress the feed-forward networks. While mathematically sound, they introduced high orchestration overhead that often negated the VRAM savings during dense eager attention execution.

## Re-evaluating FFN Sparsity under FlashAttention
Now that attention compute is explicitly bounded (using chunked prefill and FlashAttention), FFN activations are actually the *largest* remaining compute cost during a forward pass. 

1. **Late-Layer Sparsity:** Qwen2 layers exhibit high sparsity in the `up_proj` outputs. Dropping near-zero activations before the `down_proj` can eliminate ~30-40% of FFN FLOPs without hurting perplexity.
2. **Block Routing / MoE Emulation:** We can apply our chunked semantic anchors to FFN layers as well. Instead of computing the MLP on all tokens, we can route only the "active" tokens (e.g. retrieved from the semantic pool) through the FFN, leaving historical context un-updated.
3. **Reduced Active FFN Compute:** If we implement fused sparse kernels (e.g., Triton kernels that only multiply non-zero elements), the FFN FLOPs will drop proportionally to the activation sparsity.

## Recommendation
Sparse FFN optimizations are now **highly practical**. Because the attention memory bottleneck is gone, any FLOP reduction in the MLP immediately translates to lower end-to-end latency. Fused sparse Triton kernels should be the next major implementation target for FFN.
