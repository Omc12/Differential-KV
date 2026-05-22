# Phase 24.9 — Execution Regime

## Production Execution Regime

After full cutover, programmatic layer auditing, and live 25K token validation, we classify the active production serving engine under the following regime:

### **FlashAttention Dense Execution (for Prefill) + Retrieval-Aware Sparse Decode**

1. **Prefill Phase:** 
   - Uses FlashAttention (`torch.nn.functional.scaled_dot_product_attention`) in SRAM to compute exact, causally-masked attention.
   - Prevents the allocation of eager `[seq_len x seq_len]` HBM matrices.
2. **Decode Phase:**
   - Routes through `batched_sparse_attn_decode` using Triton low-rank sparse execution.
   - Bypasses dense reconstruction entirely.

This hybrid regime ensures bounded activation memory during massive prefill phases and highly compact, sublinear KV cache footprints during generations.
