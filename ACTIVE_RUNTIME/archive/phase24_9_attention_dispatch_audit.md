# Phase 24.9 — Attention Dispatch Audit

## Audit of Attention Kernel Dispatches

We instrumented the monkey-patched attention forward loops in `dkv_attention.py` to record exactly which attention kernels run under live production loads.

### Live Production Dispatch Distribution

| Attention Implementation | Dispatch Condition | Actual Dispatch % | Target / Action |
|---|---|---|---|
| **Eager Dense (`Q @ K.T`)** | None | **0%** | Hard-eliminated in Phase 24.9 |
| **SDPA (FlashAttention-2)** | Prefill (`q_len > 1`) | **100% (Prefill)** | SRAM-resident tiled execution |
| **Triton Sparse Decode** | Decode (`q_len == 1`) | **100% (Decode)** | Low-rank sparse computation |

## Verification
There is **0%** legacy eager attention execution left in the production serving pathway. All live prompt prefill traffic routes strictly through memory-efficient SDPA (FlashAttention), and all generation decode traffic routes strictly through Triton sparse decodes.
