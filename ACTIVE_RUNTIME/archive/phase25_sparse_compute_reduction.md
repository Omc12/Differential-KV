# Phase 25 — Sparse Compute Reduction

## Measuring FLOP & Latency Reduction

After identifying that the 25K prompt was bottlenecked by dense attention FLOPs and SVD compression overhead, we made two critical changes:
1. **Compute Routing:** Bypassed dense SDPA for `RetrievalAwareSparsePrefill`.
2. **SVD Backpressure Release:** Aligned the `micro_block_size` of the streaming ingest manager from 16 to 512, explicitly matching the attention chunk size. This reduced the number of SVD operations from 1,562 to 48 per layer.

### Live 25K Token Validation Results

| Configuration | Peak VRAM | Prefill Latency | Attention FLOPs (est) |
|---|---|---|---|
| **Eager Baseline (Pre-Phase 24.8)** | OOM (>17 GB) | N/A (Failed) | 1.12 TFLOPs |
| **Dense SDPA (Pre-Phase 25)** | 9,288 MB | 63.27 seconds | 1.12 TFLOPs |
| **Dense SDPA + Logits Patch** | 2,562 MB | 61.80 seconds | 1.12 TFLOPs |
| **Retrieval-Aware Sparse Compute** | **2,295 MB** | **7.95 seconds** | **~0.07 TFLOPs** |

## Conclusion
We have successfully transitioned from a *bounded-memory* system into a **bounded-compute** sparse execution system. Latency dropped by **87%** (from 63s to under 8s) for a 25K prompt by eliminating wasted dense compute and redundant micro-compressions.
