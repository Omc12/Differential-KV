# Phase 25 — Prefill Compute Distribution Trace

## The 63-Second Bottleneck
During Phase 24.9, we verified that memory footprint dropped to 2.5 GB for a 25K prompt (thanks to SDPA and the last-token logits patch). However, the prefill latency was **63.27 seconds**. Why?

## FLOP Distribution Breakdown

### 1. The Dense Attention Compute (SDPA)
We used `F.scaled_dot_product_attention` on the full `[1, 14, 25000, 64]` tensors. 
- SDPA FLOPs for 25K tokens: $2 \times 14 \times 25000^2 \times 64 \approx 1.12$ TFLOPs.
- Across 24 layers: $24 \times 1.12 = 26.8$ TFLOPs.
- On a modern GPU (e.g. RTX 3090 / 4090 with 35–80 TFLOPs FP16 compute), this SDPA operation should take **< 1.0 second**.

### 2. The FFN / QKV Projections
- QKV Projections: $25000 \times 24 \times (1024 \times 3072) \approx 3.7$ TFLOPs.
- FFN Projections: $25000 \times 24 \times (1024 \times 4864 \times 2) \approx 11.9$ TFLOPs.
- Total standard linear layer compute: $\sim 15$ TFLOPs (takes **< 0.5 seconds**).

### 3. The True Bottleneck: Synchronous SVD Backpressure
Inside `dkv_attention.py`, the streaming ingest manager executes:
```python
kv_manager.ingest_streaming(sid, captured_layer_idx, curr_k, curr_v)
```
- A 25K prompt is sliced into micro-blocks of 16 tokens.
- This creates $25000 / 16 = 1562$ micro-blocks per layer.
- Over 24 layers, that is **37,488 micro-blocks**.
- The `AsyncCompressor` queue fills up immediately (usually queue size is 64).
- The `StreamingSparseIngestManager` hits backpressure and forces synchronous execution:
```python
if not submitted:
    self.compress_fn(block, k, v) # Synchronous SVD fallback!
```
- Computing 37,488 SVDs synchronously during the prefill forward pass completely stalls the GPU execution, turning the GPU-bound prefill into a CPU-bound iterative loop. 
- **Time spent:** ~61 seconds.

## Conclusion
The prefill compute is dominated by:
1. **Synchronous SVD compression** (due to backpressure from massive instant ingest).
2. **Dense 25Kx25K Attention** (even if in SRAM via SDPA, it's still $O(N^2)$ FLOPs).

To reduce active compute, we must chunk the prefill execution (Task 4) and we must restructure compression to avoid synchronous blocking (or run it natively in a fused GPU kernel).
