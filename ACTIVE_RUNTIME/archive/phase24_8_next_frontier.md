# Phase 24.8 — The Next True Frontier

## Context
With eager attention materialization eliminated, the $O(N)$ KV Cache bottleneck and the $O(N^2)$ Activation bottleneck are solved. The prefill engine can stably digest 100K+ sequences without OOMing on attention compute.

## What is the Next Limit?
The prompt offered several candidates:
1. `logits tensor memory`
2. `FFN activations`
3. `distributed synchronization`
4. `sparse transformer routing`
5. `multi-GPU slab ownership`
6. `retrieval orchestration`
7. `activation recomputation`

### 1. FFN Activations
The Feed-Forward Network (MLP) layers project the sequence to an intermediate size (e.g., $4 \times \text{hidden\_size}$). 
For a 100K token sequence, `[1, 100000, 4864]` in fp16 consumes ~972 MB per layer. Because FFN is token-wise independent, this can trivially be chunked alongside the attention prefill. It is not a structural blocker.

### 2. Logits Tensor Memory
The final language modeling head projects the hidden states to the vocabulary size.
For a 100K sequence: `[1, 100000, 151936]` (Qwen2.5 vocab).
In fp16, this single tensor consumes **~30.3 GB**.
This will instantly OOM the system at the very last layer of the forward pass, even if all attention layers executed cleanly.

However, during prefill, we only need the logit distribution for the *final* token to generate the next word (unless we are training or calculating perplexity). We can slice the final hidden state to `[-1:, :]` before passing it to the `lm_head`, reducing the logits tensor to `[1, 1, 151936]` (300 KB). This is an easy patch, not a structural limit.

### 3. Distributed Synchronization / Multi-GPU Slab Ownership
As context lengths push into the millions (or we serve hundreds of concurrent 25K sessions), a single GPU will eventually exhaust its physical VRAM, even with high compression ratios.

The true next structural frontier is moving from a single-node memory manager to a distributed memory manager.

If Differential KV is the canonical owner of the sparse slabs, how do we shard those slabs across multiple GPUs?
- Does GPU 0 hold chunks 0-100 and GPU 1 hold chunks 101-200?
- If we use chunked sparse prefill with anchor routing, GPU 0 might need to fetch a chunk from GPU 1 during the prefill phase.
- This requires **Multi-GPU Slab Ownership** and **Distributed Synchronization**.

## Decision

The next true dominant blocker is:
**Multi-GPU Slab Ownership** (Distributed Sparse Serving).

The engine is single-node complete. It must now become distributed.
