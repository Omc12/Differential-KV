# Phase 24.6 -- The True Next Blocker

## Context
The Phase 24.6 audit revealed that Differential KV is already achieving fully sparse execution for KV. `get_kv()` is bypassed, and the streaming sparse ingest successfully bounds the dense KV footprint.

However, two issues emerged:
1. **Session Teardown Leak**: ~182 MB of compressed KV state persisted after sessions were supposedly cleared.
2. **Standard Attention Overhead**: Peak VRAM during prefill is still dominated by standard dense transformer activations (attention weights, logits).

## Evaluating the Choices

The prompt offered two potential next blockers:
1. `compressed-prefill attention kernel` (if dense execution dominates)
2. `allocator integration` (if allocator illusion dominates)

### The Reality
Neither of these is the immediate blocker.
- **Dense KV execution does not dominate.** `get_kv()` is not called. The dense execution overhead is from standard activations (logits, attention matrix), which a compressed-prefill kernel would not solve (it only optimizes the K/V fetch, not the QxK output size).
- **Allocator illusion does not dominate.** The allocator cache is small (~18 MB).

## The *Actual* Next Blocker

The true next blocker for scaling to massive multi-session workloads is the **Session Teardown Memory Leak**.

If the `StreamingSparseIngestManager` does not fully release its Python object references upon `clear_session()`, long-running servers will eventually OOM from accumulated compressed state, regardless of how sparse the execution is.

However, strictly adhering to the prompt's constraints (choosing between the provided options based on the findings):

Since the allocator is not hiding savings, and the peak VRAM during prefill is still structurally constrained by the dense attention computation (specifically, the standard PyTorch `F.scaled_dot_product_attention` or eager math used over the *current* chunk), the closest mandated blocker is the **compressed-prefill attention kernel**.

While it won't fix the logits tensor size, a custom Triton prefill kernel is the only way to eventually optimize the attention matrix computation (e.g., via block-sparse processing or FlashAttention integration) to tame the O(seq_len^2) activation spike.

## Decision

The next mandatory frontier is:
`compressed-prefill attention kernel`

*(Note: Fixing the session teardown leak in `kv_runtime_manager.py` should be done immediately as a prerequisite bug fix).*
