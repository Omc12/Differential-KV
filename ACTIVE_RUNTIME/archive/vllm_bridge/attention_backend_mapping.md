# Attention Backend Mapping

This document details how the Differential KV `TritonSparseDecode` kernel maps to the vLLM Attention Backend interface.

## The vLLM Interface
vLLM requires a custom backend to implement two primary operations:
1. `forward_prefill` (Handled by standard FlashAttention or our experimental chunked routing)
2. `forward_decode` (The target for Differential KV integration)

## The Differential KV Implementation (`DKVAttentionBackend`)

### Metadata Assembly (`make_metadata`)
vLLM builds `AttentionMetadata` on every step. Our backend intercepts this:
- Extracts the standard `block_tables`.
- Cross-references the `PersistentMetadataPool` to segregate Dense blocks (recent) from Compressed blocks (historical).
- Passes raw pointers (U, V arrays) into the metadata struct.

### Decode Execution (`forward_decode`)
Instead of calling a single `paged_attention_v1` kernel, the backend executes a split dispatch:
1. **Dense Phase:** Standard `paged_attention` runs over the Dense Recency Window blocks.
2. **Sparse Phase:** `TritonSparseDecode` runs over the Compressed blocks, directly reading the $U$ and $V$ matrices.
3. **Reduction:** The outputs of the Dense and Sparse phases are summed (using standard logsumexp reduction) to produce the final attention output.

This split execution is native to vLLM's architecture (similar to how prefix caching splits attention).
