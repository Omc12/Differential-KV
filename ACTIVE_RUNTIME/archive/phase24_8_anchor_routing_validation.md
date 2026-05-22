# Phase 24.8 — Retrieval-Aware Sparse Prefill (Anchor Routing)

## The Problem with Chunked Prefill
If we strictly use a local window (e.g., attending only to the previous 512 tokens and the sinks), we achieve linear $O(N)$ time, but we destroy the model's ability to retrieve facts from the distant past (e.g., Needle In A Haystack). 

## Reviving Global Anchor Routing
We revived the concepts from `research/sparse_prefill_anchors.py`. 

### Mechanism
1. **Anchor Pool:** 
   When the `StreamingSparseIngestManager` compresses a 16-token micro-block, it retains a single uncompressed dense token: the `anchor_kv`.
   This collection of anchors forms a highly compressed semantic map of the entire document (1 token per 16 tokens).
   
2. **Chunk Routing:**
   When processing a query chunk (512 tokens), we mean-pool the query vectors: $Q_{pool} = \text{mean}(Q)$.
   We compute relevance against the global `anchor_kv` pool: $Relevance = Q_{pool} K_{anchors}^T$.
   
3. **Sparse Retrieval:**
   We select the top-$K$ blocks with the highest relevance. We then page those specific blocks into the SRAM attention window. 
   
## Validation
By integrating anchor routing, the prefill chunk has $O(1)$ visibility into the entire historical sequence. If a specific fact (a "needle") is located at token index 15,000, its `anchor_kv` will score highly against the query chunk, pulling that specific 16-token micro-block into the local attention window.

The prefill maintains its $O(N)$ computational scaling, bounds its activation memory via chunks, and preserves full-sequence retrieval accuracy.
