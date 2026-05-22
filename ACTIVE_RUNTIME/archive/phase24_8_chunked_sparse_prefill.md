# Phase 24.8 — Chunked Sparse Prefill Revival

## Objective
Revive the chunk-local execution from Phase 13/14 to process long sequences without materializing full `[seq_len x seq_len]` matrices.

## Mechanism Integration

We integrated the core concepts from `research/sparse_prefill.py` into the active prefill path.

1. **Bounded Chunk Execution:**
   The massive 25K query sequence is sliced into chunks of `chunk_size = 512`. This strictly bounds the query length `Q` to 512 tokens at a time.
   
2. **Rolling Sparse Windows:**
   Instead of each 512-token chunk attending to the entire 25K sequence, the chunk ONLY attends to:
   - **Sinks:** The first 64 tokens of the prompt.
   - **Local Window:** The immediately preceding 512 tokens.
   - **Self:** Its own 512 tokens (causally masked).
   - **Retrieved Anchors:** Semantically relevant historical chunks.

3. **Avoidance of Eager Materialization:**
   Because the key/value context for any given chunk is bounded to `64 (sink) + 512 (local) + 512 (self) + retrieved_tokens`, the resulting attention matrix is maximum `[512 x ~2048]`. 
   
## VRAM Reduction Reality
By slicing the execution, the maximum intermediate tensor size Drops from:
- `14 * 25000 * 25000 * 2 bytes = 16.3 GB`
Down to:
- `14 * 512 * 2048 * 2 bytes = 29.3 MB`.

This represents a **556x reduction** in activation memory during prefill, allowing effectively infinite context scaling within a fixed SRAM/VRAM budget.
