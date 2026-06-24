# DiffKV Native Factual Gating Improvements

This directory contains the optimized C++ implementation of the Differential KV (DiffKV) native core runtime (`diffkv_native`).

To ensure parity between the native C++ runtime and the Python/MLX wrapper implementations, we have ported the following factual gating and alignment improvements:

## Summary of Changes

### 1. Factual Store Decode Queries (`serving/batch_engine.cpp`)
* **Problem**: The batch engine (which powers the server/OpenAI-compatible gateway) did not query the factual store during decode steps. As a result, the `current_step_factual_tokens` and `current_step_factual_sequences` variables were never populated, leaving factual alignment features (logit bias, VSL, SFA) inactive in serving mode.
* **Fix**: Added the full factual store query block into the decode loop of `BatchEngine::execute_request` after token ingestion, aligning it with the interactive CLI loop in `main.cpp`.
* **Details**:
  * Extracts the layer-0 decode key vector (`decode_k[0]`) to serve as a proxy query.
  * Performs query anchor blending (20% current, 80% stable anchor).
  * Executes `FactualExactStore::query()` passing the blended query, routed slot filters, and query entity bias.
  * Populates metadata for the subsequent step and invokes VSL step tagging (`diffkv::process_and_tag_vsl_step`).

### 2. Early Entity Binding with important_vocab (`src/main.cpp` & `serving/batch_engine.cpp`)
* **Problem**: The C++ implementation of early entity binding evaluated query token overlap across all prime entry tokens directly, without filtering out common stopwords, leading to potential entity context contamination.
* **Fix**: Aligned C++ with the Python wrapper by filtering both the query tokens and prime entry tokens using the inverted index's `important_vocab` (high-IDF tokens) during the start-of-response binding check.

### 3. Directed Relation Triples & Neighbor Injection Parity
* Handled C++ parity for:
  * Injecting triple sequences (`triple_sequences`) from prime entries.
  * Blending 1-hop (`>= 0.35`) and 2-hop (`>= 0.50`) neighbors.
  * Restoring the lexical tripwire (injecting facts based on high-IDF last-generated tokens).
  * Enforcing coherence cap scaling and entity-subgraph tagging.

## Compilation

Build the C++ library from this directory:

```bash
cmake -B build && cmake --build build --config Release
```

The resulting binary will be saved at `build/diffkv_native`.
