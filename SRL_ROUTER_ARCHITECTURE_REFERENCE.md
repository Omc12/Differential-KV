# SRL Router Architecture — Reference for Future Improvement

**Date:** 2026-07-16
**Status:** Not deleted, not deprecated — benched as the CUDA default in favor of a simpler MLX-parity router. Fully intact and reachable via `DIFFKV_ROUTER=srl`. This document exists so the design and the evidence against it are legible without re-reading `query_router.py` end to end.

## TL;DR

CUDA has two block-routing implementations behind one switch:

```
DIFFKV_ROUTER=residual   (default, 2026-07-16+)   → route_blocks_relevance()   — MLX-parity port
DIFFKV_ROUTER=srl                                 → route_query_fixed_k()      — this document's subject
```

The `srl` router was the CUDA-native design (multi-channel: lexical + semantic + graph + recency). It was switched off as the default after being measured net-negative twice in a whole-document-synthesis eval — not deleted, because (a) most of its supporting machinery (`SessionSRLState`, `ChunkGraph`, `InvertedTokenIndex`, `FactualExactStore`) is genuinely used by other features (entity/relational binding, factual-store lookup) that are independent of routing quality, and (b) it may simply be a poor fit for the benchmark it was tested against rather than a poor design outright. Read the "What is and isn't confirmed" section before assuming it should stay off.

## Where the code lives

| File | Role |
|---|---|
| `ACTIVE_RUNTIME/native_core/srl/query_router.py` | `route_query()` (the SRL scorer), `adaptive_k()`, `two_level_gate()` (anchor rerank), `route_query_fixed_k()` (the public entry point + K-padding wrapper), and — new — `route_blocks_relevance()` (the MLX-parity replacement). ~1000 lines. |
| `ACTIVE_RUNTIME/native_core/srl/chunk_graph.py` | `ChunkGraph` + `build_chunk_graph()`: block-to-block similarity graph with cluster centers ("prime nodes") and neighbor edges, used for SRL's graph-expansion channel. |
| `ACTIVE_RUNTIME/native_core/srl/session_srl_state.py` | `SessionSRLState`: per-session routing state. **Only a fraction of its fields are routing-relevant** (see below) — most belong to a separate relational/entity-binding feature set (RC1–RC8 in project history) that reuses this same dataclass. |
| `ACTIVE_RUNTIME/native_core/srl/inverted_index.py` | `InvertedTokenIndex`: lexical token → (slot, position) occurrence index with IDF weights, feeds SRL's lexical channel. |
| `ACTIVE_RUNTIME/native_core/srl/factual_store.py` | `FactualExactStore` — a **separate opt-in feature** (`DIFFKV_FACTUAL_STORE=1`), not part of routing. Do not conflate audits of the two. |
| `ACTIVE_RUNTIME/runtime/diffkv_attention.py` (~line 616–700) | The decode-time call site: gates on `srl_state.routing_threshold`, dispatches to whichever router `DIFFKV_ROUTER` selects, then maps the returned slot IDs back to `block_indices`/`anchor_indices` for the kernel. |

`SessionSRLState` fields that are **routing-relevant** (relevant to this document):
`ordered_slot_ids`, `sink_blocks`, `recent_miss_rate`/`k_multiplier`/`call_count` (adaptive-K), `recent_generated_tokens`, `segment_ids`/`current_query_segment_id` (Structured Attention Segmenting), `current_step_slots`, `k_min`/`k_max`/`k_*_frac`, `routing_threshold`, `overlap_threshold`, `graph_hop_decay`, `srl_age_penalty`.

Fields that are **not** routing — they belong to entity/relational binding and factual-sequence tracking (a different fix program, see project memory on relational binding RC1–RC8): `current_entity_id`, `dual_entity_mode`/`dual_entity_ids`, `comparison_entities`/`comparison_active_idx`/`comparison_covered`, `current_step_sequence_entity_ids`/`current_step_sequence_is_prime`/`current_step_sequence_prefixes`, `current_step_factual_tokens`/`current_step_factual_sequences`, `prompt_eagle_scores`/`prompt_anchors`/`dynamic_anchors` (EQA-DR), `factual_anchor_q`. If you extend or replace the router, leave these alone — they're consumed elsewhere in `diffkv_attention.py`'s factual-store path regardless of which block router is active.

## The SRL router's design (`route_query`, ~query_router.py:254–865)

A per-decode-step (cached across all 28 layers, recomputed at layer 0 only) multi-channel candidate-generation-then-rerank pipeline:

1. **Lexical channel** — looks up the current query's tokens (or, absent explicit query tokens, a tail of `recent_generated_tokens`) in the inverted index; scores candidate slots by IDF-weighted term overlap with a configurable decay factor (`DIFFKV_SRL_DECAY_FACTOR`) and a "query term coverage" boost (`n_unique_matches²`). Produces `lexical_slots` and a `rare_lex_slots` subset (IDF ≥ 2.0, i.e. distinctive/rare tokens).
2. **Semantic channel** — projects the query through `pool.W_proj` into a 64-dim descriptor, does ANN search over block descriptors either directly (`semantic_index.search`) or, if the chunk graph has cluster centers, via a hierarchical two-hop walk: score cluster centers, select active clusters (similarity ≥ max(0.30, 0.85·S_max)), then expand to children.
3. **Topic-switch detection** — if the best semantic match is weak (below `_TOPIC_SWITCH_THRESHOLD`), treat this as a new topic and suppress lexical/rare seeds from anchoring graph expansion (avoids "sticking" to a stale prior topic).
4. **Chunk-graph neighborhood expansion** — seeds = top semantic + rare-lexical + lexical slots; propagates an activation score one or more hops through `chunk_graph.neighbors` (weighted edges), with **segment forbidding**: if the current query is tagged to "segment 1" or "segment 2" (Structured Attention Segmenting — used to keep multi-entity comparisons from bleeding into each other), rows belonging to the *other* segment are excluded from expansion entirely.
5. **Sink blocks** — block 0 + special-token blocks (`<|system|>`, `<|user|>`, etc.) are always force-included, never filtered by any later step.
6. **Combine** — concatenate sink → semantic → rare-lex → graph → lexical → recent → any dynamic/prompt-routed slots, dedupe.
7. **`two_level_gate` rerank** (~query_router.py:72–184) — cheap anchor-only `q·k` dot product over the combined candidate set (GQA-aware head grouping), **minus an age penalty** (`age × srl_age_penalty`, age = position in `ordered_slot_ids` from the end) that biases toward recent blocks, plus a slot-reinforcement-strength boost. Truncates non-sink candidates to `K − len(sink)`.
8. **`route_query_fixed_k`** wraps this and used to zero-pad/truncate to a fixed `K_FIXED=64` — the duplicate-padding bug (fixed 2026-07-16, see git history) lived here.

Compare to MLX's router (`serving/mlx_diffkv_wrapper.py:1047`/`1086`, ported as `route_blocks_relevance` in `query_router.py`): rank every block once by `max(q·anchor, max q·residual)`, `argsort`, take top-K. No lexical index, no ANN clustering, no graph expansion, no segment logic, no age penalty. One function, ~40 lines, zero host↔device syncs.

## Known/likely defects (found during the 2026-07-16 audit, independent of the "which is better" question)

- **Age penalty biases against old blocks.** For whole-document synthesis (the eval this was tested against), that means the router systematically deprioritizes early-document content. Default changed 0.01→0.0 to match MLX (no recency term) — did **not** fully fix the degradation (see below), so this was a real defect but not the whole story.
- **Two silent `argmax(dim=1)→0` mappings** in the `diffkv_attention.py` call site (slot→anchor and anchor→block_indices) returned index 0 — i.e., silently substituted the sink block — whenever a selected slot/anchor wasn't found in a given layer's block set. Hardened to filter to actual matches (fixed 2026-07-16); latent, not confirmed to have fired in the observed runs.
- **Heavy per-token Python work**: `row_segs` list-comprehension over every semantic-index row, `slot_to_idx` dict rebuilt from scratch, multiple `.item()`/`.tolist()` calls — all per decode step, not cached. Real cost, not measured in isolation.
- **Lexical channel is likely dead in a direct-`model()`-call eval harness** (like `colab/run_nat_eval.py`): it depends on `srl_state.current_query_tokens`, which nothing in that harness populates; it falls back to a tail of `recent_generated_tokens`, which is a poor proxy for "what does this decode step need" on a single long generation.
- **fp16 semantic scoring on CUDA vs fp32 on MPS** (`desc_matrix @ q16` branches on device type) — a precision asymmetry that was noted but not chased down.

## What is and isn't confirmed

**Confirmed (from the 2026-07-16 NAT-eval runs, 13.4K-token whole-document synthesis, Qwen2.5-14B):**
- Forcing SRL routing on (`DIFFKV_SRL_THRESHOLD=16`, N_sparse≈43-45 of 49 blocks) produced shorter, more erratic outputs (64–121 tokens, several runs hitting the 256-token cap) than leaving routing off (190–250 tokens), across three separate runs.
- This held with the age penalty at its old default (0.01) **and** at 0.0 — so the age penalty is not the sole explanation.
- Routing-on tps was *lower* than routing-off tps in every run, which rules out "routing hurts quality but helps speed" as a story — at this block count it did neither.

**NOT yet confirmed:**
- **`route_blocks_relevance` (the MLX-parity port) has not been GPU-tested.** It has unit tests against synthetic pool data on CPU only (needle-via-residual retrieval, GQA head mapping, padding masks) — those confirm the math is correct, not that it produces better *NAT-eval* output than `route_query`. The "MLX's architecture wins" conclusion in project history is inferred from "SRL degrades vs. no-routing" plus "MLX's design is simpler and sync-free," not from a direct SRL-vs-MLX-router A/B on the same prompt. Run `DIFFKV_ROUTER=srl` vs the default back-to-back before treating that as settled.
- Whether SRL's degradation is a router defect at all, versus **any** block-dropping being wrong for whole-document synthesis (a task with no localized "needle," where nearly every block plausibly matters and top-K pruning of any kind may be lossy). If the MLX-style router also degrades output at N_sparse≈43-45 on this same prompt, that would point to the task/threshold rather than the router design.
- Whether SRL's real strength — the lexical + entity/relational machinery it shares state with — actually helps on tasks it's suited for (needle-in-haystack factual retrieval, table binding, entity comparison) even if it loses on document synthesis. Its `SessionSRLState` fields (dual-entity mode, comparison locking, prime-node entity tracking) were clearly built for exactly those tasks; this audit's benchmark wasn't.

## If you want to improve the SRL router later

1. **Get the A/B that hasn't been run**: `DIFFKV_ROUTER=srl` vs `DIFFKV_ROUTER=residual` (default) at the same `DIFFKV_SRL_THRESHOLD`, same prompt, same seed. That isolates the router itself instead of routing-on vs routing-off.
2. **Test it on what it was built for.** Re-run the relational-binding / factual-recall benchmarks (`benchmarks/niah_recall.py`, `RELATIONAL_BINDING_REPORT.md`, the RC1–RC8 fix history in project memory) with `DIFFKV_ROUTER=srl` vs `residual` — document synthesis may be the wrong yardstick entirely.
3. **If keeping SRL**, the cheapest wins are: cache `slot_to_idx`/`row_segs` across decode steps instead of rebuilding per token; populate `current_query_tokens` properly in serving paths that currently leave the lexical channel starved; consider whether the age penalty should be 0 by default everywhere or reserved for confirmed multi-turn scenarios only.
4. **If discarding SRL as a router** (keeping it only for its lexical/entity-binding machinery), `route_blocks_relevance` is the drop-in replacement — same call signature shape, same slot-ID contract, already wired behind `DIFFKV_ROUTER`.
