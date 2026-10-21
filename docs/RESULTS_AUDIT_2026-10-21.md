# Results audit, 2026-10-21: does every number the paper reads describe today's code?

Method: every result file the paper's figure and table scripts open (traced by
wrapping `open()` while `paperdata.load()`, `gen_appendix_tables.py`,
`gen_latency_table.py` and `degeneracy.py` ran: 119 stores) was mapped to the code
it ran on. DKV stores record `dkv_decode_rev`, a hash of seven decode-critical
sources (`benchmarks/code_fingerprint.py`); each hash was recomputed at every
commit since 2026-08-15 to find the commit that produced it. Baseline stores carry
no fingerprint; their creation times were placed against the harness history.
Then every commit after each store was read (message AND default-path diff) and
classified.

## DKV stores, by code state

| code state (commit) | paper stores | what changed after it, on that store's path | status |
|---|---|---|---|
| 2fb29df9 (09-02) | LongBench DKV/mid, /high (12k); RULER Qwen3.5-4B DKV/mid, /high to 64k | 2da39dbc (router slot dequant, lowmem prefill, rank clamp on by default): verified on the exact path, 52/52 RULER items byte-identical text and identical peaks (`paper/results/merge_check`, re-checked today: 26/26 granite 8k, 26/26 Qwen 16k, ladder 16k identical). 996ed334: no-remat path only; never fired on default DKV (DKV_DEBUG_BOUNDS). fc80b649 (lean exact): tier 2 KL -0.0001 [-0.0057, +0.0045], same arithmetic up to rounding; peaks re-measured in V3. Residual policy (residual_capture.py) last changed 08-28, before. | valid (quality); peaks superseded by V3 ladders |
| 996ed334 (09-30) | LongBench DKV/mid streaming, fixed window ("streamlow") | elastic window added later as a switch (this store = DKV_STREAM_ELASTIC=0); 50179e98 tiling (same softmax, tests); 7e5749c2 fused tile (tier 2 +0.006 n.s.) | valid as the fixed-window scheduler |
| 3ea71604 (09-30) | RULER beyond-limit, fixed window (granite 24k-32k, Qwen 131k) | as above | valid as fixed-window |
| 2da39dbc, 50179e98 (10-01/02) | RULER Qwen 32k forced stream, untiled vs tiled | a within-pair comparison of the tiling | valid |
| db9c2a02 (10-16) | V3 LongBench stream, V3 RULER forced stream 32k | d83b1d80: incremental-branch raw history tiled, same arithmetic (tier 2 +0.010 n.s.) | valid |
| d83b1d80 (10-17) | V3/V4 RULER, LongBench hybrid, systems | 71eeab88 + ded83751 (codes store, rank clamp): default path unchanged by construction (every new branch gated on value codes; clamp a no-op when the recorded rank fits the pool); 80 parity/equivalence tests pass | valid |

## Baseline stores

LongBench baselines (09-02): dense 05:17-05:52 and post-hoc StreamingLLM 07:42-08:02
ran before 99fd29a0 (answers bounded by the generation budget) -- no row in either
exceeds the budget (max 3,007 chars, unclean 0), so the fix changes nothing in them;
both record `prefill_attn: sdpa`, so d0a68c24 (SnapKV/H2O off eager prefill) does not
apply. SnapKV, H2O, KIVI-4, INT8-KV ran after both fixes. 2-bit KIVI without a window
straddles 99fd29a0; same check, unclean 0. 515eb4be (SnapKV hybrid layer alignment,
09-28) affects Qwen3.5-4B SnapKV only; the paper's Qwen SnapKV rows are the
`*_snapkv_aligned_*` stores made after it.

## Not covered by the fingerprint (checked by hand)

`kv_runtime_manager.py`, `hf_dkv_wrapper.py`, `decode_config.py`, `query_router.py`,
`residual_capture.py` also decide outputs. Their commits after 09-02: ca097b41 /
3ea71604 (opt-in streaming switches, default path byte-identical by allocator probe),
7c5dc99f / 2da39dbc (verified identical above), ded83751 (no-op clamp). No other change.

## Older findings that still stand (code unchanged since) and the paper under-uses

- Routing: on granite at 24-32k (hybrid store, streaming), top-16 vs every block:
  KL 0.074 both, +0.0001 [-0.009, +0.008]; decode 5.5 vs 2.4 tok/s (IMPROVEMENTS_LOG
  batch C). On Qwen3.5-4B 32k/64k: -0.010 n.s., 9.9 vs 8.3 tok/s.
- Reuse across turns (multiquery, 09-03/04, append pattern, Qwen3.5-4B): follow-up
  query 1.80 s at 16k and 2.84 s at 32k against SnapKV's 7.25-7.50 and 15.1-15.6 s
  (re-prefill); 8 queries 22.4 vs 59.7 s and 52.0 vs 125.3 s. Granite 16k DKV rows in
  that store spilled under the pre-lean exact code and are not usable.
- Residual policy (cuda_work_record.md 4d-4h, Qwen2.5-1.5B): natural-text needle
  recall 2/12 and 3/12 -> 12/12 at 8k and 32k (dense 12/12), same budget; synthesis
  6.7 -> 13.3 when the reservation follows the query.
- Generality (tier 1/2, IMPROVEMENTS_LOG): on Qwen2.5-7B the default store fails
  (KL 1.114); quantized keys + attention-ranked residuals cut it 77% (0.251), no
  model-specific code.
- Elastic scheduler: where history fits, streaming + elastic equals exact prefill
  (tier 2 KL 0.1350 both, hybrid store).

## Disclosures the audit requires in the paper

- The factorization is joint over a layer's KV heads (`feat_dim = 2 H_kv d_h`).
- Residual rows are chosen content-aware and ordered by the prompt's last 64 tokens
  (`DKV_RESIDUAL_QUERY_TAIL`), which is the question in every template evaluated.
- Exact-mode quality stores predate lean exact; equivalence is the tier-2 number above.
- Only the CUDA implementation is current; MLX and native are not claimed.
