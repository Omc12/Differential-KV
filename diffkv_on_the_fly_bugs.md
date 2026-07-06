# DiffKV — on-the-fly bugs & follow-ups (found while building sparse prefill)

Running log started 2026-07-06 (Opus 4.8, sparse-prefill pass). Each entry: what I saw, what I
think should happen, and where. These are NOT fixed here — they are parked for a later targeted
pass, per the user's instruction. Verify each still reproduces before acting on it.

---

## FOLLOW-UP 1 — Sparse prefill Stage 2: the MEMORY win (biggest lever, not yet built)
**Status:** by-design gap, not a bug. **Where:** `ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py`
`attention_forward` prefill branch + `MLXQwenModel.__call__`.

What ships now (`DIFFKV_SPARSE_PREFILL=1`) is **Stage 1 = COMPUTE-sparse prefill**: each chunk
attends to `[sink blocks + top-K routed history blocks + recency window + self]` instead of the
full KV, so prefill attention drops O(L²)→O(L·K). But it still calls
`cache.update_and_fetch(keys_rot, values)`, so the **full raw KV is retained** in the native MLX
prefill cache → **prefill peak memory is unchanged**. That's why sparse prefill does NOT extend
max context reach yet (a 64k prompt that OOMs dense still OOMs sparse).

The DSA/NSA "proper" version (Stage 2) also wins memory: after a block is streamed-compressed
(`compress_deferred_prefill_blocks` already runs per chunk in `niah_recall`), **evict its raw KV
from the native prefill cache**, keep only `[sink + rolling window]` raw, and have later chunks
route over the **compressed pool** and attend to **reconstructed** block K/V (reuse the
decode-cache `_execute_decode_cache` materialize helper — anchor row + comp_scale·(U·V) + exact
residuals, pre-rotated). Then prefill memory ≈ O(window + pool) instead of O(L).

Why deferred: (a) it's a much bigger change to the cache lifecycle (the native `cache` object
owns the KV; evicting mid-prefill means either not using it or a custom cache); (b) reconstructed
blocks add approximation to the *prefill* hidden states (Stage 1 only approximates by *dropping*
blocks — Stage 2 also approximates the *kept* ones), so it needs its own recall gate; (c) the
user prioritized porting Stage 1 to native first. Acceptance for Stage 2: NIAH multi-depth +
multi-needle still green at 16k/32k, AND a 48k/64k prompt that OOMs dense now completes.

## FOLLOW-UP 2 — Sparse-prefill defaults are Qwen-1.5B/NIAH-tuned
**Where:** manager `_sp_kmin=8`, `_sp_frac=0.05`. Verified only on Qwen2.5-1.5B-Instruct-4bit with
the NIAH/factual prompts. A DIFFUSE multi-fact/summarization query (synthesis_eval) or a larger
model may need a larger K — the mean-pooled block router is coarse and a diffuse query has no
sharp block to hit. Before enabling by default on any new model, sweep K vs `synthesis_eval` and
RULER, not just NIAH. (Same class of caveat the decode `SPARSE_BIAS`/`max_residual` carry.)

## FOLLOW-UP 4 — Native sparse prefill: Stage A BUILT ✅; Stage B (multi-fact routing) = TODO
**Status:** Stage A (positional StreamingLLM, single-needle) is **built + verified** —
`DIFFKV_SPARSE_PREFILL=1`, 16k prefill compute −34%, 6-cell NIAH 6/6 + conformance PASS. This entry
now tracks **Stage B: multi-fact routing** (the part that needs a query). **Where:**
`diffkv_native/src/main.cpp` — `build_prefill_ctx_graph` (`sp_ranges` param) + the chunk driver loop
(`sp_enabled`/`sp_ranges`/sparse mask). Stage A drops the ENTIRE middle (sink+window+current); it
passes single-needle because recall is a DECODE-time job (verified: MLX K=0 = single-needle NIAH
2/2, but multi-needle 0/1 — the model refuses without the far facts primed). **Stage B adds the
top-K routed history blocks so multi-fact prompts work**, matching MLX. The block-view gather is
already in place (just add more ranges); the only new piece is the router:

_Original two blockers (why a query-based router is the hard part):_

**Why native is HARDER than MLX (two real blockers):**
1. **GGML is a static graph.** In MLX the routing (`argpartition` over block scores) and the gather
   (`mx.take`) are dynamic ops inside one lazy graph — trivial. In GGML the graph is built on the
   host BEFORE it runs, and the router needs the chunk's **query**, which is computed *inside* the
   graph (`q = wq·h`). So you cannot both compute the query and use it to pick blocks in one pass.
   Options: (a) **2-pass** — pass 1 computes q_rope per layer (embed+rmsnorm+wq+rope), read back a
   pooled query, route on host, pass 2 builds sparse attention. Adds ~20-30% projection work but
   removes O(L²) attention → net win at long ctx. (b) **query-independent router** — route prior
   blocks by similarity to the current chunk's **own mean rotated-K** (K-K topical similarity, a
   rough proxy for q direction; MInference-style). No 2-pass, no query readback. Weaker router —
   MUST be gated on the 6-cell NIAH. Recommend building (b) first (low-risk), fall back to (a) only
   if (b) misses needles.
2. **The host rotated-K buffer isn't populated in the regime we need.** `k_rotated_activations[l]`
   (host, per-token rotated K) is filled per chunk at main.cpp:2937 **only when
   `!decode_use_sparse`**. But `decode_use_sparse = (L >= engage_threshold)` is TRUE for exactly
   the long prompts where sparse prefill pays off — so in that regime the host rotated-K is empty
   and router (b) would have to rotate `k_activations` (raw K, always populated at 2912) itself via
   the existing `apply_rope_neox_cpu_fast`, or read block min/max back from `persistent_k_cache`
   (GPU F16). Cheapest: maintain a host per-block [min_k,max_k] of rotated K incrementally as each
   chunk is ingested (rotate the new chunk's raw K, update the block stats), independent of the
   `decode_use_sparse` flag.

**Concrete port steps (mirrors the verified MLX `_sparse_prefill_attend`):**
- Add `DIFFKV_SPARSE_PREFILL` env (default 0) + KMIN/FRAC/WINDOW/SINK_BLOCKS/MIN knobs (same names).
- Host: maintain `blk_min[l][b][kv,d]`, `blk_max[l][b][kv,d]` of ROTATED K, updated per ingested
  chunk (aligned to block_size, absolute positions). Route once per chunk (uniform block set across
  layers to keep ONE mask + ONE gather layout): score = Quest min/max bound of chunk-mean-rotated-K
  (or pooled q if going 2-pass) vs each prior block; pick top-K = max(KMIN, ceil(FRAC·nb)); keep
  block 0 (sink) + the window `[cur_start-W, cur_start)` + current chunk, all fully attended.
- `build_prefill_ctx_graph`: when a selected-range list is passed, build `k_ctx`/`v_ctx` as
  `ggml_concat` of `ggml_view_3d` slices of `persistent_k_cache[l]` for each selected contiguous
  block range (+ window view + current chunk), instead of the single full view. The blocks are
  contiguous, so NO `ggml_get_rows` is needed — just views + concat.
- Mask: build `[Ksel, chunk_len]` on host — history rows (all selected/sink/window keys, abs pos <
  cur_start) fully visible (0); current-chunk rows causal. Exactly the MLX mask, reordered to the
  gather layout.
- Gate: only engage when `pos_start >= DIFFKV_SPARSE_PREFILL_MIN` AND there is ≥1 prunable block.
- Verify: `cd diffkv_native && cmake --build build -j4` (never while an MLX bench runs), then the
  6-cell NIAH (`diffkv_native/tests/test_niah_native.sh`) must stay 6/6 + `tools/run_conformance.sh`
  PASS + `native_margin_probe.sh` margins unchanged, with the flag ON. Leave default OFF until green.
- **Memory note:** like MLX Stage 1 this is COMPUTE-only — `persistent_k_cache` still holds full KV,
  so native prefill peak RAM is unchanged. The memory win is the same Stage-2 follow-up (FOLLOW-UP 1).

## FOLLOW-UP 5 — Native decode surfaces only ONE of multiple facts (multi-fact retrieval gap)
**Status:** characterized, NOT fixed — it's a DEEP native-decode gap, not a quick knob. **NOT a
sparse-prefill regression** (dense native == sparse native == 1/3). **Where:** native decode
retrieval path (`route_decode_slots` / decode-cache / compressed-decode attention in
`diffkv_native/src/main.cpp`), NOT prefill.

**Symptom:** on the 3-needle prompt (`diffkv_native/tests/make_multineedle_prompt.py`, passcodes at
depths 0.25/0.50/0.75, "list all three"), native emits `"The three secret passcode is
SIGMA-9923-BETA."` and stops — it KNOWS there are "three" but surfaces only the MIDDLE needle
(SIGMA @0.5) and never OMEGA (0.25) or THETA (0.75). **MLX on the SAME model gets all 3/3**
(`1. OMEGA 2. SIGMA 3. THETA`). So the 1.5B model is capable; the gap is 100% in native's decode.

**FULL DIAGNOSIS (this pass — thorough, every knob ruled out):**
- **Not generation.** Pure dense native (2k, DiffKV OFF via `DIFFKV_ENGAGE_THRESHOLD=8192`) gets
  multi-needle **3/3** ("The three secret passcodes are: …" + all 3 codes). The 1.5B model + native
  generation are fine — the gap is 100% in the DiffKV COMPRESSED-DECODE path at long ctx.
- **Degrades with context:** DiffKV-on gets 4k **2/3** (OMEGA+SIGMA), 8k **2/3** (SIGMA+THETA),
  16k **1/3** (SIGMA). The MIDDLE needle (0.5) always survives; edges drop as ctx grows → softmax
  DILUTION over more compressed keys, not a per-needle failure.
- **Not routing coverage.** `DIFFKV_MLX_PARITY=1` (attend ALL blocks, no top-k pruning) is still
  **1/3** at 16k. All 3 needle blocks are attended, yet only one surfaces.
- **Not compression fidelity.** `DIFFKV_RANK=32`, `RANK=48 MAX_RESIDUAL=256`, and
  `MLX_PARITY=1 + RANK=32 + MAX_RESIDUAL=256` combined are ALL still **1/3**. (Note native
  `NativeBlockPool::MAX_RESIDUAL` already defaults to 128 via `DIFFKV_MAX_RESIDUAL`, not 8.)
- **Not cache / interval / quant / sampler:** `DECODE_CACHE=0`, `INTERVAL=1/4`, native q4_k_m == q8,
  rep-penalty 1.15+loop-escalation — all **1/3**.
- **The decode-attention MATH is not buggy:** the CPU conformance harness passes **bit-exact
  (1.19e-7)** vs golden vectors. So native computes the intended DiffKV attention correctly; the
  gap is that native's compression POLICY / reconstruction / block+window COMPOSITION differs from
  MLX's such that a diffuse "list all three" query can't attend 3 specific blocks sharply enough.
- **MLX gets 3/3 at 16k on the SAME model** ("1. OMEGA 2. SIGMA 3. THETA"). MLX differences that
  likely matter: per-token min/max q·k routing + `DIFFKV_SPARSE_BIAS` (additive boost on compressed
  block keys so they compete with the exact dense window) — **native has NO sparse-bias equivalent**
  (grep found none). But a UNIFORM compressed boost wouldn't discriminate needle-vs-filler blocks,
  so SPARSE_BIAS alone is unlikely to be the whole fix.

**ACTIVE vs NATIVE — the concrete differences (read both decode paths + A/B, this pass):**
Native can retrieve EACH needle individually at 16k (single-needle depth 0.25 AND 0.75 both hit), so
reconstruction/per-position retrieval is fine. The gap is attention BREADTH on a diffuse "list all
three" query: native's first decode token commits SINGULAR ("...passcode **is** SIGMA") =
winner-take-all; MLX goes PLURAL ("...passcode**s are:**\n1. …\n2. …\n3. …") = distributed. The
structural differences that produce this (`compute_decode_attention_static` in mlx_diffkv_wrapper.py
vs the decode-cache flash path + `materialize_routed_kv` in main.cpp):

1. **Exact-residual placement — STRONGEST suspect.** MLX DROPS each exact-residual position from the
   compressed SVD scores (`res_mask` → −inf, line ~375) and represents those tokens ONLY in the
   **dense pool**, which is a SEPARATE, SMALL softmax (`scores_dense`, just window + routed residuals
   ~ hundreds of keys). So the 3 needles' exact tokens each get HIGH relative weight → all surface.
   NATIVE materializes residuals INTO the block buffer (`materialize_routed_kv`: anchor + U·V +
   residuals) and attends everything in ONE unified `ggml_flash_attn_ext` over [all routed blocks ++
   window ++ current] — thousands of keys — so the 3 needle tokens are DILUTED and the softmax picks
   one winner (SIGMA, the middle/strongest). This matches every symptom: single needle fine (wins
   dilution alone); 3 needles → 1 (must share); worse as ctx grows (more keys = more dilution).
2. **Attention structure.** MLX = TWO softmaxes (compressed pool, dense window) + flash LSE merge, so
   the exact-residual (dense) pool is normalized INDEPENDENTLY of the big compressed pool. Native =
   ONE softmax over the concatenation → no independent normalization for the exact tokens.
3. **Residual/KV precision.** MLX pool = fp16; native pool = **Q8_0** (`initialize(... kv_type=
   GGML_TYPE_Q8_0)`) → native's "exact" residuals are 8-bit, slightly less exact.
4. **Router signal.** MLX = per-token min/max **q·k** + a residual-token router that scores each block
   by its exact residual tokens' q·k (`_block_relevance_residual`). Native = semantic-descriptor
   search + lexical host_slots + anchor_screen (q·anchor). Different selection; not the sole cause
   (attend-all `MLX_PARITY=1` is still 1/3) but affects which blocks get materialized.

**RULED OUT as the cause:** SPARSE_BIAS (MLX is 3/3 even with `DIFFKV_SPARSE_BIAS=0`), routing
coverage (attend-all still 1/3), compression fidelity (rank 32/48 + residuals 256 no change), model
quant (q4==q8), sampler/rep-penalty, decode-cache on/off, re-route interval, per-needle
reconstruction (each individually retrievable at 16k), generation (pure dense native = 3/3).

**SHARPENED DIAGNOSIS (2nd deep pass — the residual-pool fix idea above is WRONG, disproven):**
- A two-path LSE merge is **mathematically identical** to one unified softmax over the same keys, so
  the MLX "separate dense pool" STRUCTURE cannot be the cause (confirmed: MLX is 3/3 even with
  `DIFFKV_SPARSE_BIAS=0`, i.e. its two-path ≡ a unified softmax). Do not "give native a separate
  residual pool" — it would be a no-op.
- **The gap is attention BREADTH, and breadth comes from the LOW-RANK RECONSTRUCTION of the needle
  SENTENCES, not the residuals.** Decisive test: MLX with `DIFFKV_MAX_RESIDUAL=8` still produces the
  correct PLURAL LIST format ("The three secret passcodes are:\n1.\n2.\n3.") but HALLUCINATED values
  (OMEGA-**1234**-ALPHA) — so residuals only fix the exact digits; the "there are 3 distinct facts"
  breadth signal comes from the low-rank recon. Native says SINGULAR ("passcode **is** SIGMA") → its
  low-rank recon conveys the distinct-facts signal WORSE than MLX's.
- **Not a knob:** ruled out this pass — pool precision (`DIFFKV_KV_QUANT=f16`), `DIFFKV_V_SCALE`
  (0/1/2), rank 32/48, residuals 256, attend-all, decode-cache, interval, quant, sampler. All 1/3.
- So the residual native-vs-MLX difference is in the **low-rank reconstruction VALUES themselves**
  (the SVD basis / U·VK·row_scale·blk_scale / POOL_ROT_ABS RoPE math produce K/V that carry less
  multi-fact structure than MLX's `comp_scale·(U@V)` recon), NOT any exposed parameter. This is the
  same native-vs-MLX compressed-decode fidelity frontier as HANDOFF_MLX_SYNTHESIS.md ("native
  compressed reads the paper 4-5/15 vs MLX better").

**FIX DIRECTION (deep, needs a decision — cross-engine value comparison):** dump native's
reconstructed needle-block K AND V (host `get_host_*` + the `DIFFKV_DBG_RECON_POS` machinery, extend
it to V) at 16k and diff against BOTH ground-truth exact K/V and MLX's reconstruction of the same
block; find where native's recon diverges (candidates: V basis vs K basis fidelity, blk/row scale,
randomized-SVD basis quality, RoPE-abs). This is a standalone diagnostic sub-project, not a knob.
MLX handles multi-fact today, so this only matters if native must match MLX on multi-fact.
Reproduce: `diffkv_native/tests/make_multineedle_prompt.py 16000 > p.txt` then run the binary.

## FOLLOW-UP 3 — Router uses a single MEAN-pooled query per chunk
**Where:** `_sparse_prefill_attend`, `q_rep = mx.mean(q_rot[0], axis=1)`. All L=512 query
positions in a chunk share ONE routed block set. That's standard (MInference pools query blocks)
and held recall here, but a chunk spanning a topic boundary could misroute the earlier half.
A cheaper-than-dense refinement: route per SUB-block of the query (e.g. 128-token query tiles)
if a future diffuse-query failure traces to this. Not needed for current gates.
