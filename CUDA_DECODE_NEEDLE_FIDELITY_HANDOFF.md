# CUDA DiffKV — Deep-Needle Decode Fidelity Bug — HANDOFF

**Date:** 2026-07-20 · **Status:** root-caused to a real CUDA decode-fidelity bug, NOT yet fixed.
**Owner picking this up:** you have the A100; I (assistant) can't run CUDA, only MLX locally.

---

## 1. TL;DR — the bug

On CUDA, the DiffKV wrapper **corrupts a needle's digits at decode** for certain needle placements, while plain dense HF retrieves them perfectly.

Reproduce (transformers **must** be 4.46.3 — see §8):

```bash
python colab/diffkv_isolate.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --ctxs 8000 --depth 0.1 --builder niah
```

Observed:
```
DENSE  greedy = '847291'   ✓   (dense retrieves the needle)
DiffKV greedy = '888888'   ✗   (repeats the first digit; N_sparse=16, default cfg)
prefill first-token: SAME (both '8')   ← prefill is correct; DECODE is wrong
```

The full smoke test that surfaces it:
```bash
DIFFKV_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m pytest ACTIVE_RUNTIME/tests/test_niah.py -s
# 4000 all depths PASS; 8000 all depths FAIL (garbage digits: 888888 / 89019 / 890...)
```

---

## 2. What is CONFIRMED (do not re-litigate)

1. **It's a real DiffKV bug, not the eval / not the prompt / not the model.**
   `--builder niah` runs the *exact* `test_niah.py` prompt through **dense** and DiffKV on the
   same process: dense → `847291`, DiffKV → `888888`. Dense retrieving proves the prompt is
   answerable and the 0.5B model can do it.
2. **Prefill is correct; decode is the problem.** The prefill last-position logit (first generated
   token) is `8` for BOTH dense and DiffKV. The degeneration happens in the greedy *continuation*
   (decode), which repeats the first digit instead of emitting `47291`.
3. **It is placement-specific.** `colab/diffkv_isolate.py`'s own builder (`make_prompt`,
   `--builder isolate`) at 8000/0.1 **retrieves `847291`** on CUDA. The `test_niah.py` builder
   (`make_niah_prompt`, `--builder niah`) at the same 8000/0.1 **fails**. The two prompts differ by
   ~200 tokens (plen 7530 vs 7739), which shifts the needle's absolute token position → different
   block assignment. 4000 passes for both (fewer blocks); 8000 fails for `niah`.
4. **MLX (the reference) handles both builders.** Ran locally on this Mac
   (`diffkv_venv/bin/python3`, mlx 0.31.2, `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, the exact
   `make_niah_prompt`): retrieves `847291` at 8000 × {0.1, 0.5, 0.9}. Probe scripts in the session
   scratchpad: `mlx_niah_probe.py`. So the CUDA port diverges from MLX here.
5. **The needle block IS flagged for exact storage.** Ingest logs show
   `Block anchor_idx=774 ... Exempted from SVD compression (contains digit/number)` (774 ≈ the
   depth-0.1 needle position). So in principle the digits should be stored losslessly — which makes
   the decode corruption suspicious (see §4).

## 3. Dead ends this session (RULED OUT — don't repeat these)

- **Routing coverage / top-K.** `DIFFKV_TOPK_BLOCKS=64` and `=0` (attend ALL blocks, nothing
  dropped) BOTH still fail the `niah` prompt. So it is NOT a dropped-block / routing-selection
  problem. (It *looked* like routing because the SRL-forced path also failed — see next.)
- **The SRL router.** `test_niah.py` used to force `DIFFKV_SRL_THRESHOLD=5`, which built the SRL
  index and hit a **separate** int16-index crash (now fixed, §7). That was a red herring layered on
  top. Default router is `residual`; the test now pins it.
- **RoPE-in-router hypothesis.** Hypothesized CUDA's router RoPE-rotates keys while MLX scores raw
  → WRONG: MLX captures keys post-RoPE (`mlx_diffkv_wrapper.py:4448`), so both routers are
  position-aware and equivalent. `DIFFKV_ROUTER_ROPE=0` (content-only routing) exists as an A/B knob
  but does NOT robustly fix this (helped one placement, not the `niah` one).
- **Prompt/model too hard.** Ruled out by dense retrieving (fact #1).

## 4. PRIME SUSPECTS (ranked) — where to look

The signature — *prefill correct, decode repeats the first digit, placement-dependent, block is
"digit-exempt"* — points at the **decode-time reconstruction of the needle block's exact rows**.

### Suspect A (most likely): digit-exempt block isn't served exactly at DECODE
- Exemption is decided at ingest: `kv_runtime_manager.py:2919-2920` (`any(c.isdigit() for c in
  block_text)`) and the streaming path `native_core/streaming_sparse_ingest.py:53-91`
  (`_RE_LONG_DIGITS = \d{5,}` etc.).
- But **decode reconstructs blocks from low-rank U + residuals**, not from an "exact" store:
  `diffkv_attention.py:399` `reconstruct_batch_U(pool, ...)`, plus residuals at
  `diffkv_attention.py:408-409` (`residual_K_positions`, `residual_K_values`). The combined Triton
  decode path is `[DiffKV] Triton fused-decode COMBINED path ACTIVE`.
- **Check:** does an *exempt* block actually carry all its token rows as exact residuals into the
  pool, and does the decode kernel USE them? If the exempt block still decodes via rank-32 SVD
  (which "cannot reproduce digit sequences reliably", see the note at `kv_runtime_manager.py:66`),
  the digits after the first are lost → `888888`.

### Suspect B: residual capacity / selection drops the digit tokens
- Per-block residual count is chosen at compression: `kv_runtime_manager.py:3246-3255`
  (`n_max_residual = int(n*0.15)`, capped `min(8|16, …)`), pool cap `max_residual=64`
  (`kv_runtime_manager.py:684`, `native_block_pool.py:215`).
- Memory precedent: the native-side version of this exact symptom was
  **residual capacity + int8 residuals** (`memory/project_native_needle_recall_rootcause.md`:
  "rank-16 SVD ~43% floor + residuals capped 8 not int8 → fixed per-row int8 + MAX_RESIDUAL 8→40";
  and `project_active_compressed_decode.md`: "max_residual 32→64 exact recall").
- **Check:** for the failing placement, are all of `4 7 2 9 1`'s token rows selected as residuals,
  or does the 0.15/cap selection miss some? Try bumping `DIFFKV_MAX_RESIDUAL` / rank and see if the
  `niah` case starts retrieving (quick A/B before deep instrumentation).

### Suspect C: needle digits straddle a block boundary
- `block_size = 64` (`kv_runtime_manager.py:493`; note the wrapper's config 256 is IGNORED —
  `hf_diffkv_wrapper.py:795`). The `niah` layout may split `847291` across block N / N+1 so no single
  block is fully exact, or the boundary token is mis-handled. This explains the ~200-token placement
  sensitivity directly.
- **Check:** log the token index of each needle digit and the block boundaries
  (`anchor_idx`, `block_size`) for both builders; see if `niah` splits the digit run and `isolate`
  doesn't.

### Suspect D: int16 residual position overflow — NOT at 8k, watch at ≥32k
- `residual_K_positions` is `int16` (`native_block_pool.py:215`), max 32767. Fine at 8k (pos<7739)
  but **overflows at ≥32k** → wrong RoPE positions on residual keys. Not this bug, but a latent one
  for the paper's long-context regime; fix when you touch the pool dtype.

## 5. Concrete debugging plan (A100)

1. **Cheap A/B first** (env knobs, no code): on the failing repro, try
   `DIFFKV_MAX_RESIDUAL=128`, then higher `rank` (e.g. config rank 48/64), then
   `DIFFKV_ROUTER_ROPE=0`. If any makes `--builder niah` retrieve → you've localized to Suspect B.
2. **Instrument the needle block at decode.** In `diffkv_attention.py` around the combined decode
   path (residual gather `:408-409`, `reconstruct_batch_U` `:399`), for the block whose
   `anchor_idx` covers the needle, dump: (a) the reconstructed K/V rows for the digit token
   positions, (b) which residual positions were stored, (c) whether the exempt flag routed it to an
   exact path or the SVD path. Compare the reconstructed digit-token key vectors to the RAW captured
   keys.
3. **MLX ground truth.** I can produce, locally, the exact per-token reconstructed K/V MLX uses for
   this block/prompt (MLX retrieves), so you can diff CUDA's reconstruction against it numerically
   instead of guessing. Ask and I'll build the MLX trace dump.
4. **Fix, then re-verify** with BOTH builders at 8000 × {0.1,0.5,0.9} and 4000, AND run the full
   `test_niah.py`. Watch for regressions in the other passing cells.

## 6. MLX reference (ground truth, runs on the Mac)

```bash
/Users/omchimurkar1/Desktop/Differential-KV/diffkv_venv/bin/python3 <scratchpad>/mlx_niah_probe.py
# → [MLX ctx=8000 depth=0.1] FOUND ✓ gen='847291'  (and 0.5, 0.9)
```
MLX uses `block_size=256`, rank_adaptive→48 (`mlx_diffkv_wrapper.py`). It captures keys post-RoPE
and its `_block_relevance_residual` (`:1131`) is the router MLX ports to CUDA `route_blocks_relevance`
(`native_core/srl/query_router.py:872`).

## 7. Fixes ALREADY LANDED this session (in place; keep them)

All committed to `origin/main` (auto-commit) or in the working tree:
- **Decode-cache tuple-clobber crash** — `diffkv_attention.py` ~line 1905: cache-miss branch wrote a
  bare tuple over the per-layer dict from `setdefault(...,{})`; fixed to
  `_dpc_dict[captured_layer_idx] = (...)`. (Commit `05b5774`.) This unblocked ALL engaged CUDA decode.
- **Bypass multi-chunk `is_causal` bug** — `diffkv_attention.py` bypass branch (~line 311): non-square
  SDPA `is_causal=True` used upper-left alignment, dropping chunk-2+ attention; fixed with an explicit
  lower-right causal mask. GPU-validated ctx 1500/4000.
- **int16-as-index crash in the router** — `query_router.py` `route_blocks_relevance`: `res_p` /
  `anc_pos` are int16, which PyTorch rejects as indices; added `.long()`. (This crash was ACCIDENTALLY
  protective — it fell back to attend-all; fixing it exposed the routing/fidelity issues.)
- **`DIFFKV_ROUTER_ROPE`** flag added (`query_router.py`, default 1 = unchanged) — A/B knob only.
- **`test_niah.py`** — pins `DIFFKV_ROUTER=residual`, drops the stale `DIFFKV_SRL_THRESHOLD=5`.
- **`colab/diffkv_isolate.py`** — added `--depth` and `--builder {isolate,niah}`; dense-vs-DiffKV
  head-to-head; 12-token greedy.

Engaged single/chunked prefill + decode is **bit-correct vs dense at 8k/16k** for the `isolate`
builder — the ONLY open correctness gap is this placement-specific decode-fidelity bug.

## 8. Environment gotchas

- **transformers MUST be 4.46.3.** Env resets on the box have re-installed 5.14.1, which reorders the
  attention forward signature; the guard at `diffkv_attention.py:226` correctly refuses to run (would
  otherwise produce token salad). `pip install "transformers==4.46.3"` after any rebuild.
- `git pull` on the A100 box before every run; the code is authored on the Mac and pushed.
- Pre-flight sanity that a fix landed: `grep -n "<your marker>" ACTIVE_RUNTIME/...` before spending a
  GPU run — the box has run stale code twice this session.

## 9. One-command repros

```bash
# The bug, with dense baseline (decisive):
python colab/diffkv_isolate.py --model Qwen/Qwen2.5-0.5B-Instruct --ctxs 8000 --depth 0.1 --builder niah
# Control that PASSES on CUDA (same ctx/depth, other builder):
python colab/diffkv_isolate.py --model Qwen/Qwen2.5-0.5B-Instruct --ctxs 8000 --depth 0.1 --builder isolate
# Full smoke test (4000 pass, 8000 fail):
DIFFKV_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m pytest ACTIVE_RUNTIME/tests/test_niah.py -s
```

**Reminder:** this is a 0.5B *synthetic* smoke test. Before sinking many A100 rounds into it, it is
worth one run of the real benchmark (`colab/run_a100_paper_experiments.py`, RULER/real docs) to see
whether this digit-corruption actually moves the reported metrics — if the real eval is clean, this
is a tracked edge case, not a paper blocker.
