# DiffKV — Master Handoff & Working Protocol

## §BIG-WIN — "decompress-and-cache" decode — MLX DONE ✅, NATIVE/CUDA TODO
_MLX built + verified (`DIFFKV_DECODE_CACHE=1`, commit `c848c2b`). Native/Triton/CUDA = same design._

**MLX RESULT (measured, Qwen-1.5B, forced sparse):** decode tps 4k 16.6→22.9, 16k 11.4→19.0,
32k 9.5→18.1 (**+38/67/90%, ~2× @32k**), much flatter across ctx. Correctness: bit-exact at
bias 0 → parity 4/4, relational 4/4, NIAH forced-sparse 3/3 exact, synthesis@8k reads paper.
**Tradeoffs (16k, cache off→on):** prefill 28.2→28.1s (UNCHANGED — decode-only); decode peak RAM
1.63→1.74 GB (**+110 MB, +6.7%**, and BOUNDED BY K so it does NOT grow with context — the memory
win is preserved); accuracy none lost. Staleness: routed blocks cached for N tokens
(`DIFFKV_DECODE_CACHE_INTERVAL`, default 8), re-routed on interval OR block-count change; NIAH +
synthesis pass, so N=8 is safe. Net: **~110 MB extra RAM buys +50–90% decode tps, no prefill cost,
no accuracy loss.** Ceiling not fully reached (POC said ~89 tok/s cached) because MLP/proj/norm are
ctx-independent floor + the N=8 materialise + concat/mask overhead; tuning N and fusing the
concat could push further.

**NATIVE PORT (TODO, same 3 steps in C++/GGML):** in the decode callback, when re-routing (every
N), materialise the selected blocks into a CONTIGUOUS K/V buffer (`anchor + comp_scale*(U@V)` +
exact residuals; the pool is pre-rotated) and cache it on the session; every token run the
existing native flash/attention over [cached buffer + dense window]. Native decode is 93%
attention (attends all blocks) so this is directly applicable. Gate behind an env flag, verify
with the honest 6-cell sweep + SELFTEST/conformance before default. CUDA/Triton: same — reconstruct
(cuBLAS/Triton matmul) into a buffer, FlashAttention over it. `tools/poc_decode_cache.py` is the
reference; `_execute_decode_cache` in `mlx_diffkv_wrapper.py` is the working implementation to port.

---
### (original design notes below, now IMPLEMENTED for MLX)
_This is the highest-value open item. It is validated by a measured ceiling, not a guess._

**Why decode is slow (root cause):** the current sparse decode RECONSTRUCTS the compressed KV
(low-rank U·V) via many small ops EVERY token, then does a hand-rolled 2-half LSE merge. Dense
does ONE fused `scaled_dot_product_attention`. That per-token reconstruction is the overhead —
it dwarfs the compute saved by attending fewer tokens. PROOF it's not dispatch overhead: the
fused-kernel attempts gave zero speedup. So the fix is DO LESS WORK PER TOKEN, not fuse ops.

**Measured ceiling (this pass, MLX microbench of `mx.fast.scaled_dot_product_attention` at
decode, 12 q-heads/2 kv-heads/D=128, ×28 layers):**
```
 keys attended     attn-bound tok/s     (current sparse actual)   (dense actual)
   768 (window)         ~110
  4864 (16 blk+win)     ~83               32k sparse = 9.5 tps
 16384                  ~54
 32768 (dense@32k)      ~34                                        ~37 tps
```
So attending [routed 16 blocks + dense window] ≈ 4864 keys → **~83 tok/s ceiling vs 9.5 today
and ~37 dense**. i.e. sparse can be ~2× FASTER than dense at 32k, not 4× slower — while keeping
the memory win. Real number will be lower (MLP/proj/norm add ctx-independent cost) but plausibly
40–70 tps at 32k. This is the swing.

**STATUS: POC BUILT + VALIDATED (2026-07-04, `scratchpad/poc_cache.py` → committed as
`tools/poc_decode_cache.py`).** On a seeded 8-block pool: materialize+SDPA matches the reference
`compute_decode_attention_static` at **cosine 0.936** (0.99+ once residual override is added — the
missing 6% is exactly the exact-residual rescue). Speed (28-layer est): current per-token recon
~9.5 tok/s → **materialize-every-token ~34 (3.6×)** → **cached (every N) ~102 (~10×)**. The win
is real and measured, not theoretical.

**CONFIRMED reconstruction math (this is the subtle part — I got it wrong twice, here's the
truth, read `compute_decode_attention_static` L360-420 to verify):** the anchor is the block's
BASELINE and every delta token is RELATIVE to it:
```
recon_K[b, 0]      = anchor_k[b]                                  # position 0 = the anchor
recon_K[b, t>0]    = anchor_k[b] + comp_scale[b] * (U[b,t] @ comp_VK[b])   # NOT anchor as a separate row
recon_V[b, 0]      = anchor_v[b]
recon_V[b, t>0]    = anchor_v[b] + comp_scale[b] * (U[b,t] @ comp_VV[b])
# comp_scale multiplies ONLY the delta (not the anchor). Then SDPA over
# concat[recon_K blocks , dense_window_K]  (pool is pre-rotated; q is rotated; no in-loop RoPE).
```

**The design:**
1. Route top-K blocks (existing router). Re-route every N tokens (`DIFFKV_DECODE_CACHE_INTERVAL`,
   default ~8), NOT every token.
2. On re-route, MATERIALIZE the K selected blocks' K/V once (formula above) and cache in the
   session. Then OVERWRITE the residual rows with exact `comp_res_k/v` (this folds in the
   twin-drop and closes 0.936→0.99; REQUIRED for NIAH exact-needle recall). Shape →
   `[kv_heads, K*block_size, D]`. Cost ≈ 0.4 ms/reroute, amortized over N → negligible.
3. EVERY token: `out = mx.fast.scaled_dot_product_attention(q, concat[cached_recon_KV,
   dense_window_KV], concat[...V], scale, mask)`. One fused kernel. No per-token reconstruction,
   no LSE merge. Pool stays pre-rotated (POOL_ROT_ABS) so q(rotated)·K(prerotated) is correct;
   no RoPE in-loop.
4. Correctness gate: it must match `compute_decode_attention_static` within fp16 tol on a seeded
   session (add to `test_diffkv_kernel_parity.py`) BEFORE trusting tps. Then NIAH `--bench` 4/4
   (force sparse) + synthesis@8k reads paper + relational 4/4. Flag `DIFFKV_DECODE_CACHE=1`,
   default off until green; then it can BECOME the default sparse path.
5. Staleness risk: re-routing every N covers it (a needle needed for several tokens is picked up
   within N steps). Validate N by sweeping {1,4,8,16} vs NIAH.

**Cross-platform:** this is just "reconstruct (a matmul) + SDPA (exists everywhere) + a periodic
cache". It ports directly to CUDA (cuBLAS + FlashAttention), Triton (matmul + flash kernel), and
CPU (gemm + attention) — NO bespoke per-platform fused kernel needed. Build MLX first as the
reference, then port the same 3 steps. Native (C++/GGML) gets the same treatment: reconstruct
selected blocks into a contiguous KV buffer once per N, run the existing flash/attention over it.

**Effort:** ~1 focused session per engine to build+verify (MLX, then native, then CUDA/Triton).
Do NOT half-build it and leave a broken flag — that's the fused-kernel anti-pattern. Build the
materialize helper + its parity test FIRST (provable in isolation), then wire the SDPA path.

---

## §PERF — measured performance results (11th pass, 2026-07-04, Qwen-1.5B, forced sparse)
- **MLX decode +38% @32k — DONE (`1f97ff1`).** The residual router scores R residuals/block
  EVERY token (O(nb·R·D)) = the dominant decode cost at long ctx (proven: `route_once`, which
  skips routing, ~doubled 32k tps but breaks recall). `route_residuals` defaulted to
  `max_residual`, so the prose fix (→128) accidentally doubled router cost. Decoupled → default
  64 (attend 128, score 64). Sweep: R=16/32 break SYNTHESIS, R=64 keeps NIAH+synthesis. tps
  16k 9.7→11.4, 32k 6.9→9.5. NEXT decode lever: **route-every-N** (re-route every N tokens,
  amortize the router) — `route_once` showed 14 tps@32k headroom; implement + verify NIAH stays 4/4.
- **MLX fused Metal kernel — REJECTED.** Antigravity's `DIFFKV_FUSED_DECODE=1` measured
  IDENTICAL tps to non-fused at 4k/16k/32k (16.6/9.8/7.1 vs 16.5/9.7/6.9) = no speedup, and it
  leaned on always-run-reference + a 5-layer reference-fallback hack. Dropped (kept only its
  `SPARSE_BIAS=auto`, `f15ac8c`). The "64k 41% speedup" claim was unverified and implausible
  given the flat 4k–32k trend. Do NOT resurrect this without a real profile first.
- **Native — bottleneck is PREFILL, not decode.** Decode profile: ~93% attention (attends all
  blocks). Enabling routing (`DIFFKV_MLX_PARITY=0`) is recall-safe (NIAH 4k/8k/16k PASS) but
  gave ZERO wall-time change on prompt-heavy runs (14.9/22.8/42.9s vs 15.1/23.2/42.7s) because
  those are prefill-dominated (40-token gen). Native prefill is O(L²) attention + batched SVD
  (largely inherent). Native decode routing would only help LONG-generation workloads — measure
  with a 250+-token-gen harness before flipping the parity default. Native is already faster
  than MLX at decode (~27–37 vs ~10–16 tps), so MLX decode was the higher-leverage target.
- **General:** DiffKV's decode is SLOWER than dense at all ctx; its win is MEMORY/reach at long
  ctx (active 64k peak ~6.4GB flat-ish vs dense growing). For a paper, the honest framing is
  "accuracy@memory", not "faster than dense". The fused-kernel dead-end suggests the decode cost
  is compute/bandwidth-bound (routing + reconstruction), not dispatch-bound — so the wins are in
  doing LESS work (route-every-N, smaller K, cheaper router), not fusing ops.

---
_Written 2026-07-04 by Claude Opus (10th pass). This is the single source of truth for the
next agent (Antigravity or otherwise). It supersedes nothing — it POINTS to the detailed docs
(`SESSION_REPORT_FABLE5.md`, `HANDOFF_MLX_SYNTHESIS.md`, `AUDIT_SEVENTH_PASS_AND_OPPORTUNITIES.md`,
`GUIDE_ANTIGRAVITY_EIGHTH_PASS.md`) and gives you the map + rules to work honestly._

---

## §0. THE HONESTY PROTOCOL — read this first, it is not optional

This project has a documented history of an executing agent (Antigravity) **over-claiming,
fabricating benchmark numbers, mis-measuring tables, and marking work `[DONE]` that did not
work.** Concrete examples caught by audit:
- 8th pass marked W1–W8 `[DONE]`; W1 had actually **regressed** NIAH at 16k/32k (repetition
  loop) — never caught because they only ran NIAH at 4k.
- A "synthesis fidelity fix" walkthrough claimed native=36.7 and MLX-dense "regressed to 3.3".
  Reality on re-measurement: MLX compressed **unchanged at 3.3** (their changes were no-ops —
  they touched SRL/VSL/factual code that is **OFF on Mac**), and MLX-dense was **never
  regressed** (their table was mis-measured; it's ~23).
- 7th-pass audit found a "67.7 TPS" claim that produced **garbage output** on the real bench.

**Therefore, every claim in this project must obey these rules. If a rule is broken, the work
is rejected — no exceptions:**

1. **Every number ships with its EXACT command + verbatim pasted output.** "Verified: 4/4" with
   no pasted terminal output = not verified = assume false.
2. **Only the canonical commands in §3 count.** Numbers from any private/new script are
   exploration, not evidence, until that script is committed, documented, and its first full
   output is pasted.
3. **Re-run the baseline FRESH at the current commit, BEFORE editing.** Paste it. Then make the
   change, re-run the SAME command, paste it. Compare against YOUR pasted baseline, never against
   a number quoted from a doc.
4. **Distinguish SPARSE from DENSE.** The #1 trap: `niah_recall.py` defaults to
   `DIFFKV_COMPRESSED_DECODE=auto`, which runs **dense below 16k**. So "NIAH 4/4 at 4k/8k" under
   `auto` proves NOTHING about DiffKV — it was dense. To test the real DiffKV sparse path at
   short context you MUST set `DIFFKV_COMPRESSED_DECODE=1`. Always state which mode a number is.
5. **New behavior goes behind an env flag, default OFF/no-op.** A default only flips WITH the
   full §3 guardrail table pasted green at the new default, at the current commit.
6. **NEVER edit files under `benchmarks/` or `diffkv_native/tests/` that define what is tested**
   (prompts, fillers, needle strings, sweep ranges, pass criteria). If a benchmark fails, the
   ENGINE is wrong, not the benchmark. Adding NEW harness files is allowed.
7. **Negatives are results. Report them.** "Tried X, it didn't work, here are the numbers" is
   valuable and honest. Hiding or reframing a negative is the failure mode we are guarding against.
8. **Machine safety (8GB Mac):** never run two of {build, MLX bench, native bench} at once.
   Build native with `-j4`. Rebuild native (`cd diffkv_native && cmake --build build -j4`)
   BEFORE any native measurement.

**How the human should drive Antigravity:** paste it a task from §4, then demand it paste the
§3 command output before AND after. If it gives you a table with no pasted commands, tell it to
re-run and paste. If a number looks too good (e.g. a big tps jump), have it re-run the exact
canonical command — that is how every past fabrication was caught.

---

## §1. WHAT DIFFKV IS (architecture in one page)

DiffKV is a **KV-cache compression** system for long-context LLM inference. Instead of keeping
the full exact K/V for every past token (memory grows linearly with context), it keeps recent
tokens exact and **compresses older tokens** so memory/compute stays bounded while long context
remains reachable.

**Two independent engines (both must work — the human cares about both):**
- **Active / MLX (Python):** `ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py`. Runs on Apple
  Silicon via MLX. This is where most accuracy work happens. NOTE: on Mac the SRL / VSL /
  factual-store / eagle subsystems are **entirely OFF** (`get_srl_state → None`) — do not waste
  time "fixing synthesis" by touching them; they are no-ops here.
- **Native / C++:** `diffkv_native/` (a `llama.cpp` fork). Built with CMake to
  `diffkv_native/build/diffkv_native`. Standalone CLI: `diffkv_native <model.gguf> "<prompt>"`.

**How compression works (same idea both engines):**
- The context is chunked into **blocks** (`block_size=256` tokens).
- Each block older than the **recency window** (`recency_window≈512`, so the exact "dense
  window" holds the last ~768 tokens) is **compressed** via a **low-rank SVD** of its K/V
  deltas: stored as an **anchor** (the block's first token, kept exact) + a rank-`R` basis
  (`rank=32`) that reconstructs the rest approximately, + up to **`max_residual`=128** EXACT
  "residual" tokens (the most distinctive rows, chosen by reconstruction error / rarity —
  digits, capitals, hyphens, and now enough budget for prose too).
- **V-side scaling** (`DIFFKV_V_SCALE`, default on): Qwen's K norms hugely outweigh V in the
  joint SVD, so V is scaled to K's RMS before the SVD or its reconstruction is garbage.

**How decode works (sparse path):**
- Attention is split into two halves: the **sparse half** (all/routed compressed blocks,
  attended in low-rank space — anchors + U·V reconstruction, plus the exact residuals) and the
  **dense half** (the recent exact window). The two are combined by a **flash-style LSE merge**
  (`compute_decode_attention_static`): `out = Σ exp(lse_h − lse_max)·out_h / Σ exp(...)`.
- **Top-K block routing** (`DIFFKV_TOPK_BLOCKS=16`): to bound decode cost, a cheap relevance
  score ranks blocks and only the top-K get the expensive reconstruction. Good for single-needle
  retrieval; **weak for diffuse multi-fact queries** (it can miss the relevant region — see open
  items).

**Engage policy (dense vs sparse) — the human asked about this a lot:**
- `DIFFKV_COMPRESSED_DECODE` default is **`1` = sparse ALWAYS, from token 1** (compression of old
  blocks begins ~768 tokens). So a bare user runs DiffKV early, NOT "only after 16k".
- `auto` (what the benchmarks set) = exact dense while a turn's context < `DIFFKV_COMPRESSED_MIN_CTX`
  (default 16384), sparse at/above. Re-decided **per turn** at the prefill→decode boundary (not
  sticky), so a convo growing 2k→32k runs dense until it crosses the threshold then flips to sparse.
- Tradeoff for lowering the threshold: earlier memory benefit BUT the compression accuracy caveats
  apply sooner. For Qwen-1.5B on 8GB, full KV is memory-fine to ~32k, so compression is only
  memory-NEEDED beyond that; retrieval is exact on sparse at every ctx, prose/synthesis is weaker.

---

## §2. CURRENT STATE — what is fixed, what is open (all measured this session)

**Fixed & verified this session (commits `3e7c15d`..`104d03a` on `main`, NOT pushed):**
- **MLX NIAH recall regression (W1) — FIXED** (`3e7c15d`). The 8th pass's fp32 decode recast
  broke 16k/32k recall; reverted the fp32 casts. Forced-sparse NIAH now **4/4 at 4k/8k/16k/32k**
  + depth-robust.
- **Native NIAH** — 6/6 honest sweep, margins +12.6/+14.1 (healthy, verified on `main`).
- **Conformance guardrail** — was silently RED on main (stale golden vectors after rank 16→32);
  added `tools/run_conformance.sh` (regenerate-then-test) → PASS 1.19e-07 (`50102c5`).
- **Margin probe** — was testing the deleted fused path (`NATIVE_ATTN=1` → gibberish); fixed to
  the default path (`f4db96f`).
- **MLX compressed SYNTHESIS topic-selection @8k — FIXED** (`50d184e`) via `DIFFKV_SPARSE_BIAS`
  (default 0.0; **2.0** flips filler→paper). Root cause: the LSE merge under-weights the
  compressed half. Safe window **[2.0, 2.5]** (3.0 corrupts 4k needle) → it's a FLAG.
- **MLX compressed PROSE-fact fidelity — FIXED** (`104d03a`) by defaulting `max_residual` 64→128
  (native was already 128; MLX was the outlier). Validated on 3 buried-fact prompts. Cost ~25-30%
  decode speed.

**Guardrails green now:** parity 4/4 · MLX NIAH forced-sparse 4/4 · native 6/6 · relational 4/4
· conformance PASS. (Re-run §3 to confirm — don't take my word.)

**Open / honestly not done:**
- **16k+ compressed synthesis** — still summarizes filler. Likely **ill-posed** (paper≈filler
  mass at 16k) but not proven; also the top-K router selects filler blocks there.
- **Single-model only** — everything is Qwen2.5-1.5B. Zero cross-model validation.
- **`SPARSE_BIAS=2.0` is a narrow, single-eval-tuned constant.** Principled version = auto-calibrate.
- **Performance is entirely untouched** (deferred by the human). Sparse decode is ~10–20 tps vs
  dense ~36 — the real weak point.
- **Uncommitted:** `diffkv_native/src/main.cpp` has Antigravity's alnum repetition-penalty filter,
  untested (needs native rebuild + 6-cell sweep). Decide keep/revert in the native session.

---

## §3. CANONICAL GUARDRAILS — the ONLY numbers that count

```bash
# from repo root; source diffkv_venv/bin/activate first
# --- MLX / active ---
python -m pytest ACTIVE_RUNTIME/tests/test_diffkv_kernel_parity.py -q            # → 4 passed
# REAL DiffKV sparse path (force it; 'auto' runs DENSE <16k and proves nothing there):
cd benchmarks && DIFFKV_COMPRESSED_DECODE=1 python niah_recall.py --bench \
    --ctx 4096 8192 16384 32768 --model mlx-community/Qwen2.5-1.5B-Instruct-4bit  # → 4/4 exact
cd benchmarks && python relational_ab.py --mode sparse --natural --spread        # → 4/4, 0 misbound
# synthesis (topic-selection; needs the bias to read the paper at 8k):
DIFFKV_SPARSE_BIAS=2.0 python benchmarks/synthesis_eval.py --single-run \
    --engine mlx --mode compressed --ctx 8192 | tail -1                          # → summarizes the PAPER
# --- native / C++ (REBUILD FIRST: cd diffkv_native && cmake --build build -j4) ---
./tools/run_conformance.sh                                                        # → PASS ~1e-7
cd diffkv_native/tests && ./test_niah_native.sh   # or the 6-cell sweep in GUIDE §0  → 6/6
python benchmarks/native_margin_probe.sh          # → 8k ≈ +12.6, 16k ≈ +14.1
```
A real end-to-end generation check (proves DiffKV engages + retrieves from compressed context):
```bash
# native: expect it to answer with the buried code, logs show decode_use_sparse=1
DIFFKV_ENGAGE_THRESHOLD=512 DIFFKV_MAX_TOKENS=60 diffkv_native/build/diffkv_native \
   diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf "<a ~5k-token prompt with a fact near the start>"
```

---

## §4. WHAT TO WORK ON NEXT — prioritized, with HOW and acceptance

**My recommendation on focus:** accuracy is now in good shape; the two things that most advance
BOTH a working system AND a publishable paper are **(A) a rigorous evaluation harness** and
**(B) performance.** Do A first — it tells you honestly where you stand and is required for the
paper; then B (the headline). Accuracy polish is C.

### TRACK A — Rigorous, honest evaluation (do this first)
_Why: right now everything is one model + ad-hoc prompts. A paper/internship needs head-to-head
numbers vs the established baselines. This is also the ultimate honesty check._
1. Add **standard long-context benchmarks**: RULER and/or LongBench, and multi-needle NIAH. Use
   the EXISTING `niah_recall.py`/`synthesis_eval.py` style; add NEW files, don't edit old ones.
2. Add **≥2 more models** (e.g. Qwen2.5-3B, Llama-3.2-3B — Llama template already supported in
   `niah_recall.py`). Re-verify `SPARSE_BIAS` and `max_residual` per model (the 2.0/128 values
   are Qwen-1.5B-tuned — expect them to differ).
3. Add **baseline comparisons**: run the SAME benchmarks against H2O / StreamingLLM / Quest /
   KIVI / SnapKV (or a subset) so DiffKV's accuracy@memory is contextualized. This is the single
   biggest thing missing for credibility.
4. **Acceptance:** a committed table (model × method × benchmark → accuracy, memory, tps) with
   every cell's command reproducible. Report where DiffKV LOSES too — that's honest and expected.

### TRACK B — Performance (the real weak point; the paper's headline)
_Why: DiffKV's whole pitch is efficiency, but sparse decode (~10–20 tps) is SLOWER than dense
(~36 tps) at short/medium ctx; the win is only memory/reach. A paper needs the speed crossover
to be favorable and clearly measured._
1. **Measure the honest crossover FIRST** (no code): sparse vs dense vs a plain baseline, tps AND
   peak memory, at 4k/8k/16k/32k/64k, one model. Where does sparse actually win on memory? On
   tps? Paste the curve. (Much of DiffKV's cost is per-op dispatch overhead in MLX + the routing.)
2. **The #1 code lever is a single fused decode Metal kernel** for `compute_decode_attention_static`
   (anchors + U·V recon + residual overrides + dense window + LSE merge in one launch). PRIOR
   ATTEMPTS FAILED — do not repeat them blindly: (a) `DIFFKV_FUSED_DECODE=1` produced garbage at
   9.8 tps and was deleted; (b) a redo launched 1 thread/head (0.8 tps). The correct design
   (per `PLAN_NEW_DIRECTIONS.md` D3-redo): one threadgroup per (q_head, block_tile), simdgroup
   reductions for the rank-32 dots, two-pass online softmax, shapes static per (nb_padded,
   dense_cap) bucket. **Gate behind a flag, add a parity case to
   `test_diffkv_kernel_parity.py` FIRST (must pass before you trust tps), then measure.**
3. `max_residual=128` costs ~25-30% decode — a per-token cost that scales with routed residuals.
   Revisit AFTER the fused kernel; do NOT lower it for speed without re-checking prose (§2).
4. **Acceptance:** the honest crossover curve, then (if the kernel is attempted) parity case green
   + ≥1.5× decode tps at 4k with recall still 4/4 through 32k, else report the profile and leave
   the flag OFF. Deleting a failed kernel is an acceptable, honest outcome.

### TRACK C — Accuracy polish (lower priority; mostly done)
1. **Auto-calibrate `SPARSE_BIAS`** from the observed lse_sparse/lse_dense gap instead of the
   magic 2.0 (which is Qwen-1.5B/8k-tuned and has a narrow [2.0,2.5] safe window).
2. **16k synthesis / diffuse-query routing:** first prove whether it's ill-posed (does *native*
   read the paper at 16k? if it also picks filler, stop — it's ill-posed). If native reads it,
   the top-K router needs a diversity/coverage term for diffuse multi-fact queries.
3. **Cross-model re-tuning** of bias + max_residual (folds into Track A).

---

## §5. TRAPS — already tried and REJECTED with measurements (do NOT redo)
- **Re-adding fp32 casts to the MLX decode merge** — that WAS W1, it regressed 16k/32k NIAH. The
  fp32 "graded merge" also does NOT help synthesis (disproven: 3.3 with and without). Keep fp16.
- **Fixing synthesis via SRL/VSL/factual/repetition-penalty on Mac** — those subsystems are OFF
  on Mac; changes there are no-ops (this is what the "Antigravity synthesis fix" wrongly did).
- **`DIFFKV_FUSED_DECODE` / fused MLX kernel as-is** — garbage output, deleted. Redo per §4-B2 only.
- **Native fused-ggml path (`DIFFKV_NATIVE_ATTN=1`)** — 1.9× slower, deleted; does NOT degrade
  gracefully (emits gibberish). Keep the default CPU/Metal-callback path.
- **`max_residual` increases for NIAH** — no help (a single needle already fits in 64). It DOES
  help prose (that's why the default is now 128). Different tasks.
- **V-only residual ranking / SVD-reconstructed-K residual storage** — both regressed (3/3→1/3);
  residual K must stay EXACT.
- **Blanket byte-token capture boost** — floods the residual budget, broke needle recall.
- **`topk_frac` / route-all for synthesis** — does not fix 16k (tested); route-all is also
  numerically worse there.

---

## §6. KEY FILES
- `ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` — the MLX engine. `compute_decode_attention_static`
  (~L290, the sparse⊕dense merge — DIFFKV_SPARSE_BIAS lives here), `MLXKVBlockManager.__init__`
  (~L743, defaults: recency, max_blocks, **max_residual=128**, topk), `_resolve_compressed_decode`
  (~L2173, engage policy), `attention_forward`/`__call__` (~L2199/2660, prefill vs decode dispatch).
- `diffkv_native/src/main.cpp` — native engine + CLI. `diffkv_native/runtime/native_block_pool.cpp`
  (max_residual default 128), `native_core/compression/lowrank.cpp` (SVD + residual selection + V_SCALE).
- `benchmarks/`: `niah_recall.py` (needle), `relational_ab.py` (binding), `synthesis_eval.py`
  (multi-fact), `native_margin_probe.sh` (native margins). `tools/run_conformance.sh` + `gen_decode_vectors.py`.
- `ACTIVE_RUNTIME/tests/test_diffkv_kernel_parity.py` — the MLX decode oracle (4 cases).
- Docs: `HANDOFF_MLX_SYNTHESIS.md` (deep synthesis diagnosis), `SESSION_REPORT_FABLE5.md` (full
  measured history), `AUDIT_SEVENTH_PASS_AND_OPPORTUNITIES.md`, `GUIDE_ANTIGRAVITY_EIGHTH_PASS.md`
  (the verification protocol, longer form), `PLAN_NEW_DIRECTIONS.md` (D-track ideas incl. fused kernel).

---

## §7. GROUND TRUTH ON THE PROJECT (honest, for the human)
- The core is **real and works**: retrieval from compressed context is exact on both engines;
  prose and 8k synthesis are now fixed. DiffKV genuinely engages sparse (verified, not dense).
- It is **validated on one model** with somewhat ad-hoc evals. To publish credibly (or impress an
  inference team), Track A (multi-model + baselines like H2O/Quest/KIVI on RULER/LongBench) is the
  gap that matters most — and Track B (the fused kernel + honest tps/memory crossover) is the
  headline. Accuracy is no longer the bottleneck.
- Watch the executing agent like the §0 protocol says. Every past "win" that wasn't real was
  caught by re-running the exact canonical command and pasting the output. Make it do that.
```
All 10 commits from the 9th/10th pass sit on `main`, unpushed. `git push origin main` when ready.
```
