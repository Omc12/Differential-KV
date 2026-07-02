# Session Report — Fable 5 execution pass (2026-07-02)

**Scope of this session:** executed against `PLAN_FABLE5_OPTIMIZATION.md`. The session was
consumed almost entirely by **Tier 1 native correctness** — justified below: the plan's premise
("native is coherent ≤13k, fix routing for >13k") turned out to be **false at HEAD**, and the
true failure was deeper and more valuable to fix than the planned router port. Five root-caused
bugs were fixed and committed (`996ebb5`); the remaining gap was A/B-proven to be **shared with
MLX** (architectural), overturning the plan's central native-vs-MLX framing.

Baselines, probes, and every claim below were run/measured this session on the M3/8GB dev
machine with Qwen2.5-1.5B-Instruct-q4_k_m.

---

## 1. What the baseline actually showed (vs. what the plan assumed)

| Claim in plan/memories | Measured this session |
|---|---|
| "Native coherent ≈MLX ≤13k; gibberish only >13k" | **NIAH FAILED at 4k, 8k, and 16k** (depth 0.5, harness env): "OME 4 1 2 1 2…", "OMESTE-RE-TS-…", degenerate loops |
| "Fused paths blocked on per-row U scale port" (3.1) | **Stale** — the committed fused kernel already reads `U_scale[slot*S_max+t]` (per-row) |
| "MLX residual-key router port is the #1 native fix" (1.1) | **Wrong target** — native attends ALL blocks by default (`DIFFKV_MLX_PARITY` on) and a residual-key router already exists in the callback; routing was not the failure |
| "MLX exact 4–32k, native broken → port MLX behavior" | On the **identical prompt**, MLX also fails ("again_(�ECHADZD)") — see §4 |

Kernel-parity oracle (`test_diffkv_kernel_parity.py`): 4 passed before and after (MLX untouched).

## 2. Root causes found and fixed (commit `996ebb5`)

Diagnosis chain (each step measured, in order): CPU==Metal outputs byte-identical → not a kernel
bug. `DENSE_CMP` synthetic parity at |K|=3900 → kernel math exact at failing scale. Prefill K/V
export vs ggml persistent cache → data clean (V byte-identical, K within fp16 rounding). Step-0
logits sparse-vs-bypass **identical to 4 decimals**, step-1 collapsed → state transition broken.
`DBG_WINDOW` cursor trace → found the reset.

1. **Dense-window reset (the dominant bug).** An end-of-step re-derivation of the decode dense
   window from per-block ingest buffers ran on effectively **every step** (`rebuild_needed` is
   true every step in practice), (a) replacing the contiguous prefill-activation window with
   divergent block-buffer content — the exact divergence an earlier fix's comment warns about —
   and (b) dropping freshly generated tokens (their block ingest is async and hasn't landed at
   scan time). **The model could not attend its own generation history.** Disabled by default
   (`DIFFKV_LEGACY_WINDOW_REBUILD=1` restores). The window is now owned by: pre-loop init from
   prefill activations + per-step append + MLX-parity slide.
   *Verified:* all-dense-window control (recency 5000, 4k, depth 0.5) went from
   `OMEGAUSS-7890-ALPHA` (confabulated) → **`OMEGA-7741-DELTA` EXACT**; 16k depth-0.9 NIAH
   PASSES (baseline: word salad).

2. **RoPE scheme conflict (double rotation).** Ingest pre-rotated pool K at *within-block*
   offsets (Jun-21 scheme) while the default exact decode paths rotated the reconstruction
   *again* at absolute `token_positions` (Jun-22 scheme) — every compressed token over-rotated
   by its within-offset; the approximate path was separately off by `landmark_idx` (because
   `anchor_positions` stores `anchor_idx + landmark_idx`). All rotation decisions now flow
   through one `diffkv::pool_rot_mode()`; default **`POOL_ROT_ABS`**: pool K stored fully
   rotated at absolute position (exactly MLX's representation), decode applies **no rotation**
   to pool content. Sites fixed: `execute_cpu_attention` (has_rope override), Metal kernel
   `decode_attention_metal_kernel` (new `pool_prerotated` param; 8 pool rotation sites gated,
   dense sites untouched), callback residual-key router (anchor + residual scoring).
   **Bonus:** with an absolutely-rotated pool, the cheap *project-then-attend* score structure
   becomes mathematically exact (no rotation intervenes) — it is now the default
   (`approx=true` under ABS), removing per-token trig from decode. 4k run wall time dropped
   42s → 12s in the harness config (not a controlled TPS benchmark; includes shorter output).

3. **sparse→dense fallback gap.** `k_rotated_activations` is only filled when prefill starts in
   dense mode; the "0 compressed blocks" fallback then uploaded all-zero K to the dense decode
   graph. Now backfilled at the fallback point.

4. **V-side blindness of the joint K|V SVD** *(measured; conceptually shared with MLX — same
   math in `_compress_block`)*. Qwen layer-0 |K|≈1100 vs |V|≈5 → V contributes <1% of the joint
   SVD energy → per-token **V reconstruction error 24–73%** while K is ~1% (new
   `DIFFKV_DBG_RECON_POS` probe). Attention *routes* correctly (K fine) but *reads* garbage
   (V wrong) → exact-token recall fails for any token that didn't win a residual slot; residual
   rows are rescued (V ~1%), which is precisely why "OMEGA"/"DELTA" (captured) emitted
   correctly while "7741" (not captured) confabulated. Fixes: V half scaled to K's RMS before
   the SVD, inverse baked into stored VV (decode unchanged, `DIFFKV_V_SCALE=0` disables);
   residual ranking moved to the balanced space. Measured: non-residual V error 0.70→0.50,
   0.35→0.24 at the needle rows.

5. **Content-aware residual capture.** Absolute-error ranking cannot identify recall-critical
   tokens (V misreconstruction is ubiquitous — a needle digit ranks *below median*). Digits,
   single uppercase letters, `-` (the verbatim-identifier alphabet; note: the pre-existing
   landmark "digit boost" targeted ids 48–57, which are `Q`–`Z` in this vocab, not digits),
   plus chain neighbors, get a rank boost (`DIFFKV_RESIDUAL_TOKEN_BOOST`, default 8×).
   *Verified:* needle digit rows go from un-captured (V err 35–73%) to captured (V err 0.4%).
   A blanket byte-token boost was tried and **rejected** (floods the budget with prose
   punctuation; displaced the multi-char code pieces and broke "OMEGA").

Also fixed/flagged in passing: stale per-slot RoPE cache (never invalidated on slot reuse —
moot under ABS, noted for legacy paths); fused-path guard (`DIFFKV_NATIVE_ATTN=1` refuses under
ABS until the ggml fused kernels are ported — they still rotate pool content).

## 3. Native end state (this session's final sweep, harness env, depth 0.5/0.9)

Baseline (session start): **0/6** — all outputs degenerate ("OME 4 1 2 1 2 4…").

Final sweep (harness env, `DIFFKV_ENGAGE_THRESHOLD=1024`, greedy, 40 tokens):

| ctx | depth | result | output (truncated) |
|---|---|---|---|
| 4k  | 0.5 | FAIL | `OMEGA-999-0` (block found, digits confabulated) |
| 4k  | 0.9 | **PASS** | `The secret passcode is OMEGA-7741-DELTA.` |
| 8k  | 0.5 | FAIL | degenerate `OMESTESTE…` (worst remaining row) |
| 8k  | 0.9 | FAIL | coherent echo, no needle |
| 16k | 0.5 | FAIL | coherent echo, no needle |
| 16k | 0.9 | near-miss | `OMEGA-7741-DELUG…` — **digits exact from the compressed pool** (first time ever); `TA` fragment displaced by the capture boost |

Every row was degenerate word-salad at baseline; now 1 PASS, 1 near-miss with exact digits, and
coherent (if needle-less) English elsewhere. The 16k/0.9 row is direct evidence the capture
boost works for digits — and that capture is zero-sum (the boost displaced the `TA` row that
error-ranking had been capturing). An earlier intermediate build (window fix, before
v_gain+boost) had 16k/0.9 fully PASS, confirming the capture-policy knobs now dominate the
outcome and need the systematic two-prompt-family eval called for in §6.1.

## 4. The pivotal reframing: the remaining gap is SHARED with MLX

Running **MLX (`MLXDiffKVWrapper`, sparse decode) on the byte-identical prompt** that native
fails: MLX emits `The answer currently exists - again_(�ECHADZD)` — no needle, worse than
native's `OMEGA-999-0`. (Caveat: this prompt has literal `\n` text and raw chat markers — it is
the native harness's prompt format; MLX's own `--bench` harness passes at 4–32k on its
prose-filler prompt.)

**Interpretation with the §2.4 evidence:** rank-16 joint-SVD reconstruction *cannot carry V
content* (25–70% error even after rebalancing); both engines' exact recall is entirely a
function of whether the needle tokens won per-block residual slots. On prose filler
(bench_common) needles are error-outliers → captured → "exact recall". On digit-dense filler
(the native harness's AI-history text) they are not → both engines confabulate. **"Exact
recall at 4–32k" is prompt-dependent on both engines**; the residual-capture policy, not
routing or kernels, is the real accuracy frontier. This retroactively also explains the MLX
relational_ab findings (crammed digit tables: `max_residual` 64→128 was the dominant lever).

## 5. Evaluated and rejected / de-prioritized

- **Plan 1.1 (port MLX residual-key router)** — rejected as the correctness fix: native attends
  all blocks by default and already has a residual-key router in the callback; routing was not
  the failure at any tested context.
- **Plan 3.1's premise** — stale: per-row U scale is already read by the committed fused kernel.
  The real 3.1 blocker is now the POOL_ROT_ABS port of the two ggml fused kernels (guarded off).
- **Blanket byte-token residual boost** — measured regression (see §2.5).
- **`MAX_RESIDUAL` 64→128/192 alone** — no effect on recall (ranking, not budget, was binding).
- **Async prefill ingest / GPU F16 cast as corruption suspects** — exonerated by ablation
  (identical outputs with both disabled).

## 6. Prioritized next steps

1. **Residual-capture policy as a first-class research problem (both engines).** The evidence
   says this is where exact recall lives. Concrete directions, in order:
   (a) capture contiguous *runs* around high-information tokens (chain protection is already
   in, extend to run-level); (b) session-level rarity (IDF) instead of token-class heuristics;
   (c) evaluate on BOTH prompt families (prose filler *and* digit-dense filler) — single-family
   evals hid this for weeks; port `benchmarks/niah_recall.py --bench` prompts to the native
   harness.
2. **Port POOL_ROT_ABS into the fused ggml kernels** (`kernel_diffkv_attn_partial` + subgraph):
   under ABS they need *no pool rotation at all* — strictly simpler than the WIP exact-math port
   sitting uncommitted in the llama.cpp submodule (that WIP targets the old scheme; supersede
   it). Then `DIFFKV_SELFTEST` byte-parity → flip `DIFFKV_NATIVE_ATTN` default → the already-
   written flash-decode path gives the ~3.4× decode speedup measured on 2026-06-21.
3. **MLX ports of §2.4/§2.5** (V rebalancing + content-aware capture) behind flags; A/B with
   `niah_recall --bench`, `relational_ab --natural` and the adversarial crammed case (expected
   to improve: same root cause).
4. **Tier 2 MLX perf items (2.1 fused kernel, 2.2 batched SVD, 2.4 prompt-scaled pool)** — not
   reached this session. Design notes: for 2.2, batch the rSVD per (layer, chunk) by virtually
   concatenating [dense tail | chunk] and compressing all flushable blocks in one call
   (numpy stacked QR/SVD exists; same seeded Omega reproduces serial results); avoid the
   per-block `mx.clear_cache()`. For 2.1, `compute_decode_attention_static` is the clean
   reference; note MLX needs no rotation in-kernel (pool pre-rotated) — same simplification the
   native ABS port gets.
5. **main.cpp decode-loop hygiene**: `rebuild_needed` is effectively always-true (every step
   rebuilds the decode graph — also a perf tax); the step body lives inside the rebuild branch.
   Fixing the flapping (`step_use_sparse`'s `pool_version>0` condition + `pool_grew`) is both a
   perf win and removes a whole class of state-reset hazards.
6. **Engage-crossing edge case** (L < engage < L+gen): window init from freed `k_activations`
   → garbage window on mid-generation sparse flip. Rare; documented, unfixed.

## 7. Debug instrumentation added (all env-gated, zero default cost)

`DIFFKV_DBG_EXPORT_CHECK` (prefill export vs ggml cache), `DIFFKV_DBG_RECON_POS=<pos>`
(pool reconstruction fidelity per row, K and V), `DIFFKV_DBG_STEP_LOGITS` (top-5 per decode
step, for path A/B), `DIFFKV_DBG_WINDOW` (window cursor trace), `DIFFKV_DENSE_CMP_T` was
already present. These four probes were what cracked the diagnosis; keep them.
