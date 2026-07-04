# HANDOFF — MLX compressed-synthesis bug (2026-07-04, Opus 4.8)

Read this first in the new session, then `SESSION_REPORT_FABLE5.md` and
`memory/project_w1_lse_merge_regression_fixed.md`.

## STATUS UPDATE (10th pass): root cause FOUND and 8k FIXED.
The MLX compressed-synthesis bug is **root-caused and fixed for 8k** via a new flag.
- **Root cause:** the sparse⊕dense flash merge **under-weights the compressed (sparse) half**
  because the low-rank reconstruction systematically UNDER-scores it. The sparse half attends
  the correct paper tokens but LOSES the merge to the exact recent dense window (filler) —
  proven with `DIFFKV_DBG_FORCE_HALF` (sparse-only → paper words; dense-only → filler) and the
  LSE-share probe.
- **Fix:** `DIFFKV_SPARSE_BIAS` (nats added to `lse_sparse`; default 0.0 = exact/parity-safe).
  `DIFFKV_SPARSE_BIAS=2.0` flips 8k compressed synthesis from the filler summary to the PAPER
  summary, with **NIAH 4/4 (bench) + 3/3 (16k depths) and relational 4/4 and parity 4/4 all
  still green** (NIAH's exact needle residual has a big enough margin to survive the bias).
  `+4.0` DOES break NIAH → the safe window is narrow, so it ships as a flag, not a default.
- **STILL OPEN — 16k synthesis:** at 16k the paper is only half the context and the **top-K
  router selects filler blocks, not paper** (forced-sparse@16k is empty/degenerate → the paper
  isn't even in the sparse half). The bias can't help there. This is a **routing** problem for
  diffuse multi-fact queries (the residual router is tuned for single-needle retrieval), and is
  the next work item (§3).

---

## 0. Guardrail state (all GREEN on `main` @ `d065112`, verified this session)

| Guardrail | Value |
|---|---|
| MLX parity | 4/4 |
| MLX NIAH `--bench` 4k/8k/16k/32k | **4/4 exact** |
| MLX relational | 4/4, 0 misbound |
| Native honest NIAH sweep | 6/6 |
| Native margins (8k/16k) | +12.6 / +14.1 |
| Native conformance (`tools/run_conformance.sh`) | PASS 1.19e-07 |

These held **with Antigravity's uncommitted changes applied** — they did not regress the
guardrails. Re-verify them first thing (they are the safety net for any synthesis fix).

## 1. Verdict on the Antigravity "Multi-Fact Synthesis Fidelity Fix" walkthrough

**The walkthrough's claims are mostly wrong; the fix did NOT work.** Measured directly this
session (`synthesis_eval.py --single-run --engine mlx --mode <m> --ctx 8192`):

| Cell | Walkthrough claim | ACTUAL (measured) |
|---|---|---|
| MLX compressed | 3.3 "clean summary" | **3.3, 1/15 — still summarizes the FILLER (Pride & Prejudice), not the paper. UNCHANGED.** |
| MLX dense | 3.3 (regression!) | **NOT regressed — 4/15, correctly summarizes the paper (~23). The walkthrough table was mis-measured.** |
| Native compressed | 36.7 | Plausible — native compressed genuinely reads the paper (4–5/15). |

Why the Antigravity MLX changes are no-ops for this bug: they target the **SRL / VSL /
factual-store / repetition-penalty** machinery, which is **entirely OFF on Mac/MLX**
(`get_srl_state → None`, factual_store None). So the 401-line wrapper diff cannot affect the
MLX compressed-decode output — and indeed it doesn't (byte-identical filler summary).

**Uncommitted changes in the working tree** (Antigravity's, on top of `d065112`):
`ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` (~401 lines, SRL/VSL/rep-penalty — no-op for
this bug on Mac), `ACTIVE_RUNTIME/native_core/kv_runtime_manager.py` (query-window cap),
`diffkv_native/src/main.cpp` (~8 lines, alphanumeric filter in the CLI repetition penalty —
the only plausibly-useful one; would need a native rebuild + 6-cell sweep to accept).
**Recommendation:** revert the MLX wrapper + kv_runtime_manager changes (no measured benefit,
adds risk/complexity to the file the real fix must touch); optionally keep/verify the native
`main.cpp` alnum filter separately. They were NOT committed for this reason.

## 2. The real bug — ROCK-SOLID diagnosis

**MLX compressed decode summarizes the recent dense-window filler instead of the compressed
paper. This is an MLX-compressed-decode bug, NOT a fundamental compression limit, NOT
reconstruction fidelity, and NOT the SRL/factual path.**

Failure mode (8k, paper=8030 tok fills blocks 0–30, ~118 filler tok + instruction in the
dense window):
- **MLX compressed** → "This eBook, titled Pride and Prejudice…" (1/15 facts).
- **MLX dense** (same 4-bit model) → "This paper presents a novel approach to accelerating
  kernel machines…" (4/15). ← proves the model CAN read the paper.
- **Native compressed** (same DiffKV algorithm) → "Random Fourier Features… Rahimi and Recht…"
  (4–5/15). ← proves the COMPRESSION can surface the paper.

### Hypotheses RULED OUT this session (do NOT re-test these — evidence in parens):
- **Reconstruction fidelity / rank** — rank 16, 32, 64 give the identical filler summary.
- **Top-K routing** — TOPK_BLOCKS 8, 16, 28 all give filler; route-all (999) gives 0.0 (worse).
- **V-side reconstruction / V_SCALE** — present in BOTH MLX paths (lines ~1410, ~1702);
  native reads the paper even with `DIFFKV_V_SCALE=0`. Not the differentiator.
- **Dense-window size** — MLX and native both use recency_window=512 → max_dense_len=768.
- **Model quantization** — full-precision `Qwen/Qwen2.5-1.5B-Instruct` (fp16) MLX compressed
  STILL summarizes filler. Not a 4-bit artifact.
- **Score dampening (gross)** — at layer 0, lse_sparse≈lse_dense (98 vs 99); roughly calibrated.

### What IS true (key evidence, `DIFFKV_DBG_LSE_SHARE=1`):
The MLX sparse (compressed) half **does attend the correct paper tokens** — top-scored rows
include "Rahimi" (`imi`), "Abstract", "convergence", "randomized", "performance". And the
sparse half wins the LSE merge at many mid-layer steps (share up to 0.94 at layer 20). **Yet
the generated output is still filler.** So the paper signal is present in the attention but
does not translate into the output. The remaining suspects (for the next session):
1. **out_sparse VALUE quality vs the exact dense values** — the sparse half attends the right
   paper K, but its low-rank-reconstructed V may be a blurred average that the LM can't use,
   so it falls back to the sharp exact dense-window values (filler). Note route-all→0.0
   (more compressed blocks = WORSE) is consistent with out_sparse contributing noise.
2. **The MLX sparse⊕dense LSE merge vs native's unified attention** — native attends the
   whole pool; MLX does two softmaxes + an LSE merge. Compare the two decode formulations
   directly and check the merge is combining the halves the way native's single softmax does.
3. **The MLX `use_topk=False` (all-blocks) path is separately broken** (0.0 vs native's 36.7
   on the same "attend everything" intent) — a good isolated target: fix that path to match
   native, then compare.

## 3. Remaining work for the next session — 16k synthesis (likely ill-posed)
The 8k case is fixed (`DIFFKV_SPARSE_BIAS=2.0`). **16k is harder AND may be ill-posed:** at
16k the paper is 8k tokens and the filler is ALSO ~8k tokens, so "summarize the text above" is
genuinely ambiguous (both regions ARE "the text above"). Tested this session: `SPARSE_BIAS=2.0`
with `TOPK_FRAC=0.5` and `1.0` (route half / all blocks) — both still summarize the filler at
16k. So growing K / biasing the merge cannot disambiguate two equal-mass competing regions.
Before investing here, first check whether **native** even reads the paper at 16k (if native
also picks the filler, 16k is ill-posed and not worth chasing). If native does read it at 16k,
then it's still an MLX gap and the router path below applies.
Router notes (for the still-open case, if pursued):
1. Instrument the router (`_block_relevance_residual`) at a 16k synthesis step: which block
   ids get selected, and where do the paper blocks (0–31 of ~63) rank? Expect: filler blocks
   win because the residual router scores single-token q·k peaks, which favour distinctive
   filler prose over diffuse technical paper text.
2. Design a routing signal that surfaces the answer region for a DIFFUSE multi-fact query
   (not a single needle). Candidates: (a) union the top-K with a spread of early/low-index
   blocks; (b) a coverage/diversity term so selection isn't collapsed onto one region;
   (c) `topk_frac>0` so K grows with block count at 16k+. All are experiments — measure.
3. Consider auto-calibrating `DIFFKV_SPARSE_BIAS` from the observed lse_sparse/lse_dense gap
   instead of a magic constant (the principled version of the 8k fix).
4. Every change gated by the FULL §0 table AND `niah_recall.py --bench 16k 32k` AND
   `relational_ab.py` — a synthesis/routing change must not re-break recall or binding. Flag,
   default OFF, per the repo protocol.

## 4. Exact repro commands
```bash
source diffkv_venv/bin/activate
# the fix (8k: filler -> paper):
DIFFKV_SPARSE_BIAS=2.0 python benchmarks/synthesis_eval.py --single-run --engine mlx --mode compressed --ctx 8192 | tail -1
python           benchmarks/synthesis_eval.py --single-run --engine mlx --mode compressed --ctx 8192 | tail -1   # bias off = filler
# still-open 16k (bias does NOT help — routing):
DIFFKV_SPARSE_BIAS=2.0 python benchmarks/synthesis_eval.py --single-run --engine mlx --mode compressed --ctx 16384 | tail -1
# guardrails the fix must not break:
DIFFKV_SPARSE_BIAS=2.0 python benchmarks/niah_recall.py --bench --ctx 4096 8192 16384 32768 --model mlx-community/Qwen2.5-1.5B-Instruct-4bit   # 4/4
# diagnostics used to root-cause (still available):
DIFFKV_DBG_FORCE_HALF=sparse python benchmarks/synthesis_eval.py --single-run --engine mlx --mode compressed --ctx 8192   # (probe removed from code; re-add if needed)
```
NOTE: `run_mlx` in `synthesis_eval.py` hardcodes `rank:16` (line 96) — rank is confirmed
irrelevant here. The uncommitted native `diffkv_native/src/main.cpp` (Antigravity's alnum
repetition filter) is still in the tree — decide whether to keep it in the native/perf session.
