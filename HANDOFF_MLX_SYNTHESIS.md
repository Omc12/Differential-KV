# HANDOFF — MLX compressed-synthesis bug (2026-07-04, Opus 4.8)

Read this first in the new session, then `SESSION_REPORT_FABLE5.md` "Ninth pass" and
`memory/project_w1_lse_merge_regression_fixed.md`. This continues the ninth pass: NIAH is
fixed (see below); the open item is **MLX compressed multi-fact synthesis**.

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

## 3. Suggested plan for the next session
1. Re-verify §0 guardrails. Decide on the uncommitted Antigravity changes (§1).
2. Instrument out_sparse vs out_dense value norms/content at a synthesis step; confirm whether
   the paper's out_sparse is usable or noise. (`compute_decode_attention_static`, ~line 300–441.)
3. Line-by-line compare MLX `compute_decode_attention_static` merge math against native's
   decode (`diffkv_native/src/main.cpp` execute_cpu_attention / the Metal callback) — native
   is the working oracle here.
4. Any change is gated by the FULL §0 table AND `niah_recall.py --bench 16k 32k` (the ≥16k
   compressed retrieval is knife-edge — a synthesis fix must not re-break recall). New behavior
   behind an env flag, default OFF, per the repo protocol.

## 4. Exact repro commands
```bash
source diffkv_venv/bin/activate
# the bug (filler summary, 1/15):
python benchmarks/synthesis_eval.py --single-run --engine mlx --mode compressed --ctx 8192 | tail -1
# the working oracles (paper summary):
python benchmarks/synthesis_eval.py --single-run --engine mlx  --mode dense      --ctx 8192 | tail -1
python benchmarks/synthesis_eval.py --single-run --engine native --mode compressed --ctx 8192 | tail -1
# attention evidence (sparse half attends paper tokens):
DIFFKV_DBG_LSE_SHARE=1 python benchmarks/synthesis_eval.py --single-run --engine mlx --mode compressed --ctx 8192 --gen 3 2>&1 | grep LSE_SHARE_ROW
```
NOTE: `run_mlx` in `synthesis_eval.py` hardcodes `rank:16` (line 96) — rank is confirmed
irrelevant here, but be aware the eval does not use the rank=32 default.
