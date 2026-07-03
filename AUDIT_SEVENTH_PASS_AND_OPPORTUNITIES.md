# Seventh Pass — Audit of the Antigravity C1–C3/B1–B3/D1 Execution + New Findings (2026-07-03, Fable 5)

Every verdict below was measured this session at HEAD `e34de4f` on the M3/8GB machine
unless marked *unverified*. Commands are the canonical guardrails from
`PLAN_ANTIGRAVITY_NEXT.md` §0.

---

## 1. Audit verdicts, claim by claim

| Claim (walkthrough / README / report) | Verdict |
|---|---|
| **C1 fused Metal kernel: "67.7 TPS @4k, 55.5 @16k, 100% recall up to 32k, parity 0.015"** | **FALSE on the canonical harness.** `DIFFKV_FUSED_DECODE=1 python niah_recall.py --bench --ctx 4096` → recall **FAIL** with garbage output (`materialising_nsould－…cplusplus,�实际上来物`) at **9.8 tps** — broken AND slower than the default path (19.4 tps). Their parity/tps numbers must come from a private script exercising a different call site / session shape. Two independent fused call sites exist (wrapper ~line 838 and ~line 2179); the canonical bench hits the broken one. |
| README env table: `DIFFKV_FUSED_DECODE` default `1` | **FALSE.** Code gates on `== "1"`, default OFF. Dangerous doc error: a user following the README enables a garbage-output path. Fixed this pass. |
| "Sequential Python path ~14.6–15 TPS" (the 4.5× baseline) | **Wrong baseline.** The default path measures 19.4 tps at 4k (and their own §0 edit says 20.2). The session's documents contain FIVE different 4k decode numbers (9.3, 11.1, 14.6–15, 20.2, 67.7) with no pinned configs — a §0 rule-3 violation pattern. |
| Native sweep "4/6" (plan §0 edit) vs "3/6" (their own report §6) | **Both wrong, and mutually contradictory.** Measured at HEAD (rebuilt binary): **6/6 — perfect honest sweep**, both with and without GQA routing. See §2. |
| GQA routing: "100% identical predictions", default ON | **Misleading + protocol violation.** The router only engages when resident blocks > `DIFFKV_TOPK_BLOCKS` (≥8k), so any 4k check tested nothing. At 8k the outputs happen to match, but shipping a routing-semantics change default-ON violates §0 rule 6. Its 8× routing-loop speedup is real arithmetic but routing was ~4.3ms of a ~13ms callback — end-to-end effect unmeasured. |
| GQA routing latency: "1.9–2.9ms" (walkthrough) vs "14.3ms" (README) vs "4.3ms" (report) for the same quantity | Three different numbers for one measurement across three documents. |
| C2/C3 profiling (readback ~0, routing 4.3ms, GPU spin-wait 8–9.5ms; prefill 99.2% compute-bound, scheduler recreation negligible) | **Plausible and useful; kept.** Note it contradicts the earlier D4 profile's framing (84.9ms/token total) — configs differ; re-pin before optimizing. |
| B1 synthesis eval: MLX-compressed 3.3/100 vs native-compressed 26.7 ≈ dense | *Unverified this pass* (heavy), but the harness is real, committed, and mechanically scored — good work. And the result is PLAUSIBLE for a specific reason Antigravity missed: see §3.1 (fp16 LSE merge quantization). This may be the most valuable thing the pass produced. |
| B2 multi-needle 100%, adversarial 4/5 0 misbound; B3 Llama-3.2-3B 100% recall to 32k, ~1.6GB | *Unverified this pass*; harness additions to `niah_recall.py` inspected — Llama template support only, Qwen prompts/needle/filler untouched → acceptable. B3 used Llama-3.2-3B instead of Qwen2.5-3B (fine; note the model dir sits untracked in the repo root — add to .gitignore). |
| D1 CUDA routing port | **Real and correct in shape:** `execute_cuda_attention` now receives the deduped/routed `slot_indices_cpu` + `actual_K` (signature changed in `diffkv_decode.cu`). Cannot be executed here (no CUDA box); compile-on-CUDA still unproven. D2–D4 honestly marked blocked. |
| A1 position-0 hole | **Not attempted** — the plan's stated top priority was skipped entirely. Still open (see §4.3). |
| "Committed and pushed to main" | True (origin/main up to date). |

**Guardrails re-verified by me at HEAD:** parity 4/4 · SELFTEST PASS 5.96e-08 · native
default-path honest sweep **6/6** · MLX `--bench` 4k = exact @19.4 tps · fused MLX path
0/1 garbage @9.8 tps.

## 2. The surprise: native is now 6/6 with dense-level margins — and nobody claimed it

Measured at HEAD after a fresh rebuild (their committed binary predated their last code
commit — 18:51 binary vs 19:00 commit):

- Honest 6-cell default-path sweep: **6/6** (was 3/6 at `05a3006`).
- The 8k/0.5 retrieval-step margin: `-` = 32.9–33.1, margin **+12–13** — equal to the
  dense-read control (+12.5). The ~14-point sparse-read deficit that defined D7 is GONE.
- GQA routing on vs off: both 6/6, both full margin → GQA is NOT the cause.
- The only code deltas since 3/6 are math-neutral (timers, CUDA branch, GQA-off path
  identical) → **the most likely explanation is that the fifth-pass 3/6/low-margin
  measurements ran on a stale incremental build that did not fully contain the
  fourth-pass fixes (`b16c3ac` route-all + `06ef021` int8-exact residuals).** I could not
  re-verify by clean-building `05a3006` (the attempt crashed the machine); treat the
  attribution as *probable, not proven*. What IS proven: at today's HEAD, clean state,
  the two fixes together deliver parity-level margins and a perfect sweep.

Consequences:
- **D7 is closed as a pass-count problem.** The remaining risk is *fragility*: these
  cells sat within ~1 nat for weeks; the correct standing guardrail is now the **margin**
  at the retrieval step, not the boolean sweep (see §4.4).
- **A1 (position-0 hole) is demoted from "accuracy blocker" to "robustness/coherence
  item"** — the hole is still real (probe evidence stands) and still wrong, but it is
  evidently not load-bearing for NIAH pass/fail at current margins.

## 3. New bugs found during this audit

### 3.1 MLX fp16 LSE merge quantization (pre-existing; likely explains B1's 3.3/100)

`compute_decode_attention_static` computes `lse_sparse`/`lse_dense` in the activation
dtype (fp16). Qwen2.5's massive activations push LSE magnitudes to ~1.2e4 (measured,
fifth pass), where **fp16 spacing is 8**. The sparse⊕dense merge weights
`exp(lse − lse_max)` can therefore only take values `e^0` or `e^±8·k` — the blend is
**50/50 or winner-take-all, never graded**, at every layer where activations are massive.
NIAH cannot see this (one half dominates at retrieval steps); *synthesis* is exactly
where a graded blend matters — consistent with MLX-compressed scoring 3.3/100 while
native-compressed (fp32/double LSE throughout) matched dense at 26.7. Antigravity's
fp32 cast computes logsumexp in fp32 **then casts straight back to fp16**, preserving
the bug. Their B1 result is best read as this bug's first visible symptom.

**Fix approach (small, high value):** keep `lse_sparse`, `lse_dense`, `w_sparse`,
`w_dense`, `denom` in fp32 end-to-end inside `compute_decode_attention_static` (and the
same math in `_execute_decode_attention_compiled`'s consumers); cast only `out_combined`
back to the activation dtype. **Acceptance:** parity 4/4 (update the oracle first),
`--bench` 4/4, relational 4/4, then re-run `synthesis_eval.py --ctx 8192` MLX-compressed
— if the 3.3 jumps materially toward the dense 26.7, the mechanism is confirmed and this
is the single biggest MLX quality fix available.

### 3.2 The incomplete fp32 cast (same code)

The overflow fix casts the *product* to fp32: `(q_exp * k_exp).astype(float32)` — but
the elementwise product itself is computed in fp16 first. Massive-dim products
(~q·k ≈ 1e5 per dim pair) can overflow to inf *before* the cast. Cast the operands, not
the product: `q_exp.astype(f32) * k_exp.astype(f32)` (or `mx.sum(x, dtype=f32)` where
available). Same acceptance set as §3.1; do them together.

### 3.3 Fused MLX kernel: broken on the canonical path, two divergent call sites

Beyond the §1 failure: the fused branch exists in TWO places (wrapper ~838 inside
`_execute_decode_attention_compiled`'s flow and ~2179 at the block-manager level) with
independently maintained input prep. One of them produces garbage under `--bench` at 4k.
Either fix and dedupe to a single call site with a parity case in
`test_diffkv_kernel_parity.py` (fused-vs-reference on randomized sessions — the
acceptance the plan demanded and that was skipped), or delete the branch. A kernel that
silently produces garbage behind a README-advertised flag is worse than no kernel.

### 3.4 Protocol/documentation debt from this pass

- README env table said `DIFFKV_FUSED_DECODE=1` default (false; corrected this pass) and
  advertises the GQA default-ON flip as a milestone with three inconsistent latency
  numbers.
- The §0 baseline table in the plan was edited to values (4/6) that contradict the same
  pass's own session report (3/6) — and both were wrong (6/6 measured).
- My D2–D4 task specs were deleted from the plan when marking them blocked (recoverable
  from git; restored spec pointers this pass).

## 4. Ranked opportunities (next work, in order)

1. **fp32 LSE merge + operand-level casts (§3.1+§3.2)** — MLX accuracy/quality; likely
   converts the B1 synthesis collapse into a pass; touches only the reference + compiled
   path; one afternoon including guardrails.
2. **Fused MLX kernel: fix-or-delete (§3.3)** — with the mandatory parity case. Until
   then the README must not advertise it. If fixed, re-measure honestly on `--bench`
   (not a private script); the dispatch-bound thesis says a real fused path should beat
   19.4 tps at 4k — prove it or record the negative.
3. **A1 position-0 / sink block (robustness)** — the hole is confirmed and philosophically
   wrong even though current margins tolerate it: sink mass lands somewhere arbitrary,
   and B1-style long-form generation is where it would show. Do it with the LSE2 share
   probe as the metric (not the sweep, which is saturated).
4. **Margin-based guardrail** — add a `--margin` mode to the native harness runner (env
   `DIFFKV_DBG_STEP_LOGITS` already prints step top-5): report the retrieval-step margin
   for 8k/0.5 and 16k/0.5 alongside pass/fail. The sweep is now saturated at 6/6 and
   knife-edge history says pass-counts hide erosion; margins do not.
5. **GQA-route default decision** — currently ON without evidence. Either produce an 8k+
   A/B (tps + margins + sweep, on/off) that justifies ON, or flip default OFF per §0
   rule 6. My data: no accuracy difference at 8k/16k; end-to-end tps effect unmeasured.
6. **Q8_0 default flip** — still pending only the fused-path sweep + RSS delta
   measurement (accuracy already cleared cell-for-cell at f16 parity).
7. **B1 across engines at 16k/32k** after (1) lands — the 2×2 table is the right
   instrument; also re-score with word-boundary matching (`"cvm" in text` currently
   substring-matches unrelated words) — scorer tweak allowed ONLY as a committed,
   documented change before any engine comparisons re-run.
8. **CUDA validation** (D2–D4) when a box exists; D1's port compiles on Mac but has never
   compiled under nvcc.

## 5. Architecture recommendation (explicit, as requested)

**No rewrite is warranted. The costly problem is decode-path divergence, and the fix is
consolidation + a shared conformance oracle:**

DiffKV currently carries **five** sparse-decode implementations (MLX compiled reference,
MLX fused Metal kernel, native CPU op, native Metal callback kernel, native fused-ggml
subgraph — plus the CUDA kernel as a sixth). Essentially every bug found across seven
passes — the anchor_screen selection pollution, the residual twin-mask ledger asymmetry,
the fused-16k salad, the fp16 LSE merge, today's broken fused MLX branch — is a
*divergence* bug: one path drifting from the reference semantics, undetected because
each path has its own ad-hoc checks.

Concrete proposal (medium effort, high leverage, no algorithm change):
1. Write the decode semantics ONCE as a versioned spec + **golden test vectors**
   (seeded sessions serialized to disk: pool tensors, dense window, query → expected
   output at fp32, tolerance schedule for fp16/int8 paths).
2. Every path (MLX compiled, MLX fused, native CPU, native Metal, native fused, CUDA)
   must load the SAME vectors and pass in CI — one conformance harness instead of five
   ad-hoc ones. `test_diffkv_kernel_parity.py` and `DIFFKV_SELFTEST` become thin
   frontends over it.
3. Kill paths that don't pay rent: the native fused-ggml subgraph is default-off, slower
   (1.9×), broke at 16k, and duplicates the Metal callback kernel — delete it once the
   callback path has the conformance harness. That's minus one entire divergence
   surface.

This is the change that stops the project from re-finding the same class of bug every
session. I recommend it explicitly and it should be its own work item before any further
kernel work.

## 6. Corrected baselines (measured this pass, HEAD `e34de4f`)

| Guardrail | Value |
|---|---|
| Parity | 4/4 |
| SELFTEST | PASS 5.96e-08 |
| Native honest sweep, default path | **6/6** |
| Native 8k/0.5 retrieval-step margin | +12.2 (GQA on) / +13.0 (GQA off) — dense control is +12.5 |
| MLX `--bench` 4k | exact, 19.4 tps |
| MLX fused (`DIFFKV_FUSED_DECODE=1`) 4k | **0/1, garbage, 9.8 tps — do not enable** |
| Unverified this pass (machine constraints) | B1/B2/B3 tables, 16k/32k benches, fused sweep re-run |
