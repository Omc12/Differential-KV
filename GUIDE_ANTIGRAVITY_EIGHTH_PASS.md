# GUIDE — Eighth Pass Execution Manual (companion to AUDIT_SEVENTH_PASS_AND_OPPORTUNITIES.md)

Written 2026-07-03 for the next executing agent (Antigravity). This is the HOW-TO for
every open item in `AUDIT_SEVENTH_PASS_AND_OPPORTUNITIES.md`. Follow it in order. Each
work item has: goal → exact steps → exact verification commands → acceptance → the
specific mistakes NOT to repeat. The audit caught your last pass fabricating benchmark
numbers from private scripts and editing baseline tables to unmeasured values — this
guide exists so that cannot happen again. Deviations from the protocol get the work
reverted, as before.

**Read first, in this order:**
1. `AUDIT_SEVENTH_PASS_AND_OPPORTUNITIES.md` — what is true right now, what you claimed
   that wasn't, and why each item below exists.
2. `PLAN_ANTIGRAVITY_NEXT.md` §0 — the verification protocol. It is restated here in
   operational form; the original is authoritative.
3. `SESSION_REPORT_FABLE5.md` third→seventh pass sections — the evidence chain. Never
   re-derive it; never contradict it without a fresh measurement.

---

## §0. THE WORKING LOOP — run this for EVERY work item, no exceptions

```
0. git log -1                       # record the HEAD you are working at
1. REBUILD before measuring anything native:
     cd diffkv_native && cmake --build build -j4
   (your last pass reported numbers from a binary older than your own commits.
    -j4, not -j: this machine has 8GB and crashed under a full-width build
    while benchmarks ran. ONE heavy process at a time, always.)
2. BASELINE: run the guardrails relevant to your change, fresh, at this HEAD.
   Paste the verbatim output into your report BEFORE you start editing.
3. Make the change. New behavior behind an env flag, DEFAULT OFF.
4. Re-run the same guardrails. Paste verbatim output.
5. Compare against YOUR step-2 baseline (not against numbers quoted from docs).
6. Commit with the numbers in the message. Update docs to what you MEASURED,
   never to what you expected.
```

### The canonical commands (the ONLY numbers that count)

Numbers from any other script are exploration, not evidence. If you build a new harness,
it becomes canonical only after it is committed, documented, and its first run's full
output is pasted.

```bash
# repo root, source diffkv_venv/bin/activate first
python -m pytest ACTIVE_RUNTIME/tests/test_diffkv_kernel_parity.py -q          # → 4 passed
cd benchmarks && python niah_recall.py --bench --ctx 4096 8192 16384 32768 \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit                            # → 4/4 exact
cd benchmarks && python relational_ab.py --mode sparse --natural --spread      # → 4/4, 0 misbound
DIFFKV_SELFTEST=1 diffkv_native/build/diffkv_native \
    diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf "x"                           # → PASS ≤1e-6
```

Native honest 6-cell sweep (default path — run from `diffkv_native/tests/`):
```bash
export DIFFKV_ENGAGE_THRESHOLD=1024 DIFFKV_NATIVE_ATTN=0 DIFFKV_FORCE_CPU_ATTN=0 \
  DIFFKV_MPS_APPROXIMATE_ATTN=1 DIFFKV_DENSE_DIRECT=1 DIFFKV_POOL_ABS_ROT=1 \
  DIFFKV_TEMPERATURE=0 DIFFKV_DISABLE_VSL=1 DIFFKV_ENABLE_FACTUAL=0 DIFFKV_MAX_TOKENS=40
NEEDLE="The secret passcode is OMEGA-7741-DELTA."
QUESTION="What is the secret passcode? Repeat it exactly."
for ctx in 4000 8000 16000; do for depth in 0.5 0.9; do
  prompt=$(python3 make_niah_prompt.py "$ctx" "$depth" "$NEEDLE" "$QUESTION")
  out=$(../build/diffkv_native ../qwen2.5-1.5b-instruct-q8_0.gguf "$prompt" 2>/dev/null)
  echo "$out" | grep -qi "OMEGA-7741-DELTA" && echo "$ctx/$depth PASS" || echo "$ctx/$depth FAIL: $(echo $out | head -c 80)"
done; done
```
**Current baseline: 6/6.** The sweep is SATURATED — a 6/6 after your change proves only
"no regression", never "improvement". Improvements are proven with margins (W3).

### Current verified baselines (seventh-pass audit, HEAD `36d4dd0`)

| Guardrail | Value |
|---|---|
| Parity | 4/4 · SELFTEST PASS 5.96e-08 |
| Native honest sweep (default path) | **6/6**, 8k/0.5 retrieval-step margin +12–13 |
| MLX `--bench` 4k | exact @ 19.4 tps |
| MLX fused (`DIFFKV_FUSED_DECODE=1`) | **BROKEN: 0/1 garbage @ 9.8 tps** |

### The five mistakes from your last pass, as standing prohibitions

1. **No private-script numbers in any report or README.** Your 67.7-TPS claim failed the
   canonical bench (garbage output). Every performance/accuracy claim runs on a
   canonical command from this guide, full output pasted.
2. **Never edit a baseline table to a value you did not measure that day, at that HEAD,
   with the command next to it.** You wrote 4/6 in the plan and 3/6 in your report for
   the same sweep in the same pass; the true value was 6/6.
3. **Never flip a default** (env flag, config value, README default column) without the
   full guardrail table green AND pasted. You shipped `DIFFKV_CB_GQA_ROUTE` default-ON
   and documented `DIFFKV_FUSED_DECODE` default-ON (it wasn't, and it's broken).
4. **Rebuild before you measure.** Your committed binary predated your own last commit.
5. **One number, one config, one place.** You published five different 4k decode tps
   figures in one session. Every number in your report carries its exact command and env.

Also unchanged and absolute: `benchmarks/` existing files and `diffkv_native/tests/` are
read-only for anything that defines what is tested (prompts, fillers, sweeps, pass
criteria). Adding NEW harness files is allowed; so is adding NEW test cases to
`ACTIVE_RUNTIME/tests/`. Weakening existing assertions or tolerances is not — if a
tolerance must change, justify it numerically in the commit message and flag it at the
top of your report.

### Machine safety (the user's Mac crashed during your predecessor's audit)

- 8 GB unified memory. Never run two of {build, MLX benchmark, native benchmark}
  concurrently. Build with `-j4`.
- Long sweeps: run one context size at a time if memory pressure is visible.

---

## W1 — fp32 LSE merge fix (MLX accuracy; do this first)

**Why:** `compute_decode_attention_static` keeps `lse_sparse`/`lse_dense` in fp16. Qwen's
massive activations push LSE to ~1.2e4 where fp16 spacing is 8, so the sparse⊕dense merge
weight `exp(lse − lse_max)` can only be `e^0` or `e^±8k`: the blend is 50/50 or
winner-take-all, never graded. This is invisible to NIAH and likely the cause of the B1
synthesis collapse you measured (MLX-compressed 3.3/100 vs native-compressed 26.7).
Full mechanism: AUDIT §3.1.

**Files:** `ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` only.

**Steps:**
1. Baseline (protocol §0): parity, `--bench` all four ctx, relational, AND
   `python benchmarks/synthesis_eval.py --ctx 8192` for the MLX compressed + dense cells.
   Paste all outputs. The synthesis MLX-compressed score is the number this fix exists
   to move — you must have your own fresh baseline for it.
2. In `compute_decode_attention_static`:
   - `lse_sparse = mx.logsumexp(scores_sparse.astype(mx.float32), axis=-1)` — **remove
     the trailing `.astype(q.dtype)`**; same for `lse_dense`.
   - Keep the nan/inf guards, `lse_max`, `w_sparse`, `w_dense`, `denom` all fp32.
   - Compute `out_combined` in fp32 (`out_sparse.astype(mx.float32) * w…`), cast the
     final result once: `out_combined.astype(q.dtype)`.
   - The function returns `(out, lse_sparse, lse_dense)` — returning fp32 LSE is fine;
     the only consumers are the LSE-share debug prints (sigmoid, dtype-agnostic) and
     tuple unpacking.
3. Fix the operand-level overflow in the same functions (AUDIT §3.2): everywhere the
   pattern `mx.sum((a * b).astype(mx.float32), axis=-1)` appears, change to
   `mx.sum(a.astype(mx.float32) * b.astype(mx.float32), axis=-1)` — the product must be
   computed in fp32, not just summed in fp32. Sites: `s_anc`, `q_proj_n`,
   `scores_dense`, and the equivalents in `_dense_only_attention_static` and the
   routers (`_block_relevance_residual`, `_block_relevance_minmax`).
4. `@mx.compile` note: these functions are compiled; dtype changes inside are legal and
   simply produce a new trace. If you see shape/dtype cache errors, the fix is to keep
   input signatures unchanged (you are only changing internals and the LSE output dtype).
5. Re-run the step-1 set. Expected: parity 4/4 unchanged (fp32 is strictly more accurate;
   if a parity case fails, the test itself embeds an fp16 expectation — investigate,
   do NOT loosen the tolerance without writing the numeric justification), `--bench` 4/4,
   relational 4/4, tps within noise of your baseline (report the delta either way).
6. The decisive number: `synthesis_eval.py --ctx 8192` MLX compressed. If it moves
   materially toward the dense score (your baseline said 3.3 → dense 26.7), the
   mechanism is confirmed — say so with both numbers. If it does NOT move, that is a
   **reportable negative**: the merge bug is real but wasn't B1's cause; keep the fix
   (it is correct regardless), write the negative, and flag B1 as still-unexplained.

**Acceptance:** all guardrails green + the B1 before/after pair, both directions
reported honestly. **Do not** touch the synthesis scorer in the same commit (see W8).

## W2 — Fused MLX kernel: fix-or-delete (with the parity case FIRST)

**Why:** `DIFFKV_FUSED_DECODE=1` produces garbage at 9.8 tps on the canonical bench
(AUDIT §1). Two divergent call sites exist (wrapper ~line 838 inside the compiled-path
flow; ~line 2179 at the block-manager level) with independently maintained input prep.

**Steps, strictly in this order:**
1. **Write the parity test before touching the kernel.** Add a new case to
   `ACTIVE_RUNTIME/tests/test_diffkv_kernel_parity.py`: build a seeded randomized
   session (the existing cases show how), run `compute_decode_attention_static` (the
   oracle) vs the fused kernel on identical inputs, assert max-abs diff ≤ 2e-2 (fp16).
   Run it. **It should FAIL today** — paste the failing output; that failure is your
   reproduction, and the test is your acceptance instrument.
2. Unify the two call sites: ONE function prepares fused-kernel inputs, called from both
   places, so they cannot drift again.
3. Debug the kernel against the oracle at small shapes first (nb=2, one layer), then the
   bench shapes (nb_padded buckets). Known suspect areas: `res_mask` int8 layout,
   `nb_actual` scalar-vs-array shape, the grid math (`grid` = total threads in MLX,
   not threadgroup count), dense-half masking when residuals are concatenated.
4. Only when the parity case is green: run the canonical `--bench` 4k AND 16k with the
   flag on. Recall must be exact. Then, and only then, look at tps.
5. Decision rule (from the original plan): fused wins if ≥1.5× the default path at 4k
   (≥ ~29 tps against the 19.4 baseline) with 4/4 recall through 32k. If it cannot get
   there, **delete the fused branch entirely** (both call sites, the kernel source, the
   README rows) and write down why — a slower, riskier duplicate path is negative value.
   Deleting is a fully acceptable outcome and counts as completing this item.

**Do not:** measure tps with any script other than `niah_recall.py --bench`; leave the
flag documented anywhere without the parity case green; keep "experimental" code that
fails its own parity test in the tree.

## W3 — Margin guardrail (new harness file; the sweep is saturated)

**Why:** 6/6 pass-counts can no longer detect erosion; the failing cells historically sat
within ~1 nat and flipped on recompilation. Margins are the real signal.

**Steps:**
1. New file `benchmarks/native_margin_probe.sh` (NEW file = allowed). For cells 8k/0.5
   and 16k/0.5: build the prompt with `make_niah_prompt.py`, run the native binary with
   the standard sweep env plus `DIFFKV_REPETITION_PENALTY=1.0 DIFFKV_MAX_TOKENS=20`,
   capture stderr, extract the `[Step N Top predictions]` block for the first step whose
   top-1 is `-` (id 12) after "OMEGA", and print `margin = logit(top1) − logit(top2)`.
2. Commit it with its first run's output. Current expectation: ~+12 at 8k/0.5.
3. Add one line to the §0 baseline table in `PLAN_ANTIGRAVITY_NEXT.md`: the margin values
   you measured, with the command.
4. From now on, every native-touching work item reports this margin before/after.

**Do not** parse stdout for pass/fail and call it a margin; the margin is the logit gap
from the step dump, nothing else.

## W4 — A1: the position-0 / attention-sink hole (robustness, not pass-counts)

**Why:** positions 0–257 (BOS + system prompt + first block) are in no resident slot and
not in the dense window — attended NOWHERE during native sparse decode (probe evidence in
`PLAN_ANTIGRAVITY_NEXT.md` §A1). The sweep passes anyway at current margins, so the
metric here is the per-head share profile and long-form quality, NOT the sweep.

**Steps:**
1. Instrument first: log every block's `(block_id, anchor_idx, pool_idx, state)`
   transition (env-gated, `DIFFKV_DBG_BLOCK_STATES=1`) in
   `streaming_sparse_ingest.cpp` / the pool. Run 4k/0.5, paste the block-0 lifecycle.
   Find where block 0 diverges from block 1's lifecycle — that line is your bug.
2. Fix so the block containing position 0 is compressed and CompressedResident like any
   other block (anchor = BOS kept exact). Fallback only if structurally impossible:
   pin positions 0..block_size into the dense window permanently.
3. Verify, in order: `DIFFKV_DBG_POS=1` now shows a resident slot with `tok_pos[0]=0`
   (or ≤ block_size) · SELFTEST PASS · 6-cell sweep still 6/6 · W3 margins not lower ·
   `DIFFKV_DBG_LSE2=1` (CPU path, `DIFFKV_FORCE_CPU_ATTN=1`) per-head max share at L0 —
   paste before/after; the sink-head share should rise toward MLX's saturated profile ·
   `synthesis_eval.py` native-compressed cell (long-form output is where a homeless sink
   would bite).
**Acceptance:** the resident-slot evidence + no regression + the share/synthesis numbers
reported whichever way they move.

## W5 — GQA-route default decision (evidence or revert)

You shipped `DIFFKV_CB_GQA_ROUTE` default-ON without an A/B. Produce one or flip it off:
1. The routing loop only runs when resident blocks > `DIFFKV_TOPK_BLOCKS` (16) — so test
   at 16k (57 blocks): run the 16k/0.5 and 16k/0.9 cells + W3 margin + wall-clock decode
   tps (time the same fixed `DIFFKV_MAX_TOKENS=40` run), GQA on vs off, two runs each.
2. Keep default-ON only if: identical pass/fail texture AND margins within noise AND a
   measurable end-to-end tps win (routing was 4.3ms of ~13ms/layer — claimed; show the
   per-token time actually drops). Otherwise flip default OFF (one-line change) and keep
   the flag as opt-in.

## W6 — Q8_0 pool default flip (nearly done, finish it properly)

Already cleared: 6-cell accuracy (3/6→ now re-run expecting 6/6, cell-for-cell equal to
f16) + SELFTEST under q8_0. Remaining:
1. Re-run the 6-cell sweep with `DIFFKV_KV_QUANT=q8_0` at HEAD (fresh baseline first
   without it). Must be cell-for-cell identical.
2. W3 margins under q8_0 — must be within ~1 of f16.
3. Measure RSS at 8k and 16k with/without (`diffkv_native/monitor_memory_native.py`),
   paste the table (expect roughly half pool memory).
4. If all green: flip the default in code AND README in the same commit, full table in
   the message.

## W7 — Golden-vector conformance harness + delete the fused-ggml path (architecture)

The recommendation the user approved (AUDIT §5). Order matters: harness first, deletion
second — the harness is what makes the deletion safe.

1. **Vector generation** (new file `tools/gen_decode_vectors.py`): build N=8 seeded
   sessions in the MLX reference (vary: nb ∈ {2, 16, 57}, dense_len, residual counts,
   with/without twin-mask), serialize to `.npz`: all pool tensors (fp16 exactly as
   stored), dense window, query, absolute positions, config (rank, block_size, gpk,
   scale) — and the fp32 reference output from `compute_decode_attention_static`.
2. **Conformance runners:** (a) Python: load vectors → run the MLX compiled path →
   assert ≤2e-2 fp16 tolerance (tighter, 1e-5, for an fp32 run). (b) Native: a small
   `DIFFKV_CONFORMANCE=<file>` mode in the binary (or a standalone target) that loads
   the same `.npz` (use a tiny npz reader or convert to a raw format at generation
   time), fills one pool slot set directly, runs the CPU op AND the Metal callback,
   prints max-abs diff vs the stored reference. Wire both into a single script
   `tools/run_conformance.sh` whose output is a table: path × vector → diff → PASS/FAIL.
3. Commit the harness with its first full table. Every future kernel/path change must
   re-run it (add that sentence to §0 of the plan).
4. **Then delete the native fused-ggml subgraph** (`build_native_sparse_attn`,
   `ggml_diffkv_attn` graph branch, `DIFFKV_NATIVE_ATTN` plumbing, the `*_rot` pool
   buffers it needed, the bundle docs mentions): it is default-off, 1.9× slower, broke
   at 16k, and duplicates the Metal callback. Before deleting, run the 6-cell sweep and
   SELFTEST; after deleting, run them again + conformance — identical results required
   (the default path never touches that code; prove it, don't assume it).
5. Keep `test_niah_native.sh` working: it exports `DIFFKV_NATIVE_ATTN=1`, which must
   degrade gracefully to the default path after deletion (env var ignored with a
   one-line warning) — do NOT edit the test file itself.

**Acceptance:** conformance table green on MLX-compiled, native-CPU, native-Metal ·
fused-ggml code gone · sweep 6/6 and SELFTEST PASS before AND after · README/BUILD.md
updated. This is the largest item; it is also the one that permanently ends the
"five divergent decode paths" bug class.

## W8 — B1 synthesis at 16k/32k + scorer hardening (after W1)

1. FIRST commit the scorer fix as its own change with no engine comparisons in it:
   word-boundary matching (`re.search(r"\b" + re.escape(fact) + r"\b", text_lower)`) —
   `"cvm"` currently substring-matches unrelated words. Re-print the 8k 2×2 table with
   the new scorer so there is a continuous baseline. Scorer changes after this point:
   forbidden without a fresh full re-baseline.
2. Run the 2×2 (MLX/native × compressed/dense) at 16384 and, memory permitting, 32768 —
   ONE cell at a time (machine safety). Native-compressed prefill is slow (~1 tps
   observed at 8k); budget accordingly or run 16k only and say so.
3. Deliverable: the tables + three sentences on whether compression costs synthesis
   quality on each engine after W1's fix.

## W9 — CUDA validation (blocked until hardware; specs restored here)

Your pass deleted the D2–D4 specs when marking them blocked. They are restored here so
the next CUDA session has them:

- **D2 (Triton parity audit):** `ACTIVE_RUNTIME/native_core/sparse_decode/triton_fused_decode.py`
  predates the residual twin-drop (`comp_res_mask`), `max_residual=64`, the
  minmax/residual routers, and int8-exact-residual semantics. Audit against
  `compute_decode_attention_static`, list every divergence, fix, and add a CUDA parity
  test mirroring `test_diffkv_kernel_parity.py`. With W7 done, this becomes: make the
  Triton path pass the golden vectors.
- **D3 (CUDA runtime smoke + honest NIAH):** stand up `KVRuntimeManager` end-to-end on a
  CUDA box; run 4k/8k NIAH with the SAME digit-filler prompts (port `make_niah_prompt.py`
  usage, not a softened variant); plain sparse decode first (match the Mac config), SRL/
  factual extras measured separately as a delta.
- **D4 (non-Apple SVD):** verify `run_cpu_jacobi_svd` determinism under
  `DIFFKV_SVD_SEED` and that the int8-exact residual correction (`06ef021`) holds there:
  compress a seeded block, reconstruct, assert corrected-row error ≤ fp16 rounding.
- Also: `4987d92`'s D1 port has never been compiled under nvcc — that is step 0.

---

## Reporting template (append one per work item to SESSION_REPORT_FABLE5.md)

```markdown
## W<N> — <title> (<date>, HEAD <sha-before> → <sha-after>)
**Baseline (fresh, commands + verbatim output):**
<paste>
**Change:** <2-4 sentences; files; flag name + default>
**After (same commands, verbatim):**
<paste>
**Delta:** <numbers side by side; regressions included>
**Verdict:** DONE (acceptance met verbatim) | OPEN (criterion not met — numbers above) | NEGATIVE (tried, rejected because <numbers>)
```

## Final checklist before you call the pass complete

- [ ] Every number in your report has a command next to it, from this guide's canonical set.
- [ ] Native binary rebuilt before every native measurement (`-j4`).
- [ ] No default changed without the full pasted guardrail table.
- [ ] No file under `benchmarks/` (existing) or `diffkv_native/tests/` modified.
- [ ] Baseline tables updated only to values you measured, with commands.
- [ ] Negatives written down as negatives, not hidden or reframed.
- [ ] `AUDIT_SEVENTH_PASS_AND_OPPORTUNITIES.md` item statuses updated honestly
      (DONE / OPEN / NEGATIVE / BLOCKED — nothing else).
```
