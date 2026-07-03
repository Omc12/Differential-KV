# New Directions — Implementation Plan (handoff, written 2026-07-03)

This is the work plan for the NEW sparse-attention directions, written for whichever
agent/tool executes next. It assumes HEAD ≥ `bb0f343` (verification pass complete).
Read `SESSION_REPORT_FABLE5.md` (last two sections) and `PLAN_FABLE5_OPTIMIZATION.md` §8
first — several superficially-attractive ideas are **already tried and rejected with
measurements**; do not re-implement them.

---

## 0. Ground rules (non-negotiable — the last pass violated these and shipped a regression)

1. **Never modify benchmark prompts, fillers, sweep ranges, or pass criteria.** The digit
   token `2010s` in the NIAH fillers is load-bearing (digit-vs-digit residual competition
   is the thing being tested). If a benchmark fails, the ENGINE is wrong, not the benchmark.
   Any change under `benchmarks/` or `diffkv_native/tests/` that alters what is tested = the
   work is rejected.
2. **Every claim ships with the exact command and its verbatim output.** "Verified" without
   a pasted number is not verified. Negative results are good results — report them.
3. **New behavior goes behind an env flag, default OFF.** Defaults flip only with before/after
   numbers from the guardrail commands below, run at HEAD, same machine.
4. **Run the guardrails before AND after each work item** (baselines at `bb0f343` given):

   | Command (from repo root, `diffkv_venv`) | Baseline |
   |---|---|
   | `python -m pytest ACTIVE_RUNTIME/tests/test_diffkv_kernel_parity.py -q` | 4 passed |
   | `cd benchmarks && python niah_recall.py --ctx 4096 --depths 0.1 0.5 0.9 --model mlx-community/Qwen2.5-1.5B-Instruct-4bit` | 3/3, ~20 tps |
   | `cd benchmarks && python niah_recall.py --bench --ctx 4096 8192 16384 32768 --model mlx-community/Qwen2.5-1.5B-Instruct-4bit` | 4/4; tps 19.7/15.8/13.5/10.6 |
   | `cd benchmarks && python relational_ab.py --mode sparse --natural --spread` | 4/4, 0 misbound |
   | `cd diffkv_native/tests && ./test_niah_native.sh` | **1/6** (4k/0.9 only) — this is the honest baseline; do not expect 6/6 |
   | `DIFFKV_SELFTEST=1 diffkv_native/build/diffkv_native <model> "x"` | PASS, maxAbsDiff ≤ 1e-6 |

5. **Do not touch the llama.cpp submodule** unless required; if you commit inside it,
   regenerate `diffkv_native/third_party/diffkv-fused-op.bundle` (command in BUILD.md §1)
   in the same superproject commit.
6. Commit per work item, message states what was measured. Update this file's checkboxes.

**Already REJECTED with measurements — do not redo:** V-only residual ranking (3/3→1/3);
SVD-reconstructed-K residual storage (3/3→1/3; residual rows are by selection the
worst-reconstructed rows, so exact K is mandatory); blanket byte-token boost (2026-07-02);
`MAX_RESIDUAL` increases alone; fused-native default-on (1.9× slower, reverted `efbc87e`).

---

## D1 — Native needle-capture root cause (TOP PRIORITY, correctness)

**Problem:** native fails 5/6 honest NIAH cells (`OMEGA-999-0`-style digit confabulation)
even though the IDF rarity weighting and string-based token classification ARE present and
wired (`lowrank.cpp:843-991`, `token_to_piece_fn` set in `streaming_sparse_ingest.cpp:699`,
verified 2026-07-03). MLX passes the same prompt family. Something between the two differs.

**Protocol (evidence first, fix second):**
1. Build the 4k/0.5 prompt (`make_niah_prompt.py 4000 0.5 ...`), tokenize, find the row
   indices of the needle tokens (`OMEGA`, `7741`, `DELTA`) → their absolute positions.
2. Run with `DIFFKV_DBG_RECON_POS=<pos>` for each needle position. Classify the failure:
   - **(a) row never captured** as residual → capture-policy bug. First suspect, and the
     known MLX/native difference: MLX's benchmark registers the ENTIRE prompt's token ids
     before prefill, so IDF counts are corpus-accurate from block 0; native's
     `session_token_ids_` grows during streaming ingest (`streaming_sparse_ingest.cpp:403`),
     so early blocks compute IDF against a partial corpus — early filler digits ('2010')
     look rare and get boosted like the needle. **Fix if confirmed:** register the full
     prompt token ids into `session_token_ids_` before ingestion starts (native knows the
     whole prompt in `main.cpp`); IDF then matches MLX semantics.
   - **(b) captured, then displaced** by a later boost pass / slot reuse → port
     `DIFFKV_RESIDUAL_COVERAGE_FRAC` from `mlx_diffkv_wrapper.py` (`_coverage_bonus`) into
     `lowrank.cpp`'s ranking, same semantics, default 0.
   - **(c) captured but mis-read at decode** → routing/apply bug; probe with
     `DIFFKV_DBG_STEP_LOGITS` + the callback router logs before touching anything.
3. Fix ONLY what the probe implicates. Re-run the full 6-cell sweep.

**Acceptance:** `./test_niah_native.sh` improves from 1/6 with prompts untouched; report
the cell-by-cell table. Even 3/6 with a root-cause explanation beats 6/6 by any shortcut.
**Effort:** 1-2 days. **Risk:** low (probes exist).

## D2 — LSE-gated block re-expansion (the most novel direction)

**Idea:** decode already computes logsumexp per component when merging sparse and dense
attention halves. When (i) the compressed pool's LSE share for a step is high (the answer
lives in a compressed block) and (ii) next-token entropy is high (model unsure), fetch the
top-routed block's EXACT tokens and splice them into the dense window for the next few
steps. Exactness on demand — sidesteps the residual-budget zero-sum game entirely.

**Phase A — measurement harness (do this first; it can kill the idea for ~a day's work):**
- MLX: env `DIFFKV_DBG_LSE_SHARE=1` → per decode step, log
  `share_comp = exp(lse_comp) / (exp(lse_comp)+exp(lse_dense))` (per layer or maxed over
  layers) + the top-routed block id. Hook where the halves merge
  (`_combine`/LSE logic in `execute_decode_attention` / `compute_decode_attention_static`;
  the LSE diagnostics added in commit `4ed59f5` are a starting point).
- Run `niah_recall.py --bench --ctx 16384 32768`; compare shares on the steps that emit
  the passcode vs ordinary prose steps.
- **Decision rule:** if needle-steps' compressed-share does not clearly separate from
  prose-steps', STOP — write the negative result in the session report and skip Phase B.

**Phase B — re-expansion (only if Phase A shows signal):**
- Spill: at compression time, optionally write each block's exact K/V to a per-session
  memmap file (`DIFFKV_BLOCK_SPILL=1`, `np.memmap`, fp16, block-indexed). Unified memory
  stays flat; SSD is fast on Apple Silicon.
- Gate at decode: when share_comp > threshold AND entropy > threshold, load the top-routed
  block's exact K/V, append to the dense half (positions are absolute, pool is pre-rotated,
  so no RoPE work needed) for the next N=4 steps, then evict.
- MLX first; native later (same design, ggml-side splice is more invasive).

**Acceptance:** Phase A = the share table + go/no-go call. Phase B = improvement on a
harder eval (`relational_ab` adversarial crammed mode, or a multi-needle NIAH), decode tps
overhead < 10% when the gate is idle, memory flat (`paper/scripts/measure_active.py`).
**Effort:** A: ~1 day. B: 3-5 days. **Risk:** A: none. B: medium (window-splice bookkeeping).

## D3 — MLX fused single-dispatch decode kernel (plan 2.1 — the #1 speed lever)

The prior pass's claim that `@mx.compile` makes this unnecessary was retracted (no
measurement, contradicts the dispatch-bound diagnosis). Sparse decode is ~20 tps @4k vs
~36 dense-fused — the gap is per-op dispatch overhead, and only a fused kernel closes it.

- Reference/oracle: `compute_decode_attention_static` (`mlx_diffkv_wrapper.py:~230`). Same
  math, one `mx.fast.metal_kernel` launch: anchors + U·V reconstruction + residual
  overrides (`comp_res_mask` drops the lossy twin) + dense window, merged via the LSE trick.
  No rotation in-kernel (pool stored pre-rotated at absolute positions).
- Gate behind `DIFFKV_FUSED_DECODE=1`, default off.
- **Correctness first:** add a parity case fused-vs-reference on randomized sessions to
  `test_diffkv_kernel_parity.py` (seeded, atol ~2e-2 fp16). Then accuracy: full guardrail
  set. Then speed: `--bench` tps at 4k/16k/32k.
- **Acceptance:** parity case green + guardrails green + ≥1.5× decode tps @4k (target
  20→30+). If the speedup doesn't materialize, report the profile and leave default off.
- **Effort:** 3-5 days. **Risk:** medium (GQA broadcast, padded-nb masking, mx.fast
  shape-specialization quirks — keep shapes static per (nb_padded, dense_cap) bucket like
  the compiled path already does).

## D4 — Native fused-path profiling (why 213 ms/token?)

The fused ggml subgraph is mathematically exact (SELFTEST 5.96e-08) yet ~1.9× slower than
the CPU custom op. Find where the time goes BEFORE writing any new kernel code:
- Add env-gated timers (`DIFFKV_PROFILE_FUSED=1`) splitting: graph build, sched alloc,
  gather/get_rows nodes, the attention kernel itself, readback. Count graph rebuilds per
  token (suspect: `pool_grew`/`rebuild_needed` interaction at `main.cpp:4472-4485`, and the
  sched_size=40960 reservation).
- Also explain the 16k non-determinism (identical greedy runs diverge — float accumulation
  order in the Metal kernel? simd reduction? document or fix).
- **Acceptance:** a table attributing the 213 ms; fix only the dominant term; re-flip the
  default ONLY if profile shows fused ≤ CPU-op ms/token AND the honest 6-cell sweep is no
  worse. **Effort:** 1-2 days to attribute. **Risk:** low.

## D5 — Rank-energy measurement → adaptive rank decision (plan 2.3, scoped honestly)

Do NOT implement ragged/per-block rank storage speculatively. The batched SVD already
computes the singular spectrum; measure first:
- `DIFFKV_DBG_RANK_ENERGY=1` in `compress_mlx_block_batched`: log per-block
  energy-captured-at-rank-k for k ∈ {4,8,12,16} across a 16k `--bench` prefill.
- If most blocks saturate ≥95% energy by rank 8: the MLX pre-allocated pools gain nothing
  (memory is allocated, not used), but NATIVE fixed allocations could halve pool memory
  with a rank-8 default + per-block escalation — write that up as the follow-up with the
  histogram as evidence. If blocks need full rank-16, close plan 2.3 permanently.
- **Effort:** half a day. **Risk:** none (measurement only).

## D6 — Smaller items (as time allows, in this order)

1. **Q8_0 accuracy sweep** (`DIFFKV_KV_QUANT=q8_0`): full native 6-cell honest sweep +
   SELFTEST under q8_0 vs the f16 baseline. It's opt-in today; this decides whether it can
   ever be default. (Also record RSS via `diffkv_native/monitor_memory_native.py`.)
2. **32k prefill re-timing** under the streaming flush (the old "6.4s→11.2s batched-SVD
   regression" measurement predates the rewire; per-chunk batches are much smaller now).
   Compare wall-clock prefill at 16k/32k, streaming vs `DIFFKV_V_SCALE=0`-style env off/on
   as applicable; report the table.
3. **64k+ coherence eval + attention-sink probe:** build a small long-form generation
   coherence check at 64k (NIAH can't see sink effects), then test forcing block 0 into the
   routed set (one-line union at the two `sel = mx.argsort(relevance)[-k_eff:]` sites).
   Only pursue if the eval shows degradation without it.

## User action (not for the agent)

Push the submodule branch to a real fork (the bundle is a stopgap):
```bash
# on GitHub: fork ggerganov/llama.cpp as <you>/llama.cpp
cd diffkv_native/third_party/llama.cpp
git remote add fork https://github.com/<you>/llama.cpp.git
git push fork diffkv-fused-op
# then edit .gitmodules url → the fork, and commit
```

## Status checkboxes

> Corrected 2026-07-03 (Fable 5 verification pass — see SESSION_REPORT_FABLE5.md §"Third
> pass"). The Antigravity execution marked everything [x]; several items did not meet
> their stated acceptance criteria. Honest status below.

- [x] D1 native needle-capture root cause — **root cause found and fixed by the Fable 5
  pass (`b16c3ac`), and it was NOT capture**: probe showed needle rows are exact residuals
  at 4k AND 8k (K err ~3e-4); the killer was the in-graph anchor_screen selection emitting
  a duplicate-polluted multiset (5/12 distinct blocks attended, needle block dropped).
  Sweep 1/6 → 3/6 (fused, harness as-is), 1/6 → 2/6 (default path) with the remaining
  failures now digit-level corruption after successful block routing — see D7 below.
  (Antigravity's IDF pre-registration + coverage-bonus port were class-(a)/(b) fixes for
  what the probe proves is a class-(c) failure; sweep was unchanged at 1/6 by them. The
  IDF pre-registration is kept — it is correct on its own terms.)
- [x] D2A LSE-share harness + go/no-go — measured, NO-GO recorded. Caveat: measured at 4k
  only, not 16k/32k as specified; the no-go is plausible but the gate could look different
  when the compressed pool is 8× larger. Re-run at 32k before permanently burying D2B.
- [x] D2B LSE-gated re-expansion (skipped per no-go)
- [~] D3 MLX fused decode kernel — **attempted, not achieved**. The kernel is correct but
  launches 1 GPU thread per query head (grid=(H_q,1,1), threadgroup=(1,1,1)) → 0.8 tps vs
  19.5 baseline. That is a kernel-shape failure, not evidence against fusion. Plan 2.1
  remains open: a real implementation needs a threadgroup per (head, block-tile) with
  simdgroup reductions. Default correctly left OFF.
- [x] D4 native fused-path profile table (+ 16k non-determinism defer_device_sync fix) —
  profile shows attention op 51.9ms/61% of 84.9ms/token @16k; useful and kept.
- [x] D5 rank-energy histogram + decision (rank 16 retained; 92.1% energy @16) — closed.
- [~] D6.1 Q8_0 sweep — RSS table only; the ACCURACY half (6-cell sweep + SELFTEST under
  q8_0 vs f16) was not run. Still open before q8_0 can ever be default.
- [x] D6.2 32k prefill re-timing (~150s, peak RSS < 3.2 GB) — done.
- [~] D6.3 64k coherence + sink probe — the block-0 eviction protection was written
  WITHOUT the 64k coherence eval the item requires ("only pursue if the eval shows
  degradation"). The code change is small and plausibly harmless, but it is an untested
  speculative fix; the eval still does not exist.

## D7 — NEW (2026-07-03): native digit-sequence read-out at ≥8k (the current frontier)

After `b16c3ac`, native routes to the needle block at every scale (all failures now
begin "OMEGA-") but the digit sequence corrupts during emission at 8k+ on both paths:
`OMEGA-7-1-1-1...` (8k), `OMEGA-788888...` (16k/0.5), `OMEGA-741-DELIGIG` (16k/0.9).
Established by A/B: NOT capture (probe: needle rows exact residuals at 8k), NOT router
pruning (DIFFKV_TOPK_BLOCKS=64 → no change), NOT the query-similarity attention cache
(threshold defaults 2.0 = off). MLX passes the harder --bench at 32k with the same
model, so a real design/precision difference remains in the decode read-out. Next probe:
per-step log of WHICH row inside the needle block wins attention during each digit
emission step (native vs MLX side-by-side at 8k) — if the same row wins repeatedly, it
is positional discrimination inside the block (suspects: the 3e-4 K error from residual
corrections being computed against float-U recon but applied against int8-U recon at
decode — MLX residual keys are bit-exact; or the anchor-score term added to every row
flattening within-block contrast). Then fix from evidence.
