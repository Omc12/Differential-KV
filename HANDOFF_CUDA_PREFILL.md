# DKV CUDA — Handoff (prefill router alignment)

Give this whole file to the next agent. It is written to be read top-to-bottom
once, then used as a reference. Everything here is measured, not assumed; where
something is a hypothesis it says so.

---

## 0. STANDING RULES — read before touching anything

1. **Never add `Co-Authored-By` or any co-author trailer to commits.** Commit
   messages should be detailed and explain *why*, not just what.
2. **Never edit MLX.** `ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py` is the
   known-good REFERENCE implementation. Read it, never change it. The job is to
   find every CUDA/MLX difference and make CUDA match. The only exception is
   where CUDA is genuinely better for performance AND produces the same output.
3. **Don't patch around problems — fix roots.** A fix that only half-works is a
   signal the root is elsewhere.
4. **No test loops.** Every GPU run must answer a specific question decided
   *before* the run, with the reading of each possible outcome written down
   first. This codebase has burned entire sessions on runs whose result could not
   discriminate anything.
5. **Don't rely on memory or on comments — check the code.** Several comments in
   this repo describe behaviour that is no longer true.
6. **State a check's COVERAGE, not just its result.** "I verified the inputs
   match" cost several sessions when it turned out to mean 8 of 256 elements.

---

## 1. WHERE THINGS STAND

`python colab/validate_cuda_dkv.py --long` (Qwen3.5-2B, 9 NIAH cases):

| config | result |
|---|---|
| default | **8/9** — `32k@depth0.9` gives `'None'`, deterministic at temp 0 |
| `DKV_SPARSE_PREFILL=0` | **9/9** ✅ |
| MLX reference (`mlx_needle_parity.py --long`, on a Mac) | 9/9 |
| dense control (same weights, DKV disengaged) | 9/9 |

So the failure is DKV's, not the model's or the prompt's, and it is **in prefill**.

`DKV_SPARSE_PREFILL=0` is a *diagnostic*, not the fix — it restores O(L²) prefill
and discards the entire point of sparse prefill. **Your job is to make sparse
prefill correct, i.e. match MLX.**

---

## 2. THE ROOT CAUSE

CUDA runs block-sparse attention **during prefill** (`DKV_SPARSE_PREFILL` default
`"1"`, `ACTIVE_RUNTIME/runtime/dkv_attention.py:412`). Each chunk attends only its
routed blocks plus a recency window. When prefill routing misses the needle's
block, **the model's own hidden states never absorb that sentence.** The query at
decode is then not "degraded by noise" — it is the query of a model that
effectively never read the needle.

This is why ~8 rounds of decode-side fixes produced byte-identical output: they
were all downstream of a defect that had already happened.

### The specific divergence — `dkv_attention.py` ~line 497

```python
anchor_ks = torch.stack([b.anchor_kv[0, 0] for _, b in valid], dim=0)  # [nb,H_kv,D]
q_repr    = chunk_q[0].mean(dim=(0, 1)).float()                        # [D]
scores    = torch.einsum("nhd,d->nh", anchor_ks, q_repr).mean(dim=1)   # [nb]
top_idx   = torch.topk(scores, k=k_eff).indices.tolist()
```

Two departures from MLX:

**(a) ANCHOR-ONLY SCORING.** An anchor is a single token — the block's first.
A needle at within-block offset 232 of 257 contributes *nothing* to it, so this
router is structurally blind to content buried deep in a block. MLX scores
`_block_relevance_residual` = **anchor + the block's top-R exact residual keys**
(`mlx_dkv_wrapper.py:1156`), and the residuals are precisely each block's
highest-error / most distinctive tokens — which is what a random code is.

This is the SAME bug already fixed on the decode side in `e48cc31`
(`route_blocks_relevance` scored bare `q·rk` on anchor-relative residuals without
adding `s_anc` back, so `maximum(s_anc, q·rk) → s_anc` always ⇒ anchor-only).
**Prefill never received that fix.**

**(b) MEAN-POOLED QUERY.** `chunk_q.mean(dim=(0,1))` collapses all heads *and* up
to 1024 chunk tokens into one `[D]` vector. Retrieval is head-specialised;
averaging a retrieval head with seven others erases exactly the signal that finds
a needle. MLX scores per-head and reduces with max.

### The fix

CUDA already has the MLX-aligned router: `route_blocks_relevance` in
`ACTIVE_RUNTIME/native_core/srl/query_router.py` (the DECODE router, MLX-matched
and fixed). **Make prefill call it instead of the ad-hoc scoring above.**

Watch for: the decode router takes per-head `q` and pool-resident blocks; prefill
has `chunk_q` `[1, H, T, D]` and `history_blocks` objects. You will need a small
adapter (per-head q, probably `max` over the chunk's tokens rather than `mean` —
a needle-matching token must not be averaged away). Verify the residual keys are
read in the same anchor-relative EXACT form the decode path uses
(`_exact_keys_enabled`), or the scores will be silently wrong.

### Parameters are NOT the divergence — don't waste a run there

`DKV_SPARSE_PREFILL_MIN` 2048, `..._WINDOW` 1024, `..._KMIN` 8 already claim MLX
parity, and `..._FRAC` is 0.25 vs MLX's 0.05 — CUDA attends **strictly more**
blocks than MLX. The block *count* is not the problem. The block *scoring* is.

---

## 3. VERIFICATION PLAN (pre-decide the reading)

1. Baseline on the new box first: `python colab/validate_cuda_dkv.py --long`
   must reproduce **8/9** with `fallback_count=0`. This suite has been
   invalidated before by config drift (a validator that didn't apply
   `BEST_DECODE_DEFAULTS`; an env reset that bumped `transformers` to a version
   the track can't use). Do not read any new number until the baseline matches.
2. `DKV_SPARSE_PREFILL=0` must give **9/9**. Confirms the diagnosis transferred.
3. After the router fix, default config must give **9/9** — *and* prefill must
   still be sparse. Check the sparse path actually ran; if your change silently
   makes `k_eff >= len(routable)` the function returns all blocks and you have
   accidentally reproduced `=0` while believing you fixed routing.
4. Re-check throughput. The point of sparse prefill is speed; a "fix" that
   attends everything is the workaround wearing a disguise.

---

## 4. WHAT IS ALREADY PROVEN CORRECT — do not re-investigate

Each of these was established by measurement this session. Re-testing them is
how you lose a day.

* **The stored key is EXACT.** `colab/probe_residual_values.py`:
  `anchors_K + residual_K_values` vs `RoPE(k_proj(h),pos)` = **cos 1.0000**,
  rel_err 3e-4, at every layer, *identically on the failing 0.9 and passing 0.5*.
* **Decode routing works** — the needle's block ranks 0–1 of 16, its row is
  unmasked, its offset resolves correctly.
* **Sparse-half math is MLX-equivalent** — `delta_s + s_anc` with residual twins
  masked, `1/√D` applied equivalently, padding masks matching.
* **The merge is not the cause** — the MLX partition plus a genuinely adaptive
  `auto` moved nothing; remat (a single unbiased softmax with no merge at all)
  fails identically; MLX passes 9/9 at its own `0.0` default.
* **Also eliminated:** residual capacity (128, already MLX-equal) and selection,
  attend-all, TF32, the RoPE clamp (instrumented — never fires), and the rotation
  convention (documented A/B in `pool_stores_rotated_k`'s docstring:
  `32k@0.9 unchanged, still 0/3` both ways).
* **Per-layer output cosine vs dense is a DEAD observable** — the *passing* case
  measures a worse cosine (0.276) than the failing one (0.291).

---

## 5. TRAPS THAT MADE EXPERIMENTS VACUOUS

1. **`DKV_RESIDUAL_EXACT_ROPE` is dead on the remat path by construction.**
   `do_rot = (... and not pool_stores_rotated_k())`, while `_remat_attend`
   *declines* unless `pool_stores_rotated_k()` is true. Toggling the flag is
   byte-identical — that is evidence the flag is dead, **not** evidence about
   rotation.
2. **The rotation A/B was already in the source** —
   `triton_fused_decode.py:237-310`. Read docstrings before designing a run.
3. **A `tl.constexpr` declared on a Triton kernel but not passed at the call site**
   raises `TypeError`, which the `try/except` swallows into a silent PyTorch
   fallback. Production then runs something different from what you are testing.
   This already happened once with `S_MAX`. Always confirm `fallback_count=0`.
4. **Swallowed exceptions generally.** `except Exception: # SRL failure is
   non-fatal` hid a router that never ran for SEVEN consecutive "fixes."
   **When N changes in a row do nothing, verify the code RUNS before making an
   (N+1)th change.**

---

## 6. PROBE DISCIPLINE (learned the hard way, twice each)

* **Compare at decode step 0, not the last step.** By the last step the two runs
  have emitted different answers, so their hidden states differ *because* of the
  wrong answer. That is an effect, not a cause. Step 0 has identical token history
  in both runs by construction.
* **Align hidden-state comparisons on ABSOLUTE POSITION, never list index.** The
  runs emit different token counts once they diverge.
* **Layer 0's input is the token embedding and MUST read cos 1.0000.** A `0.0319`
  there is misalignment, not a discovery. *When a measurement violates an
  invariant, the measurement is wrong until proven otherwise.*
* **Always read the failing depth against the passing 0.5 control.** A number
  identical in both explains nothing. This trap was hit twice.
* **State coverage.** "The block is routed" from a probe that prints once is a
  statement about token 0, not about the token that produces the answer.

---

## 7. TOOLS

| file | what it answers |
|---|---|
| `colab/validate_cuda_dkv.py --long` | the 9-case NIAH suite; `--dense` for the control |
| `colab/probe_query_vs_dense.py` | DKV's q vs dense's q against the same exact key. **This is what found the bug.** `--mode dkv`, then `--mode dense`; `--mode compare` re-analyses caches with no GPU |
| `colab/probe_residual_values.py` | is the stored key correct? (`--depth 0.5` = passing control) |
| `colab/probe_needle_block.py` | is the needle selected into its block's residual set? |
| `colab/mlx_cuda_parity.py` | side-by-side MLX/CUDA harness — this found the dead router that reading missed 7 times |
| `ACTIVE_RUNTIME/tests/` | CPU tests; run before any GPU turn |

Known rough edge: `probe_query_vs_dense.py --mode compare` throws
`TypeError: iteration over a 0-d tensor` on caches written before the
position-alignment change. Delete `/tmp/dkv_qprobe_*.pt` and re-run both passes.

---

## 8. OPEN ITEMS

* **Prefill router alignment** — section 2. The main task.
* **64k** — untested. Depth 0.9 there puts the needle in the same relative
  position, so the same failure is *expected* but unverified. Re-check after the
  fix. Note the routed-row count does not grow with context (K=16 regardless), so
  "more context = more competitors" is NOT the mechanism.
* **`DKV_RESIDUALS_IN_DENSE`** (commits `53d928d`, `6dab025`) — real MLX parity:
  exact residual rows belong in the DENSE half (`mlx_dkv_wrapper.py:1031`), and
  with them in the sparse half `DKV_SPARSE_BIAS=auto` had the wrong sign and was
  pinned near +2.0, then disabled outright. Verified safe (no regression at bias
  0.0 or `auto`) but it does **not** fix `32k@0.9`. Default OFF. Ship it only on
  its own merits, with its own measurement.
* **`DKV_REMAT_CACHE`** — stays default OFF. Its old speed numbers came from a
  broken 1.5B build; on the 2B it measured 21.1 vs 20.1 ms/token, inside noise.
* Prefill throughput work generally — the reason sparse prefill exists.

---

## 9. ARCHITECTURAL NOTE — why this class of bug exists on CUDA and not MLX

MLX compresses as a **side-effect of the model's own forward pass**:
`keys_rot = self.rope(keys, offset)` (`:4565`), ordinary attention with those
keys, then `manager.ingest_streaming(keys_rot, ...)` (`:4613`). Its hidden states
are structurally identical to dense's, and there is no separate prefill
implementation that *can* drift.

CUDA **replaces the attention implementation** (`DKV Attention Interception
Applied`) in prefill as well as decode. That is what allows prefill to produce
different hidden states than dense — the precondition for this entire bug.

If you find yourself repeatedly patching prefill to behave like dense, the
deeper fix is to make CUDA's ingest a side-effect of normal attention the way
MLX's is, rather than a substitute for it. That is a large change; do not start
it without agreement.
