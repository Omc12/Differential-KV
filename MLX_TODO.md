# MLX work order

**Read this top to bottom once, then use it as a checklist.** Written
2026-08-23 from a Windows/CUDA box (RTX 4070 SUPER). Nothing here was run on
Apple silicon — MLX cannot be installed on this machine at all, so every item
below is either *unverified on MLX by construction* or *a CUDA finding that
needs porting*. Where something is a hypothesis it says so.

Companion documents, in the order worth reading:

* `MLX_PORT_FROM_CUDA.md` — what CUDA learned that MLX needs, with line
  numbers into `mlx_dkv_wrapper.py`. **§1 is the big one.**
* `HANDOFF_CUDA_PREFILL.md` §0 — standing rules. The one that governs this
  file: **never edit `ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py`.** It is the
  known-good reference implementation. Item 3 below is the single exception
  being *proposed*, and it is written as a proposal for that reason.

---

## 0. Setup and the baseline check

```bash
python -m venv dkv_venv && ./dkv_venv/bin/pip install -r ACTIVE_RUNTIME/requirements.txt
./dkv_venv/bin/pip install mlx
```

Confirm the MLX-only tests collect — four of them are skipped on CUDA and have
therefore not run in this work at all:

```bash
python -m pytest ACTIVE_RUNTIME/tests/test_unrotated_pool.py ACTIVE_RUNTIME/tests/test_residual_budget_clamp.py ACTIVE_RUNTIME/tests/test_dkv_kernel_parity.py ACTIVE_RUNTIME/tests/test_decode_cache_fused_parity.py -q
```

Then the whole suite. On CUDA it is **216 passed, 6 skipped, 0 failed** with
five files excluded (two MLX-only, one Windows-path, and three added by the
MLX/Mac port that import `mlx`). On a Mac they should all collect:

```bash
python -m pytest ACTIVE_RUNTIME/tests -q
```

**Record the number before changing anything.** Several items below are only
interpretable against a known-green baseline.

---

## 1. HIGHEST VALUE — run `logit_fidelity_mlx.py` and settle the open question

```bash
python colab/logit_fidelity_mlx.py --arms dense baseline --ctx 8192
```

**Why this is first.** On CUDA, DKV's first-step next-token distribution sits
absurdly far from a dense control at 8k on Qwen2.5-1.5B:

| arm | top-1 agree | KL(dense‖arm) | dense-top1 rank | top-5 overlap |
|---|---|---|---|---|
| dense (self-check) | 5/5 | 0.00000 | 0.0 | 5.0/5 |
| **CUDA DKV baseline** | **0/5** | **10.579** | **1254.6** | **0.2/5** |

DKV is not tracking dense *at all* at that operating point. That single fact
explains why every recall-based accuracy attempt on CUDA failed to discriminate
anything — the arms were being compared on top of a baseline already off the
map — and it is why shared bases shipped opt-in rather than as a default.

**Pre-decide the reading before you run it:**

* **MLX baseline KL small (say < 1) and top-1 agreement high** → the instrument
  has real resolving power on MLX, *and* CUDA's gap is a CUDA defect rather than
  the price of compression. That makes it the highest-priority CUDA bug in the
  repo, and it would be found by bisecting CUDA against MLX layer outputs
  (`colab/probe_mlx_layer_output_diff.py` and its CUDA twin already exist).
* **MLX baseline KL also ~10 with top-1 rank in the hundreds** → this is what
  DKV compression costs at 8k on a 1.5B model on both engines, the CUDA number
  is not a defect, and any accuracy claim about *any* DKV feature at this
  operating point is unmeasurable without a larger model or shorter context.
* **Anything in between** → report the number, do not round it to either story.

Run `--ctx 4096` as well. On CUDA, 4k is where dense recalls the needle but DKV
does not compress at all (pool 0.0 MB), so the two ranges never overlapped;
MLX's block sizing differs (default `block_size` 1024) and may overlap.

---

## 2. Port shared low-rank bases (`MLX_PORT_FROM_CUDA.md` §1)

The one feature merged from this work. **−23.6% pool** on CUDA at retained
delta energy 0.969 with zero forced joins. `MLX_PORT_FROM_CUDA.md` §1 has the
math, the exact `mlx_dkv_wrapper.py` line numbers for every slot-indexed read
that must be redirected, and §2 has eight traps that each cost a debugging pass
on CUDA. Do not start without reading both.

The five things most likely to bite on MLX specifically:

1. **`:3605`, the sliding eviction.** `comp_VK[:-1] = comp_VK[1:]` shifts basis
   ROWS as if they were blocks. Under sharing that renumbers every group at
   once. CUDA has no equivalent code path, so this trap is MLX-only and is not
   covered by any CUDA test.
2. **MLX has more slot-indexed reads than CUDA did** — `:993`, `:1533`, `:2289`,
   `:2983`, `:3373`, `:3405`, `:4007`, `:4229`, and writes at `:2943`, `:3338`,
   `:3906`. Miss one and it reads another group's basis. On CUDA the same class
   of miss surfaced as a CUDA device-side assert; on MLX it may just be wrong
   numbers, so **assert, do not rely on a crash**.
3. **Assign in compress, not at write time.** Residual selection scores
   `delta − recon`, and under a shared basis the recon the decoder rebuilds
   comes from the *group* basis. MLX's seam is
   `compress_deferred_prefill_blocks`, before the `capture_scores` ranking at
   `:2846`.
4. **Checkpoint shapes at `:2081-2082` change.** Version them or old
   checkpoints load onto a differently-shaped store.
5. **Do not put it on a 4-bit-KV preset.** See §2 of the port file: on CUDA's
   `low` (`kv_quant=q4_0`) quantisation noise takes voluntary joins to **zero**
   and retained energy to 0.685, *at identical pool MB*. Report
   `basis_stats()['joined']`; `0` is the signature.

**Port the CPU-side tests too** — `ACTIVE_RUNTIME/tests/test_basis_group.py`
(27 tests, pure math, no CUDA) should translate to `mx` almost unchanged and is
the cheapest way to know the projection math is right before touching the pool.

---

## 3. PROPOSED MLX EDIT — the `_pending_query` gate (read the standing rule first)

**This is the one place this file proposes changing `mlx_dkv_wrapper.py`, and
it is a proposal, not an instruction.** The standing rule says never edit MLX.
Weigh this against it and decide.

CUDA had two faults here; MLX has the second one.

`current_query_tokens` is read as a **lexical query** by the lexical router and
the decode-time `query_toks` set. MLX sets it correctly *when the pin is
filled* (`:1978-1979`):

```python
pq = self._pending_query.pop(session_id, None)
srl_state.current_query_tokens = list(pq) if pq else list(token_ids[cached_len:])
```

But the write that fills `_pending_query` is gated on
`getattr(self.manager, "_factual_enabled", False)` (around `:5500`), which is
**off by default**. So on the default path the pin is empty and MLX takes the
whole-prompt fallback — naming every token in the document as part of the
question, IDF ~uniform, discriminating nothing. `query_span.py`'s own docstring
says that fallback "would pin nothing useful".

CUDA's fix (commit `0d3b12a9`) ungated it, because the consumers are *routing*,
not the factual store. Measured there: `current_query_tokens` 7986 → 11 tokens,
and per-block lexical overlap std 0.0818 → 0.1264.

**Before changing anything, measure whether it matters on MLX:** print
`len(srl_state.current_query_tokens)` after a single-turn `generate()`. If it
equals the prompt length, MLX has the same latent issue. Whether to fix it is a
judgement call against the standing rule — the safe alternative is to pass
`query_text=` explicitly at every call site instead of touching the wrapper.

---

## 4. MPS `_validate_this_step` double rotation (`dkv_attention.py:4746`)

Carried from `HANDOFF_CUDA_PREFILL.md` §8, still open. `_apply_rope_single` is
called on `dense_k_valid` inside `if _is_mps_decode:` — a path that only exists
on Apple silicon and therefore **cannot be tested from CUDA at all**.

The helper itself is now partial-RoPE-correct (it slices by `cos.shape[-1]`,
not by head_dim), and its sibling caller at `:1699` is guarded by
`_pool_rotated_k()`. The site at `:4746` is not obviously guarded. Under
`DKV_ROTATED_POOL=1` (the default) the dense-history keys may already be
rotated, in which case this rotates them a second time.

The handoff deferred it as "not production CUDA, fix it if that validation path
is ever trusted". On a Mac it *is* reachable. Either verify it is correct, or
guard it the way `:1699` is guarded — and if you touch it, add a test, because
a double rotation preserves norms (RoPE is orthogonal) and only corrupts
angles, so it fails silently.

---

## 5. Frame consistency — confirm MLX has no equivalent hole

CUDA had four prefill capture sites that stored K in the wrong frame. Fixed
this session (`ACTIVE_RUNTIME/tests/test_ingest_frame_consistency.py`, 4 tests)
by routing every site through one `_ingest_k(rot_k, unrot_k)` helper.

MLX captures keys **post-RoPE** by design (`mlx_dkv_wrapper.py:4448`), so it
likely has no equivalent split — but that is an inference from one line, not a
verification. Worth confirming that every MLX capture site agrees on the frame,
because the failure mode is silent: identical norms, wrong angles, and a
depth-graded quality loss that looks like a retrieval problem.

---

## 6. Measurements that only make sense on MLX

* **`colab/linkbench_mlx.py`, not the needle sweep**, for anything touching
  `DKV_ROTATED_POOL`. The needle sweep has no confusable distractors and cannot
  see the effect. Recorded in the README knob table.
* **`DKV_SVD_SEED` must be varied for any accuracy A/B.** Temperature-0
  replication is deterministic and proves nothing; at a fixed config this seed
  alone spans ~30 synthesis points. Use `colab/synthesis_power.py`, which is
  replicated, paired and interval-bounded.
* **`colab/bench_decode_interval_mlx.py`** for `DKV_DECODE_CACHE_INTERVAL`.

---

## 7. Explicitly NOT worth redoing

Each of these was built on CUDA, measured, and rejected. The full reasoning and
numbers are in `MLX_PORT_FROM_CUDA.md` under "Three sibling features". In
short, so nobody re-derives them:

| idea | why it is dead |
|---|---|
| Anchor-delta residual budget | Residual fill is already at the full budget on every block at `max_residual` **40 and 128** — zero blocks reach the tier the override targets, so it is inert. Not a budget-size problem. |
| Shared-basis chunk-graph edges | **Zero** new edges when basis groups are contiguous, which is what a real document produces; co-members are already neighbours. |
| Learned SRL router — scoring head | val recall@16 **0.599–0.609 vs 0.595** rule-based. Expected: the teacher label derives from the same q·k the rule computes. |
| Learned SRL router — K-head | Genuinely **14.3% faster at 32k** (paired, CI [-9.00, -6.91] ms, 8/8 rounds) but drops >50% of attention mass on the worst 5% of queries. Not fixable by a floor (needs 0.90, returning all the speed-up), by asymmetric retraining (predicts 115% of K), or by a guard (best signal separates at 0.38 sd). An **oracle static-K sweep** shows no K below 16 has an acceptable tail *at all*, so it is not a training problem. |

The one condition under which the K-head is worth revisiting: it was measured
on NIAH filler, which is adversarial for it — a needle is precisely a query
where one distant block carries everything. A **prose-synthesis** workload could
have a benign tail. That is a measurement, not an argument.

---

## 8. Quick reference — what this session changed on CUDA

| commit | what |
|---|---|
| `0bca51e8` | shared low-rank bases merged, opt-in |
| `0d3b12a9` | question-span pin — never worked on CUDA (see §3 above) |
| `be2897a7` | shared bases become a config key; `low` preset is the *wrong* home |
| `e6069eea` | README + port-file documentation |

CUDA suite at time of writing: **216 passed, 6 skipped, 0 failed**.
