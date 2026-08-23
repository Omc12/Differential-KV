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
| `758e0c26` | four more prefill capture sites stored K in the wrong frame (§5) |

CUDA suite at time of writing: **220 passed, 6 skipped, 0 failed**.

CUDA validation, Qwen3.5-2B, RTX 4070 SUPER, 2026-08-23 — **ALL CHECKS PASSED**
at every rung including the two the handoff listed as outstanding:

    2k / 8k / 32k / 64k, depths 0.0 / 0.5 / 0.9   3/3 recall each
    determinism at temperature 0                   1 distinct output across 3 runs
    Triton kernel used                             fallback_count=0

`32k@0.9` and `64k@0.9` — both previously expected to fail — now pass. **This
is the bar MLX should be held to**: `benchmarks/niah_recall.py` and
`colab/mlx_needle_parity.py` are the equivalents, and any MLX rung that does
not reach 3/3 with deterministic output is a real gap rather than a tuning
question.

---

# MLX RESULTS — 2026-08-23, M3 (8 GB), mlx 0.32.0 / mlx_lm 0.31.3

Run on Apple silicon against this file. Section numbers refer to the work order
above. `transformers 5.14.1`, `torch 2.13.0`, venv `dkv_venv`.

## 0. Baseline

The four MLX-only files collect and pass (17 tests). The full suite did NOT
start green:

| stage | result |
|---|---|
| as pulled | **263 passed, 9 failed**, 1 skipped |
| after the fixes below | **277 passed, 0 failed**, 1 skipped |

The nine split into two groups, and the split is the useful part: four failed in
ISOLATION (real defects, all of them macOS-only and therefore invisible to CUDA),
five failed ONLY in the suite (cross-test pollution).

**Four real, all unreachable from CUDA:**

1. `test_triton_combined.py` ×3 — `native_triton_sparse_attn_decode` returns a
   ZERO-WIDTH tensor for a zero-block step. The N==0 dense-only fix of
   2026-08-17 was written into the Triton branch's `else`, but the function
   short-circuits at its top on `if not HAS_TRITON:` and returns before ever
   reaching it. CUDA always has Triton, so CUDA always skipped the broken
   branch; Apple silicon and CPU always take it. Fixed by applying the same
   `_dense_only_attend` guard in the non-Triton early return.
2. `test_sparse_residual.py::test_metal_residual_and_fact_parity` — the test
   passes `block_indices` as `torch.long` while the Metal binding requires
   Int32 (the guard added with `39a4a9d1`, the fp16-RoPE-table fix). The test
   predates the guard and CUDA never runs it. **The production call sites had
   the same omission**: `39a4a9d1` converted `anchor_indices` and left
   `block_indices` raw at all four `decode_attention_metal` sites, so the
   conversion its own error message prescribes was never applied.
   `_DKV_HAS_METAL_ATTN` is True on this machine, i.e. that branch is live here.
   Fixed at the four call sites and in the test.

**Five from pollution — `test_residual_budget_clamp.py`**, which passes alone
and fails in the suite. Root cause is a global side effect, not a test bug:
`native_core/config.py:814,819` EXPORT preset-derived values into the process
environment with `setdefault()`, and `MLXKVBlockManager.__init__` resolves
`self.rotated_pool` from that same `DKV_ROTATED_POOL` (`mlx_dkv_wrapper.py:1783`,
default `"1"`). `setdefault` means the FIRST config built in the process wins for
the whole process, so one test building a config on a preset whose
`rotated_pool` is False silently flips the pool frame for every MLX manager
constructed afterwards. Measured at the clamp test: env `DKV_ROTATED_POOL='0'`,
`DKV_SVD_ENERGY='0.9999'`, and a freshly built manager reporting
`rotated_pool=False`. Fixed in `tests/conftest.py` by restoring those two keys
per test — the same isolation the `DKV_POOL_BUDGET_GB` pin already does, and for
the same stated reason.

*Method note:* the first two probes for this were change-detectors and printed
nothing, which reads exactly like "no pollution". Only a probe that printed the
VALUE (and built a manager to show what it resolved to) found it. State a
check's coverage, not just its result.

## 1. HIGHEST VALUE — logit fidelity. ANSWERED, and it is the first reading.

**The harness as written could not have answered the question, for two reasons,
both fixed in `colab/logit_fidelity_mlx.py`:**

* `MLXDKVWrapper(model_id=...)` — `config` is a required positional argument, so
  every arm died before loading. Nothing had run.
* The `dense` arm was `DKV_COMPRESSED_DECODE=0`. On MLX that forces exact
  full-KV attention **at decode only**; the prompt is still read through
  block-sparse PREFILL, which gates on `manager._sparse_prefill` (default on)
  and context length alone (`mlx_dkv_wrapper.py:5493`). This harness measures
  the FIRST decode step, which is a pure function of what prefill produced, so
  that arm would have compared DKV-prefill against DKV-prefill and reported a
  reassuringly small KL for a reason having nothing to do with fidelity. The
  real control is plain `mlx_lm` with DKV never loaded — the convention
  `mlx_needle_parity.py:104` and `linkbench_mlx.py:70` already use.

Two diagnostics were added because the headline number is otherwise ambiguous:
`blocks` (compressed blocks actually in the pool — a KL of 0 from an empty pool
means nothing) and `max|dlogit|` (which separates "very close" from
"bit-identical", the latter meaning the compressed path never ran).

**Qwen2.5-1.5B-Instruct, 5 depths, `mid` preset, dense = plain mlx_lm:**

| ctx | arm | top-1 agree | KL(dense‖arm) | dense-top1 rank | top-5 overlap | blocks | max\|Δlogit\| |
|---|---|---|---|---|---|---|---|
| 8192 | dense (self-check) | 5/5 | 0.0 | 0.00 | 5.0/5 | n/a | 0.0 |
| 8192 | **MLX DKV baseline** | **5/5** | **5.135e-12** | **0.00** | **5.0/5** | 168 | 3.125e-02 |
| 4096 | dense (self-check) | 5/5 | 0.0 | 0.00 | 5.0/5 | n/a | 0.0 |
| 4096 | **MLX DKV baseline** | **5/5** | **2.475e-13** | **0.00** | **5.0/5** | 56 | 3.125e-02 |

Against CUDA's baseline row — 0/5 agreement, KL 10.579, dense's top-1 at rank
1254.6 — on the same model, prompt and context.

`max|Δlogit|` is 3.125e-02 = 2⁻⁵ at both contexts, which is **exactly one fp16
ULP** at logit magnitude ~30. So the arms are not bit-identical (the compressed
path genuinely ran and moved the numbers) but differ by the smallest step fp16
can represent. 168 and 56 compressed blocks confirm the pool is not empty.

**This is §1's first pre-decided reading, and it is not marginal — KL is twelve
orders of magnitude below the "<1" bar that reading asks for.** Therefore:

* The instrument has real resolving power on MLX. Anything measured on top of
  this baseline is meaningful.
* **CUDA's gap is a CUDA defect, not the price of compression.** By this file's
  own §1, that makes it the highest-priority CUDA bug in the repo. Bisect CUDA
  against MLX layer outputs (`colab/probe_mlx_layer_output_diff.py` and its CUDA
  twin).
* The 4k range OVERLAPS on MLX (56 blocks compressed), unlike CUDA where 4k had
  an empty pool — the hypothesis in §1's last paragraph is confirmed, so 4k is a
  usable operating point here.

**Consequence for §2 that was not foreseeable when it was written:** shared bases
scored "no measurable harm" on CUDA *because the baseline was already off the
map*. On MLX the baseline is exact, so `mean_kept` 0.969 would be a MEASURABLE
fidelity regression from a perfect starting point. The trade is now visible for
the first time — which is an argument for measuring it, and against assuming it
is free.

## 3. The `_pending_query` gate — MEASURED, and MLX does NOT have the fault.

§3 asked for a measurement before any decision. Run on a single-turn 8k
`generate()`, default config:

    prompt tokens              7986
    _factual_enabled           False
    _pending_query has sid     False
    srl_state                  None
    _sp_instr_pin              False

The whole-prompt fallback **never executes**. The only writer of
`current_query_tokens` is `finalize_srl_index`, which returns at its first line
when the factual store is off (`mlx_dkv_wrapper.py:2113`), so `srl_state` is
never built; `get_srl_state` returns None and the routing consumers at `:5128`
and `:5176` cannot run either.

CUDA's fault was a DESYNC — the consumers were live while the producer was
gated. On MLX the producer and every consumer sit behind the same
`_factual_enabled` flag, so there is nothing to desync. **§3's proposed edit is
declined; the standing rule stands and `mlx_dkv_wrapper.py` was not touched.**

## 4. MPS `_validate_this_step` double rotation — CONFIRMED REAL, and it was
## never validation-only. FIXED.

Not a validator quirk: the PRODUCTION `_is_mps_decode` path has the same defect.
It hands `dense_k_assembled` to the Metal shader together with cos/sin and the
shader rotates it — but `dense_k_assembled` comes from
`assemble_dense_window_kv`, i.e. the blocks' `active_k`, which under
`DKV_ROTATED_POOL=1` is ALREADY post-RoPE. Both the shader and the validator
then rotate a second time.

**Reachable in production, and on a shipped preset.** `low` is the only preset
that keeps `rotated_pool=True` (`config.py:143`) and it also sets
`approximate_attn=True` on macOS (`config.py:37`) — which is exactly the
`_is_mps_decode` gate. So `low` on Apple silicon double-rotated its entire dense
window. CUDA cannot observe any of it; the branch requires MPS.

Fixed by guarding both sites on `_pool_rotated_k()`. Disabling rotation by
passing EMPTY cos/sin is the correct mechanism: `metal_runtime.mm:453` derives
`has_dense_rope` separately from `has_dense`, so the dense window is still
attended, just not rotated.

`tests/test_mps_dense_rope_guard.py` (5 tests) added as §4 requires. It asserts
against the singly-rotated reference and pins the guard text at BOTH sites,
because RoPE is orthogonal: a norm-based assertion passes while the bug is
present. One of its tests demonstrates exactly that, so the trap is recorded
rather than described. Negative control run: removing the guard fails the pin.

## 5. Frame consistency — CONFIRMED, and it is stronger than an inference.

MLX has exactly TWO K ingest sites — `capture_prefill_kv` (`:3799`) and
`ingest_streaming` (`:3954`) — with exactly one caller each (`:5519`, `:5089`).
Both callers pass `keys_rot` as the post-RoPE frame and `keys` as the pre-RoPE
frame under the same `manager.rotated_pool` condition, and **both sinks raise a
RuntimeError rather than storing the wrong frame** when the unrotated key is
missing. MLX therefore cannot have CUDA's four-wrong-frame-capture-site problem
structurally, not just by convention. (Those guards are what surfaced the §0
pollution above — they were doing their job.)

## 2. Shared low-rank bases — NOT STARTED. Blocked on a standing-rule decision.

The math and CPU-side tests are already present and green on this Mac:
`basis_group.py` plus `test_basis_group.py` (27), `test_shared_basis_pool.py`
(26) and `test_shared_basis_preset.py` (14) — 67 passing. What does NOT exist is
any MLX integration: `mlx_dkv_wrapper.py` contains **zero** references to shared
bases.

Everything remaining is inside `mlx_dkv_wrapper.py` — the allocation change at
`:2033-2034`, the ~12 slot-indexed reads and writes §1 of the port file
enumerates, the sliding-eviction trap at `:3605`, and the checkpoint version at
`:2081-2082`. That is precisely the file HANDOFF §0 rule 2 forbids editing, and
this work order's own §3 treats a ONE-LINE change to it as needing an explicit
decision. A twelve-site change to the same file cannot need less.

Deliberately left for the owner to decide, with the §1 result above as new
input: the port's fidelity cost is now measurable rather than hidden.
