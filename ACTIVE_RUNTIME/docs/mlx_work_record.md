# MLX work record — the 2026-08-23 cycle

Supersedes `MLX_TODO.md` and the second edition of `MLX_PORT_FROM_CUDA.md`, both
deleted from `main` now that every item in them is answered. They were work
orders written from a Windows/CUDA box by someone who could not run MLX at all;
this file is what happened when each item was actually measured on an M3.

What the CUDA side should do NEXT lives in `CUDA_TODO.md`, not here. This file
is the record.

The previous cycle's record is `cuda_port_record.md`, and its adoption bar still
governs: **a CUDA change ships on MLX only if it is measured to benefit MLX.**
Working on CUDA is a reason to try it, never a reason to ship it.

---

# MLX RESULTS — 2026-08-23, M3 (8 GB), mlx 0.32.0 / mlx_lm 0.31.3

Run on Apple silicon against the retired work order. Section numbers are ITS
section numbers, kept so the two can be read together if it is recovered from
git history. `transformers 5.14.1`, `torch 2.13.0`, venv `dkv_venv`.

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
* **CUDA's gap is a CUDA defect, not the price of compression.** By the work
  order's own §1, that makes it the highest-priority CUDA bug in the repo. Bisect CUDA
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

---

# ITEM 2 — SHARED BASES: PORTED ON BRANCH `mlx-shared-basis`

Done on a branch at the owner's direction, so `main`'s reference implementation
is untouched until review. `main` has only the macOS fixes above.

## What was built

`native_core/compression/basis_group_mlx.py` — the `mx` twin of the torch
`basis_group.py`. `tests/test_basis_group_mlx.py` (18) asserts the two AGREE on
identical inputs rather than testing the port against itself, plus the
properties that agreement cannot cover (orthonormal rows; kept == 1.0 against a
block's own basis; U' Vg really is the CLOSEST point in span(Vg), not merely a
point in it).

Pool integration in `mlx_dkv_wrapper.py`: `comp_VK`/`comp_VV` allocate
`ceil(frac * max_blocks)` BASIS ROWS, reached through a per-layer `basis_of`
map, with `basis_claimed` carrying trap 1's distinction. Every block-indexed
read goes through ONE accessor, `_basis_rows()`, so kernels and callers are
unchanged. `tests/test_shared_basis_mlx_pool.py` (12).

**The port file's line numbers are STALE** — it cites `:3605` for the sliding
eviction, which now lives at `:4256`, and every other cited line has drifted
similarly. Following them literally edits the wrong code.

## Two measured MLX-specific constraints

* `mx.linalg.qr` is **CPU-only** and raises on GPU, so `row_orthonormalize`
  pins a CPU stream and runs once per compress batch, never per block.
* mx `__setitem__` mutates the store in place and the caller sees it, same as
  torch. This was written up as the OPPOSITE first and corrected against a
  measurement; had it copied, every founded basis would land in a temporary and
  blocks would decompress from a store of zeros — right shapes, finite numbers,
  nothing raised.

## Paths that REFUSE rather than half-work

Three paths have no assignment seam, and in each an unguarded write stays IN
RANGE with only the contents wrong — no exception, no shape error:

* the **sliding eviction** (`_compress_block`) — `comp_VK[:-1] = comp_VK[1:]`
  shifts basis ROWS as if they were blocks, renumbering every group at once.
  MLX-only, exactly as the port file predicted; no CUDA test covers it.
* the **streaming single-block compress** path.
* the **multi-layer batched compressor**, which declines and hands off to the
  per-layer one that does have the seam (slower; the trade is deliberate while
  this is opt-in).

Correction-form residuals are refused at construction, and 4-bit KV warns.

## Measured — Qwen2.5-1.5B, 8k NIAH, `mid`, seeds pinned

**Memory (deterministic), 28 layers, block 1024, 64 slots:**

| arm | V-store | total pool | vs baseline |
|---|---|---|---|
| baseline | 88.08 MB | 412.79 MB | — |
| `frac=0.50` | 44.04 MB | 368.75 MB | **−10.7%** |
| `frac=0.25` | 22.02 MB | 346.73 MB | **−16.0%** |

**Not CUDA's −23.6%, and the gap is structural rather than a porting defect.**
MLX's `block_size` is 1024 against CUDA's 257, so `comp_U` (block_size−1 × rank
per slot) dominates a slot here and V is a much smaller share of it. The V store
itself halves exactly as designed.

**Grouping — and this is the finding that matters:**

| config | founded | joined | forced | mean_kept |
|---|---|---|---|---|
| MLX, block 1024, frac 0.50 | 140 | **0** | 28 | 0.934 |
| MLX, block 256, frac 0.50 | 560 | **10** | 186 | 0.903 |
| CUDA, block 257, frac 0.50 | — | **463** | **0** | 0.969 |

`joined == 0` is precisely the degeneracy signature the port file names — but
here it is NOT the 4-bit cause it documents. `kv_quant` is f16 throughout, and
`DKV_V_SCALE=0` vs `1` changes nothing (10 joins either way), so the per-block
v_scale undo is not destroying the alignment either.

At MLX's own block size, **adjacent blocks essentially never clear the 0.90
retained-energy bar**. Shrinking to CUDA's block size and reproducing its exact
block count (756) recovers only 10 voluntary joins against CUDA's 463. So the
premise the whole feature rests on — "adjacent blocks of one document share a
subspace, so the saving is close to free" — **does not reproduce on MLX**. What
the memory saving buys here is bought with FORCED lossy joins, not with dedup.

**Fidelity cost — none that this instrument can resolve:**

| arm | top-1 | KL(dense‖arm) | rank | top-5 | blocks | max\|Δlogit\| |
|---|---|---|---|---|---|---|
| baseline | 5/5 | 5.135e-12 | 0.00 | 5.0/5 | 168 | 3.125e-02 |
| `frac=0.50` | 5/5 | 5.135e-12 | 0.00 | 5.0/5 | 168 | 3.125e-02 |
| `frac=0.25` | 5/5 | 5.135e-12 | 0.00 | 5.0/5 | 168 | 3.125e-02 |

Identical to every printed digit, which is the harness's own "arms are inert"
warning condition — so it was checked directly rather than reported as a
result. Same prompt, sharing off vs on, in one process: the logit SUMS agree to
8 decimals but the **byte hashes differ**, and `basis_stats()` reports
`enabled: true, founded 140, forced 28, mean_kept 0.9336`. Sharing really is
running and really does move the logits; the movement is just smaller than one
fp16 ULP, which is where `max|Δlogit|` saturates. **Read the table as "below
this instrument's floor", not as "bit-identical".**

## Open, and deliberately not done

1. **The premise gap above is unexplained.** Same block count and same document
   give CUDA 463 voluntary joins and MLX 10. Block size and v_scale are ruled
   out. Until it is understood, `frac` on MLX is a lossy-compression dial, and
   should be argued for on that basis rather than as free dedup.
2. **Eviction, streaming compress and the batched multi-layer compressor** all
   refuse instead of working. The pool must be sized so eviction never fires.
3. **Checkpoint versioning is documented, not implemented.** A checkpoint
   written with sharing on has a different `comp_VK` row count and is
   meaningless without `basis_of`; loading it into a one-row-per-block store
   would broadcast cleanly and be wrong. Needs a real version field before this
   ships.
4. **Peak-memory (as opposed to pool-size) effect is unmeasured.** `_basis_rows`
   materialises one row per block at read time, so the transient is
   full-size even though the persistent store is not. The pool numbers above
   are exact; a peak-RSS claim is not supported by them.
5. **No long-context or recall validation.** `mlx_needle_parity.py --long` and
   `linkbench_mlx.py` have not been run against the sharing arm.

---

# ROOT CAUSE FOUND — why MLX grouped 10 blocks where CUDA grouped 463

The port reproduced shared bases' memory saving but not its GROUPING, and the
first write-up left that open. It is now closed, and the cause is one line of
configuration rather than anything in the port.

## The finding

**Shared bases compare SUBSPACES, and RoPE rotates every key by its ABSOLUTE
POSITION.** Two blocks holding the same text at different offsets therefore have
subspaces rotated apart, and the retained-energy test — which is exactly a
weighted average of cos² of the principal angles between them — collapses.

MLX's pool stores POST-RoPE keys by default (`DKV_ROTATED_POOL=1`). CUDA's does
not, on any preset where sharing is enabled.

Same document, same block size, `frac=0.50`, Qwen2.5-1.5B @8k, only the pool
frame differing:

| pool | best-partner retained energy | founded | joined | forced | mean_kept |
|---|---|---|---|---|---|
| rotated (MLX default) | mean 0.486, max 0.541, **0/27** over 0.90 | 560 | **10** | 186 | 0.903 |
| unrotated | mean 0.972, median 1.000, **26/27** over 0.90 | 236 | **520** | **0** | 0.968 |
| *CUDA, for reference* | — | 293 | *463* | *0* | *0.969* |

The unrotated row lands on CUDA's numbers. That is what identifies rotation as
the entire mechanism rather than one contributing factor.

At MLX's own default block size of 1024 it also works once unrotated: founded
89, joined 73, forced 6, mean_kept 0.968.

## What was eliminated first, and why each mattered

Each of these was a plausible story that would have been wrong, and each is
cheap to re-derive if someone doubts the conclusion:

* **Block size.** MLX 1024 vs CUDA 257. Shrinking MLX to 256 reproduced CUDA's
  exact block count (756) and moved voluntary joins only 0 → 10. Real but tiny.
* **Zero padding to the store rank.** CUDA slices to the REALISED rank before
  scoring; MLX passed the padded array. Scoring both ways gives identical
  numbers to four decimals (0.4862 either way) — the zero rows are inert, as
  the math says they should be.
* **The v_scale undo.** `DKV_V_SCALE=0` vs `1`: 10 voluntary joins either way.
* **Energy truncation.** CUDA selects rank by an `svd_energy` target while MLX
  truncates at a fixed rank, so "MLX keeps low-energy directions that do not
  align" was the leading hypothesis. It is BACKWARDS: restricting to the top-r
  energy directions makes alignment strictly WORSE (rank 24 → 0.486, rank 8 →
  0.387, rank 2 → 0.145). The top direction holds 66% of a block's delta energy
  and still does not align across blocks — which in hindsight is exactly the
  signature of a per-position rotation.

## Method notes worth keeping

* **The first probe measured one pair and called it a distribution.** MLX
  compresses 1–2 blocks per `_assign_shared_basis` call (392 calls at 8k), so
  capturing "a batch" captured two blocks. The blocks of a layer have to be
  ACCUMULATED across calls before any pairwise statistic means anything.
* **`joined == 0` is a signature with more than one cause.** The port file
  documents it as the 4-bit-KV signature; here it fired at f16 for an unrelated
  reason. Treat it as "grouping failed, find out why", not as a diagnosis.

## What changed as a result

`MLXKVBlockManager.__init__` now REFUSES `DKV_SHARED_BASIS=1` on a rotated pool,
with `DKV_SHARED_BASIS_ALLOW_ROTATED=1` as an escape hatch so the bad
configuration stays measurable. It refuses rather than warns because the failure
is silent and expensive: **pool MB is identical either way**, so a rotated run
reports the full memory win while having force-joined nearly every block.

`DKV_ROTATED_POOL=0` in turn requires `DKV_DECODE_CACHE=1`, which the pool
already enforced with its own error.

## The cost side, which is now the open question

The unrotated pool is not free — `dkv_attention.py`'s remat-cache decline
documents it as **43% slower decode on Qwen3.5-2B and 137% on Qwen2.5-1.5B**,
because declining remat disables that cache for the whole session. So shared
bases on MLX are currently a memory-for-decode-speed trade, not a free win, and
the two knobs are coupled:

    shared bases  -->  unrotated pool  -->  no remat cache  -->  slower decode

Anyone pushing this further should attack the remat decline first; that file
already sketches the fix (plumb the dense window's absolute token positions
into the function and rotate with `_partial_rope_apply`), and it would make the
unrotated pool nearly free — which would benefit the unrotated pool's OWN
retrieval win (40/48 → 47/48 on distractor retrieval) quite apart from bases.

## Fidelity, measured in the correct configuration

| arm | top-1 | KL(dense‖arm) | rank | top-5 | blocks | max abs dlogit |
|---|---|---|---|---|---|---|
| baseline (rotated) | 5/5 | 5.135e-12 | 0.00 | 5.0/5 | 168 | 3.125e-02 |
| baseline (unrotated) | 5/5 | 5.135e-12 | 0.00 | 5.0/5 | 168 | 3.125e-02 |
| `frac=0.50` | 5/5 | 5.135e-12 | 0.00 | 5.0/5 | 168 | 3.125e-02 |
| `frac=0.25` | 5/5 | 5.135e-12 | 0.00 | 5.0/5 | 168 | 3.125e-02 |

An unrotated baseline arm was added so the sharing change is not confounded with
the rotation change. All four rows are identical **because this instrument's
floor is one fp16 ULP**, not because the arms are equal — the same prompt with
sharing off vs on gives logit sums agreeing to 8 decimals while the byte hashes
differ. Read the table as "below the floor". Resolving differences among MLX DKV
variants needs a sharper instrument than first-step logits at 8k.

## Still open on the shared-basis port

1. Eviction, streaming compress and the batched multi-layer compressor all
   REFUSE under sharing rather than working. The pool must be sized so eviction
   never fires.
2. Checkpoint versioning is documented, not implemented.
3. Peak-RSS effect unmeasured — `_basis_rows` materialises one row per block at
   read time, so the transient is full-size even though the store is not. The
   pool-size numbers are exact; a peak-memory claim is not supported by them.
4. No long-context or recall validation of the sharing arm
   (`mlx_needle_parity.py --long`, `linkbench_mlx.py`).
5. The decode cost of the unrotated pool has not been measured WITH sharing on.

---

# THE PORT WAS ALSO BROKEN — two defects the logit harness could not see

The section above closed the 10-vs-463 grouping gap. Running the needle suite
against the fixed port then found that the port ITSELF was defective, in ways
that never showed up in any of the measurements taken up to that point.

**Correction to the section above.** It reported "no fidelity cost that this
instrument can resolve" and listed recall validation as owed. The recall
validation, once run, showed a real regression. The fidelity table was not
wrong — it was measuring below its own floor, which is exactly what it said.

## Symptom

With sharing on, `mlx_needle_parity.py` returned a TRUNCATED code at 8k depths
0.5 and 0.9 — `ZEBRA-4471`, dropping `-QUARTZ` — deterministically, while 2k
and 8k@0.0 passed.

Two controls made it a port defect rather than a property of sharing:

* it reproduced with sharing effectively DISABLED (`frac=1.0`, threshold 1.01,
  so every block founds its own group), and
* it reproduced at a second `DKV_SVD_SEED`, so it was not the ±15-point seed
  noise this repo's rules warn about.

The unrotated pool alone passed 6/6, so the rotation change was not the cause
either.

## Defect 1 — the registry lived on the MANAGER

Basis groups are SESSION state. A group records a subspace of ONE document's
keys, and its row is refcounted against that session's slots. Held on the
manager, a request's groups outlived it, so a later request's blocks scored
against a PREVIOUS DOCUMENT's bases and force-joined them once the store filled.

**Every single-session probe said the port was fine, and they were all
right — the bug needs two sessions to exist.** Prefill state was BYTE-IDENTICAL
(md5 over every `comp_*` array, the dense window, all six attended layers)
between a passing and a failing configuration. One case run alone passed with
`forced=0, mean_kept=1.0`. Only the needle suite — six cases in ONE process —
could see it, and it failed the LATER cases.

The registry now lives in the session dict, so `clear_session` disposes of it.

## Defect 2 — founders were not bit-exact

A block that founds its own group stores its own V, so the projection is
analytically the identity. Routing U through the solve returned it only to
within floating-point noise, and that noise moved the int8 U quantisation by a
couple of LSBs — enough to flip the needle. Founders are now returned
untouched, so with nothing sharing the feature is bit-identical to being off.

## Two corrections found on the way

* **`reproject_U` needed the PSEUDO-INVERSE.** `U (V Vg^T)` is the projection
  only when Vg's rows are ORTHONORMAL, and the joint `[K | V]` basis this pool
  stores is not — measured row norms 0.78–0.83, because the halves are sliced
  out of one orthonormal `Vh` and the V half is then divided by the per-block
  `v_scale` gain. Founders store their RAW V for the same reason: a
  unit-normalised basis leaves `U V` unchanged but rescales U and V against
  each other, and the ROUTER reads them separately. That is also why the
  failure was DEPTH-DEPENDENT — depth-invariant would have been reconstruction.
* **The `v_scale` undo was applied twice.** The shared basis is necessarily
  taken post-undo (one row backs many blocks; `v_gain` is per block), so
  dividing again scored residual selection against a V reconstruction the
  decoder never produces.

Plus two found by inspection while fixing the above: `_ensure_block_capacity`
grew `comp_VK` one row per NEW BLOCK (silently returning the store to one row
per block while still claiming `frac`) and did not grow `basis_of` at all; and
`snapshot_session` deep-copied mx arrays but let numpy fall through, so a
snapshot SHARED `basis_claimed` with the live session.

## Measured after the fixes — Qwen3.5-2B-4bit, unrotated pool

| arm | needle | determinism |
|---|---|---|
| sharing off | **6/6** | 1 distinct output across 3 runs |
| `frac=1.0` (no sharing) | **6/6** | 1 distinct |
| `frac=0.50` (real sharing) | **6/6** | 1 distinct |

Suite: 316 passed, 0 failed.

## The method lesson, which is the reusable part

**The instrument that could not resolve the change was not the instrument that
found the bug.** First-step logit fidelity bottoms out at one fp16 ULP on this
model and context; both defects lived far under that floor and it reported
"identical" for every arm — correctly, and uselessly. The needle suite found
both immediately.

Two properties made it able to: it runs SIX CASES IN ONE PROCESS (which is the
only reason the session leak was observable at all), and it asserts on an EXACT
STRING rather than a distance. A distance metric would have shown a small
number and been believed.

Corollary worth keeping: a byte-identical state comparison is not proof of
equivalence when the state you compared is not the state that differs. The
prefill pool matched exactly while the defect lived in state that accumulates
ACROSS sessions.

---

# CLOSING THE REMAINING OPEN ITEMS

## Long-context recall — 9/9 WITH SHARING ON

`mlx_needle_parity.py --long`, Qwen3.5-2B-4bit, `DKV_SHARED_BASIS=1`,
`frac=0.50`, unrotated pool:

    2k  @ 0.0 / 0.5 / 0.9    3/3 each, 1 distinct output across 3 runs
    8k  @ 0.0 / 0.5 / 0.9    3/3 each, 1 distinct
    32k @ 0.0 / 0.5 / 0.9    3/3 each, 1 distinct

**ALL PASS, including `32k@0.9`** — the rung the work order named as the bar and
which CUDA also passes. This is the validation that was owed, and it is the
check that found both port defects in the first place.

Incidentally it also shows the eviction guard never fires at these contexts: a
32k prompt completed without hitting the `RuntimeError` that sliding eviction
raises under sharing. The guard stays, but the pool is sized well enough that
2k–32k does not reach it.

## Peak memory — the pool saving does NOT reach peak, and that is the headline

Qwen2.5-1.5B @8k, one process per arm, decode rate by DIFFERENCE (a 1-token and
a 128-token call on identical fresh sessions, subtracted — a fresh session
re-prefills the whole prompt, and dividing an N-token call by its wall time
reports ~0.2 tok/s and charges prefill to decode):

| arm | decode tok/s | MLX peak | pool (allocated) |
|---|---|---|---|
| baseline (rotated) | 17.53 | 6.81 GB | 412.79 MB |
| unrotated | 15.49 | **7.12 GB** | 412.79 MB |
| unrotated + `frac=0.50` | 11.96 | **7.12 GB** | 368.75 MB (−10.7%) |

**Peak memory is IDENTICAL with and without sharing.** The *reason* given here
was wrong, and is corrected in the next section: this put a SYNTHETIC pool
number (28 layers, 64 slots, 412 MB) next to a REAL peak, and blamed
`_basis_rows`. Measured properly, in one process, the pool is 1% of peak at 8k
and 3.4% at 32k — the saving is real but far too small to move peak, and no
change to the read path can alter that.

**Decode cost, item 5:** the unrotated pool costs ~12% (17.53 → 15.49) and
sharing costs ~23% more on top (15.49 → 11.96), so the configuration the feature
requires runs at ~68% of baseline decode. These are SINGLE RUNS per arm, not
paired or replicated, so treat the decode numbers as point estimates; the memory
equality is the robust half, since the allocator peak is deterministic.

### What that means for the feature

As it stands on MLX, shared bases buy **pool bytes that do not become peak
bytes**, at roughly a third of decode. The chain is:

    shared bases --> unrotated pool --> no remat cache --> slower decode

and the memory win is currently swallowed by eager gathers. Two things would
change the verdict, in this order:

1. **Make `_basis_rows` lazy** — compose `basis_of` into the kernels' own gather
   instead of materialising a per-block view. That is what would turn the
   allocated saving into a peak saving, and it is the difference between this
   feature being worth its cost and not.
2. **Fix the remat-cache decline** for unrotated pools, which
   `dkv_attention.py` already sketches (plumb the dense window's absolute token
   positions in and rotate with `_partial_rope_apply`). That would pay back most
   of the 12%, and it benefits the unrotated pool's own retrieval win
   (40/48 → 47/48 on distractor retrieval) quite apart from bases.

Until at least (1) lands, `DKV_SHARED_BASIS` should stay opt-in on MLX — not
because it is unsafe (9/9 says otherwise) but because it currently costs decode
for a saving that does not show up where it matters.

## Snapshot / restore — was BROKEN under sharing, now fixed

`copy.deepcopy` cannot pickle an `mx.Dtype`, and the registry holds one. Both
`snapshot_session` and `restore_session` raised `TypeError: cannot pickle
'mlx.core.Dtype' object` the moment sharing was enabled — a crash, not a wrong
answer, but on a path no shared-basis test touched. `SharedBasisRegistryMLX`
now defines `__deepcopy__`, carrying the dtype by reference (it is an immutable
singleton) and deep-copying everything mutable.

This also closes the "checkpoint versioning" item from the earlier list, which
was mis-stated: MLX has **no on-disk checkpoint**. What it has is in-memory
`snapshot_session` / `restore_session`, and the shape concern the port file
raised belongs to `_ensure_block_capacity`, which is fixed and tested.

## Status of the port's open items

| item | status |
|---|---|
| long-context / recall validation | **CLOSED** — 9/9 incl. 32k@0.9 |
| checkpoint versioning | **CLOSED** — no on-disk path; snapshot/restore fixed |
| peak-RSS unmeasured | **CLOSED** — measured, and it does NOT improve |
| decode cost with sharing | **CLOSED** — ~68% of baseline, single-run estimate |
| eviction / streaming / batched compress refuse | **OPEN** — never reached up to 32k, guards retained |
| make `_basis_rows` lazy | **OPEN, and now the item that matters most** |

---

# THE LAZY-GATHER PLAN, MEASURED AND ABANDONED — and the cost that is real

The previous section concluded that `_basis_rows` materialising a full-size
per-block view was why the pool saving never reached peak, and named making it
lazy as the item that mattered most. **Both halves of that were wrong**, and
measuring before implementing is what showed it.

## Why the premise was wrong

**The pool is not where the memory is.** Real pool bytes and MLX peak, measured
in the SAME process (the earlier comparison put a synthetic pool number next to
a real peak, which is what produced the bad conclusion):

| ctx | arm | blocks | pool | V store | MLX peak |
|---|---|---|---|---|---|
| 8k | unrotated | 168 | 65.22 MB | 13.76 MB | 5.725 GB |
| 8k | `frac=0.50` | 168 | 58.34 MB | **6.88 MB** | 5.727 GB |
| 32k | unrotated | 840 | 221.74 MB | 46.79 MB | 6.574 GB |
| 32k | `frac=0.50` | 840 | 198.34 MB | **23.40 MB** | 6.571 GB |

The V store halves EXACTLY, as designed. But the whole pool is **1.1% of peak at
8k and 3.4% at 32k** — peak is dominated by weights (~3.1 GB fp16) plus prefill
activations. A 23 MB saving cannot move a 6.6 GB peak, and no read-path change
alters that arithmetic.

**And `_basis_rows` was never the cost anyway.** Counting calls by site through
a real 8k generate under sharing:

    line 4899  kind=nb  calls=140  total rows gathered=840

That is the ONLY site that executes. It gathers ~6 rows per call. The fused
decode kernel site, the decode-cache eval site and the routed-subset site never
run in this configuration at all. Making a 6-row gather lazy saves nothing.

**Verdict: not implemented, deliberately.** The one real defect found while
looking was an `mx.eval` target being handed a GATHERED view instead of the
store — a full-size materialisation purely to force evaluation. Fixed; it is
the same mistake already fixed once in the streaming-compress eval list.

## The cost that IS real, and where it actually was

Chasing the decode regression instead found a genuine one, in the SCORING math.

`retained_energy` had been changed to orthonormalise Vg internally (so callers
could pass the raw stored basis). That is a QR of an `[F, r]` matrix — F=512 —
**per candidate, per block**, and `mx.linalg.qr` has no GPU kernel, so it ran on
a CPU stream. Block compression also runs during DECODE, so it landed straight
in the token loop:

    frac=1.0, QR in the scorer      1.39 tok/s
    frac=1.0, [r,r] solve instead   9.86 tok/s        (7x)

The projector onto span(Vg) is now obtained with a batched `[r, r]` solve
(r=48), one per GROUP rather than per (block, group). It reduces to exactly the
old `C C^T` when Vg's rows are orthonormal, and the torch-agreement tests still
pass unchanged.

## Decode cost of sharing — bounded, not point-estimated

`colab/bench_decode_paired.py`'s own calibration says cross-process comparison
here "cannot resolve anything below ~20%", and that is generous: alternating
three rounds, the SAME arm spanned 6.39–10.31 tok/s. Absolute numbers from
single runs — including the "17.53 / 15.49 / 11.96" reported earlier — are not
quotable, and that earlier "~12% and ~23%" claim is **retracted**.

What survives is the PAIRED comparison, alternated over three rounds:

| round | unrotated | `frac=0.50` | ratio |
|---|---|---|---|
| 1 | 9.25 | 6.03 | 0.65 |
| 2 | 10.31 | 6.26 | 0.61 |
| 3 | 6.39 | 3.26 | 0.51 |

Sharing is slower in **3/3 rounds** at a consistent ratio, so the effect is real
even though its size is not pinned: roughly **0.5–0.65x decode**. Prefill runs
~10% longer (15.8s → 17.3s). Peak was **7.12 GB in all six runs**.

That harness cannot be used directly here, and the reason is worth recording:
its whole design is ONE process with the config flipped by rebinding a
module-level constant. `DKV_SHARED_BASIS` changes the POOL LAYOUT and is read at
manager construction, so the two arms cannot share a process. Measuring this
properly needs a different instrument, not a different run.

## Where that leaves the feature

Shared bases on MLX halve the V store exactly as designed, pass 9/9 needles
including 32k@0.9, and cost roughly a third to a half of decode for a pool
saving that is 1–3% of peak. **It should stay opt-in, and on this model there is
no operating point where it is currently worth enabling.**

It would become worth revisiting if the pool ever dominated peak — a much
longer context, a smaller model, or a configuration where weights are not 3 GB
of the footprint. That is a different measurement, not an argument.
