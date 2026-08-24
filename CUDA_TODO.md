# CUDA work order — written from the Mac, 2026-08-23

The reverse of `MLX_TODO.md`, which has been retired: its items are all
answered and the record lives in `ACTIVE_RUNTIME/docs/mlx_work_record.md`.

Everything here was measured on an M3 (8 GB), MLX 0.32.0 / mlx_lm 0.31.3,
`transformers 5.14.1`. **Nothing here was run on CUDA** — that is the whole
point of the file. Where an item is a hypothesis it says so.

Standing rules are unchanged and still govern this file
(`HANDOFF_CUDA_PREFILL.md` §0), with **one amendment you need to know about**:
`mlx_dkv_wrapper.py` HAS been edited, and the change is now on `main`
(`227cdd7f`). It adds shared low-rank bases, opt-in and off by default.

**The reference implementation is still a valid reference.** Before merging,
the default path was shown BYTE-IDENTICAL to the previous `main` — md5 over the
entire compressed pool plus the dense window, every attended layer, on an 8k
prompt: `cf4582197fb9a530` both sides, with first-step logits hashing to
`47f057f05d700cbe` both sides. With `DKV_SHARED_BASIS` unset, MLX produces the
same bytes it did before. Diff CUDA against it exactly as you did.

What that does mean for you: when you read MLX for a behaviour, **check whether
the line is inside a `self._shared_basis` branch**. If it is, it is not the
reference path and CUDA should not be made to match it.

---

## 1. HIGHEST VALUE — DKV's logit fidelity is a CUDA DEFECT. Bisect it.

This is now settled, not suspected.

| runtime | top-1 agree | KL(dense‖DKV) | dense-top1 rank | top-5 overlap |
|---|---|---|---|---|
| **MLX** | **5/5** | **5.135e-12** | **0.00** | **5.0/5** |
| CUDA | 0/5 | 10.579 | 1254.6 | 0.2/5 |

Same model (Qwen2.5-1.5B-Instruct), same prompt builder, same context (8k),
same 5 depths, each runtime against its own dense control. MLX's
`max|Δlogit|` is 3.125e-02 = 2⁻⁵, **exactly one fp16 ULP** at logit magnitude
~30, with 168 compressed blocks in the pool — so it is genuinely compressing
and genuinely tracking dense. 4k behaves the same (KL 2.475e-13, 56 blocks).

The (now retired) MLX work order pre-committed this reading before the run —
see `ACTIVE_RUNTIME/docs/mlx_work_record.md` §1: MLX baseline KL < 1 with high
top-1 agreement ⇒ **CUDA's gap is a CUDA defect rather than the price of
compression, and it is the highest-priority bug in the repo.** The measured KL
is twelve orders of magnitude below that bar, so this is not a marginal call.

**What this unblocks.** Every failed recall-based accuracy attempt on CUDA was
comparing arms on top of a baseline already off the map. Fix this and those
experiments become interpretable — including the shared-basis fidelity question
that had to ship opt-in for want of an instrument.

**Reference data is checked in for you.** `mlx_reference/` holds per-layer
attention outputs from a known-good MLX run:

    mlx_layers_qwen25_1p5b_8k_d05.npz     28 layers x (dense, dkv) + token ids
    mlx_layers_qwen25_1p5b_8k_d09.npz     same at depth 0.9

Produced by `colab/probe_mlx_layer_output_diff.py --model
Qwen/Qwen2.5-1.5B-Instruct --label 8k --depth 0.5 --save <path>`. Its CUDA twin
already exists. **Check the stored `__model` field before diffing raw vectors**
— they are only comparable across runtimes because both sides load the same
fp16 HF checkpoint; the script's own 4-bit default would make them meaningless.

MLX's per-layer cos(DKV, dense) at 8k/depth 0.5 runs 0.12–0.77 (layer 3 =
0.554) against CUDA's 0.291/0.276. Note that this is a **weaker** signal than
the logit result and should not be the primary instrument: MLX's per-layer
attention outputs deviate substantially from dense while its final logits do
not, so per-layer cosine evidently does not determine end behaviour. Bisect on
logits, use the layer dump to localise.

---

## 2. Shared bases will SILENTLY DEGENERATE on CUDA's `low` preset. Add the guard.

**The single most transferable finding from the MLX port**, and CUDA is exposed
to it today.

Shared low-rank bases compare SUBSPACES. RoPE rotates every key by its
ABSOLUTE POSITION, so two blocks holding the same text at different offsets
have subspaces rotated apart and the grouping collapses. Measured on MLX, same
document, same block size, `frac=0.50`, only `DKV_ROTATED_POOL` differing:

| pool | best-partner retained energy | founded | joined | forced | mean_kept |
|---|---|---|---|---|---|
| rotated | mean 0.486, **0/27** clear 0.90 | 560 | **10** | 186 | 0.903 |
| unrotated | mean 0.972, **26/27** clear 0.90 | 236 | **520** | **0** | 0.968 |

The unrotated row lands on CUDA's own published numbers (joined 463, forced 0,
mean_kept 0.969), which is what identifies rotation as the whole mechanism.

**CUDA has never seen this because it got lucky:** every preset it enables
sharing on — `mid`, `high`, `ultra` — already sets `rotated_pool=False`
(`native_core/config.py:147, 328, 464`). But `low` is CUDA's one ROTATED preset
(`config.py:143`), so `DKV_SHARED_BASIS=1` on `low` degenerates exactly as
above.

CUDA already warns at pool construction when `low`'s `kv_quant` is 4-bit. **It
does not check rotation**, and these are two INDEPENDENT reasons sharing fails
on that preset — fixing or excusing one does not cover the other.

The trap is the same one the port file already names: **pool MB is identical
either way**, because the saving comes from allocating fewer basis rows rather
than from grouping succeeding. A rotated run therefore reports the full memory
win with its fidelity quietly bought by forced lossy joins.

**Suggested fix**, mirroring what MLX now does: refuse at pool construction
when `shared_basis` is on and the pool stores rotated K, with an
`ALLOW_ROTATED` escape hatch so the bad configuration stays measurable. Refuse
rather than warn — the failure produces no error, no shape change, and a
memory number that looks correct.

---

## 2b. CUDA's shared-basis code has the SAME TWO DEFECTS the MLX port had

Both were found by running the needle suite against the MLX port. CUDA has
never run an equivalent check on this feature — its own recall validation
scored 0/3 for the DENSE control at that operating point, and its logit
instrument sits at KL 10.579, so neither could discriminate anything. **Shared
bases on CUDA are therefore unvalidated against any working accuracy test**,
not validated-and-clean.

**(a) `reproject_U` assumes Vg has ORTHONORMAL ROWS, and it does not.**
`basis_group.py`'s `U (V Vg^T)` is the projection onto span(Vg) only for
orthonormal Vg. The joint `[K | V]` basis both pools store is not orthonormal:
the halves are sliced out of one orthonormal `Vh` and the V half is then
divided by the per-block `v_scale` gain — `lowrank.py:1260` does exactly that,
**before** the assignment at `:1384`. Measured row norms on MLX: 0.78–0.83.
The fix is the pseudo-inverse, `U' = U V Vg^+`, which also makes a founder
exact by construction.

**(b) Founders store the ORTHONORMALISED basis rather than their own V**
(`basis_group.py:413`, `basis_store[row] = V_on[n]`). That keeps `U V`
unchanged, so reconstruction stays exact and no distance metric notices — but
it rescales U and V against each other, and **the ROUTER reads them
separately**, so it retains a different set of blocks. On MLX the signature was
a needle that passed at depth 0.0 and failed at 0.5 and 0.9. Depth-DEPENDENT is
routing; depth-invariant would have been reconstruction.

**(c) If you adopt (a) and (b), do NOT reintroduce a per-candidate QR.**
CUDA today calls `row_orthonormalize` ONCE per compress batch
(`basis_group.py:366`) and `retained_energy` requires an already-orthonormal
Vg, so CUDA does not have the performance bug MLX hit. But the fix for (a)/(b)
is to store the founder's RAW basis, and the obvious way to keep scoring
working after that is to orthonormalise Vg inside `retained_energy` — which is
a QR of an `[F, r]` matrix per CANDIDATE per BLOCK. On MLX that cost 1.39 tok/s
against ~9.9, because block compression also runs during DECODE. Use a batched
`[r, r]` solve for the projector instead (it reduces to exactly the old
`C C^T` when Vg is orthonormal); `basis_group_mlx.retained_energy` has the
form.

**(d) Check the memory claim against PEAK, not just pool bytes.** On MLX the V
store halves exactly as designed and the saving is still only **1.1% of peak at
8k and 3.4% at 32k** — peak is dominated by weights and prefill activations, so
the pool is simply not where the memory is. CUDA's README frames its −23.6% as
a CAPACITY result ("the budget holds proportionally more blocks"), which is a
defensible and different claim — but if it is ever quoted as a memory saving,
it needs a peak measurement behind it. Measure pool bytes and peak IN THE SAME
PROCESS: putting a synthetic pool number next to a real peak is exactly the
mistake that produced a wrong conclusion here, and it took a second measurement
to catch.

There is also a third, which CUDA appears to survive but should confirm:
**basis groups are SESSION state.** On MLX the registry lived on the manager,
so one request's groups outlived it and a later request's blocks force-joined a
PREVIOUS DOCUMENT's bases once the store filled. CUDA's registry lives on the
pool and is refcounted, with `release_basis` on slot free, so dead groups
should be reclaimed — but that is only true if EVERY slot-free path calls it.
Worth checking directly, because the MLX version of this bug was invisible to
every single-session test: prefill state was byte-identical between a passing
and a failing configuration, and only a six-cases-in-one-process run exposed it.

---

## 3. Stop citing line numbers across runtimes; they are already stale.

The retired `MLX_PORT_FROM_CUDA.md` §1 listed exact `mlx_dkv_wrapper.py` line numbers for
every slot-indexed read to redirect. **Every one had drifted** by the time the
port was attempted — the sliding eviction it cited at `:3605` was at `:4256`,
and the rest moved similarly. Following them literally edits the wrong code,
and the failure mode is silent because neighbouring lines are often
structurally similar.

Cite a FUNCTION NAME and a distinctive code fragment instead. The port was done
by re-deriving the site list with a grep for `comp_VK|comp_VV`, which took
under a minute and cannot go stale.

---

## 4. CUDA-side items that are still open on CUDA's own list

Carried forward from `HANDOFF_CUDA_PREFILL.md` §8, unchanged by this work:

* **Decode is ~12% slower** after the §0.5 prefill fixes and it is NOT
  explained. A prefill-only change moving decode at all points at the working
  set: correct routing retains a different set of blocks, so decode gathers
  differently. Measure with LONG generations — the same harness reported −28%
  at `max_new_tokens=32` and −12% at 128 on the same build, because the rate is
  derived by subtracting a 1-token call and fixed overhead inflates small N.
* **Sparse prefill selectivity was inferred, not measured.** The 9/9 result
  verifies `fallback_count=0` directly, but the `k_eff < nb` half — "prefill was
  genuinely selective rather than degenerating to attend-all" — was not
  separately checked. Re-run with `DKV_SP_TRACE_TOKEN` if that distinction
  matters.

---

## 5. What the Mac fixed in SHARED code — worth knowing, no CUDA action

These are all in files CUDA also compiles, but every one is on a path CUDA does
not execute. Listed so nobody re-derives them or "fixes" them back.

* `native_triton_sparse_attn_decode` returned a **ZERO-WIDTH tensor** for a
  zero-block step. The N==0 dense-only fix of 2026-08-17 went into the Triton
  branch, but the function short-circuits at `if not HAS_TRITON:` and returns
  before reaching it. CUDA always has Triton and always skipped the broken
  branch; CPU and Apple silicon always take it.
* `decode_attention_metal` rejects Long `slot_indices`. The guard commit
  (`39a4a9d1`) converted `anchor_indices` at the call site and left
  `block_indices` raw at all four sites, so the conversion its own error message
  prescribes was never applied. macOS-only.
* The **MPS dense window was rotated twice**, in production and not only in the
  validator. `low` is the only preset keeping `rotated_pool=True` and also sets
  `approximate_attn=True` on macOS — exactly the `_is_mps_decode` gate. (Note
  the recurring theme with item 2: `low` is where the rotated-pool assumptions
  keep biting.)
* `tests/conftest.py` now restores `DKV_ROTATED_POOL` and `DKV_SVD_ENERGY` per
  test. `native_core/config.py:814,819` EXPORT preset-derived values into the
  process environment with `setdefault()`, so the FIRST config built in a
  process wins for the whole process. **This one can bite CUDA too** — any CUDA
  test that reads those variables after another test has built a config is
  order-dependent in the same way.

**The MLX exception is now merged** (`227cdd7f`), off by default, with the
default path proven byte-identical to the previous `main` — see the amendment
at the top of this file. Item 2b above IS CUDA work that came out of it: the
two projection defects live in `basis_group.py`, which CUDA runs today.

**Do not port the MLX shared-basis feature's SETTINGS to CUDA.** On MLX it is
merged but NOT recommended: the V store halves exactly as designed, yet the
saving is 1.1% of peak at 8k and 3.4% at 32k, while decode is slower in 3/3
paired rounds at 0.5–0.65x. Whether CUDA's own trade is better is an open
question — its slot composition differs (block 257 vs 1024, so V is a much
larger share of a slot there) and its `_bytes_per_block` framing makes the
saving a CAPACITY result rather than a peak one. That is item 2b(d) to measure,
not to assume either way. It exists because the owner directed
it explicitly, and `main` is untouched. Suite on that branch: 307 passed, 0
failed, with 295 of those passing with the feature OFF.

---

## 6. Measured on MLX, for calibration when comparing runtimes

* **MLX's fidelity instrument saturates at one fp16 ULP.** At 8k every DKV
  variant tested — baseline, unrotated baseline, `frac=0.50`, `frac=0.25` —
  prints identical KL and identical `max|Δlogit|`. That is the instrument's
  FLOOR, not equality: the same prompt with sharing off vs on gives logit sums
  agreeing to 8 decimals while the byte HASHES differ. Do not read those rows
  as "bit-identical", and do not expect this harness to resolve small
  compression changes on MLX.
* **MLX compresses 1–2 blocks per call** (392 `_assign_shared_basis` calls at
  8k), where CUDA batches a whole layer. Anything reasoning about per-batch
  behaviour has to account for that difference.
* `mx.linalg.qr` is **CPU-only** and raises on GPU.
* MLX's `block_size` default is 1024 against CUDA's 257. That alone changes
  slot composition enough to move the shared-basis memory saving from CUDA's
  −23.6% to −10.7% on MLX, because `comp_U` dominates a slot at the larger
  block size. It is not a porting defect.
