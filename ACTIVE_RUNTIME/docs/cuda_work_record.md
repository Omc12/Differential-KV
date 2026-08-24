# CUDA work record — 2026-08-24

The answers to `CUDA_TODO.md`, which is retired by this file. Everything below
was measured on an RTX 4070 SUPER (12 GB), torch 2.11.0+cu130, triton 3.6.0,
`transformers 5.14.1`, Qwen2.5-1.5B-Instruct fp16. Where something is a
hypothesis it says so.

`ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py` was NOT edited.

---

## 1. §1's premise was an INSTRUMENT DEFECT. The bug it named does not exist.

`CUDA_TODO.md` §1 opened "This is now settled, not suspected" and called CUDA's
KL 10.579 against MLX's 5.135e-12 "the highest-priority bug in the repo". It is
not a bug at all. **`colab/logit_fidelity.py` was measuring two different token
positions.**

`generate()`'s decode loop (`hf_dkv_wrapper.py:1836`) samples token 1 from the
logits PREFILL already produced and *then* runs a forward, so even at
`max_new_tokens=1` a decode step executes. The harness kept the LAST `lm_head`
call — that decode step — while its control kept the position one token earlier.
It was scoring DKV's token *N+2* against dense's token *N+1*.

Three things were wrong with the control as well, and each is fixed:

* the `dense` arm was `DKV_COMPRESSED_DECODE=0`, which is **not** a dense
  control: it shares DKV's block-sparse prefill with every other arm and can
  only ever measure the decode half. It is now `dense_arm`, and `dense_true`
  (plain `transformers`, DKV never imported) is the control. The MLX port hit
  the same trap and fixed it the same way (`logit_fidelity_mlx.py:60`);
* the output of `lm_head` could not tell prefill from decode — this stack slices
  hidden states to the last position before the head, so a 1024-token prefill
  chunk and a decode step both arrive as `L == 1` (measured: 9 calls at 8k, all
  `shape[1] == 1`). The discriminator is now the MODEL's `input_ids`;
* there was no engagement readout, so a small KL could not be told from "DKV
  never compressed anything".

**Corrected numbers, 8k, 5 depths, against a true dense control:**

| arm | position | top-1 | KL | dense-top1 rank | blocks |
|---|---|---|---|---|---|
| `dense_true/dec` | token 2 | 5/5 | **0.00000** | 0.00 | 0 |
| `baseline`       | token 1 | 5/5 | **0.00024** | 0.00 | 896 |
| `baseline/dec`   | token 2 | 5/5 | **0.00125** | 0.00 | 896 |

CUDA's prefill tracks dense. The `dense_true/dec` row is a permanent
self-check: it contains no compression anywhere, so anything but ~0 there means
the positions are misaligned again and no row in the table can be believed.

**What §1 got right by accident.** It said "bisect on logits, use the layer dump
to localise" and warned that per-layer cosine does not determine end behaviour.
Both held. Per-layer cosine was actively misleading here — see §1b.

**Note on the reference dumps.** `mlx_reference/*.npz` were not needed. The
useful ground truth turned out to be a CUDA-side dense control's own post-RoPE
K/V at layer 0, because at layer 0 the decode query is a pure function of token
and position and is therefore identical between arms.

---

## 1b. There IS a real decode defect, and §1's instrument could not have found it

Found by the corrected harness, on the row §1 never had.

**`_remat_attend` is called from two sites with dense windows in DIFFERENT
ROTATIONAL FRAMES, and it assumes the production one.**

`_use_combined` (`dkv_attention.py:4363`) is true when `DKV_SPARSE_BIAS` is unset
or `"0.0"` — the LIBRARY DEFAULT. That branch builds `_dk_combined` for
`native_triton_sparse_attn_decode_combined`: a fixed-width workspace re-rotated
into each token's TRUE frame. The compressed half `_remat_attend` materialises is
in the PRODUCTION frame — anchors and `V_K` straight out of a pool that already
stores post-RoPE keys, no second rotation. Measured at layer 0, 8k, against a
dense control's own keys:

| branch | window handed to remat | mean abs(dk − K_true) |
|---|---|---|
| combined (`DKV_SPARSE_BIAS` default) | `_dk_combined` | **0.047** |
| production (`DKV_SPARSE_BIAS=auto`)  | `dense_k_assembled` | 43.79 |

One plain SDPA over a union of two frames is meaningless. Nothing raises, the
shapes are right, and the pool reports the same block count.

**End to end, first decode step, against a true dense control:**

| configuration | serves | KL | top-1 | needle |
|---|---|---|---|---|
| `DKV_SPARSE_BIAS` default | remat | **11.76** | **0/5** | **LOST** |
| `DKV_SPARSE_BIAS=auto` | remat | 0.00125 | 5/5 | `OMEGA-7741-DELTA` |
| `DKV_REMAT_CACHE=0` | kernel | 0.00125 | 5/5 | `OMEGA-7741-DELTA` |

**Why nothing caught it.** `BEST_DECODE_DEFAULTS` sets `DKV_SPARSE_BIAS=auto`,
so everything that goes through the serving defaults — including
`validate_cuda_dkv.py`, whose 9/9 is unaffected — takes the production branch.
A caller using the library without those defaults got the broken pairing
silently. `colab/logit_fidelity.py` does not apply them, which is why the
corrected harness walked straight into it.

**Fix.** `_remat_attend` takes `combined_window=True` from that call site and
declines, with a reason code. The combined branch's own kernel is what served
before remat was wired into it, and it is correct. After the fix all three rows
above read KL 0.00125 / 5-of-5, `validate_cuda_dkv.py --long` is unchanged at
ALL CHECKS PASSED, and `colab/needle_suite_cuda.py` reports the arms identical.

**Ruled out by measurement before the frame was found** — recorded so nobody
re-walks them:

| suspect | test | result |
|---|---|---|
| routing dropping blocks | `DKV_TOPK_BLOCKS=0` (attend all 27, one chunk) | 11.20 vs 11.76 — no |
| low-rank capacity | rank 32 → 96 → 128 | 11.76 / 11.42 / 11.70 — flat |
| residual budget | `DKV_MAX_RESIDUAL` 40 → 200 | 12.23 — slightly worse |
| fp16 precision | attention in fp32 | 11.75 — no |
| stratified U | `pool.U` vs `reconstruct_batch_U` | `n_semantic` is None; identical |
| residual plumbing | gather → `reconstruct_blocks` | `has_res=True`, 640 valid, passed |
| score algebra | remat's q·K vs project-then-attend, same inputs | agree to 0.48 mean |
| row set / coverage | truth over remat's rows vs all rows | cos 0.99988 — no |
| V reconstruction | true K + remat V | cos 0.99880 — no |

**Two traps inside this investigation, both of which produced a wrong reading
before being caught.** They are the reason the table above is worth keeping.

1. **Comparing `_remat_attend`'s return against the attention module's output.**
   remat returns `[1, H_q, 1, D]` BEFORE the output projection; the module's
   output is `[1, 1, hidden]` after it. On this model both flatten to 1536, so
   the comparison ran and reported cos ≈ 0 at *every* layer — which reads
   exactly like a catastrophic bug. Compare at `o_proj`'s INPUT.
2. **A "truth" that omitted the self row.** Attention at the decode step attends
   the current token's own K/V. A dense control stopped after prefill does not
   have it, and leaving it out put BOTH arms at cos ≈ 0.75 from "truth" for a
   reason having nothing to do with compression. With it included the fallback
   reads **0.99929** and remat **0.71110**.

**And a coverage note in the spirit of §0's rule 6.** Cosine on layer-0 KEYS is
worthless here: key norms are ~1102 while token-to-token variation is O(10), so
anchor-plus-anything reads cos 0.9997. V norms are ~6, so V is the sensitive
side. Every K conclusion in this file rests on attention OUTPUT or on score
error, never on key cosine.

---

## 2. Rotated pools now REFUSE shared bases

Implemented as §2 prescribed, at pool construction (`native_block_pool.py`),
with `DKV_SHARED_BASIS_ALLOW_ROTATED=1` as the escape hatch so the bad
configuration stays measurable. The refusal names the setting to change; a bare
"ignored" is what sent three earlier debugging passes after the wrong knob.

**This immediately exposed that the existing tests were measuring the degenerate
arm.** `DKV_ROTATED_POOL` defaults to `"1"`, and every shared-basis test enabled
sharing without naming the pool — so 11 of them had been exercising rotated
sharing, which is the configuration that silently force-joins at the full
advertised memory saving. They now set `DKV_ROTATED_POOL=0` explicitly.
`test_low_preset_pool_actually_saves` was asserting the saving on exactly the
combination §2 warns about.

---

## 2b. Two of the three projection defects ported; the third is a real divergence

**(a) `reproject_U` — FIXED.** Now `U' = U V Vg^+` via an `[r, r]` solve.

**(c) `retained_energy` — FIXED.** Projector via a batched `[r, r]` solve, one
per GROUP with the block axis folded into the right-hand side. No per-candidate
QR: `test_retained_energy_still_matches_the_orthonormal_form` pins that it
reduces to the old `C C^T` when Vg really is orthonormal, so previously measured
sharing numbers stay comparable.

**(b) founders storing the orthonormalised basis — NOT PORTED, and the reason is
measured.** §2b called this a defect and prescribed storing the raw V. On CUDA
that is a **regression**, because this pool quantises U to INT8 with one
per-block scale (`native_block_pool.py`, `scale_u = max_abs/127`). A raw joint
`[K | V]` basis is ill-conditioned — measured cond 46.6, row norms 2.42–3.45 —
and `U' = U V Vg^+` carries that conditioning into the tensor being quantised.
Measured on `test_shared_blocks_still_reconstruct`, six blocks on one basis:

| store | founder rel | joiner rel |
|---|---|---|
| orthonormalised | 0.0070 | **0.0058–0.0079** |
| raw V | 0.0072 | **0.0379–0.0791** |

An exact founder for a factor of 5–10 on every joiner. MLX does not quantise U,
so its trade is the other way round; the choice does not port. Both runtimes use
the pseudo-inverse regardless, since it costs nothing when Vg is orthonormal.
Noted in `basis_group_mlx.py` so the divergence is visible from both sides.

**Session state — a REAL leak was found.** `pool.free_block` does call
`release_basis`, so the refcounting §2b hoped for is intact. **`pool.reset()`
was not.** It deletes `V_KV`, but `basis_store` is a `_JointVAdapter` holding a
reference to it, and `basis_registry` / `basis_of` were never cleared. On the
LAZY path — CUDA's default — `reset()` does not re-allocate, so the registry
survived with every group it had, its rows indexing a store no reader uses and
its capacity already spent; the next document's blocks were force-joined to the
previous one's bases. Same defect MLX hit with the registry on the manager.
Fixed, with `test_lazy_pool_drops_basis_state_on_reset` (which fails on the old
code; the non-lazy test passes either way, because that path re-allocates).

**(d) the memory claim.** Not re-measured. CUDA's −23.6% remains a CAPACITY
result via `_bytes_per_block`, which is a different and defensible claim; it is
still not backed by a same-process PEAK measurement, so it must not be quoted as
a memory saving. Open.

---

## 3. The instrument CUDA was missing

`colab/needle_suite_cuda.py` — several cases in ONE process, asserting on an
EXACT STRING. `validate_cuda_dkv.py` runs each case in its own process AND
applies `BEST_DECODE_DEFAULTS`, so it could see neither state leaking between
requests nor any defect that only appears without those defaults. §1b was the
second kind. The suite deliberately does NOT apply the serving defaults;
`--serving-defaults` measures the other configuration.

Current reading, all arms identical: 3/4, the one failure being an
arm-independent partial recall at `8k@0.1` (the model answers `7741-DELTA`,
dropping `OMEGA-`). That is a real model/config result, not an arm difference.

**One trap worth carrying.** Isolating the completion by re-tokenising
`generate()`'s output is NOT safe: `decode(encode(x)) != x` here, and the round
trip clipped correct answers to `-DELTA` and scored them FAIL. Slice by
character when the prompt is a literal prefix.

---

## 4. Still open, unchanged by this work

* **§4 decode −12%** — not investigated.
* **§4 sparse prefill selectivity** — not re-measured; `DKV_SP_TRACE_TOKEN`
  still the way in.
* **§2b(d) peak memory** — see above.
* **The root inside `_dk_combined`** — the fix declines rather than reconciling
  the two frames. Making remat serve the combined branch would mean giving it a
  window in the pool's frame, or teaching it which frame it holds. Worth doing
  only if the combined branch is ever measured faster; it is not today.
* **§3 of the old file (stale line numbers)** — every line number in this record
  is paired with a function name or a distinctive fragment for that reason.
