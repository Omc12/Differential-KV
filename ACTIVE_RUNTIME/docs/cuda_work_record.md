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

**The combined branch handed `_remat_attend` an ALREADY-ROTATED dense window,
and remat rotated it a second time.** Same defect family as §0.5's
double-RoPE-on-history, at a new site.

The default preset leaves the pool UNROTATED — `mid`, `high` and `ultra` all set
`rotated_pool=False` and `config.py` exports it into the environment; only `low`
keeps a rotated pool (see §4b, which turns on the same fact). On an unrotated
pool `_remat_attend` rotates the dense window itself, at the positions from
`dense_blocks[].token_indices`, to match the compressed half it also rotates. So
it must be given the UNROTATED window.

`_use_combined` (`dkv_attention.py`) is true when `DKV_SPARSE_BIAS` is unset or
`"0.0"` — the LIBRARY DEFAULT. That branch builds `_dk_combined` for
`native_triton_sparse_attn_decode_combined`, which wants the window PRE-ROTATED,
and passed that to remat. Measured at layer 0, 8k, against a dense control's own
post-RoPE keys — note that the tensor CLOSER to truth is the wrong one to pass,
precisely because it already carries the rotation remat would apply:

| branch | window handed to remat | mean abs(dk − K_true) |
|---|---|---|
| combined (`DKV_SPARSE_BIAS` default) | `_dk_combined` | **0.047** |
| production (`DKV_SPARSE_BIAS=auto`)  | `dense_k_assembled` | 43.79 |

Rotating `_dk_combined` again put the dense rows in no frame at all while the
compressed rows landed correctly — one plain SDPA over a union of two frames,
which is meaningless whichever frame is "right". Nothing raises, the shapes are
right, and the pool reports the same block count.

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

**Fix: the combined branch now passes `dense_k_assembled`**, the same window the
production branch hands remat and the one its contract is written for.

**Declining was tried first and REJECTED on measurement.** Making
`_remat_attend` refuse the combined branch is correct — it returns that branch to
its own kernel — but `colab/bench_decode_paired.py` prices remat there at
**29.9% of decode** (54.75 vs 78.43 ms/token, paired over 8 rounds, CI ±0.7% of a
token). Declining bought correctness and paid all of it; passing the right window
buys correctness for nothing, and is also markedly more stable (cv 1.3% against
the old broken path's 22.9%, because remat's cache serves whole intervals).

After the fix all three rows above read KL 0.00125 / 5-of-5,
`validate_cuda_dkv.py --long` is unchanged at ALL CHECKS PASSED, and
`colab/needle_suite_cuda.py` reports the three arms identical.

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

**(d) the memory claim — MEASURED.** `colab/bench_shared_basis_peak.py`, pool
bytes summed from the pool's REAL tensors and peak from
`torch.cuda.max_memory_allocated()`, both in the same process, both arms:

| ctx | pool off | pool on | pool Δ | peak off | peak on | peak Δ |
|---|---|---|---|---|---|---|
| 8k | 95.9 MB | 73.2 MB | **−23.7%** | 3619.2 MB | 3598.1 MB | **−0.6%** |
| 32k | 366.1 MB | 279.4 MB | **−23.7%** | 4614.5 MB | 4532.6 MB | **−1.8%** |

The −23.6% pool figure reproduces exactly and the V store halves as designed
(924 → 462 rows at 8k, 3528 → 1764 at 32k). Peak barely moves, because weights
alone are 3087 MB of every peak above. MLX measured 1.1% / 3.4% for the same
feature; CUDA is 0.6% / 1.8% — same conclusion. **The CAPACITY framing is the
only one the measurement supports**, and the README now says so.

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

## 4. The two §4 items — both ANSWERED, and both were instrument problems

### 4a. The "~12% slower decode" cannot be salvaged: both numbers are impossible

`HANDOFF_CUDA_PREFILL.md` §8 records decode 259.8 tok/s (HEAD) vs 229.4 (fixed)
at 8k on Qwen3.5-2B, "**-11.7%** ... and that is NOT explained", alongside a
dense arm at 309.0. All three come from

    decode_tok_s = (GEN - 1) / (total_s - ttft_s)

where `total_s` and `ttft_s` are the walls of TWO SEPARATE `generate()` calls,
each running its own full prefill (`benchmarks/clean_sweep_v2.py:100-128`).

**A bandwidth bound settles it without needing the old build.** Decode streams
every weight once per token, so tok/s <= bandwidth / weight_bytes. Qwen3.5-2B is
1.882 B params fp16 = 3.76 GB; this card is 504 GB/s:

| accounting | ceiling |
|---|---|
| all weights | 133.9 tok/s |
| excluding the embedding table entirely (over-generous — it is TIED, so `lm_head` streams it) | 183.5 tok/s |

Against the most generous ceiling, dense 309.0 is **1.68x** it, HEAD 259.8 is
**1.42x**, fixed 229.4 is **1.25x**. All three are above what the hardware can
do, so none is a decode rate and their difference is not a regression.

**And the estimator's noise is prefill wall noise.** `colab/bench_decode_
estimator_check.py` runs it against itself — same build, same prompt, nothing
changed between repetitions — and reports it beside per-token times taken from
`DKV_TIME_ATTN`, which need no subtraction:

| run | prefill wall cv | ESTIMATOR range | ground-truth range |
|---|---|---|---|
| clocks ramping | 11.06% | **17.2%** | 2.0% |
| warm and quiet | 1.15% | 0.3% | 1.2% |

So its resolution is not a property of the method alone — it is whatever the
prefill wall happens to be doing, amplified by prefill/decode ≈ 0.6x here. The
original −12% was measured across two builds in separate runs, which is the
worst case for that. It also reads **25% low** in both runs (20.2–20.6 against a
true 27.3–27.4 tok/s), because the two prefills do not cancel.

**The real number.** Qwen3.5-2B, 8.4k, DKV under the serving defaults:
**27.4 tok/s (36.5 ms/token)**, range 1.2% across reps. Use
`colab/bench_decode_paired.py` for comparisons — its A/A control resolves ±0.3%
of a token.

**What decode timing DID find.** Pricing this session's own change with that
harness: remat is worth **29.9%** of decode on the combined branch (54.75 vs
78.43 ms/token, paired, CI ±0.7%). That is what made handing remat a correctly
framed window the right fix rather than letting it decline.

### 4b. Sparse prefill selectivity — measured, and it is OFF by default

§4 asked whether prefill was "genuinely selective rather than degenerating to
attend-all". It degenerates, and not subtly. `_sparse_prefill_filter_blocks` has
four early returns that all mean *attend every block*, and its only instrument
(`DKV_SP_TRACE_TOKEN`) prints AFTER all four, so a decline is silent. Counting
every call instead:

| pool | ctx | selective calls | what prefill attended |
|---|---|---|---|
| unrotated (**the default**) | 8k | **0 of 196** | every block, every chunk |
| unrotated | 32k | **0 of 868** | every block, every chunk |
| rotated | 8k | 0 of 196 | every block — `k_eff >= nb` |
| rotated | 32k | 616 of 868 | nb 9–30, k_eff 8, dropping 10–71% |

The decline's own comment called `DKV_ROTATED_POOL=0` "a non-default diagnostic
path". **It is the default**: `mid` sets `rotated_pool=False` and `config.py`
exports it into the environment, as do `high` and `ultra`; only `low` keeps a
rotated pool. So on the shipped configuration there is no prefill sparsity at
any context, and even on a rotated pool it does not engage below ~32k. §8's
"prefill is STILL SPARSE — k_eff=30 of 120" is the 32k rotated case and does not
generalise. The comment now carries this table.

### 4c. Prefill sparsity on an UNROTATED pool — built, correct, and OFF by default

The decline's stated blocker was "the keys cannot be rotated without their true
per-token positions". **That premise was false.** A block's anchor is token
`anchor_idx`, its active row j is `anchor_idx + 1 + j`, and a compressed block's
residual j is at `anchor_idx + 1 + residual_K_positions[j]` — the same mapping
`_remat_attend`'s trace already resolves. The second blocker, "q cannot be
un-rotated", is moot: nothing un-rotates q, the KEYS move instead.

`_prefill_block_key_boxes` now takes a `rope` callback and rotates each key at
its own absolute position before the min/max, which puts the box in the same
frame as the post-RoPE `chunk_q`. Behind `DKV_SPARSE_PREFILL_ROTATE=1` the
unrotated pool reaches **616 of 868 selective at 32k, dropping 10–71%** — the
rotated column exactly. `validate_cuda_dkv.py --long` is ALL CHECKS PASSED with
the flag on, all three 32k cases 3/3 and deterministic.

**It ships OFF, on evidence.** Paired prefill A/B at 32k
(`colab/bench_prefill_paired.py`, A/A control ±4.2% then ±1.0–1.3%):

| pool | sparse prefill vs off | reading |
|---|---|---|
| rotated | **9.2% faster**, CI ±1.0% | routing pays |
| unrotated + rope | **no effect resolvable**, CI [−241, +87] ms | it does not |

The reason is structural: an unrotated pool's history reader must rotate keys
for the attention anyway, so skipping blocks saves rotation — and the router has
to rotate to decide what to skip. The saving and the cost are the same work.
It is not free either: at `block_size` 256 the router engages at 8k and the
first-token KL against a dense control goes **0.00024 → 0.00585** (still 5/5
top-1, dense's top-1 at rank 0). Paying fidelity for a measured-zero speedup is
the wrong default.

**Both numbers re-taken at ONE operating point.** An earlier note claimed they
came from different block sizes; that was wrong — `bench_prefill_paired.py` and
`logit_fidelity.py` both run block_size 256, preset mid. The context differed,
so both were re-taken at 32k, where routing actually engages:

| 32k, first token, vs a plain-transformers control | KL | top-1 | rank |
|---|---|---|---|
| ROTATE off (default) | **0.00036** | 3/3 | 0 |
| ROTATE on | **0.10580** | 3/3 | 0 |

294× the KL at the exact context where the paired throughput A/B reports no
resolvable change. The fidelity is spent and nothing is bought with it.

### Two ways to make it pay — both tried, both fail

**1. "Decide without rotating": transform the BOX, not the keys.** A box is
`[nb, H_kv, D]` against keys at `[nb, S, H_kv, D]`, so 257× less work, and RoPE
acts on 2-D pairs so a rectangle maps to a rotated rectangle whose enclosing box
is exact *at one angle*. The block is the problem: it spans S positions, pair i
sweeps `theta_i * S`, and at theta=1e6, S=257 the fast pairs wrap many times —
pair 0 sweeps 257 radians. The only enclosure valid at every position is the
RADIUS, which discards direction. Counting pairs that keep a tight box against
the sweep a single angle may cover:

| sweep ≤ | pairs tight | sound? |
|---|---|---|
| 0.5 | 35/64 | **no** — an enclosure test finds keys outside the box |
| 0.05 | 24/64 | no |
| 0.001 | 6/64 | yes, and 58/64 ranking on magnitude alone |

Sound and discriminative are mutually exclusive here. Sub-block boxes do not
rescue it (pair 0 wraps within ~6 positions). It was built, tested, and reverted;
the enclosure test is what caught the unsoundness, not a recall run.

**2. Lower the floor.** `k_eff = max(KMIN, 0.25*nb)` with KMIN=8 against nb=9–30
is what actually caps the win: at `k_eff≈2` the unrotated pool DOES pay — **5.1%
at 32k, CI ±1.2%**. Recall survives it: needle suite unchanged, and
`validate_cuda_dkv.py --long` **9/9 including all three 32k cases at KMIN=2**.

**It still must not ship, and only one harness could tell:**

| `multifact_eval_cuda.py`, 16k, Qwen2.5-1.5B | multi-needle | relational | synthesis |
|---|---|---|---|
| KMIN=8 | 3/3 | **4/4** | 13.3 |
| KMIN=2 | 3/3 | **3/4** | 30.0 |

Asked for Dr. Quillfeather's number at KMIN=2 the model returns **8857 — Dr.
Braxanible's**. A BINDING failure, which is the characteristic compressed-KV
failure and which NIAH cannot see by construction. Synthesis rising at the same
time is not a counterweight: 13.3 is this model's floor with routing OFF too, so
it was never measuring the router. **KMIN=8 stays**, and if it is ever revisited
the gate is multifact, not the needle suite.

That also answers the separate "8k never engages" item: it is the same KMIN, and
the same reason not to move it.

**Two performance defects were found getting there, and both outlive the flag:**

* `_history_cos_sin` keyed its single-entry cache on `max_pos` and `.clear()`s
  on a miss. Two callers in the same prefill chunk ask for different extents, so
  they evicted each other and rebuilt a context-length rotary table every layer
  of every chunk. Now keyed on (model, device, dtype) with a longer table
  serving shorter requests by slicing — a view. This helps every prefill,
  routed or not: the un-routed arm went 13173 → 12772 ms at 32k.
* the per-block key box was rebuilt from ALL of a block's keys every time the
  block grew — O(n²) over a prefill. It is now incremental: only rows added
  since the cached box are measured, and min/max folds associatively. Pinned by
  `tests/test_prefill_key_boxes.py`, which also caught a real off-by-one in the
  first version of this (the growth path prepended the anchor's position for a
  row that is not in its slice).

---

## 5. TASKS — open, in priority order

**T1. Does CUDA lose the needle at depths nobody tests? (owner's report, believed)**
The suites that say "9/9" sample THREE depths: `validate_cuda_dkv.py` is
`[2k, 8k, 32k] x [0.0, 0.5, 0.9]`. `pool_stores_rotated_k`'s claim of "9/9 at
every depth and every length" was an overclaim and is corrected in that
docstring. `colab/needle_suite_cuda.py` samples 0.1 and FAILS there on
Qwen2.5-1.5B while all three validator depths pass.

Two candidate causes, and they must be separated before anything is changed:

* **The PTA phase error.** This is THE architectural CUDA/MLX divergence, and it
  is depth-dependent by construction. MLX ingests `keys_rot` (post-RoPE), so its
  reconstruction lands in each token's true frame and its only error is low-rank
  truncation. CUDA's default presets — `mid`, `high`, `ultra` — set
  `rotated_pool=False` (measured), so the decode gather rotates the anchor and
  the whole `V_K` basis at the ANCHOR's position and every compressed token
  carries a RoPE phase error of up to a full block. `_ingest_k` records the same
  gradient from the other side: at depth 0.0 the block sits near position 0
  where RoPE is ~identity, so it passes while deeper needles degrade. `low` is
  the only preset that stores what MLX stores.
* **The needle's tokenisation.** `niah_recall`'s `OMEGA-7741-DELTA` splits
  `' O'|'ME'|'GA'`, which `_assert_needle_unambiguous` exists to reject and
  which the repo measures as a 0.1875-logit coin flip on small models. The
  observed failure returns `7741-DELTA` — exactly the fragmenting prefix
  missing.

`colab/needle_depth_sweep.py` (added, NOT YET RUN) separates them: a fine depth
grid, the validator's unambiguous needle, and a DENSE control at every point, so
a depth where both fail is the prompt and a depth where only DKV fails is DKV.
Run it at 8k and 32k, then — only if DKV alone fails — A/B `DKV_ROTATED_POOL=1`
at the failing depths. Note that flag is measured as −18% to −24% decode and
+1.1 GB VRAM, so it is a trade, not a free fix.

**T2. Shared bases have still never seen an accuracy test.**
That was §2b's actual complaint. Its projection defects are fixed, its rotated-
pool guard is in, and its memory is measured — but `DKV_SHARED_BASIS=1` has
never been run through the needle suite, multifact, or the corrected logit
harness, all three of which now exist and work. Until it is, the feature is
unvalidated, not validated-and-clean.

**T3. Everything here is one model.** Every accuracy number in this record is
Qwen2.5-1.5B. The one time a second harness was brought in (multifact) it
immediately overturned a decision the other two had cleared. A second model is
the highest-value coverage this record is missing.

---

## 6. Closed, with the reasoning worth keeping

* **Prefill sparsity on an unrotated pool is CLOSED, not open.** It works, it is
  correct, and it does not pay — the two available levers are measured and
  rejected in §4c. Reopening it needs a genuinely different idea, not another
  pass at those two.
* **Nothing in this record has been measured on more than one model.** Every
  accuracy number here is Qwen2.5-1.5B, and the one place a second harness was
  brought in (multifact) immediately overturned a decision the first two had
  cleared. That is the most likely place a conclusion here is wrong.
* **A KEY COLLISION was found and fixed in passing.** `dkv_attention.py`'s
  combined branch and `hf_dkv_wrapper.py`'s pre-rotation both used the workspace
  key `"dense_rot_state"` for values of INCOMPATIBLE type — a dict vs a tuple
  `(sig, valid_len)`. Whichever wrote second poisoned the other, and the forward
  dereferenced it unguarded: `'tuple' object has no attribute 'get'`. Reachable
  only with the combined branch AND mutation-out both live, i.e. without the
  serving defaults — the same blind spot again. The wrapper's key is now
  `dense_prerot_state` and the forward's read is `isinstance`-guarded so a future
  collision degrades to a rebuild. Reproduced on 32d66345 to confirm it predates
  this work.
* **§3 of the old file (stale line numbers)** — every line number in this record
  is paired with a function name or a distinctive fragment for that reason.
