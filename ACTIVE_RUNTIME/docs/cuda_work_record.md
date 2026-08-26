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

**Current reading: 4/4, all arms identical.** It was reported as 3/4 for most of
this work, with an "arm-independent partial recall at 8k@0.1". That failure was
this harness, twice over, and both bugs are fixed:

* **the needle was contaminated.** It used `niah_recall`'s
  `OMEGA-7741-DELTA`, which Qwen splits `' O'|'ME'|'GA'` — the exact
  partial-word shape `validate_cuda_dkv._assert_needle_unambiguous` exists to
  reject, measured in this repo as a 0.1875-logit coin flip on small models.
  The suite now uses the validator's needle AND runs the unambiguity check, so
  swapping in a fragmenting one later fails loudly.
* **the answer extraction was length-based.** The needle is IN THE PROMPT, so a
  slice that misses the boundary scores the prompt. A token slice with a
  4-token backward margin reaches into the prompt's own copy of the needle at
  shallow depths, which is what produced `'9427-6183'`. Extraction is now
  anchored on the LAST occurrence of QUESTION — the prompt ends with it, so
  everything after is the completion and nothing else, at every depth.

Length-based answer extraction has now produced a wrong reading three times in
this file. Anchor on a marker, never on a length.

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

## 4d. THE HAYSTACK WAS HIDING A REAL DEFECT — recall on natural text

**Every needle harness here fills the context with ONE SENTENCE tiled to
length.** `niah_recall.FILLER` is 291 characters, 38 unique words;
`validate_cuda_dkv.py` builds from a list of eight. A random alphanumeric code
dropped into that is a colossal outlier, and DKV's residual budget spends its
slots on the WORST-RECONSTRUCTED tokens of each block — so the needle is all but
guaranteed one. The suites were measuring "is the needle distinctive", which it
is by construction, not "does the compressed representation retain it".

Swap the haystack for real papers already in this repo (`nat_paper.txt`,
`berry_paper.txt`, `random_features_paper.txt` — 1544 unique words in the first
alone), hold everything else fixed. Qwen2.5-1.5B, `mid`, block_size 256, needle
`Falcon-9427-6183`, DENSE control at every point:

| filler | ctx | dense | DKV |
|---|---|---|---|
| tiled sentence | 8k | 11/11 | **11/11** |
| tiled sentence | 32k | 11/11 | **21/21** |
| **natural text** | 8k | **12/12** | **3/12** |
| **natural text** | 32k | **12/12** | **3/12** |

Dense is perfect in every condition, so the prompt, needle and extraction are
sound. This is DKV.

**IT IS NOT A RETRIEVAL FAILURE.** The needle is found and corrupted:

    Falcon-9427-6185   for   Falcon-9427-6183
    "Falcon911"        for   Falcon-9427-6183

Right word, wrong digits.

**The residual budget is the dominant lever.** `DKV_MAX_RESIDUAL` 40 → 200 on
natural filler at 32k takes DKV from **3/12 to 8/12**. `DKV_EXACT_ROPE_REMAT=1`,
which removes the PTA phase error, gives **2/12** — no help, and an independent
confirmation that the phase error is not this.

So on realistic text the needle's tokens do not win one of the 40
worst-reconstructed slots in their block, come back through pure low-rank
reconstruction, and their digits flip. On tiled filler they always win a slot at
any budget, which is why every suite here reads 9/9.

**Raising it is not free.** Residuals are already 40 × 2·128·2·2 B = 40,960 B of
an 83,136 B slot — 49% of it. At 200 the slot roughly triples, which is most of
DKV's memory case. And 8/12 is still not 12/12.

**How this differs from the owner's external benchmark.** That one reports DKV
perfect in the LATE band, degrading early/mid, worsening with context, and
attributes it to the exact recency window shrinking as a fraction of the
sequence. This does not reproduce that shape: the rate is 3/12 at BOTH 8k and
32k, and the late band is 0/4 at 32k. Same family of conclusion — everything
outside the exact window is lossy — but the lever here is the residual budget,
not the window size. Different configurations; the gap has not been chased.

---

## 4e. T0 attacked and NOT fixed — five candidate causes eliminated

> **SUPERSEDED BY 4f, which fixes it.** The eliminations below stand and
> the closing prescription -- capture a RUN as a unit -- was right. What is
> wrong here is the LEVER: "the budget is the only thing that has moved
> recall" reads as "raise max_residual", and 4f shows every token exact
> still scores only 6/12. The fix spends the SAME forty slots on a
> complete run. Two eliminations below are also confounded by that
> defect and should not be carried forward without re-measuring: the
> `DKV_EXACT_ROPE_REMAT` row, and the query-capture row.

The defect of §4d, pursued. **It is not fixed.** What follows is the mechanism
and the eliminations, so the next attempt starts from here.

**The mechanism is PARTIAL CAPTURE of the answer's token run.** Reading the pool
after prefill and asking whether the needle's absolute positions appear in its
block's `residual_K_positions` — natural filler, 32k, 40 residual rows either
way, so this is purely WHICH rows are chosen:

| depth | needle tokens captured exact | answer |
|---|---|---|
| 0.50 | 8 of 17 | correct |
| 0.67 | 7 of 11 — the run STRADDLES a block boundary | `"Falcon-942"` |
| 0.83 | **3 of 17** | `"Falcon-947"` |

Failure tracks capture. Half a code is worth nothing: the right word and the
wrong digits. At 0.67 the run crosses a block boundary, so its tokens are split
across two independent 40-slot budgets and two independent error rankings,
neither of which knows about the other.

**Eliminated, each measured against the natural-filler sweep (baseline 3/12,
dense 12/12):**

| candidate | result |
|---|---|
| RoPE phase error (`DKV_EXACT_ROPE_REMAT=1`) | **2/12** — no help |
| coverage quota (`DKV_RESIDUAL_COVERAGE_FRAC=0.25 / 0.50`) | **0/12** — WORSE |
| rarity pass (`DKV_RESIDUAL_RARITY_CAPTURE=0`) | **3/12** — no change |
| atomic segment weighting (run shares its rarest IDF) | capture unchanged at 8/17; reverted as a dead knob |
| query-proximity capture (new, `DKV_RESIDUAL_QUERY_CAPTURE`) | capture 3→5 and 7→8 of 17, end recall **3/12** |
| **residual budget 40 → 200** | **8/12** — the only lever, and it roughly triples the pool slot |

Coverage is worse because CUDA's `_topk_with_coverage` EXCLUDES coverage
positions from the ranked selection, so at `frac=0.25` the answer competes for 30
slots instead of 40.

**Two hypotheses that sounded right and were wrong**, recorded because both will
be re-proposed:

* *"natural text is full of digits, so the shape heuristic stops discriminating"*
  — backwards. `is_core` fires on 4.5–5.8% of tokens in the papers against 14.5%
  in the tiled filler, i.e. 11–33 core tokens per block against a 40-row budget.
  Shape is not the constraint.
* *"the boost never fires on this path"* — it fires on every block, but boosts
  **256 of 256 rows**, so MLX's budget-floor formula (`boosted + n_cov + margin`)
  always evaluates to the full budget and the adaptive tier is inert.

**What was added and left OFF.** `DKV_RESIDUAL_QUERY_CAPTURE` boosts a
distance-weighted window around tokens whose id appears in the pinned query
(`manager._pending_query`, set by the wrapper before prefill). Selection is
otherwise blind to the query — it ranks by how badly the low-rank basis fits a
token, never by whether anyone will ask for it, which is exactly the difference
against attention-selected methods. It measurably captures more of the answer and
does not change end recall, so it ships off with that number attached rather than
being deleted or switched on.

**Where the next attempt should start.** The budget is the only thing that has
moved recall, and the reason is that an answer needs MOST of its run exact.
Selection is per-token and per-block; an answer is neither. Something that
captures a RUN as a unit, and can span a block boundary, is the shape of the fix
— not another per-token score. The methods that beat DKV on the owner's external
benchmark (SnapKV, kv_quant) have no position or distinctiveness dependence at
all.

---


## 4f. T0 FIXED — selection was PER-TOKEN and an answer is a RUN

§4e's prescription was right and its diagnosis of the lever was not. "The budget
is the only thing that has moved recall" pointed at `DKV_MAX_RESIDUAL_TOKENS`;
the thing that actually moved recall was making the same forty slots hold a
COMPLETE run instead of a truncated one. Nothing here raises a budget.

**Environment for every number below:** RTX 4070 SUPER, Qwen2.5-1.5B-Instruct,
preset `mid`, `block_size` 256, `max_residual` 40, needle `Falcon-9427-6183`,
natural filler, DENSE control at every depth, twelve depths at k/12.

### Three candidate causes eliminated first, each with its reading fixed before the run

| arm | 8k natural | reading |
|---|---|---|
| dense control | **12/12** | the prompt, needle and extraction are sound |
| DKV, as shipped | **2/12** | the defect, reproduced |
| `DKV_TOPK_BLOCKS=0` (attend ALL blocks) | **2/12** | **routing is NOT the cause** |
| `DKV_MAX_RESIDUAL_TOKENS=256` (every token exact) | **6/12** | the budget is not the whole story either |
| `DKV_RESIDUALS_IN_DENSE=1` | **2/12**, byte-identical at all 12 depths | the flag is INERT on this path |

Two of these are worth keeping.

**`DKV_RESIDUALS_IN_DENSE` does not reach the default decode path.** It is a
Triton-kernel flag, and this configuration logs `REMAT ACTIVE — materialise-then-
SDPA (MLX's decode form)`. The output is byte-identical at every depth with it on
and off. Anyone reaching for it as the MLX-partition fix is measuring nothing.

**Every token exact is still only 6/12**, and some of its failures are WORSE than
the baseline's (`Falcon-1000`, `20212021`, `Falcon942` against the baseline's
tidy `Falcon-9427-618x`). So "the answer's rows were not exact" is not a
sufficient account of this defect, and neither raising the budget nor spending it
more cleverly can reach dense on its own.

### The mechanism, read out of the pool rather than inferred from the answer

For the block bracketing the needle, which of the needle's own rows appear in
`residual_K_positions` after prefill. `n_res_written` is 40/40 throughout, so the
adaptive tier is NOT capping anything — this is purely WHICH rows were chosen:

| depth | layer 13 captured | model answered |
|---|---|---|
| 0.83 | `' Falcon' '-' '9' '4' '2' '7' '-'` — **dropped `'6' '1' '8' '3'`** | `Falcon-9427-6137` |
| 0.50 | all but the final `'3'` | `Falcon-9427-6185` |
| 0.25 | 5 of 11 | `Falcon`, and nothing after it |

The model reproduces EXACTLY the captured prefix and invents the rest. Capture is
also different at every layer, because each layer ranks its own reconstruction
errors independently.

That is the whole defect. Qwen splits `Falcon-9427-6183` into eleven tokens;
selection ranks tokens one at a time and takes the top forty; the tail of the run
loses. **Half a code is not half an answer — it is a wrong answer with the right
shape**, so the seven slots spent on the surviving prefix bought nothing.

### The fix — `atomic_runs` + run-atomic selection, at the SAME budget

`residual_capture.atomic_runs` returns the spans that are worth nothing unless
kept whole: a maximal non-prose segment carrying a core token, extended back over
its OWNER (`Falcon` is prose by shape, so only the owner walk-back pulls it in),
merged across a spacing artefact but never across a sentence or line break, and
dropped when a single segment is longer than `DKV_RESIDUAL_RUN_MAX` (32) because a
whole table row is not an atomic unit.

`lowrank._select_residual_rows` then REORDERS the existing error ranking: whole
runs, best run first, all-or-nothing, fill the slots the pool will actually keep
(`_res_cap`, not `n_max_residual` — the content boost can raise the latter to 256
and the pool would truncate the runs at 40 all over again), and everything else
keeps its old relative order behind them. Same row count in, same row count out.

A run cut by a block edge needs no special case: it is whole within its own block
and its other half is whole within the neighbour, which is what lets a straddled
answer survive two independent forty-slot budgets — §4e's "can span a block
boundary", for free.

`DKV_RESIDUAL_RUN_ATOMIC=0` restores the old ranking exactly, and gates the extra
pass so an A/B of the flag does not pay for it in both arms.

Capture after the change, same probe, same budget: **11 of 11 at essentially
every layer** at all three depths.

### Results

| | dense | DKV before | DKV after |
|---|---|---|---|
| needle 8k TILED (the old suites' filler) | 11/11 | 12/12 | **12/12** |
| needle 8k natural | 12/12 | **2/12** | **11/12** |
| needle 32k natural | 12/12 | **3/12** (§4d) | **9/12** |
| multifact relational 16k | — | 4/4 | **4/4** |
| multifact multi-needle 16k | — | 3/3 | **3/3** |
| multifact synthesis 16k | — | 13.3 | **6.7** |

The relational row — the binding test NIAH cannot see — is unchanged at 4/4, and
`validate_cuda_dkv.py --long` on Qwen3.5-2B stays ALL CHECKS PASSED (9/9 recall,
9/9 determinism, `fallback_count=0`), so nothing the old suites can see moved.

The tiled row is the point of §4d restated: on that filler the needle is a
guaranteed outlier, it already won its slots, and a fix aimed at the real defect
correctly does nothing there. Do not gate on it.

### What it costs in TIME

`colab/bench_prefill_paired.py`, which grows a `run_atomic` arm here. A/A control
run first and it reported NO EFFECT RESOLVABLE, CI [-51.8, +70.7] ms, so the
harness resolves what it is being asked to resolve.

    as first written           +427 ms   15.9%   CI [+362, +492]
    batched score transfer     +419 ms   15.5%   CI [+348, +490]
    no device sync at all      +177 ms    6.6%   CI [+107, +247]
    numpy-vectorised reorder   +214 ms    7.9%   CI [+170, +257]

A third run of the same arm after 4g's query-ranking pass was added read
**+56 ms / +1.8%, CI [-43, +155]** -- NO EFFECT RESOLVABLE at its own +-3.6%
resolution, which is wider than the effect the tighter runs measured. Three
paired runs across three processes therefore read 6.6%, 7.9% and 1.8%. Take the
prefill cost as single-digit percent and NOT precisely pinned: the paired CI is
within-run, and these differ by more than it across runs.

Two thirds of the original cost was a DEVICE SYNC per block per layer, and it
took three tries to remove because it kept moving: `scores.cpu()`, then
`base.indices.tolist()`, then a boolean mask index -- whose output shape is
data-dependent, so it synchronises too. Each drain waits on that block's queued
int8-recon matmul instead of letting the pipeline run ahead, ~900 times on an 8k
prefill. Most of the rest was `float(cpu_tensor[lo:hi].max())` paying the torch
dispatcher ~15 us a call, ~21k calls; numpy does the same slice in ~2 us.

**Decode and memory are unchanged.** This is a compression-time reordering of a
fixed number of rows: the pool stores the same `max_residual` residuals and the
decode kernel reads the same array.

### What this COSTS in accuracy — RECOVERED IN 4h, read that first

> The synthesis regression below is real and was measured correctly, but
> it is no longer the shipped behaviour. It came from reserving a whole
> run for EVERY atomic span; scoping the reservation to the runs the
> query points at (4h) restores synthesis to 13.3 AND takes 32k needle
> recall to 12/12. The analysis below stands as the reason the scope
> matters.

**Synthesis regresses 13.3 → 6.7** (facts 4/15 → 2/15), each setting reproduced
twice and identical both times, so it is real and not run-to-run spread. Run
reservation spends slots on codes and entities, and document synthesis is made of
the scattered rare PROSE words the rarity pass protects.

Capping the reservation is the obvious guard and it does not work.
`DKV_RESIDUAL_RUN_FRAC=0.5`, i.e. at most 20 of the 40 slots reservable:

    needle 8k natural     11/12 -> 10/12   (d=0.25 'Falcon-4278-6183')
    synthesis 16k          6.7  ->  6.7    (reproduced twice)

It costs recall and recovers none of the synthesis loss, so the cost is not slot
VOLUME. The knob was removed rather than shipped, on the same reasoning §4e used
to revert atomic segment weighting: a parameter whose only measured effect is
negative is worse than no parameter. The remaining account of the synthesis loss
is that it is compositional — which forty rows, not how many — and it is NOT
explained here.

### What is still open

* **8k depth 0.0 (1 of the 2 remaining 8k failures).** The needle lands in block
  0, which the deferred path releases as `force_exact` — and `force_exact` is
  only lossless while `T_active <= _res_cap`. At 256 active rows against 40 slots
  it falls through to the same ranked selection as any other block, so the
  comment at `streaming_sparse_ingest._is_block_compression_eligible` claiming
  block 0 is "protected from LOSSY compression by being compressed LOSSLESSLY" is
  false for every block size this runtime actually uses. Block 0 also carries the
  chat template and the paper's title, authors and URLs — 14 runs, 63 tokens, for
  40 slots — so the answer's run loses the run ranking at some layers. It does
  not reproduce at 32k, where depth 0.0 passes.
* **32k depths 0.33 / 0.50 / 0.58.** The signature CHANGED: `Falcon-94276-831`
  and `Falcon-94276-618` carry the right digits in nearly the right order with
  the separator misplaced, where the pre-fix failures dropped the tail outright.
  That is positional, not content, which makes the Project-Then-Attend phase error
  the obvious next suspect — and §4e's elimination of it (`DKV_EXACT_ROPE_REMAT=1`
  gave 2/12) was measured while capture was still truncating the run, i.e. under a
  condition where it could not have helped. **That elimination should be
  re-measured now, not trusted.**
* **`max_residual` 40 against MLX's 128** (`mlx_dkv_wrapper.py:1964`) remains an
  unexplained divergence in the exact quantity this section is about.

---


## 4g. The rest of the gap — routing, not representation

§4f left 8k at 11/12 and 32k at 9/12 against dense's 12/12. This closes 8k and
takes 32k to 11/12. Same environment as §4f throughout.

### The measurement that reframed it

| arm, 32k natural | result | reading |
|---|---|---|
| §4f as shipped | 9/12 | |
| `DKV_MAX_RESIDUAL_TOKENS=128` (MLX's own default) | **8/12** | budget is not the lever — it is WORSE |
| `DKV_TOPK_BLOCKS=32` (double the routed budget) | **9/12** | no change |
| **`DKV_TOPK_BLOCKS=0`** (attend every block) | **12/12** | **dense parity** |

Attend-all reaching 12/12 is the decisive one: with run-atomic capture at a
budget of 40, the compressed REPRESENTATION is already good enough for
dense-equal recall. There is no third lossy path. Everything still missing was
the router declining to look at the block.

And K=32 changing nothing while K=all fixes everything says the needle's block is
not ranked 17th–32nd of ~128 — it is ranked far down. That is a RANKING failure,
so a bigger K cannot buy it.

### Fault 1 — the router scored a key that exists in no frame

`route_blocks_relevance` rotates the ANCHOR at the anchor's position, rotates each
RESIDUAL row at its own true position, and then adds the two scores:

    s_res = q . R(p_r)(res)  +  q . R(p_anchor)(anchor)

A residual row's true key is `R(p_r) . (anchor_raw + res_raw)`, so that sum is the
true score for neither frame. It is the same frame split
`triton_fused_decode.pool_stores_rotated_k` describes for decode — "the two live
in different frames and their sum is not the exact key in either one" — except
here it decides which blocks are VISIBLE AT ALL, and a buried code's entire claim
on the top-K lives in that term.

Fixed by reconstructing the exact key BEFORE rotating it: fold the raw anchor into
`rk`, rotate the sum at each row's true position, and stop adding `s_anc`
afterwards. `DKV_ROUTER_EXACT_KEY=0` restores the split form.

**32k 9/12 → 10/12.** Decode cost: NO EFFECT RESOLVABLE, +0.150 ms/token, CI
[-0.139, +0.439], A/A control run first at ±1.4% of a token.

**This only exists on an UNROTATED pool**, which is the case that rotates anything
here — and preset `mid` sets `rotated_pool=False`. Two comments in this repo say
otherwise and are stale: `config.py`'s "low/mid/high keep rotated_pool=True", and
`pool_stores_rotated_k`'s "DEFAULT ON". The config object built from preset `mid`
reports False, and `DKV_ROTATED_POOL` lands in the environment as `0`.

Note what did NOT work: `DKV_ROUTER_ROPE=0`, which deletes the rotation entirely,
also scores 9/12 — the same count with a DIFFERENT set of failing depths
(0.33/0.58/0.83 against 0.33/0.50/0.58). Removing position-awareness just shuffles
which needle survives. The frame has to be repaired, not discarded.

### Fault 2 — the run ranking spent the budget on runs nobody asked for

With capture now all-or-nothing, the only question left is WHICH run wins the
slots, and §4f ranked them by reconstruction error alone. Reading the needle's own
rows out of the pool at 8k depth 0.0, per layer:

    24 of 28 layers   captured 11 of 11
    L16 L18 L26 L27   captured  0 of 11

Block 0 holds the chat template plus the paper's title, authors and a GitHub URL —
14 runs, 63 tokens, for 40 slots. Those competitors are genuinely hard to
reconstruct and completely irrelevant to "what is the secret passcode", which is
the one thing an error score cannot express. All-or-nothing turns losing that
ranking into losing everything, and L26/L27 are the layers that most directly
shape the emitted token.

`residual_capture.rank_runs_by_query` annotates each run with whether it sits
within a window of a QUERY term, and the greedy fills priority runs first. On the
real block 0 exactly 3 of 14 runs are marked and the needle's is the longest of
them.

**8k 11/12 → 12/12 — dense parity.**

**This is NOT the query boost that failed in §4e.** That one multiplies a
per-token score and could not lift a code at all: a core token already sits at
`tok_boost * idf/2` = 24 and the query pass writes `if w > boost[i]`, whose
maximum is also 24. It moved capture and left recall at 3/12. Ordering whole runs
is a different lever, and it only became available once selection was atomic.

**Self-limiting by construction.** If more than half a block's runs look
query-relevant the signal carries no ranking information and is dropped. That is
the production case, where `_pending_query` falls back to the whole prompt when no
question span can be extracted — a fallback that would otherwise mark everything
relevant and quietly randomise the order.

### Where it stands

| | dense | before §4f | after §4f | now |
|---|---|---|---|---|
| needle 8k natural | 12/12 | 2/12 | 11/12 | **12/12** |
| needle 32k natural | 12/12 | 3/12 | 9/12 | **11/12** |
| multifact relational 16k | — | 4/4 | 4/4 | **4/4** |
| multifact multi-needle 16k | — | 3/3 | 3/3 | **3/3** |
| multifact synthesis 16k | — | 13.3 | 6.7 | **6.7** |

`validate_cuda_dkv.py --long` on Qwen3.5-2B stays ALL CHECKS PASSED. The synthesis
cost recorded in §4f is unchanged by either fault above — it belongs to
run-atomic selection and is still unexplained.

### Still open

* **32k depth 0.58, the last failure.** `DKV_EXACT_ROPE_REMAT=1` takes 32k to
  **11/12** on its own and composes with both fixes above to the same 11/12; 0.58
  survives every combination tried, including attend-all-plus-everything-exact
  reaching 12/12, which means it is reachable — just not by these levers.
* **`DKV_EXACT_ROPE_REMAT` is now ON by default — DECIDED, not left open.**
  §T1b kept it off on a 10/11-against-11/11 result measured on the tiled filler
  and under the pre-§4f capture defect. Its own gate run:

      natural 32k   OFF 10/12   ON 11/12
      natural 8k    OFF 12/12   ON 12/12
      tiled 8k/32k  ON 12/12 / 12/12   (the regression that kept it off is GONE)
      validate_cuda_dkv --long          ALL CHECKS PASSED
      multifact 16k  relational 4/4, multi-needle 3/3, synthesis 6.7 unchanged
      decode cost    -0.055 ms/token, CI [-0.655, +0.545], NO EFFECT RESOLVABLE

  So 32k natural ships at **11/12**, and the last failure is depth 0.58 alone.
* **`max_residual` 40 against MLX's 128 is ANSWERED, and the answer is no.**
  Raising it to MLX's default scores 8/12 against 40's 9/12. The divergence is
  real and is not what separates the two runtimes.

---

## 4h. DENSE PARITY — the reservation only ever needed to cover the query

§4f bought needle recall by reserving a whole run for EVERY atomic span in a
block, and paid for it in document synthesis (13.3 → 6.7). §4g closed 8k and left
32k at 11/12. Scoping the reservation to the runs the query actually points at
closes both, and gives the synthesis back.

`DKV_RESIDUAL_RUN_RESERVE` (default `query`) chooses which runs may claim slots
all-or-nothing:

* `query` — only runs marked by `rank_runs_by_query`. If NO run is marked, the
  block reserves nothing and the whole budget goes to the per-token ranking.
* `all` — §4f's behaviour, every run best-scoring first.

### The synthesis cost was compositional, and nothing around the selection touched it

Before changing anything, the other candidates were eliminated. multifact
synthesis at 16k, Qwen2.5-1.5B:

| arm | score |
|---|---|
| shipped (K=16, remat cache on) | 6.7 |
| `DKV_TOPK_BLOCKS=0` — attend every block | 6.7 |
| `DKV_REMAT_CACHE=0` — no decode staleness | 6.7 |
| both | 6.7 |
| `DKV_RESIDUAL_RUN_ATOMIC=0` | **13.3** |

Routing does not move it. Decode staleness does not move it — even though
`remat_cache.remat_enabled`'s own docstring records a synthesis cost for the
cache on this exact model, that cost is not what is happening here. ONLY which
forty rows are spent moves it, which is what made the reservation's scope the
thing to change.

### Result — strictly better on every gate, not a trade

Dense control 12/12 at every point, twelve depths at k/12:

| gate | `all` (§4f/§4g) | `query` (shipped) |
|---|---|---|
| needle 8k natural | 12/12 | **12/12** |
| needle 32k natural | 11/12 | **12/12** |
| needle 8k tiled | 12/12 | 12/12 |
| needle 32k tiled | 12/12 | 12/12 |
| multifact relational 16k | 4/4 | 4/4 |
| multifact multi-needle 16k | 3/3 | 3/3 |
| multifact synthesis 16k | 6.7 | **13.3** |
| `validate_cuda_dkv --long` | ALL PASSED | ALL PASSED |

**Natural-text needle recall is now at dense at BOTH contexts**, from 2/12 and
3/12 where this work started, and §4f's synthesis regression is gone rather than
merely documented.

The 32k row deserves a note. Under `all`, depth 0.58 failed and NO other lever
reached it — not `DKV_TOPK_BLOCKS=32`, not a doubled residual budget, not
`DKV_EXACT_ROPE_REMAT`, not the §4g router fix. Handing the un-asked-for runs'
slots back to the error ranking did. So the last failure was not a missing
guarantee, it was a guarantee spent in the wrong place.

### Why this is safe where §4e's query work was not

The signal is the same pinned query. What differs is where it is applied:

* §4e multiplied it into a per-token score, where it could not move a code at all
  — a core token already sits at the pass's own ceiling of 24 — and left recall
  at 3/12.
* §4g used it to ORDER whole runs, which only became meaningful once selection
  was atomic.
* This uses it to SCOPE the reservation, which is the first place where being
  wrong is cheap: an unmarked run is not evicted, it simply competes on
  reconstruction error exactly as it did before §4f existed.

That asymmetry is the safety argument. `rank_runs_by_query` already drops its own
signal when more than half a block's runs look relevant, which is the degenerate
production case where `_pending_query` falls back to the whole prompt; combined
with `query` mode, a useless query now costs nothing rather than reserving
everything.

### Still open

* **`DKV_TOPK_BLOCKS=0` reaching 12/12 is no longer the only route there**, and
  it was never a shippable one. Attend-all is a GROUND-TRUTH mode, not a
  configuration: measured with `colab/bench_decode_paired.py` at 33,637 tokens,
  `EXPERIMENT=topk TOPK_ON=0`, it costs **51.1% of decode** — 17.86 → 11.82
  tok/s, +28.6 ms/token, CI [+28.2, +29.0], resolution ±0.8% of a token. The
  same A/B at 8,437 tokens is NO EFFECT RESOLVABLE, because there K=16 already
  keeps 16 of ~33 blocks; the cost is the ratio, so it grows with context and
  32k is where it bites. Sparse attention exists to not pay that, and recall no
  longer needs it.

  It remains true that the router ranks a needle's block far below 32 of ~128,
  which is worth understanding on its own — but it is no longer costing recall.
* **Synthesis is 13.3 against a >= 30 bar** and dense could not be measured here
  (`multifact_eval_cuda --dense` forwards unchunked and OOMs at 16k on a 12 GB
  card). This work restored what §4f cost; it did not make synthesis good, and
  nothing in §4f–§4h ever moved it above the pre-existing 13.3.

---

## 5. T1 and T2 — DONE. What the depth question actually was.

### T1. SUPERSEDED BY §4d — this held only for the TILED haystack

> **Read §4d first.** Everything below is correct for the filler these suites
> use, and that filler is the problem: on natural text the same sweep reads
> **DKV 3/12 against dense 12/12**. The conclusion "no depth-dependent CUDA
> failure survives a correct instrument" was true of the instrument and false of
> DKV. The needle and the extraction needed fixing, and so did the haystack.

### T1 (as measured, on tiled filler)

The owner reported CUDA losing needles at depths MLX handles, and the suites
could not have refuted it: `validate_cuda_dkv.py` samples three depths
(`[2k, 8k, 32k] x [0.0, 0.5, 0.9]`), and `pool_stores_rotated_k`'s claim of
"9/9 at every depth and every length" was an overclaim, now corrected in that
docstring.

`colab/needle_depth_sweep.py` answers it properly: eleven depths, the
validator's unambiguous needle, and a DENSE control at every point (its dense
arm chunks the prefill — `model.generate` on a 32k prompt asks for 46 GiB on a
12 GB card, so the control was the arm that could not run at the context that
mattered).

| ctx | dense | DKV |
|---|---|---|
| 8k, 11 depths | 11/11 | **11/11** |
| 32k, 11 depths | 11/11 | **11/11** |

So on this model, with a clean needle and correct extraction, CUDA matches dense
at every depth. **The reported failure was instrument, not engine** — see §3 for
the two bugs, both mine, both in harnesses written during this work. That is not
a refutation of the owner's observation in general: it is one model, and any
harness still using `niah_recall`'s needle reproduces the same false failure.

### T1b. The PTA phase error is real, the fix works — and it SHIPS ON as of 4g

> **The heading below said "and it still ships OFF". That is no longer
> true.** Both rows of the table were measured on the TILED filler, which
> §4d showed cannot see this defect, and under the pre-§4f capture defect.
> Re-measured on natural text with run-atomic capture: 32k **10/12 → 11/12**,
> 8k 12/12 either way, tiled 12/12 at both contexts, every other gate
> unchanged, and decode cost NO EFFECT RESOLVABLE. The docstring's own
> guess — "the old path was lucky at a coin-flip margin" — was right, and
> the margin is gone. Default flipped ON; see §4g.

The one architectural CUDA/MLX divergence is real and worth naming: MLX ingests
`keys_rot`, so its reconstruction lands in each token's true frame and its only
error is low-rank truncation. CUDA's default presets set `rotated_pool=False`,
and the decode gather rotates the anchor and the whole `V_K` basis at the
ANCHOR's position — so a token j into its block carries j positions of RoPE
error, and the exact residual (rotated at its TRUE position) corrects a base in
a different frame.

`DKV_EXACT_ROPE_REMAT` removes it: gather raw, reconstruct raw, then rotate the
MATERIALISED key at each row's own absolute position. Affordable only on the
remat path, which already materialises the keys, and it sits inside the
RematCache entry so it is paid per refresh rather than per token.

| | exact-RoPE ON | OFF (default) |
|---|---|---|
| decode-step KL, 8k | **0.00029** | 0.00125 |
| depth sweep, 8k | 11/11 | 11/11 |
| depth sweep, 32k | **10/11** | 11/11 |

At 32k depth 0.80 it returns `Falcon-9427-6123` for `...-6183` — deterministic
over repeats, one digit wrong, which is the "right letters, wrong digits"
signature this repo already associates with a RoPE phase error. So the more
accurate keys measure 4.3x closer to dense in KL and lose the needle anyway.
Recall is the gate, so the default follows the recall column. **Not understood**
— the position mapping matches three independent statements of the block layout,
the raw gather is complete, and the routed set is identical either way. The
likeliest explanation is that the anchor-frame error was suppressing a
competitor at a coin-flip margin, but that is a hypothesis.

### T2. Shared bases ARE validated now — §2b's actual complaint, answered

§2b's real objection was "unvalidated against any working accuracy test". Its
defects were fixed, its guard added and its memory measured, but the feature had
never been run through an accuracy suite. All three now exist and all three were
run at `DKV_SHARED_BASIS=1, frac=0.50` (unrotated pool, which the default preset
already gives):

| harness | OFF | ON |
|---|---|---|
| needle suite (4 cases) | 4/4 | **4/4** |
| multifact multi-needle | 3/3 | **3/3** |
| multifact relational (BINDING) | 4/4 | **4/4** |
| multifact synthesis | 13.3 | 10.0 |
| logit fidelity, first token | 0.00024 | **0.00024** |
| logit fidelity, decode step | 0.00125 | **0.00146** |

No regression anything here can resolve. The synthesis move is inside the
±15-point RSVD-seed band this repo already records for that metric, and both
numbers are far below its ≥30 bar, which this model does not clear with every
feature off either. The row that would have mattered — relational, the binding
test that caught KMIN=2 — is unchanged at 4/4.

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
