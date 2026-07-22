# CUDA active-runtime: why VRAM is high and throughput is low (2026-07-17)

Investigation of the A100 NAT-eval run (Qwen2.5-14B-Instruct nf4, 13.4K prompt,
48 layers, 8 KV heads, head_dim 128). Every claim below is tagged **PROVEN**
(derived from the code plus arithmetic, reproducible on any machine) or
**HYPOTHESIS** (consistent with the numbers, needs the A100 to confirm).

---

## 1. The headline: the "26x KV reduction" is a broken counter — **PROVEN**

`kv_logical` reads **0.101 GB** in every DKV config, against dense's 2.643 GB.
That implies 26x. It is not real.

`analytic_kv_bytes` (colab/run_nat_eval.py) reads `mgr.sessions`, a property on
`KVRuntimeManager`. That property had two independent bugs, each of which alone
zeroes the store:

1. It iterated `self.session_blocks`. On the streaming path (the CUDA default)
   `init_session` creates `self.session_blocks[sid] = {i: [] for i in range(L)}`
   and **nothing ever appends to it** — the real blocks live in
   `self._streaming_mgr.session_blocks[sid]`. The eval's own diagnostic reads
   the streaming dict and prints 53 blocks; the metric read the empty one and
   got `num_blocks = [0]*48`.
2. It read `getattr(b, "slot_idx", -1)`. Blocks carry `pool_idx`
   (`StreamingKVBlock` dataclass field, streaming_sparse_ingest.py:151).
   `slot_idx` does not exist, so every residual count was 0 regardless.

With `nb = 0` and `res_tokens_used = 0`, the formula collapses to the dense
recency window alone:

```
store_used = L * dense_len * kv_tok = 48 * 512 * 4096 = 100,663,296 B = 0.1007 GB
```

which is exactly the reported 0.101 GB. The decisive tell: **the number is
byte-identical across all six configs** even though `max_residual` varies
40/64/128 and `pool_rank` varies 48/96. A real store measurement cannot be
invariant to those.

### What the real numbers are

`pool_physical_mb` was already in the JSON and is the honest figure:

| preset      | pool (real) | dense KV | **real ratio** | kv_logical implied |
|-------------|-------------|----------|----------------|--------------------|
| low         | 1039 MB     | 2643 MB  | **2.54x**      | 26.2x              |
| mid         | 1282 MB     | 2643 MB  | **2.06x**      | 26.2x              |
| high        | 1931 MB     | 2643 MB  | **1.37x**      | 26.2x              |
| early_boost | 1875 MB     | 2643 MB  | **1.41x**      | 26.2x              |

Cross-check: with the property fixed, the *logical* formula returns **1.039 GB**
for `low` — landing on the measured physical pool of **1039 MB** to three
significant figures. The pool is ~98% occupied (2544 blocks in 2592 slots) and
every block is full, so logical and physical genuinely coincide here. Both
routes now agree the store is ~1 GB, not 0.101 GB.

**This is the whole "kv cache shows lower but total VRAM is high" mystery.**
Total VRAM was always telling the truth; the KV metric was not. At `high`,
DKV's pool is only 1.37x smaller than the dense KV it replaces — while
costing 6.1 s of compression and 40% of decode throughput.

Fixed in `KVRuntimeManager.sessions`; the eval now also prints
`kv_phys=... vs dense ... = N.NNx REAL`.

---

## 2. The rank boost fires on 100% of blocks — **PROVEN**

`_block_boost_rank` (lowrank.py) raises a block to `ceil(rank*1.5)` when its text
contains any digit, or matches `_RE_MATH_BOOST`, or `_RE_DEFINITIONS_BOOST`.
`_RE_MATH_BOOST` begins with the character class `[\+\-\*\/=]` — **a single
hyphen matches**. Measured against the actual eval corpus
(`ACTIVE_RUNTIME/nat_paper.txt`, ~256-token blocks):

```
any-digit  fires: 47/47 = 100.0%
math regex fires: 46/47 =  97.9%
=> BOOSTED      : 47/47 = 100.0%
```

Direct unit check: `"we use self-attention in this model"` → rank 32 becomes 48.

So on technical prose this is not a heuristic, it is an unconditional 1.5x on
rank, wearing a heuristic's clothes. Consequences:

- `pool_rank = ceil(max_rank*1.5) = 48`, and it is **fully used** — so the pool's
  largest tensor `V_KV` is 1.5x bigger than "rank=32" implies. That is most of
  why `low` only reaches 2.54x.
- The rSVD runs at `r_proj = 48+5 = 53` instead of `32+5 = 37`.
- **MLX has no SVD-rank boost at all** — it runs a flat rank 32. Every `1.5` in
  `mlx_dkv_wrapper.py` is the model name "Qwen-1.5B". This is a CUDA-only
  divergence, and it is part of why MLX looks better earlier.

The log line `rank=32 (pool_rank=48)` reads like headroom. It is not headroom;
it is the operating point.

Added `DKV_RANK_BOOST=off` (MLX parity) and a per-session boost-rate counter
now printed by the eval. Default unchanged (`auto`) — this changes accuracy, so
**A/B it on the GPU**, don't assume.

---

## 3. Compress: ~99% of the 6.1 s is not arithmetic — **PROVEN (that it isn't math)** / **HYPOTHESIS (that it's cuSOLVER)**

The rSVD matmuls at the real shapes (49 blocks/layer, T=256, feat=2048, r_proj=53,
48 layers) total **784 GFLOP**:

```
A100 fp32, TF32 OFF : 0.040 s
A100 fp32, TF32 ON  : 0.005 s
measured compress   : 6.100 s
```

So **the math is 0.7% of compress**. Batching the finalization (last session's
work) was correct but attacked the wrong 6 seconds.

**HYPOTHESIS:** the cost is `torch.linalg.svd(B)` with `B = [49, 53, 2048]`.
cuSOLVER's genuinely batched SVD (`gesvdjBatched`) only covers matrices up to
32x32, so a 53x2048 batch falls back to a per-element loop → **49 x 48 = 2,352
sequential cuSOLVER SVDs per prefill**. At 6.1 s that is ~2.6 ms each, which is
the right order for `gesvdj` on that shape. `linalg.qr` on `[49,256,53]` may loop
for the same reason.

**Proposed fix (needs GPU confirmation):** the tail of an rSVD only needs
`U_b [r,r]`, `S [r]`, `Vh [r,feat]`. Those come out of the eigendecomposition of
the small Gram matrix `B B^T` (`[49,53,53]`):

```
B B^T = U_b diag(S^2) U_b^T ;   Vh = diag(1/S) U_b^T B
```

trading a 53x2048 SVD for a 53x53 symmetric eigh.

Numerics (measured on CPU at the real shapes, decaying synthetic spectrum):

| spectrum decay | cond(B) | SVD recon | Gram recon |
|----------------|---------|-----------|------------|
| 1.00           | 1.0e+00 | 5.7e-07   | 4.9e-07    |
| 0.90           | 2.4e+02 | 8.3e-07   | 5.5e-07    |
| 0.80           | 1.1e+05 | 7.1e-07   | 2.4e-04    |
| 0.70           | 1.1e+08 | 8.0e-07   | 2.2e-04    |

Squaring the condition number does cost the tail ~2.2e-4 relative. For scale:
the pool **already** stores U as int8 with a per-block scale, injecting
**9.2e-3** relative error — **41x larger**. The Gram route's error sits well
under the noise floor the pipeline already imposes, so it is an unlikely
candidate to move recall. That is an argument, not a result — A/B it.

**Run `colab/profile_compress_stages.py` on the A100 first.** It times all 11
stages with proper synchronisation, counts the SVD calls, and A/Bs the Gram
replacement. It needs no model download.

---

## 4. TF32 was never enabled — **PROVEN**

The run log contains PyTorch's own warning:

> TensorFloat32 tensor cores for float32 matrix multiplication available but not
> enabled.

No `allow_tf32` / `set_float32_matmul_precision` anywhere in the repo. The entire
compress path is fp32 (`deltas = flat_batch.float() - ...`), so it ran at ~19.5
TFLOPS instead of ~156. Now enabled in `_configure_cuda_allocator`
(`DKV_TF32=0` to opt out). Honest expectation: this only reclaims ~0.035 s of
the 6.1 s — the math was never the bottleneck. Take it, don't celebrate it.

Also fixed there: the function printed
`CUDA allocator config: garbage_collection_threshold=0.6, max_split_size_mb=128`
unconditionally, but it uses `os.environ.setdefault` and `run_nat_eval.py` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at import — so the message was
**false** on every eval run. It now prints what is actually in effect.

---

## 5. Decode: sparse attention cannot pay off at 13.4K on this GPU — **PROVEN**

This is the strategic answer to "MLX wins pretty early but CUDA seems off".

```
@ 13,440 tokens
  KV cache               : 2.64 GB   -> 1.70 ms/token to read ALL of it
  weights (14.7B nf4)    : 7.35 GB   -> 4.74 ms/token
  measured dense step    : 87.7 ms/token (11.4 tps)
  measured DKV step   : 149.3 ms/token (6.7 tps)   => +61.5 ms/token
```

**KV read is 1.9% of the dense decode step.** DKV attends ~5.5K of 13.4K
tokens (`N_sparse=16, L_dense=1419`), so the absolute most it can save is
**~1.0 ms/token** — while it adds **61.5 ms/token**.

For the saved KV bandwidth to merely pay for that overhead you would need
**~95 GB of KV, i.e. ~485,000 tokens of context**. KV only equals the *weight*
read at ~37,000 tokens.

At 87.7 ms/token, dense is ~20x off its own bandwidth roofline (4.7 ms) — decode
is bound by bnb-nf4 dequant and eager launch overhead, not by KV. DKV then
adds 48 layers of per-token routing/reconstruction on top, with **CUDA graphs
correctly disabled** (mutable routing state), so it pays full eager launch cost.

**Implication:** at 13.4K the eval is measuring DKV's overhead against a
benefit that is arithmetically ~2% of the step. This is not a tuning problem. The
run is far below the crossover, and no amount of kernel work on the sparse path
changes that ordering. MLX "wins early" partly because unified memory makes its
KV read a much larger share of its step, and partly because its baseline is a
1.5B model — different regime entirely.

The honest ways forward, in order:
1. **Report the crossover instead of hiding it.** Sweep 32K/64K/128K. That is
   where the architecture has something to show on an A100.
2. **Attack the 61.5 ms/token overhead**, which is the actual blocker. Until it
   is under ~2 ms it cannot pay for itself at any context this GPU can hold.
3. Stop quoting 13.4K decode tps as a DKV result. It is a measurement of
   overhead.

---

## 6. Pool composition — **PROVEN (arithmetic)**

Per slot at `low` (max_seq_len 257, pool_rank 48, max_residual 40), ~410 KB:

| component                    | bytes/slot | share | note |
|------------------------------|-----------:|------:|------|
| `V_KV`                       | 196,608    | 47.9% | scales with pool_rank (=48) |
| residual K+V values          | 163,840    | 39.9% | scales with max_residual |
| `U_fact` (fp16)              |  24,672    |  6.0% | **dead on CUDA GPU-compress** |
| `U` (int8)                   |  12,336    |  3.0% | the actual low-rank payload |
| `fact_anchors_K/V`           |  12,288    |  3.0% | **dead on CUDA GPU-compress** |
| `U_sem` + scale              |   6,240    |  1.5% | **dead on CUDA GPU-compress** |

~10.5% of every slot (~42 KB) is the legacy stratified/fact slots that only the
**CPU** compress path ever writes. `native_block_pool.py` already documents this
and deliberately defers it, because eight getters index those tensors directly
and would need guarding first. That reasoning still holds — it is ~110 MB at
13.4K. Worth doing only after the items above.

The two levers that actually matter are `pool_rank` (see §2) and `max_residual`
— which is exactly why `high` (max_residual 128) degrades to a 1.37x ratio.

---

## 7. Still unexplained — **HYPOTHESIS, needs the A100**

- **Prompt 2 costs more VRAM than prompt 1, every config.** `after_fwd` 12.58 →
  14.34 GB (+1.76), `after_comp` 11.46 → 12.13 (+0.67). Dense is flat at 13.27
  both. ~1.04 GB is explained (the pool is lazily allocated during prompt 1's
  *compress*, so prompt 1's `after_fwd` predates it). The remaining ~0.7 GB of
  live tensors is not explained. `clear_session` looks thorough on inspection.
- **DKV forward is 1.47x slower than dense** (8.62 s vs 5.88 s at matched
  chunk size), before any compression. Per chunk that is ~0.21 s of ingestion
  overhead — block partitioning, `torch.cat`, buffer copies — across 48 layers.
  Bigger chunks help (`high` at CH=2056 → 7.27 s), which fits a per-chunk fixed
  cost.
- **`factual_store` collapses to 1.2 tps** with prefill 31.7 s. Consistent with
  the 2026-07-06 finding that the factual store is net-negative (~32% slower);
  this is worse and worth a separate look.
- **Block 0 never compresses** (`state=ACCUMULATING`, `tcount=256`, full). 4 of
  53 blocks stay ACCUMULATING. Anchors advance by 257 while blocks hold 256
  tokens, so `block_capacity=257` vs `tcount=256` may be an off-by-one in
  eligibility.

---

## Changes made (none GPU-validated)

| file | change |
|------|--------|
| `native_core/kv_runtime_manager.py` | `sessions` reads the streaming block store and `pool_idx`; drops `_rank_boost_stats` in `clear_session` |
| `colab/run_nat_eval.py` | `analytic_kv_bytes` uses the real block size + pool rank, returns physical/dense-equivalent bytes; report prints the REAL ratio; prints boost rate |
| `native_core/compression/lowrank.py` | `DKV_RANK_BOOST=off`; boost-rate telemetry; documented the 100% fire rate |
| `serving/hf_dkv_wrapper.py` | TF32 on (`DKV_TF32=0` opts out); allocator log prints what is actually in effect |
| `colab/profile_compress_stages.py` | **new** — stage-by-stage compress profiler + Gram-eigh A/B |

Test suite: 27 files, all pass except three failures that are **pre-existing and
identical on clean `e42a593`** (`test_multidim_srl::test_dynamic_decay_and_weights`,
`test_reasoning_mitigations::test_python_dynamic_rank_boosting_decision`,
`test_residual_capture::test_table_rows_selected_as_residuals`).

Note: `tests/test_reasoning_mitigations.py::test_pool_rank_boosting` is a
tautology — it recomputes `ceil(16*1.5)==24` in the test body and never calls the
product code. It protects nothing.

## Suggested order on the A100

1. Re-run the eval. Confirm `kv_phys` and the REAL ratio, and the ~100% boost rate.
2. `python colab/profile_compress_stages.py` — attribute the 6.1 s. Then
   `--rank 32` and `--tf32` to separate the rank-boost cost from cuSOLVER.
3. A/B `DKV_RANK_BOOST=off` on VRAM, compress time, **and recall**.
4. If the profiler confirms the SVD loop, implement the Gram-eigh swap behind a
   flag and A/B recall.
5. Sweep 32K/64K/128K. That is where this architecture can actually win here.

---

# SECOND PASS — CONFIRMED ON A100 (2026-07-17)

The A100 run validated every prediction. Numbers below are measured, not
projected.

## Profiler confirmed the cuSOLVER hypothesis exactly

```
stage                                  seconds   share
9. linalg.svd [N,r,feat]  <-- SUSPECT    3.917   84.5%
7. linalg.qr  [N,T,r]                    0.330    7.1%
cuSOLVER (qr + svd)                      4.247   91.6% of compress
svd call count = 2352 sequential decompositions, 1.665 ms each
```

So compress is **91.6% cuSOLVER**, and the rSVD matmuls are ~0.1 s — as
predicted. The 6.1 s was never arithmetic.

## Metric fix confirmed — the honest ratio is now printed

```
low  : kv_phys=1.090GB vs dense 2.643GB = 2.43x REAL   (was "26x")
mid  : kv_phys=1.345GB vs dense 2.643GB = 1.97x REAL
high : kv_phys=2.025GB vs dense 2.643GB = 1.31x REAL
```

`kv_logical` also became meaningful (0.94–1.49 GB with `[blk0=53/54]`), landing
on the physical pool — both routes now agree the store is ~1–2 GB.

## Rank boost confirmed at 100% live

`[DIAG] rank boost fired on 49/49 blocks = 100.0%` in every DKV config.

## TF32 confirmed useless for compress → refactored to a scoped context manager

TF32 on: compress 4.453 s vs 4.635 s off — a **4%** difference, because compress
is cuSOLVER-bound. The first-pass change enabled TF32 **process-globally** in
`_configure_cuda_allocator`, which also alters the fp32 math in decode
reconstruction and the block router (perturbing generated output across presets
for a 4% compress win). Replaced with `lowrank._tf32_matmul()`, a context manager
scoped to the compression call only. `DKV_TF32=0` disables it.

## Gram-eigh swap: 1.9x on the SVD, proven equivalent, now implemented

Profiler on the same A100: `eigh(B Bᵀ)` = 2.106 s vs `svd` 3.917 s → **1.9x**,
projected compress 4.635 s → **2.824 s**. Reconstruction 8.2e-6 vs the SVD's
8.5e-6 (equal). Implemented behind `DKV_COMPRESS_GRAM_SVD=1` in
`compress_layer_blocks_gpu`; CPU parity verified that the downstream
reconstruction (`U_scaled @ Vh`, the KV the pool stores) is bit-identical to the
SVD path (1.2e-6, fp32 noise), and the existing compress parity tests pass with
the flag on. **Opt-in — A/B recall before defaulting.**

## THE REAL LEVER — the r≤32 cuSOLVER cliff is CONFIRMED, and it is a wall

`probe_batched_cliff` on the A100 measured a **130× cliff** in one rank step:

```
   r    eigh ms/call    svd ms/call
  30       0.0062         0.8710     <-- BATCHED
  32       0.0063         0.8838     <-- BATCHED
  33       0.8120         1.3483     <-- fell off the cliff
  48       0.9682         1.6100
  53       0.9778         1.6719
```

cuSOLVER's batched Jacobi solvers (`syevjBatched`/`gesvdjBatched`) run r≤32
genuinely batched; at r=33 they drop to a per-matrix loop. So across the 2,352
calls/prefill, Gram eigh at `r_proj ≤ 32` costs **~0.015 s** vs **~2.1 s** at
r_proj=53 — and QR (same cuSOLVER family) drops with it. **Gram + r_proj ≤ 32
takes compress from ~6.1 s to well under ~1 s** — the ~6x win, not the 1.9x that
Gram alone gives at r_proj=53.

The catch the probe exposed: `r_proj = max_rank + oversamples`, so even
`DKV_RANK_BOOST=off` (rank 32) gives 32+5 = **37, still over the cliff**. Two
knobs now make r_proj ≤ 32 reachable (both in `compress_layer_blocks_gpu`):

- `DKV_RSVD_OVERSAMPLES` (default 5) — randomized-SVD slack; the 2 power
  iterations already there cover most of what oversampling buys.
- `DKV_RSVD_MAX_RPROJ` (default 0=off) — hard cap on r_proj. Blocks that
  wanted a higher rank are capped to it. The per-block dynamic-rank clamp was
  updated so `dynamic_rank ≤ r_proj` (CPU-verified: forcing block_rank 48 with a
  cap of 32 yields dynamic_rank 32 and consistent U/V, no crash).

**Recommended batched recipe to A/B on recall:**
```
DKV_COMPRESS_GRAM_SVD=1  DKV_RANK_BOOST=off  DKV_RSVD_MAX_RPROJ=32
```
This caps r_proj 37→32 (dropping ~5 oversamples' worth of subspace slack; the
dynamic rank was already ≤32 with boost off, so the stored fidelity barely
moves), and should collapse compress. Verify recall (needle + synthesis) —
0 effective oversamples leans entirely on the 2 power iterations.

## Bugs fixed this pass (were blocking the run)

- **`lowrank.py` `import contextlib` was missing** — a `@contextlib.contextmanager`
  (`_tf32_matmul`, added when TF32 was scoped) referenced it, so the whole module
  failed to import. This broke `test_compress_gpu_smoke` and `test_vscale_parity`
  collection (both now pass). The A100 eval only ran because it used the earlier
  committed global-TF32 version.
- **Two tests errored pytest collection on the A100** —
  `test_decode_cache_fused_parity` and `test_dkv_kernel_parity` do a bare
  `import mlx.core`, which aborts the entire `pytest ACTIVE_RUNTIME/tests/` run on
  a box without MLX. Changed to `pytest.importorskip("mlx.core")` so they skip.

## factual_store / combined are catastrophic — confirmed

`factual_store` 1.9/1.8 tps, `combined` 1.5/0.8 tps, prefill 25–27 s. Consistent
with the 2026-07-06 finding that the factual store is net-negative; here it is
far worse. Keep it off; investigate separately if it is ever wanted.

## Corrected pre-existing failure count

There are **4** pre-existing test failures (the first pass said 3 — a `-x` run
masked the second `test_multidim_srl` failure). All four are identical on the
clean committed tree and unrelated to any change here:
`test_multidim_srl::test_dynamic_decay_and_weights`,
`test_multidim_srl::test_prime_node_and_adaptive_k`,
`test_reasoning_mitigations::test_python_dynamic_rank_boosting_decision`,
`test_residual_capture::test_table_rows_selected_as_residuals`. The first two
assert `10x in [range]` — nondeterministic SRL routing, stably failing.

## Updated next steps

1. **A/B the batched recipe on recall** —
   `DKV_COMPRESS_GRAM_SVD=1 DKV_RANK_BOOST=off DKV_RSVD_MAX_RPROJ=32`.
   Expect compress ~6.1 s → ~1 s. If needle + synthesis hold, this is the win;
   make it the CUDA default. (The cliff is confirmed; this is now an accuracy
   check, not a perf question.)
2. If recall dips, back off toward the cliff instead of over it:
   `DKV_RSVD_OVERSAMPLES=0` with rank 32 (r_proj=32), or a base rank of ~26
   with the default oversamples — both land r_proj at 32 with some slack.
3. `DKV_RANK_BOOST=off` also cuts pool VRAM (pool_rank 48→32) — fold its
   recall check into step 1 since the recipe already sets it.
4. Sweep 32K/64K/128K — still the only regime where sparse *decode* can win here
   (decode KV is 1.9% of the step at 13.4K; break-even ~485K tokens).

---

# THIRD PASS — batched recipe CONFIRMED, pool_rank waste fixed (2026-07-17)

## The recipe works: compress 6.1 s → 2.6 s, prefill 14.9 s → ~10.5 s

`DKV_COMPRESS_GRAM_SVD=1 DKV_RANK_BOOST=off DKV_RSVD_MAX_RPROJ=32` on the
A100:

```
preset   compress before   compress now   prefill now
low          6.2 s            2.6 s          11.0 s
mid          6.1 s            2.6 s          10.6 s
high         6.1 s            2.6 s           9.4 s
```

Recall signal is good: the first decoded token matches dense in every config
(`first_id=8813` / `785`), token counts unchanged. (Confirm full `output_text`
in the JSON before defaulting.)

**Why 2.6 s, not the projected ~1 s:** the eigh dropped to ~0.015 s as predicted,
which *exposed the finalization floor* — residual selection, pool writes, the two
fp32 `deltas`/token-norm copies, medians — that the 3.9 s SVD had hidden. That is
now the dominant cost. Further compress wins mean attacking finalization, with
diminishing returns; the big lever is spent.

## Fixed: pool_rank stayed 48 even with the boost off (VRAM did not drop)

The batched run left the memory ratio at 2.43× / 1.97× / 1.31× — unchanged — with
the log still showing `rank=32 (pool_rank=48)`. The pool's V_KV/U slots are sized
by `pool_rank = ceil(max_rank * 1.5)`, and that 1.5× is headroom *for the content
rank-boost*. With `DKV_RANK_BOOST=off` the boost never fires, so no block's
stored rank exceeds `max_rank` — the 1.5× over-allocates every slot by 50% for
capacity that cannot fill.

Fixed in `KVRuntimeManager`: the multiplier is now `1.5` only when the content
boost is active, `1.0` when `DKV_RANK_BOOST=off`, and `pool_rank` is
additionally capped by `DKV_RSVD_MAX_RPROJ`. Provably safe — compress receives
the base `manager.rank` and clamps every block's `dynamic_rank ≤ r_proj ≤` this
value (CPU-verified: a rank-32 pool accepts the capped blocks with no "V rank
exceeds pool" error). Default path (boost auto) is unchanged: still `pool_rank=48`.

Impact (low preset): slot 368 KB → ~300 KB (**~18% smaller pool**), so the batched
recipe should now move the ratio too — low ~2.43× → **~2.9×** — not just the
compress time.

**CONFIRMED on A100** (recipe run, `pool_rank=32` in every config):

```
preset   pool before   pool now   ratio before   ratio now
low       1039 MB       842 MB      2.43×          2.99×
mid       1282 MB      1085 MB      1.97×          2.32×
high      1931 MB      1733 MB      1.31×          1.45×
```

Slots: low 368→300 KB, mid 464→396 KB, high 721→653 KB. Worst-case ceilings fell
too (low 20.8→17.0 GB). First decoded token still matches dense every config.
Note: with `DKV_RSVD_MAX_RPROJ=32`, `early_boost` collapses to mid (pool_rank
capped 96→32, ratio 2.32×) — the cap correctly dominates the early-layer boost,
so that config is redundant under this recipe.

## Where this leaves it

- **Prefill**: compress is no longer the story (2.6 s of a ~10.5 s prefill; the
  ~8 s forward is now the bulk — that is the model, not DKV overhead).
- **Memory**: with the pool_rank fix, the boost-off recipe should land low near
  ~2.9×, mid ~2.3×, high ~1.5×. `high` (max_residual 128) stays weakest — its
  ratio is residual-bound, not rank-bound.
- **Decode**: unchanged and still not the place to invest at 13.4 K.
- **Open**: `factual_store`/`combined` remain catastrophic; the prefill-forward
  1.4× vs dense and the prompt-2 VRAM creep are still unexplained.

---

# FOURTH PASS — why MLX wins early and CUDA doesn't (2026-07-18)

The user's question: MLX beats dense on memory early in context; CUDA takes
~the same peak RAM as dense while being slower. Diagnosed from the eval numbers
and the code — no theorizing.

## Peak VRAM: DKV is lighter at REST but heavier at the PEAK

```
low_preset prompt1        VRAM
  dense peak              13.27 GB
  DKV after-compress   11.25 GB   <- 2.0 GB LIGHTER than dense (the pool wins)
  DKV PEAK             15.07 GB   <- 1.8 GB HEAVIER than dense (the spike)
```

The pool (842 MB) genuinely beats dense's KV (2.6 GB) — `after_comp` proves it,
11.25 < 13.27. But `max_memory_allocated` (what OOMs you, and what the eval
reports) is the **compression spike**: `weights + full raw KV (2.6 GB) + fp32
compression transients + the pool being built`, all resident at once.

**Root cause:** CUDA runs *exact prefill* — it holds every token's raw KV for the
whole prompt (`after_fwd` = 12.58 GB) and compresses only at the boundary. So at
the peak moment the full raw KV and the new pool coexist.

## Why MLX doesn't have this spike

`MLXKVBlockManager.compress_deferred_prefill_blocks` is *"safe (and intended) to
call after EVERY prefill chunk... keeps peak uncompressed KV bounded by
~(recency_window + chunk) tokens instead of the whole prompt"*. MLX compresses
each chunk as it goes and frees the raw KV, so its peak never contains the whole
prompt's raw KV. **That is the entire "wins early" difference** — not the
compression quality, the *timing* of it.

## The fix already exists but is off by default: `DKV_STREAMING_COMPRESS=1`

The eval's CUDA loop calls `compress_deferred_prefill_blocks` + `empty_cache`
after each chunk when `DKV_STREAMING_COMPRESS=1`. `compress_deferred_blocks`
only compresses blocks past the recency window (`window_ok`), and the fresh-
prefill attention path already handles compressed history blocks (it splits
`comp_blocks` vs dense in the history cross-attention). So streaming should bound
peak raw KV to ~(512 + chunk) ≈ 1.5 K tokens instead of 13.4 K — dropping peak
from 15 GB toward ≤ dense. Cost: far-back history is attended in lossy compressed
form during prefill (exactly MLX's tradeoff), plus the per-chunk `empty_cache`
slows allocation slightly. **This is the lever for "CUDA peak ≥ dense" — A/B it.**

## Why prefill FORWARD is 1.4× dense (8.2 s vs 5.9 s)

The fresh-prefill path (`dkv_attention.py` ~2242) does, per chunk: local causal
self-attention **plus** history cross-attention that re-assembles previous blocks
(`torch.cat` of per-block K/V), re-applies RoPE to all history, `repeat_kv`s it,
and LSE-merges the two streams. Dense does one fused flash-attention over a
contiguous growing cache. The re-assembly + redundant RoPE per chunk per layer is
the overhead — architectural to attending a *block* store during prefill. MLX
keeps a contiguous dense prefill buffer and avoids it. Closing this on CUDA means
a contiguous raw-prefill cache (big change; GPU-only to validate).

Small safe win taken: `unrot_query_states` was `.clone()`d in full every layer
every chunk, but prefill uses only its last token (the router uses are
decode-only). Now sliced to the last token in prefill — a few hundred MB less
transient, no behavior change (verified: all four uses accounted for; tests pass).

## Why TPS is lower (7.7 vs 11.2) — and why it can't be "fixed" at 13.4 K

Decode KV read is **1.9 % of the step** at 13.4 K (weights dominate the
bandwidth). DKV adds per-token block routing + KV reconstruction from U/V,
run eager (CUDA graphs are correctly OFF for mutable routing). That overhead is
real and is not amortized by any KV saving until KV is a large fraction of the
step — i.e. long context (break-even ~485 K tokens by the roofline). At 13.4 K
this is simply the wrong regime to judge decode; the honest move is the
32K/64K/128K sweep, not decode-kernel micro-optimization here.

## Bottom line for the user's question

- **"Same RAM as dense"** → true only at the PEAK, and only because of the
  exact-prefill compression spike. At rest DKV is 2 GB lighter. Turn on
  `DKV_STREAMING_COMPRESS=1` to bound the peak (MLX's mechanism) and CUDA
  should win early too.
- **"Worse prefill"** → the block-store history re-attention per chunk; ~1.4×,
  architectural. Compress itself is already fixed (6.1→2.6 s).
- **"Worse tps"** → correct at 13.4 K and expected; sparse decode pays off only
  when KV dominates the step. Not a bug, a regime.

---

# FIFTH PASS — streaming verdict, TF32, contiguous prefill, tps roofline (2026-07-18)

## `DKV_STREAMING_COMPRESS=1` is a NET LOSS here — do not use it

The A/B (user, A100): prefill **11 s → 70–84 s** (7× worse), peak barely moved
(15.07 → 14.57 GB). Streaming bounds raw KV, but then every later chunk's forward
attends *compressed* history, and CUDA's compressed-history prefill attention is
~7× slower than attending raw KV. Wrong trade at this context. Reverted the
recommendation — keep exact prefill.

## TF32 enabled globally — a real decode win

The prior scoped-only TF32 left the log begging (`set_float32_matmul_precision`).
Enabled it globally in `_configure_cuda_allocator` (`DKV_TF32=0` to disable).
It does NOT touch the fp16 prefill attention or compress (cuSOLVER-bound), but it
DOES speed the fp32 decode reconstruction JIT kernels — the ones the warning fires
on. Small, free, and aligned with "even small gains pay off." Slight fp32-
accumulation perturbation (TF32 = 10-bit mantissa); the project has opted in.

## Contiguous dense prefill — built (MLX-parity), EXPERIMENTAL, `DKV_CONTIGUOUS_PREFILL=1`

MLX's `_sparse_prefill_attend` keeps a rotated K/V buffer of ALL tokens so far and
does ONE flash SDPA per chunk. The CUDA default re-assembles history blocks +
re-RoPEs + eager matmul + LSE-merge every chunk (the ~1.4× forward). New branch
(`dkv_attention.py`, fresh-prefill) replicates MLX: a per-(session,layer)
rotated buffer + one mem-efficient SDPA with an EXPLICIT bottom-right causal mask
(not `is_causal` — its non-square alignment is torch-version-dependent). KV is
still captured into blocks for boundary compression, so prefill holds the rotated
buffer AND unrotated blocks (~2× raw prefill KV); the buffer is freed at the
prefill→decode boundary and on clear_session.

Correctness of the core math is CPU-proven: the chunked contiguous attention with
the mask reproduces a single full-sequence causal SDPA to **0.00e+00** rel error.
But the block-capture integration and the pool/decode handoff are **UNVALIDATED on
GPU** — A/B `output_text` vs the default before trusting it. Expected: forward
~8 s → ~6 s (dense-like); peak similar to the current compression spike (the 2×
buffer ≈ the spike it replaces). If the 2× peak is a problem, the follow-up is to
drop block-capture and un-rotate the buffer at the boundary (RoPE is invertible)
for 1× memory — not done (numerical round-trip risk, GPU-only to validate).

## A fused decode kernel will NOT move tps at 13.4 K — roofline says so

```
per-token @13.4k:   weights(nf4) 7.35 GB=4.74 ms   DKV store 0.95 GB=0.61 ms
dense step 89.3 ms (11.2 tps)     DKV step 129.9 ms (7.7 tps)     gap +40.6 ms
```

The DKV KV-store read a fused kernel could optimize is **0.61 ms (~0.5 % of the
step)**. The +40.6 ms gap is **per-token eager launch / Python overhead** — 48
layers × (routing + reconstruction + sparse attend), with CUDA graphs OFF for
mutable routing. Both dense and DKV run ~19× off their bandwidth roofline
(launch-bound + nf4 dequant), so DKV's *extra* launches are the gap, not kernel
math. A better kernel optimizes the 0.5 %; it cannot close a launch-overhead gap.
The real tps levers, in order: (1) CUDA graphs for the static-routing steps
(route-cadence makes selection static between routes → capturable; hard, needs
fixed addresses/shapes), (2) TF32 on reconstruction (done), (3) fewer kernels per
layer per token. There is no kernel that makes 7.7→11 at this context; the honest
path is the long-context regime where the store read is a real fraction of the
step.

---

# SIXTH PASS — contiguous prefill CONFIRMED, 1× variant, CUDA-graph reality (2026-07-18)

## Contiguous dense prefill WORKS — forward now faster than dense

`DKV_CONTIGUOUS_PREFILL=1` on the A100:

```
preset   fwd before   fwd now   dense fwd
low        8.6 s       4.85 s      5.65 s
mid        8.5 s       4.85 s      5.65 s
high       7.3 s       4.01 s      5.65 s
```

The ~1.4× forward overhead is gone — DKV's forward is now *below* dense. The
single-SDPA-over-a-rotated-buffer replaced the per-chunk block re-assembly +
re-RoPE + eager matmul. Cost: peak rose (15.07 → 16.74 GB low) from the buffer
coexisting with the captured blocks (2× raw prefill KV).

## 1× variant built — `DKV_CONTIG_UNROTATE=1` (with CONTIGUOUS_PREFILL=1)

"Delete the old blocks": keep ONLY the rotated buffer during prefill; at the
compression boundary, recover the unrotated K the pool needs by **inverse RoPE**
(rotation is orthogonal), then replay `capture_prefill_kv` per recorded chunk so
the block layout is byte-identical. Drops prefill KV from 2× back to 1× (buffer
only ≈ dense KV), so peak should return to ~dense while keeping the fast forward.

CPU-verified: inverse RoPE round-trip fp16 = 1.4e-4 (≪ the 9.2e-3 int8-U floor);
the finalize's un-rotate+chunk-slice reproduces the exact per-chunk unrotated K
(fp32 6e-8). So the blocks it builds are identical to the 2× path's. **The
block-capture/pool integration is UNVALIDATED on GPU** — A/B `output_text` vs the
2× variant (should be bit-identical up to the fp16 round-trip). Also enabled TF32
globally last pass (`set_float32_matmul_precision('high')`).

Diags: removed the verbose per-block state dump from `run_nat_eval.py` (the
kv_logical investigation it served is closed); kept the rank-boost rate and the
prefill-next first-token line (recall proxy). `DKV_DIAG=1` still gives the
runtime's own compress trace.

## CUDA graphs: honest assessment — a real refactor, NOT shippable blind

`CUDAGraphDecodeRunner` (`native_core/graph_runtime/static_decode_graph.py`)
exists with static input/output buffers and capture/replay, but capture is gated
behind `model._dkv_cuda_graph_safe`, which is never set — correctly. CUDA-graph
replay records **kernels only**, not host Python. The DKV decode forward mutates
Python/session state every token (appends KV, updates routing slots, dense-window
membership), so a captured graph would (a) replay stale routing and (b) never
actually append the token. That is the whole reason it is off.

Readiness audit:
- ✓ Pool is pre-allocated → fixed addresses.
- ✓ Dense window uses a **ring buffer** (Phase 29, "zero torch.cat per token") →
  fixed address, in-place writes.
- ✗ Routing builds a fresh `block_indices` tensor per token (dynamic shape/values).
- ✗ Per-token host Python: dict lookups, `get_streaming_blocks`, workspace mgmt,
  `.item()`/`.tolist()` syncs — none capturable.

To make it graph-safe (the vLLM pattern the user described — "insert new chunks/
tokens into the graph as they come"):
1. A **static block-table tensor** of fixed capacity K_max; routing writes indices
   in-place (masked padding), the graph gathers from it. Route eagerly every N
   tokens (route-cadence), replay the graph for the N−1 static steps.
2. The KV append must be an **in-place ring-buffer write at a fixed address**,
   inside the captured region (the ring buffer exists; the append must be
   graph-captured, not Python-orchestrated).
3. A **static seq-len tensor** (or fixed-max + mask) so no shape changes per token.
4. Strip host Python from the captured region — all decisions become tensor ops.

This is a multi-stage refactor of the decode hot path, and **CUDA-graph replay
cannot be validated anywhere but the A100** — a stale-address bug is silent
garbage, not a crash. Shipping capture blind would corrupt runs. Recommend doing
it as staged, GPU-tested increments (start with the static block-table + cadenced
routing), each A/B'd for output parity, rather than a single blind change. The
runner scaffolding + ring buffer mean stages 1–2 are the real work, not stage 0.

## MLX: it already has both

MLX wins early precisely because it *already* does the contiguous rotated buffer
(`_sparse_prefill_attend`, `all_k` = all tokens) AND compiles its decode with
`@mx.compile` over **pure array functions with state passed explicitly** — the
graph-safe ABI CUDA lacks. So "do it in MLX too" is already true; the CUDA work
is about reaching MLX's design, not adding to MLX.

---

# SEVENTH PASS — 1× prefill CONFIRMED, factual-store sync fix, CUDA-graph plan (2026-07-18)

## `DKV_CONTIG_UNROTATE=1` (1× memory) CONFIRMED on A100

The un-rotate variant worked exactly as designed:

```
low_preset      after_fwd   peak_prefill   fwd     kv_phys
2× contiguous     15.23 GB     16.74 GB    4.85s   2.99×
1× un-rotate      12.58 GB     14.09 GB    5.03s   2.99×   ← ≈ dense peak (13.27)
```

`after_fwd` dropped 15.23 → 12.58 GB (the duplicate buffer is gone); peak is now
basically dense while the forward stays faster than dense. Compress rose (2.7 →
3.8s) because block-building moved into the compress phase (un-rotate + replay
capture). Net: best of both — dense-speed forward AND dense-level peak, with a 3×
KV store. (The "tps dropped 7.6→6.7" in that run was the whole machine being
slower — dense also fell 11.1→9.1; the ratio held.)

## Factual store: the 1.9-tps cause fixed (the "CPU micro-spikes in a pattern")

`FactualExactStore.query()` runs PER DECODE TOKEN and did
`torch.dot(q_desc, entry.descriptor).item()` per entry in three loops (prime
seeds, merged candidates, fallback) — thousands of host syncs per generation, the
periodic CPU spikes. Replaced with one cached `[E, DESC]·[DESC]` matmul + a single
`.tolist()`; bit-identical (2.4e-7). `build()` (prefill) had eight
`token_ids[i].item()` loops → one `.tolist()`. CPU-verified equal; no new test
failures. This targets `factual_store`/`combined` (1.9 tps), not the base presets.

## CUDA graphs — researched (vLLM pattern) and scoped; the concrete blockers

vLLM/SGLang pattern (deepwiki, vLLM docs): pre-allocated **static input buffers**
(copy_ actual data in before replay), capture per fixed batch size (pad smaller),
**block-table indirection** via a static tensor, replay = one driver call, read
from static output. KV writes use a static **slot_mapping** updated before replay.

Why DKV can't capture today (verified, not assumed): `CUDAGraphDecodeRunner`
captures the model forward *inside* `torch.cuda.graph()`, and graph capture
**forbids host syncs**. The decode forward has them:
- routing: entropy `.item()`, centroid `.tolist()`, score `.cpu()` — but routing
  is cadenced and can run OUTSIDE the graph (eager, every N tokens).
- per-layer compute: `k_avg = curr_k...cpu()`, `active_slots = set(block_indices
  .tolist())`, an `.item()` — these are INSIDE what would be captured.
- the ring-buffer write position changes per token (needs a static slot tensor).
- the eval also `.item()`s the sampled id per token for the stop check.

Staged plan (each stage GPU-tested for output parity — capture bugs are silent
garbage, so no stage ships without the A100):
1. **Make the per-layer decode compute sync-free** — move routing fully outside
   the captured region (cadenced, writes a STATIC `block_indices` tensor in place),
   drop the per-layer `.cpu()`/`.tolist()`/`.item()`. This is the prerequisite;
   until it's done, capture throws.
2. **Static slot_mapping / seq-len** — the ring-buffer append becomes an in-place
   write at a device-resident position the kernels read.
3. **Static block-table of fixed capacity K_max** — routing writes indices in
   place, padding masked; graph gathers K_max blocks.
4. **Capture + self-check** — capture the sync-free forward; on replay,
   periodically compare against one eager step and auto-disable on mismatch (a
   safety net so testing can never corrupt output).

Stage 1 is the real first increment and is independently testable (each removed
sync is verifiable). Recommend doing it next as a focused change, GPU-validated,
before touching capture. Not shipping capture blind — it would silently corrupt.

---

# EIGHTH PASS — CUDA-graph stage 1 done (the finding: decode is already sync-free) (2026-07-18)

## Stage-1 outcome: the base decode path is ALREADY sync-free

Audited every host sync in the decode forward. Result — the base (low/mid/high)
presets have **none** on the hot path:
- `k_avg.cpu()` (SRL recent-key trail): behind `if srl_state is not None` — SRL is
  off in base presets. Now ALSO gated by `DKV_GRAPH_SAFE_DECODE=1` (skips it so
  even SRL sessions are sync-free; heuristic loses its key trail, routing still
  works). This is the one code change this pass.
- `active_slots = set(block_indices.tolist())`: behind the factual-store gate — off
  in base presets.
- the `[SRL Validate] ....item()`: behind a debug validate flag — off.
- `get_cached_decode_blocks`: metadata is CPU-resident (Phase 29), and it returns a
  **version-cached, stable-address** `block_indices` GPU tensor — no GPU→CPU sync,
  and the address is stable between block-creations (the cache only rebuilds when
  `metadata_version` changes, i.e. a new block).

So "make decode sync-free" (stage 1) is effectively already satisfied for the
eval. Capture will NOT fail on host syncs.

## The actual blocker (stage 2): the ring-buffer append changes shape per token

`streaming_sparse_ingest.py:990` — the decode KV append is
`buf_k[0, :, fill, :] = k[...]` with `fill` a **Python int** advancing each token,
and attention reads `current_block.active_k = buf_k[:, :, :fill, :]` — a
**different-length slice every token**. CUDA graphs require static shapes and
static write positions, so this is what breaks capture (not syncs). It is exactly
the case vLLM solves with `slot_mapping` (a device write-position tensor) + a
static `seq_len` (attend a FIXED max length, mask the tail).

Stage-2 spec (the real work, GPU-only to validate):
1. Attend the dense window at a FIXED length (`micro_block_size`) always; pass the
   valid count as a **device tensor** the combined kernel masks against (the kernel
   already takes a `dense_len` arg — it must become a static tensor, and the tail
   masked). Behaviour-preserving vs the current sliced read (CPU-verifiable: masked
   fixed-length == sliced).
2. The append writes into `buf_k` at a **device-resident `fill`** via scatter
   (in-place, fixed address) so it is graph-captured or done eagerly before replay.
3. `block_indices` already stable between block-creations (cache) — re-capture the
   graph on block-creation (~every 32 tokens; a cheap eager step at the boundary).
4. Then capture the 48-layer forward over each stable window and replay.

## Why I am NOT shipping capture this pass

The prefill wins (Gram, contiguous, un-rotate) were shippable-untested because the
MATH was CPU-provable to 0.00e+00. **CUDA-graph replay correctness has no CPU
equivalent** — a stale-address replay is silent wrong tokens, not a crash, and the
decode is stateful (you cannot run eager alongside to self-check without
double-appending the token). So each stage-2 piece must be GPU-validated directly
(output A/B) as it lands. Stage 1's deliverable is the audit result above + the
`DKV_GRAPH_SAFE_DECODE` guard; stage 2 (the fixed-length masked ring read + the
device `fill`) is the next focused, GPU-tested change — the point where the tps
win actually starts.

---

# NINTH PASS — the "arg12" cudagraph skip is a RED HERRING; the tps blocker is the eager nf4 model (2026-07-18)

Chased the log line `skipping cudagraphs due to cpu device (arg12_1)`. Traced it
precisely, then reverted the fix, because it does not help the eval — and the
tracing revealed where the tps actually goes.

## What `arg12` is, and why fixing it doesn't matter here

`arg12 == D` (head_dim), an **int argument** to `_attend_and_reconstruct_v`. Under
`torch.compile(dynamic=True)` a scalar int is lifted to a 0-d CPU tensor input,
and Inductor won't cudagraph a function with a CPU input → the skip. Deriving the
ints from tensor shapes removes the CPU input and would let the cudagraph engage.
**But**: that compiled function is on the **SEPARATE** decode path (`bias>0`). The
eval runs the **COMBINED** path — a raw Triton kernel (`_fused_decode_combined_
kernel[grid]`) that never calls the compiled function. So fixing the skip changes
nothing for the eval, and the shape-derivation broke the N=0 dense-only test
(empty `anchors_V` ⇒ derived D=0). Reverted.

## Where the eval's decode time actually goes

- Both dense AND DKV pay the **eager nf4 model forward** (`torch.compile
  disabled` / "skipping torch.compile to avoid graph-break errors" for bnb-quant).
  That is the ~89 ms of the dense step — 48 layers of eager QKV/MLP/norm kernels.
  DKV cannot touch this; it is the base model, and torch.compile breaks on
  bnb-nf4. It is also why BOTH runtimes sit ~19× off their bandwidth roofline.
- DKV's **extra** ~41 ms/token (the 11.2→7.7 tps gap) is its own per-layer work:
  routing (layer 0) + **reconstructing all 49 blocks** from U/V + the combined
  Triton kernel, ×48 layers, eager. `N_sparse=49` means routing is NOT pruning
  (the residual router needs `pool.W_proj`; unset here) — every compressed block
  is reconstructed every token.

## The honest tps levers at 13.4K (all hard or tradeoff-laden)

1. **Prune blocks via the routing** (49 → topk 16): ~3× less reconstruction, so
   the ~41 ms DKV overhead could drop toward ~14 ms → real tps gain. Needs
   `W_proj` wired + `DKV_TOPK_BLOCKS=16`; accuracy-sensitive (MLX uses exactly
   this successfully, but the CUDA SRL router was measured net-negative before —
   the `residual` router is the one to try). GPU-only to validate; the user's new
   output printing is exactly the A/B tool for it.
2. **CUDA-graph the DKV hook** — the stage-2/3 ABI work (still the biggest, and
   still un-CPU-validatable).
3. **Get the nf4 model itself graphable** — a bnb/torch.compile-compatibility
   problem, separate from DKV, and it would help dense too.

There is no safe one-line tps win at this context. The most tractable *testable*
one is #1 (routing prune), which the user can A/B on outputs now that the eval
prints them. `DKV_EVAL_OUTPUT_CHARS` caps the printed length; default full.

---

# TENTH PASS — MLX-parity decode pruning ported to CUDA + eval reworked (2026-07-18)

## How MLX prunes (studied) and the CUDA port

MLX (`mlx_dkv_wrapper`): `use_topk = (topk_blocks>0 and nb>k_eff)`, then
`sel = argsort(relevance)[-k_eff:]` where relevance = `_block_relevance_residual`
(max over query heads of max(q·anchor, max q·residual-key)). Default K=16. NO
W_proj / SRL state — a pure per-block q·k top-K. It gathers only the K selected
blocks and attends those; that is why MLX reconstructs ~16 not ~50 blocks/token.

CUDA already had the identical router (`route_blocks_relevance`, a direct port of
`_block_relevance_residual`) but only ran it inside the SRL gate, so decode kept
`N_sparse=49` (every block reconstructed). Added `DKV_DECODE_PRUNE_K` (default
0=off; set 16 to match MLX) in the decode dispatch: when `nb > K` and SRL didn't
already reroute, run the residual router, keep the top-K, prune both
`block_indices` and `anchor_indices`. Routes once at layer 0 (caches the selected
ANCHORS, which are layer-invariant) and maps them to each layer's slots — verified
on CPU (anchor→slot mapping keeps exactly K blocks across layers). Fewer blocks
reconstructed ⇒ the ~41 ms/token DKV overhead should drop toward ~14 ms at K=16.
**Accuracy-sensitive — A/B the printed output; MLX does this successfully, but the
CUDA SRL router was net-negative before (this is the `residual` router, not SRL).**

## Eval reworked (`colab/run_nat_eval.py`)

- **Single prompt** (the researcher-claim evaluation) instead of two — shorter runs.
- **Prints model outputs** already (`↳ output:` per prompt) — judge on OUTPUT.
- **Removed debug noise**: the `[DIAG] prefill-next ... top5` lines (both branches)
  and the repeated `[NAT eval] Aligned CUDA chunk size`. Kept the rank-boost rate.
- **TIME / STORAGE BREAKDOWN table** at the end: per config, `fwd_s | comp_s |
  dec_s | tps | peak_pf | peak_dec | pool_MB | kv_phys | vs_dense` — so where the
  time and VRAM go across dense vs presets is one glance.

Run to test the prune (K=16) with outputs:
`DKV_COMPRESS_GRAM_SVD=1 DKV_RANK_BOOST=off DKV_RSVD_MAX_RPROJ=32
DKV_CONTIGUOUS_PREFILL=1 DKV_CONTIG_UNROTATE=1 DKV_DECODE_PRUNE_K=16 ...`
Watch tps rise, `N_sparse` drop to ~16-17, and confirm the output still answers.

## RESULT (A100): prune is a DEAD END — keep `DKV_DECODE_PRUNE_K=0`

Ran K=16. `N_sparse` dropped 49→16 as intended. Two decisive findings:

1. **tps did NOT change**: low 7.0, mid 6.9, high 6.8 — identical to the un-pruned
   runs. Reconstructing 16 blocks instead of 49 changed decode speed by **zero**.
   This is the definitive proof that decode is **100% bound by the eager nf4 model
   forward**, not by block reconstruction. **There is no DKV-side tps lever at
   this context** — the block count is irrelevant to speed. (The only tps levers
   left touch the base model: compile/graph the nf4 forward, out of DKV scope.)
2. **Output collapsed to garbage** on every pruned preset (repetition loops, mixed
   scripts). At K=16 the CUDA residual router drops blocks the answer needs — MLX
   tolerates K=16, CUDA does not here. So prune both fails to help AND hurts.

Verdict: **prune stays OFF by default (0).** Kept the flag for the record, but it
is a confirmed dead end for tps. The un-pruned presets are the ones to use.

## Output-quality ranking (this prompt, A100)

- **dense, factual_store, combined**: coherent, correct — all five required
  distinctions covered. factual_store/combined match dense quality.
- **low/mid/high/early_boost UN-pruned** (prior runs): coherent, correct.
- **low/mid/high/early_boost PRUNED (K=16)**: garbage.

So: the presets already answer correctly without pruning at ~2–3× KV savings and
~7 tps; factual_store adds nothing to quality here but costs a slow O(T²) build
(compress 14.7 s) — its query-path fix did help decode (1.9→2.5 tps). The tps
ceiling (~7 vs dense 11) is the eager nf4 model, shared by dense, and not movable
from the DKV side at 13.4K.

## Factual-store build made lighter (exact, this pass)

Two more costs in `FactualExactStore.build()` (the 14.7 s factual/combined compress):
- **Eagle scores** built the full `[T, T]` form — `sim`, a `triu` mask, the
  `masked_fill` result and the `softmax` all resident (~4× T·T fp32 ≈ 3 GB at 13K)
  plus a T·T `nan_to_num`. Rewrote it as a **row-chunked** running column-sum:
  peak drops to O(chunk·T) (~55 MB/chunk) and the giant mask/nan_to_num are gone.
  EXACT same R (CPU-verified 9.5e-7 vs the full form; strictly-`j<i` causal
  preserved).
- Two remaining per-span `token_ids[idx].item()` / `eagle_scores[idx].item()` loops
  (device syncs over most content tokens) → index the pre-materialised
  `token_ids_list` and a cached eagle-score list.
Both are exact/behaviour-preserving; verified `build()`+`query()` run end-to-end.
This trims the factual memory spike and some sync time; the residual O(T²) *compute*
(the K·Kᵀ softmax) is unchanged, so profile on the A100 to see how much of the
14.7 s was the spike vs the compute.

## Validated defaults now applied in the eval (not global runtime)

`run_nat_eval.run_worker` now `setdefault`s the A/B'd-good combo so a no-flag run
gets it: `DKV_COMPRESS_GRAM_SVD=1`, `DKV_RANK_BOOST=off`,
`DKV_RSVD_MAX_RPROJ=32`, `DKV_CONTIGUOUS_PREFILL=1`, `DKV_CONTIG_UNROTATE=1`.
`DKV_DECODE_PRUNE_K` is deliberately NOT set (confirmed dead end). Scoped to the
eval (setdefault, overridable) rather than changed in the runtime, because the rank
cap is fidelity-affecting and needle-sensitive workloads should validate it first —
here the synthesis output held. GRAM/contiguous/un-rotate are recon-equivalent and
safe to promote globally later; the two rank knobs want a needle sweep first.

## Where does decode time go? — `colab/profile_decode_step.py` (rewritten)

Runs the eval setup for a chosen `--preset` (or `dense`), prefills, decodes N tokens
under `torch.profiler`, and reports CUDA self-time bucketed into **model** (nf4
GEMM/dequant, shared with dense), **dkv** (the fused sparse-attention + recon
kernels), and **other** (generic ops left unattributed to keep the split honest),
plus the top-K ops. This empirically shows the tps story: run it per preset; if the
**model** bucket dominates, tps is bound by the eager nf4 forward and no DKV
change moves it (the 2026-07-18 finding) — the long-context sweep is where the
dkv bucket, and thus DKV's advantage, becomes the real cost.
`python colab/profile_decode_step.py --preset low` (and `--preset dense` for the
baseline).

## `DKV_FAST=1` — one toggle for the combo, and what to validate

Added `DKV_FAST=1` (wrapper `_apply_fast_mode`, runs at init before the
manager): setdefaults GRAM_SVD + CONTIGUOUS_PREFILL + CONTIG_UNROTATE (all
recon-equivalent, safe) **and** RANK_BOOST=off + RSVD_MAX_RPROJ=32 (fidelity-
affecting). DECODE_PRUNE is NOT bundled. An explicit individual flag still wins
(`DKV_FAST=1 DKV_RANK_BOOST=auto` keeps the boost). The eval now defaults to
`DKV_FAST=1`.

**The test that gates the two rank knobs: `tests/test_niah.py`.** Its needle is a
6-DIGIT NUMBER ("847291") — and the content-aware rank boost existed precisely to
give DIGIT blocks extra rank. So capping rank at 32 with boost off is exactly the
change that could lose a number. Validate before trusting FAST on number-heavy
retrieval:
```
DKV_MODEL=Qwen/Qwen2.5-14B-Instruct python -m pytest ACTIVE_RUNTIME/tests/test_niah.py -v
DKV_MODEL=Qwen/Qwen2.5-14B-Instruct DKV_RANK_BOOST=off DKV_RSVD_MAX_RPROJ=32 \
  DKV_COMPRESS_GRAM_SVD=1 python -m pytest ACTIVE_RUNTIME/tests/test_niah.py -v
```
(depths 0.1/0.5/0.9 × ctx 4k/8k). Both green ⇒ safe to promote FAST's rank knobs to
global runtime defaults; a regression ⇒ keep the boost (or a higher cap) for
number/formula content. `benchmarks/niah_recall.py` is the deeper sweep.

## Two findings from the A100 profiler run (2026-07-18)

### BUG fixed: `wrapper.generate()` `NameError: _PREFILL_CHUNK`

`hf_dkv_wrapper.generate()` line 904 referenced `_PREFILL_CHUNK` (undefined)
instead of `PREFILL_CHUNK` — a hard crash on the CUDA `generate()` path. The eval
never hit it (it drives `model()` directly), but `test_niah` uses
`wrapper.generate()`, so it crashed before generating AND before `wrapper.stop()`,
leaking the 14B model so every subsequent parametrized test OOM'd (two 14B models
on 40 GB). So the niah run did NOT actually validate the rank knobs — it died on
this typo. Fixed; re-run the niah A/B.

### CORRECTION: high preset IS DKV-kernel-bound (earlier "eager-model-bound" was low/mid only)

`profile_decode_step.py` per preset:
- **dense**: `model 48%`, `dkv 0%` (flash attention) — model-bound, as expected.
- **high**: `_fused_decode_combined_kernel = 55.5 ms/token = 49%` of the step —
  the DKV combined kernel is the SINGLE LARGEST cost.

So the blanket "decode is eager-model-bound, no DKV lever" holds for **low/mid**
(cheap kernel, 40/64 residuals) but is **WRONG for high**: `max_residual=128` makes
the combined kernel dominate. The combined-kernel cost scales with
`N_blocks × max_residual`, so high's residual attention is the bottleneck. Real
DKV tps levers FOR HIGH (not low/mid): (a) lower `max_residual` (128→64,
speed/recall trade), or (b) optimize the combined kernel's residual path (GPU
work). This is the profiler doing its job — it found a real DKV cost the
pruning experiment (run on low/mid) had masked.

---

# SEVENTH PASS — 1× confirmed, CUDA-graph reality, factual-store cause (2026-07-18)

## 1× contiguous prefill (`DKV_CONTIG_UNROTATE=1`) — CONFIRMED on A100

```
                fwd     after_fwd   peak_prefill
2× contiguous   4.85s   15.23 GB    16.74 GB
1× un-rotate    5.03s   12.58 GB    14.09 GB   ← near dense (13.27), fast fwd kept
dense           6.07s   —           13.27 GB
```

after_fwd dropped 15.23→12.58 (the 2.6 GB buffer duplicate is gone); peak 16.74→
14.09 ≈ dense. Forward stays faster than dense. Compress rose (2.7→3.8 s) because
block-building moved into the boundary finalize — expected. The apparent tps drop
this run was **machine-wide** (dense also fell 11.1→9.1 tps; slower instance), not
a regression; the DKV/dense ratio held. Recommended prefill config now:
`DKV_COMPRESS_GRAM_SVD=1 DKV_RANK_BOOST=off DKV_RSVD_MAX_RPROJ=32
DKV_CONTIGUOUS_PREFILL=1 DKV_CONTIG_UNROTATE=1`.

## CUDA graphs: researched, scoped — NOT shippable blind (would be silent garbage)

Studied vLLM/SGLang (sources in session): the pattern is a **two-stage** step — a
PREP stage (CPU+GPU: routing, slot_mapping → written into STATIC tensors) then a
REPLAY stage (only captured kernels). Requirements: static input/position/
block-table/seq-len buffers (`copy_` in place before replay), block-table
indirection into a fixed-address paged KV pool, and the KV write
(`reshape_and_cache`) as an in-place op at a static slot **inside** the graph.

DKV readiness (confirmed by reading the code):
- ✓ pool fixed-address; ✓ dense-window ring buffer (Phase 29).
- ✗ **decode ingest is deeply stateful**: `ingest_chunk` (streaming_sparse_ingest
  ~935) starts a NEW block every 32 decode tokens, manages ring fill in Python,
  and triggers compression mid-decode — none capturable.
- ✗ routing builds a fresh `block_indices` tensor per token.
- ✗ the hot decode path is the raw **Triton COMBINED kernel** launched eagerly
  ×48 layers/token (not torch.compile), so Inductor's own cudagraphs don't cover
  it. (The `arg12` CPU-scalar that skips Inductor cudagraphs is on a NON-hot
  fallback path — fixing it would not touch the real decode.)

**Verdict:** a real decode-path rewrite (static ingest, static block table,
graph-captured ring append) — the vLLM/SGLang-scale effort, and CUDA-graph replay
correctness **cannot be validated anywhere but the A100** (a stale-address/stale-
state bug is silent garbage). Capturing the current stateful forward would replay
stale routing AND never append the new token. Path forward = staged, GPU-tested
increments (stage 1: fixed-capacity static block table + graph-safe ring append,
cadenced routing, eager fallback at 32-token block boundaries), each A/B'd. Not a
one-shot flag.

## Factual store 1.9 tps — cause found: per-token `.item()` host-sync loops

`native_core/srl/factual_store.py` runs many `for i in range(total_seq_len):
tid = int(token_ids[i].item())` loops (lines 216, 221, 243, 259, 279, 339, 348,
387, 493, 612…). Each `.item()` is a GPU→CPU sync; at 13.4k tokens that is tens of
thousands of host stalls per call — the "CPU micro-spikes in a pattern" the user
observed, and why factual_store adds ~4 s to compress and runs at 1.9 tps. Fix:
move token_ids to CPU ONCE (`token_ids[:T].tolist()`) and index the Python list in
the loops instead of `.item()`-ing per element; vectorize the salience/threshold
reductions. Tractable and testable — separate from the CUDA-graph work.
