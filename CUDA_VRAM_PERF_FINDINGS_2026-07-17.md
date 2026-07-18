# CUDA active-runtime: why VRAM is high and throughput is low (2026-07-17)

Investigation of the A100 NAT-eval run (Qwen2.5-14B-Instruct nf4, 13.4K prompt,
48 layers, 8 KV heads, head_dim 128). Every claim below is tagged **PROVEN**
(derived from the code plus arithmetic, reproducible on any machine) or
**HYPOTHESIS** (consistent with the numbers, needs the A100 to confirm).

---

## 1. The headline: the "26x KV reduction" is a broken counter — **PROVEN**

`kv_logical` reads **0.101 GB** in every DiffKV config, against dense's 2.643 GB.
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
DiffKV's pool is only 1.37x smaller than the dense KV it replaces — while
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
  `mlx_diffkv_wrapper.py` is the model name "Qwen-1.5B". This is a CUDA-only
  divergence, and it is part of why MLX looks better earlier.

The log line `rank=32 (pool_rank=48)` reads like headroom. It is not headroom;
it is the operating point.

Added `DIFFKV_RANK_BOOST=off` (MLX parity) and a per-session boost-rate counter
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
(`DIFFKV_TF32=0` to opt out). Honest expectation: this only reclaims ~0.035 s of
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
  measured DiffKV step   : 149.3 ms/token (6.7 tps)   => +61.5 ms/token
```

**KV read is 1.9% of the dense decode step.** DiffKV attends ~5.5K of 13.4K
tokens (`N_sparse=16, L_dense=1419`), so the absolute most it can save is
**~1.0 ms/token** — while it adds **61.5 ms/token**.

For the saved KV bandwidth to merely pay for that overhead you would need
**~95 GB of KV, i.e. ~485,000 tokens of context**. KV only equals the *weight*
read at ~37,000 tokens.

At 87.7 ms/token, dense is ~20x off its own bandwidth roofline (4.7 ms) — decode
is bound by bnb-nf4 dequant and eager launch overhead, not by KV. DiffKV then
adds 48 layers of per-token routing/reconstruction on top, with **CUDA graphs
correctly disabled** (mutable routing state), so it pays full eager launch cost.

**Implication:** at 13.4K the eval is measuring DiffKV's overhead against a
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
3. Stop quoting 13.4K decode tps as a DiffKV result. It is a measurement of
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
- **DiffKV forward is 1.47x slower than dense** (8.62 s vs 5.88 s at matched
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
| `native_core/compression/lowrank.py` | `DIFFKV_RANK_BOOST=off`; boost-rate telemetry; documented the 100% fire rate |
| `serving/hf_diffkv_wrapper.py` | TF32 on (`DIFFKV_TF32=0` opts out); allocator log prints what is actually in effect |
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
3. A/B `DIFFKV_RANK_BOOST=off` on VRAM, compress time, **and recall**.
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

`[DIAG] rank boost fired on 49/49 blocks = 100.0%` in every DiffKV config.

## TF32 confirmed useless for compress → refactored to a scoped context manager

TF32 on: compress 4.453 s vs 4.635 s off — a **4%** difference, because compress
is cuSOLVER-bound. The first-pass change enabled TF32 **process-globally** in
`_configure_cuda_allocator`, which also alters the fp32 math in decode
reconstruction and the block router (perturbing generated output across presets
for a 4% compress win). Replaced with `lowrank._tf32_matmul()`, a context manager
scoped to the compression call only. `DIFFKV_TF32=0` disables it.

## Gram-eigh swap: 1.9x on the SVD, proven equivalent, now implemented

Profiler on the same A100: `eigh(B Bᵀ)` = 2.106 s vs `svd` 3.917 s → **1.9x**,
projected compress 4.635 s → **2.824 s**. Reconstruction 8.2e-6 vs the SVD's
8.5e-6 (equal). Implemented behind `DIFFKV_COMPRESS_GRAM_SVD=1` in
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
`DIFFKV_RANK_BOOST=off` (rank 32) gives 32+5 = **37, still over the cliff**. Two
knobs now make r_proj ≤ 32 reachable (both in `compress_layer_blocks_gpu`):

- `DIFFKV_RSVD_OVERSAMPLES` (default 5) — randomized-SVD slack; the 2 power
  iterations already there cover most of what oversampling buys.
- `DIFFKV_RSVD_MAX_RPROJ` (default 0=off) — hard cap on r_proj. Blocks that
  wanted a higher rank are capped to it. The per-block dynamic-rank clamp was
  updated so `dynamic_rank ≤ r_proj` (CPU-verified: forcing block_rank 48 with a
  cap of 32 yields dynamic_rank 32 and consistent U/V, no crash).

**Recommended batched recipe to A/B on recall:**
```
DIFFKV_COMPRESS_GRAM_SVD=1  DIFFKV_RANK_BOOST=off  DIFFKV_RSVD_MAX_RPROJ=32
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
  `test_decode_cache_fused_parity` and `test_diffkv_kernel_parity` do a bare
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
   `DIFFKV_COMPRESS_GRAM_SVD=1 DIFFKV_RANK_BOOST=off DIFFKV_RSVD_MAX_RPROJ=32`.
   Expect compress ~6.1 s → ~1 s. If needle + synthesis hold, this is the win;
   make it the CUDA default. (The cliff is confirmed; this is now an accuracy
   check, not a perf question.)
2. If recall dips, back off toward the cliff instead of over it:
   `DIFFKV_RSVD_OVERSAMPLES=0` with rank 32 (r_proj=32), or a base rank of ~26
   with the default oversamples — both land r_proj at 32 with some slack.
3. `DIFFKV_RANK_BOOST=off` also cuts pool VRAM (pool_rank 48→32) — fold its
   recall check into step 1 since the recipe already sets it.
4. Sweep 32K/64K/128K — still the only regime where sparse *decode* can win here
   (decode KV is 1.9% of the step at 13.4K; break-even ~485K tokens).

---

# THIRD PASS — batched recipe CONFIRMED, pool_rank waste fixed (2026-07-17)

## The recipe works: compress 6.1 s → 2.6 s, prefill 14.9 s → ~10.5 s

`DIFFKV_COMPRESS_GRAM_SVD=1 DIFFKV_RANK_BOOST=off DIFFKV_RSVD_MAX_RPROJ=32` on the
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
rank-boost*. With `DIFFKV_RANK_BOOST=off` the boost never fires, so no block's
stored rank exceeds `max_rank` — the 1.5× over-allocates every slot by 50% for
capacity that cannot fill.

Fixed in `KVRuntimeManager`: the multiplier is now `1.5` only when the content
boost is active, `1.0` when `DIFFKV_RANK_BOOST=off`, and `pool_rank` is
additionally capped by `DIFFKV_RSVD_MAX_RPROJ`. Provably safe — compress receives
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
Note: with `DIFFKV_RSVD_MAX_RPROJ=32`, `early_boost` collapses to mid (pool_rank
capped 96→32, ratio 2.32×) — the cap correctly dominates the early-layer boost,
so that config is redundant under this recipe.

## Where this leaves it

- **Prefill**: compress is no longer the story (2.6 s of a ~10.5 s prefill; the
  ~8 s forward is now the bulk — that is the model, not DiffKV overhead).
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

## Peak VRAM: DiffKV is lighter at REST but heavier at the PEAK

```
low_preset prompt1        VRAM
  dense peak              13.27 GB
  DiffKV after-compress   11.25 GB   <- 2.0 GB LIGHTER than dense (the pool wins)
  DiffKV PEAK             15.07 GB   <- 1.8 GB HEAVIER than dense (the spike)
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

## The fix already exists but is off by default: `DIFFKV_STREAMING_COMPRESS=1`

The eval's CUDA loop calls `compress_deferred_prefill_blocks` + `empty_cache`
after each chunk when `DIFFKV_STREAMING_COMPRESS=1`. `compress_deferred_blocks`
only compresses blocks past the recency window (`window_ok`), and the fresh-
prefill attention path already handles compressed history blocks (it splits
`comp_blocks` vs dense in the history cross-attention). So streaming should bound
peak raw KV to ~(512 + chunk) ≈ 1.5 K tokens instead of 13.4 K — dropping peak
from 15 GB toward ≤ dense. Cost: far-back history is attended in lossy compressed
form during prefill (exactly MLX's tradeoff), plus the per-chunk `empty_cache`
slows allocation slightly. **This is the lever for "CUDA peak ≥ dense" — A/B it.**

## Why prefill FORWARD is 1.4× dense (8.2 s vs 5.9 s)

The fresh-prefill path (`diffkv_attention.py` ~2242) does, per chunk: local causal
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
bandwidth). DiffKV adds per-token block routing + KV reconstruction from U/V,
run eager (CUDA graphs are correctly OFF for mutable routing). That overhead is
real and is not amortized by any KV saving until KV is a large fraction of the
step — i.e. long context (break-even ~485 K tokens by the roofline). At 13.4 K
this is simply the wrong regime to judge decode; the honest move is the
32K/64K/128K sweep, not decode-kernel micro-optimization here.

## Bottom line for the user's question

- **"Same RAM as dense"** → true only at the PEAK, and only because of the
  exact-prefill compression spike. At rest DiffKV is 2 GB lighter. Turn on
  `DIFFKV_STREAMING_COMPRESS=1` to bound the peak (MLX's mechanism) and CUDA
  should win early too.
- **"Worse prefill"** → the block-store history re-attention per chunk; ~1.4×,
  architectural. Compress itself is already fixed (6.1→2.6 s).
- **"Worse tps"** → correct at 13.4 K and expected; sparse decode pays off only
  when KV dominates the step. Not a bug, a regime.

---

# FIFTH PASS — streaming verdict, TF32, contiguous prefill, tps roofline (2026-07-18)

## `DIFFKV_STREAMING_COMPRESS=1` is a NET LOSS here — do not use it

The A/B (user, A100): prefill **11 s → 70–84 s** (7× worse), peak barely moved
(15.07 → 14.57 GB). Streaming bounds raw KV, but then every later chunk's forward
attends *compressed* history, and CUDA's compressed-history prefill attention is
~7× slower than attending raw KV. Wrong trade at this context. Reverted the
recommendation — keep exact prefill.

## TF32 enabled globally — a real decode win

The prior scoped-only TF32 left the log begging (`set_float32_matmul_precision`).
Enabled it globally in `_configure_cuda_allocator` (`DIFFKV_TF32=0` to disable).
It does NOT touch the fp16 prefill attention or compress (cuSOLVER-bound), but it
DOES speed the fp32 decode reconstruction JIT kernels — the ones the warning fires
on. Small, free, and aligned with "even small gains pay off." Slight fp32-
accumulation perturbation (TF32 = 10-bit mantissa); the project has opted in.

## Contiguous dense prefill — built (MLX-parity), EXPERIMENTAL, `DIFFKV_CONTIGUOUS_PREFILL=1`

MLX's `_sparse_prefill_attend` keeps a rotated K/V buffer of ALL tokens so far and
does ONE flash SDPA per chunk. The CUDA default re-assembles history blocks +
re-RoPEs + eager matmul + LSE-merge every chunk (the ~1.4× forward). New branch
(`diffkv_attention.py`, fresh-prefill) replicates MLX: a per-(session,layer)
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
per-token @13.4k:   weights(nf4) 7.35 GB=4.74 ms   DiffKV store 0.95 GB=0.61 ms
dense step 89.3 ms (11.2 tps)     DiffKV step 129.9 ms (7.7 tps)     gap +40.6 ms
```

The DiffKV KV-store read a fused kernel could optimize is **0.61 ms (~0.5 % of the
step)**. The +40.6 ms gap is **per-token eager launch / Python overhead** — 48
layers × (routing + reconstruction + sparse attend), with CUDA graphs OFF for
mutable routing. Both dense and DiffKV run ~19× off their bandwidth roofline
(launch-bound + nf4 dequant), so DiffKV's *extra* launches are the gap, not kernel
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

`DIFFKV_CONTIGUOUS_PREFILL=1` on the A100:

```
preset   fwd before   fwd now   dense fwd
low        8.6 s       4.85 s      5.65 s
mid        8.5 s       4.85 s      5.65 s
high       7.3 s       4.01 s      5.65 s
```

The ~1.4× forward overhead is gone — DiffKV's forward is now *below* dense. The
single-SDPA-over-a-rotated-buffer replaced the per-chunk block re-assembly +
re-RoPE + eager matmul. Cost: peak rose (15.07 → 16.74 GB low) from the buffer
coexisting with the captured blocks (2× raw prefill KV).

## 1× variant built — `DIFFKV_CONTIG_UNROTATE=1` (with CONTIGUOUS_PREFILL=1)

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
prefill-next first-token line (recall proxy). `DIFFKV_DIAG=1` still gives the
runtime's own compress trace.

## CUDA graphs: honest assessment — a real refactor, NOT shippable blind

`CUDAGraphDecodeRunner` (`native_core/graph_runtime/static_decode_graph.py`)
exists with static input/output buffers and capture/replay, but capture is gated
behind `model._diffkv_cuda_graph_safe`, which is never set — correctly. CUDA-graph
replay records **kernels only**, not host Python. The DiffKV decode forward mutates
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
