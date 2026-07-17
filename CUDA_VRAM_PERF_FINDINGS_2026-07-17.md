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
