# Changes to port from CUDA to MLX

Written 2026-08-13, from the CUDA work in commits `a5289a18..7c63bdca`.

**Scope.** Most of that work was CUDA catching up to MLX, or fixing things that
only exist on CUDA (the PyTorch caching allocator, Triton kernel grids, CUDA
graphs). Those are **not** listed here. This file lists only the items where the
CUDA work found something MLX plausibly shares, ordered by how much they matter.

**Confidence labels.** `VERIFIED` means I read the MLX source and the same
construct is there — line numbers are from `ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py`.
`LIKELY` means the mechanism is shared but I did not confirm the MLX code path.
I did not run anything on MLX; nothing here is measured on Apple silicon.

---

## 1. Routed-block count K=16 is tuned for retrieval, not synthesis

**Priority: highest — this is an accuracy change, and it is free on speed.**

**MLX status: VERIFIED.** `mlx_dkv_wrapper.py:1713-1724` — "Default is fixed
K=16 (topk_frac=0)", read from `DKV_TOPK_BLOCKS`. Identical default to CUDA's
(CUDA derives 16 from the pool span; same number, same effect).

**What CUDA found.** K=16 is enough to find a needle but not to hold a topic. On
`colab/multifact_eval_cuda.py` at 16k on Qwen3.5-2B, each setting reproduced
twice:

| K | synthesis | facts | links |
|---|---|---|---|
| 16 (default) | 46.7 | 8/15 | 2/5 |
| 32 | **60.0** | 9/15 | 3/5 |
| 48 | 60.0 | — | — (saturated) |

**A dense control scores 60.0 (9/15, 3/5).** So at K=16 DKV is measurably below
dense on synthesis, and at K=32 it reaches parity. That is the single largest
accuracy finding of the session.

Cost on CUDA was nil — 2.4% *faster* at 8.4k (95% CI [-3.5%, -1.2%]) and no
resolvable difference at 32k — because more routed blocks give the decode kernel
more parallelism, which offsets the extra work. **That offset is CUDA-specific
(it comes from the kernel's grid), so measure the cost on MLX rather than
assuming it.**

**Caveat, and why CUDA did not make it the default.** The same K=32 reproducibly
*hurt* Qwen2.5-1.5B: synthesis 26.7 -> 16.7, identical facts (5/15), one fewer
link. It helps the capable model and hurts the weak one, so on CUDA it stayed a
knob. Decide for MLX with your own models.

**To try:** `DKV_TOPK_BLOCKS=32`, then run the multifact synthesis case and a
needle sweep. Needle recall did not move on CUDA (9/9 either way) — synthesis is
the metric that responds.

---

## 2. Decode-cache interval 16 buys nothing over 4

**Priority: medium — pure staleness reduction, no measured cost.**

**MLX status: VERIFIED.** `mlx_dkv_wrapper.py:1784` —
`DKV_DECODE_CACHE_INTERVAL` defaults to `16`. CUDA's port of this
(`DKV_REMAT_INTERVAL`) now defaults to `4`.

**What CUDA found.** Throughput is flat across the interval range: 18.28 / 18.61
/ 18.52 tok/s at 16 / 8 / 4 on Qwen2.5-1.5B at 8.4k, and a paired benchmark
(interval 16 vs 4, 8 rounds) reported no resolvable difference: +0.425 ms, 95% CI
[-0.152, +1.002], resolution ±1.5%. Skipping reconstruction on 3 steps in 4
already captures nearly all of the saving, so 16 is paying staleness for nothing.

Interval 4 is therefore the least staleness the speedup will pay for. The cache
freezes the routed block set for the interval, so a shorter interval means a
needle whose block is routed late becomes visible sooner.

**To try:** `DKV_DECODE_CACHE_INTERVAL=4`. Expect no speed change; the benefit is
that the frozen-routing window is 4x shorter.

---

## 3. Pool is sized by total layers, not attended layers

**Priority: medium on hybrid models (Qwen3.5), none on dense-attention models.**

**MLX status: LIKELY.** `mlx_dkv_wrapper.py:2030-2033` allocates one array per
layer for `range(self.num_layers)` — `comp_U`, `comp_U_scale`, `comp_VK` etc.,
each sized `max_blocks`. That is the same shape of allocation CUDA had. I did not
confirm whether MLX skips linear-attention layers when filling them.

**What CUDA found.** Only full-attention layers hold compressed blocks. On
Qwen3.5-2B that is **6 of 24 layers**; on Qwen3.5-9B, **8 of 32**. Sizing by
`config.num_hidden_layers` therefore reserved 4x the slots the session could ever
use — measured 1246 MB of pool at 32k where the blocks needed ~312 MB.

CUDA's fix publishes the attended-layer count after the attention patch runs and
sizes from that. Under-estimating is safe there because slots come from one
global free list with a grow path; **check that MLX's allocator has the same
property before shrinking anything.**

Effect on CUDA (Qwen3.5-2B peak tensor bytes): 8.4k 4.42 -> 4.17 GB, 32k 5.95 ->
4.97 GB. Non-hybrid models are unaffected (28 of 28 layers attended, so the count
is identical).

**To check on MLX:** print the number of layers that actually receive compressed
blocks versus `num_layers` on a Qwen3.5 model. If they differ, the same
over-allocation is present.

---

## 4. Benchmark trap: reasoning models fail on the answer budget, not on recall

**Priority: high if you validate Qwen3.5-9B or any thinking model — otherwise
you will read a working build as broken.**

**MLX status: N/A to the runtime; applies to whatever validator you run.**

Qwen3.5-9B emits a long `<think>` preamble before answering. At the CUDA
validator's default budget of 24 answer tokens it never reached the code and
scored **0/3 at every depth and every length** — which reads as total recall
failure. At `DKV_VALIDATE_MAX_NEW=400` the same build scores **3/3 at all three
32k depths** and 3/3 across the 2k set.

CUDA's validator now detects an unclosed `<think>` in a failing case and prints
`TRUNCATED MID-<think>` with the budget to re-run at, instead of reporting a
recall failure that is not one. Worth mirroring in whatever harness you use on
the MacBook.

---

## Not portable — CUDA-only, listed so you do not go looking

* **Occupancy-driven decode chunking** (`a5289a18`) — sizes a Triton kernel grid
  from SM count. No Metal analogue.
* **`expandable_segments` and the post-prefill `empty_cache`** (`48e6b844`,
  `a411e033`) — PyTorch caching-allocator fragmentation. 2.6 GB on CUDA; MLX's
  unified memory does not have this failure mode.
* **Fixed-shape routing** (`fbfd25fe`, `cce2e0d2`) — removes `torch.nonzero`
  because its data-dependent output shape blocks CUDA graph capture and syncs.
  Worth +8.4% on CUDA (95% CI [4.4%, 12.4%]). MLX has no equivalent capture
  constraint.
* **`gc.collect()` removals** (`48e6b844`, `fbfd25fe`) — CPython + PyTorch
  allocator interaction. Measured: the collect cost 4.3 s of prefill and freed
  **zero** bytes, because refcounting had already released the tensors.
* **CUDA graphs** generally — bypass path works (1.25x, bit-exact); routed path
  captures but replays stale state and is not faster.

---

## Method note, if you re-measure any of this on the MacBook

Decode throughput on the CUDA box moved ~20% between runs of the same config,
which is wider than every effect above except item 1's accuracy delta. Three
claims made from unpaired runs turned out to be wrong when measured properly (one
reversed sign entirely). `colab/bench_decode_paired.py` is the harness that fixed
it: both arms in one process against one model load, interleaved with alternating
order, paired statistic, **minimum** rather than median per round (timing noise is
one-sided), and the session cleared between runs. It is calibrated both ways — an
A/A control must report no effect, and a known effect must be detected.

Also recorded there: do not size a change from cProfile self-time. It charges its
own per-call overhead to the callee, so high-count cheap builtins look expensive.
Cutting one such call 229.8 -> 56.4 per token produced no measurable change,
because `Tensor.to()` returns self when the tensor is already on-device.
