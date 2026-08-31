# MLX's K=4 default was resting on a benchmark with no power — the floor is now 8

**Status: APPLIED.** `_K_MIN` 2 -> 8 in `ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py`,
so `max(_K_MIN, 4096 // block_size)` gives **K=8** at the shipped `block_size=1024`
(K=16 at 256 is unchanged, so every pre-1024 measurement stays valid).

**Date:** 2026-08-30 · **Hardware:** Apple silicon, 8 GB
**Model:** `mlx-community/Qwen3.5-2B-4bit` · **Harness:** `colab/synthesis_power.py`

## Why the old evidence did not support K=4

`1157f3a2` set the floor to 2 citing *linkbench 16k, 24 seeds: 24/24, unchanged
from dense*. Run **with a dense control** on CUDA, dense scores the same as DKV on
that benchmark (10/24 at 16k, 6/12 at 32k). When the dense arm cannot beat the
routed arm, the benchmark is at the model's ceiling and has **no power** to detect
a routing regression: "24/24, unchanged" is equally consistent with K=4 being fine
and with K=4 being much worse. Synthesis, which does have power, was never run on
MLX — the gap this closes.

## The measurement

16k context, `block_size=1024`, paired (every arm sees the same replicate list),
**n=32**:

| K | mean | sd | vs attend-all (paired, 95% CI) | reps < 30 | worst rep |
|---|---|---|---|---|---|
| 4 (shipped) | 41.8 | 11.9 | **−5.42  [−9.99, −0.84]  RESOLVED** | **7 / 32** | 16.7 |
| 8 | 44.9 | 7.9 | −2.29  [−5.09, +0.51]  not resolvable | 0 / 32 | 30.0 |
| 16 | 47.2 | 7.5 | — (attend-all at this ctx) | 0 / 32 | 36.7 |

**n mattered.** At n=16 the same K=4 comparison read −6.67 with CI [−13.69, +0.35],
straddling zero. Do not re-decide this at n=8 or n=16.

**The tail decided it, not the mean.** K=4 does not degrade gracefully — it fails
badly on roughly a fifth of inputs, producing answers (16.7 / 20.0 / 23.3) that
neither other arm ever produced. A 5-point mean difference understates that.

K=8 is therefore the smallest K that is not resolvably behind attend-all and never
produced a catastrophic answer. CUDA independently put its knee at 8 as well.

## Read the K=16 arm correctly

At 16k with `block_size=1024` there are **nb=15** blocks, and both MLX decode sites
guard routing with `nb > k_eff`. With `k_eff=16` that guard is **false**, so the
K=16 arm is **attend-all** — the quality ceiling, not "routing 16 blocks". Verified
directly before any score was read, by instrumenting `_route_k`:

    DKV_TOPK_BLOCKS=4   -> nb=15, k_eff=4,  top-K branch fired 12/12  (routing)
    DKV_TOPK_BLOCKS=16  -> nb=15, k_eff=16, top-K branch fired 0/12   (attend-all)

This makes it the strictest available control — K=8 tying it means K=8 loses
nothing to routing at this context.

## The routing-vs-routing comparison, at 32k

Run afterwards to close that gap, since at 16k the K=16 arm is not routing at all.
At 32k there are **nb=31** blocks, so both arms genuinely route (verified: K=16
fired the top-K branch 6/6 calls). Same harness, paired, n=32:

| K | mean | sd | vs K=16 (paired, 95% CI) | reps < 30 | worst rep |
|---|---|---|---|---|---|
| 8 | 43.0 | 9.8 | **−3.44  [−7.78, +0.91]  not resolvable** | 2 / 32 | 20.0 |
| 16 | 46.5 | 9.5 | — | 1 / 32 | **16.7** |

Per-replicate, K=8 was worse in 15 of 32, better in 11, tied in 6 — close to even.

So K=8 vs K=16 is unresolved at BOTH contexts, and in the same direction each time
(−2.29 at 16k, −3.44 at 32k). Two cautions before reading that as a real ~3-point
deficit:

* they are **not the same comparison** — at 16k the K=16 arm is attend-all, at 32k
  it routes 16 of 31 — so the two should not simply be pooled;
* **the tail argument that killed K=4 does not reproduce here.** K=4's case rested
  on 7 of 32 reps below 30 points against zero for the other arms. At 32k both K=8
  and K=16 produce the occasional bad rep and K=16 owns the single worst one, so
  there is no failure-mode difference between them, only a small mean gap.

That is why the floor is 8 rather than 16: the resolved, catastrophic failure is at
K=4, and K=8 removes it while keeping half of K=16's decode cache. The residual
K=8-vs-K=16 gap is real in direction but unresolved in size at n=32, and sits under
this harness's ~3-point noise floor. **If it needs settling, n=64 at 32k is the
run** — resolution at n=32 was ±4.34, so roughly n≈51 would resolve a −3.44.

## Cost, and why CUDA still floors at 16

MLX's decode cache is ON and sized `k_eff * block_size`, so K controls memory as
well as quality here and this floor gives back part of the 377.9 -> 151.1 MB saving
that motivated K=4. That trade is the point: the saving was being paid for in
quality that nobody had measured. CUDA's cache is OFF by default
(`DKV_DECODE_CACHE_CUDA=0`), so a smaller K there collects no memory and only costs
quality — hence 16 there, 8 here. The asymmetry is measured on both sides and is
pinned with its reasoning in `ACTIVE_RUNTIME/tests/test_routing_k_budget_parity.py`.

## What the floor actually costs (measured, not extrapolated)

The whole case for K=4 was memory, so raising the floor without measuring the cost
would have been the same mistake in the other direction. Qwen3.5-2B-4bit, 16k,
block_size=1024, live `_cache_kv` bytes and MLX peak:

| K | decode cache | peak memory |
|---|---|---|
| 4 | 151.1 MB | 2329.9 MB |
| **8** | **251.9 MB** | **2329.9 MB** |
| 16 | 428.2 MB | 2329.9 MB |

K=4's 151.1 MB reproduces the figure in `1157f3a2` exactly, which says the probe is
measuring the same quantity that decision was made on.

**Peak is byte-identical at all three.** The decode cache is not what sets peak —
prefill activations are, and they happen before it is allocated. So the floor costs
+100.8 MB of *resident decode-cache* footprint and nothing in peak. That is the
same pattern CUDA reported (peak VRAM identical at K=4 and K=16), now confirmed on
MLX, and it makes the original memory argument for K=4 considerably weaker than it
looked: the 377.9 -> 151.1 MB saving was in a buffer that never drove the ceiling.

Two honest qualifications: peak is not the only thing that matters — resident bytes
still bound how many sessions fit concurrently — and this is one context on one
model. But on the axis the K=4 decision was actually argued on, the cost of moving
to 8 is smaller than that decision assumed.

## Caveats

* **Hybrid model.** DKV patches only 6 of 24 layers on Qwen3.5-2B
  (`[3, 7, 11, 15, 19, 23]`), which bounds how large any routing effect can be and
  likely explains why MLX's −5.42 is smaller than CUDA's −16.2 for the same change.
* **One context, one model.** The quality numbers are 16k (plus the 32k
  routing-vs-routing run); the memory numbers are 16k only.
* Reproduce with:

      BLOCK=1024 DKV_TOPK_BLOCKS=8 python3 colab/synthesis_power.py \
          --arm dkv --reps 32 --ctx 16384 --out mlx_k8.json
