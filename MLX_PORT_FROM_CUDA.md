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

## 5. The pool stores POST-RoPE keys, which costs accuracy at long context

**Priority: highest accuracy item found so far. Not yet fixed on CUDA either —
this is a shared open bug, not a port.**

**MLX status: CHECK THIS.** CUDA's flag is `DKV_ROTATED_POOL` (default `1` =
store rotated). Find MLX's equivalent before assuming it differs.

**What CUDA found.** A new benchmark (`colab/linkbench_cuda.py`) plants 16
near-identical sentences ("The X Institute is located in Y") and asks for one of
them. On 24 seeds at 16k on Qwen3.5-2B:

    dense   24/24
    DKV     14/24     (Fisher p ~ 0.0004)

This is a SINGLE-HOP lookup, and the needle benchmark cannot see it, because one
unique code in bland filler has no confusable distractors.

It is not a fidelity problem. Every knob leaves it at **exactly** 14/24 -- routing
K (0/32/default), residual budget (32/128/224), recency window (512/4096), SVD
rank (32/96). Only context length moves it: 4k 24/24, 8k 19/24, 16k 14/24.

`DKV_ROTATED_POOL=0` gives **24/24 at 8k, 16k and 32k** -- the whole gap closes.
Storing post-RoPE keys bakes in the position a block held at compression time,
so the values were never wrong; the positions were.

**Why it is not simply flipped on:** with `DKV_ROTATED_POOL=0` the needle sweep
falls 9/9 -> 6/9, including a one-character near-miss (`Falcon-9427-613` for
`-6183`) and two failures at 2k, where nothing is compressed at all. Failing at
2k means the un-rotated READ path is buggy, not that this is a real trade. The
fix is to make the rotated path carry correct positions.

**MLX almost certainly shares this.** `triton_fused_decode.py:242` documents the
two designs explicitly and cites `mlx_dkv_wrapper.py:4565/4613`: MLX rotates keys
and THEN compresses them, which is exactly the `DKV_ROTATED_POOL=1` design. So
the distractor-heavy weakness measured here is a property of the shared
architecture, not of the CUDA port, and `linkbench_cuda.py` should reproduce it
on MLX.

**Both modes are wrong, in complementary ways.** Neither is a fix to adopt:

| | linkbench (content) | needle sweep (order) |
|---|---|---|
| `=1` rotated, MLX parity | 14/24 | 9/9 |
| `=0` unrotated | 24/24 at 8k/16k/32k | 6/9 |

The `=0` needle failures are all EDIT DISTANCE 1 and all order errors --
`Falcon-94276-6183`, `Falcon-9427-6138`, `Falcon-9427-613` for `Falcon-9427-6183`.
Right digits, wrong order. That is the fingerprint of the phase error the
docstring predicts: the unrotated path rotates the anchor and the whole V_K basis
at the ANCHOR's position, so every token in a block shares one rotation and
carries a positional error of up to a full block (256 positions). Raising the
residual budget to 224 of 256 does NOT repair it (still 4 failures, still all
near-misses), because a residual is rotated at its token's true position while
the base term it corrects is rotated at the anchor -- they are in different
frames and their sum is exact in neither.

**The actual fix, for whichever runtime does it first:** store keys UNROTATED (so
the SVD fits un-mixed vectors, which is what recovers the content discrimination)
and apply RoPE PER TOKEN at read, rather than once per block at the anchor. The
CUDA docstring rejected this as "a D-dim reconstruction per token", but the fused
decode kernel already reconstructs per token to score it, so the rotation can ride
along inside that loop instead of being a separate pass. That is kernel work on
CUDA and would be `mx.fast.rope` inside the equivalent MLX path.

---

## 5b. Block size 256 -> 512 (CUDA default changed; check MLX's)

**Priority: high — it is the only knob that moves either, and MLX has the same
constant.**

**MLX status: LIKELY** — MLX takes `block_size` in `MLXKVBlockManager.__init__`
(`mlx_dkv_wrapper.py:1656`). Check its default against CUDA's 256.

Measured on Qwen3.5-2B at 16k, CUDA:

| block | linkbench (24 seeds) | multifact synthesis | needles |
|---|---|---|---|
| 128 | 11/24 | — | — |
| 256 | 14/24 | 46.7 (8 facts, 2 links) | 9/9 |
| 512 | 15/24 | — | — |
| 1024 | **24/24** | 30.0 (6 facts, 1 link) | 9/9 |
| dense | 24/24 | 60.0 (9 facts, 3 links) | — |

Big blocks keep an association intact inside one block, which distractor-heavy
retrieval needs. Small blocks give routing finer granularity to assemble diverse
content, which synthesis needs.

**CUDA now defaults to 512**, because with prefill and VRAM measured too, 512
turned out to DOMINATE 256 rather than trade against it:

| block | linkbench | synthesis | TTFT (1.5B, 32k) | peak_alloc | needles |
|---|---|---|---|---|---|
| 256 (old default) | 14/24 | 46.7 | 15.17 s | 5.44 GB | 9/9 |
| **512 (new)** | **15/24** | **50.0** | **11.58 s** | **5.10 GB** | 9/9 |
| 1024 | **24/24** | 30.0 | 11.43 s | 4.96 GB | 9/9 |
| dense | 24/24 | 60.0 | 5.70 s | — | — |

512 is better on prefill (-24%), VRAM, synthesis and distractor retrieval at
once; decode measured 2-4% lower, inside the run-to-run band. Fewer, larger
blocks mean fewer per-block compressions, and that is where prefill time goes --
on a non-hybrid model where all 28 layers are compressed, the model forward is
only about a third of prefill.

1024 is NOT taken despite reaching dense parity on distractor retrieval, because
it costs synthesis 50.0 -> 30.0. Use `micro_block_size=1024` for
distractor-heavy retrieval specifically.

**On MLX: check the block_size default and sweep it.** This is the strongest
single lever found in the whole CUDA effort, and it moved four metrics at once.

### What actually drives it (measured, and it is not what it looks like)

Distractor retrieval tracks the NUMBER OF BLOCKS the representation is split
into, and reaches dense parity whenever that number is small, regardless of how
it got small:

| block | ctx | ~blocks | linkbench |
|---|---|---|---|
| 256 | 4k | 15 | 24/24 |
| 1024 | 16k | 15 | 24/24 |
| 2048 | 32k | 14 | 24/24 |
| 1024 | 32k | 29 | 23/24 |
| 512 | 16k | 29 | 15/24 |
| 256 | 16k | 58 | 14/24 |

**It is NOT routing coverage.** `DKV_TOPK_FRAC` at 0.0 / 0.5 / 1.0 all give
15/24, and `DKV_TOPK_BLOCKS=0` (attend every block) gives 14/24. Letting the
model see every block does not help at all.

**It is NOT reconstruction fidelity.** SVD rank (32/96), residual budget
(32/128/224) and recency window (512/4096) all leave it at EXACTLY 14/24 --
and note a 1024-token block has WORSE per-token fidelity than a 512-token one
(same rank 32, same 128 residuals spread over 4x the tokens) yet scores better.

So what a bigger block buys is not more information and not more of it being
looked at -- it is that an association stays inside ONE unit. Splitting a
document into more pieces destroys cross-piece associations no matter how
faithfully each piece is stored or how many are retrieved.

That is why the fix is multi-scale blocks rather than a better constant: the same
content compressed at two granularities, with attention seeing both, so
associations survive at the coarse scale while the fine scale keeps the
granularity synthesis needs. Note the corollary -- because routing is provably
irrelevant here, a multi-scale ROUTER would not help; the two scales have to
both reach attention.

This also supersedes the rotated-pool theory in item 5 as the *practical* lever:
`DKV_ROTATED_POOL=0` closes linkbench too, but breaks needle ORDER (edit-distance-1
transpositions), whereas block size closes it with needles intact. Both point at
the same underlying thing -- how much positional and associative structure
survives one block -- and the real fix for either is multi-scale blocks or
per-token rotation, not a different constant.

---

## 6. Model load kept a full CPU copy of the weights (CUDA-only, fixed)

**MLX status: N/A** — different loader entirely. Recorded because it explains a
number you may have seen.

The CUDA wrapper loaded with `from_pretrained(...).to(device)`, which
materialises every weight on the CPU and leaves the host pages resident:
Qwen3.5-2B held 6.22 GB RSS against dense's 2.07 GB for the same work. Switching
to `device_map=device` streams shards straight to the GPU: **RSS 6.22 -> 2.60 GB**,
GPU unchanged. DKV's own structures held 0.00 GB of CPU tensors, so this was the
entire system-RAM gap.

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
