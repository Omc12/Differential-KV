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

**Per-token RoPE is NOT cheap — correcting an earlier note in this file.**

An earlier revision said the fix was to store keys unrotated and rotate per token
at read, on the grounds that "the fused kernel already reconstructs per token, so
the rotation can ride along inside that loop". That is wrong, and the original
CUDA docstring's cost objection was right.

`triton_fused_decode.py:567` computes the query projection ONCE per chunk:

    q_proj_k = tl.sum(q[None, :] * vk_data, axis=1) * scale   # [RANK]
    # "Q projection -- computed once for this entire chunk"

and the token loop then only does a RANK-dim dot product against it. RoPE acts in
D-space while that projection is D -> rank, so rotating per token means
recomputing a D x rank projection for every token instead of once per block: S x
more work in the hottest loop, with S up to the block size. Rotating the key
instead is worse (it needs V_K rotated per token, a rank x D matrix each time).

The relative-position identity does not rescue it either. Within a block
positions are consecutive, so R(p_q - anchor - t) factors as R(p_q - anchor) .
R(-t) and the R(-t) part is shared across blocks -- but it still has to be applied
BEFORE the D -> rank projection, so the projection is still per token.

**So the practical path is dual-scale, not per-token rotation.** Small blocks
under an unrotated pool do bound the phase error -- block 128 with
`DKV_ROTATED_POOL=0` fails 2 needle cases against 3-4 at block 256/512 -- but it
does not reach clean, so this is a real dead end rather than a tuning problem.

---

## 5b. Block size 256 -> 1024 (CUDA default changed; check MLX's)

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

**CUDA now defaults to 1024**, chosen for LINKAGE. 1024 is the smallest block
that reaches dense parity on distractor retrieval (24/24), and it is also the
best point measured for prefill and VRAM. It costs synthesis, 50.0 -> 30.0 --
taken deliberately, because retrieving the right fact from a document full of
similar ones is the workload this system is for, and at 512 it lost 9 of 24 such
lookups that dense got right. Use `micro_block_size=512` for synthesis-shaped
work.

| block | linkbench | synthesis | TTFT (1.5B, 32k) | peak_alloc | needles |
|---|---|---|---|---|---|
| 256 (old default) | 14/24 | 46.7 | 15.17 s | 5.44 GB | 9/9 |
| **512 (new)** | **15/24** | **50.0** | **11.58 s** | **5.10 GB** | 9/9 |
| 1024 | **24/24** | 30.0 | 11.43 s | 4.96 GB | 9/9 |
| dense | 24/24 | 60.0 | 5.70 s | — | — |

512 beats 256 on all four; 1024 then buys the remaining 9 linkbench points for
synthesis. Decode differences across these sizes were inside the run-to-run band. Fewer, larger
blocks mean fewer per-block compressions, and that is where prefill time goes --
on a non-hybrid model where all 28 layers are compressed, the model forward is
only about a third of prefill.

512 remains the best single point for synthesis if that is your workload.

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

### The two metrics want opposite block sizes, and nothing bridges them

Full sweep on Qwen3.5-2B at 16k:

| block | linkbench | synthesis (facts/links) |
|---|---|---|
| 256 | 14/24 | 46.7 (8/2) |
| 512 | 15/24 | **50.0** (6/3) |
| 1024 | **24/24** | 30.0 (6/1) |
| 1536 | — | 26.7 (5/1) — fails the >=30 bar |
| 2048 | **24/24** (at 32k) | 33.3 (4/2) |

Synthesis peaks at 512 and degrades as blocks grow; distractor retrieval rises
monotonically with block size. There is no crossover point that serves both.

Everything that might have bridged them was measured and does not:

* rank scaled WITH block size (1024 at rank 32 / 64 / 128) -> synthesis 30.0 in
  all three, so the synthesis loss is not lost per-token fidelity
* routing coverage (`DKV_TOPK_FRAC` 0.0/0.5/1.0, `DKV_TOPK_BLOCKS=0`) -> no
  effect on either
* residual budget, recency window -> no effect

### So dual-scale is required, and here is the shape of it

The same content compressed at BOTH granularities, with attention seeing both:
the coarse scale keeps associations inside one unit, the fine scale keeps the
granularity synthesis needs.

Because routing is provably irrelevant here, a multi-scale ROUTER cannot help --
both scales have to reach ATTENTION. The cheapest correct structure found on the
CUDA side, not yet implemented:

1. a second pool at the coarse scale (a single pool cannot hold both: its per-slot
   tensors are sized `[n_blocks, block_size-1, rank]`, so sizing for the coarse
   scale inflates every fine slot by the same factor)
2. ingest writing both scales from the same token stream
3. decode running the existing fused kernel once per scale and merging the two
   outputs by log-sum-exp -- the prefill path already does exactly this merge for
   chunked attention, so the kernel itself does not need to change

Cost is roughly 2x pool memory and a second decode kernel launch per layer.

This also supersedes the rotated-pool theory in item 5 as the *practical* lever:
`DKV_ROTATED_POOL=0` closes linkbench too, but breaks needle ORDER (edit-distance-1
transpositions), whereas block size closes it with needles intact. Both point at
the same underlying thing -- how much positional and associative structure
survives one block -- and the real fix for either is multi-scale blocks or
per-token rotation, not a different constant.

---

## 5c. Why the VRAM saving looks small — it is the denominator, not the compression

**MLX status: applies to any comparison you run.** If MLX appears to save far more
than CUDA, check what each number is a fraction OF before concluding the
compression differs.

Measured on Qwen3.5-2B at 32k (6 attended layers, 2 kv_heads, head_dim 256), pool
against the dense KV it replaces for the same tokens:

| block | pool | dense-KV equivalent | ratio | residual share |
|---|---|---|---|---|
| 256 | 311.6 MB | 393.9 MB | 0.79x | 64% of pool |
| **1024** | **105.3 MB** | 393.9 MB | **0.27x** | 49% |

So the compression itself is fine — at the current default the pool is **3.7x
smaller** than the KV it stands in for. The reason total VRAM barely moves is that
the pool is 105 MB of a ~5 GB footprint: the model weights (~4 GB in fp16) dominate,
so a 289 MB KV saving is ~5% of the total.

A KV-compression system only shows a large TOTAL saving where KV outweighs
weights. At 32k with 6 attended layers and 2 KV heads that is nowhere near true.
It becomes true with much longer context, more KV heads (a non-GQA or wide-GQA
model), batching, or a quantized model where weights shrink but KV does not.

**Report the KV-side ratio, not total device memory**, or the number is dominated
by whatever the weights happen to cost.

Two consequences worth carrying:

* Block size drives this too. At 256 the residual budget (a fixed 128 tokens per
  block regardless of block size) was 64% of the pool, so most of the "compressed"
  store was exact tokens. At 1024 the same budget covers 4x the tokens and the
  ratio improves 0.79x -> 0.27x.
* **Dual-scale is affordable.** A second pool at a coarser scale roughly doubles
  105 MB, i.e. ~2% of the footprint, not "double the VRAM". An earlier note in
  this file worried that dual-scale would give back the VRAM result; that was
  wrong by an order of magnitude.

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

## 7. Prefill's history attention was eager — fuse it

**Priority: high — it is the single biggest prefill win found, and it improved
accuracy rather than costing it.**

**MLX status: CHECK THIS.** CUDA's site is the history branch of
`_sparse_prefill_attend` in `runtime/dkv_attention.py`; MLX has a function of the
same name. If it builds a full score matrix and calls softmax itself, the same
change applies.

**What CUDA found.** Profiling prefill at 32k on Qwen2.5-1.5B contradicted the
obvious assumption that compression dominates. Of an 8.4 s kernel budget the SVD
is only **8%** (eigh 457 ms + qr 214 ms); GEMM is ~25%, softmax ~8%, elementwise
~22%. That is the history cross-attention, which ran as
`matmul -> logsumexp -> softmax -> matmul` and materialised the whole
`[B, H, Lq, Lk]` score matrix.

Replacing it with a fused attention call gave:

    TTFT        12.11 -> 8.25 s   (-32%)
    peak_alloc   5.17 -> 5.03 GB  (the score matrix stops existing)

**The constraint that shapes the fix.** This path's output is merged with the
local and compressed paths by log-sum-exp, so the call must return the LSE, not
just the output. On CUDA the public `scaled_dot_product_attention` is therefore
unusable and the change goes through
`torch.ops.aten._scaled_dot_product_efficient_attention`, which returns
`(out, logsumexp, ...)`. **Check whether MLX's fused attention primitive
(`mx.fast.scaled_dot_product_attention`) can return the LSE.** If it cannot, this
item does not port as written — do not approximate the merge to get the speed.

No mask, `is_causal=False`: every key on this path is history strictly before the
chunk, so it is visible to every query in it.

**Accuracy moved, and up.** A fused kernel reduces in a different order and flips
the occasional greedy tie, so output is not bit-identical and this needs the full
suite rather than an assumption. On CUDA: multifact synthesis **43.3 -> 50.0**
(links 2/5 -> 3/5), linkbench still 24/24, needles 9/9 on both models.

---

## 8. Fidelity is a preset ladder, not one constant

**Priority: medium — it is what fixed synthesis without touching block size.**

**MLX status: LIKELY.** CUDA's rank-selection site keeps singular values until
cumulative energy reaches a hardcoded `0.999`. Look for the same constant in
MLX's low-rank path.

**What CUDA found.** With block size pinned at 1024 for linkage, synthesis was
stuck at 30.0. The lever that moved it was not routing, residuals or block size
(all measured, all flat — see 5b) but **how much spectral energy the SVD keeps**.
`0.999` truncates harder than it looks: it is the knee of a fast-decaying
spectrum, so the tail it drops still carries the distinctions synthesis needs.

CUDA now ties energy and rank to the existing quality presets rather than picking
one number:

| preset | svd_energy | rank |
|---|---|---|
| low | 0.999 | 32 |
| mid | 0.9999 | 64 |
| high | 0.99999 | 128 |

This is the "smarter routing instead of more rank" question answered the other
way: routing genuinely cannot help (5b measured that), so the fidelity has to
come from the spectrum — but paying for it is now the *caller's* choice per
preset instead of a constant everyone pays.

Related measurement, worth not repeating: trading residual budget for rank does
NOT work once rank carries more of the fidelity. `max_residual` 128/64/32 gives
synthesis 43.3/40.0/23.3, so the residual bytes still earn their place and are
not a place to reclaim memory from.

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
