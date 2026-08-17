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

> ## ⚠ READ ITEM 10 BEFORE TRUSTING ANY `multifact` NUMBER HERE
>
> The randomised SVD draws its projection from a seeded generator whose SHAPE
> depends on the configured rank. Changing `DKV_RSVD_SEED` alone, at a fixed
> config, moves the synthesis score **63.3 / 33.3 / 50.0** — a 30-point spread
> with nothing else altered.
>
> Temperature-0 replication is deterministic and therefore proves nothing: a
> number "reproduced twice" is one sample, not two. **Any synthesis difference
> under ~15 points is not a difference.** Several items below quote single-seed
> numbers from before this was understood; each is now flagged inline, and
> item 10 is a full retraction. MLX runs the same randomised SVD
> (`DKV_SVD_SEED`) and has the same noise floor.
>
> **Use `colab/synthesis_power.py` instead** - replicated, paired against dense,
> with a confidence interval. It resolves differences the old harness could not
> (item 12), and reports how many replicates a given effect size needs.

---

## 1. RETRACTED: K=16 vs K=32 makes no difference. Routing is not a lever.

**Priority: read before spending any time on routing. This closes the question
with three independent methods.**

**MLX status: the knob is VERIFIED to exist** (`mlx_dkv_wrapper.py:1713-1724`,
`DKV_TOPK_BLOCKS`, same default). **Do not bother changing it.**

This item used to claim K=32 lifted synthesis 46.7 -> 60.0 and called it "the
single largest accuracy finding of the session". That was one deterministic run
per setting on a fixed document window, i.e. one sample, inside a +-15-point seed
band. Re-measured with `colab/synthesis_power.py` -- replicated, paired,
interval-bounded -- it does not reproduce.

**Measured where routing actually BINDS.** At 16k with block 1024 there are only
~15 blocks, so K=16 already routes every one of them and K cannot do anything;
an earlier re-test at 16k was therefore also meaningless. At 32k there are 31
blocks and K=16 routes about half. Qwen3.5-2B, 4 paired replicates at 32k:

| arm | mean | sd |
|---|---|---|
| K=16 (default) | 62.5 | 4.2 |
| K=32 | 62.5 | 6.3 |

    paired mean_diff = +0.00   95% CI [-4.33, +4.33]

Exactly zero, with a resolution of +-4.3 points -- tight enough that a 5-point
effect would have shown.

**Three independent methods now agree that routing does nothing:**

| method | result |
|---|---|
| linkbench, 48 seeds, K=16 vs attend-EVERY-block | 47/48 vs 47/48 |
| generated prose (`qual_routing_cuda.py`), K=16 vs attend-all | **byte-identical** |
| synthesis, paired and powered, K=16 vs K=32 @32k | +0.00 +-4.33 |

**Why, structurally.** Retrieval and synthesis were never *selection* problems.
Showing the model every block changes nothing, so the router is not missing
anything -- what limits DKV is what the blocks CONTAIN, not which are chosen.
That is why the unrotated-pool change (item 5) moved linkbench 40 -> 47 while no
routing knob has ever moved anything.

**The one caveat worth keeping:** "attend everything" is not a strict upper bound
on quality, since attending more can dilute attention -- that is what killed
dual-scale (item 10d). A cleverer *subset* could in principle beat both. But no
evidence supports it, and three methods say the current router is not the
constraint.

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

## 5. RESOLVED: store keys UNROTATED — it reaches dense parity and the blocker is gone

**Priority: highest actionable accuracy item in this file. This is the one change
that measurably closes a gap to dense on a metric that can resolve it.**

**MLX status: MLX SHARES THE PROBLEM AND SHOULD TAKE THE FIX.**
`triton_fused_decode.py` cites `mlx_dkv_wrapper.py:4565/4613`: MLX rotates keys
and THEN compresses them, which is exactly the rotated design. So the weakness
below is a property of the shared architecture, not of the CUDA port.

**The problem.** `colab/linkbench_cuda.py` plants 16 near-identical sentences
("The X Institute is located in Y") and asks for one. Storing POST-RoPE keys
bakes in the position a block held at COMPRESSION time, so near-identical
distractors collapse together at long context. The values were never wrong; the
positions were. The needle benchmark cannot see this — one unique code in bland
filler has no confusable distractors.

**It is not a fidelity problem.** Every knob left it unchanged: routing K, residual
budget, recency window, SVD rank. Only context length moved it.

**What changed.** This item previously said unrotated keys could not be adopted
because the needle sweep fell 9/9 → 6/9, *including failures at 2k where nothing
is compressed*. That was read as a broken unrotated READ path rather than a real
trade — and that reading was correct. **It is now fixed.** On the current build,
`DKV_ROTATED_POOL=0` scores **9/9 with 9/9 determinism on BOTH Qwen3.5-2B and
Qwen2.5-1.5B-Instruct**, at every depth and every length.

With the blocker gone, linkbench at 32k over **48 seeds** (48 samples per point —
unlike multifact, whose ±15-point seed band cannot resolve anything, see item 10):

| arm | 48 seeds |
|---|---|
| rotated (default) | 40/48 |
| **UNROTATED** | **47/48** |
| dense | 47/48 |

**Exact parity with dense**, up from a 7-point deficit. Fisher p ≈ 0.03 against
the rotated arm.

> **THESE NUMBERS ARE `QMODE=direct`. RECORD THE MODE WITH THE SCORE.**
> They reproduce exactly -- dense 47/48, unrotated 47/48, rotated 40/48 -- but
> only in the mode they were taken in, and the mode was never written down.
>
> linkbench has two question modes and `chain` is the DEFAULT. `direct` names the
> intermediate entity outright, collapsing the multi-hop chain to a single lookup
> over the same context, and is much easier. The same three arms in `chain`:
>
> | arm | `direct` | `chain` |
> |---|---|---|
> | rotated | 40/48 | 21/48 |
> | **unrotated** | **47/48** | **23/48** |
> | dense | 47/48 | 23/48 |
>
> This cost an afternoon. Re-run in `chain` the scores looked halved, the DENSE
> arm halved with them, and since dense shares no DKV code that was read as an
> environment shift -- a `transformers`/`torch` update was the leading suspect.
> It was not. Packages have not changed since 2026-08-10, before these were
> recorded; the transformers 5.14.1 bump predates them by three weeks; the
> harness has not been touched since before them either. Two different benchmarks
> were being compared.
>
> **The finding holds in BOTH modes**: `unrotated == dense` exactly, rotated
> below both. That is what makes it worth porting.
>
> Two lessons. **Record the harness MODE next to every score, not just the
> harness name.** And **a score is only meaningful next to a control in the same
> configuration** -- the dense arm is what turned "DKV regressed" into "these are
> different tasks", and without it the wrong conclusion fit the data perfectly.

**The cost, which is why it is not the global default.** Rotating at read time
costs decode and memory. Qwen3.5-2B at 32k, interleaved and reversed:

    decode       17.60 -> 13.37  and  15.55 -> 12.70 tok/s   (-18% to -24%)
    TTFT          9.82 -> 10.15 s
    device VRAM   5.21 -> 6.31 GB

**It is a STANDALONE knob, not an `ultra` feature.** `mid` with the unrotated pool
also scores **47/48** over the same 48 seeds -- identical to ultra. The whole win
comes from the unrotated pool and none of it from ultra's rank or energy, so
setting it on any preset buys dense-parity distractor retrieval without ultra's
other costs. That is the single most valuable line in this file for MLX.

CUDA keeps `rotated_pool=True` in low/mid/high and sets it **False in `ultra`**:
the cost is ~24% of decode and +1.1 GB, `mid` is the default preset, and ordinary
needle recall is 9/9 either way -- so nothing is lost by default and only
confusable-content retrieval gains. Decide the same way for MLX, on your own
decode budget.

**For MLX:** find the equivalent of `DKV_ROTATED_POOL`, run `linkbench` on both
settings with at least 24 seeds, and confirm the needle sweep is clean before
adopting. Do not judge it on multifact.

### Routing has ZERO headroom here — do not spend time on a smarter router

Re-tested at this operating point, where it actually binds (32k, ~29 blocks,
K=16 routed), with `DKV_TOPK_BLOCKS=0` so the model attends EVERY block:

| routing | 48 seeds |
|---|---|
| K=16 (default) | 47/48 |
| **every block** | **47/48** |

Identical. Showing the model the entire context changes nothing, so the router is
already selecting correctly and the one remaining failure is not a selection
failure. This confirms, at the new and much better operating point, the earlier
finding that `DKV_TOPK_FRAC` 0.0/0.5/1.0 were indistinguishable.

### Confirmed again on generated PROSE, not just on retrieval scores

The objection to the above is fair: linkbench scores one extracted fact, so it
cannot see whether routing changes the quality of an actual answer.
`colab/qual_routing_cuda.py` tests exactly that — a 16k structured handbook (five
departments, each with a budget, headcount, policy and a dependency on another),
and a four-part question needing facts from four sections plus a dependency chain.

**K=16 and attend-every-block produced BYTE-IDENTICAL answers.** Not similar —
identical. Both got the budget comparison, the headcount total (416) and the full
dependency chain including its loop; both missed the same fourth part. Dense also
got parts 1-3 and missed part 4 differently. So routing changed nothing about the
generated response either.

**One honest caveat on the ceiling argument.** "Attend everything" is not a strict
upper bound on quality — attending more CAN dilute attention, which is exactly
what killed dual-scale (item 10d), so a cleverer *subset* could in principle beat
both. What the data rules out is the assumption that the router is currently
*missing* something: it is not, because showing it everything changes nothing.

**The consequence for design:** distractor retrieval was never a *selection*
problem, it was a *representation* problem — which is exactly why the rotated/
unrotated fix moved it 40 → 47 while every routing knob ever tried moved it
nothing. Graph-structured routing, learned routing, multi-scale routing: none of
them can beat "attend everything", and attending everything is already measured
at 47/48. Spend effort on what is stored, not on what is selected.

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

**⚠ Partly superseded by item 10.** The ladder is right and now rests on realised
per-block rank rather than on the synthesis scores quoted below, which are
single-seed. Read item 10 first.

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

## 8b. The `ultra` preset — what CUDA ships, and what each setting is worth

**MLX status: the settings are portable; the JUSTIFICATIONS are what matter.**

CUDA added a fourth preset above `high`. Recording it in full because two of its
four defining choices are things MLX should copy and two are things MLX should
NOT copy, and the difference is only visible with the measurements attached.

| setting | low | mid | high | **ultra** |
|---|---|---|---|---|
| `svd_energy` | 0.999 | 0.9999 | 0.99999 | **0.999999** |
| realised mean per-block rank | 35 | 53 | 67 | **94** |
| `rank` (ceiling) | 32 | 64 | 128 | **224** |
| `rotated_pool` | True | True | True | **False** |
| `prefill_chunk_size` | 1024 | 1024 | 2048 | 1024 |
| `max_residual_tokens` | 40 | 128 | 128 | 128 |

**Take `rotated_pool=False`** — item 5. This is the one setting that measurably
closes a gap to dense on a metric with real statistical power: linkbench 40/48 →
47/48 over 48 seeds, exactly matching dense's 47/48. It costs 18-24% of decode
and +1.1 GB, which is why it lives here and not in the defaults.

**Do NOT take a high energy rung or a raised rank into `ultra`.** CUDA shipped
both there and then removed them, because measured against the version without
them they bought nothing and cost a lot. Qwen3.5-2B at 32k, interleaved:

| ultra variant | decode | device VRAM |
|---|---|---|
| rank 224 + energy 0.999999 | 8.16 / 8.23 tok/s | 9.22 GB |
| **mid settings + unrotated pool** | **10.05 / 10.07 tok/s** | **6.28 GB** |

22% of decode and 2.9 GB, for NO difference on anything measurable — linkbench
identical, needle sweep clean either way, synthesis unable to resolve it at all.
`ultra` is now **mid + `rotated_pool=False`, and nothing else.**

Two reasons those knobs failed to earn their place, both worth knowing before
repeating them on MLX: the rank choice came from a sweep that was randomised-SVD
projection noise (item 10), and the energy rung is nearly inert on real prose
because the rank CEILING binds there rather than the energy target (also item 10).
Neither is a general statement that fidelity does not matter — it is that these
two dials, at these settings, on this workload, do not move anything a benchmark
can see.

**Do NOT take `rank=224` as meaningful on its own** — item 10. It was originally
chosen from a rank sweep that turned out to be randomised-SVD projection noise.

**Do NOT assume `ultra` wins on synthesis.** It does not: over three seeds it
means 48.9 against `mid`'s 53.3 and dense's 60.0. The preset is justified on
linkbench and on realised rank, both of which can be measured; multifact cannot
resolve any of it.

`ultra` is otherwise a copy of `mid`, not of `high` — that is deliberate and is
the configuration everything above was measured on.

---

## 8c. Determinism: greedy decode is NOT reproducible at long context

**Priority: read before running ANY A/B on either runtime.**

**MLX status: LIKELY SHARED** — the cause is a reduction inside the decode
attention, not anything CUDA-specific.

At 32k, greedy decoding produced a small set of recurring distinct outputs across
runs; 16k was byte-stable. Everything upstream was eliminated by measurement:

* compression is deterministic — post-prefill state is byte-identical run to run
  (same block-state counts, same realised-rank sum), and disabling async SVD
  changes nothing
* background tiering, prefetch and eviction: disabling them changes nothing
* cuBLAS workspace config: changes nothing

The cause is the SDPA inside the re-materialisation path, whose reduction splits
differently once the concatenated key set is large (~33k rows at 32k against ~15k
at 16k). CUDA's fix is `DKV_DETERMINISTIC=1`, which forces the math backend:
byte-stable over 10 runs, at ~8% of decode.

**It is OPT-IN, not default**, because at 32k it is paid with nothing to offset
it. **Turn it on for any run you will compare.** Roughly fifteen 32k comparisons
in this project were made without it and were not measuring what they appeared
to — that produced one full retraction.

**Do not accept a cheaper backend without testing it properly.**
`EFFICIENT_ATTENTION` was tried as a faster deterministic option: 3 runs agreed,
8 runs did not (6/8). Three-run agreement is not evidence.

---

## 8d. A speed flag needs a speed measurement — two that shipped without one

Both of these were gated on *correctness* checks, passed them, and shipped. Neither
was ever timed. Both cost real throughput, and the second one cost it in a regime
where it could not possibly help.

**`DKV_GRAPH_SAFE_DECODE` — measured at last, −7.1%.** It removes device→host
syncs so the forward can be captured. On the bypass path that machinery (a
`StaticCache`, a full-buffer mask, a position-derived `cache_position`) is what
makes capture possible at all. But it *also* forced `changed = True` on the routed
path every step, which discards the gather cache on every token. Paired harness,
Qwen3.5-2B at 32k, 10 rounds:

    relaxed  80.28 ms/tok      forced  86.32 ms/tok
    paired mean_diff -6.148 ms, 95% CI [-7.015, -5.281]  ->  A faster by 7.1%

10/10 rounds the same sign. Fixed by splitting the flag: the bypass sites keep the
broad one, the routed sites take `_GRAPH_SAFE_ROUTED = safe AND mutation_out`,
which is only true when a graph is genuinely going to be captured over the routed
forward. **If you port a "make it capturable" flag, check what it costs on the
paths that are not being captured.**

**Mutation-out is now PER SESSION, which is what made `--fastdc` safe.** Deferring
the forward's mutation costs in proportion to attended-layer count. It buys a large
win when a graph is captured (16k on Qwen2.5-1.5B: 17.3 → 10.2 s wall,
byte-identical) and buys *nothing* when the selectivity gate declines one — where
it was a ~9% loss on wide models at 32k. That asymmetry is why the flag could not
simply be switched on for everybody.

The fix is to ask the question the gate already answers — *will a graph engage for
this session?* — and honour the request only when the answer is yes. Two
consequences worth copying:

* **Re-evaluate per step, not once per session.** Blocks keep getting compressed
  during decode, so a session can start at 15 blocks with K=16 and cross the line
  mid-generation. Cached, it keeps deferring after the graph has been declined.
* **The gate must be cheap enough to sit on the hot path, and that needs a
  number.** Measured 4.60 µs/call against an ~80 ms token — 0.006%, against the
  ~9% it avoids.

Verified at 32k with `DKV_DETERMINISTIC=1`: eager and gated-`--fastdc` both give
`7c291f42ece7d897`, with the gate logging that it disabled mutation-out. The
general lesson is portable even though CUDA graphs are not: **a knob whose payoff
depends on a regime should be gated on that regime, not on the user remembering.**

---

## 8e. The routed graph now captures — the blocker was a DEFAULT MISMATCH

For most of this project the routed CUDA graph never captured in the wrapper
decode path. The refusal said *"no decode cache supplied"*, which reads like
missing machinery and sent two investigations looking for one. It was not.

`DKV_GRAPH_SAFE_ROUTING` was read in two places with **two different defaults**:

    dkv_attention.py       default "1"   (fixed-shape routing is ON)
    static_decode_graph.py default "0"   (…so capture refuses)

The variable is not in `BEST_DECODE_DEFAULTS`, so it is normally unset and the two
modules simply disagreed: fixed-shape routing was on, and the capture that needs
it declined anyway. **If one flag is read in two modules, read it in one place.**

Fixing the default was not the right fix either, because fixed-shape routing is
the wrong question. Capture wants a cache so it can roll back its warmup writes;
what actually makes a cache unnecessary is that **the forward does not write** —
which is exactly what mutation-out provides, and nothing else does. So the gate is
now `_MUTATION_OUT_ACTIVE`. Fixed-shape routing keeps shapes static, which capture
also needs, but says nothing about whether warmup mutates state.

**Result: routed capture works and is byte-exact.** Qwen2.5-1.5B, every decode step
replaying, `DKV_DETERMINISTIC=1` on both arms:

    16k  eager and replay both 84b3e292c52be1bb   (and b5054d61 at 8 tokens)
    32k  eager and gated-decline both 7c291f42ece7d897

**On the speed, be careful — it does not reproduce the number in the old notes.**
This file previously recorded 1.48x / "17.3 → 10.2 s". Measured properly, with
prefill excluded and over the tokens actually generated, the gain at 16k is about
**5%**, and the run-to-run spread of the method is larger than that:

    fastdc off  86.06, 84.33 ms/token        fastdc on  85.05, 76.39

So the honest statement is "byte-exact and not slower, with a small gain this
method cannot resolve", not a speedup figure. `bench_decode_paired.py` cannot
settle it either, because `_FAST_DECODE` is read at import and the wrapper
republishes the effective flag every step.

### Two bugs found while verifying this, both worth avoiding in the port

**Do not fold a per-session flag into a module constant.** `_GRAPH_SAFE_ROUTED`
was computed once at import from the *requested* mutation-out flag, so under
`--fastdc` it stayed true on sessions where the gate had turned mutation-out off.
Those sessions took the relaxed branch anyway — gather cache cleared every step,
SRL recent-key trail skipped — which changes routing and therefore the text. It
showed up as 32k `--fastdc` returning `9a9cbc07` against eager's `7c291f42` with
determinism on. The routed sites must read the value the wrapper rebinds.

**"Not selective" and "not engaged" are different questions.** The gate first
asked only whether routing was selective, and answered "no" both when DKV was
engaged with few blocks *and* when DKV was not engaged at all. The second case is
a short prompt running the ordinary dense forward, which mutates its cache
normally — so claiming the forward was read-only let capture skip a rollback it
genuinely needed. Gates went to **3/8** with output degenerating to
`29dfulfulfulful`. Three cases, not two:

    ncomp is None    state unreadable        -> off, conservatively
    ncomp == 0       DKV not engaged         -> off
    0 < ncomp <= K   engaged, non-selective  -> ON, the win case
    ncomp > K        engaged, selective      -> off, frozen set would drift

Restored 8/8. **A boolean that answers two different questions with the same
`False` will eventually be asked the one it gets wrong.**

---

## 9. Measure VRAM against the POOL, and check for slots nothing fills

**Priority: high if you are trying to show a memory win — this is where CUDA's
went.**

**MLX status: CHECK THIS.** MLX allocates one array per layer for `comp_U`,
`comp_U_scale`, `comp_VK` etc. (item 3). Check whether it also carries
stratified-U / fact-anchor equivalents, and whether anything writes them.

**What CUDA found.** Attributing decode-time VRAM by owner on Qwen3.5-2B at 32k:

| owner | DKV | dense |
|---|---|---|
| model weights | 3.764 GB | 3.764 GB |
| **KV store** (pool vs HF cache) | **0.114 GB** | 0.433 GB |
| activations | 0.293 GB | 0.009 GB |
| fragmentation (reserved − allocated) | 0.417 GB | 2.163 GB |
| CUDA context | 0.261 GB | 0.219 GB |
| **device total** | **4.96 GB** | **6.59 GB** |

Two things only this breakdown shows:

* **The compression works and the total hides it.** The pool is **0.26x** the KV
  it replaces — a 3.8x saving — but it is 114 MB of a 4.96 GB footprint, so it
  moves the total by 2%. Weights are 76% of DKV's total and no KV scheme touches
  them. Report the KV-side ratio.
* **31% of CUDA's pool was slots nothing ever wrote.** Stratified-U and fact
  anchors are filled only by the CPU compress path; on CUDA the GPU path owns
  compression, so they sat all-zero at 52 MB — and were still handed to the
  decode kernel every token, which looped three dead fact slots per block per
  layer. Dropping them: pool 166 -> 114 MB, and decode got slightly FASTER
  (18.20 vs 16.90 and 17.42 vs 16.98 tok/s, interleaved arms, change ahead in
  both rounds).

Note dense's 2.16 GB of fragmentation against DKV's 0.42 — that is
`expandable_segments`, which DKV sets and the plain HF path does not. Most of
DKV's apparent total-memory win on this model is that, not compression. Be
honest about which is which; MLX's unified memory has no equivalent.

---

## 10. RETRACTED: the rank sweep was randomised-SVD noise. Energy is the dial.

**Priority: read this BEFORE trusting any multifact number in this file, mine or
your own.**

An earlier revision of this item claimed rank 224 beat the dense synthesis
control (63.3 vs 60.0) and that `svd_energy` did nothing. **Both halves are
withdrawn.** What actually happened:

**`rank` is a CEILING that almost never binds.** The compressor keeps the
smallest k reaching `svd_energy`, then clamps to `rank`. Instrumented on
Qwen3.5-2B at 16k, the realised per-block rank at configured 216 / 224 / 232 is
52-137 with **mean ~67 in all three cases — the ceiling binds for 0.0% of
blocks.** Configuring 216 versus 232 changes nothing about what is stored.

**Which of the two binds depends on the SPECTRAL RICHNESS OF THE INPUT.** On
repetitive filler, energy binds and the ceiling never does:

| energy | realised mean rank | | energy | realised mean rank |
|---|---|---|---|---|
| 0.999 | 35 | | 0.999999 | 94 |
| 0.9999 | 53 | | 0.9999999 | 180 |
| 0.99999 | 67 | | | |

**On REAL PROSE the opposite holds** — realised rank tracks the CEILING and barely
responds to energy (Random Features paper, 16k):

| rank ceiling | realised mean, energy 0.999 -> 0.999999 |
|---|---|
| 64 | 66.5 -> 66.7 |
| 128 | 130.3 -> 133.3 |
| 224 | 215.3 -> 233.3 |

A document's spectrum does not decay fast enough to meet the target under the
cap, so the cap binds. **On the workloads this system is for, `rank` is the dial
and `svd_energy` is nearly inert** — which is why an energy A/B on the paper
corpus gave BYTE-IDENTICAL text at 0.9999 / 0.99999 / 0.999999 even though
`pool.U` differed at every setting.

**Do not quote the filler table as if it described documents.** Measure realised
rank on YOUR corpus before choosing either knob.

**What a different `rank` DOES change is `r_proj = rank + 5`**, the width of the
randomised-SVD projection — so it redraws Omega and yields a different
approximate basis at the same realised rank. The "rank landscape" was that
redraw.

**The benchmark cannot resolve any of it.** Holding the config fixed at rank 224
and changing only `DKV_RSVD_SEED`:

| seed | 0 | 1 | 2 |
|---|---|---|---|
| synthesis | 63.3 | 33.3 | 50.0 |

A **30-point spread from the random draw alone** — the entire range the rank
sweep "discovered". Over three seeds:

| config | seeds | mean |
|---|---|---|
| mid (rank 64) | 50.0 / 56.7 / 53.3 | 53.3 |
| ultra (rank 224) | 63.3 / 33.3 / 50.0 | 48.9 |
| **dense** | seed-independent | **60.0** |

So rank 224 is **worse on average** than rank 64 and far less stable, and **DKV
does not beat dense on synthesis at either.**

**Method, and it applies to MLX too** (`DKV_SVD_SEED` there): temperature-0
replication is deterministic and proves nothing. The seed is the axis that must
be varied. Treat any multifact difference under ~15 points as no difference, and
never quote a single-seed number. MLX uses the same randomised SVD, so it has the
same noise floor.

**What to port instead:** the energy ladder, justified by realised rank — which
is deterministic and directly measurable — rather than by benchmark scores.
CUDA's presets now differ on `svd_energy` (0.999 / 0.9999 / 0.99999 / 0.999999),
with `rank` set only high enough not to clip the target.

---

## 10b. RETRACTED with it: the prefill_chunk_size score

The bisect finding stands as a MECHANISM and falls as a measurement.

Real and visible in the code: the wrapper rounds the prefill chunk UP to a
multiple of block capacity, so at `micro_block_size` 1024 (capacity 1025) a 1024
chunk becomes 1025 — exactly ONE block per chunk — and 2048 becomes 2050, two.
Worth checking whether MLX has the same rounding.

Not established: that this is worth 60.0 → 33.3 of synthesis. That is a
single-seed comparison sitting inside the ±15-point seed band from item 10. The
bisect that isolated `prefill_chunk_size` from the other four `high` settings has
the same problem. Re-run across seeds before believing the direction.

---

## 10c. Block size: the retrieval half survives, the synthesis half does not

Re-stated after the retraction. Of the sweep at rank 224:

| block | synthesis (single seed — **not** reliable) | linkbench @32k (24 seeds — reliable) |
|---|---|---|
| 1024 | 63.3 | 20/24 |
| 1536 | 33.3 | 22/24 |
| 2048 | 46.7 | **23/24** (= dense) |

**Linkbench is the trustworthy column** — it already averages 24 seeds per point,
and its ordering is monotone in block size, which matches the independent
block-count finding in item 5b. The synthesis column is one draw per point and
spans exactly the noise band; do not read a trade-off out of it.

So the honest statement is narrower than before: **larger blocks help distractor
retrieval, and whether they cost synthesis is unmeasured.** Dual-scale was
motivated by a trade-off whose synthesis half is not established — see 10d, where
it fails anyway.

---

## 10d. Dual-scale storage does NOT work — and the design note's structure is why

**Priority: highest, because this closes the item every earlier note pointed at
as the fix. Do not build it on MLX.**

Item 5b proposed dual-scale as the answer to the block-size trade, with this
structure: a second pool at the coarse scale, ingest writing both, and decode
"merging the two outputs by log-sum-exp". CUDA has now implemented it
(`DKV_DUAL_SCALE=1`, default off) and measured all three ways of combining the
scales. None works.

| policy | result |
|---|---|
| **union** (attend both scales; what the note proposed) | synthesis **63.3 → 33.3** |
| **extend** (coarse only where fine routing covered nothing) | 63.3, 20/24 — no change |
| **swap** (coarse replaces fine where it split an association) | linkbench 20/24 → **19/24** |

**The log-sum-exp merge is not a fix, it is the failure.** Merging two softmaxes
over disjoint key sets is arithmetically identical to one softmax over their
union — so the note's proposed structure is exactly the union row. Every token
then appears TWICE, as two different lossy reconstructions of itself; its
attention mass splits between them and the exact dense window is diluted in the
same proportion.

Isolated with `DKV_DUAL_SCALE_ATTEND=0` (keeps the coarse ingest and the widened
pool, skips attending the coarse rows): 63.3 either way. So the second
compression pass and the pool sizing are harmless, and it is combination at
attention that fails.

**Two structural things worth taking even though the feature is off:**

* **A second pool is not needed.** Blocks are already keyed by session id and
  block size is already per-session, so the coarse scale is a SHADOW SESSION and
  every existing path works on it unchanged. MLX has per-session state too.
* **Widening slots is cheap.** The note assumed sizing slots for the coarse scale
  inflates the whole pool. Only `U` scales with block width — `V_KV`, anchors and
  residuals are indexed by rank or by the residual budget — so it is one doubling
  of `U`, ~20 MB of a 114 MB pool.

**The conclusion is narrower and more useful than "dual-scale does not work":
combining two granularities AT ATTENTION cannot work, because attention over a
set of representations is not attention over a set of tokens.** Granularity has
to be chosen where the representation is BUILT. The block-size trade in 10c is
still open, and now has one fewer candidate solution.

---

## 11a. THE finding on speed: DKV is HOST-BOUND, not kernel-bound

**Priority: highest for anyone optimising either runtime. It reframes what to
work on, and it explains why five separate kernel-level attempts measured zero.**

**MLX status: MEASURE THIS FIRST on MLX.** Metal's launch overhead is much lower
than CUDA's, so MLX may not share it — but if it does, the same conclusion
follows and no amount of kernel tuning will help.

Both halves of CUDA's runtime spend roughly 40% of their time with the GPU IDLE,
waiting on Python:

| phase | GPU kernel time | wall | host-bound share |
|---|---|---|---|
| prefill (1.5B, 32k) | 4.52 s | 7.83 s | **~42%** |
| decode (2B, 16k) | 27.19 ms/tok | 44.7 ms/tok | **~39%** |

And within decode's GPU time, attention is a minority:

| | ms/token | share |
|---|---|---|
| attention kernels | 5.35 | 19.7% |
| everything else | 21.84 | 80.3% |

The single largest decode kernel is `cublasGemv` at **8.08 ms/token** — the
model's own weight reading, which decode is memory-bound on and which no
KV-compression scheme touches.

**What this explains.** Every kernel-level optimisation tried this session
measured zero or near-zero: removing 46% of prefill's GPU syncs, caching the
history rotary tables across layers, `enable_gqa`, chunk-size tuning, streaming
compression. That is not a coincidence — they were all aimed at a GPU that was
already idle 40% of the time. The binding constraint is host dispatch, not kernel
efficiency.

**The fix, and why CUDA cannot take it yet.** Host-bound decode is what CUDA
graphs exist for. CUDA's graph runner works and is bit-exact on the BYPASS path
(1.25x), but `config.cuda_graph` is hard-disabled because the routed path
captures mutable Python routing state and replays it stale. Fixing that means a
device-resident routing/session ABI — a redesign, not a tuning change, and by far
the highest-value speed work remaining.

**Two measurement lessons to carry over:**

* `DKV_TIME_ATTN` (MLX has the same style of timer) measures the attention path
  with host time around it, NOT attention GPU kernels. It reports 39.4 ms/token
  where attention kernels are 5.35. Quote it as an attention-path figure, never
  as a serving figure.
* Wall-clock throughput is ~12% below the reported tok/s at 16k. Harnesses in
  `colab/` now measure both: `decode_wall_vs_timer.py` and
  `decode_kernel_split.py`.

---

## 11. Prefill: what was tried and did NOT work

**Priority: read before spending time on TTFT, so you do not repeat this.**

CUDA prefill sits at ~7.9 s against dense's ~5.5 s on Qwen2.5-1.5B at 32k, with
4.5 s of that GPU kernel time. Everything below was measured and rejected:

* **Removing GPU syncs — no effect.** A sync probe
  (`torch.cuda.set_sync_debug_mode`) found **3858 syncs** in one prefill. Killing
  the two biggest sites took it to 2094 (−46%) and TTFT did not move
  (7.89/8.09 s vs 7.96/8.23 s). The prefill gap is not host stalls. Both fixes
  were kept anyway since they are strictly correct: `seq_lens[i] = 0` →
  `.zero_()` (assigning a Python scalar into a CUDA tensor synchronises), and
  `torch.tensor(list, device="cuda")` → a pinned staging ring.
* **`DKV_STREAMING_COMPRESS=1`, which is MLX's default — much worse on CUDA.**
  At 8k: TTFT 5.69 s vs 1.88 s, peak_alloc 7.07 vs 3.78 GB. **Do not port CUDA's
  OFF default back to MLX either** — the reason is CUDA-specific dispatch
  overhead, and MLX's low launch cost is exactly why it is on there.
* **Caching the history rotary tables across layers — no effect.** cos/sin depend
  only on position, and were being rebuilt in all 28 layers per chunk. Correct to
  cache, and cached now, but GPU time was unchanged, so the cost was never there.
* **`DKV_CONTIGUOUS_PREFILL=1`** alongside the fused history attention: 8.91 s and
  5.76 GB against 7.83 s and 4.81 GB.
* **Prefill chunk 2048 / 4096** against 1024: TTFT 8.21 and 9.18 s vs 7.96 s.

What the profile actually says, after the fused history attention landed: GEMM
60%, elementwise 20%, SVD ~12%, softmax 4%. The single largest kernel is the
fused attention itself at 25%. **DKV prefill does everything dense prefill does
and then compresses**, so parity is not reachable; the SVD alone is ~0.5 s.

---

## 11b. Every remaining single-seed synthesis claim: NO RESOLVABLE DIFFERENCE

**Priority: read before acting on any knob in this file. This is the clean-up
sweep after item 10's retraction.**

All four surviving single-seed findings re-run with `colab/synthesis_power.py`,
paired, 4 replicates, Qwen3.5-2B at 16k on `ultra`:

| knob | paired diff vs default | 95% CI | replicates a 5-pt effect needs |
|---|---|---|---|
| block 512 vs 1024 | +7.50 | [-13.9, +28.9] | **28** |
| block 2048 vs 1024 | +8.33 | [-4.3, +21.0] | 10 |
| block 512 vs 2048 | -0.83 | [-12.6, +10.9] | 9 |
| prefill_chunk 2048 vs 1024 | +5.83 | [-23.6, +35.3] | **53** |
| K=32 vs K=16 (at 32k) | +0.00 | [-4.3, +4.3] | 2 |
| svd_energy 0.9999 vs 0.999999 | +0.00 | [0.0, 0.0] | 2 |

**Not one is resolvable.** Every original claim -- block size trading synthesis
against retrieval, `prefill_chunk_size` being worth 27 points, K=32 lifting
synthesis, the energy ladder ordering -- came from a single sample and none
survives replication.

**Note the replicate column.** Block 512 vs 1024 would need **28** replicates and
prefill chunk **53**. Those effects, if they exist at all, are far smaller than
the single-seed numbers implied. Any future synthesis claim should quote this
column.

**What survives from 10c:** only the linkbench half. That column already averages
24-48 seeds per point and its ordering is monotone in block size. The synthesis
half is withdrawn -- **so the "block size trades retrieval against synthesis"
premise, which is what motivated dual-scale (10d), was never established.**
Dual-scale failed anyway, but it was chasing a trade-off that may not exist.

**One open oddity, flagged rather than explained.** The `svd_energy` arm returned
score vectors IDENTICAL to the default across all four replicates
(`[40.0, 66.67, 63.33, 50.0]`), where the block, chunk and K arms all produced
different vectors. The override does reach the compressor -- `_svd_energy_target()`
returns the overridden value -- and it demonstrably changes the realised
per-block rank (35/53/67/94 across the ladder, item 10). So either the generated
summary is genuinely insensitive to that much of the spectrum, or something
downstream of the rank selection is not consuming it. **Worth resolving before
the energy ladder is trusted as a quality dial**; it is currently justified on
realised rank, which is measured, not on scores.

---

## 12. Synthesis, measured properly: `ultra` reaches dense, `mid` does not

**Priority: highest accuracy result in this file, and the first synthesis
comparison in the project with a confidence interval attached.**

**MLX status: PORT THE HARNESS FIRST.** `colab/synthesis_power.py`. Nothing below
is obtainable without it.

The old harness had three defects, all fixed:

1. **One sample per config** - now R replicates.
2. **Dense had n=1 BY CONSTRUCTION.** The document window was fixed and dense has
   no randomised SVD, so dense could only ever emit one number and a DKV
   distribution had nothing to be compared against. Replicates now vary the
   DOCUMENT WINDOW as well as the SVD seed, giving both arms a real distribution.
3. **Unpaired** - every arm now walks the SAME replicate list, so the statistic is
   the per-replicate difference, whose variance is far below either arm's own.

Qwen3.5-2B at 16k, 4 paired replicates:

| arm | mean | sd | 95% CI |
|---|---|---|---|
| dense | 61.7 | 1.9 | [59.8, 63.6] |
| **DKV `ultra`** | **63.3** | 4.7 | [58.7, 68.0] |
| DKV `mid` | 45.0 | 6.4 | [38.7, 51.3] |

| paired | diff | 95% CI | verdict |
|---|---|---|---|
| ultra - dense | **+1.67** | [-7.5, +10.9] | **no difference resolvable -> parity** |
| ultra - mid | +18.3 | [+3.0, +33.6] | ultra genuinely ahead |
| mid - dense | -16.7 | [-28.1, -5.2] | mid genuinely behind |

**So `ultra` reaches dense on synthesis and `mid` does not.** This is not the
retracted claim returning: that one was a single seed on a fixed window, this is
replicated, paired and interval-bounded.

Two things worth carrying:

* **Dense's sd is 1.9 against DKV's 4.7-6.4.** DKV matches it on the mean while
  remaining a higher-variance system, so a single reading of either can mislead.
* **The fixed window the old harness used was EASIER than average** - `mid`
  scores 53.3 on it against 45.0 over varied windows. A harness with one fixed
  document is partly measuring that document.

**Power:** the tool prints the replicates a target effect needs. At the observed
spread, 5 points needs ~6 replicates; 4 suffices for the ~17-point gaps above but
not for ultra-vs-dense.

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
* **CUDA graphs** generally — bypass path works (1.25x, bit-exact). The routed
  path captures and is byte-exact where routing is non-selective (a small gain,
  see 8d. Nothing here is portable — Metal has no capture/replay constraint — but
  8d's *gating* idea is, because it is really about not paying a cost that only
  pays off in one regime.

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
