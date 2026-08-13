# Handoff: dual-scale block storage

Written 2026-08-13. Everything below is measured on this machine (RTX 4070 SUPER,
12 GB) unless marked otherwise. Start here rather than re-deriving.

## Why this work exists

Two accuracy benchmarks want opposite block sizes and nothing bridges them.
Qwen3.5-2B at 16k, `linkbench` = 24 seeds of `colab/linkbench_cuda.py`:

| block | linkbench | multifact synthesis | TTFT (1.5B @32k) | pool |
|---|---|---|---|---|
| 256 | 14/24 | 46.7 (8 facts, 2 links) | 15.17 s | 311.6 MB |
| 512 | 15/24 | **50.0** (6, 3) | 11.58 s | — |
| **1024 (current default)** | **24/24** | 30.0 (6, 1) | 11.43 s | **105.3 MB** |
| 1536 | — | 26.7 (5, 1) — fails the >=30 bar | — | — |
| 2048 | 24/24 (at 32k) | 33.3 (4, 2) | — | — |
| dense | 24/24 | 60.0 (9, 3) | 5.70 s | 393.9 MB |

Big blocks keep an association inside ONE unit, which distractor-heavy retrieval
needs. Small blocks give routing the granularity to assemble diverse content,
which synthesis needs. The current default (1024) buys linkage at dense parity
and pays for it in synthesis.

## What is already ruled out — do not re-test these

Every one of these leaves linkbench at **exactly** 14/24 at block 256:

* routing coverage — `DKV_TOPK_FRAC` 0.0 / 0.5 / 1.0, and `DKV_TOPK_BLOCKS=0`
  (attend EVERY block). Letting the model see everything does not help.
* residual budget — 32 / 128 / 224 of 256 tokens exact.
* recency window — 512 / 4096.
* SVD rank — 32 / 96. And rank scaled WITH block size (1024 at rank 32/64/128)
  leaves synthesis at 30.0 in all three, so the synthesis loss is not lost
  per-token fidelity either.

Two more dead ends, both with the measurement that killed them:

* **`DKV_ROTATED_POOL=0`** closes linkbench (24/24 at 8k/16k/32k) but breaks
  needle ORDER: every failure is edit-distance-1 with transposed digits
  (`Falcon-94276-6183`, `Falcon-9427-6138`), including at 2k where nothing is
  compressed. Raising residuals to 224 does not repair it. The unrotated path
  rotates the anchor and the whole V_K basis at the ANCHOR's position, so every
  token in a block shares one rotation and carries a phase error of up to a full
  block.
* **Per-token RoPE** is NOT cheap, contrary to an earlier note. The kernel
  computes the query projection once per chunk
  (`triton_fused_decode.py:567`, "Q projection -- computed once for this entire
  chunk") and the token loop only does a RANK-dim dot against it. RoPE acts in
  D-space while that projection is D -> rank, so per-token rotation means a
  D x rank projection per TOKEN instead of per block: S x more work in the hottest
  loop. Rotating the key instead needs V_K rotated per token, which is worse.

## The design to build

**One pool, not two.** Set the pool's `max_seq_len` to the COARSE size and store
both fine and coarse blocks in it, distinguished by their `seq_len`.

Why this beats the two-pool version:

* the decode path needs NO changes — `_gather_routed_blocks_for_kernel` is
  already parameterised by pool, and the kernel already masks per block by
  `blk_sz`, so a coarse block is just a block with a larger extent
* no second kernel launch, no log-sum-exp merge between pools
* routing already scores whatever blocks it is given

**Memory.** ~460 KB/slot at `max_seq_len=2048` (residuals are a FIXED 128 tokens
per block regardless of block size, so they do not scale), times roughly 480
slots for fine+coarse at 32k on Qwen3.5-2B = ~221 MB against today's 105 MB.
That is ~+116 MB on a ~5 GB footprint, about 2%. An earlier concern that
dual-scale would give back the VRAM win was wrong by an order of magnitude.

**Where to hook ingest.** `streaming_sparse_ingest.py:_submit_blocks`, around the
`_by_T` grouping. At that point `blocks_list` still holds raw `active_k`, which is
what a second scale has to be built from — after compression it is freed. Group
consecutive fine blocks, concatenate their raw K/V, compress that into coarse
blocks, and register them.

**The risky part, and it is the only risky part.** Block creation and
registration: `session_blocks`, per-layer metadata, and the
`anchor_idx` / `token_indices` / `pool_idx` bookkeeping that routing reads. This
sits directly under the accuracy gates and is where a mistake corrupts stored KV
rather than merely misreporting a number. Budget the validation, not just the
edit.

## Gates that must pass

    validate_cuda_dkv.py --long --model Qwen/Qwen2.5-1.5B-Instruct   9/9 + 9/9
    validate_cuda_dkv.py --long --model Qwen/Qwen3.5-2B              9/9 + 9/9
    multifact_eval_cuda.py --model Qwen/Qwen3.5-2B --ctx 16384       9/9
    linkbench_cuda.py QMODE=direct SEEDS=$(seq -s, 31 54) CTX=16000  24/24
    pytest ACTIVE_RUNTIME/tests                                      147 passed

Target for dual-scale: keep linkbench at 24/24 AND lift synthesis from 30.0
towards the 50.0 that block 512 reaches, at roughly +116 MB.

## Measurement discipline (this cost real time to learn)

* Decode throughput moves ~20% run to run. Use `colab/bench_decode_paired.py`
  (paired, interleaved, one process, minimum-per-round). It resolves ~1.5% and it
  overturned three claims made from unpaired runs.
* Do NOT size a change from cProfile self-time. It charges its own per-call
  overhead to the callee, so cheap high-count builtins look expensive: cutting one
  from 229.8 to 56.4 calls per token produced no measurable change.
* Check that a benchmark actually reads the knob you are sweeping.
  `multifact_eval_cuda` silently ignored `BLOCK` and returned three identical
  scores for three different block sizes, which nearly produced a misattributed
  regression report.
* Verify the GPU is idle before benchmarking. A stray run from an earlier command
  invalidated several measurements before it was noticed.
