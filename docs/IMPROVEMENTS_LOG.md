# DKV improvements log

Every change to DKV's store, decode or prefill made in the improvement batches
is recorded here: what it is, the switch that controls it, what each test tier
measured, and when (and why) its default changed. Newest entries on top within
each batch.

Tiers: **T1** offline screen on captured KV (`benchmarks/tier1_screen.py`),
**T2** real decode on 6 fixed prompts (`benchmarks/decode_fidelity.py`),
**T3** batch confirmation. Guardrails a change must not trade away: store bytes,
reach (peak memory), decode speed, query-agnosticism, exact prefill = dense.

## Batch A: store quality (exact and streaming)

| Date | Change | Switch | Default | T1 | T2 | T3 | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-03 | Separate K/V factors, block-diagonal factor | `DKV_KV_SPLIT=rK,rV` | off | 1.29-1.52x error (worse) | - | - | Dropped: K and V share token structure; joint factor wins at equal bytes |
| 2026-10-03 | Channel normalization before the SVD | `DKV_CHANNEL_NORM=1` | off | 1.05x (worse) | - | - | Dropped |
| 2026-10-03 | Equalize U columns before int8 | `DKV_U_COLSCALE=1` | off | 1.00x | - | - | No gain alone; used inside 4-bit U |
| 2026-10-03 | U on a 4-bit grid, per-column scale | `DKV_U_BITS=4` | off | with 8-bit window: -31% bytes at 1.04x | in win-att: mean KL 0.176 vs base 0.157 (better on 4 of 6 prompts) | - | Quality only: pool still stores int8 until packing is implemented |
| 2026-10-03 | Rank 32 -> 64 | `DKV_RANK=64 DKV_RSVD_MAX_RPROJ=64` | off | 0.77x, +14% bytes | see win-att | - | Check compression time (batched solver limit 32) |
| 2026-10-03 | V gain off in the joint SVD | `DKV_V_SCALE=0` | off (gain on) | 0.95x, attention KL -20% | win vs win-vg0: no difference | - | Existing switch |
| 2026-10-03 | Residuals ranked by error x attention received (256 sampled prompt queries/layer, per-block normalized) | `DKV_RESID_ATTN=1` | off | 0.69x at equal bytes | MIXED: RULER KL -55..-60%, but 3 LongBench steps spike (KL 6-12); mean KL 0.560 vs 0.157. Fired on 2,760 blocks | - | Query-agnostic; architecture-generic (finds the rotary module by role); engagement counter `_qres_used` |
| 2026-10-03 | Combined: 4-bit U + 8-bit window + rank 64 + V gain off + resid-attn | (all of the above) | off | **0.57x error, -18% bytes** | mean KL 0.460 vs 0.157 (driven by the resid-attn spikes); decode speed unchanged, peak +0.26 GB | - | 8-bit window not yet wired in the runtime |

**T2 note (2026-10-03):** 6 prompts / 65 forced steps is underpowered: per-step KL is
heavy-tailed (medians ~0.0001 for every arm) and the means are set by 2-4 spikes.

**T2 enlarged (2026-10-04):** 24 prompts (12 LongBench >=8k, 12 RULER 16k), 374 paired
steps, granite. Mean KL base 0.314; paired dKL [95% CI]:
- resid-attn: +0.044 [-0.077, +0.175], top-1 +1.1% -> **no effect; parked (stays off)**
- 4-bit U + rank 64 + V gain off: -0.049 [-0.120, +0.005], large-KL steps 31 vs 35 ->
  **keep**: at least as good, and -18% store once U is packed (pool still int8: +14% today)
- all combined: -0.032 [-0.150, +0.087]
Reading: 91% of decode steps already match dense (KL < 0.5); the gap lives in ~9% of
steps, which batch A does not move. Next: pack 4-bit U + 8-bit window (realize bytes),
then batch B (per-token low-bit residual) aimed at those steps.

## Batch B: exactness (tier 1, 2026-10-04, granite, 6 prompts x 6 layers)

Lesson first: **attention KL, not output error, predicts real decode.** KIVI-4 has a
larger output error than DKV (0.096 vs 0.076) yet matches dense in real decode; its
attention KL is 30x lower (0.003 vs 0.094). resid-attn improved output error but not
attention KL, and did nothing in tier 2. From here, T1 ranks by attention KL.

| Change | T1 out_err | T1 attn KL | B/tok (cmp) | Decision |
|---|---|---|---|---|
| shipped DKV | 0.076 | 0.094 | 962 (4.3x) | reference |
| KIVI-4 (reference arm) | 0.096 | 0.003 | 1215 (3.4x) | reference |
| B1 2-bit per-token residual on all rows | 0.045 | 0.015 | 1510 (2.7x) | works, too many bytes |
| B1 3-bit / 4-bit per-token residual | 0.016 / 0.007 | 0.003 / 0.0006 | 1729 / 1948 | too many bytes |
| B1 2-bit + 4-bit U + 8-bit window + rank 64 + V gain off | 0.035 | 0.008 | 1335 (3.1x) | superseded by the hybrid |
| B3 block-adaptive residual budget | 0.074 | 0.093 | 962 | no gain; dropped |
| **Hybrid: keys 4-bit per channel (groups of 32 tokens), values low-rank r32 + 64 residuals, 4-bit U, 8-bit window** | **0.077** | **0.003** | **980 (4.2x)** | **lead candidate** |
| Hybrid, value rank 16 + 32 residuals | 0.096 | 0.003 | 917 (4.5x) | KIVI-4 quality, 25% fewer bytes |
| Hybrid, keys 3-bit | 0.105 | 0.012 | 871 (4.7x) | keys need 4 bits |
| Reverse hybrid: low-rank keys, 4-bit values | 0.131 | 0.092 | 898 | confirms keys are the weak side |

Reading: keys have outlier channels and their errors are amplified by the softmax, so
they want per-channel quantization; values are averaged by attention and compress well
in low rank. Being verified in real decode by a probe (`DKV_PROBE_EXACT_SIDE=K|V`)
before the runtime/kernel change for quantized keys is made.

### Batch B tier 2 (2026-10-04, 24 prompts, 374 paired steps, granite)

| Arm | Switch | dKL vs base [95% CI] | KL>0.5 (base 35) | top-1 | Reading |
|---|---|---|---|---|---|
| probe: exact keys | `DKV_PROBE_EXACT_SIDE=K` | **-0.145 [-0.278, -0.040]** | 26 | +0.5% | keys limit real decode |
| probe: exact values | `DKV_PROBE_EXACT_SIDE=V` | -0.023 [-0.144, +0.085] | 35 | +1.3% | values do not |
| **hybrid: per-channel 4-bit keys + low-rank values + 64 V residuals** | `DKV_KEY_QUANT=pc4` | **-0.112 [-0.228, -0.021]** | 32 | +1.9% | **significant; ~77% of the exact-key gain** |
| hybrid + 4-bit U | `+ DKV_U_BITS=4` | -0.106 [-0.223, -0.015] | 34 | +1.9% | same, smaller |
| hybrid, streaming | `+ DKV_STREAMING_COMPRESS=1` | +0.082 [-0.059, +0.223] | 59 | -1.6% | needs the streaming baseline on this set (running) |

Peak memory and decode speed in these arms are NOT the design's: keys are written as
16-bit residuals (fake 4-bit) in 1,024-slot arrays, so the store is oversized and spills.

**Design for real packed keys (not yet built; needs review):**
- Pool: per-channel 4-bit key codes per slot `[n, T, H_kv, D/2]` uint8 + scale/zero per
  32-token group `[n, T/32, H_kv, D]` fp16 (~640 KB per 1,024-token block vs 2 MB fp16).
- The value factor only (V half of `V_KV`); no K half, no K residuals.
- Decode: reconstruct keys from codes in the main loop of the fused kernel and the remat
  path (NOT via residual substitution: kernels assume equal K/V residual slot counts, and
  a 1,024-slot residual loop per block is slow).
- Router: scores blocks from the anchor + residual keys today; with quantized keys it can
  score a few key landmarks per block instead (ties into batch C3).
- Streaming history attention: same key reconstruction.

### Batch B tier 2, design parameters (2026-10-04)

| Arm | dKL vs base [95% CI] | Reading |
|---|---|---|
| base, streaming | +0.213 [+0.113, +0.331] (KL 0.527) | streaming's cost today |
| hybrid, streaming (previous run) | KL 0.395 | 25% better than base streaming |
| hybrid, no value residuals (`DKV_KEY_QUANT_VRES=0`) | -0.015 (n.s.) | loses most of the gain: keep 64 |
| hybrid, value rank 16 | +0.069 (n.s.) | worse: keep rank 32 |
| hybrid, rank 16, no value residuals | +0.259 [+0.148, +0.378] | dropped |

**Settled hybrid design:** per-channel 4-bit keys (groups of 32 tokens), values low-rank
rank 32 + 64 value residuals. ~Same store as today; KL -36% exact, -25% streaming.

## Batch C: routing (tier 1, 2026-10-04, Qwen3.5-4B, 5 RULER prompts x 4 attention layers)

Attention mass the routed blocks keep / attention-output error:

| Scheme | 32k mass / err | 64k mass / err | blocks at 64k |
|---|---|---|---|
| all blocks | 1.00 / 0.079 | 1.00 / 0.094 | 61 |
| top-16 (shipped) | 0.93 / 0.096 | **0.80 / 0.165** | 16 |
| top-32 | 1.00 / 0.079 | 0.89 / 0.126 | 32 |
| top-p 0.99 | 0.96 / 0.091 | 0.94 / 0.109 | 42 |
| per-head top-16 | 0.94 / 0.091 | 0.83 / 0.145 | 16 |
| oracle 16 | 0.95 / 0.090 | 0.85 / 0.143 | 16 |
| landmarks (16 per block) | = top-16 | = top-16 | 16 |

Reading: at 64k routing drops 20% of attention mass and doubles the error of compression
alone; even an oracle 16-block choice drops 15%, so attention is spread and the fix is
MORE blocks (cost: decode speed), not a cleverer 16. Per-head helps slightly; landmarks
add nothing. Tier 2 (real decode, Qwen 32k/64k: all / top-32 / half the blocks) running.

### Batch C tier 2 (2026-10-04, Qwen3.5-4B exact, 12 RULER prompts at 32k + 64k, 213 steps)

| Arm | dKL vs top-16 [95% CI] | decode tok/s |
|---|---|---|
| all blocks (`DKV_TOPK_BLOCKS=0`) | -0.010 [-0.028, +0.005] n.s. | 8.3 (-16%) |
| top-32 | -0.007 n.s. | 9.2 |
| half the blocks (`DKV_TOPK_FRAC=0.5`) | -0.003 n.s. | 10.0 |
| top-16 (shipped) | - | 9.9 |

**Decision: keep top-16.** Routing is not a bottleneck in real decode on Qwen at 32k-64k,
despite tier 1's 20% dropped mass (only 8 of 32 layers attend; recurrent layers carry
the rest). Second time tier 1's single-layer metric overstated an effect: tier 2 decides.
Still to check: granite (all 40 layers attend) beyond 16k.

### Batch C on granite (2026-10-04, hybrid streaming, 8 RULER prompts at 24k-32k, 140 steps)

| Arm | KL | dKL vs top-16 [95% CI] | dec tok/s |
|---|---|---|---|
| top-16 (shipped) | 0.074 | - | 5.5 |
| all blocks | 0.074 | +0.0001 [-0.009, +0.008] | 2.4 |
| top-32 (= all at 32k) | 0.074 | identical | 2.2 |

**Decision: keep top-16; batch C closed on both architectures** (Qwen hybrid-attention
and granite all-attention). The hybrid in streaming mode at 24-32k reads KL 0.074.

## Batch D (2026-10-04)

**D3 granite decode defect -- already resolved.** Root cause (wrong attention scale in the
decode kernels) was fixed 2026-09-02; the remaining open item then (NaN on the
project-then-attend path) is not observed: today's streaming arms take that path and
return finite, sensible logits on all 374 tier-2 steps. No change.

**D1 memory-sized remat cache for the DEFAULT exact mode** -- `DKV_REMAT_GATE=1` (off by
default; the hybrid already uses the gate). Short ladder, granite exact, 128-token answers:

| rung | no gate (alloc / reserved, status) | gate |
|---|---|---|
| 16,384 | 11.49 / 11.93 ok, 2.31 s/1k | 11.05 / 11.39 ok, 2.61 s/1k |
| 20,480 | 11.69 / 12.46 **spilled** | 11.28 / 11.95 **ok** |
| 24,576 | - | 11.37 / 12.15 spilled (exact prefill holds dense KV) |

Reading: the gate lifts granite's exact-mode ceiling 16k -> 20k (+25%, = preallocated
dense) by declining to cache layers when memory is short; cost is per-step rebuilds for
those layers (16k decode ~13% slower); nothing changes when memory is ample. Candidate
for default-on after the paper ladders.

**D2 elastic streaming window** -- `DKV_STREAM_ELASTIC=1` (off by default;
streaming_sparse_ingest.py `_elastic_hold`). In streaming mode, blocks that left the
recency window stay EXACT while allocated memory + 2 GB headroom
(`DKV_STREAM_ELASTIC_RESERVE_GB`) is under 94% of the card; once it is not, the oldest are
compressed, at most 4 per layer per chunk (`DKV_STREAM_ELASTIC_BATCH`). The prefill
boundary compresses everything as before (`compress_deferred_blocks(final=True)`), so
the decode store and its bytes are unchanged; only the prefill sees exact history.

- v1 gated only the per-layer compress path. Streaming also runs a per-session pass after
  every chunk, which compressed the held blocks a moment later, so v1 measured
  bit-identical to plain streaming (KL 0.2193 per step). v2 gates both mid-prefill
  paths; a debug print confirms both hold.
- Tier 2 (granite, hybrid store, 374 steps):

| arm | KL | dKL vs base [95% CI] | top-1 | store GB | peak GB | dec tok/s |
|---|---|---|---|---|---|---|
| hybrid exact | 0.1350 | -0.179 [-0.319, -0.067] | 0.949 | 0.809 | 11.11 | 8.7 |
| hybrid streaming | 0.2193 | -0.094 [-0.238, +0.039] | 0.920 | 0.809 | 10.94 | 8.3 |
| **hybrid streaming + elastic** | **0.1350** | -0.179 [-0.319, -0.067] | 0.949 | 0.809 | 10.91 | 8.1 |

  Reading: where the history fits (all 24 prompts, up to 16k), streaming + elastic IS
  exact mode, bit for bit, at a lower peak. Streaming only pays its quality cost when
  memory forces it.
- Reach: the 65k spill seen in v1's ladder (TEST_granite_hybrid_elastic_ladder.jsonl; v1
  was inert, so that ladder is plain hybrid streaming) is NOT elastic's: plain hybrid
  streaming spills too (TEST_granite_hybrid_stream_65k.jsonl: 11.91 alloc / 12.17
  reserved; line 12.11). Shipped tiled DKV at 65k peaks 9.36. Cause: the memory-sized
  remat cache fills headroom up to its gate, and the gate's fixed 1.5 GB step headroom
  (`DKV_REMAT_RESERVE_GB`) is smaller than the 65k decode transient. Test of 2.5 / 3.5 GB
  queued (benchmarks/run_remat_reserve.cmd); v2 elastic 65k running.

## Generality (tier 1, 2026-10-04): the hybrid on three architectures

| Model | Store | out_err | attn KL | B/tok |
|---|---|---|---|---|
| granite-4.2-8b | DKV / KIVI-4 / hybrid | 0.076 / 0.096 / 0.077 | 0.094 / 0.003 / 0.003 | 962 / 1215 / 980 |
| Qwen3.5-4B (32k/64k) | DKV / KIVI-4 / hybrid | 0.085 / 0.032 / 0.078 | 0.204 / 0.003 / 0.003 | 658 / 1195 / 870 |
| Qwen2.5-7B (12k) | DKV / KIVI-4 / hybrid | 0.539 / 0.077 / 0.627 | 1.60 / 0.004 / 0.009 | 504 / 611 / 500 |

- Quantized keys fix the attention pattern on all three (KIVI-level attn KL).
- Low-rank VALUES are model-dependent: better than 4-bit on granite, worse on Qwen3.5,
  and broken on Qwen2.5-7B, concentrated in layers 17 and 24 (err 0.7-1.6; KIVI 0.07-0.13).
  Likely cause: massive-activation tokens (huge norms) defeat the token-norm-scaled factor,
  and 4-bit per-column U rounds ordinary tokens to ~0 next to them. Shipped DKV fails
  there too (layer 24 err 1.6), so this is a pre-existing DKV generality defect, not one
  the hybrid introduced. **Top open item:** a value store robust to outlier tokens
  (e.g. outlier tokens kept exact before factoring, per-token U scales, or per-token
  quantized values where low rank fails), decided per layer from measured error -- not
  per model.

## Packed hybrid store (2026-10-04): implemented behind `DKV_KEY_QUANT=pc4` (off by default)

- Pool: per-channel 4-bit key codes `[n, T, H_kv, D/2]` uint8 + fp16 scale/zero per
  32-token group; allocated/grown/reset/written only when the switch is on; quantized
  against fp16-rounded scales so encode and decode agree exactly. Per-block/CPU write
  path refuses (raises) rather than store a slot without keys; batched write without
  keys raises. CUDA graphs auto-disabled when on (keys decode through materialise).
- Compressor: key half of the factorization zeroed (all rank to values); exact key
  deltas handed to the pool; residual selection unchanged (one set, worst joint rows).
- Decode: materialise path rebuilds keys from codes (anchor + dequant), values from the
  factor; serves the hybrid even with the remat cache off (streaming) by rebuilding per
  step and storing nothing; any decline on a hybrid pool prints a loud warning.
- Streaming prefill: history attention takes compact codes, sliced per block tile, and
  dequantizes per tile (keeps the tiling's memory bound).
- Tests: tests/test_key_quant_hybrid.py (8: round trip vs reference for 4/8 bits incl.
  ragged groups, pool alloc/write/grow/read, refusals, remat k_delta, history-attention
  equivalence); smoke test learns the hybrid (key error 0.078 vs 0.867 low-rank);
  existing parity suites unchanged (33 pass).
- Store accounting (dkv_kv_bytes) counts the key codes.

### Packed hybrid, tier 2 (2026-10-04, granite, 24 prompts, 374 steps)

| Arm | KL (base 0.314) | dKL vs base [95% CI] | top-1 | store GB (cmp) | peak GB | dec tok/s |
|---|---|---|---|---|---|---|
| packed, exact | **0.183** | **-0.131 [-0.228, -0.054]** | +2.4% | 0.985 (2.26x; base 0.624, 3.56x) | 11.34 (+0.6) | 6.7 (-4%) |
| packed, streaming | **0.270** | -0.044 (n.s.) | +0.8% | 0.985 | 9.33 (+1.3) | 3.0 (-45%) |
| base, streaming | 0.527 | +0.213 [+0.113, +0.331] | -2.9% | 0.624 | 7.99 | 5.5 |

No fallback warnings: every hybrid decode step was served by the materialise path.
Streaming packed is as good as today's EXACT mode. Costs to address before it can ship:
store +58% (keys at 4 bits are ~0.64 B/element incl. scales; on granite that would cut
streaming reach from 131k to ~86k by extrapolation) and streaming decode -45% (routed
blocks rebuilt every step). Store trims planned: drop the factor's zero key half,
serve key residuals from the codes (~18%).

Also fixed: the tiered block store pages slots to host memory and back, and did not
know about the key codes, so a restored hybrid slot would have held stale keys. It now
pages them. **Pre-existing, logged not fixed:** the pager also does not page residuals.
**Audit item:** block-level `V` properties (kv_runtime_manager / streaming_sparse_ingest)
rebuild keys from the factor; any consumer of them on a hybrid pool reads anchor copies.

### Hybrid + attention-ranked residuals, tier 2 (2026-10-04)

| Model | Arm | KL | dKL vs base [95% CI] | KL>0.5 | top-1 | dec tok/s |
|---|---|---|---|---|---|---|
| granite (374 steps) | base | 0.314 | - | 35 | 0.914 | 7.0 |
| granite | packed | 0.183 | -0.131 [-0.228, -0.054] | 25 | +2.4% | 6.7 |
| granite | **packed + `DKV_RESID_ATTN`** | **0.130** | **-0.184 [-0.326, -0.071]** | 24 | **+3.5%** | 5.9 |
| granite | packed + resid-attn, streaming | 0.222 (base stream 0.527) | -0.091 (n.s.) | 27 | +0.8% | 3.5 |
| Qwen2.5-7B (103 steps) | base | **1.114** | - | 26 | 0.816 | 13.5 |
| Qwen2.5-7B | packed | 0.906 | -0.208 (n.s.) | 29 | -1.9% | 13.5 |
| Qwen2.5-7B | resid-attn only | 0.926 | -0.188 (n.s.) | 16 | +4.9% | 13.6 |
| Qwen2.5-7B | **packed + resid-attn** | **0.251** | **-0.863 [-1.345, -0.458]** | **6** | **+10.7%** | 13.3 |

Reading: on a third architecture, neither fix alone is enough; together they cut KL by
77% -- quantized keys fix the attention pattern, attention-ranked residuals fix the values
attention actually reads. This is the architecture-level result: no model-specific code.

### Store trims (2026-10-04, active only with `DKV_KEY_QUANT`)

- Factor stores the value half only (`V_KV` second dim 1); `V_K` reads as a broadcast view
  of zeros (no memory), so every reader's shapes stay valid; shared bases refused in the
  hybrid. Block-level `V` properties now read through `pool.V_K/V_V` (raw half indexing
  would raise with one half).
- K residual values are no longer stored: `get_residual_k` dequantizes the key codes at
  the residual positions (only the gathered rows). Gather and router recognise the codes
  as a residual source (gather checks them first so it never dequantizes the whole pool).
- Tests: 10 hybrid tests (adds value-only factor, codes-backed K residuals incl. padding
  and growth, for int8 and fp16 residual formats); parity suites and smoke unchanged.
- Tier 2 re-check (2026-10-04): quality unchanged -- granite exact KL 0.1325 (-0.181
  [-0.323, -0.068], top-1 +3.7%), streaming 0.220; Qwen2.5-7B 0.249 (-0.865 [-1.354,
  -0.459]). Peak -0.19 GB and exact decode 5.9 -> 6.6 tok/s vs the untrimmed hybrid.
- Store (accounting fixed to count a value-only factor and V-only residual rows), granite
  per 1,024-token block per layer: shipped DKV 676 KB (6.1x), hybrid 996 KB (4.1x, was
  1,201 KB before the trims), dense 4,096 KB. The +47% is the inherent cost of 4-bit keys.
- Committed f913d7db (switch off by default; revert = leave it off or git revert).

**Open costs before the hybrid can be a default:** streaming decode 3.4 tok/s vs 5.5 for
streaming base (routed blocks rebuilt every step; a fused kernel reading the codes, or a
bounded remat cache, is the fix); reach on granite (store +47% -> ceiling to be measured by
ladder in the overnight runs).

### Hybrid streaming decode speed (2026-10-04; hybrid-only unless noted)

Profiled with benchmarks/profile_hybrid_decode.py (synchronized timers around each
decode-path function; granite, 16k, 48 tokens; ms/token incl. sync overhead):

| Change | ms/token | Notes |
|---|---|---|
| start (hybrid streaming) | 435 | base streaming 217 |
| dequant broadcasts scales (no [N,S,H,D] gather), fp32 math, one rounding | -- | all readers agree bit-for-bit |
| skip key residuals in the materialise path (the codes already give those keys) | 391 | |
| rebuild in the factor dtype (fp16), no fp32 intermediates | 322 | substitution, not small-delta addition, so no precision loss |
| memory-sized remat cache (`_remat_fits`): keep a layer's rebuilt blocks while reserved memory after the entry + 1.5 GB step headroom stays under 94% of the card; always look up, drop stale entries that cannot refresh | 274 | 33/40 layers cached at 16k, no paging. First two gate versions were wrong and are recorded: driver "free" ignores reserved-but-unused memory (refused at 1.3 GB "free"); allocated-only + 1 GB let reserved pass the card (attention 98 -> 149 ms from paging) |
| GQA fold in `attend_with_remat`: one decode token, the query heads of a KV head laid along the query axis, no K/V copies | **232** | bit-identical to expansion (test); on for the hybrid, `DKV_REMAT_GQA_FOLD=1` extends it to every materialise step (default path untouched until measured) |

Tests: test_key_quant_hybrid.py now 11 (adds the GQA fold equivalence).

Tier 2 (granite, 374 steps): quality unchanged and decode now FASTER than shipped DKV --
hybrid exact KL 0.135 (-0.179 [-0.319, -0.067]) at 8.7 tok/s (base 7.0); hybrid streaming
KL 0.219 at 8.3 tok/s (base streaming 0.527 at 5.5). Store 0.809 GB (2.75x; base 0.624,
3.56x) = +30% with the corrected accounting. Streaming peak 10.94 GB vs 7.99: the
memory-sized cache using free headroom under the spill line (gives way to the store;
reach to be confirmed by ladder).

## Batch E: prefill speed and baseline fairness (2026-10-04)

**E2 larger prefill chunks -- no code.** `DKV_PREFILL_CHUNK_SIZE` already sets it
(rounded up to whole blocks: 2048 -> 2050 = 2 blocks per chunk). Measurement queued
(benchmarks/run_e2_chunks.cmd): tier 2 at 2048/4096 exact and streaming, and 32k prefill
time. An older single-seed note in config.py links chunk 2048 to a lower score; tier 2
decides.

**E1 fused history attention** -- `DKV_PREFILL_SDPA=1` (off by default;
`history_attend_sdpa_tile` in triton_fused_decode.py). Streaming prefill attends the
compressed history by rebuilding each block tile's K/V and scoring every query against
every key in explicit [H, Q, keys] tensors, sliced into 256-query pieces to bound them.
E1 rebuilds K/V once per tile at KV-head width (every `groups`-th head of the repeated
inputs is the KV head), folds the GQA group into the query length, drops padded columns
instead of masking them, and runs torch's memory-efficient attention, which returns the
log-sum-exp the tile merge needs. No query slicing, no score tensors, no per-model code.
Tile 64 blocks (`DKV_PREFILL_SDPA_TILE`).
- Tests: tests/test_prefill_sdpa.py (5): both residual semantics, tiled vs untiled, no
  GQA, hybrid key codes. Both paths sit the same distance from an fp32 reference (0.011
  on outputs of 8.8 each), so the bound is relative.
- Real-model fidelity and 32k prefill time: queued with E2.

**E3 tiled KIVI-4 baseline** -- new arm `kivi4_tiled` (benchmarks/kv_baselines.py). Same
quantizer and bytes as `kivi4_chunked`. kivi4_chunked's ceiling (49k granite, 98k Qwen)
was its DECODE peak (12.96 GB at 65k vs 10.49 prefill): update() returned the whole
dequantized history every step. kivi4_tiled returns only the 16-bit window, and a
registered attention function (transformers AttentionInterface, so any model on that
interface) reads the 4-bit history in 8k-token tiles, merged by log-sum-exp, the same
treatment DKV's streaming path gets. One difference, in KIVI's favour: tokens are
quantized at the start of the next forward, so a prefill chunk attends its own tokens in
16 bits (as KIVI's reference prefill does); it also removes masking from the codes.
The model's own attention implementation is restored after each item.
- Tests: tests/test_kivi_tiled.py (3): prefill chunk with mask, first chunk without, decode;
  multi-tile; equals attention over the dequantized cache.
- Reach smoke (granite 65k/98k) and RULER 24k agreement vs kivi4_chunked: queued
  (benchmarks/run_e3_kivi_tiled.cmd).

## Untried ideas on the hybrid's value side (tier 1, 2026-10-04)

All on top of the hybrid with attention-ranked residuals (`Hv_att`); keys unchanged, so
attention KL is identical across rows and out_err decides. benchmarks/tier1_screen.py,
CPU, all captures (granite 6 prompts, Qwen2.5-7B 3 prompts).

| idea | variant | granite out_err / B/tok | Qwen2.5 out_err / B/tok | verdict |
|---|---|---|---|---|
| reference | Hv_att | 0.0624 / 1279 | 0.1011 / 661 | |
| B2 attention-weighted SVD (rows weighted after token-norm normalization) | U_B2_svdw | 0.0640 / 1279 | 0.1049 / 661 | drop: no gain |
| B5 per-KV-head factors, rank 4 / 8 per head | U_B5_head4/8 | 0.1032 / 0.0860 | 0.1393 / 0.1280 | drop: much worse; the joint factor's shared U is what makes it cheap |
| DuoAttention-style: heads with the most far-history mass keep 4-bit values | U_duo12/25 | 0.0614 / 1337, 0.0667 / 1395 | 0.0952 / 719 | drop: worse per byte than more rank |
| B4 energy-adaptive rank per block (e=0.80, cap 64) | U_B4_e80 | 0.0587 / 1297 | 0.0853 / 686 | marginal, closed (below) |
| rank 64 for comparison | Hv_r64att | 0.0524 / 1361 | 0.0808 / 716 | |
| KIVI-4 | kivi4 | 0.0955 / 1215 | 0.0769 / 611 | |

B4 closed: the runtime already truncates rank by energy (`DKV_SVD_ENERGY`), and its
stored rank is capped at 32 by the batched-solver limit (`_pool_rproj_cap`, see
memory note on preset rank), so B4's gain is the blocks that would go above 32. That
needs the cap lifted (measured +13% forward, ~+1.9 s/prefill), and value-side precision
was not significant in tier 2 on granite (exact-values probe n.s.). Revisit only if a
model shows value-limited tier-2 error (Qwen2.5-7B is the candidate).

## Values robust to outlier tokens (tier 1, 2026-10-04)

| Value side (keys 4-bit per channel) | Qwen2.5-7B out_err | granite out_err | bytes |
|---|---|---|---|
| low rank, residuals by error | 0.594 | 0.076 | ref |
| + per-row U scale / + 16 outliers excluded from the SVD | 0.594 / 0.594 | 0.076 / 0.077 | ~same |
| + rank 64 / + 128 residuals by error | 0.548 / 0.586 | 0.062 / 0.070 | +7% / +5% |
| per-layer fallback to 4-bit values (rel-error rule) | 0.076 | 0.094 (switched layers that were fine) | +22-25% |
| **residuals ranked by error x attention received (whole prompt, per-block normalized)** | **0.101** | **0.060** | **same** |
| KIVI-4 (reference) | 0.077 | 0.095 | |

Reading: Qwen2.5's value failure was never low rank itself -- per-layer value error and
rank-32 energy are the same as granite's (~0.6 / ~0.73), and its attention outputs are not
small. It is WHICH tokens get exact values: a few heavily-attended tokens were missed by
error-ranked selection. Attention-ranked selection (`DKV_RESID_ATTN`, already built, which
did nothing alone because key errors dominated) fixes it at no byte cost. Next: tier 2 of
packed hybrid + `DKV_RESID_ATTN` on granite and on Qwen2.5-7B.

## Instrument fixes

| Date | Fix | Why |
|---|---|---|
| 2026-10-03 | Decode-fidelity disables CUDA graphs in DKV arms | The forcing hook cannot be captured; replays returned stale buffers (KL 4-15 on prompts under `DKV_GRAPH_MAX_CTX`=8192) |
| 2026-10-03 | Tier-1 base uses the runtime's residual rule | Runtime ranks joint ABSOLUTE error with one index set for K and V; the first screen used relative error, overstating attention weighting's gain |
| 2026-10-03 | Tier-1 tools read RoPE / scale / q-k norms from the live model | No per-architecture code |
| 2026-10-04 | Probe / hybrid row selection moved after BOTH residual branches | With a residual budget >= block size every block takes the store-everything (`force_exact`) branch; logic placed only in the ranking branch never ran, so the first K/V probe read KL 0.0008 for both sides (both fully exact). The smoke test's pool (64 slots) never reaches that branch, which is why it passed. Verified in the real run: `force_exact=True K_rows=1024 V_rows=128` |
| 2026-10-04 | decode_fidelity reuses finished arms | Power cut mid-run; a rerun now skips arms whose output is newer than the dense control |

## Open items found along the way

- Paper says the factorization is per head; the code factors all KV heads of a layer jointly (`feat_dim = 2*H*D`). Correct the paper.
- On short contexts CUDA uses adaptive (smaller) blocks; the paper's "1024-token blocks, 1,536-token window" describes long contexts only. Verify and state.
- Residual selection falls back to the last 64 prompt tokens as a query (`DKV_RESIDUAL_QUERY_TAIL`); within equal information, but not strictly query-agnostic. State in the paper or replace with resid-attn.
