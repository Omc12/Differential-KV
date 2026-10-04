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
