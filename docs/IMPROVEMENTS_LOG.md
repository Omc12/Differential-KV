# DKV improvements log

Every change to DKV's store, decode or prefill made in the improvement batches
is recorded here: what it is, the switch that controls it, what each test tier
measured, and when (and why) its default changed. Newest entries on top within
each batch.

Tiers: **T1** offline screen on captured KV (`benchmarks/tier1_screen.py`),
**T2** real decode on 6 fixed prompts (`benchmarks/decode_fidelity.py`),
**T3** batch confirmation. Guardrails a change must not trade away: store bytes,
reach (peak memory), decode speed, query-agnosticism, exact prefill = dense.

## Router: adaptive growth at any block count; head vote opt-in (V20-V22, 2026-11-27)

Two router defects examined on Qwen2.5-7B and granite; neither change looks at
the model. One became the default; the other only helps some models.

| Change | Switch | Default | Evidence |
|---|---|---|---|
| **Adaptive growth at any block count.** The grow-only block count (`_adaptive_select`) only ran above 2K = 32 candidate blocks, i.e. never at 32k. The cutoff came from a LOW-RANK store measurement (extra blocks added reconstruction error); on the codes store extra blocks help | `DKV_ROUTE_ADAPTIVE_MIN_BLOCKS` | **0** (was 2K) | 32k decode KL, max router (V22b): Qwen2.5-7B **0.101 -> 0.036** (every block 0.047, KIVI-4 0.016), RULER 4 items **63.8 -> 95.0** (dense 95); granite 0.111 -> 0.105; SmolLM3-3B 0.0077 -> 0.0060. Phi-4-mini decode 11.7 vs 11.6 tok/s (growth bounded by the estimated mass, P=0.80) |
| **Head vote, opt-in.** The router ranks a block by the largest raw `q.k` of ANY head; a raw logit means nothing outside its head, so the head with the largest logit scale can choose every block (Qwen2.5-7B, k_proj bias to 414: one head decided all 16 kept blocks in most layers; layer 27 logits -4493..643). `vote`: each (head, query) row spreads one vote over the blocks by its own softmax, ranked by the mean vote | `DKV_ROUTE_HEADS=max\|vote` | **max** (vote was the default for one commit, 1fc5e8c9) | real router, whole-layer attention-output error (`benchmarks/diag_route_runtime.py`), Qwen2.5-7B two items: max 0.079 / 0.134, vote 0.056 / 0.076. But decode KL with growth, max vs vote: **SmolLM3-3B 0.0060 vs 0.044**, Qwen2.5-7B 0.036 vs 0.060 (RULER 95.0 vs 82.5), granite 0.105 vs **0.075** (0.066 without growth), Phi-4-mini 0.0069 vs 0.0073, Qwen3.5-2B 0.0012 vs 0.0009, Ministral 0.0021 vs 0.0021. A block one head needs is diluted by the heads that do not care; max keeps each head's sharpest block |
| Rejected: normalise each head, then the max | (V20) | - | exact keys said yes (`diag_route_oracle.py`, missed mass 0.087 -> 0.061); real decode said no: Qwen2.5-7B KL **0.101 -> 0.217**, RULER 63.8 -> 57.5. From the anchor plus <=64 residual rows some heads are badly estimated and the max lets the worst one win (layer 0 output error 0.03 -> 0.36) |
| Residual rows as the router's view: kept | - | - | under the codes store the rows hold no values (keys are read back from the codes) and exist only for routing. Per-head rows by deviation from the block's mean key, same count: 0.057 vs current 0.054 output error (one item). The current ranking stays |

Lessons: score routing rules in real decode across architectures, not on exact
keys (V20) and not on mean attention-output error alone (vote won that and
lost SmolLM3). Short checks only, 4 prompts per model, one item per RULER
task: `benchmarks/campaigns/run_v21_head_vote.cmd`, `run_v22_router_fidelity.cmd`,
`run_v22b_max_grow.cmd`; logs `paper/results/campaign/v20.log`, `v21.log`,
`v21_ruler.log`, `v22.log`, `v22_smollm3.log`, `v22b.log`, `v22c.log`;
RULER `paper/results/ruler/V20_*`, `V21_*` (vote+growth), `V22b_*` (default).
V20's codes_v19 / codes_v20 arms in the summaries were the rejected rule; the
summaries keep V19b's rows and one `codes_v20_hnorm` row. Granite and
Ministral decode speeds are unreliable (both spill past the card).

## Value codes hold the values, not anchor deltas; the Qwen2.5-7B gap explained (V18-V19c, 2026-11-26)

Asked: is Qwen2.5-7B struggling because of something model-specific in DKV?
Audit: no model-name branches anywhere in the runtime; head geometry, sliding
windows, rotary and NoPE layers are all read from the live model. What was
found instead, in order:

| Finding | Evidence | Action |
|---|---|---|
| **Bug (generic): values were coded anchor-relative.** A per-token quantizer's range is the row's own; subtracting the block's anchor widens every row's range to the anchor's, so an outlier anchor ruins its whole block. KIVI codes values as they are, and values carry no RoPE, so the anchor frame bought nothing | `diag_store_recon.py`, Qwen2.5-7B 32k, every block, real first decode query: attention-output error at layer 5 **1.245 -> 0.119** (KIVI-4 0.120); layers 12/17/24 0.34/0.26/0.37 -> 0.068/0.067/0.104 (KIVI 0.07-0.14) | fixed: the compressor hands the codec the values themselves; decode reads them without adding the anchor. Codec contract: keys arrive as anchor-relative deltas, values as they are (`runtime/kv_codecs.py`); `get_value_deltas` is now `get_values` |
| Keys: DKV's stored blocks are better than KIVI-4's | same diagnostic against the model's own arithmetic (keys rotated in bf16, as HF does): attention KL layer 0 0.0045 vs 0.0121, layer 5 0.0010 vs 0.0012, layer 27 0.0109 vs 0.0188 (KIVI post-RoPE, dequantized into bf16). A float32 reference had shown DKV worse at layer 27; that reference sits below bf16's grid at Qwen2.5's key magnitudes (k_proj bias up to 414 at layer 27, 171 at layer 0) | none; the diagnostic now prints both references |
| **The 4-bit store is not the remaining gap.** A lossless codec (`examples/lossless_codec.py`: key deltas fp32, values bf16) through the whole DKV pipeline, every block read | decode KL vs dense 0.0418 vs 4-bit codes 0.0467 (KIVI-4 0.0155) | the lossless codec is kept as the instrument for "store or machinery?" |
| **Qwen2.5-7B's prefill is numerically fragile on its own.** First-token KL of DKV's exact prefill is 0.42 / 0.23 on two of four prompts, identical for every store (so not compression) | `diag_prefill_layer_diff.py` 32k: DKV vs dense (SDPA) layer-0 rel err 0.066, worst 0.14; dense EAGER vs dense SDPA, same weights: 0.039, layers 2-9 **0.17-0.22**. Two correct dense implementations differ more than DKV does | none; not a DKV defect. KIVI's baseline runs HF's own kernels, which is why it tracks the SDPA reference more closely on this model |
| Routing is the largest share of what is left | RULER 32k (4 items): routed 63.8, every block read 88.8, KIVI-4 100. Fidelity KL routed 0.101 vs every block 0.047; granite routed 0.111 vs every block **0.0013 (KIVI 0.0066)** | addressed in V21-V22 (top section): growth at any block count. Was open. Reading every block is no slower at 32k on Qwen2.5 (10.7 vs 10.2 tok/s) but 5x slower on Phi-4-mini (2.0 vs 9.8), so it is not a default |

**V19** (`benchmarks/campaigns/run_v19_ruler_raw_values.cmd`): V18's DKV arm
again after the fix, same items; KIVI-4 arms from V18. RULER 32k, DKV codes /
KIVI-4 tiled:

| model | V18 DKV (pre-fix) | **V19 DKV** | KIVI-4 |
|---|---|---|---|
| SmolLM3-3B (13 tasks) | 76.2 | **78.5** | 86.9 |
| Qwen2.5-7B | 63.8 | 63.8 | 100 |
| Phi-4-mini | 70 | 70 | 70 |
| Qwen3.5-2B | 100 | 100 | 100 |
| gemma-4-e2b | 88.8 | 88.8 | 88.8 |
| gemma-4-e4b | 100 | 100 | 100 |
| gemma-4-12B | 100 | 93.8 | 93.8 |
| Ministral-3-8B | 100 | 100 | 100 |

One item per task (n=4, SmolLM3 n=13): a one-item move is 6-25 points, so only
Qwen2.5-7B's gap is outside noise, and V19b puts most of it on routing.
Logs: `paper/results/campaign/v18.log`, `v19.log`, `v19b.log`,
`v19c_lossless.log`.

## vLLM backend: DKV codec pages in vLLM 0.34.0 (2026-11-26)

`integrations/vllm/` (pip package `dkv-vllm`, `vllm.general_plugins` entry
point, `--attention-backend CUSTOM`). Measured on one model and GPU (Qwen2.5-
1.5B, RTX 4070 SUPER under WSL2, CUDA graphs on for every arm); full table in
`integrations/vllm/README.md`.

| Phase | What | Gate | Result |
|---|---|---|---|
| 1 | backend delegates to vLLM's Triton backend | token-identical to stock on 20 prompts, backend verified in use | 20/20 PASS |
| 2 | `k8v4` codec pages (8-bit per-token keys, KIVI 4-bit values; page size set through `customize_spec`), fused Triton store + split-K decode kernels, CUDA-graph capturable decode | capacity, quality, speed vs stock and vLLM's own TurboQuant k8v4 | 32k: 2.01x KV tokens, needle 6/6, decode 107.8 vs stock 102.4 tok/s; greedy agreement with stock 29.4/48 vs TurboQuant 23.9 |
| 3 | routed decode over pages (per-page key bounds, Quest's bound, Triton score kernel) | faster than reading every page, no quality loss | NOT met at 16k-32k (99.9 vs 107.8 tok/s, needle 5/6): opt-in, `DKV_VLLM_ROUTE_TOKENS` |

Bugs found on the way, each with its fix: padding rows of a page's tail
indexed past the chunk (device assert); persistent page bounds reallocated
after CUDA-graph capture, so captured decode routed against bounds nothing
wrote (needle 2/6 under graphs, 6/6 eager; now grow-only); routing gathered
`[batch, pages, heads, D]` bounds (OOM at vLLM's 512-batch capture; now a
Triton score kernel); the reference prefill's temporaries were invisible to
vLLM's memory profiling (OOM at 32k; now tiled with `enable_gqa`).

## Codes (KIVI-4 layout) is the default store; pluggable codecs; vLLM backend (V16-V17b, 2026-11-26)

**What changed, and the evidence for each**

| Change | Switch | Default | Verification |
|---|---|---|---|
| The default store is KIVI-4's layout (keys per-channel, values per-token, 4-bit), read through DKV's routed decode | `DKV_STORE=codes`; `DKV_LOWRANK=1` (or `DKV_STORE=lowrank`) opts back | **codes** (was lowrank) | V17b below: quality beats low rank on every model where both ran, at the same decode speed; 2.3x the store bytes |
| Pluggable codecs: a codec owns the K and/or V deltas of a block; the runtime (pool, growth, offload, prefill tiling, gather, router) is codec-agnostic. Built-ins re-expressed as codecs, bit-identical | `DKV_STORE=<name>` or `package.module:Attr`; `config={"codec": ...}`; `register_codec(name, encode=, decode=)` | - | `tests/test_kv_codecs.py` (12): built-ins bit-identical to the old quantizers; an outside two-function codec (`examples/int8_codec.py`) runs compress, growth, router residual keys, tiered paging and the prefill tile; prefill tile through a codec == through the old keywords, bit for bit |
| The codes store no longer runs the (never read) low-rank SVD of the values | `DKV_CODES_FACTOR=1` restores it | off | V17b `codes_factor` vs `codes`: same quality on all 8 models, 0.3 s less to first token on SmolLM3 |
| Hybrid-era switches keep their low-rank base when `DKV_STORE` is unset (`DKV_KEY_QUANT=pc4` alone is the hybrid again; `=off` no longer leaves value codes without key codes) | - | - | `test_legacy_key_quant_keeps_its_lowrank_base` |
| head_dim not a multiple of 32: the default falls back to low rank (an explicit `DKV_STORE=codes` still raises) | - | - | `test_default_falls_back_on_ungroupable_head_dim` |
| Run metadata records the RESOLVED store (`dkv_store`), and `DKV_LOWRANK` | - | - | an unset `DKV_STORE` meant lowrank before this day and codes after it |
| **Bug: NoPE layers rotated in streaming prefill.** The prefill RoPE-slice cache was keyed on block anchors only and shared across layers, so a NoPE layer (SmolLM3, Llama 4) reused a rotary layer's tables whenever prefill attended compressed history | - | fixed | SmolLM3 8k per-layer diff: first NoPE layer 0.60 -> 0.017 rel err, last layer 1.22 -> 0.055; any store |
| **Bug: residual weighting used the VISION tower's rotary on VLMs.** `_find_rotary` took the first `*RotaryEmbedding`; on Ministral-3 that is Pixtral's 2-D table: a device-side gather out of bounds at 32k (every codes arm failed), silently wrong rotation below it; on gemma-4-e2b it raised inside a try and the weighting was skipped | - | fixed (uses `_resolve_rotary_emb`) | Ministral 32k codes: crash -> clean; `tests/test_rotary_resolution.py` |
| Fidelity harness gains a KIVI-4 reference arm (`kivi4`, tiled where the model allows) and `kivi4w` (KIVI given DKV's 16-bit window, so a gap cannot come from window size); decode tok/s and time-to-first-decode recorded | `decode_fidelity.py --arms kivi4 kivi4w:KIVI_RESIDUAL=dkv` | - | three harness bugs found and fixed on the way: sliding-window cache layers (Phi-4's declared 262k window: KL 3.8-17 from a mask/length mismatch), gemma-4 KV-shared layers (index out of range; must read the source layer's history), and a cache leak across prompts (a reference cycle; KIVI's peak climbed 8 -> 12 GB). KIVI peaks in the table below for SmolLM3, Qwen2.5-7B and Phi-4 predate the last fix and are high; their KL is unaffected |

Removed after measurement: a "KIVI-style" streaming default for the codes
store (`DKV_CODES_STREAM`: compress each block at once, no elastic window, no
sparse prefill). It lowered peak memory but lost on quality and prefill time
on every model, and an automatic exact/stream switch misjudged the card in
both directions. One default instead: **exact prefill** on every store, with
streaming as the single explicit switch (`DKV_STREAMING_COMPRESS=1`).

**V17b** (`benchmarks/campaigns/run_v17b_routing.cmd`, `decode_fidelity.py`,
RULER 32k prompts, dense's own answer forced through each arm, KL per decode
step; paired difference vs tiled KIVI-4 with a bootstrap 95% CI;
`paper/results/campaign/v17b.log`):

| model | KIVI-4 | DKV codes | paired vs KIVI | codes, every block read | DKV low rank | decode tok/s KIVI / codes | peak GB KIVI / codes |
|---|---|---|---|---|---|---|---|
| SmolLM3-3B | 0.0077 | 0.0055 | -0.002 [-0.008, +0.003] tie | 0.0041 | 0.128 | 3.6 / 8.0 | 4.4* / 6.1 |
| Qwen2.5-7B | 0.0155 | 0.101 | **+0.085 [+0.027, +0.161] worse** | 0.053 (+0.038) | 0.789 | 4.6 / 10.9 | 7.5* / 8.7 |
| Phi-4-mini | 0.0223 | 0.0077 | **-0.021 [-0.031, -0.011] better** | 0.0052 | 0.211 | 3.8 / 9.8 | 7.7* / 9.5 |
| Qwen3.5-2B | 0.0071 | 0.0016 | **-0.0055 [-0.0094, -0.0022] better** | 0.0022 | 0.0042 | 11.9 / 16.3 | 2.7 / 3.1 |
| Ministral-3-8B | 0.0013 | 0.0020 | +0.0007 [-0.0008, +0.0021] tie | 0.0020 | 0.324 | 3.7 / 1.9 (spill) | 8.1 / 13.3 |
| granite-4.2-8b | 0.0066 | 0.101 | **+0.095 [+0.007, +0.209] worse** | **0.0014 (-0.005, better)** | 0.372 | 3.1 / 2.0 (spill) | 7.9 / 13.9 |
| gemma-4-e2b | 0.171 | 0.012 | -0.108 [-0.325, +0.007] | 0.012 | 0.016 | 8.0 / 12.1 | 8.0 / 7.9 |
| gemma-4-e4b | 0.0048 | 0.067 | +0.062 [+0.027, +0.106] | 0.067 | 0.068 | 7.1 / 7.1 | 10.7 / 10.9 |

(* KIVI peak from a run with the leak.) Reading it:

- **The codes store is not the problem; routing is.** With every block read,
  DKV codes is at or below KIVI-4 on 6 of 6 non-gemma models (granite 0.0014
  vs 0.0066). The losses on Qwen2.5-7B and granite come from the router
  skipping blocks: all of granite's gap, half of Qwen2.5-7B's (the other half,
  0.053 vs 0.016 with every block read, was traced on 2026-11-26 to the
  model's own numerics, not the store: see "Value codes hold the values"
  above).
- **Decode is 1.4-2.7x faster than tiled KIVI-4** wherever DKV fits the card.
- **Memory is the cost.** Exact prefill holds the whole prompt in 16 bits, so
  DKV peaks higher than KIVI and the two 8B models SPILL at 32k on a 12.9 GB
  card (13.3 / 13.9 GB; decode then 1.9-2.0 tok/s). `DKV_STREAMING_COMPRESS=1`
  fits them (10.2-10.4 GB) at equal quality on Ministral and a cost on
  granite (0.148).
- **gemma is not comparable yet.** Every DKV arm, low rank included, lands on
  the same KL on both gemma models, so DKV's gemma result does not come from
  the store at all; the e2b KIVI number (0.171) is also out of line. Open.

## Residual rows no longer crowded out by token runs; Phi-4-mini fixed (V15, 2026-11-26)

V13's open defect 4 (Phi-4-mini second-token KL 0.68) was NOT the joint K|V
factor. `benchmarks/diag_store_recon.py` reads what the decode path actually
rebuilds (it wraps `reconstruct_blocks` and the routed-block gather) and compares
it with the exact pre-RoPE K/V from prefill, then attends both with the real
first decode query. On Phi-4-mini at 32k, layer 9, block 0: two rows (positions
156 and 172, value-heavy tokens: |dV| 3x the block median) were rebuilt WORSE
than the anchor alone (key error 1.5x their own delta, ranks 1 and 2 of 1025 by
both key and value error) and took 66% of head 22's attention (exact: 1%).
Attention KL over the routed blocks: layer 9 2.43, layer 16 0.84.

Neither row had a residual. The block's 128 slots went to rows with key error as
low as 0.50, because with no question pinned (the default since the 2026-11-24
review) `_greedy_whole_runs` reserved EVERY token run first. A run is any span
with a digit or all-caps word, so each 'AI' / '2010s' / 'GPU' of the repeated
filler sentence is one, and together they filled the budget before the per-row
ranking was read. The joint factor makes the miss visible (separate factors never
rebuild a row past its anchor: worst row 1.00), but the selection defect is
model-independent: the panel improved on five other models.

Also found: `benchmarks/tier1_screen.py` (the offline store model) predicted
layer 9 at 4% output error -- it ranks residuals per row with no runs, so it
could not see this. A model of the store is not the store.

| Change | Switch | Default | Verification | Notes |
|---|---|---|---|---|
| Bounded run reservation: the best run is reserved whole, further runs may take at most a share of the budget; the rest goes to the per-row error ranking. Only when no query priority exists | `DKV_RESIDUAL_RUN_SHARE` (0.5; 1 = old runs-first) | **0.5** | Phi-4-mini layer 9 attention KL 2.43 -> 0.0011, layer 16 0.84 -> 0.019; V15 panel below; `test_residual_capture.py` 37 pass (the best-run guarantees behind the 3/12 -> 11/12 sweep still hold) | Query-marked reservation unchanged. First tried: runs competing with single rows on score -- broke the tested guarantee that a buried code is kept whole when filler out-errors it, so dropped |
| `benchmarks/diag_store_recon.py` | `--layers --ctx --depth` | - | Found the defect | Runtime reconstruction vs exact K/V per block, worst rows, residual membership, and attention with the real decode query |
| `tier1_capture.py` reads fused `qkv_proj` (Phi-3 / Phi-4) | - | - | Captured Phi-4-mini at 32k | Split by widths from the live layer |

**V15 panel** (`colab/logit_fidelity.py`, 32k, nf4, `mid`, 5 depths; log
`paper/results/campaign/v15_panel.log`), second-token KL (first compressed
decode step), V13 -> V15; first-token rows unchanged:

| model | V13 | V15 | top-1 |
|---|---|---|---|
| Phi-4-mini | 0.679 | **0.0063** | 4/5 -> 5/5 |
| SmolLM3-3B | 0.110 | **0.012** | 5/5 |
| Qwen3.5-2B | 0.034 | 0.0012 | 5/5 |
| gemma-4-e2b | 0.010 | 0.0036 | 5/5 |
| Ministral-3-8B | 0.011 | 0.0007 | 5/5 |
| Qwen3.5-4B | 0.0018 | 0.0007 | 5/5 |
| Qwen2.5-7B | 0.0000 | 0.0000 | 5/5 |
| gemma-4-e4b | 0.0001 | 0.0001 | 5/5 |
| gemma-4-12B | not comparable (step-1 near-tie at depth 0.3) | same | 4/5 first token |
| granite-4.2-8b | 0.00002 | 0.00000 | 5/5 |

V13 defect 4 is closed. The `DKV_STORE=codes` / `DKV_KV_SPLIT` workarounds are no
longer needed for Phi-4-mini.

RULER rerun with the fix (V15, `benchmarks/campaigns/run_v15_ruler_dkv.cmd`,
low-rank store pinned, same items as V14): identical to V14 on 7 of 8 models;
SmolLM3 71.7 -> 64.7, one item (`niah_multikey_2`, 7615230 for 7618353: the
answer's run was no longer reserved whole -- the cost side of the bounded
reservation, at n = 1). Fidelity improved and retrieval did not.

## Cross-architecture fidelity panel: 10 models, nothing set per model (V13, 2026-11-25)

Does the method run unchanged on attention layouts it was not developed on?
`colab/logit_fidelity.py` on 10 cached models at 32k (nf4 both sides, shipped `mid`
preset, 5 needle depths), KL(plain transformers || DKV) on the first generated token
(prefill only) and the second (the first compressed-decode step). Campaign:
`benchmarks/campaigns/run_v13_panel.cmd`; logs `v13*.log`.

| model | what it tests | 1st token top-1 / KL | 2nd token top-1 / KL | status |
|---|---|---|---|---|
| SmolLM3-3B | Llama-style, NoPE on every 4th layer | 5/5 / 0.00000 (was 0/5, 9.21) | 5/5 / 0.110 | fixed (NoPE) |
| Phi-4-mini | fused qkv_proj, partial rotary 0.75, LongRoPE | 5/5 / 0.0081 | 4/5 / **0.679** | ~~open defect~~ fixed in V15 (residual selection, 0.0063) |
| Qwen2.5-7B | dense GQA, qkv bias | 5/5 / 0.0017 | 5/5 / 0.0000 | pass |
| Qwen3.5-2B | hybrid linear + full | 5/5 / 0.0000 | 5/5 / 0.034 | pass |
| gemma-4-e2b | sliding + global, KV sharing, 1 KV head | 5/5 / 0.0014 (was 2.57) | 5/5 / 0.010 | fixed (torch + harness) |
| gemma-4-e4b | same, 2 KV heads | 5/5 / 0.0044 (was 1.29) | 5/5 / 0.0001 | fixed (harness) |
| gemma-4-12B | unified composite | 4/5 / 0.012 (was 0.90) | not comparable (step-1 near-tie at depth 0.3) | fixed (harness); pass |
| Ministral-3-8B | composite mistral3 wrapper | 5/5 / 0.015 | 5/5 / 0.011 | pass |
| Qwen3.5-4B | re-anchor (V12: 0.00180) | 5/5 / 0.0000 | 5/5 / 0.00180 | pass, reproduces V12 |
| granite-4.2-8b | dense GQA, 40 full layers | 5/5 / 0.0000 | 5/5 / 0.00002 | pass |

**Four defects found; three fixed, one open.**

1. **NoPE layers were rotated (DKV bug).** SmolLM3 and Llama 4 set
   `use_rope` on layers with no positional encoding. DKV never checked it and
   rotated q/k and stored history on them. Now every rotary table a NoPE
   layer's forward builds becomes the identity (`layer_uses_rope`,
   `_nope_tables`), and `resid_attn_weight` plus the opt-in contiguous-unrotate
   finalize skip those layers. The first version tested `is not False` and
   missed SmolLM3's int `0`; the rerun measured the identical 9.21, which
   caught it.
2. **torch's efficient SDPA kernel is wrong on stride-0 K/V with a mask.**
   transformers' `repeat_kv` makes those views for one-KV-head layers (gemma-4
   sliding), and DKV's 1025-token prefill chunk is one of the bad lengths
   (L % 32 == 1). This hit layers DKV leaves NATIVE, and plain transformers
   reproduces it at chunk 1025. Isolated by a per-layer diff
   (`benchmarks/diag_prefill_layer_diff.py`): putting DKV's layer back native
   left the error unchanged; math backend and eager were correct; a pure-torch
   repro gives 5.9e-2 vs 1.7e-3. Known upstream (pytorch/pytorch#191937, fixed
   in torch 2.14). Guard: `install_sdpa_expanded_kv_guard` makes such K/V
   contiguous, installed at load. Details: `docs/upstream/torch-sdpa-expanded-kv.md`.
3. **The harness compared capped with uncapped logits (instrument bug).**
   The DKV arm hooked `lm_head`, but gemma caps logits after it (softcap 30)
   while the control read `model(...).logits`. One plain forward's capped vs
   own `lm_head` output alone is KL 0.165 at 2k. That was the rest of gemma's
   "gap": every decoder layer matched dense to bf16 noise at 8k (e4b) and 32k
   (e2b). The harness now applies the model's own softcap. ad2b88d3's gemma
   numbers (0.29 / 0.91 / 1.15 at 8k) carried both errors.
4. **Phi-4-mini compressed decode is wrong with the low-rank store (closed in
   V15, above: residual rows crowded out by token runs; the joint-factor
   conclusion below was wrong).**
   The prefill is clean (0.008), and no block is compressed before decode at
   32k (its 3.29 GB budget holds the whole prompt exact). Bisect at 32k
   (`run_v13d` .. `run_v13n` under `benchmarks/campaigns/`), second-token KL:

   | arm | KL | top-1 |
   |---|---|---|
   | baseline (lowrank, joint K\|V factor) | 0.679 | 4/5 |
   | compressed decode off (`dense_arm`) | 0.047 | 5/5 |
   | **codes store** (`DKV_STORE=codes`) | **0.0017** | 5/5 |
   | **separate K and V factors, same rank 16+16** (`kvsplit`) | **0.095** | 5/5 |
   | U column scale before int8 (`colscale`) | 0.471 | 5/5 |
   | every block attended (`allblocks`) | 1.049 | 3/5 |
   | no remat cache | 0.548 | 4/5 |
   | residual rows 0 / 200 | 0.893 / 1.165 | 3/5 / 2/5 |
   | rank 128 (64 stored) | 0.807 | 2/5 |
   | energy truncation off (`energy1`) | 0.672 | 4/5 |
   | attention-ranked residual rows (`DKV_RESID_ATTN=1`) | 0.799 | 3/5 |
   | rotated pool (`rot1`) | 16.07 | 0/5 |

   Decode-step per-layer diff (`--at decode`): layer 9 is at 104% relative
   error with the store and 8.9% with exact history. Ruled out with a run each:
   LongRoPE (Phi-4-mini's long and short factor lists are identical; tables
   are call-length independent), its rope `attention_scaling` 1.19 (forced
   to 1 in both arms: layer 9 still 142%), K/V outliers (smaller than
   granite's), the rotary base (the `mid` preset stores keys UNROTATED, so
   `unrot` equals baseline, which is why it measured identical), routing,
   rank, residual budget and truncation. The rotated pool is far worse here:
   at rotary base 10k a block of ROTATED keys is high-rank (control: the same
   arm is clean on Qwen2.5-7B, base 1M).

   A per-layer chooser that picks joint or split by measured reconstruction
   error was built and NOT shipped (its per-layer measurements are in
   `paper/results/campaign/v13o.log`, a run stopped after Phi-4): joint has the
   LOWER Frobenius error on all 32 layers (layer 9: K 0.195 / V 0.253 joint vs 0.307 / 0.386 split), yet
   split decodes far better. So what breaks decode is not reconstruction error.
   The 2026-10-03 screen that dropped the split measured the same quantity.
   Next: an attention-space criterion (error in q.k under sampled prompt
   queries) and the int8 U one-scale-per-block, which `colscale` implicates in
   part. Workaround today: `DKV_STORE=codes` (0.0017 on Phi-4; only Phi-4 was
   run with it in V13) or `DKV_KV_SPLIT=16,16` (0.095).

   Also found: Phi-3-mini-4k sets `sliding_window=2047` in its config, but its
   attention module exposes no `is_sliding` / `layer_type`, so DKV would treat
   it as full attention. Not on the panel; untested.

Also measured on the way: gemma's chunking noise floor (plain transformers,
chunk 512 vs 1024 / 1026 / 2048) is KL 0.0005-0.004, so gemma results here are
not chunk-size sensitive once the kernel bug is guarded.

| Change | Switch | Default | Verification | Notes |
|---|---|---|---|---|
| NoPE layers are not rotated | none (reads the layer's `use_rope`) | on | SmolLM3-3B 32k: first token 0/5 KL 9.21 → 5/5 0.00000; unit tests (`test_nope_layers.py`: int and bool flags, identity tables at partial rotary, residual ranking) | Covers SmolLM3 and Llama-4-style layers. The opt-in attention-interface path (`DKV_USE_ATTENTION_INTERFACE=1`) is NOT covered |
| Guard torch's efficient-SDPA stride-0 K/V defect | none | on (installed at load) | gemma-4-e2b 8k per-layer: first DKV layer 7.6% → 1.8% rel err (noise); 32k first token 2.57 → 0.96 before the harness fix; unit test against the math kernel | Makes K/V contiguous only when a mask is given and the head axis has stride 0; every other call untouched. Upstream fix in torch 2.14. Plain transformers chunk 2050 also broke and does not fit the L % 32 == 1 rule; unexplained |
| Fidelity harness applies the model's final logit softcap to the DKV arm | none | on | gemma-4 e2b / e4b first token 0.96 / 1.29 → 0.0014 / 0.0044 | Instrument fix; uncapped models unchanged |
| `benchmarks/diag_prefill_layer_diff.py` | `--chunk --dkv-chunk --unpatch --attn --sdpa-backend` | - | Found defects 2 and 3 | Per-decoder-layer last-position hidden state, plain transformers vs DKV |

**V14 RULER smoke test at 32k** (`benchmarks/campaigns/run_v14_ruler_panel.cmd`,
data newly generated per tokenizer with `generate_ruler_data.sh`, 1 item per
task; SmolLM3 all 13 tasks, the rest 4 tasks: `niah_multikey_2`,
`niah_multivalue`, `vt`, `qa_1`; paired against dense on shared items):

| model | DKV | dense | difference | tasks DKV lost |
|---|---|---|---|---|
| SmolLM3-3B (13 tasks) | 71.7 | 84.6 | -12.95 [-30.13, -1.54] | multikey_3, fwe, multivalue, cwe |
| Qwen2.5-7B | 63.8 | 95.0 | -31.25 [-75.00, +0.00] | multikey_2, multivalue (-25) |
| Qwen3.5-2B | 100.0 | 100.0 | 0 | - |
| gemma-4-e2b | 88.8 | 88.8 | 0 | - |
| gemma-4-e4b | 100.0 | 100.0 | 0 | - |
| gemma-4-12B | 75.0 | 100.0 | -25.00 [-75.00, +0.00] | multikey_2 |
| Ministral-3-8B | 75.0 | 100.0 | -25.00 [-75.00, +0.00] | multikey_2 |
| Phi-4-mini | 68.8 | 70.0 | -1.25 [-18.75, +15.00] | multivalue -25, vt +20 |

No architecture fails outright. Nothing was set per model, and every model ran
end to end. The losses are retrieval. `niah_multikey_2` (one key among many
near-duplicate distractor keys) was missed by DKV on 3 of 7 models and by
dense on none, the low-rank store's known weak spot (it trails dense by ~20
on RULER, V10). n = 4 per model is a smoke test, not an estimate: every
interval spans 0 except SmolLM3's. Phi-4-mini, despite its open decode
defect (second-token KL 0.68), is level with dense on these four items.

## The key-box router cost 22 points at 131k; residual router restored (2026-11-25)

V11 (below) measured the adaptive block count against fixed K on identical
code, and both sat far under V10 on the same 65 items. Paired against V10's
question-free arm (`V10P_..._dkv_mid_noquery_131k`, the like-for-like store,
since 6db9743a made residual rows question-free):

| V11 arm (same 65 items) | score | vs V10 noquery (70.5) |
|---|---|---|
| fixed K, key-box router | 48.5 | -22.00 [-31.38, -13.38] |
| adaptive K, key-box router | 57.3 | -13.21 [-20.64, -6.51] |

The run metadata differ only in the code revision. The wrong answers are
confident and garbled (`ugly-map: 100`, repeated or truncated numbers), not
refusals. Bisect on the two `niah_multiquery` items at 131k (V10 answers both),
all arms on the memory window so wall time matches V10:

| arm | answers | wall/item |
|---|---|---|
| V10-era code (6fb7e184), instrument check | 2/2 | 96-99 s |
| current code, memory window, key-box router | 0/2 | 99 s |
| current code, every 2026-11-24 decode switch off | 0/2 | 98 s |
| current code, `DKV_ROUTE_BOX=0` | **2/2** | 98 s |

The boxes are not mis-framed: `benchmarks/diag_key_box_frame.py` compares each
recorded box with the true post-RoPE box from a plain forward of the same
prompt (16k, streaming forced) and they agree to 0.2-0.7% of the box span on all
8 attention layers (unrotated keys would be off by 2-5%). The box is the right
bound and the wrong ranker: a per-channel min/max over ~1k-token blocks is
loose, which still covers most of the attention MASS (what the offline screen
and the 32k decode-step KL measured, on natural text) but loses the rare
needle tokens that the residual rows score exactly.

The token-budget elastic window is not the quality cause (the memory window
alone did not fix the bisect items), but it is why V11 took 137 s per item
against V10's 95 s: the newest 8,192 tokens exact means later prefill chunks
attend compressed history through the router. V12 p1 measures its quality.

| Change | Switch | Default | Verification | Notes |
|---|---|---|---|---|
| Residual router is the decode and prefill router again | `DKV_ROUTE_BOX=1` opts into the key box | **off** (was on since 6db9743a) | Bisect above; V12 | The prefill router's exact-box shortcut for compressed blocks is gated on the same switch, so the default prefill is V10's again |
| Adaptive block count runs on either router | `DKV_ROUTE_ADAPTIVE` as before | **on** past 32 candidates | Unit tests (the caller's top-k is the floor; no table / explicit K / small N / switch off fall back); V12 | `query_router._adaptive_select`: the residual router's top-16 (after edge propagation) is always routed and the key moments add blocks by estimated mass. Key boxes and moments are recorded whenever either needs them (`key_box.boxes_recorded`) |
| `benchmarks/diag_key_box_frame.py` | - | - | Ran (above) | Recorded box vs true post-RoPE box per block and layer |

**V12** (`benchmarks/campaigns/archive/run_v12_router.cmd`, `run_v12b_fixedk.cmd`,
`run_v12c_auto.cmd`, `run_v12d_repeatable.cmd`, `run_v12e_projected.cmd`;
Qwen3.5-4B, RULER 131k with the answer prefix, paired
bootstrap over shared items):

| arm | items | score | paired against | difference | wall/item (median) | decode peak |
|---|---|---|---|---|---|---|
| p1 residual router + adaptive, memory window | 39 | 70.6 | V10 noquery | -0.38 [-10.51, +10.26] | 91 s | 8.33 GB |
| p1 | 39 | | V11 adaptive (key box) | +14.87 [+5.64, +25.38] | | |
| p3 p1 with `DKV_ROUTE_ADAPTIVE=0` (fixed K=16) | 39 | 71.0 | V10 noquery | +0.00 (identical answers) | 89 s | 8.20 GB |
| p1 (adaptive) | 39 | | p3 (fixed K) | -0.38 [-10.51, +10.26] | +2 s | +0.13 GB |
| p2 same, token window (8192, as shipped by 6db9743a) | 65 | 63.4 | p1 (shared 39) | -7.18 [-17.05, +0.64] | 141 s | 5.77 GB |
| p2 | 65 | | V10 noquery | -7.05 [-13.77, -1.21] | | |
| p2 | 65 | | V11 adaptive (key box) | +6.15 [+2.31, +10.77] | | |
| p2 | 65 | | V11 fixed K (key box) | +14.95 [+6.41, +24.00] | | |
| p4 p1 with no policy set (card-derived `auto`) | 39 | 64.9 | p1 (memory window) | -5.77 [-13.46, +0.00] | 92 s | 8.62 GB |
| p4 | 39 | | p2 (token window, shared 39) | +1.41 [-3.85, +8.08] | | |
| p5 auto, slack 0.40, store so far | 39 | 69.4 | p1 (memory window) | -1.28 [-7.69, +3.08] | 94 s | 10.00 GB |
| p6 auto, slack 0.40, projected store (**shipped**) | 39 | 71.0 | p1 (memory window) | +0.38 [-7.31, +8.08] | 95 s | 9.65 GB |
| p6 | 39 | | p4 (auto, guard-driven) | +6.15 [+0.00, +14.36] | | |
| p6 | 39 | | p5 | +1.67 [-2.05, +7.44] | | |

p5 and p6 printed no memory-guard warning on any of the 39 items, so their
answers do not depend on what ran earlier in the process (p4 and p1 do, at
~125k). Wall is ~3 s/item above p1: the window now compresses earlier, at
roughly 85k tokens (budget arithmetic: 3.29 GB less a 0.52 GB projected store
at 32 KB/token; not traced), where the guard used to cut in at ~125k.

**p4 is not a policy difference.** On this card the auto budget is 5.49 GB
(12.88 x 0.94 - 3.12 resident - 3.5 reserve), a 167k-token window at Qwen3.5-4B's
32 KB/token, so it holds the whole 131k prompt exact; a `DKV_DBG_ELASTIC=1`
trace shows the only mid-prefill compression is the memory guard at ~125k
(reserved 8.6 GB + 3.5 reserve reaching 94% of the card), which the memory
policy shares. With adaptive off, auto and the memory window answered 5 of 6
`niah_multikey_2` / `niah_multiquery` items identically (the sixth is wrong
both ways). p4 differs from p1 on 3 items (2 multikey_2, 1 multiquery), the
same fragile items adaptive vs fixed moved in p1 vs p3.

**Answers depended on what ran earlier in the process (found 2026-11-25).** The
same config on `niah_multikey_2#0` answered `8143511` in p4 and `8143507` in a
fresh process. Not adaptive-specific and not randomness: fresh-process repeats
were bit-identical (64k: all 64 routing calls; 131k: every elastic decision),
and running `niah_multikey_1#0-2` first reproduced p4's six answers exactly.
The fixed-K arm moves the same way (`multikey_2#0` changed when run first). Cause:
the memory guard reads LIVE memory, and at ~125k (whole prompt held exact) the
card is at its line; after earlier requests allocated memory differed by 70 MB,
so the guard compressed layer 15's oldest blocks one chunk later
(`DKV_DBG_ELASTIC=1` traces, first difference at seq 125049) and every routing
input downstream differed (`DKV_DBG_ROUTE_DUMP`, new). The memory policy has the
same dependence. Fix below: the auto window is sized so the guard does not fire.

**Granite reach under auto** (granite-4.2-8b, `niah_single_1` at 114,688, the
longest generated granite length): budget 2.82 GB = a 17,241-token window at
164 KB/token, set with nothing configured; answered correctly, decode peak
10.46 GB, reserved at most 9.09 GB in prefill (under the 12.1 GB spill line),
398 s. With slack 0.40 (g2) the budget is 1.56 GB and the guard fired at 68,674
tokens: granite's compressed store (1.97 GB by then, 2.23 GB at the end)
outgrows the budget, and since the window was budget minus the store SO FAR it
closed only as memory ran out, so the guard cut in on the last few blocks
(`DKV_DBG_ELASTIC=1` trace: window 2,643 tokens, 2 blocks eligible, reserved
8.74 GB vs the 8.61 line). Sizing the window against the PROJECTED store (whole
prompt compressed) closes it from the start on granite (g3): no guard warning,
answered correctly, decode peak 10.46 GB, 332 s (was 384-398 s).

**Block size** (`colab/logit_fidelity.py`, Qwen3.5-4B nf4, 32k, 5 prompts;
table in `paper/results/campaign/v12.log`): decode-step KL vs dense 0.00180 at
1024 (default), 0.00037 at 512, 0.00035 at 256; top-1 5/5 everywhere. Not a
clean block-size effect: at 32k the default has 32 blocks per layer, so the
adaptive count does not engage (K=16, 16k tokens), while 512 and 256 engage it
(mean K 26 and 41). No default change: n=5 at 32k, and a block-size change
needs a 131k RULER pair (KMAX=64 blocks of 512 is half the tokens of 64 of 1024).

The router fix holds under either window. The 8192-token window costs ~7
points and ~50% wall time: every later prefill chunk attends compressed
history through the router, and the question at the end of the prompt is
encoded against it. Adaptive K chose mean 54.6 / median 60 blocks (p1) and
54.1 / 60 (p2) of a 64 ceiling.

p3 answers every item exactly as V10 did, so the current decode code (fast
attend, incremental refresh) reproduces V10 at fixed K. On the residual router
the adaptive block count measured no RULER effect at 131k: -0.38 [-10.51,
+10.26], moving single items both ways (multikey_2 +2 items, qa_1 / qa_2 / cwe
-1 each) for ~2% wall and 0.13 GB. Its +8.79 in V11 was compensation for the
key-box ranker. It stays on: it raises the routed share of attention mass
(0.79 -> 0.92 at 64k) at that small cost, and no measured harm; a RULER gain
from it is not shown.

| Change | Switch | Default | Verification | Notes |
|---|---|---|---|---|
| Auto window sized so the live-memory guard stays out of normal runs | `DKV_STREAM_AUTO_SLACK` (0.40) | on | Fresh vs after-history `multikey_2#0`: same answer, guard never fired (at slack 0.30; peak reserved 8.60 vs the 8.61 line, hence 0.40); unit tests; V12 p6 (39 items, no guard warning, 71.0, +0.38 vs memory window) and g3 (granite 114k, no warning, correct) | Budget = (94% card - reserved at load - 3.5 GB) x (1 - slack), minus the store the whole prompt will occupy once compressed (layers x ceil(prompt / block) x slot bytes, from the session's prompt length -- not an allocator reading, and fixed per prompt). Minus the store so far (p5) was not enough on granite, whose store outgrows the budget. The guard still protects against spill and warns only when it changes a decision, since then output depends on memory state. `DKV_DBG_ROUTE_DUMP=<path>` logs each routing call's inputs and selection for diffing runs |
| Exact-history budget read from the card once at load | `DKV_STREAM_ELASTIC_POLICY=auto\|tokens\|memory`; fixed budgets `DKV_STREAM_EXACT_TOKENS` / `DKV_STREAM_EXACT_GB` | **auto** (was tokens, 8192) | Unit tests (budget converted per model from live blocks; fixed budgets still win; roomy card holds all); V12 p4 + trace (same decisions as the memory window on this card) | Plug-and-play: nothing to set for any model. Budget = 94% of the card - resident memory after load - 3.5 GB reserve, recorded once (`record_exact_budget_at_load`), so co-resident load and other sessions no longer move it (the memory policy re-read it every chunk). Bytes per token are read from the blocks, so one budget gives a 4B hybrid ~5x the tokens of granite. Output differs between card sizes; `tokens` stays for reproducible runs |

## Adaptive block count and a faster decode read (2026-11-24)

Short tests only (unit tests, offline screens on captured KV, 32k logit
fidelity, paired decode timing); the long-context quality run is V11 below.

**Why K was stuck at 16.** The 2026-08-30 adaptive-K audit found K=16 to be a
cost knee: each routed block above 16 cost ~1.67 ms/token against ~0.3 below.
Profiling the materialise path found the cause in the attend, not the
attention: every step re-concatenated all routed rows with the dense window,
GQA-expanded them, and ran a masked memory-efficient SDPA, which does not split
along keys and so ran one thread block per KV head. Per layer, Qwen2.5-1.5B
shapes (2 KV x 6 query heads, D=128, 3.5k window), RTX 4070 SUPER:

| routed blocks | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| SDPA path (before) | 1.37 ms | 2.48 | 4.81 | 10.37 |
| `attend_fast` | 0.22 | 0.27 | 0.50 | 0.94 |

| Change | Switch | Default | Verification | Notes |
|---|---|---|---|---|
| Head-major remat entries + batched-matmul attend (`remat_cache.attend_fast`) | `DKV_FAST_ATTEND=0` restores SDPA | **on** | Output vs SDPA path: max diff 9.8e-4 (fp16 rounding), mean error vs an fp32 reference 1.04e-4 vs SDPA's 0.95e-4; 32k fidelity decode-step KL 0.00411 vs 0.00410; unit tests incl. padding garbage and a bf16 model over an fp16 pool (granite NF4, which the first version rejected) | Scores in fp32 per segment (KV head = batch, its query group = rows), one softmax, fp32 value accumulation; no K/V copy, no mask kernel. Deterministic run to run (cuBLAS bmm). Not used under CUDA-graph capture, dual-scale or the row trace |
| Incremental refresh: the block arena (`sparse_decode/block_arena.py`) | `DKV_REMAT_ARENA=0` | **on** | Unit tests: reuses kept blocks, rebuilds only new ones, rebuilds a rewritten slot, masks departed blocks, grows, declines duplicates | Per-slot write epochs on the pool (bumped where key boxes are invalidated, and all on `reset`) so a recycled slot is never read as its old content. Any failure falls back to the full rebuild |
| Grow-only adaptive block count (`srl/key_box.select_adaptive`) | `DKV_ROUTE_ADAPTIVE=0`; `_P` (0.80), `_KMAX` (64), `_MIN_BLOCKS` (2K = 32) | **on** past 32 candidate blocks | Offline (below); 32k logit fidelity (below); V11 at 131k: +8.79 [+1.82, +16.26] | (Since 2026-11-25 the floor is the residual router's top-16; top section.) The fixed router's box top-16 (after edge propagation) is always routed; blocks are added by estimated mass until the routed set holds P of it. Estimate: per-block mean and per-channel variance of the exact post-RoPE keys, recorded with the key box, clipped to the two bounds every block satisfies. Off when `DKV_TOPK_BLOCKS` is set. Gated to >32 candidates because at 32k growth bought nothing in real decode |
| Lazy routed-row RoPE tables | `DKV_LAZY_SLICED_ROPE=0` | **on** | granite 32k codes store: 0.336 GB -> 0 (below) | K x (1 + block) x D per layer (0.34 GB at granite 32k, growing with K) built only when a non-materialise kernel runs |
| One shared rotated-window scratch instead of one per layer | `DKV_DENSE_ROT_PER_LAYER=1` | **on** | granite 32k codes store: 0.294 GB -> 0.007 (below) | The unrotated-pool remat path rotates the dense window into a buffer every step; per-layer buffers held a second copy of every layer's window (0.29 GB on granite) |
| `benchmarks/route_adaptive_screen.py`, `colab/bench_k_ladder.py`, `colab/profile_decode_generate.py` | - | - | - | Offline routing screen on tier1 captures; in-process K ladder; generate()-loop profiler |

**Offline routing screen** (`route_adaptive_screen.py`, Qwen3.5-4B RULER
captures, exact keys, so routing error alone; mass = share of the true
attention mass the routed set plus window covers):

| rule | 32k mass | blocks | 64k mass | blocks |
|---|---|---|---|---|
| box top-16 (shipped) | 0.935 | 16 | 0.788 | 16 |
| box top-32 | 1.000 | 26.3 | 0.877 | 32 |
| oracle top-16 | 0.962 | 16 | 0.855 | 16 |
| estimate top-16 (diag. variance) | 0.955 | 16 | 0.837 | 16 |
| grow-only P=0.80 (default) | 0.950 | 16.9 | 0.924 | 32.9 |
| grow-only P=0.90 | 0.975 | 19.7 | 0.961 | 42.6 |

The moment estimate ranks blocks nearly as well as the oracle (8 principal
directions or the full covariance add ~1 point over the diagonal and need an
eigendecomposition per block, 217 ms per 16 blocks). At equal blocks the
estimate adds the right ones (0.924 vs 0.877 at ~32).

**32k real decode** (`colab/logit_fidelity.py`, Qwen2.5-1.5B fp16, 5 prompts,
dense_true self-check 0; prefill-step KL 0.00019 for every arm):

| arm | mean blocks | decode-step KL |
|---|---|---|
| fixed box top-16 | 16 | 0.00411 |
| grow-only P=0.80 (before the gate) | 17.8 | 0.00517 |
| estimate-ranked, P=0.80 | ~17 | 0.00521 |
| estimate-ranked, P=0.90 | 20.4 | 0.00553 |
| estimate-ranked, P=0.95 | 23.7 | 0.00561 |
| all 31 blocks | 31 | 0.00582 |

At 32k the routed 16 already hold most of the mass and each extra compressed
block adds its reconstruction error, so growth is gated to >32 candidates and
32k now routes exactly as before. Top-1 agreed 5/5 in every arm.

**Decode is host-bound.** Profiling generate() at 32k (Qwen2.5-1.5B): ~2,900
kernel launches per token, ~100 per layer; with `attend_fast` the routed
attention is ~5 ms of a ~95 ms token. The remaining cost of a larger K is the
refresh (every 4th token rebuilt every routed block), which the arena removes.

**Decode speed, in generate()** (`colab/bench_k_ladder.py`, Qwen2.5-1.5B 32k, 3
interleaved rounds, best token of each run; medians sit at ~90-100 ms on every
arm, set by host jitter):

| | fixed 16 blocks | all 31 blocks |
|---|---|---|
| before (SDPA, full refresh, eager tables) | 67.3 ms | 93.0 ms, peak 5.04 GB |
| after | 69.6 ms (no change resolvable) | 64.3 ms, peak 4.61-4.79 GB |

So routing all 31 blocks now costs what 16 did. At 16 the step is host-bound
and the attend was never the limit.

**granite codes store, 32k** (`benchmarks/diag_store_memory.py`, NF4, streaming):
allocated 10.45 -> 9.96 GB, peak 10.74 -> 10.21. The rotated-window buffers fell
0.294 -> 0.007 GB and the RoPE tables 0.336 -> 0. The memory-gated decode cache
takes part of the room back by design (0.81 -> 0.94 GB, more layers kept), so
the store's reach past 65k is checked directly in V11 p3.

**V11** (`benchmarks/campaigns/archive/run_v11_adaptive.cmd`): Qwen3.5-4B RULER at 131k,
5 items per task (65), adaptive vs fixed K on identical code, both on the
key-box router; granite codes store at 65k.

- Adaptive vs fixed: **+8.79 [+1.82, +16.26]** (57.3 vs 48.5). Gains on
  niah_multiquery +40, niah_multivalue +35, niah_single_2 +40, niah_single_3
  +20; qa_1 -20, fwe -6.7. Mean routed blocks 53.2 (median 59, p90 64, cap 64)
  out of ~120 candidates.
- Cost: none measurable. Wall 137 s/item in both arms, peak 5.80 GB in both.
- Both arms are far under V10 on the same items: the router regression in the
  section above.
- granite codes store at 65k: 3/3 correct, peak 11.17-11.32 GB (under the
  12.1 GB line), wall 134-163 s against the low-rank store's 127-162 s on the
  same items in V8 (no spill). The decode-buffer cuts give the codes store
  granite 65k; 131k is not measured, so it is still not the default.


## Review fixes (2026-11-24, code-first after the MLSys desk rejection)

Verified with short tests only (unit tests + one 32k logit-fidelity run); nothing here
has a long-context benchmark yet. `benchmarks/code_fingerprint.py` now also hashes
`query_router.py`, `key_box.py` and `residual_capture.py`, so stores written before
these changes cannot be resumed into.

| Change | Switch | Default | Verification | Notes |
|---|---|---|---|---|
| Residual rows ordered without the question (no span pin, no 64-token prompt tail) | `DKV_RESIDUAL_QUERY=1` opts in; legacy `DKV_RESIDUAL_NO_QUERY` still honoured | **off** (was on) | V9/V6 already measured it: granite LongBench 12k -0.25 [-1.20, +0.89], RULER 4k-16k +0.61 [-0.92, +2.17]; Qwen3.5-4B RULER 131k +0.17 [-2.33, +2.67]. Unit test of the switch | A compressed block's rows were picked for the first question and never redone, so later questions read a cache shaped by someone else's. `residual_capture.residual_query_enabled` is the one reader |
| Elastic window by token position, not free VRAM | `DKV_STREAM_ELASTIC_POLICY=tokens\|memory`, `DKV_STREAM_EXACT_TOKENS` (8192) | ~~**tokens** (was memory)~~ **auto since 2026-11-25** (V12: 8192 tokens cost 7 RULER points at 131k) | Unit tests: same blocks compressed regardless of `_elastic_room`; legacy policy still holds | The same prompt compressed different blocks on a 12 GB card vs 80 GB, or under different co-resident load. Memory now only forces EARLIER compression, with a one-time warning that output has become card-dependent. Pool *capacity* is still sized from free VRAM; it only limits how many slots exist, not which blocks are compressed |
| Router-owned key boxes at decode (Quest-style bound) | `DKV_ROUTE_BOX=0` restores the residual-key router | ~~on~~ **off since 2026-11-25** (-22 points at 131k, top section) | 32k logit fidelity, Qwen2.5-1.5B fp16, 5 prompts, self-check 0: decode-step KL box 0.00410, residual router 0.00510, all blocks 0.00579; prefill-step 0.00019 for all. Unit tests: bound never under-scores q.k; needle block ranks first; unboxed slots route first | Min/max of each block's EXACT post-RoPE keys per KV head, recorded by the GPU compressor before the store drops them (`native_core/srl/key_box.py`), invalidated on any pool write/free. Scoring reads 2 vectors per block instead of anchor + R residual rows, and no longer depends on which rows the store kept. The prefill router uses the same exact box for compressed blocks instead of residual rows. n=5 means "no regression", not a measured gain |
| Logit-fidelity arm `routebox0` | - | - | - | A/B arm for the row above |
| `benchmarks/diag_store_memory.py`: attributes allocated memory after a prefill to weights, pool fields, block tensors, decode workspace keys, module globals and unowned tensors (`--trace-shape` names the holder) | - | - | Ran on granite 32k, both stores (below) | Built for the codes-store spill |
| Repo tidy: campaign `.cmd` drivers to `benchmarks/campaigns/`; root notes to `docs/`; 36 one-off scripts nothing references (5 MLX benchmarks, 8 July A/B scripts, 4 probes, 11 colab hunts/notebooks, 8 native tools) to `archive/one_off_scripts_2026/` | - | - | `git grep` finds no reference to any moved script outside archives; unit tests pass | Moves only, nothing deleted; `keep_awake.py`, `run_v4_campaign.py`, `collect_results.py` stay since they are run by hand |

**Granite codes-store spill (V5): diagnosed, not a leak.** granite-4.2-8b NF4, 32k, streaming,
1-token generate, GB allocated above the 5.78 GB of weights:

| | codes4 | lowrank |
|---|---|---|
| store (pool) | 2.13 | 0.72 |
| dense recency buffers (fixed per layer, ~3.5k tokens) | 0.59 | 0.59* |
| rotated dense buffer | 0.29 | 0.29* |
| gathered RoPE tables (per layer, routed set, reused for 16 tokens) | 0.34 | 0.34* |
| decode remat cache (fp16 K/V of routed blocks; budget = free memory minus `DKV_REMAT_RESERVE_GB`) | 0.81 | - |
| exact active blocks | 0.33 | 0.33 |
| total | 4.67 | 2.06 |

(*lowrank's workspace was measured as one total, 0.95 GB.) Nothing is unowned: every
tensor has a holder. The codes store is 3x the low-rank store's bytes, the same as KIVI-4
plus anchors and scales, and DKV adds ~1.2 GB of fixed decode buffers plus a remat cache
that fills whatever headroom is left. Scaled to 65k, the store alone is ~4.3 GB, and with
weights and fixed buffers it sits at the 94% line before any prefill transient: that is
the spill, not a component. V5-F (remat cache held off) still spilling is consistent with
this. Fixing it means making the fixed decode buffers smaller or optional under the codes
store; not attempted here (it needs a 65k granite run to verify, which is long). The
exact-token default above is 8192, not 16384, for the same budget reason: 16384 would
hold ~2.7 GB of exact KV through a granite prefill.


Not changed, with the evidence that decided it:

| Review item | Why left |
|---|---|
| Keys compressed after RoPE | Already pre-RoPE: `mid`/`high`/`ultra` set `rotated_pool=False`; keys are re-rotated at their own positions on reconstruction (`DKV_EXACT_ROPE_REMAT`) |
| Separate K and V factors | Tried 2026-10-03 (`DKV_KV_SPLIT`, Batch A): 1.29-1.52x more error at equal bytes. Dropped |
| Oracle router / recall diagnostic | Done in Batch C: oracle-16 keeps 0.85 of attention mass at 64k vs 0.80 for the router; tier 2 found routing is not the bottleneck to 64k on either model. Past 64k the lever is K (V5-C: all blocks +51.67 at 131k), not a cleverer 16 |
| Per-KV-head selection | Batch C: per-head top-16 0.83 vs 0.80 mass. Needs per-head block sets in the read path and kernels; small measured gain |
| Codes store as the default | It wins on quality (V5-C/V10), but on granite it reaches less than half of low-rank's context on a 12.9 GB card (spill from 65k, diagnosed above; low-rank reaches 131k). Flip after the fixed decode buffers shrink under it. Select with `DKV_STORE=codes` |
| Block-size sweep (256/512 vs 1024) | A quality measurement at long context; needs long runs. Block size is `DKV_MICRO_BLOCK_SIZE` |
| Streaming prefill cost (routes 25% of history per chunk) | Sparse prefill routing is already on (`DKV_SPARSE_PREFILL`, now scoring compressed blocks by their exact box). A constant cap on routed blocks only changes anything past 128k, where checking prefill quality is a long run |
| Fixed-rank record buckets for vLLM | Planned with the vLLM integration, not before |
| ~~Adaptive K (top-p) as default~~ | Done differently, same day: grow-only adaptive block count with a mass estimate, plus a 6-12x faster attend and incremental refresh so extra blocks are cheap (section above). Top-p over box scores stays off |

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
  (`DKV_REMAT_RESERVE_GB`) is smaller than the 65k decode transient. **2.5 GB: 65k ok,
  11.37 alloc / 11.70 reserved, 383 s** (1.5 GB: spilled, 478 s; shipped tiled DKV 354 s).
  Default raised to 2.5 GB (affects the hybrid and the opt-in `DKV_REMAT_GATE`; the
  shipped path does not use this gate). The 3.5 GB point was cut by a runner time limit
  and is not needed.
- v2 elastic at 65k: **spilled badly** -- 12.74 alloc / 16.54 GB reserved, 2,095 s for one
  point (plain hybrid streaming 478 s). Its gate read allocated memory only (the remat
  gate's first mistake again). v3: allocated AND reserved must leave the headroom; when
  reserved alone blocks, free cached blocks are returned once and it is re-read.
  v3 at 65k: still spilled, 11.84 alloc / 12.60 reserved, 429 s (v2 2,095 s).
- Elastic headroom 3.5 GB (now the default): hybrid 65k ok, 11.37 / 11.70 GB, 191 s.
- Shipped store (granite): tier 2 KL 0.378 vs plain streaming 0.527, **paired -0.148
  [-0.241, -0.074]**, vs exact +0.065 [+0.009, +0.136] (the 3.5 GB headroom compresses
  earlier on this store, so it does not reach exact here as the hybrid did); 131k ok,
  10.87 / 11.26 GB, 499 s (416 s without elastic).
- **ON by default from 2026-10-04** (`DKV_STREAM_ELASTIC=0` disables). Streaming quality
  numbers measured before this date used plain streaming.

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

## Regression found by V3 part A: Qwen3.5-4B streaming spilled at 131k (fixed 2026-10-17)

Symptom: Qwen3.5-4B auto streaming at 131,072 peaked 18.58 GB alloc / 26.4 reserved
(paper ladder before the October defaults: 6.71 GB, reach 262k). Granite unaffected.

Diagnosis (instrument first):
- Bisect, Qwen forced streaming at 65k, one default off per arm: only elastic changes
  the peak (8.56 -> 5.04 GB); fused history attention and lean exact do not.
- Memory trace per chunk (DKV_DBG_ELASTIC) and per history step (DKV_DBG_MEM): elastic
  held ~113k raw tokens (~80-120 blocks per layer) because after-chunk memory showed
  room; the spike (8.2 -> 18.6 GB) came in the first chunk after the first in-forward
  compression. From then on those layers take the INCREMENTAL prefill branch
  (compressed history present), and that branch attended raw history EAGERLY: the
  whole [H, Q, raw tokens] score matrix plus a softmax copy plus repeat_kv'd K/V. With
  only the recency window raw (pre-elastic) it was small; with 113k raw it was ~10 GB.
  The lean exact fix (10-11) had tiled the other branch only.

Fix: the incremental branch uses the same tiled raw-history attention
(`_dense_history_attend_tiled`) under DKV_EXACT_LEAN (default on, CUDA). Same
arithmetic. Unit tests 27/27.

Check (benchmarks/campaigns/archive/run_inc_tiled_check.cmd): Qwen auto streaming 131,072 ok **10.10 GB**,
196,608 ok **10.37 GB** (were: spill at 131k, 18.58 GB). Elastic now borrows up to
~3.5 GB of headroom by design, so the peak sits above the pre-elastic 6.7 GB but under
the spill line.
- 262,144 then still spilled on RESERVED memory (12.97 GB reserved, 9.49 allocated). The
  boundary compresses everything elastic held in 64-block batches, ~3 GB of transients
  each, when memory is already high. Fix: `_boundary_batch_cap` -- at the prefill
  boundary, return cached blocks, and if under DKV_BOUNDARY_HEADROOM_GB (4.0) remains
  below the 94% line, compress 16 blocks per batch (~0.7 GB). Arithmetic unchanged.
  Re-run: **262,144 ok, 9.14 / 11.30 GB**, 484 s. Qwen streaming reaches the model's
  maximum again.
- Tier 2, granite streaming (elastic), fixed vs pre-fix: +0.010 [-0.048, +0.066] (no
  change; KL 0.389 vs 0.378, rounding from the tiled merge compounding over steps).

The V3 part A streaming ladders measured on the eager path moved to
paper/results/superseded/v3_inc_eager/ and are re-run; part A's LongBench/RULER rows
stand (outputs identical; only memory/time differed).

## Exact mode reach: DKV_EXACT_LEAN (2026-10-06, code only, NOT yet tested)

Why exact mode's ceiling sits below a preallocated dense cache (Qwen3.5-4B: 98,304 vs
122,880): not chunking (prefill is already 1,025-token chunks) and not KV growth (each
chunk's raw KV is one fixed tensor; boundary compression is per layer, synchronous, and
frees each block's raw KV). It is the exact-prefill history attention: every layer, every
chunk, it copies all raw history block by block, concatenates it, rotates it, and
repeat_kv's K and V to every query head -- a transient of several times one layer's
full KV. Plus the decode cache, which fills spare memory.

`DKV_EXACT_LEAN=1` (default off): `_dense_history_attend_tiled` in dkv_attention.py
assembles 8k-token tiles (`DKV_EXACT_TILE_TOKENS`) with one concatenation at KV-head
width, rotates per tile, folds the GQA group into the query axis, attends with the
efficient kernel and merges tiles by log-sum-exp; and turns on the decode-cache memory
gate (`DKV_REMAT_GATE`). Same arithmetic up to rounding. Expected: exact ceiling close
to the preallocated dense one; it cannot pass it (exact prefill must hold the raw KV).
Tests written (tests/test_exact_lean.py, 5) and benchmarks/campaigns/archive/run_exact_lean.cmd (unit
tests, tier 2 vs default exact, exact ladders granite 16k-28k, Qwen 98k-131k).

Results (2026-10-11):
- Unit tests 24/24 (5 new).
- Tier 2, granite, 374 steps, vs default exact: KL 0.3134 vs 0.3136, **paired -0.0001
  [-0.0057, +0.0045]** (identical up to rounding); peak 10.10 vs 10.72 GB; decode 6.2 vs
  7.0 tok/s (graph-free tool).
- Exact-mode reach:

| model | default exact | DKV_EXACT_LEAN | preallocated dense |
|---|---|---|---|
| granite-4.2-8b | 16,384 | **24,576** (11.37 / 11.68 GB); 28,672 spills | 20,480 |
| Qwen3.5-4B | 98,304 | **131,072** (11.13 / 11.75 GB), all of 98k-131k clean; slope predicts ~157k | 122,880 |

  Exact mode now reaches past a preallocated dense cache on both models.
- Ablation, tiled prefill alone (`DKV_REMAT_GATE=0`): tier 2 +0.0001 [-0.0055, +0.0048],
  decode **7.0 tok/s (no cost)**, Qwen 122,880 / 131,072 still clean (all of Qwen's gain),
  granite spills at 20,480. So the tiled prefill is free; granite's reach needs the
  decode-cache gate, which costs ~11% decode when memory is tight.
- **ON by default on CUDA from 2026-10-11** (`DKV_EXACT_LEAN` default auto; `=0` restores
  the previous path; `DKV_REMAT_GATE=0` keeps the tiled prefill but drops the gate, i.e.
  speed over granite's extra reach). Exact QUALITY results stand (outputs identical);
  exact reach, peak and timing are re-measured in V3 part A0.

## Store modes (2026-10-04)

`DKV_STORE=lowrank` (default) | `hybrid`. `hybrid` is shorthand for `DKV_KEY_QUANT=pc4` +
`DKV_RESID_ATTN=1` (an explicitly set switch wins) and works in exact and streaming mode
alike (runtime/native_block_pool.py `store_mode`). RULER / LongBench runners now record
DKV_STORE, DKV_KEY_QUANT, DKV_RESID_ATTN, DKV_STREAM_ELASTIC, DKV_PREFILL_SDPA and
DKV_REMAT_GATE (and since 2026-10-11 DKV_EXACT_LEAN) in their config when set (they recorded a fixed list, so a hybrid run
would have matched a low-rank store), and the code fingerprint now also hashes
streaming_sparse_ingest.py, lowrank.py and native_block_pool.py.

**Which old results stand** (after E1 and elastic went default-on):

| results | status |
|---|---|
| DKV exact: quality, peak, reach, decode speed | valid (nothing changed on that path) |
| DKV streaming: peak, reach | valid (equal or lower peak, 131k still reached) |
| DKV streaming: prefill time, s/1k | stale (E1: 2.4-2.9x faster) |
| DKV streaming: quality | stale for the new default (elastic: better); valid as "plain streaming" |
| KIVI-4 reach | stale (kivi4_chunked harness); use kivi4_tiled |
| KIVI-4 quality | re-run with kivi4_tiled for consistency |
| hybrid mode | none yet: LongBench, RULER, ladders in both modes, both models |

### V3 paper campaign (scripted 2026-10-04, not yet run)

`python benchmarks/run_v3_campaign.py [a] [b] [c1] [c2]`; logs in
paper/results/campaign/v3_*.log; every result in a new `V3_*` file, nothing old touched;
resumable. Estimates from the old runs' wall times.

| part | script | what | est. |
|---|---|---|---|
| A | run_v3_a_dkv_stream.cmd | A0 exact-mode ladders under DKV_EXACT_LEAN (+1 h); DKV streaming, new defaults: ladders (granite 24k-131k, Qwen 131k-262k), LongBench forced stream, RULER Qwen forced 32k, RULER beyond limits (granite 24k-32k, Qwen 131k) | 4-5 h |
| B | run_v3_b_kivi_tiled.cmd | kivi4_tiled: ladders both models, LongBench 12k, RULER granite 24k-32k, Qwen 32k and 131k | 6-7 h |
| C1 | run_v3_c_hybrid.cmd c1 | hybrid exact: ladders, LongBench 12k, RULER Qwen 4k-32k and 64k | 7-8 h |
| C2 | run_v3_c_hybrid.cmd c2 | hybrid streaming: the part-A set | 6-7 h |
| S | run_v3_s_systems.cmd | the MLSys systems table, systems-v2: TTFT, decode tok/s, peak alloc+reserved, KV bytes vs context; dense (defrag), dense static, SnapKV, KIVI-4 tiled, chunked StreamingLLM/H2O, DKV exact/auto, hybrid exact/auto; both models; median of 3 after warmup | 7-9 h |

Pre-flight for part C (2026-10-05): the hybrid had never run on Qwen3.5-4B (hybrid
architecture, head_dim 256) in the runtime. Smoke, 4 RULER 16k items, 32 steps: exact
hybrid KL **0.0006** vs base 0.297 (paired -0.297 [-0.612, -0.039]), top-1 1.000, decode
11.2 vs 2.7 tok/s, store 0.750 vs 0.595 GB. The streaming arms were not run (stopped:
GPU needed by another project); run them before part C2.

## Codes store (2026-10-21): KIVI's codes as a DKV store

`DKV_STORE=codes` [`DKV_QUANT_BITS=4|2`]: keys per-channel (groups of 32 tokens) and
values per-token (groups of 32 channels), both at `DKV_QUANT_BITS`, the tiled KIVI
baseline's layout, held in DKV's block pool and read by DKV's router, schedulers and
tiled read path. Individually: `DKV_KEY_QUANT=pc2|pc4|pc8`, `DKV_VALUE_QUANT=pt2|pt4|pt8`.
The factor is allocated at rank 1 and never read; no value residual is stored (positions
stay -1); key residual positions stay, for the router, read from the key codes. Only the
fused history tile reads value codes (the other prefill paths raise). Decode goes through
the materialise path, as the hybrid's does. Accounting (`dkv_kv_bytes`) counts the value
codes and scales. Tests: `ACTIVE_RUNTIME/tests/test_codes_store.py` (2-bit key codes,
value round trip, pool write/read/grow, reconstruct, fused tile vs reference attention).

T2, Qwen3.5-4B, 6 prompts / 65 forced steps (`paper/results/diag/fidelity_codes_qwen.log`):

| arm | KL | top-1 | KV GB | cmp | peak | dec/s |
|---|---|---|---|---|---|---|
| lowrank (default) | 0.7286 | 0.815 | 0.137 | 3.32 | 4.64 | 12.4 |
| hybrid | 0.3127 | 0.938 | 0.178 | 2.56 | 4.96 | 12.6 |
| codes, 4 bits | **0.0003** | **1.000** | 0.215 | 2.12 | 5.21 | 13.1 |
| codes, 2 bits | **0.0127** | **0.985** | 0.156 | 2.92 | 4.89 | 12.4 |

Paired vs lowrank: codes4 dKL -0.728 [-1.163, -0.377], codes2 -0.716 [-1.150, -0.366];
neither has a step with KL > 0.5 (lowrank: 16). V5 campaign: `benchmarks/campaigns/archive/run_v5_codes.cmd`.

V5 results (RULER on the paper's prompt, no answer prefix; 12-task averages, vt excluded):

| run | codes4 in DKV | paired vs lowrank DKV | paired vs tiled KIVI-4 |
|---|---|---|---|
| Qwen3.5-4B 131k, 5/task (p1) | 75.3 | +8.58 [+1.58, +16.83] | -11.42 [-20.75, -2.58]; on the 9 tasks other than multikey_2/3 and multiquery +2.00 [-2.22, +7.56] |
| granite 32k, 5/task (V5-B) | 56.1 | +8.50 [-0.83, +18.36] | -4.08 [-10.83, +2.58] |

Reach / decode (ladder, one call, 128 tokens): Qwen codes4 131k ok 8.88 / 11.43 GB,
decode 7.03 tok/s; 262k ok 10.35 / 11.58 GB, 7.18 tok/s (tiled KIVI-4: 3.80 and 2.16).
Granite codes4 **spilled at 98,304** (14.38 / 18.06 GB) where tiled KIVI-4 fits (10.66 /
11.50) with 0.6 GB fewer stored bytes; suspected cause: the fused history tile
dequantizes 64 blocks of key AND value codes in fp32 per layer call. V5-D re-runs the
granite ladder with `DKV_PREFILL_SDPA_TILE=16`. V5-C: codes4 attending every block at
131k on the three tasks where it trails tiled KIVI-4.

V5-C: attending every block with the same codes scores 100 on all three (15 items),
+51.67 [+28.33, +75.00] over top-16 and identical to tiled KIVI-4: at 131k the gap is
the router. Granite codes4 reach, 128-token answers, every run spills at 65,536 near
12.2 GB allocated where tiled KIVI-4 fits (9.11): 16-block tile (V5-D; first token
133 s, decode 0.35 tok/s), fixed window (V5-E; 131 s, 0.53), decode cache held off
(V5-F, DKV_REMAT_RESERVE_GB=30; 192 s, 2.86). Not the tile, the scheduler or the decode
cache. Unverified lead: the codes store still runs the low-rank encoder's full
factorization and residual pass (only rank 1 is kept), whose fp32 temporaries the
low-rank store's smaller pool leaves room for. Open.

## Fairness controls (V6, 2026-10-21)

`benchmarks/campaigns/archive/run_v6_fairness.cmd`. p1: a 16-bit cache read through DKV's tiled,
LSE-merged history attention (`dense_tiled`), ladders, 128-token answers.

| model | dense_tiled clean to | spills at | peak there (alloc / reserved, prefill) | streaming DKV past it | preallocated dense |
|---|---|---|---|---|---|
| granite-4.2-8b | 28,672 | 32,768 (12.12 GB reserved) | 10.97 / 11.44 | 4.6x (131,072) | 20,480 |
| Qwen3.5-4B | 196,608 | 229,376 (12.13 GB reserved) | 10.36 / 10.95 | 1.3x (262,144) | 122,880 |

The read path alone lifts dense 1.4x (granite) and 1.6x (Qwen) past preallocated dense;
DKV's store adds 4.6x and 1.3x. The paper's earlier bound ("at most 5.3x and 2.0x",
from exact DKV's ceiling) overstated the hybrid: on Qwen3.5-4B, where 8 of 32 layers
cache, most of DKV's reach is the read path's; on granite, where all 40 do, most is the
store's.

p2/p3, RULER 131k on Qwen3.5-4B, 5 items per task, the V3_BEYOND items (60 paired):

| arm | RULER | KV GB | paired |
|---|---|---|---|
| DKV/mid as shipped (V3_BEYOND) | 66.7 | 0.52 | ref. |
| DKV/mid, residuals ordered with no question information (`DKV_RESIDUAL_NO_QUERY=1`) | 66.9 | 0.52 | +0.17 [-2.33, +2.67] vs shipped |
| chunk-wise StreamingLLM, 23,808 tokens (V4S) | 29.4 | 0.78 | |
| chunk-wise SnapKV ranking told the question (`snapkv_chunked_q`), 23,808 tokens | 75.9 | 0.78 | +46.56 [+32.81, +59.67] vs StreamingLLM; +9.25 [+3.19, +16.61] vs DKV; -0.67 [-4.86, +4.19] vs DKV codes4 |

DKV's lead over eviction does not come from seeing the question (no change without
it). Eviction recovers once its ranking knows the question before the prompt is read,
and then what it keeps serves that question alone; DKV's records are built without it.

V7 (`benchmarks/campaigns/archive/run_v7_fill.cmd`), the dense reference past dense's preallocated
ceiling and DKV on granite from 4k, RULER on the paper's prompt:

| run | RULER | paired |
|---|---|---|
| dense on DKV's tiled read path, Qwen3.5-4B 131k, 5/task | 87.1 | DKV/mid streaming -20.42 [-31.67, -9.83]; codes4 -11.83 [-21.08, -3.08]; tiled KIVI-4 -0.42 [-1.25, +0.00]; question-aware eviction -11.17 [-21.17, -1.94] |
| dense on DKV's tiled read path, granite 24k, 10/task | 66.6 | DKV/mid streaming -16.21 [-24.46, -8.22]; tiled KIVI-4 +1.53 [-2.50, +5.83] |
| DKV/mid on granite 4k / 8k / 16k, 20/task | 72.5 / 58.4 / 56.6 | vs dense (72.7 / 73.1 / 70.9) -9.75 [-12.11, -7.48] |

Tiled KIVI-4 is at dense quality at long context; the low-rank store pays 16-20
points there for the fewest bytes.

V8 (`benchmarks/campaigns/archive/run_v8_fill.cmd`, 2026-10-26), granite between 32k and 114k and its
eviction arms at dense's lengths (RULER items generated at 65,536, 5/task):

| run | RULER | paired |
|---|---|---|
| DKV/mid streaming, granite 65k, 5/task | 41.7 | tiled KIVI-4 - DKV +3.61 [-7.08, +14.44] |
| tiled KIVI-4, granite 65k, 5/task | 45.3 | |
| SnapKV, granite 4k / 8k / 16k, 10/task | 69.4 / 63.8 / 55.1 | vs dense -9.70 [-13.04, -6.60]; DKV/mid - SnapKV -1.12 [-5.00, +2.85] |
| StreamingLLM chunk-wise, granite 4k / 8k / 16k, 10/task | 40.9 / 32.3 / 18.0 | vs dense -42.06 [-47.49, -36.46]; DKV/mid +31.24 [+25.90, +36.50] |
| H2O chunk-wise, granite 4k / 8k / 16k, 10/task | 37.6 / 21.1 / 19.0 | vs dense -46.56 [-51.82, -41.14]; DKV/mid +35.74 [+30.42, +41.01] |

On granite every cache that fits loses ground past 32k (DKV 49.0 -> 41.7 -> 28.2 at
32k / 65k / 114k; tiled KIVI-4 63.1 -> 45.3), and within dense's limit DKV and SnapKV
tie at about 9.7 points below dense.

V9 (`benchmarks/campaigns/archive/run_v9_noquery.cmd`, 2026-10-26), DKV's question look-ahead
(`DKV_RESIDUAL_NO_QUERY=1`) off on granite, current code:

| run | score | paired (off - on) |
|---|---|---|
| LongBench 12k, look-ahead on (control, 120 items) | 38.74 | vs the paper's 2026-10 store -0.07 [-0.86, +0.57] |
| LongBench 12k, look-ahead off | 38.48 | -0.25 [-1.20, +0.89] |
| RULER 4k / 8k / 16k, look-ahead off, 10/task | 72.0 / 57.7 / 57.0 | +0.61 [-0.92, +2.17] vs V7 (360 items) |

With Qwen3.5-4B at 131k (+0.17 [-2.33, +2.67]), turning the look-ahead off moves
no measured score. V10 (`benchmarks/campaigns/archive/run_v10_prefix.cmd`) re-runs the past-the-limit
comparisons with RULER's answer prefix at 10 items per task.

V10 (2026-10-27, three power cuts, resumed by `run_v10_prefix.cmd resume`): RULER
with the answer prefix, all 13 tasks (vt now scores 98-100), 10 items per task,
130 per arm, same items for every arm.

| arm | Qwen3.5-4B 131k | vs dense, tiled | granite 24k | vs dense, tiled |
|---|---|---|---|---|
| dense on DKV's tiled read path | 93.5 | ref. | 81.6 | ref. |
| tiled KIVI-4 | 94.2 | +0.69 [-0.23, +2.31] | 82.6 | +0.97 [-0.78, +2.85] |
| DKV, codes store (4-bit) | 87.2 | -6.27 [-11.27, -1.62] | | |
| SnapKV chunk-wise, told the question, 1.5x bytes | 77.7 | -15.77 [-22.54, -9.62] | | |
| DKV/mid, low-rank store, streaming | 73.3 | -20.19 [-27.69, -13.23] | 63.7 | -17.88 [-25.14, -11.04] |
| DKV/mid, no look-ahead | 72.5 | -21.00 [-28.62, -13.77] | | |
| StreamingLLM chunk-wise, 1.5x bytes | 32.7 | -60.86 [-68.20, -53.50] | | |
| StreamingLLM chunk-wise, 2k | 12.7 | -80.78 [-87.21, -73.76] | | |

Every comparison resolved on the old prompt (12 tasks, 5 per task) keeps its sign
and stays resolved. Two change resolution: the codes store now beats question-told
eviction, +9.50 [+4.12, +15.54] (was -0.67 ns), and DKV without look-ahead is no
longer resolved below it, -5.23 [-10.54, +0.19] (was -9.08). The low-rank store's
loss stays in mk2/mk3 (0 and 0 vs 100) and cwe (56 vs 99).

Loader fix found while adding V10: `paperdata._correct_dkv_bytes` corrected every
DKV row to int8 residual width, including rows written after 1b3773c9 that already
counted it (both carry `kv_acct: 2`), so V6-V10 low-rank stores read low (V9
LongBench 0.335 GB, true 0.427). The loader now skips campaigns run after the fix,
and the harness writes `kv_acct: 3`, which the loader also skips. No paper number
used the affected bytes.

Paper-side finding (2026-10-26, not a code change): dense on DKV's tiled read path
decodes faster than streaming DKV where it fits (Qwen3.5-4B ladder: 10.34 vs 7.00
tok/s at 131k, 7.76 vs 7.11 at 196k); the router's decode gain is over dequantizing
codes in PyTorch, not over 16-bit attention.

Paper audit (2026-10-25, not a code change): the `mid` preset exports
`DKV_SVD_ENERGY=0.9999` at config load (`config.py`, setdefault), so every run kept
99.99% of a block's energy (rank capped at 32), not the 99.9% default in `lowrank.py`
that the paper had quoted; the ladder rows record 0.9999.

## Batch E: prefill speed and baseline fairness (2026-10-04)

**E2 larger prefill chunks -- no code.** `DKV_PREFILL_CHUNK_SIZE` already sets it
(rounded up to whole blocks: 2048 -> 2050 = 2 blocks per chunk). Measurement queued
(benchmarks/campaigns/archive/run_e2_chunks.cmd): tier 2 at 2048/4096 exact and streaming, and 32k prefill
time. An older single-seed note in config.py links chunk 2048 to a lower score; tier 2
decides.
- Result (granite hybrid, 374 steps; paired vs chunk 1024): exact 2k +0.026 [-0.002,
  +0.060], 4k +0.010 [-0.009, +0.033]; streaming 2k **+0.039 [+0.012, +0.072]**, 4k +0.011
  [-0.014, +0.038]. 32k streaming wall: 1024 110.4 s, 2048 97.6 s (-12%), 4096 spilled
  (12.28 GB reserved). **Closed: chunk stays 1024.** The speed is small, the quality cost is
  real for 2k, and 4k breaks reach; E1 gives 2x without either.

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
- Result (granite hybrid streaming): tier 2 KL 0.2253 vs 0.2193, paired +0.006 [-0.005,
  +0.017] (no difference); 32k ladder point **52.6 s vs 110.4 s wall (2.1x), 1.60 vs 3.37
  s/1k**, peak 11.56 GB both.
- Shipped store (granite, streaming auto at 16k, as the shipped ceiling ladder): tier 2
  +0.006 [-0.004, +0.017] vs plain streaming; **65k 149 s vs 354 s (2.4x), 131k 416 s vs
  1,226 s (2.9x)**, peaks 9.36 / 10.86 GB vs 9.36 / 11.00. Same reach.
- **ON by default on CUDA from 2026-10-04** (`DKV_PREFILL_SDPA` default `auto` = on when
  the queries are on CUDA; `=0` restores the previous path). Paper speed numbers measured
  before this date used the previous path.

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
- Result (granite): 65k ok, peak 9.11 GB (prefill 9.47), 155 s; 98k ok, peak 10.66
  (prefill 11.06), 255 s. kivi4_chunked spilled at 65k (12.96). So KIVI-4's reach on
  granite is at least 98k once tiled (shipped DKV tiled streaming: 131k at 11.00). RULER
  24k, 1 item per task (13): tiled 62.8 vs chunked 59.0, decode 3.5 vs 2.5 tok/s, peak 7.15
  vs 8.47 GB; the score gap is within one item per task and in the expected direction
  (own-chunk attention in 16 bits). The paper's KIVI reach numbers must use kivi4_tiled;
  131k point to run with the paper ladders.

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
| 2026-10-22 | **Cloud campaign Y1** (`benchmarks/cloud/`, runbook `docs/CLOUD_CAMPAIGN.md`): 1,123 jobs (140 smoke configs + 983 real, ~734 GPU-h estimated; every series paperdata.py reads has a cloud twin with the recipe its .meta.json recorded, plus dense past 65k, 262k, Qwen3.5-9B, concurrency) across many GPU machines, coordinated only by the orphan branch `cloud-results` (claim = atomic push). Each worker runs a logit-fidelity preflight per model, pushes outputs every 10 min, and resumes a dead machine's job from its pushed rows. RULER data is generated per machine and hash-checked against the paper's items. Quality DKV arms set `DKV_STREAM_ELASTIC_RESERVE_GB` so the elastic window's line falls at the 12.878 GB card's 8.605 GB (`dkv_native` keeps the big card's own line). `bench_concurrency_cuda.py --timeout` (was a hardcoded 2 h per arm); `decode_fidelity.py` reads RULER items from `DKV_RULER_DATA` when set | The sponsored GPUs answer the limitations section: n of 5-10 per task beyond the exact limit, no dense reference past 65k, no datacenter-GPU systems numbers, no serving concurrency, two models. Without the reserve, DKV's elastic window on 80 GB would keep nearly everything exact and measure something other than the paper's method. Regenerated Qwen3.5-4B 4k data was byte-identical to the paper's. tests/test_cloud_queue.py (9, CPU) |
| 2026-10-21 | **Fairness controls (V6, `benchmarks/campaigns/archive/run_v6_fairness.cmd`)**: `dense_tiled` arm (a 16-bit cache held in chunks, read by the tiled LSE-merged attention tiled KIVI uses: the dense ceiling with no history-sized transient); `snapkv_chunked_q` arm (chunk-wise eviction scored by the post-RoPE queries of the prompt's last 64 tokens, captured once at their true positions: the same look-ahead DKV's residual ordering has); `DKV_RESIDUAL_NO_QUERY=1` (residual runs ordered with no question information, neither the extracted span nor the tail) | Blind review: the reach ratios used a dense baseline that still had a history-sized transient, and DKV's store sees the question while the eviction arms do not. tests/test_dense_tiled.py (3, equals plain attention, 16-bit bytes); CPU check of `snapkv_chunked_q`: holds every layer at the budget, equals dense when nothing is evicted |
| 2026-10-21 | **KV accounting v3: residual rows at their stored width** (`dkv_kv_bytes`) | It charged every residual row at 16 bits (K+V; V for the hybrid) while the pool stores them int8 with an fp16 scale and offset per 64 channels (`DKVConfig.residual_quant` default int8): 4,096 vs 2,176 B per row on granite, 4,096 vs 2,176 on Qwen3.5-4B. Every recorded DKV byte count was high: LongBench DKV/mid 0.525 -> 0.431 GB, Qwen3.5-4B 131k 0.772 -> 0.524, granite 114k 3.32 -> 2.24; store growth on granite 0.029 -> 0.019 GB per 1k tokens. Recorded totals rebuild exactly from each row's own fields, so the paper's loader (`paperdata._correct_dkv_bytes`) corrects old rows exactly. Found by a blind reviewer recomputing the 676 KB record (true: 436 KB at the full 128-row budget). tests/test_kv_accounting.py gains an int8 case |
| 2026-10-20 | **RULER answer prefix attached** (`run_ruler_cuda.build_ruler_prompt`; switch `RULER_ANSWER_PREFIX`, default 1, 0 restores the old prompt; stores record `answer_prefix`, so old and new rows cannot merge) | RULER's generator stores the answer prefix BESIDE the input, and the harness only re-attached it when the input ended with it, so no task ever got one. Models opened with a preamble ("Based on the text provided, here is...") that spent variable tracking's 30-token budget: vt scored ~0 for every arm, dense included. With the prefix, dense answers vt exactly (pilot, Qwen3.5-4B 4k). Every RULER store before 2026-10-20 is on the old prompt; the V4 campaign re-runs all of them. Paired deltas between old stores stay valid (same prompt for every arm) but absolute scores and averages were depressed |
| 2026-10-20 | New baseline options: `kivi2_tiled` arm (KIVI-2 as published: per-channel keys in groups of 32, per-token values, 128-token 16-bit window, tiled attention like `kivi4_tiled`; 4-bit path bit-identical); dense `{"offload": true}` (exact KV in host memory, transformers' offloaded DynamicCache); SnapKV/H2O post-hoc `{"defrag": true}` (the allocator fix dense gets) | Review asked for a properly configured 2-bit baseline, an offloading baseline, and SnapKV given the same allocator treatment as dense. Smoke-tested on both models (`run_v4_z_smoke.cmd`): correct answers, 2-bit packing round-trips |
| 2026-10-20 | `DKV_TOPK_BLOCKS` recorded in RULER store configs when set | The attend-all routing ablation (`DKV_TOPK_BLOCKS=0`) would otherwise have the same config as a routed run |
| 2026-10-17 | **KV accounting v2** (`dkv_kv_bytes`): count only layers that hold KV, each with its own block/residual counts; rows carry `kv_acct: 2`, `kv_layers` | It multiplied layer 0's counts by every decoder layer: on Qwen3.5-4B (8 of 32 layers hold KV) every DKV absolute GB, store and dense-equivalent, was 4x high (32k dense-equiv 4.29 vs the true 1.07 GB the baselines report), and layer 0 there holds no blocks, so its counts were fallbacks. Ratios were right. Granite (all layers attention) unaffected. tests/test_kv_accounting.py (3, CPU) |
| 2026-10-17 | **systems-v2** (`bench_systems_cuda.py`): TTFT and decode rate from inside ONE call per repeat (DKV: per-token timestamps after the sampler's sync, `hf_dkv_wrapper.last_timing`; baselines: run_baseline's own prefill/decode split); peaks max over repeats, allocated AND reserved, spill on either; every DKV_* switch in the config; arm stops after a spill; per-point timeout | v1 differenced two separate calls, decode/s = K / (T(1+K) - T(1)), folding one prefill's jitter into decode (at 131k ~100 s prefill vs ~6 s decode); spill detection read allocated only; configs had no DKV switches, so a hybrid or streaming run could merge into a low-rank store |
| 2026-10-17 | Context ladder records `ttft_s` / `decode_tps` for DKV rows (same timestamps) | DKV ladder rows had only wall time; prefill and decode were not separable |
| 2026-10-03 | Decode-fidelity disables CUDA graphs in DKV arms | The forcing hook cannot be captured; replays returned stale buffers (KL 4-15 on prompts under `DKV_GRAPH_MAX_CTX`=8192) |
| 2026-10-03 | Tier-1 base uses the runtime's residual rule | Runtime ranks joint ABSOLUTE error with one index set for K and V; the first screen used relative error, overstating attention weighting's gain |
| 2026-10-03 | Tier-1 tools read RoPE / scale / q-k norms from the live model | No per-architecture code |
| 2026-10-04 | Probe / hybrid row selection moved after BOTH residual branches | With a residual budget >= block size every block takes the store-everything (`force_exact`) branch; logic placed only in the ranking branch never ran, so the first K/V probe read KL 0.0008 for both sides (both fully exact). The smoke test's pool (64 slots) never reaches that branch, which is why it passed. Verified in the real run: `force_exact=True K_rows=1024 V_rows=128` |
| 2026-10-04 | decode_fidelity reuses finished arms | Power cut mid-run; a rerun now skips arms whose output is newer than the dense control |

## Open items found along the way

- Paper says the factorization is per head; the code factors all KV heads of a layer jointly (`feat_dim = 2*H*D`). Correct the paper.
- On short contexts CUDA uses adaptive (smaller) blocks; the paper's "1024-token blocks, 1,536-token window" describes long contexts only. Verify and state.
- ~~Residual selection falls back to the last 64 prompt tokens as a query (`DKV_RESIDUAL_QUERY_TAIL`); within equal information, but not strictly query-agnostic.~~ Resolved 2026-11-24: residual ordering is question-free by default (`DKV_RESIDUAL_QUERY=1` opts back in); see "Review fixes" below.
