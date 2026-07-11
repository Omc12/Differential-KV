# Native (C++) port plan: pool right-sizing + lego streaming prefill

Status: PLAN (2026-07-11). The MLX reference implementation landed in
`ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` (`DIFFKV_LEGO_PREFILL`); this doc
maps it onto `diffkv_native/src/main.cpp`. Read the MLX flag comments first —
all fidelity findings (studs-only default, knife-edge margins, synthesis
sensitivity) were measured there and apply here.

## Why (measured on MLX, same structure here)

Native prefill holds, per layer, BOTH a device raw cache and a host mirror for
the FULL prompt, plus the compressed pool:

- `persistent_k_cache[l]` / `persistent_v_cache[l]` — F16 device tensors `[head_dim, kv_heads, L]` (main.cpp ~2860).
- `k_rotated_activations[l]` / `v_activations[l]` — F16 host vectors `[L * F]` (main.cpp ~2577-2583).

At 16k on Qwen2.5-1.5B that is ~940 MB of raw KV duplication on unified memory.
The MLX lego mode cut peak process memory 4.38→3.29 GB @16k by bounding raw KV
to O(sinks + ring + chunk) and attending the far field via compressed pieces.

## Mapping

| MLX piece | Native equivalent | Status |
|---|---|---|
| per-chunk flush (`compress_deferred_prefill_blocks`) | streaming ingest (`IngestManager`, prefill ingest thread) | EXISTS |
| block materialisation (decode route math) | `materialize_routed_kv()` (main.cpp ~5032) | EXISTS — needs a studs-only variant (anchor + residual rows, skip low-rank recon) |
| residual router | anchor+residual scoring in the decode router; prefill has host-side lexical routing (`sp_block_tokens`) | EXISTS (lexical is fine for prefill — no readback needed) |
| sparse ranges in the prefill graph | `build_prefill_ctx_graph(..., sp_ranges)` | EXISTS — needs a second source (materialised far rows) besides the persistent cache |
| raw recency ring | replace full-[L] `persistent_*_cache` with ring-sized tensors + rolling offset | NEW |
| sinks raw copy | first `sp_sink_blocks*micro_block_size` rows pinned in the ring buffers | NEW (trivial) |
| engage gate | `sp_min` + "compressed far blocks exist" + total-prompt-will-decode-sparse | NEW — NOTE: below the decode memory-budget gate (~64k) slots stay dense-resident, so lego-prefill requires EAGER compression + raw-drop for far blocks, i.e. it overrides the budget gate for prefill state. This is the main design decision. |

## Steps

1. **Studs-only materialiser**: `materialize_routed_studs(engine, cands, ...)`
   writing only [anchor + residual rows] per candidate block (reuse the
   residual-extraction inside `materialize_routed_kv`; content-guard skip
   (cands+pool_ver) applies unchanged).
2. **Ring buffers**: allocate `persistent_*_cache` at
   `sink_end + DIFFKV_LEGO_RING + chunk` rows; keep `ring_start` per prefill
   (block-aligned trims). Host mirrors `k_rotated_activations`/`v_activations`
   shrink the same way — everything the ingest thread needs beyond the ring is
   already in the pool. Prefix-reuse decompression (cached_len path) reads
   host mirrors — gate lego OFF when `cached_len > 0` (mirrors MLX
   stream-start-0 guard).
3. **Graph**: pass the materialised far rows as an extra prior-KV range
   (a second device tensor pair, `[far_rows]`), concatenated before the ring
   ranges in `sp_ranges` handling. Mask: far rows fully visible (like other
   history), chunk causal (already the case).
4. **Decode boundary**: the pool is already the decode source; the ring's last
   `window` rows seed the decode dense window (today that comes from the full
   persistent cache — same bytes).
5. **Gates (MANDATORY per repo protocol)**: native NIAH 6-cell sweep + margins
   (`benchmarks/native_margin_probe.sh`), multi-needle, synthesis
   (`benchmarks/synthesis_eval.py --engine native`), conformance vs MLX.
   Expect the 16k/0.1-style knife-edge: judge on margins + content, not a
   single exact-match cell.

## CUDA

The torch/CUDA path (`hf_diffkv_wrapper` + Triton kernels) cannot be tested on
this Mac. Port ONLY behind default-OFF env gates and add cert items to
`CUDA_TRITON_AUDIT.md`'s GPU checklist. The MLX session-pool growth
(`_ensure_block_capacity`) has a direct torch analogue if the HF wrapper's
session store pre-allocates (check `DiffKVHFWrapper` before porting — on Mac it
aliases to MLX, so the torch store is only exercised on CUDA machines).
