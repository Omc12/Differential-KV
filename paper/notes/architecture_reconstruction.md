# Architecture Reconstruction — DiffKV Active Runtime (MLX)

> Reconstructed directly from the implementation that produces the measured benchmark
> numbers. Source of truth: `ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` and the
> measured outputs in `benchmarks/results/`. Where docs disagree with code, code wins.

## 0. What "Active Runtime" actually is (provenance)

- The benchmark worker `benchmarks/bench_worker.py::run_active` instantiates
  `DiffKVHFWrapper` from `ACTIVE_RUNTIME/serving/hf_diffkv_wrapper.py`.
- On macOS with MLX present, `hf_diffkv_wrapper.py:1220` rebinds
  `DiffKVHFWrapper = MLXDiffKVWrapper` (`serving/mlx_diffkv_wrapper.py`).
  Confirmed log line in every active run: `"[DiffKV] macOS + MLX detected: using
  native MLX DiffKV wrapper."`
- **Therefore the measured ACTIVE runtime = the MLX wrapper.** The PyTorch/HF +
  Triton path (`runtime/diffkv_attention.py`, `native_core/streaming_sparse_ingest.py`,
  `native_core/sparse_decode/triton_fused_decode.py`, `runtime/native_block_pool.py`)
  is the *portable backend / original design*, described in `docs/runtime_architecture.md`,
  but it is NOT the engine behind the numbers. The paper centers on the MLX path and
  references the PyTorch backend as an alternative implementation.
- **SRL / factual-store relational-binding subsystem is GATED OFF in these benchmarks.**
  `MLXKVBlockManager.get_srl_state()` returns `None` and `_session_srl` is never
  populated (`mlx_diffkv_wrapper.py:352,354`). Every factual-store / VSL / logit-bias
  block in the decode and generate loops is guarded by `srl_state is not None`, so it is
  inert during the NIAH long-context runs. Needle recovery is purely from the core
  compressed+dense attention. The SRL module is described as an optional, separately
  evaluated capability — not part of the measured memory/throughput results.

## 1. Model under test

Qwen2.5-1.5B-Instruct, MLX int4 (`mlx-community/Qwen2.5-1.5B-Instruct-4bit`,
group_size=64, bits=4). Dimensions (from model config):

| symbol | meaning | value |
|---|---|---|
| L | transformer layers | 28 |
| H | attention (query) heads | 12 |
| H_kv | key/value heads (GQA) | 2 |
| d | head dimension | 128 |
| d_model | hidden size | 1536 |
| G = H/H_kv | GQA group size | 6 |
| V | vocab | 151936 |
| — | rope_theta | 1e6 |
| — | trained context | 32768 |

## 2. DiffKV hyper-parameters (as instantiated)

From `MLXDiffKVWrapper.__init__` / `MLXKVBlockManager.__init__`, with the bench config
`{"quantization":"int4","rank":16,"block_size":256,"micro_block_size":256,"preset":"mid"}`:

| symbol | meaning | value | source |
|---|---|---|---|
| B | block size (tokens per compressed block) | 256 | config |
| r | SVD target rank | 16 | config |
| W | dense recency window | 512 | `recency_window` default |
| D_max | dense buffer capacity | W + B = 768 | `max_dense_len` |
| M | max compressed blocks | 256 | `DIFFKV_MAX_BLOCKS` |
| — | max compressed tokens | M·B = 65536 | derived |
| — | KV store dtype | float16 | `_create_empty_session` |
| — | prefill chunk | 512 | `generate()` / worker |

Capacity envelope: M·B + D_max = 65536 + 768 ≈ 66k compressed-addressable tokens,
which is exactly why the active runtime reaches the 64k (65,615-token) NIAH cell.

## 3. Per-session memory layout (`_create_empty_session`)

Per layer ℓ ∈ [0,L), the session holds two regions:

**(a) Dense recency window** (uncompressed, sliding):
- `dense_keys[ℓ]`   : [1, H_kv, D_max, d]  fp16
- `dense_values[ℓ]` : [1, H_kv, D_max, d]  fp16
- `dense_lens[ℓ]`   : int (current fill)

**(b) Compressed block pool** (pre-allocated, fixed M):
- `comp_U[ℓ]`      : [M, B-1, r]   fp16   (per-token low-rank coefficients)
- `comp_VK[ℓ]`     : [M, H_kv, r, d] fp16 (K reconstruction basis)
- `comp_VV[ℓ]`     : [M, H_kv, r, d] fp16 (V reconstruction basis)
- `comp_anc_k[ℓ]`  : [M, H_kv, d]   fp16 (anchor key, token 0 of each block)
- `comp_anc_v[ℓ]`  : [M, H_kv, d]   fp16 (anchor value)
- `comp_scale[ℓ]`  : [M]            fp32 (SVD normalization scale)
- `comp_seq_len[ℓ]`: [M]            int32 (valid tokens per block)
- `num_blocks[ℓ]`  : int

## 4. Compression (`_compress_block` + `compress_mlx_block`)

Per block of B contiguous tokens in the dense buffer:
1. anchor = token 0 (per kv-head): a_k = K[:,0,:], a_v = V[:,0,:]
2. deltas: ΔK = K[:,1:,:] − a_k, ΔV = V[:,1:,:] − a_v   (shape [H_kv, B-1, d])
3. flatten+concat across K and V into X ∈ R^{(B-1) × 2·H_kv·d}
4. per-token (row) L2 normalize: n_i = ||X_i||, X̂_i = X_i / n_i
5. randomized truncated SVD of X̂ (NumPy/CPU): X̂ ≈ U_k Σ_k V_kᵀ, with
   - scale s = max|X̂| (global), X̂ ← X̂/s
   - oversampled range finder (r+5 cols), 2 power iterations, QR, projected SVD
   - adaptive rank: keep k = smallest components covering 99.9% energy, clamped [4, r]
   - U ← (U_k·Σ_k) in fp16, V ← V_kᵀ in fp16
6. re-apply per-token norms: U ← U ⊙ n   (so reconstruction recovers the true delta scale)
7. zero-pad U to [B-1, r], V to [r, 2·H_kv·d]; split V into V_K | V_V; reshape to
   [H_kv, r, d]; store U, V_K, V_V, anchors, scale s, seq_len.

Reconstruction (implicit, never materialized as dense KV):
  ΔK̂_i ≈ s · (U_i · V_K),  K̂_i ≈ a_k + ΔK̂_i   (and similarly for V).

Notes: joint K/V SVD shares one U (token-coefficient matrix) across K and V bases —
halves the per-token coefficient storage. No exact-token residual path and no int4
packing of U in the MLX path (the PyTorch `lowrank.py` backend adds both).

## 5. Memory complexity (per layer, per block)

Dense full-KV block (baseline): B·H_kv·d·2(K,V)·2 bytes = 256·2·128·2·2 = 262,144 B (256 KiB).

Compressed block:
- U:        (B-1)·r·2          = 255·16·2      = 8,160 B
- V_K,V_V:  2·H_kv·r·d·2       = 2·2·16·128·2  = 16,384 B
- anchors:  2·H_kv·d·2         = 2·2·128·2     = 1,024 B
- scalars:  scale(4)+seq_len(4)= 8 B
- total ≈ 25,576 B (≈ 25 KiB)

Per-block compression ratio ≈ 262,144 / 25,576 ≈ **10.25×** (for KV state, fp16 vs fp16).
Asymptotic state per token: O(r·d) basis amortized over B + O(r) coefficient
→ dominated by the (B-1)·r coefficient term, i.e. O(r) per token vs O(H_kv·d) dense
(rank 16 vs 256 effective → the source of the flat memory slope).

## 6. Prefill path (`attention_forward`, L>1 branch + `MLXQwenModel.__call__`)

1. Chunked prefill (chunk = 512 tokens). Each chunk is a forward pass through the
   patched model with a *native MLX KVCache* (`make_prompt_cache`) so every chunk
   attends over all prior tokens → exact causal hidden states (no approximation in
   prefill). This is `mx.fast.scaled_dot_product_attention` over the growing cache.
2. After SDPA, the chunk's rotated K and raw V are captured token-by-token into the
   DiffKV dense buffer (`capture_prefill_kv`); when the dense buffer overflows D_max,
   the oldest B tokens are compressed out (`_flush_oldest_block`) — streaming compression.
3. `compress_deferred_prefill_blocks` is a no-op hook in the MLX path (compression is
   inline during capture).

## 7. Prefill→Decode boundary (`MLXQwenModel.__call__`, decode branch)

At the first decode step the runtime:
- `mx.eval()` to flush lazy ops, then `mx.clear_cache()` + `gc.collect()` to return the
  peak GQA-expanded prefill activations to the OS;
- drops the native prefill KVCache entirely (`_prefill_caches.pop`) when compressed
  decode is enabled, so decode-time memory reflects only the DiffKV store (compressed
  pool + dense window), not a retained full-context cache. This is the mechanism behind
  the active runtime's low, flat decode-time footprint.

## 8. Decode path (`attention_forward`, L==1 branch)

Per decode step, per layer:
1. Ingest current token's (rotated K, V) into the dense buffer (`ingest_streaming`),
   then flush+compress any block that pushes dense fill past W+B
   (`_compress_eligible_blocks`). The current token is ingested BEFORE scoring so the
   query can attend to itself.
2. `execute_decode_attention` → `compute_decode_attention_static` (a single
   `@mx.compile`'d kernel) computes the fused compressed+dense attention.

### Fused decode attention math (`compute_decode_attention_static`)
For query q ∈ R^{H×d} (GQA-expanded to match H_kv groups):

Sparse / compressed branch (no KV decompression):
- anchor scores: s_anc = (q · a_k)·scale            [per block]
- project query into each block's K-basis: q̃ = (q · V_K)·scale  ∈ R^r
- per-token delta scores: δs = (q̃ · Uᵀ)·s + s_anc   (reconstructs scores directly in
  low-rank space — the key trick: score reconstruction costs O(r·B), never O(B·d))
- mask invalid tokens (seq_len) and invalid blocks (num_blocks) to −∞
- per-block softmax over [anchor, δ_1..δ_{B-1}], collected across all blocks
- weighted value: O_anc = Σ w_block · a_v ; O_delta = Σ (w_d·U)·s · V_V
- block log-sum-exp `lse_sparse`

Dense branch: exact attention over the dense window (recency tokens).
- `lse_dense`, `out_dense`

Flash-style merge: combine the two branches by their log-sum-exps
  w = softmax([lse_sparse, lse_dense]); out = w_s·out_sparse + w_d·out_dense.
Extensive NaN/inf sanitization guards a fully-masked branch (e.g. zero compressed
blocks) from poisoning the merge — documented as a real bug fix in the source.

`@mx.compile` fuses the whole decode attention into one Metal graph; the Python layer
only orchestrates dispatch.

## 9. Generation / sampling (`MLXDiffKVWrapper.generate`)

Greedy (benchmark) or temperature/top-p sampling, repetition penalty with a
loop-detection escalation (n-gram repeat ratio ≥ 0.35 → widen window, raise penalty;
40 tokens unrecovered → force-stop). All factual-bias / VSL logic is present but inert
when `srl_state is None` (benchmark case). Benchmark decode is pure greedy
(`np.argmax`) for 128 fixed tokens, EOS ignored, for comparable TPS.

## 10. Serving stack (context, not benchmarked here)

`openai_compatible_api_gateway.py` (FastAPI, OpenAI-compatible) →
`batch_engine.py` (continuous-batching decode loop) →
`production_session_manager.py` (multi-session lifecycle, LRU residency) →
wrapper/manager. Session ops in the manager: init/clear/clone/snapshot/restore/
rollback — a full KV-cache lifecycle for multi-turn serving.

## 11. Knobs (env) that matter for the paper

- `DIFFKV_MAX_BLOCKS` (256) → caps compressed tokens / VRAM.
- `DIFFKV_ENGAGE_THRESHOLD` → overrides recency window W.
- `DIFFKV_COMPRESSED_DECODE` (default 1) → real sparse decode; 0 = exact full-KV (debug).
- Presets low/mid/high (`native_core/config.py`) tune chunk size, kv_quant, dense budget.
</content>
