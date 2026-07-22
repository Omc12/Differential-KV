# CUDA ↔ MLX end-to-end trace diff

**Method:** walk the pipeline input → output, one stage at a time, and record every
behavioural difference between `serving/mlx_dkv_wrapper.py` (MLX, the working
reference) and the CUDA path (`serving/hf_dkv_wrapper.py` → `runtime/dkv_attention.py`
→ `native_core/*` → `runtime/native_block_pool.py`), including options that are on/off.

**Status legend:** 🟢 aligned · 🟡 differs, understood/deliberate · 🔴 differs, likely a real gap
· ⚫ CUDA-only dead weight (MLX has no equivalent)

---

## 1. Model load / preset resolution

| | MLX | CUDA | |
|---|---|---|---|
| preset → quantization | low → auto int4 | low → auto NF4 | 🟢 |
| CAD (relational accuracy) | high/quality/max → `DKV_CAD_ALPHA=0.5`, `CAD_MAX_STEPS=32` | **was never auto-set** → now aligned (hf_dkv_wrapper) | 🟢 *(fixed)* |
| allocator/cache cap | `DKV_CACHE_LIMIT_GB=1` (`mx.set_cache_limit`) | `PYTORCH_CUDA_ALLOC_CONF` (expandable_segments via eval) | 🟡 different mechanism |

## 2. Session init / block pool

| | MLX | CUDA | |
|---|---|---|---|
| store layout | `comp_U, comp_VK, comp_VV, comp_anc_k/v, comp_min_k/max_k, comp_scale, comp_seq_len, comp_res_k/v, comp_res_n, comp_res_mask` | `U(int8)+U_scale, V_KV, anchors_KV, scales, seq_lens, desc, residual_K/V_{positions,values}` | 🟡 equivalent core |
| U storage | fp16 `comp_U` | **int8 + per-block scale** | 🟡 CUDA is smaller here |
| **dead tensors** | — none — | ⚫ `U_sem`, `U_sem_scale`, `U_fact`, `n_semantic`, `fact_anchors_K`, `fact_anchors_V`, `fact_anchor_positions` — allocated per slot but **never written by the GPU compress path** (only the CPU path at kv_runtime_manager:1730/2971 writes them). ~42 KB/slot ≈ **11% of every pool slot**. `fact_*` are still handed to the decode kernel each token → `HAS_FACT=True` → kernel loops 3 dead slots per block per layer per token. | ⚫ **actionable** |
| dominant slot cost | `comp_VK+comp_VV` ≈ 192 KB | `V_KV` ≈ 192 KB | 🟢 same |

## 3. Prefill — scheduling

| | MLX | CUDA | |
|---|---|---|---|
| chunk size | 512 | preset `prefill_chunk_size`, rounded to block capacity (1028/2056) | 🟡 |
| **compression schedule** | **streams per chunk** in `generate()` (compress_deferred_prefill_blocks + `mx.eval` + `mx.clear_cache` after EVERY chunk) → raw KV bounded ≈ recency window | **defers ALL to the boundary** (hf_dkv_wrapper:891; eval loop) → **holds the whole prompt's raw KV**, then builds the pool on top | 🔴 **root cause of peak VRAM ≈ dense_peak + pool at every context** |
| per-chunk cache release | `mx.clear_cache()` | none (opt-in only) | 🔴 |

## 4. Prefill — attention

| | MLX | CUDA | |
|---|---|---|---|
| formulation | `_sparse_prefill_attend`: **sinks + top-K routed history + recency window + self** (DSA/NSA), `DKV_SPARSE_PREFILL=1` default | **exact decomposed**: `_flash_local_attention` (local chunk) + `_project_then_attend_history` over **ALL** history | 🔴 **CUDA has NO sparse prefill.** `runtime/sparse_prefill.py` (`RetrievalAwareSparsePrefill`) is imported only by its own test — never wired. Explains fwd 8.5s vs dense 5.9s. |
| tuning knobs | `SPARSE_PREFILL_{FRAC,KMIN,MIN,WINDOW,SINK_BLOCKS}` | none | 🔴 |

## 5. Compression

| | MLX | CUDA | |
|---|---|---|---|
| batch granularity | ONE batched SVD over `num_layers × num_blocks` | **per layer** (48 dispatches) | 🟡 cross-layer was tried → **regressed VRAM 15.07→17.16 GB, reverted** (natural per-layer batch is only ~49; the chunk cap acted as a floor). Cap now 64 = bounds only 64k+ layers. |
| rSVD params | oversamples 5, n_iter 2, seeded | oversamples 5, n_iter 2 | 🟢 |
| **V-scale** | `sqrt(eK/eV)` on V before joint K\|V SVD, unscale factor after | **was absent** → now ported (`DKV_V_SCALE=1`), V recon err 0.97→0.56 | 🟢 *(fixed)* |
| rank | **fixed** `self.rank` | **dynamic**: 99.9%-energy truncation + 1.5× content boost | 🔴 real divergence; changes stored rank/quality per block |
| per-block scale | `scales` = max-abs, recon `× scale` | `block.scale = 1.0` (folded into U) | 🟡 equivalent-ish |
| **residual selection** | **JOINT**: `joint_errors = sqrt(eK² + (eV·v_gain)²)` → **one** top-K → same positions for K and V | **SEPARATE**: independent `top_k_K` and `top_k_V` → **K and V may capture different tokens** | 🔴 real divergence |
| **residual representation** | stores the **actual K/V value**; `res_mask` drops the low-rank twin so the token is attended once, exactly | stores **`delta − recon`**; kernel **adds** it back onto the recon | 🟡 mathematically equivalent, different layout |
| adaptive residual budget | tiers 8/16/full by median err | same tiers | 🟢 |
| finalization | batched over whole batch | **was 2,352 per-block iters** → now batched recon/errors/medians + batched pool write (bit-identical) | 🟢 *(fixed)* |
| int4 stratified pack | n/a | ⚫ was computed per block then **never written** → removed | 🟢 *(fixed)* |

## 6. Prefill → decode boundary

| | MLX | CUDA | |
|---|---|---|---|
| drop raw prefill cache | yes (`_prefill_caches.pop` + `clear_cache` + gc) when compressed decode | frees `active_k` per block during compress | 🟡 |
| lego state release | frees ring/sink buffers | lego unwired on CUDA | 🟡 |

## 7. Decode

| | MLX | CUDA | |
|---|---|---|---|
| engagement | `DKV_COMPRESSED_DECODE=1` → **sparse from token 1** (accepts 16 vs 36 tps @4k, "always exercise DKV") | dense **bypass below `ENGAGE_THRESHOLD=4096`** | 🟡 CUDA is *faster* short-ctx — deliberate |
| **routing fire** | prunes to **top-16** whenever `nb > topk_blocks(16)` (+ sinks + recency) | only when `nb > srl_threshold(50)` → **at 13k (49 blocks) NEVER fires → attends ALL 49** | 🔴 **~3× the sparse work; the tps gap** |
| attention form | **materialise routed K/V once per interval + cache**, then one masked SDPA | **PTA in Triton**: scores straight from low-rank factors, no materialisation | 🟡 different strategy; PTA is cheaper per-token in theory (R≪D) — unmeasured which wins |
| decode cache var | `DKV_DECODE_CACHE` (works) | ⚫ **`DKV_DECODE_CACHE` is read NOWHERE on CUDA** (only in `decode_config.py`'s defaults dict). CUDA's real one is `DKV_DECODE_CACHE_ENABLED` → the gathered-KV workspace cache in the *PyTorch fallback*, off for accuracy (Issue-6 stale tensors) | 🔴 the "~2× tps default" is a **no-op on CUDA** |
| sparse bias | applied inside the unified SDPA | `SPARSE_BIAS=auto` **disables the fast combined Triton path** (gate requires bias∈{0,0.0,off}) | 🔴 perf/accuracy tension unique to CUDA |
| fact anchors | n/a | ⚫ dead (`HAS_FACT=True` over all -1 slots) | ⚫ |
| routed-row gather | n/a (materialises) | already routed-only + `data_ptr`-keyed cache | 🟢 |
| dense window | dense tail | incremental (`dirty`-gated) fixed workspace + cached RoPE | 🟢 |

## 8. Output / sampling

| | MLX | CUDA | |
|---|---|---|---|
| CAD prior stream | short question-only session per token, capped | same PyTorch port (hf:932) | 🟢 |
| sampler | on-device | vocab-scale mask allocs when factual features on | 🟡 factual off by default |

---

## Ranked actionable diffs

1. 🔴 **Streaming compression** (§3) — the only fix for peak VRAM ≈ dense+pool. *But* streaming currently OOMs at 16k because the prefill history-attention over compressed blocks allocates more than the raw KV it frees. Needs §4 fixed first.
2. 🔴 **Sparse prefill** (§4) — CUDA has none; explains fwd 8.5s vs 5.9s. Also the precondition that makes streaming viable (bounded history attention).
3. 🔴 **Routing threshold** (§7) — CUDA attends all 49 blocks at 13k vs MLX's 16. One-env-var A/B (`DKV_SRL_THRESHOLD=16`); accuracy-gated.
4. ⚫ **Dead pool tensors + dead fact-anchor kernel loop** (§2/§7) — **measured and DEFERRED.** Real (~42 KB/slot ≈ 11% of a slot) but that is only ~110 MB at 13k = **<1% of the ~15 GB peak**, while `n_semantic`/`U_sem`/`U_fact` are indexed by eight block-property getters (streaming_sparse_ingest 277/292/309/327, kv_runtime_manager 242/257/274/292) that `getattr(b, "U_sem_int4", None)` triggers. Not worth a crash risk for <1%. The allocation gate + write guards are in place (`_needs_legacy_slots`, forced True); flip it only after guarding all eight readers.
5. 🔴 **Dynamic vs fixed rank**, **separate vs joint residual selection** (§5) — quality divergences; need NIAH to judge.
6. 🔴 **`DKV_DECODE_CACHE` is a no-op on CUDA** (§7) — the documented "2× tps" default does nothing here.
</content>
