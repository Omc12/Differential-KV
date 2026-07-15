# CUDA vs. MLX Performance Audit — ACTIVE_RUNTIME

**Date:** 2026-07-15  
**Scope:** `ACTIVE_RUNTIME` serving path: MLX (`serving/mlx_diffkv_wrapper.py`) compared with the Linux/CUDA PyTorch/Triton path (`serving/hf_diffkv_wrapper.py` → `runtime/diffkv_attention.py` → `native_core/*`).  The separate `diffkv_native/` binary is out of the primary runtime path; its CUDA code is noted only where it affects a claimed CUDA optimization.

## Executive verdict

The CUDA implementation is not a GPU-equivalent port of the working MLX path.  It has a capable Triton decode kernel, but the end-to-end CUDA runtime is dominated by CPU-offloaded compression, blocking host/device transfers, Python-side routing/factual work, and a CUDA-graph mechanism that cannot safely represent the mutable KV/routing state.  Several GPU-native components exist in the tree but are not wired into the live Python serving path.

The largest explanation for high CPU usage is confirmed by control flow, not inferred from profiling: normal CUDA prefill copies every full KV block to CPU, synchronizes the transfer, and runs randomized SVD with CPU `torch.linalg` workers.  GPU compression is an opt-in environment switch defaulting to off.

No CUDA device was available in the audit environment, so findings marked **static** require GPU profiling/functional certification before estimating exact milliseconds or changing production behavior.  The source-level execution paths and defaults were inspected directly.

## Architecture comparison

| Area | MLX path | CUDA/PyTorch path | Performance consequence |
|---|---|---|---|
| Runtime ownership | Purpose-built MLX manager and kernels in one wrapper | HF model monkey patch + manager + streamer + Python router + Triton | CUDA has substantially more host orchestration per token. |
| Compression default | MLX block compression stays in MLX operations | Full CUDA blocks are staged to CPU; CPU rSVD is the default | PCIe/NVLink transfer, CPU saturation, and delayed pool publication. |
| Factual store | Explicitly gated by `DIFFKV_FACTUAL_STORE` | Built and queried whenever capture exists; no matching gate | CUDA pays an undocumented, heavy feature cost by default. |
| Sparse decode | Dedicated MLX static decode path | Triton kernel plus Python preparation, dense RoPE materialization, and fallbacks | Triton launch is only a fraction of CUDA decode cost. |
| Scheduling | Single wrapper owns session state | Python mutation on every layer/request; attempted CUDA graph replay | Dynamic state prevents the proposed static graph from being correct. |

## Findings

### P0-1 — Default CUDA compression is CPU SVD, not CUDA SVD (**confirmed**)

**Evidence.** `streaming_sparse_ingest.py:1249-1258` uses the GPU compression path only when `DIFFKV_GPU_COMPRESS=1`; its default is `"0"`.  Otherwise `:1276-1311` concatenates GPU block K/V, copies them into CPU staging buffers, records an event, then immediately calls `event.synchronize()` at `:1284-1288`.  The worker is given CPU slices.  `lowrank.py:159-160` explicitly moves non-CPU input to CPU; `:195-209` and `:872-936` run QR/SVD through CPU PyTorch operations.  The default config enables async SVD for CUDA (`config.py:47-59`).

**Impact.** Every completed block across every layer requires D2H traffic and CPU randomized SVD.  The immediate event synchronization means the transfer is not overlapped at this handoff.  Two Python workers can also contend with tokenizer/SRL/factual processing and CPU thread pools.  This directly explains high CPU utilization and can make long-prefill throughput CPU-bound even while CUDA attention is underutilized.

**MLX contrast.** MLX compression is implemented with MLX arrays/rSVD in `mlx_diffkv_wrapper.py:558-657`; it does not route normal blocks through a separate discrete-memory CPU staging pipeline.

**Priority.** First performance blocker.  Measure D2H time, CPU SVD time, and GPU idle gaps before any kernel micro-optimization.

### P0-2 — The GPU-compression escape hatch is not production-safe as written (**confirmed**)

**Evidence.** The only live GPU-compression selection is the disabled `DIFFKV_GPU_COMPRESS` branch above.  Its `compress_layer_blocks_gpu()` implementation still has host scalar conversions and token processing (`lowrank.py:462+`, especially `:509`, `:566-577`, `:642-698`).  The standard `compress_lowrank_batch()` still unconditionally copies non-CPU tensors to CPU at `lowrank.py:883-886`, so it is not a GPU batch-SVD fallback.

The native extension advertises a cuSOLVER compressor, but no active Python code constructs `DiffKVCompressorThread`; repository call sites are only bindings/tests.  More importantly, its `process_job()` is a schematic stub: the actual cuSOLVER calls are commented out in `native_core/diffkv_core/src/compressor_thread.cpp:70-80`.

**Impact.** Simply enabling the switch does not establish a complete GPU-resident compression pipeline and can exchange a known CPU bottleneck for an unvalidated, potentially synchronization-heavy path.  The native extension cannot presently solve the issue without wiring and completing it.

### P0-3 — CUDA always retains and concatenates CPU copies of full prefill K/V; factual-store opt-in is broken (**confirmed**)

**Evidence.** `KVRuntimeManager.capture_prefill_kv()` copies K/V to CPU for every layer and grows each buffer with `torch.cat` (`kv_runtime_manager.py:1037-1045`) before also streaming it.  This is repeated for chunks, creating O(number-of-chunks²) copy/allocation behavior per layer for the retained capture.  `finalize_srl_index()` always builds `FactualExactStore` when that capture exists (`:964-988`).  Decode then enters factual-store logic whenever the store exists (`diffkv_attention.py:755-1086`).  There is no `DIFFKV_FACTUAL_STORE` check anywhere in the CUDA/PyTorch path, while the README says the feature is opt-in/default-off and the MLX manager explicitly gates it (`mlx_diffkv_wrapper.py:1731-1733`).

**Impact.** At default settings CUDA retains a complete host KV duplicate and pays factual-store construction plus layer-0 Python query work.  This increases RAM, D2H volume, prefill latency, and CPU load; it also invalidates the documentation's claimed default behavior.  At long contexts this may be as material as the CPU SVD issue.

**Priority.** First remediation group with P0-1; determine whether factual capture is needed for the request before retaining it.

### P1-4 — “Async” staging synchronizes before enqueue and adds several extra copies (**confirmed**)

**Evidence.** The ingest path makes GPU concatenation tensors (`streaming_sparse_ingest.py:1275-1277`), copies them to pinned CPU buffers (`:1281-1282`), synchronizes the event (`:1284-1288`), clones each exact CPU slice (`:1299-1300`), pins those clones again (`:1303-1304`), and submits them to the worker.  The same CPU buffer is not reused by the worker because the clone is needed for lifetime safety.

**Impact.** This defeats the claimed compute/compression overlap at the transfer boundary and creates allocation/copy pressure in addition to the SVD itself.  Calling it asynchronous hides CPU compute after the handoff, but not the handoff latency.

### P1-5 — CUDA graph replay is incompatible with the live mutable decode path (**confirmed design defect; GPU behavior untested**)

**Evidence.** `CUDAGraphDecodeRunner.capture()` performs three normal model forwards and one captured forward (`static_decode_graph.py:91-111`).  The attention patch mutates Python/session state during each decode forward: it calls `kv_manager.ingest_streaming()` for each layer/request (`diffkv_attention.py:521-529`), performs routing, assembles dense workspaces, and changes caches/metadata.  Graph replay (`static_decode_graph.py:127-134`) executes no Python state updates.  The runner invalidates only on explicit prefill, not when a decode token changes block state/routing/dense layout.  Batch mode also changes `model._diffkv_session_ids` each step (`batch_engine.py:1392-1396`) while graph identity is keyed only by input shapes.

The one-shot wrapper has an additional interface failure: graph replay returns `_GraphOutput`, which has only `.logits` (`static_decode_graph.py:145-159`), then accesses `outputs.past_key_values` (`hf_diffkv_wrapper.py:1316-1318`).

**Impact.** A graph that captures successfully is stale/incorrect after replay, and normal failures are swallowed by broad `except` blocks in the wrapper/engine.  In practice this makes the graph path unreliable or silently eager, so it cannot be credited as a CUDA launch-overhead optimization.

**Priority.** Disable from performance claims until it is redesigned around a truly static, device-resident decode ABI with explicit state buffers and graph invalidation/versioning.

### P1-6 — Continuous batching is not fused at the DiffKV attention level (**confirmed**)

**Evidence.** The batch engine builds a bucketed batch (`batch_engine.py:1366-1396`), but patched attention loops over batch elements (`diffkv_attention.py:521-548`) and invokes sparse decode once per element (`:1636-1672`).  `native_triton_sparse_attn_decode()` asserts batch size one (`triton_fused_decode.py:1882-1883`); the combined kernel has the same assertion (`:2384-2385`).

**Impact.** A batch of B active requests means roughly B × layers sparse dispatch/preparation operations, rather than a batched decode kernel.  Bucket padding adds model work but does not turn DiffKV attention into a batched kernel.  This caps serving throughput and worsens CPU launch/orchestration overhead as concurrency rises.

### P1-7 — CUDA decode still does Python/GPU preparation and allocations per layer/token (**confirmed**)

**Evidence.** The combined CUDA path builds a fresh position tensor and fills it in a Python loop over dense blocks (`diffkv_attention.py:1613-1622`), creates a full `torch.cat` to rotate the dense workspace (`:1630-1634`), and invokes the kernel once per batch item/layer.  The sparse-only fallback performs dense attention and LSE merge in PyTorch (`triton_fused_decode.py:1982-2078`).  The manager also walks/copies dense blocks into workspaces per layer (`kv_runtime_manager.py:1945-2014`).

**Impact.** Even where Triton is live, the hot path launches non-fused PyTorch ops and allocates index/temporary tensors.  At short and medium contexts, this host/launch/bandwidth work can dominate the sparse kernel.

### P1-8 — SRL routing causes recurrent D2H synchronizations and Python graph work (**confirmed when SRL activates**)

**Evidence.** SRL is enabled by default (`kv_runtime_manager.py:925-927`) and activates above the configured block threshold.  The router pulls GPU scalars/lists to Python: entropy `.item()` (`query_router.py:196-204`), centroid `.item()`/`.tolist()` (`:359-367`), semantic score vector `.cpu()` (`:537-550`), and many CPU list/dictionary graph operations (`:570-635`).  It is invoked at layer 0 on each decode step (`diffkv_attention.py:587-638`).  Factual routing additionally calls `block_indices.tolist()` at `diffkv_attention.py:768` and traverses entry graphs in Python.

**Impact.** With discrete CUDA memory, each conversion is a synchronization point; transferring all semantic scores to CPU is O(number of compressed blocks) host traffic per token.  This is materially unlike unified-memory MLX and can erase savings from routing when the context is large.

### P1-9 — Stratified-U compatibility path introduces per-layer D2H key generation and whole-pool traffic on cache misses (**confirmed when stratified storage is populated**)

**Evidence.** Normal low-rank compression always sets `n_semantic >= 1` (`lowrank.py:254-265`), so the stratified path is normally enabled.  `_build_stratified_U_for_triton()` creates `active_key = tuple(sorted(active_idx.tolist()))` (`triton_fused_decode.py:989-997`) before cache lookup; on CUDA this synchronizes.  On a cache miss it dequantizes/clones `U` for the entire pool (`:1010-1015`) even though only active rows are used.

**Impact.** The gather optimization avoids whole-pool cloning for K/V inputs, but U remains O(pool size) on a stratified cache miss and the `tolist()` lookup occurs in every layer's dispatcher.  Pool writes invalidate the generation cache, so the cost recurs whenever blocks complete compression.

### P2-10 — Silent Triton fallback makes observed CUDA performance ambiguous (**confirmed**)

**Evidence.** Both sparse dispatchers catch arbitrary exceptions and return the PyTorch vectorized decoder unless `DIFFKV_TRITON_STRICT=1` (`triton_fused_decode.py:2087-2103`, `:2522-2534`).  Logging is one-time only.  The README/open-web documentation normalizes automatic fallback instead of making it an operational error.

**Impact.** A CUDA benchmark may measure the much slower Python/PyTorch fallback after a compile/runtime failure, with no repeated signal.  Any performance comparison with MLX is invalid until strict mode/telemetry confirms the intended kernels execute.

### P2-11 — Recurrent allocator/cache churn hurts latency and graph stability (**confirmed**)

**Evidence.** CUDA `empty_cache()` is called every 100 batch decode steps (`batch_engine.py:1492-1502`), and pool growth allocates an entire replacement pool then copies all arrays (`native_block_pool.py:213-318`).  CUDA allocator configuration is conservative for fragmentation (`hf_diffkv_wrapper.py:247-261`) but cannot remove growth-copy spikes.

**Impact.** `empty_cache()` is a latency/throughput trade-off, not a free memory optimization; repeated calls can increase allocator work.  Pool growth has a large allocation/copy peak and invalidates cached assumptions.  This is particularly hostile to CUDA graph capture.

### P2-12 — Sampling and factual-logit controls allocate vocabulary-scale tensors on the GPU (**confirmed when factual features are active**)

**Evidence.** The HF generation loop builds a full-vocabulary boolean penalty mask and new index tensor (`hf_diffkv_wrapper.py:1113-1119`) and again creates a full-vocabulary VSL mask (`:1166-1177`).  It also calls `.item()` for sampled tokens (`:1180-1181`, `:1223`).

**Impact.** These allocations/synchronizations are outside the model graph and cannot be captured by the current decoder graph.  They are secondary for a large model, but become meaningful for small models/short decode and compound the CPU-bound factual path.

### P2-13 — The advertised native CUDA compressor is not live, and its build paths disagree (**confirmed**)

**Evidence.** `setup.py` is the platform-aware extension build used by the project and includes decode/SRL sources plus CUDA compressor/paging sources on Linux (`native_core/diffkv_core/setup.py:135-166`).  The adjacent CMake file instead lists only compressor, paging, and bindings (`CMakeLists.txt:11-15`), omitting the decode/SRL implementations expected by the bindings.  No live manager code invokes the native compressor/pager classes.

**Impact.** A CUDA build produced through CMake is not equivalent to the setuptools extension, and neither route accelerates the live Python manager's compression today.  Treat native-extension performance claims as unvalidated until a single supported build and live integration exist.

## What is already good / not a current primary issue

- The routed-row gather in `triton_fused_decode.py:1051-1148` avoids the prior full-pool K/V clone in the normal Triton decode dispatch.  Keep it; validate on CUDA rather than reverting it.
- CPU-resident block metadata in `get_cached_decode_blocks()` avoids scalar CUDA synchronization for basic block-state inspection (`kv_runtime_manager.py:1773-1837`).
- Fixed-size dense workspaces avoid repeated workspace allocation, although their contents are still assembled and rotated outside the fused kernel.
- The existing `CUDA_TRITON_AUDIT.md` correctly records kernel-specific validation still required.  This audit adds runtime-level issues that file intentionally did not cover: default CPU compression, ungated factual-store retention, routing, batching, and graph validity.

## Recommended remediation order (no implementation performed)

1. **Establish runtime truth on a CUDA host.** Run with strict Triton mode and collect Nsight Systems/Compute traces separating prefill transfer, CPU SVD, CUDA SVD, routing, dense preparation, and Triton.  Record CPU utilization and D2H/H2D bytes.
2. **Make feature gating truthful.** Ensure factual K/V retention, factual-store construction, factual query, and logit machinery are absent unless the opt-in flag is enabled.  This is both a default-performance fix and a documentation-correctness fix.
3. **Choose one compression strategy.** Prefer a complete GPU-resident batched randomized-SVD/cuSOLVER path, using dedicated streams/events without host synchronization.  If CPU compression remains necessary, make it an explicit fallback with bounded pinned buffers and measured admission control—not the CUDA default.
4. **Remove mandatory D2H barriers/copies from ingest.** Do not synchronize immediately after staging; give the consumer a correct event dependency and keep ownership/lifetime safe without a second cloned CPU buffer.
5. **Rebuild decode around a static CUDA ABI.** Keep routing indices, dense K/V, lengths, pool generation/version, and session mapping in device buffers.  Only then reintroduce CUDA graph capture with invalidation tied to all mutable state—not only shapes.
6. **Batch the sparse attention kernel.** Add a batch dimension or a packed-ragged request dimension so continuous batching reduces launches and preparation, rather than wrapping sequential B=1 executions.
7. **Move SRL/factual selection off the critical host path.** Keep descriptors/index data on GPU; avoid score-vector D2H and Python list graph traversal each token.  At minimum, cache/reroute less frequently and measure quality impact.
8. **Fuse or cache dense-window RoPE preparation.** Precompute/maintain dense absolute positions and rotate within the combined Triton kernel or persistent workspace; eliminate the Python list/tensor construction and `torch.cat` per layer/token.
9. **Harden observability.** In benchmark/production-performance modes, fail fast on Triton fallback, report active path per request, and expose transfer/SVD/router counters.
10. **Unify and wire the native extension only after correctness.** Do not rely on the current native compressor stub or contradictory CMake target as an optimization path.

## Required CUDA validation before changes are accepted

1. Confirm `HAS_TRITON=True`; set `DIFFKV_TRITON_STRICT=1`; capture logs proving sparse and combined kernels are active.
2. Profile default settings first.  Expected signature from this audit: repeated GPU→CPU copies, event synchronization, CPU QR/SVD, and host routing gaps during prefill/decode.
3. A/B factual store default-off versus on.  Default-off should eliminate retained prefill K/V CPU copies and factual decode work while preserving normal sparse attention.
4. A/B CPU compression versus a genuinely GPU-resident implementation; measure prefill tokens/s, p50/p99 TTFT, CPU%, D2H bytes, and peak VRAM/RAM at 4k/16k/32k/128k.
5. Test batch sizes 1/2/4/8.  Measure kernel launches/token/request to demonstrate whether batching is real.
6. Run the existing CUDA parity tests (`tests/test_sparse_residual.py::test_triton_matches_reference_on_gpu`, `tests/test_triton_combined.py`) plus long-context retrieval before and after any runtime change.
7. Test CUDA graph correctness with changing session IDs, routing sets, dense-window growth, and compression transitions; do not use replay for throughput reporting until it passes.

## Audit limitations

- This workspace is on Apple Silicon and the base shell environment lacks Torch, Triton, and MLX, so CUDA kernels were not compiled or executed here.
- Static findings identify executed source paths and defaults; exact cost ranking between P1 findings needs a CUDA trace.
- No source code was changed.  This file is an audit artifact only.
