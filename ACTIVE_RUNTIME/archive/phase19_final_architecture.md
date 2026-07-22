# Phase 19 Final Architecture

This is the definitive structure of the Differential KV project following Phase 19 architectural convergence.

## `native_core` (The Execution Engine)
The heavily-optimized, physically executing core runtime.
- `/sparse_decode`: `triton_sparse_attn.py`, `triton_dkv.py`
- `/compression`: `async_compressor.py`, `lowrank.py`
- `/paging`: `paged_kv_store.py`
- `/metadata_pool`: `metadata_pool.py`
- `/graph_runtime`: `static_decode_graph.py`
- `/`: `kv_runtime_manager.py`, `recon_cache.py`

## `serving` (Orchestration & Integration)
The API and token scheduling layers.
- `/`: `batch_engine.py`, `hf_dkv_wrapper.py`

## `research` (Experimental R&D)
Mathematically valid systems awaiting native C++ integration (e.g., vLLM or FlashAttention-3) to overcome Python orchestration bottlenecks.
- `sparse_prefill_anchors.py`, `fused_sparse_prefill.py`
- `async_tiered_ffn.py`, `tiered_ffn.py`
- `sparse_mlp_fused.py`

## `ARCHIVED_SYNTHETIC_SYSTEMS` (Quarantine)
Permanently rejected architecture theater. Contains all legacy "cognitive", "geometry", and simulated telemetry systems.

## Conclusion
The architecture is now clean, highly maintainable, and directly oriented toward production deployment. 
