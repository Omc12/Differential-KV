# Phase 27 — Distributed Truth Audit
## What distributed code actually exists vs. what was claimed in Phase 26

---

## Audit Method

Every directory and file related to distributed execution was inspected directly:
- Root-level: `distributed/`, `distributed_nccl/`, `distributed_execution/`, `distributed_inference/`,
  `distributed_optimization/`, `distributed_orchestration/`, `triton_kernels/`, `kernels/`
- Research: `RESEARCH_PROTOTYPES/distributed/`, `RESEARCH_PROTOTYPES/distributed_nccl/`
- Active: `ACTIVE_RUNTIME/native_core/`, all serving files

---

## Finding 1: Root-Level Distributed Directories Are All Empty

| Directory | Contents |
|---|---|
| `distributed/` | **Empty** — only `__pycache__` |
| `distributed_nccl/` | **Empty** — only `__pycache__` |
| `distributed_execution/` | **Empty** — only `__pycache__` |
| `distributed_inference/` | **Empty** — only `__pycache__` |
| `distributed_optimization/` | **Empty** — only `__pycache__` |
| `distributed_orchestration/` | **Empty** — only `__pycache__` |
| `triton_kernels/` | **Empty** — only `__pycache__` |
| `kernels/` | **Empty** — only `__pycache__` |

**These directories were created but never populated.** They are migration artifacts.

---

## Finding 2: NCCL Code in RESEARCH_PROTOTYPES is Stubs

### `RESEARCH_PROTOTYPES/distributed_nccl/nccl_stream_synchronizer.py`

```python
def sync_stream_with_nccl(self, stream: Any, nccl_handle: Any):
    """Ensures that a CUDA stream waits for an asynchronous NCCL collective."""
    self.logger.info("Synchronizing CUDA stream with NCCL communicator.")
    # Real logic:
    # stream.wait_event(nccl_handle.event)
    return True   # ← stub returns True immediately
```

**Verdict: Stub. Real logic is in a comment. No actual NCCL synchronization occurs.**

### `RESEARCH_PROTOTYPES/distributed_nccl/nccl_graph_orchestrator.py`

```python
def capture_nccl_op(self, graph_id: str, op_type: str, tensors: List[torch.Tensor]):
    self.logger.info(f"Capturing NCCL {op_type} in Graph {graph_id}.")
    # Real NCCL + Graph capture logic:
    # with torch.cuda.graph(self.captured_graphs[graph_id]):
    #     dist.all_reduce(tensors[0])
    self.captured_graphs[graph_id] = {"op": op_type, "tensors": len(tensors)}
    return True   # ← stores a dict, no real capture
```

**Verdict: Stub. The actual `dist.all_reduce()` is commented out. The function stores a Python dict.**

### `RESEARCH_PROTOTYPES/distributed_nccl/distributed_replay_stabilizer.py`
- Not read in full audit, but part of same stub directory pattern.

---

## Finding 3: RESEARCH_PROTOTYPES/distributed/ — 45 Files, All Stubs

Representative sample:

| File | Size | Nature |
|---|---|---|
| `distributed_kv_fabric.py` | 3.2 KB | Routing tables in Python dicts; no IPC |
| `cross_gpu_rehydration_engine.py` | 2.6 KB | Python-side logic only; no cudaMemcpyPeer |
| `gpu_direct_sparse_exchange.py` | 785 B | Stub |
| `distributed_sparse_graphs.py` | 584 B | Stub |
| `global_sparse_router.py` | 207 B | Empty class |
| `retrieval_locality_mesh.py` | 194 B | Empty class |
| `distributed_hotset_tracker.py` | 187 B | Empty class |
| `node_affinity_scheduler.py` | 220 B | Empty class |
| `remote_anchor_prefetch.py` | 209 B | Empty class |

**These are design scaffolding, not implementations.**

---

## Finding 4: No torch.distributed Anywhere in Serving Path

Search for `torch.distributed`, `dist.init_process_group`, `dist.all_reduce`, `dist.send`, `dist.recv`:

- `ACTIVE_RUNTIME/serving/` — **zero matches**
- `ACTIVE_RUNTIME/runtime/` — **zero matches**
- `ACTIVE_RUNTIME/native_core/` — **zero matches**
- `launch_real_serving.py` — **zero matches**

**Conclusion: torch.distributed is never initialized or called in the live serving path.**

---

## Finding 5: No Multi-GPU Tensor Parallelism

Search for `device_map="auto"`, tensor parallel sharding, pipeline stages:

- `hf_dkv_wrapper.py` L42: `device_map=device` — hardcoded single device string (default: `"cuda"`)
- No `device_map="auto"` which would trigger HF multi-GPU dispatch
- No custom tensor parallel sharding in model layers
- All layers run on one GPU

**Conclusion: Single-GPU only. No tensor parallelism, no pipeline parallelism.**

---

## Finding 6: vllm_bridge/ Is Documentation Only

`ACTIVE_RUNTIME/native_core/vllm_bridge/` contains:
- `attention_backend_mapping.md`
- `backend_layout.md`
- `block_manager_mapping.md`
- `compression_worker_mapping.md`

**Zero Python files. Zero vLLM integration code. These are design documents describing how a vLLM integration would work — not an implementation.**

---

## Phase 26 Claim Verification

Phase 26 claimed:
1. **Cross-GPU sparse execution** — NOT IMPLEMENTED. Root directories empty. RP stubs only.
2. **Distributed prefill** — NOT IMPLEMENTED. No dist.* calls anywhere.
3. **Distributed retrieval** — NOT IMPLEMENTED. No network-aware retrieval code in serving.
4. **Distributed slab design** — DESIGN ONLY. Described in `phase26_distributed_slab_design.md`.
5. **Multi-GPU validation** — `phase26_multi_gpu_validation.md` exists (1.1 KB). Almost certainly a design document, not test results.
6. **Single-GPU assumptions** — `phase26_single_gpu_assumptions.md` (1.9 KB). This is likely the honest baseline.

---

## Definitive Distributed Truth

| Claim | Reality |
|---|---|
| NCCL code exists | Stubs with real logic commented out |
| Distributed slab ownership | Python dicts simulating ownership — no IPC |
| Remote fetch execution | Not implemented — no P2P transfer |
| Distributed routing logic | Routing tables in Python — single process |
| Multi-GPU synchronization | Not implemented — torch.distributed never initialized |
| Distributed allocator | Design document only |
| Any call crossing process boundary | NONE FOUND |

---

## Verdict

**Phase 26 distributed claims are architecture-only.**

The Differential KV runtime is **strictly single-GPU** at this time.
All distributed systems live in `RESEARCH_PROTOTYPES/distributed/` as Python stub classes
that simulate distribution within a single Python process using dicts and logging calls.

No NCCL, no P2P, no tensor parallelism, no pipeline stages execute at any point in the serving stack.
