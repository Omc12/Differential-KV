# CUDA-graph decode for DKV — design plan

Status: **plan only, nothing implemented.** Written 2026-08-01 after the routing
top-K fix, because that fix removed the main structural blocker without anyone
intending it to.

## Why this is now the top lever

Measured on Qwen2.5-1.5B-Instruct, 15,849-token prompt, RTX PRO 4000, preset
mid, fused Triton kernel confirmed running:

| | value |
|---|---|
| DKV decode | 102.1 ms/token |
| dense decode | 9.3 ms/token |
| DKV tokens attended | 5,650 (16 blocks x 257 + 1,538 dense) |
| dense tokens attended | 15,849 |

**DKV attends one third the tokens and is 11x slower.** It cannot be compute.
The earlier decode profile put 89% of wall time outside GPU compute
(204.7 of 231.1 ms/token), i.e. host-side dispatch. At 28 layers that is
~3.3 ms of CPU per layer per decoded token.

Two data points bound how much is left to win:

* Cutting routed tokens 4x (57 -> 16 blocks, 14,649 -> 4,112 tokens) moved
  decode only 125.7 -> 102.1 ms, i.e. **19%**. The compressed-block work is not
  the cost.
* Removing 27 of every 28 heat-update syncs contributed to that same 19%.

So the remaining ~90 ms/token is per-layer Python and launch overhead. That is
exactly and only what CUDA graphs remove.

## What the current blocker actually is

`native_core/graph_runtime/static_decode_graph.py` disables capture with:

> The DKV attention patch mutates Python/session state on every decode forward
> (routing slots, dense-window layout, SRL state, session IDs). A captured graph
> replays without executing any of this Python.

That reasoning is correct but the conclusion is stronger than it needs to be.
CUDA graphs do not require state to be *immutable*. They require:

1. **Fixed tensor ADDRESSES** across replays. Contents may change freely — you
   update the buffer in place before replaying.
2. **Fixed SHAPES**.
3. **No host-side control flow inside the captured region** that depends on
   device values (no `.item()`, `.tolist()`, `if tensor:`).
4. **No allocation inside the region** (or a graph-owned memory pool).

Mutable routing is fine. Freshly-allocated, variably-shaped routing is not.

## How others solve the same problem

The pattern is consistent across production inference stacks, and DKV's routing
is structurally the same problem as paged-attention block tables:

* **vLLM** — decode is CUDA-graphed despite PagedAttention's per-step block
  tables. The block table lives in a pre-allocated device tensor at a fixed
  address, written in place each step; the graph reads whatever is there.
  Variable batch size is handled by capturing a **set of graphs for bucketed
  batch sizes** and padding up to the nearest bucket. `enforce_eager` disables
  the whole mechanism for debugging.
* **TensorRT-LLM** — same principle: persistent I/O buffers, graphs captured per
  shape configuration, inputs copied into the fixed buffers before enqueue.
* **SGLang** — a graph runner that captures per batch-size bucket over a fixed
  KV pool-index tensor.
* **PyTorch Inductor "cudagraph trees"** (`mode="reduce-overhead"`) — automates
  the memory-pool side, but still requires static shapes and refuses to capture
  regions containing host syncs or data-dependent branching.
* **HF transformers `StaticCache`** — exists precisely because `DynamicCache`
  reallocates and so cannot be graphed.

The transferable idea: **pad to a fixed shape, keep every buffer at a fixed
address, and hoist all host-side decisions out of the captured region.**

## Why today's routing fix changed the calculus

Before: `routing_topk_default = max(16, 4096 // 64)` = **64**, and the router
returned however many blocks existed (observed `N_sparse=57`, varying with
context). Variable shape => not capturable without padding logic.

After: K is **16 and fixed**, independent of context length. `L_dense` is
already a fixed-size workspace (`max_dense_len`) whose live extent the kernel
masks via `L_dense_valid`. So the two shapes that matter are now constant by
construction. That is the precondition the original comment says is missing.

## Staged plan

Each stage is independently valuable and independently revertible. Do not skip
Stage 0 — it decides whether the rest is worth building.

### Stage 0 — bound the prize (no code changes to the runtime)

Measure the host-side cost that a graph would remove, per layer per token:

* Wrap one decode step with `torch.profiler` on the **generate() path** (the
  existing `profile_decode_step.py` cannot: it bypasses `generate()` and never
  engages the fused kernel — see handoff §10u).
* Record: wall time, GPU busy time, number of kernel launches per token, and
  number of host syncs per token.

Accept/abort criterion: if GPU busy is already >60% of wall after the routing
fix, graphs are not the answer and the remaining cost is real compute. If it is
still <25%, proceed. **Do not build Stage 1+ on the old 89% figure — that was
measured before the routing and sync fixes.**

### Stage 1 — static ABI for the decode region

**PARTIALLY IMPLEMENTED (opt-in, unvalidated).** `DKV_STATIC_GATHER=1` switches
`_gather_routed_blocks_for_kernel` from advanced indexing (`pool.U[indices]`,
which allocates) to `index_select(..., out=persistent_buffer)`. Verified on CPU:
values identical, writes into the buffer, address stable across repeated gathers
-- the graph precondition. Default OFF; refuses to reuse buffers whenever
deferred batch dispatch is possible (fails closed, see `_batch_queue_active`).
Still to do: q/out/lse buffers, `block_indices` itself, and the dense workspace.

**MEASURED: no effect.** 102.7 vs 102.1 ms/token, peak VRAM identical, recall
unchanged (all six depth cases pass). The gather allocations were not the cost.
Keep it only as a graph PREREQUISITE (fixed addresses), not as an optimisation
in its own right -- and note that it is the third change predicted to help that
returned ~0:

| change | predicted | measured |
|---|---|---|
| Path A vs Path B (fused kernel) | large | 0% |
| routing top-K 64 -> 16 (4x less work) | large | 19% |
| static gather buffers | moderate | 0% |

All three were inferred from op tables rather than measured on the generate()
path. Stage 0 exists to stop this; it was skipped. `colab/profile_decode_host.py`
now implements it.


Pre-allocate, once per session, at fixed addresses:

| buffer | shape | notes |
|---|---|---|
| `block_indices` | `[K]` fixed (16) | pad with a repeat of slot 0 and mask, or a `-1` sentinel the kernel already skips |
| `seq_lens`, `scales`, `U_scale` | `[K]` | gathered in place |
| dense K/V workspace | `[1, H_kv, max_dense_len, D]` | already fixed-size |
| `L_dense_valid` | scalar on device | kernel already masks on it |
| q, out, lse | fixed | |

Rule: the decode path may only `copy_` into these. Any `torch.empty`/`cat`/
`stack` inside the region breaks capture.

Known offenders to convert (all identified while profiling today):
* `block_indices` freshly built per step
* the `_rot_state` dense-rotation cache (already caches across steps — verify it
  never reallocates once warm)
* `torch.cat([zeros_pad, deltas_k], dim=1)` in the compiled reconstruct path

### Stage 2 — evict host syncs from the region

Must be zero inside the captured region:

* `active_key = tuple(sorted(active_idx.tolist()))` (`triton_fused_decode.py`)
  — a D2H sync used only for a Python cache key. Either hoist the lookup above
  the region or key it on the device-side routing generation counter.
* `TieredBlockStore.update_heat` — throttled to 1-in-32 today, but must be moved
  strictly outside, not merely made rare.
* Any `.item()` / `bool(tensor)` — the numeric guards are already behind
  `DKV_DEBUG_NUMERICS`; keep it that way.

### Stage 3 — capture and replay

* Warm up >= 3 eager steps (Triton JIT + Inductor must finish first — note the
  Inductor warmup only started succeeding after the einsum fix).
* Capture one graph per shape bucket. With K fixed and the dense workspace
  fixed, the expectation is **one** graph per session; verify rather than assume.
* Invalidate on: routing-version change that alters the *pool base pointers*
  (not routing contents), pool reallocation, session change, sequence length
  crossing the dense-workspace bound.
* Keep `DKV_DISABLE_CUDA_GRAPH=1` as the default until Stage 4 passes.

### Stage 4 — correctness gate (non-negotiable)

A graph that replays stale routing produces *plausible* wrong answers, which is
the worst failure mode this project has. Required before flipping the default:

* `colab/validate_cuda_dkv.py` — the needle **depth sweep** (0.0/0.5/0.9 at 2k
  and 8k). Depth is the discriminating test: a needle at position 0 sits in a
  block that sink/recency rules retain regardless of routing, so it passes even
  when routing is broken.
* Bit-exactness vs eager for >= 32 consecutive decode steps, not 1. A stale
  graph is correct on step 1 by construction and wrong later.
* A test that deliberately forces a routing change mid-generation and asserts
  the graph invalidates.

## Risks

* **Silent staleness** is the whole risk. Every other failure is loud.
* Graph memory is held for the session lifetime; on an 8 GB card where DKV
  already peaks at 4.0 GB this is not free.
* Multi-session/batch serving multiplies graph count — out of scope for now.

## Explicitly out of scope

Prefill. It is variable-shaped by nature and is 4.83 s vs dense 1.45 s — a real
problem, but a different one, and chunked prefill is not graph-shaped.
