# CUDA / Triton Audit + GPU Validation Checklist

**Date:** 2026-07-06 (updated 2026-07-10)  •  **Auditor:** static review on Apple Silicon
(Triton falls back to PyTorch here — *no Triton/CUDA kernel was executed on CUDA*).
•  **Status (2026-07-10):** F1/F3/F6/F7/F9/F10/F11 applied AND committed (`745739a`).
**F2 FIXED this pass** — per-token whole-pool clones replaced by a routed-row gather
+ generation-keyed cache (`_gather_routed_blocks_for_kernel`; equivalence certified on
CPU by `ACTIVE_RUNTIME/tests/test_triton_gather_equiv.py`). **F8 APPLIED this pass** —
per-token `cudaDeviceSynchronize()` → `cudaStreamSynchronize(0)` (minimal barrier for
the legacy default stream all launches use). Also fixed: all three
`decode_attention_metal` call sites (2 live + 1 test) had drifted from the compiled
binding after it grew 4 dense-window args — they now pass empties (dense merged
separately), and `test_metal_residual_and_fact_parity` passes again on Mac.
**Triton compile/run + `.cu` changes still require a GPU box — see checklist.**
This file was restored from git history (`745739a`) after being removed in a docs
cleanup; the checklist below is still pending, so it stays at the repo root.

Scope: the active CUDA/Triton decode surface only —
- `ACTIVE_RUNTIME/native_core/sparse_decode/triton_fused_decode.py` (Triton kernels + PyTorch/MPS fallbacks)
- `diffkv_native/native_core/diffkv_core/src/diffkv_decode.cu` (native GGML CUDA decode kernel)
- `diffkv_native/CMakeLists.txt` (`-DGGML_CUDA=ON` build)

The `archive/`, `EXPERIMENTAL_RUNTIME/`, and `RESEARCH_PROTOTYPES/` Triton/CUDA trees are
parked (see `PARKED_SYSTEMS.md`) and out of scope.

---

## Dispatch map (verified statically)

```
diffkv_attention.py:1517/1618
  └─ native_triton_sparse_attn_decode()          triton_fused_decode.py:1344   ← unified CUDA entry
       ├─ HAS_TRITON & N>0 → _fused_sparse_decode_kernel[grid]      (line 164)  ← THE live kernel
       │                     + _fused_sparse_decode_reduction_kernel (line 347) when num_chunks>1
       │                     + PyTorch dense-block LSE merge         (line 1504-1548)
       ├─ not HAS_TRITON  → _pytorch_vectorized_sparse_attn_decode  (line 940)  ← reference
       └─ except Exception → _pytorch_vectorized_sparse_attn_decode (line 1555) ← silent fallback

MPS path: fused_decode_mps() (line 705) — separate, already validated on Mac.
Native C++/GGML CUDA: execute_cuda_attention() → decode_attention_cuda_kernel (diffkv_decode.cu)
```

Reference of record for the Triton kernel = `_pytorch_vectorized_sparse_attn_decode`.
`fused_decode_mps` is a second, independently-validated reference that agrees with it on the
residual formulation (see F1).

---

## Findings — Triton path (`triton_fused_decode.py`)

### F1 — 🔴 CORRECTNESS — ✅ FIXED (pending GPU cert): residual correction now aligned to the Mac reference
**Root cause (two bugs):**
1. **Append vs correct-in-place.** The kernel (old lines 291-328) treated C1 residuals as
   *extra standalone tokens* appended to the block softmax, while both references
   (`fused_decode_mps:806`, `_pytorch_vectorized_sparse_attn_decode:1164`) correct the
   existing delta token **at its position**. `residual_*_values` store the DIFFERENCE
   `(exact - lowrank_recon)` (`lowrank.py:659/666`) — the kernel's own comment mislabeled
   them "exact K/V tokens." Scoring a small difference-vector as a standalone token adds a
   spurious low-weight ghost while the approx token still remains.
2. **K/V position conflation.** The kernel used `residual_K_positions` for *both* K and V,
   but K and V select **different** worst-reconstructed tokens (observed: K `[6,2,4,7,3,5,1]`
   vs V `[3,5,1,6,4,2,7]`). The reference uses `residual_V_positions` for the V correction.

**Mac-verified evidence (no GPU needed for the math):**
- Principle A/B (`scratchpad/residual_principle_test.py`): concentrated-error case — recon-only
  0.39, **append (old) 0.47 (worse than doing nothing)**, correct-in-place **0.024**; full
  correction → correct-in-place **3e-07 (exact)**, append 0.65.
- Real compressed data (`tests/test_sparse_residual.py::test_sparse_residual_correctness`):
  reference correct-in-place = **2.9e-4 mean err vs dense** (vs 1.5e-2 without). Confirms
  residuals-are-differences and correct-at-position recovers exact attention end-to-end.

**Fix applied** (`_fused_sparse_decode_kernel` + dispatcher): residual K correction adds
`q·resK` to the delta score at `res_pos` **before** the block softmax (so P and the denominator
become exact there); residual V correction adds `p_delta[res_pos_v]·resV` to the output
numerator, normalized by `l_i` at the end — mathematically equivalent to the reference. Both
loops reuse the kernel's existing, proven fact-override idiom (`tl.where(offs_s==pos, …)`).
Added `res_pos_v` kernel arg + stride; `res_k` is now pre-rotated (anchor RoPE) in the
dispatcher like `V_K`. Signature↔launch alignment verified. `MAX_RESIDUAL` now = actual column
count (unrolled scalar loop, no padded `tl.arange`).
**Cert:** run `tests/test_sparse_residual.py::test_triton_matches_reference_on_gpu` on the GPU
box — asserts the real Triton kernel ≈ reference ≈ dense (skips on Mac).
- Note: the C2 *fact-anchor* override (score replace + V diff) was already correct and is
  untouched. My dispatcher `res_k.clone()` adds one more per-token whole-pool clone (see F2).

### F2 — 🟠 PERF — ✅ FIXED 2026-07-10 (pending GPU cert): whole-pool `.clone()` per token
- Was: `anchors_K_rot = pool.anchors_K.clone(); V_K_rot = pool.V_K.clone()` (+ `res_k.clone()`),
  then only `[indices]` rows RoPE-rotated — O(pool) bandwidth ×3 per decode token.
- **Fix:** `_gather_routed_blocks_for_kernel` gathers ONLY the N routed rows of every
  per-block tensor the kernel reads, rotates those, and passes block_indices remapped to
  `arange(N)` — bit-identical kernel inputs (the kernel uses the slot id purely as a row
  index; verified). Applied to BOTH dispatchers (`native_triton_sparse_attn_decode` and
  `..._combined`). The gathered set is q-independent, so it is cached keyed on
  (pool id, `_stratified_generation`, exact index order) — the MLX fused-decode
  route-interval-reuse lesson ported to CUDA: steady routing costs zero gathers.
- CPU-certified: `tests/test_triton_gather_equiv.py` (equivalence for rot/no-rot ×
  res/no-res × fact, cache hit/invalidate/evict, order sensitivity). GPU cert = C8 below.

### F3 — 🟠 HARDENING / OBSERVABILITY: silent, un-instrumented fallback
- Lines 1551-1560: `except Exception` swallows **any** kernel/compile error and drops to the
  PyTorch decoder, printing a warning **once** (`fallback_fired`). There is no positive signal
  that Triton is actually running, and a numerically-wrong-but-non-throwing kernel is invisible.
- On a GPU box you cannot currently tell from logs whether you are measuring Triton or the
  slow PyTorch fallback. **Applied fix (Mac-safe, see below):** one-time "Triton path ACTIVE"
  confirmation + `DIFFKV_TRITON_STRICT=1` to re-raise instead of falling back.

### F4 — 🟡 DEAD CODE (landmines inside the active file)
- `diffkv_fused_decode_kernel` (lines 42-160): unreferenced anywhere. Also contains a
  **hardcoded `kv_heads==2`** assumption (`slot * tl.constexpr(2) * HEAD_DIM`, lines 113/120)
  — if anyone wires it up thinking it's the live kernel, it silently mis-indexes for any model
  whose KV-head count ≠ 2.
- `lowrank_recon_kernel` (line 382): only referenced from `archive/dist/…`, dead in the active file.
- Recommend deleting both (or moving to `PARKED_SYSTEMS.md`) so the live kernel
  (`_fused_sparse_decode_kernel`) is unambiguous. Not deleted here pending your call (repo
  convention is catalog-not-delete).

### F5 — 🟡 ROBUSTNESS: shared mutable workspace + uninitialized `torch.empty`
- Workspace tensors (lines 1401-1405) are cached and reused across calls keyed on
  `(H_q, num_chunks, D_pad, device)`. Safe under the current single-stream, sequential decode,
  but **not** safe if decode is ever batched across heads/layers concurrently or run on multiple
  CUDA streams. `m_workspace`/`l_workspace` are `torch.empty` (uninitialized) — correctness relies
  on the grid writing every `(h_q, chunk)` slot before the reduction reads it. Validate under
  batch_size>1 (checklist C4).

---

## Findings — native CUDA kernel (`diffkv_decode.cu`)

### F6 — 🔴 CORRECTNESS/SAFETY: hardcoded scratch dimensions
- Lines 333-335: `cudaMalloc(&d_split_out, 64 * 8 * 128 * sizeof(float))` (and `_m`, `_d`).
  This assumes **n_q_heads ≤ 64, S_split ≤ 8, head_dim ≤ 128**. Current local models fit
  (Qwen 12/28 heads, Llama 24 heads, all D=128), but a model with head_dim > 128 or > 64 query
  heads (e.g. 70B) writes out of bounds → memory corruption / wrong output / illegal-access crash.
  Parameterize from `data->n_q_heads`, `S_split`, `data->D`, and re-allocate on growth (as is
  already done for `d_slot_indices_scratch`).

### F7 — 🟠 HARDENING: unchecked CUDA API calls
- `cudaMalloc` (333-335, 344), `cudaMemcpy` (347), and the kernel launches (361, 399) have no
  error checking, and there is no `cudaGetLastError()` after launch. A failed launch/alloc is
  silent until a later, confusing sync error. Add a `CUDA_CHECK(...)` macro + post-launch
  `cudaGetLastError()`.

### F8 — 🟠 PERF — ✅ APPLIED 2026-07-10 (pending GPU cert)
- Was: `cudaDeviceSynchronize()` on **every** decode call — a full device barrier per token.
- **Fix:** `DIFFKV_CUDA_CHECK(cudaStreamSynchronize(0))` — all launches in
  `execute_cuda_attention` go to the legacy default stream, so stream-sync(0) is the
  minimal barrier that still guarantees `d_out` is ready for the host read that follows.
  Going fully async (sync only when the caller consumes dst) requires plumbing the ggml
  CUDA stream through the custom-op callback — deliberately NOT attempted blind; measure
  the stream-sync win first (checklist C5).

### F10/F11 — ✅ IMPLEMENTED (pending GPU cert): full DiffKV native CUDA kernel
New kernel `diffkv_full_decode_kernel` ([diffkv_decode.cu](diffkv_native/native_core/diffkv_core/src/diffkv_decode.cu))
replaces the anchor-only stub as the **CUDA default**. It implements the complete DiffKV decode
attention, ported line-by-line from the validated CPU reference (`execute_cpu_attention`,
approximate_attn path) + the ACTIVE Triton kernel:
- anchor score + **low-rank delta tokens** (project-then-attend: `q_proj[r]=q·VK[r]`,
  `delta[t]=Σ_r U[t,r]·q_proj[r]·rowscale·blk_sc`) — the entire content the stub ignored;
- **exact residual correction** K-side (add `q·resK` to the token score at `res_K_pos`) and
  V-side (add `w·resV` at `res_V_pos`), `res*=exact−recon` — same correct-in-place semantics as
  the F1 Triton fix; K/V use their separate position arrays;
- **dense window** online-merged; split-K preserved (merge kernel unchanged, layout verified);
- `block(D)` threads — fixes the F11 `block(64)`-vs-D=128 half-output bug.
Targets the native default **POOL_ROT_ABS** scheme (pool pre-rotated → no in-kernel RoPE);
`has_rope=1` (legacy rotation) is **not** handled by the delta path — use CPU there.
**Verification status: written blind (no CUDA on this Mac); NOT compiled or run.** Every memory
layout is documented at the kernel head and mirrors the CPU indexing, but must be certified on
a GPU (procedure C7 below). Safety valves: `DIFFKV_CUDA_ANCHOR_ONLY=1` (old stub, A/B) and
`DIFFKV_FORCE_CPU_ATTN=1` (exact CPU reference).

### F10-orig — context: why the stub was wrong
Original `decode_attention_cuda_kernel` is a STUB (not just missing residuals)
**Upgraded finding.** `decode_attention_cuda_kernel` ([diffkv_decode.cu](diffkv_native/native_core/diffkv_core/src/diffkv_decode.cu:64))
attends to **block anchors + dense window ONLY**. `U_pool`, `VK_pool`, `VV_pool`, `U_scale_pool`,
`rank`, `S_max` are passed in but **never used in the kernel body** (verified) — so the entire
low-rank *delta-token* content of every compressed block is ignored (each 256-token block
collapses to its single anchor). Residuals and fact overrides are absent too. The native **CPU**
path ([diffkv_attention.cpp:62-460](diffkv_native/runtime/diffkv_attention.cpp:62)) implements the
full DiffKV attention (project-then-attend deltas + residuals + facts); the CUDA path
([diffkv_attention.cpp:1055](diffkv_native/runtime/diffkv_attention.cpp:1055)) does not, yet is the
**default** on a CUDA build (unless `DIFFKV_FORCE_CPU_ATTN=1`). → native CUDA output would be far
from Mac/CPU (severe recall loss). **Making native DiffKV attention correct on CUDA = a from-scratch
kernel port of the full CPU/Triton math, which cannot be verified on a Mac.** GPU device buffers for
residuals exist (`pool.get_res_K_pos/val()`, `get_res_V_pos/val()`), so plumbing is available.

### F11 — 🔴 native `.cu` kernel: `block(64)` threads but D=128 → half the output dims unwritten
- Launch uses `dim3 block(64,...)` ([diffkv_decode.cu:352](diffkv_native/native_core/diffkv_core/src/diffkv_decode.cu:352))
  while the value accumulation writes `out_buf[... + tid]` for `tid < 64` ([line 253](diffkv_native/native_core/diffkv_core/src/diffkv_decode.cu:253)).
  For head_dim=128 (all current models) only dims 0-63 are computed; 64-127 are left uninitialized.
  Must launch `block(D)` (and bound `thread_val[D]`). Part of the from-scratch rewrite (F10).

### F6/F7/F9 — ✅ APPLIED this pass (safe, mechanical; still need GPU compile)
- F6: split-K scratch now sized from actual `n_q_heads * S_split_max * D` (+ realloc on growth),
  replacing the hardcoded `64*8*128` (was OOB for >64 heads / D>128).
- F7: `DIFFKV_CUDA_CHECK` macro on `cudaMalloc`/`cudaMemcpy` + `cudaGetLastError()` after both
  kernel launches.
- F9: split-K scratch is freed before re-alloc on growth (no leak on model change).

### F9 — 🟡 LEAK: persistent split-K scratch never freed
- `d_split_out/_m/_d` (statics allocated at 332-336) are never `cudaFree`d. One-time persistent
  allocation, low severity, but couple it with the F6 re-parameterization (free+realloc on growth).

---

## GPU-box validation checklist (run in this order)

**Environment**
- [ ] `-DGGML_CUDA=ON` builds clean (`find_package(CUDAToolkit)`, cusolver/cublas/cudart link — CMake 73-107).
- [ ] `python -c "import triton; print(triton.__version__)"` inside the serving venv; confirm `HAS_TRITON` is True at import.
- [ ] Start serving with the applied `DIFFKV_TRITON_STRICT=1` (below) and confirm the one-time
      **"Triton path ACTIVE"** log fires and **no** fallback warning appears.

**C1 — 🔴 Certify the F1 residual-alignment fix (kernel unit test)**
- [ ] `pytest ACTIVE_RUNTIME/tests/test_sparse_residual.py::test_triton_matches_reference_on_gpu`
      on the GPU box. Asserts the REAL Triton kernel ≈ reference (max-diff < 1e-2) AND ≈ exact
      dense (mean-diff < 1e-2). (Fix already applied + Mac-verified for math; this certifies the
      Triton *translation* compiles and runs correctly.)
- [ ] Also run `test_sparse_residual_correctness` on CUDA (device auto-selects) — reference path
      sanity on GPU.

**C2 — 🔴 End-to-end recall parity (the real gate)**
- [ ] `DIFFKV_COMPRESSED_DECODE=auto python benchmarks/niah_recall.py --bench --ctx 4096 8192 16384 32768`
      on CUDA. Must match the Mac baseline: exact needle recall (`OMEGA-7741-DELTA`) at every ctx.
- [ ] `benchmarks/niah_recall.py --multi-needle` and `benchmarks/relational_ab.py` — no regression vs Mac.

**C3 — 🟠 Perf: confirm Triton actually wins**
- [ ] Decode tps with Triton vs forced PyTorch fallback at 4k/16k/32k. Triton must be faster; if
      not, F2 (whole-pool clone) and/or F8 (per-token device sync) are likely eating the win.

**C4 — 🟡 Robustness (F5)**
- [ ] `--batch-size 2+` through the gateway; confirm workspace reuse doesn't cross-contaminate heads.
- [ ] Last-chunk partial-fill (N not a multiple of `BLOCKS_PER_CHUNK=16`): correct output, no NaN.

**C5 — native `.cu` (F6/F7/F8)**
- [ ] Run a model with head_dim ≠ 128 or > 28 query heads to exercise the F6 bound (expect
      corruption **before** the fix; clean **after** parameterizing scratch).
- [ ] `compute-sanitizer --tool memcheck` over one decode — zero out-of-bounds / illegal accesses.
- [ ] Before/after F8 sync reduction: decode tps + `nsys` timeline (per-token device barrier gone).

**C7 — 🔴 certify the new native DiffKV CUDA kernel (F10/F11)**
- [ ] Build native with `-DGGML_CUDA=ON`; confirm `diffkv_decode.cu` compiles (all pool accessors +
      `NativeBlockPool::MAX_RESIDUAL` resolve; shared-mem size fits).
- [ ] `compute-sanitizer --tool memcheck` over one decode — zero races / OOB (the kernel is blind-written).
- [ ] **3-way A/B on one needle prompt:** default (new CUDA) vs `DIFFKV_CUDA_ANCHOR_ONLY=1` (old stub)
      vs `DIFFKV_FORCE_CPU_ATTN=1` (exact CPU reference). Assert **default ≈ CPU** per-token (the new
      kernel is correct) and **anchor-only ≠ CPU** (confirms deltas/residuals now matter). Default must
      recall the needle; anchor-only should not.
- [ ] Decode tps: new CUDA kernel vs CPU path — confirm the "faster" claim (and vs F8 sync fix below).
- [ ] If numerics are off, check the documented layout assumptions at the kernel head first
      (U/VK/VV/residual strides, `U_row_scale` per-token vs `scales` per-block) — most-likely failure point.

**C6 — regression guard**
- [ ] Full guardrail suite green on CUDA (NIAH bench, multi-needle, relational, parity) matching Mac.

**C8 — 🔴 certify the F2 gather remap (2026-07-10) on GPU**
- [ ] `python ACTIVE_RUNTIME/tests/test_triton_gather_equiv.py` on the CUDA box (tensor math is
      device-agnostic, but re-run there for dtype/device coverage).
- [ ] `pytest ACTIVE_RUNTIME/tests/test_sparse_residual.py::test_triton_matches_reference_on_gpu`
      — the kernel now receives gathered [N]-row tensors + arange indices; this certifies the
      Triton translation end-to-end.
- [ ] Decode tps before/after F2+F8 at 16k/32k (expect the O(pool)-per-token clone traffic and
      the per-token device barrier gone; if tps is flat, profile with `nsys` before reverting).
- [ ] Cache-hit sanity: steady decode (no pool writes, stable routing) should log zero extra
      gathers — verify `_gathered_rot_cache` hit path with a counter if in doubt.

---

## Changes applied this pass
- **F1 residual alignment (Triton kernel + dispatcher)** — full fix described above. Math
  Mac-verified; Triton compile/run certified by `test_triton_matches_reference_on_gpu` on GPU.
  Mac path provably unaffected (kernel only runs under `HAS_TRITON`, i.e. CUDA).
- **F3 observability (Mac-verified)** — one-time "Triton path ACTIVE" log + `DIFFKV_TRITON_STRICT=1`
  re-raise instead of silent PyTorch fallback. Default behavior byte-unchanged.

## Recommended but NOT applied (need GPU compile/verify — your call)
- **F2 clone elimination** (Triton) — now three per-token whole-pool clones (`anchors_K`, `V_K`,
  and the `res_k` I added). Rotate only the `[indices]` rows into small scratch. Perf only.
- F6 scratch parameterization, F7 CUDA_CHECK, F8 sync reduction, F9 free-on-growth (`.cu`) —
  cannot compile-verify on Mac; propose applying together behind a single GPU build+memcheck pass.

**C9 — LEGO prefill port (torch/CUDA path) — design notes, 2026-07-12**
The MLX reference (`mlx_diffkv_wrapper.py`, `DIFFKV_LEGO_PREFILL`) and native C++
stage 1 (`main.cpp`, same flag) both exist; see `docs/NATIVE_LEGO_PORT_PLAN.md`.
Before porting to the torch path, note what the two implementations taught us:
- The torch `KVRuntimeManager` already right-sizes its pool from a memory budget
  (`dynamic_max_blocks`) — the MLX pool-growth fix has no CUDA equivalent to port.
- The HF wrapper's prefill retains full `past_key_values` (raw) alongside the
  DiffKV store — the same duplication MLX had. A lego port = windowed
  `past_key_values` + far blocks materialised from the manager's pool (the
  torch equivalent of `materialize_routed_kv` already exists in the decode fill).
- NATIVE STAGE-1 LESSON (2026-07-12): shrinking ONE raw copy may not move the
  process peak — decompose the peak (phase-tagged footprint sampling) BEFORE
  building, or the port can be correctness-perfect and win nothing. On the Mac
  native build the ring cut the device cache 81% at 32k with zero footprint
  change (peak set by host mirrors / engine slots / allocator reserves).
- CPU-runnable pre-checks stay green after the 2026-07-12 changes:
  `test_triton_gather_equiv.py` PASS, `test_sparse_residual.py` 2 passed/1 GPU-skipped.
- [ ] On the GPU box: profile prefill peak composition FIRST (torch.cuda.memory_summary
      per phase), then port lego only if `past_key_values` actually dominates.

**C10 — Owner-capture residual selection (torch path) — design note, 2026-07-12**
MLX + native C++ now boost the OWNER of a fact into the exact-residual set
(`DIFFKV_RESIDUAL_OWNER_CAPTURE`, default ON; `_apply_owner_capture` in
`mlx_diffkv_wrapper.py`, mirrored block in `lowrank.cpp`). Root cause it fixes:
values (digits) were captured exactly while entity names (title-case → is_prose)
survived only as rank-r recon — the MLX binding probe (`benchmarks/binding_probe.py`)
showed compressed list-all 1/6 vs dense 5/6 with real values bound to CORRUPTED
names ("Okazaki"→"Okinawa"); owner capture takes it to 6/6 with zero recall
regression (NIAH 4k-32k exact incl. 16k/0.9, multi-needle 3/3, native 6-cell
sweep + margins re-run).
The torch `lowrank.py` path CANNOT take this port directly: it has NO
content-aware boost machinery at all (no token ids plumbed into
`compress_lowrank_block`, K/V residuals ranked separately by rel-error only) —
so it also still has the ORIGINAL failure this fixed, plus the digit-boost gap.
- [ ] On the GPU box: reproduce the binding failure on the torch path first
      (binding_probe pattern), then plumb token ids into the torch compressor
      and port the boost + owner-capture + budget-floor block as one unit
      (default OFF until the probe passes there).
