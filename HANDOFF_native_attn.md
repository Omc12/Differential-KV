# DiffKV Native Attention — Handoff / Continuation Doc

_Last updated: 2026-06-16. Read this top-to-bottom to resume in a fresh chat._

## 0. The goal (unchanged)
Make the **native ggml/Metal attention subgraph** (`build_native_sparse_attn` in
`diffkv_native/src/main.cpp`) a faithful, fast replacement for the 24 per-layer
`ggml_map_custom3` custom-op dispatches. Same algorithm as `ACTIVE_RUNTIME` / the C++
custom-op reconstruction, executed natively in one fused graph. **Gated behind
`DIFFKV_NATIVE_ATTN=1`, default OFF.** Goal is NIAH-equivalence, not bit-identity.
Rule: ACTIVE_RUNTIME is source of truth; only change `diffkv_native` to match it; if a
genuine bug exists in the reference, patch it and mirror. Log to
`diffkv_reconstruction_logs.md`.

## 1. CURRENT STATUS (one line)
Native path works on realistic prompts (coherent, terminates), but **echoes the prompt
verbatim on a pathological 12×-repeated prompt** (`/tmp/longprompt.txt`). The subgraph
**math is PROVEN exact**; the echo is a downstream integration/precision issue.

| Test | Result |
|---|---|
| Default path (no native) | **bit-identical**: token id 382, logit 13.7686 (`"The capital of France is"`) |
| `DIFFKV_SELFTEST` (math, no rope) | **PASS** maxAbsDiff 3e-8 |
| `DIFFKV_SELFTEST_ROPE` (math, with rope) | **PASS** maxAbsDiff 2e-4 (fp16) |
| realprompt native (`/tmp/realprompt.txt`) | coherent, terminates with `. <\|im_end\|>` (works) |
| longprompt native (`/tmp/longprompt.txt`) | **ECHOES** 700–2000+ tokens of prompt, inflated logits 35–45, no EOS |
| longprompt CPU reference (`DIFFKV_FORCE_CPU_ATTN=1`) | coherent, 24 tokens, ends `. <\|im_end\|>` (logit ~14) |

## 2. THE BIG PROVEN FACT — the subgraph math is exact
`run_native_attn_selftest()` (in `src/main.cpp`, run via `DIFFKV_SELFTEST=1`) builds a
tiny pool with KNOWN values (incl. **massive anchor-K ~per-elem 30** and **near-duplicate
slots** to mimic the repetitive prompt), runs `build_native_sparse_attn` on the CPU
backend, and compares **element-wise** to the real reference:
`execute_cpu_attention` (sparse) + `cpu_dense_attention` (dense) + 3-way LSE combine.
- No rope: maxAbsDiff **3e-8**. With rope (`DIFFKV_SELFTEST_ROPE=1`): **2e-4** (fp16 level).
- ⇒ sparse + dense + current + merge + rotation are **all mathematically correct**, even
  on extreme/repetitive data. **The bug is NOT in `build_native_sparse_attn`.** It is in
  the **inputs fed to it / the full-model integration**.

## 3. ROOT-CAUSE INVESTIGATION (the DIFFKV_DBG_CMP tool — use this!)
We added an **in-process per-layer/per-step comparison** (gated `DIFFKV_DBG_CMP`, in
`src/main.cpp` decode loop ~line 2785). It captures the native run's layer-`L` `q_rope`,
`selected_slots`, current K/V, and `attn_out`, then computes the **exact CPU reference**
(`execute_cpu_attention` + `cpu_dense_attention` + LSE combine) on the host using the SAME
inputs, and prints `maxAbsDiff`. Layer is selectable with `DIFFKV_DBG_CMP_LAYER=N`.

Findings on longprompt:
- **Layer 0, all 12 steps: maxAbsDiff ≈ 0.002** (matches CPU at fp16). First real token
  `"It"` matches the CPU. So **step 0 / layer 0 is correct.**
- **Layer 11, default sync: maxAbsDiff 0.23–0.30** (DIVERGED; norms 3.79 vs 3.67).
- **Layer 11, `DIFFKV_SYNC_ALL=1`: maxAbsDiff 0.0008** (matches; norms 3.755 vs 3.755).
  ⇒ Deep-layer **device pool tensors (`VK_rot`/`anchorK_rot`/…) were STALE vs host.**
- BUT: **`DIFFKV_SYNC_ALL=1` generation STILL echoes** (726+ tokens, prompt content,
  logits 35–45). So device staleness was a *real contributor* at deep layers, but
  fixing it does NOT stop the echo. The residual ~0.002/layer device-vs-host fp16
  difference (precomputed `VK_rot`/`anchorK_rot` rotation vs `execute_cpu_attention`'s
  runtime rotation) appears to **cascade over 24 layers × hundreds of steps** on the
  near-degenerate repetitive attention, flipping tokens → echo. The **inflated logits
  (35–45 vs ref ~14)** are suspicious and suggest possible attention collapse rather than
  pure precision — needs confirmation (see Next Steps A).

### Slot-level detail (DIFFKV_DBG_SEL → DBG_DEVSLOT)
On longprompt L0 the selected slots were e.g. `497,494,504,511,0,0,6,7,8,9`. Slot 511 had
`seq_len=0` but stale `host_anchors_K`=367 while device `anchorK_rot`=0. The CPU reference
**always** includes every selected slot's anchor entry (see
`runtime/diffkv_attention.cpp:72-99` and `:209` — no `seq_len` guard); the native was
**masking** `seq_len==0` anchors via `anc_mask`. That was a genuine native-only divergence
→ fixed (see §4 anc-fix).

## 4. FIXES APPLIED (currently in the working tree, UNCOMMITTED)
`git diff` touches `src/main.cpp` (+57) and `native_core/kv_runtime_manager.cpp` (+12).

1. **anc-fix (CORRECT, keep).** `src/main.cpp` `build_native_sparse_attn` (~line 496):
   the anchor entry is now ALWAYS included (matches `execute_cpu_attention`), instead of
   being masked for `seq_len==0` slots. Revert with `DIFFKV_MASK_EMPTY_ANCHOR=1`.
   This made the **sparse-only L0** output match the CPU (maxAbsDiff 0.008).
2. **no-reuse decode alloc (native).** Decode graph uses `ggml_backend_alloc_ctx_tensors`
   + `ggml_backend_graph_compute` when `native_attn_on` (instead of the sched path), to
   rule out buffer reuse. (Did NOT fix the echo; buffer reuse was not the cause. Can stay
   or be reverted — low risk; realprompt works with it.)
3. **sync-all (DIAGNOSTIC ONLY, default OFF).** `native_core/kv_runtime_manager.cpp`
   `sync_device_for_native` (~line 643): `DIFFKV_SYNC_ALL=1` uploads every slot every step.
   **Too slow for production** (512 slots × 24 layers per step) and does NOT fix the echo.
   Keep it gated OFF; replace with the efficient targeted sync (Next Steps C).
4. **Debug/capture infra** (all env-gated, in `src/main.cpp`): `g_dbg_qrope/attn0/sel0/
   curk/curv` capture pointers + the `DIFFKV_DBG_CMP` comparison block; `DIFFKV_DBG_SEL`/
   `DBG_DEVSLOT` device-vs-host slot check. `cpu_dense_attention` was made **non-static**
   (`runtime/diffkv_attention.cpp`) + forward-declared in `src/main.cpp` so the harness can
   call it.

## 5. ENV FLAGS / TOOLS (all in diffkv_native binary)
- `DIFFKV_NATIVE_ATTN=1` — enable native subgraph (the feature under test).
- `DIFFKV_ENGAGE_THRESHOLD=64` — force sparse engagement at low context (for test prompts).
- `DIFFKV_SELFTEST` / `DIFFKV_SELFTEST_ROPE` — run the standalone math proof, exit.
- `DIFFKV_DBG_CMP` [+ `DIFFKV_DBG_CMP_LAYER=N`] — **in-process native-vs-CPU per-layer diff.**
- `DIFFKV_SYNC_ALL=1` — brute device→host pool sync (diagnostic).
- `DIFFKV_FORCE_CPU_ATTN=1` — run the CPU reference path (the "correct" baseline).
- `DIFFKV_MASK_EMPTY_ANCHOR=1` — revert the anc-fix.
- Component isolation: `DIFFKV_DBG_DENSEOFF` (sparse only), `DIFFKV_DBG_CUROFF`,
  `DIFFKV_DBG_DSOFF`, `DIFFKV_DBG_T1OFF`/`T2OFF` (sparse value terms), `DIFFKV_DBG_NOROPE`.
- `DIFFKV_DBG_SEL` — dump selected slots + `DBG_DEVSLOT` device-vs-host anchor norms.

### Repro commands
```bash
cd diffkv_native
cmake --build build --target diffkv_native -j4
M=./qwen2.5-0.5b-instruct.gguf; P="$(cat /tmp/longprompt.txt)"
# math proof:
DIFFKV_SELFTEST=1 ./build/diffkv_native; DIFFKV_SELFTEST_ROPE=1 ./build/diffkv_native
# per-layer divergence (THE tool); try LAYER=0,5,11,17,23:
DIFFKV_NATIVE_ATTN=1 DIFFKV_DBG_CMP=1 DIFFKV_DBG_CMP_LAYER=11 DIFFKV_ENGAGE_THRESHOLD=64 ./build/diffkv_native "$M" "$P" 2>&1 | grep DBG_CMP
# reference vs native generation:
DIFFKV_FORCE_CPU_ATTN=1 DIFFKV_ENGAGE_THRESHOLD=64 ./build/diffkv_native "$M" "$P" 2>&1 | grep -E "^  0:" | tail
DIFFKV_NATIVE_ATTN=1     DIFFKV_ENGAGE_THRESHOLD=64 ./build/diffkv_native "$M" "$P" 2>&1 | grep -E "^  0:" | tail
```
⚠️ Native echo runs generate thousands of tokens & take many minutes — always
`pkill -9 -f diffkv_native` after, and prefer capturing the FIRST few `^  0:` tokens or
`[Response]` rather than waiting for EOS.

## 6. NEXT STEPS / FUTURE PLAN (in priority order)
**A. Determine: precision cascade vs residual bug.** With `DIFFKV_SYNC_ALL=1` + `DBG_CMP`,
   sweep `DIFFKV_DBG_CMP_LAYER=0,3,6,…,23` at step 0 and find the first layer where
   maxAbsDiff jumps above ~0.01. If it climbs smoothly (0.002→…→small) it's an fp16
   cascade; if one layer SPIKES, there's a layer-specific input bug there. The **inflated
   logits (35–45)** argue something may still be genuinely wrong — chase this first.
   Also: compare native vs `DIFFKV_FORCE_CPU_ATTN` token-by-token for the first ~5 real
   decode tokens to see exactly where the trajectories split.
**B. If precision cascade:** reduce per-layer device-vs-host diff. Options:
   (i) store `VK_rot`/`anchorK_rot` device tensors as **fp32** not fp16;
   (ii) make `upload_slot`'s precomputed rotation bit-match `execute_cpu_attention`'s
        runtime rotation; (iii) accept as a known limitation on pathological repetitive
        inputs (document it) since realistic prompts work.
**C. Replace sync-all with an efficient targeted sync (needed regardless).** Deep-layer
   staleness is real. Cheap fix candidates: in `sync_device_for_native`, (a) drop the
   `block->device_synced` skip so OCCUPIED `CompressedResident` blocks re-sync each step
   (handles re-compression staleness), and (b) also upload the routing
   `physical_candidates` (host-known, bounded — see `src/main.cpp` ~line 2680/2704) to
   cover stale-but-reselected slots. Then re-test L11 maxAbsDiff stays ~0.001 without
   uploading all 512 slots. NOTE: `selected_slots` also includes in-graph `sem_slots`
   (semantic-descriptor matches) which are NOT host-known; those correspond to compressed
   blocks and should already be CompressedResident/synced.
**D. Verify & measure.** Confirm default still bit-identical (id 382/13.7686), realprompt
   still coherent, NIAH still passes, and re-measure the ~4.6× decode speedup with the
   efficient sync.
**E. Cleanup.** Remove/guard the `DIFFKV_DBG_*` instrumentation and `g_dbg_*` captures
   once resolved; keep `DIFFKV_SELFTEST` (it's the math regression guard).

## 7. KEY CODE LOCATIONS
- `src/main.cpp`
  - `build_native_sparse_attn` (~line 430): the subgraph. **anc-fix at ~496** (`ae`).
  - `run_native_attn_selftest` (~line 1095): math proof harness (calls non-static
    `execute_cpu_attention` + `cpu_dense_attention`).
  - `build_decode_graph` (~line 582): native branch ~line 707–729; `g_dbg_*` captures set
    at `l == DIFFKV_DBG_CMP_LAYER`.
  - decode loop `DIFFKV_DBG_CMP` block ~line 2785; `DIFFKV_DBG_SEL`/`DBG_DEVSLOT` ~line 2793.
  - native decode alloc/compute branch (`native_attn_on`) ~lines 2238 / 2534 / 2727.
- `native_core/kv_runtime_manager.cpp`: `sync_device_for_native` (~643, has sync-all).
- `runtime/diffkv_attention.cpp`: `execute_cpu_attention` (reference; anchor handling
  72–99, combine 207–214); `cpu_dense_attention` (now non-static, ~261).
- `runtime/native_block_pool.cpp`: `upload_slot` (~270) — the host→device push + the
  precomputed `VK_rot`/`anchorK_rot` rotation that must match the CPU's runtime rotation.

## 8. THINGS ALREADY RULED OUT (don't re-chase)
- Subgraph math (sparse/dense/current/merge/rope) — PROVEN exact via self-test.
- ggml_backend_sched buffer reuse — switched to no-reuse alloc; didn't change the echo.
- The async device-upload bug (`sync_device_for_native` existence) — already fixed earlier.
- `native_maxd` cap (was 256, now 2048) — already fixed; dense window not truncated (Td=509).
- The anchor-mask over-masking — fixed (anc-fix); sparse L0 now matches CPU.
- Whole-pool staleness as the SOLE cause — sync-all fixes per-layer diff but NOT the echo.
