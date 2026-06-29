# Handoff — Compressed Decode: speed (Options A/B) + porting MLX wins to C++

Date: 2026-06-29. Author context: see `benchmarks/COMPRESSED_DECODE_OPTIMIZATION.md`
(full results) and memory `project_active_compressed_decode`.

## Where we are
The MLX active runtime (`ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py`) now has, validated on
the HARD on-topic NIAH prompt (`benchmarks/niah_recall.py --bench`, exact buried passcode in
*generated* text):

| Win | What | State |
|---|---|---|
| **Accuracy** | `DIFFKV_MAX_RESIDUAL` 32→64 (capture the needle's distinctive tokens) | exact 4k–64k |
| **Routing** | residual-key router (`_block_relevance_residual`) — rank blocks by exact q·k over anchor + residual keys | exact 4k–**64k** |
| **Speed** | top-K block routing makes decode ~context-independent | ~10–16 tok/s flat |

**Honest speed status:** the MLX path is **overhead-bound, not FLOP-bound** at ≤64k. The
sparse kernel is already `@mx.compile`d; the cost is the *uncompiled* per-layer orchestration
(routing, `mx.eval(sel)` sync, the Python residual-gather loop). So it loses to dense's single
fused kernel below ~128k, and only wins at very long context / 1M (where dense OOMs). Two
proven negative results to NOT repeat:
- **Fused-SDPA reconstruction** (reconstruct selected blocks → one fused SDPA): slower AND less
  accurate. Don't.
- **min/max (Quest) and SVD-score routers**: too loose at 256+ blocks — they average away the
  single outlier token you're searching for. The residual-key router is the fix. Don't go back.

Goal for next phase: **competitive decode speed** at moderate context, and **any model / 1M+**
with no hardcoding. The MLX path can be pushed (Option B) but real parity lives in the native
C++/Metal path (Option A).

---

## Option A — Native fused Metal kernel (the real speed path) — RECOMMENDED

The native path (`diffkv_native/`) is the right vehicle for competitive speed: it already has a
Metal fused sparse-attention op, it's just incomplete. Today the **accurate** path is forced to
a CPU custom op whenever residuals are present.

What exists:
- Fused op `build_native_sparse_attn` / `ggml_diffkv_attn` — see the comment block at
  `diffkv_native/src/main.cpp:48-64`. It assumes a **single block scale** and **cannot read the
  per-row int8 U** scales, and it does **not handle residuals**.
- So `diffkv_native/runtime/diffkv_attention.cpp:784` **forces CPU** "if any selected block has
  residuals (Metal doesn't handle them)". The CPU custom op is correct but is the architectural
  speed bottleneck (memory: `project_native_decode_tps_sampler`).

Work (in priority order):
1. **Teach the fused Metal op the per-row int8 U scales.** `lowrank.cpp` switched U to per-token-row
   int8 scales to fix needle recall; the CPU op reads them, the fused graph still assumes one
   block scale. Until the kernel reads per-row scales, `DIFFKV_NATIVE_ATTN=1` stays wrong.
2. **Add residual K/V to the fused op** so residual-bearing blocks no longer fall back to CPU
   (remove the `:784` force-CPU). Residuals are exact fp16 K/V appended per block
   (`native_block_pool.hpp` `res_K_val/res_V_val`, `MAX_RESIDUAL` slots).
3. **Add top-K block selection inside/around the fused op** (see Port item #2). Score all blocks
   cheaply, then run value reconstruction + residual attention only for the top-K.
4. Validate against the CPU op byte-for-byte (the existing selftest harness already does this for
   the no-residual single-scale case — extend it to per-row + residuals).

Payoff: this is where dense-class throughput is achievable (fused Metal vs hand-rolled), and it's
the only path that gets the *accurate* (residual) configuration off the CPU.

---

## Option B — Vectorized / compiled MLX decode (bounded but real)

Cut the uncompiled per-layer overhead in `execute_decode_attention`
(`ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py`). Today, per layer per token, it:
`mx.argsort` + **`mx.eval(sel)` (GPU→CPU sync)** + **Python loop** gathering selected blocks'
residuals + `mx.concatenate` + 7× `mx.take`. Each breaks the graph / forces a sync.

Plan:
1. **Vectorize the residual gather.** Replace the Python `for bi in res_blocks` loop with
   `mx.take(comp_res_k[:nb], sel)` → `[K, max_res, kv_heads, D]`, reshape to
   `[kv_heads, K*max_res, D]`. Build a validity mask `arange(max_res) < n_res[sel]` (keep
   `comp_res_n` as an mx array). This removes the loop AND the `mx.eval(sel)` sync (sel stays
   lazy, used only in `mx.take`).
2. **Kernel takes a boolean validity mask** instead of a prefix `dense_len`. The padded residual
   slots have scattered validity, so `compute_decode_attention_static` must mask by an explicit
   `[cap]` bool mask in its dense-path softmax (replace `dense_idx < dense_len`). Update the
   `nb==0` and non-topK call sites to build the equivalent prefix mask.
3. With (1)+(2), routing+gather+kernel can live in one (mostly) compiled graph → fewer syncs.
4. Estimated ceiling ~14→~20 tok/s at ≤32k (still below dense's fused kernel — that's why Option
   A matters more). Worth it to raise the *flat* line (lowers the dense-crossover context).

Note: the residual router (default) scores all residuals (`O(nb·R·D)`), which is real compute —
Option B reduces *orchestration* overhead, not router FLOPs. Keep `DIFFKV_ROUTER=minmax` as the
fast path for users who stay ≤32k.

---

## What to do in C++ that we did in MLX and is NOT in native

The native already does the *expensive* part (per-block anchor + residual scoring, with RoPE, in
`diffkv_attention.cpp:401-473`) — it just **attends every block and never selects**, uses a
too-small residual budget, and runs accurate decode on CPU. Port these, in order:

1. **Capture fix (accuracy).** Native `MAX_RESIDUAL = 40` (`native_block_pool.hpp:139`) + a
   15%-fraction / err>0.08 selection (`lowrank.cpp:733-767`, `DIFFKV_RESIDUAL_FRAC`). This mirrors
   MLX's old *failing* 32. We found the needle's tokens can be **mid-rank** residuals (e.g. the
   "DELTA" token was residual #33+), so:
   - Raise the cap to **64** (make it runtime-configurable, not a `constexpr`).
   - Prefer **fixed top-N by reconstruction error** over the fraction+threshold (the threshold
     drops mid-rank-but-critical tokens). Validate on `niah_recall.py --bench` until exact.

2. **Top-K block routing (speed + 1M).** Native softmaxes over ALL `active_K` blocks; `max_score`
   (`diffkv_attention.cpp:373`) is only LSE stabilization, not selection. Add a selection pass:
   compute per-block relevance (you already have it — see #3), `nth_element`/`partial_sort` to the
   top-K, then run the value reconstruction + residual attention only for those K. Makes decode
   cost scale with K, not block count — essential at 1M.

3. **Residual-key router (the generalizable retrieval fix).** Rank each block by
   `max(anchor_score, max over its residual token scores)` — native ALREADY computes
   `anchor_score` and per-token `res_score` (`diffkv_attention.cpp:401,448`). Use them to *select*
   top-K (#2), not just to attend. Score (near-)all residuals per block for deep-needle
   correctness (the R=full finding). This is what fixes 64k+ where min/max routing fails. The
   residual selection is content/model-agnostic (it's just "this block's most distinctive
   tokens"), so it generalizes across models with no tuning.

4. **DON'T port the dead ends.** No min/max-bound router (too loose at scale — proven). No
   fused-reconstruction decode (slower + less accurate — proven). And note the subtle bug that
   wasted a cycle in MLX: combining an exact score with an upper bound via `max(bound, score)` is
   a **no-op** (the bound always wins) — rank by the exact scores directly.

5. **Residual ordering.** MLX now stores residuals **highest-error-first** so the router can score
   the top-R cheaply (`mlx_diffkv_wrapper.py`, `np.argsort(errors)[-max_res:][::-1]`). If native
   adds a "route on top-R residuals" fast mode, store/scan them in the same order.

---

## Cross-impl validation (do this for every change, both sides)
- **Recall gate:** `python benchmarks/niah_recall.py --bench --ctx 8192 16384 32768 65536`
  (exact passcode in generated text). This is the hard prompt that exposed every bug; the easy
  prompt hides them.
- **MLX kernel parity:** `cd ACTIVE_RUNTIME && python -m pytest tests/test_diffkv_kernel_parity.py -q`.
- **Native vs CPU op:** the existing byte-for-byte selftest (extend it for per-row int8 + residuals).
- **Speed/mem table:** `benchmarks/bench_worker.py` (active/dense/native) — compare flat-vs-declining
  curves; the compressed win is the high-context crossover + not OOMing.

## Key files
- MLX: `ACTIVE_RUNTIME/serving/mlx_diffkv_wrapper.py` — `_block_relevance_residual` (router),
  `execute_decode_attention` (routing+gather+kernel), `_compress_block` (residual selection),
  `compute_decode_attention_static` (sparse kernel). Env: `DIFFKV_ROUTER`,
  `DIFFKV_ROUTE_RESIDUALS` (0=all), `DIFFKV_TOPK_BLOCKS`, `DIFFKV_MAX_RESIDUAL`.
- Native: `diffkv_native/runtime/diffkv_attention.cpp` (CPU decode + per-block scoring + :784
  force-CPU), `diffkv_native/runtime/native_block_pool.hpp` (`MAX_RESIDUAL`, residual buffers),
  `diffkv_native/native_core/compression/lowrank.cpp` (SVD + residual selection),
  `diffkv_native/src/main.cpp:48-64` (fused Metal op status / `build_native_sparse_attn`).

## Known ceilings (flag to stakeholders)
- **1M memory:** residual K/V at `max_residual=64` ≈ 7.5 GB across layers at 1M → needs
  datacenter RAM, or a smaller `max_residual` (trades recall). Independent of routing.
- **Router cost** is `O(nb·R·D)` — linear in context. Fine at 1M on big hardware; the two-stage
  (cheap pre-filter + residual rerank) is risky because the cheap filter can drop the needle.
