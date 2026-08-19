# Changes to port from CUDA to MLX

Written 2026-08-19, from the CUDA work on branch `srl-adaptive-memory`
(`552332ec..79342f4a`), merged as branch `srl-shared-basis`.

**The previous edition of this file is retired**, and was deleted from `main`
once that port landed; its record lives in
`ACTIVE_RUNTIME/docs/cuda_port_record.md`. It covered commits
`a5289a18..7c63bdca`. Two of its conclusions are load-bearing for what follows
and are restated here rather than left to be rediscovered — see "Carried
forward" below.

This edition starts the cycle again for a new batch of CUDA work.

**Scope.** This branch adds ONE thing to CUDA: shared low-rank bases across
blocks, **off by default** (`DKV_SHARED_BASIS`). This file is what MLX needs to
do to reach parity, and — as important — what it should NOT expect to gain.

Three sibling features were built alongside it on `srl-adaptive-memory` and
**deliberately not merged**, because measuring them said not to. They are not
described here; the branch and its commit messages hold the full record. In
short, so nobody re-derives them:

  * *Anchor-delta residual budget* — inert at the shipping config. Residual
    fill is already 40.0 of a 40-slot budget on every block, so the tier the
    override targets never binds.
  * *Shared-basis chunk-graph edges* — adds ZERO edges when basis groups are
    contiguous, which is what a real document produces.
  * *Learned hybrid SRL router* — the scoring head reproduces the rule-based
    ranking (val recall@16 0.599 vs 0.595). The K-head IS 14.3% faster at 32k
    (paired, CI [-9.00, -6.91] ms) but drops >50% of attention mass on the
    worst 5% of queries, and an oracle static-K sweep shows no K below 16 with
    an acceptable tail on that workload — so it is not a training problem.

**Confidence labels.** `VERIFIED` means I read the MLX source and the construct
is there; line numbers are from `ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py` at
this commit. `LIKELY` means the mechanism is shared but I did not confirm the
code path. Nothing here was run on Apple silicon.

---

## Carried forward from the previous edition

### The randomised-SVD seed is a ±15-point noise floor

The rSVD draws its projection from a seeded generator whose SHAPE depends on
the configured rank. At a fixed config, changing `DKV_RSVD_SEED` alone moved the
old synthesis harness **63.3 / 33.3 / 50.0**. Temperature-0 replication proves
nothing: a number reproduced twice is one sample, not two.

**Any synthesis difference under ~15 points is not a difference.** MLX runs the
same randomised SVD (`DKV_SVD_SEED`) and has the same floor. Use
`colab/synthesis_power.py` — replicated, paired, interval-bounded.

### Routing is not a quality lever, settled by three independent methods

| method | result |
|---|---|
| linkbench, 48 seeds, K=16 vs attend-EVERY-block | 47/48 vs 47/48 |
| generated prose, K=16 vs attend-all | **byte-identical** |
| synthesis, paired and powered, K=16 vs K=32 @32k | **+0.00**, 95% CI [-4.33, +4.33] |

Showing the model every block changes nothing, so the router is not missing
anything — what limits DKV is what the blocks CONTAIN, not which are chosen.
This is why the learned-router work summarised in the scope note above was
not merged, and why nothing here promises a routing gain.

The single caveat still open: "attend everything" is not a strict upper bound,
since attending more can dilute attention. A cleverer *subset* could in
principle beat both. No evidence supports it.

---

## 1. Shared low-rank bases

**Priority: highest. This is a VRAM change, and VRAM is deterministic.**

**MLX status: VERIFIED to have the same redundancy and the same layout
problem.** `comp_VK` / `comp_VV` are allocated per layer as
`[max_blocks, kv_heads, rank, head_dim]` (`:2033-2034`), i.e. one basis per
block, exactly as CUDA's `V_KV` was.

**The observation.** Every block stores its own basis `V`. On the `mid` preset
(rank 32, kv_heads 2, head_dim 128, pool block 257) a CUDA slot is:

```
U        257 * 32 * 1      =   8,224 B
V_K+V_V   32 * 2*128*2*2   =  32,768 B   <- 39% of the slot
anchors      2*128*2*2     =   1,024 B
residuals 40 * 2*128*2*2   =  40,960 B
```

`V` is the largest item after the residual store, and it is the one adjacent
blocks of one document most nearly agree on — they are prose from the same
source spanning nearly the same subspace of key/value space.

**The math, which is why it costs nothing to apply.** A block is stored as
`D ≈ U V`. For a group basis `Vg` with ORTHONORMAL ROWS, the best approximation
of `U V` inside span(Vg) is

```
U' = U (V Vgᵀ)        and        D ≈ U' Vg
```

one `[k,F] × [F,r]` matmul. No re-decomposition, no touching the original K/V.

Threshold on RETAINED ENERGY, not principal angles — angles weight every basis
direction equally while a block's energy sits in the first few. With `G = UᵀU`
and `C = V Vgᵀ`:

```
kept = tr(G C Cᵀ) / tr(G V Vᵀ)  ==  ||U' Vg||_F² / ||U V||_F²
```

Both traces are `[k,k]`, so scoring a block against a group never touches the
token dimension.

**What to port.** `ACTIVE_RUNTIME/native_core/compression/basis_group.py` is
pure math plus a greedy registry and translates directly to `mx`. Then:

- Allocate `comp_VK`/`comp_VV` at `ceil(frac * max_blocks)` rows (`:2033-2034`)
  and add a `basis_of` map per layer.
- Redirect **every** slot-indexed read. MLX has more of these than CUDA did:
  `mx.take(comp_VK, sel, axis=0)` (`:993-994`), `mx.take(session["comp_VK"]
  [layer_idx], sel_abs, 0)` (`:1533-1534`), the direct slices at `:2289`,
  `:2983`, `:3373`, `:3405`, `:4007-4008`, `:4229`, and the writes at `:2943`,
  `:3338`, `:3906`. **`:3605` is the one to be careful with** — the sliding
  eviction `comp_VK[:-1] = comp_VK[1:]` shifts basis ROWS as if they were
  blocks, which under sharing renumbers every group at once.
- Checkpoint shapes at `:2081-2082` change; version the checkpoint or old ones
  load onto a differently-shaped store.

**Assign in COMPRESS, not at write time.** Residual selection scores
`delta − recon`, and under a shared basis the recon the decoder rebuilds comes
from the GROUP basis, not the block's own SVD factor. Assigning at write time
leaves every residual chosen against a reconstruction that is not the one
stored — repairing errors that do not exist and missing the ones sharing
introduced. MLX's equivalent seam is `compress_deferred_prefill_blocks`, before
the `capture_scores` ranking at `:2846`.

**Residual FORM is a hard precondition.** Residuals in correction form are a
delta against a block's OWN reconstruction, so sharing invalidates every one of
them. Exact form (`DKV_RESIDUAL_EXCLUDE_SVD`, MLX default `"1"`) stores the
anchor-relative true K/V and is basis-independent. CUDA's pool refuses to
enable sharing when exact form is off and says so; MLX should do the same.

**Measured on CUDA** (RTX 4070 SUPER, Qwen2.5-1.5B, 8k NIAH, `mid` preset,
`colab/srl_tradeoffs.py`, seeds pinned): see the table in §3.

**What to watch.** The fidelity cost is real and shows up as `mean_kept`. At
`frac=0.25` on this document the basis store filled and blocks were
FORCE-JOINED — capacity is a hard contract, so exhausting it degrades fidelity
rather than failing a write. Report `forced` and `mean_kept`; a run where
`forced` is large and `mean_kept` is well under 1.0 is spending accuracy for
VRAM, which may or may not be the trade you want.

---

## 2. Traps found implementing this on CUDA — MLX will hit most of them

Every one of these cost a debugging pass. None are CUDA-specific except where
noted.

1. **An unwritten slot must resolve to a VALID basis row** (row 0) or any gather
   over it reads out of bounds. But "points at row 0" is then indistinguishable
   from "holds a refcount on row 0". Releasing a claim never made decremented
   the founding block's own group, returned the row to the free list while
   blocks were still reading it, and let a later block RE-FOUND it with a
   different basis — silently changing what earlier blocks decompress to. Keep
   an explicit claimed flag.

2. **Release before assign.** A slot being overwritten must give up its previous
   claim first, or the write decrements the group it just joined.

3. **`r_proj` can be NARROWER than the store rank.** It is
   `min(max_rank + oversamples, T_active, feat_dim)`, so a short block makes the
   SVD narrower than the pool rank. `U'` has one column per BASIS direction and
   does not fit back into the `[N, T, r_proj]` buffer it came from. Rebuild at
   the store width.

4. **Slot ids must exist before the basis claim.** CUDA allocated slots at write
   time, ~600 lines after the assignment, so `pool_idx` was None. Allocate early
   — but only under sharing, or you widen the window in which a failed compress
   leaks slots.

5. **Compiled kernels that take the whole V array and index it by slot cannot be
   redirected from Python.** CUDA has five such dispatch sites (dkv_core fused,
   Metal, aten). They must DECLINE under sharing and fall through to the gather
   path. Three of the five are macOS-only or need an opt-in build, so a missing
   guard stays invisible on a Linux/CUDA box indefinitely — CUDA now checks all
   five by parsing its own source. **MLX is the runtime where those paths are
   live**, so this is the item most likely to bite there.

6. **`getattr(b, "layer_idx", -1) or -1` makes layer 0 falsy.** Every layer-0
   block reported -1 and shared a basis search space with genuinely unknown
   blocks. Grouping still worked; it grouped the wrong set.

7. **Do not report the config back as a measurement.** `sharing_factor` computed
   over pool CAPACITY yields exactly `1/frac` whenever the store is full,
   regardless of how many blocks were written. Count slots that actually hold a
   claim.

8. **CPU unit tests cannot see any of items 3-5.** They drive the pool directly
   and never reach a kernel dispatch. Four separate faults survived 100 passing
   tests and appeared on the first real-hardware run. Run the hardware harness.

---

## 3. Measured

RTX 4070 SUPER, Qwen2.5-1.5B-Instruct, 8k NIAH prompt, `mid` preset,
`DKV_RSVD_SEED`/`DKV_SVD_SEED` and `DKV_POOL_BUDGET_GB` pinned,
`colab/srl_tradeoffs.py`, n=3 depths (0.1 / 0.5 / 0.9).

| arm | pool MB | vs DKV base | sharing | mean kept | forced joins | recall |
|---|---|---|---|---|---|---|
| dense (control) | 0.0 | — | — | — | — | 0/3 |
| DKV baseline | 91.4 | +0.0% | — | 1.000 | 0 | 0/3 |
| `frac=0.50` | 69.8 | **-23.6%** | 2.6x | 0.969 | 58-142 | 0/3 |
| `frac=0.25` | 59.0 | **-35.5%** | 3.3x | 0.905 | 128-703 | 0/3 |
| `frac=0.125` | 53.6 | **-41.4%** | 6.5x | 0.759 | 505-1487 | 0/3 |

**The recall column is uninformative here and must not be read as "no
regression".** The DENSE control also scores 0/3: Qwen2.5-1.5B does not answer
this needle at 8k at all (it replies fluently with the wrong code — `The secret
passcode is "AI revolution"`). At 4k dense recalls 1/1, but there DKV does not
compress (pool 0.0 MB), so the model's working range and DKV's active range do
not overlap on this model. Qwen3.5-2B spends 400+ tokens in `<think>` without
reaching an answer. **A recall-based accuracy claim for these features needs a
model/context where the dense control passes and the pool is non-empty; that
was not found here.**

What IS measured exactly: pool VRAM (deterministic), and `mean kept` — the
fraction of each block's delta energy that survives the shared basis, which is
the quantity that determines reconstruction error. At `frac=0.25` blocks keep
90.5% of their delta energy; at `0.125`, 75.9%.

**The premise is only partly borne out.** The idea was that adjacent blocks of
one document share a subspace, making the saving nearly free. At `frac=0.50`
that holds: 58-142 forced joins and `kept` 0.969. Below it the basis store
fills and most of the saving is bought by FORCED lossy joins, not by genuine
redundancy — 1487 forced joins at `frac=0.125`. So `frac=0.50` is the regime
where this is close to free; the deeper settings are lossy compression by
another name, and should be argued for on that basis.



---

## 4. Nothing to port: CUDA caught up to MLX on the question-span pin

**MLX status: MLX was already right.** This is recorded so the divergence is
not re-introduced, not because MLX needs a change.

`current_query_tokens` is read as a lexical query by the lexical router and the
decode-time `query_toks` set. MLX sets it from the extracted question span
(`mlx_dkv_wrapper.py:1978-1979`, `pq = self._pending_query.pop(...)`); CUDA only
ever took the whole-prompt fallback, so a single-turn request named every token
in an 8k document as part of the question.

Two silent faults on the CUDA side, now fixed:

1. `KVRuntimeManager` never declared `_pending_query`. `hf_dkv_wrapper` has
   always written to it inside a bare `except Exception: pass`, so every write
   raised `AttributeError` and was swallowed. MLX declares it at `:1900`.
2. The write was gated on `_factual_enabled` (off by default), while the
   consumers are routing, not the factual store.

Measured after the fix, Qwen2.5-1.5B @8k NIAH: `current_query_tokens`
7986 → 11, decoding to exactly the question, and per-block lexical overlap goes
from std 0.0818 (near-uniform — every block matches the document) to 0.1264.

**The lesson worth keeping**, since it generalises past this field: a pin
written into a bare `except` and read through a fallback fails *completely
silently*. Nothing errors, the fallback is plausible, and the only symptom is a
signal that quietly discriminates nothing. If MLX grows a similar pin, assert
the attribute exists rather than trusting the write.
