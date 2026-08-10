# DKV CUDA — Handoff (prefill router alignment)

Give this whole file to the next agent. It is written to be read top-to-bottom
once, then used as a reference. Everything here is measured, not assumed; where
something is a hypothesis it says so.

---

## 0. STANDING RULES — read before touching anything

1. **Never add `Co-Authored-By` or any co-author trailer to commits.** Commit
   messages should be detailed and explain *why*, not just what.
2. **Never edit MLX.** `ACTIVE_RUNTIME/serving/mlx_dkv_wrapper.py` is the
   known-good REFERENCE implementation. Read it, never change it. The job is to
   find every CUDA/MLX difference and make CUDA match. The only exception is
   where CUDA is genuinely better for performance AND produces the same output.
3. **Don't patch around problems — fix roots.** A fix that only half-works is a
   signal the root is elsewhere.
4. **No test loops.** Every GPU run must answer a specific question decided
   *before* the run, with the reading of each possible outcome written down
   first. This codebase has burned entire sessions on runs whose result could not
   discriminate anything.
5. **Don't rely on memory or on comments — check the code.** Several comments in
   this repo describe behaviour that is no longer true.
6. **State a check's COVERAGE, not just its result.** "I verified the inputs
   match" cost several sessions when it turned out to mean 8 of 256 elements.

---

## 0.5 UPDATE — TWO prefill defects, and §2's prescription was incomplete

Everything below §1 was written before the RTX 4070 session. It remains accurate
about the router, but it named ONE defect where there are TWO, and it pointed at
the wrong MLX reference. Corrections, all measured:

**(1) There is a second, independent prefill defect: DOUBLE RoPE on history.**
`DKV_ROTATED_POOL` defaults to `"1"`, so `_ingest_k` writes POST-RoPE keys and a
block's `anchor_kv` / `active_k` are already in their true rotational frame. The
decode gather knows this (`do_rot = ... and not pool_stores_rotated_k()`,
`triton_fused_decode.py:1695`). **Every prefill history reader rotated anyway** —
`_apply_rope_single` at `dkv_attention.py:3602` and `:3862`, and
`_prefill_fused_history_attend`, which rotates unconditionally at
`triton_fused_decode.py:1234`. So every key the prompt was read against carried a
second full rotation. Same omission as the router, same cause: the decode-side fix
never reached prefill.

**(2) MLX's PREFILL router is `_block_relevance_minmax` (`:1098`), not
`_block_relevance_residual`.** §2 says to make prefill call
`route_blocks_relevance` (the decode router). That is the wrong reference and it
is also infeasible: `_sparse_prefill_attend` (`:1265`) scores a Quest-style
min/max box over each block's REAL keys — "Blocks are not yet compressed during
prefill", its own docstring — and MLX uses the residual router only at decode.
The residual form for a 3D query materialises `[H_kv, gpk, L, nb, R]` (~1 GB at
L=1024), which is exactly why `route_blocks_relevance` gates residuals off for 3D
q behind the undocumented, never-set `DKV_ROUTE_PREFILL_RESID`. Calling it from
prefill would have given you anchor-only scoring again, or an OOM.

CUDA can use the min/max box directly, because during a FRESH prefill NO block is
compressed: `DKV_STREAMING_COMPRESS` defaults to `"0"` and SVD publication is
deferred to the prefill boundary, so every history block still holds exact
`active_k`. Same inputs MLX has.

### CONFIRMED ON THE REAL TARGET — `validate_cuda_dkv.py --long`, Qwen3.5-2B

Run on an RTX 4070 SUPER, clean GPU, `transformers 5.14.1`, `DKV_ROUTE_TRACE=1`:

| config | result |
|---|---|
| HEAD | **8/9** — `32k@depth0.9` emits `'None'`, deterministic at temp 0 |
| both §0.5 fixes | **9/9**, every case deterministic, `fallback_count=0` |

HEAD reproduces §1's table exactly, down to the literal `'None'`, so this box is
measuring the same defect the table describes. §3's verification plan is
satisfied in full: baseline reproduced (1), fix passes (3), and prefill is STILL
SPARSE — `DKV_SP_TRACE_TOKEN` reports `k_eff=30` of `120` candidates with the
needle's block at rank 0-1 (516 kept vs 60 dropped over all layers/chunks/cases),
so this is not §3's "accidentally reproduced `DKV_SPARSE_PREFILL=0`" trap.

**`transformers` matters.** `qwen3_5` does not exist before v5; on 4.57.6
`AutoConfig` raises "Transformers does not recognize this architecture" and the
model cannot load at all. `requirements.txt` already says `>=5.14.1`. If a box
has 4.x, nothing about Qwen3.5-2B measured on it means anything.

### Cost of the fix — Qwen3.5-2B, RTX 4070 SUPER, fp16, serving defaults

| metric | HEAD | fixed | delta |
|---|---|---|---|
| TTFT 8k (11,007 tok) | 5.078 s | 5.128 s | +1.0% |
| TTFT 32k (32,717 tok) | 14.239 s | 14.472 s | +1.6% |
| decode, 8k, 128 new tok | 259.8 tok/s | 229.4 tok/s | **-11.7%** |
| peak VRAM 8k / 32k | 4.62 / 5.06 GB | 4.62 / 5.06 GB | none |

Prefill is FLAT despite the router now building min/max boxes over 120 candidate
blocks — the two-GEMM form plus the RoPE work the fix REMOVES roughly cancel it.
Both TTFT deltas are inside run-to-run spread (the 3 reps overlap), so read them
as "no measurable prefill cost", not as a 1% regression.

Decode is ~12% slower and that is NOT yet explained. A prefill-only change moving
decode at all points at the working set: correct routing retains a different set
of blocks, so decode gathers differently. Worth a look before this matters for
long generations.

**Measure decode with LONG generations.** The same harness reported 68.1 vs
94.3 tok/s ("-28%") at `max_new_tokens=32` and 229.4 vs 259.8 ("-12%") at 128 --
same build, same prompt. Decode rate here is derived by subtracting a separate
1-token call from an N-token call, so fixed per-call overhead inflates the gap at
small N. The 32-token number is an artifact of the method; do not quote it.

### DKV vs DENSE — Qwen3.5-2B, RTX 4070 SUPER (12 GB), fp16, one harness

Accuracy, `validate_cuda_dkv.py --long` vs `--long --dense`:

| context | dense | DKV (fixed) |
|---|---|---|
| 2k, 3 depths | 3/3 | 3/3 |
| 8k, 3 depths | 3/3 | 3/3 |
| 32k, 3 depths | **OOM — cannot run** | **3/3** |

Dense dies in `sdpa_attention_forward` trying to allocate **31.77 GiB** on a 12 GB
card. That is the whole case for DKV on this hardware: at 32k the comparison is
not "faster or slower", it is "runs or does not". Below that, DKV matches dense
exactly — 6/6 vs 6/6, every case deterministic, so the fixed router costs nothing
in recall where both can run.

Cost, 8k (11,007 tok), 128 new tokens, both arms measured by the SAME harness:

| metric | dense | DKV (fixed) | DKV vs dense |
|---|---|---|---|
| TTFT (prefill) | 2.013 s | 5.128 s | **2.5x slower** |
| decode | 309.0 tok/s | 229.4 tok/s | **26% slower** |
| peak VRAM | 5.21 GB | 4.62 GB | 11% lower |
| 32k | OOM | 5.06 GB | dense cannot run |

Read this honestly: below the OOM cliff DKV is STRICTLY SLOWER than dense at
equal accuracy, and prefill is where it loses most. ~3.8 GB of each peak figure
is model weights, so the KV working set is ~1.4 GB dense vs ~0.8 GB DKV. DKV's
win is the context ceiling, not throughput.

### WHY DKV IS SLOWER THAN DENSE: it is LAUNCH-BOUND, not compute-bound

`torch.profiler` over one 11,007-token prefill, Qwen3.5-2B, production config
(async compress), both engines through the same script:

| | DKV | dense |
|---|---|---|
| **self CUDA** | **2,714 ms** | **2,974 ms** |
| **self CPU** | **3,993 ms** | **2,066 ms** |
| `aten::mm` | 2,475 calls @ 212 us | 187 calls @ 2,202 us |
| `aten::copy_` | **57,278** calls | 7,230 calls |
| `aten::mul` | 31,582 calls | 13,935 calls |
| `aten::bmm` | 17,785 calls | 15,535 calls |

**DKV uses LESS GPU time than dense and still loses.** The sparse algorithm is
doing its job — fewer FLOPs, 2,714 ms vs 2,974 ms of device time. It loses on the
HOST: 3,993 ms of CPU against dense's 2,066 ms, and because CPU time exceeds CUDA
time, the GPU is idle waiting for Python to hand it the next launch. The
bookkeeping around the algorithm costs more than the algorithm saves.

The `aten::mm` row is the clearest single tell: near-identical total GEMM work
(524 ms vs 412 ms) split into **13x more launches**, 212 us of work per launch
instead of 2.2 ms. Same for `copy_` at 8x the call count. This is a dispatch
problem, so faster KERNELS cannot fix it — only FEWER OF THEM.

**What MLX does that sidesteps this entirely.** (1) Lazy evaluation with kernel
fusion: MLX builds a graph and fuses elementwise chains at eval, so the 31k `mul`
+ 23k `add` + 42k elementwise launches collapse into a handful; PyTorch eager
dispatches every one. (2) Unified memory — no staging copies, which is a large
part of the 8x `copy_` gap. (3) A FIXED `block_size=256` with whole-array ops,
where this side blocks adaptively (32-256) and loops per block in Python.

**The CUDA options, ranked, with status measured not assumed:**

1. **CUDA Graphs — the biggest available win, currently disabled BY DESIGN.**
   `graph_runtime/static_decode_graph.py:63-74`: the DKV attention patch mutates
   Python/session state every forward (routing slots, dense-window layout, SRL,
   session ids), so a replayed graph goes stale and **silently emits wrong
   output**. Re-enabling is not a flag flip — it needs the static,
   device-resident state ABI that file describes. This is the standard fix for
   exactly this profile and it is the one worth designing for.
2. ~~**Batch the remaining per-block loops.**~~ **ATTEMPTED AND EXHAUSTED — this
   work is already done, do not re-open it.** cProfile over the same prefill puts
   *all* DKV Python `tottime` at ~0.2 s of 5.4 s, and the per-block loops that do
   remain (`_gather_block_token_ids` 228 calls, `_should_skip_compression` 258)
   total ~0.14 s — about 2.5% of prefill, so there is no headroom here even in
   principle. The paths that matter are already batched:
   `write_blocks_batched` collapsed ~2,352 `write_block` calls, and
   `_submit_blocks_batched` groups blocks by `T_active` and compresses in
   sub-batches of 64.

   Measured, not assumed: caching `_gather_block_token_ids` per (session, block)
   instead of recomputing it per layer -- which removes 190 of 228 calls, each
   costing an H2D upload plus a full D2H sync -- changed nothing. Back-to-back
   A/B, production async config:

   | | baseline | with cache |
   |---|---|---|
   | 8k TTFT | 5.114 s | 5.036 s |
   | 32k TTFT | 14.580 s | 14.672 s |

   Both inside run-to-run spread. The change was REVERTED: it adds a
   (session, anchor)-keyed cache to a subsystem that had just produced a
   stale-cache bug, and buying that risk for zero measured gain is a bad trade.

   The conclusion that matters: **the launch storm does not come from
   Python-level per-block loops.** It comes from the many tensor ops issued
   INSIDE the already-batched compression (the `[n, T, feat]` fp32 intermediates
   -- deltas, recon, U_masked) and the attention patch. That is option 3's
   territory, not this one's.
3. **Hand-fuse the low-rank reconstruction into one Triton kernel.** The
   `mul`/`add`/`sum` storm is `U@V + anchor` executed as separate ATen ops.
4. **torch.compile — MEASURED, DOES NOT HELP.** `DKV_USE_TORCH_COMPILE=1` compiles
   FFN-only (`inductor`, `mode='reduce-overhead'`): TTFT 5.090 s vs 5.128 s
   (unchanged) and decode **193.7 vs 229.4 tok/s — 16% WORSE**. It compiles the
   part that is not the bottleneck. Do not reach for this again without changing
   WHAT is compiled.

**Environment note.** `cl.exe` (MSVC) is not installed, so the DKV decode JIT
prewarm fails with `Compiler: cl is not found` on every run and the first decode
pays JIT. Plain `torch.compile` on GPU-only graphs still works (verified), so
this blocks the prewarm path specifically, not Inductor as a whole.

### "MLX gets 9/9" — SO DOES CUDA. The difference is the BENCH, not the engine.

Held the engine fixed (CUDA DKV), the model fixed (Qwen2.5-1.5B-Instruct), the
nine (context, depth) points fixed, and swapped ONLY the bench, in one process:

| bench | result |
|---|---|
| MLX's own (`tests/test_mlx_niah.py`) | **9/9**, every case deterministic |
| `validate_cuda_dkv.py`'s | **1/9** |

So the CUDA engine reproduces MLX's 9/9 exactly when it is given MLX's test. Any
comparison that reads "MLX 9/9 vs CUDA 1/9" as an engine gap is comparing two
different tests. They differ in three ways that all point the same direction:

* **Needle.** `OMEGA-7741-DELTA` vs `ZEBRA-4471-QUARTZ`. Qwen splits the latter
  as `ZEBR|A|-|447|1|-|QU|ART|Z`, so recall requires preserving a LONE `A`.
* **Filler.** MLX repeats ONE 4-sentence paragraph to length; the validator
  samples randomly from an 8-sentence pool. Repetitive filler is far easier for a
  compressed cache -- near-duplicate blocks reconstruct almost perfectly, so the
  needle's reconstruction error towers over everything and wins residual slots
  uncontested. Randomised filler makes it compete.
* **Scoring.** Substring anywhere in the output vs normalised-alnum match inside
  24 tokens.

**This does NOT mean CUDA is fine.** On the harder bench, the like-for-like
comparison against dense is damning, and dense is the ceiling:

| Qwen2.5-1.5B, validator bench | 2k+8k (6 cases) | 32k |
|---|---|---|
| dense (no compression) | **4/6** | OOM — cannot run |
| DKV | **0/6** | 1/3 |

DKV loses four cases the model demonstrably CAN answer. The failure is always the
same single dropped token: dense emits `ZEBR-A-4471-QUARTZ` (a hit), DKV emits
`ZEBR4471QUARTZ` -- digits and tail intact, the lone `A` gone. Closing 0/6 -> 4/6
is the real recall target, and MLX's bench cannot see it at all.

Reproduce both arms with `colab/bench_mlx_vs_validator.py` (`BENCH=mlx|dkv`); the
`dkv` arm reproduces the real validator's 1/9 exactly, which is what makes the
comparison trustworthy.

**The 2k drop is NOT residual capture — proven, not inferred.** `DKV_DBG_RESIDUAL_TOKEN=<abs
token index>` (new, diagnostic-only, `lowrank.py`) reports, for the block that
actually owns that token, whether it won a residual slot. On 1.5B 2k@0.0 the
needle tokenises as `Z|EB|RA|-|4|4|7|1|-|QU|ART|Z`, so the missing `A` lives
inside token `RA` at abs 32. That token's block reports:

    anchor=0 tok=32 row=31 T_active=256 len(token_indices)=257
    n_sel=128 row_selected=True neighbours_selected=[27..35]
    rows[-3:+5]=[' Z','EB','RA','-','4','4','7','1']

The ENTIRE needle run wins exact slots, the budget is not truncating (n_sel=128 ==
max_residual), and the anchor `+1` offset arithmetic checks out. Both halves are
exact and share one index set (`lowrank.py:1686-1703`). The compressed half's RoPE
is also correct — under `DKV_ROTATED_POOL` the prefill reader passes `cos=1, sin=0`
so the JIT'd kernel's unconditional rotation is a no-op. So the `A` is stored
exactly and still does not survive: **the fault is downstream of compression, in
how prefill CONSUMES these blocks.**

*Per-layer divergence: THERE IS NONE, and the earlier "layer 0" reading was a
PROBE BUG.* The caveat recorded here — that both engines emit the same first
token, which cannot coexist with a near-uncorrelated layer-0 output — was the
right thread to pull. `Capture` enabled its hook for the whole DKV `generate()`
and every hook OVERWRITES, so the DKV vector saved was the last forward to run:
a DECODE step. `run_dense` enables the hook only on the final PREFILL chunk. The
comparison was therefore dense-prefill vs DKV-decode — two different queries.
Fixed by skipping `L == 1` forwards, which aligns both arms on the last prefill
position (accounting now printed: DKV `taken=84 skipped_decode=28`, i.e. exactly
one decode forward per layer had been clobbering the result).

Re-measured, 1.5B 2k@0.0, **every one of 28 layers**:

    cos >= 0.99967  (worst, layer 2)      rel_err <= 0.026
    |dense| 9.8415 vs |dkv| 9.8459 at layer 0

**DKV's prefill is numerically near-identical to dense's.** So the earlier
statement in this file that the token is "lost on the PREFILL side" is WRONG and
is retracted: prefill reproduces dense to 4 significant figures, and the `A` is
still exact in the pool (see the residual trace above). The loss therefore
happens during DECODE — across the 24 generated tokens — even at 2k where
`DKV_COMPRESSED_MIN_CTX=8192` means the *compressed* decode path is off. "Off"
there still means exact-dense attention over RECONSTRUCTED KV, not over the
original KV, so it is not a no-op.

*Decode-step diff — DONE, and it closes the question.* `run_dense` now generates
greedily and both arms capture decode steps (`--mode compare` reports them).
1.5B 2k@0.0:

| step | min cos | @layer | max rel | mean cos |
|---|---|---|---|---|
| 0 | 0.99451 | **27** | 0.132 | 0.99952 |
| 6 | 0.98980 | **27** | 0.181 | 0.99934 |

Every step's worst layer is 26 or 27 — the last two — while the mean across
layers is 0.9995. So DKV's decode is near-exact overall with a ~0.5-1% error
concentrated at the top of the stack.

**Why that is enough to fail the case, and why it is not a bug.** The probe now
also prints the greedy top-2 margin per step. The 1.5B's very first output
decision is a coin flip:

    step 0   'BR' 18.703   runner-up 'BA' 18.516   margin 0.1875
    step 1    '4' 20.078   runner-up  '-' 18.656   margin 1.4219

Those two branches are exactly the three outputs seen all along —
`ZEBR4471QUARTZ`, `ZEBA-4471-QUARTZ`, and the passing `ZEBR-A-4471-QUARTZ`. A
0.19-logit margin does not survive a 0.5-1% perturbation in the final layers.

The clincher is that **dense disagrees with dense**: `validate_cuda_dkv.py`'s dense
control emits the passing `ZEBR-A-4471-QUARTZ`, while this probe's dense arm — same
model, same prompt, same fp16, same greedy decode, chunked AND unchunked — emits
the failing `ZEBR4471QUARTZ`. Two correct dense implementations land on opposite
sides. A case that cannot distinguish dense from dense cannot be evidence about
DKV.

*So the `A` is never "dropped".* It is exact in the pool (residual trace), prefill
reproduces dense to 4 significant figures, and decode is a sub-1% perturbation.
The 1.5B validator cases fail because the model is nearly undecided and any
approximation tips it. Qwen3.5-2B is 9/9 because it is confident; the 1.5B on
MLX's bench is 9/9 for the same reason.

*Also eliminated (byte-identical output):* `DKV_LAYER_ADAPTIVE_RANK=0`, i.e. flat
rank with no late-layer halving. Worth recording because the drift DOES sit in
layers 26-27, which `get_layer_rank` gives `0.5 * base_rank` — an inviting story
that the measurement refutes. Rank cannot matter here: the needle is already
served exactly, so its fidelity does not depend on rank at all.

**Do not "fix" the 0/6.** The remaining lever is not DKV fidelity — it is that
this bench, on a 1.5B, decides on 0.19-logit margins. If small-model recall
matters, make the bench discriminative first (a needle the model tokenises
unambiguously, or scoring that tolerates the `ZEBR`/`ZEBA` split), then re-measure.

*Hardcodes found while looking (the user asked; both are real):*
* `native_block_pool.py:121` — `self._needs_legacy_slots = True`, with the real
  form `not (_is_cuda_dev and _gpu_compress)` sitting COMMENTED OUT on the line
  directly above. This is what made the stratified-slot race live on CUDA.
* `kv_runtime_manager.py:589` and `:1002` — `max(pool_block_size, 257)` and
  `max(block_size, 257)`, a magic 257 in two places that the routing-K formula
  (`4096 // 257 -> 15 -> clamped to 16`) then depends on.

**Closing 0/6 -> 4/6: what it is NOT.** Four knobs tested on the 1.5B validator
bench, none of them the cause — do not re-run these:

| tried | result |
|---|---|
| `DKV_MAX_RESIDUAL=256` (2x budget) | 1/9 -> 2/9, and `8k@0.5` REGRESSED to `'ZE654'`. Noise, not a fix. |
| `DKV_RANK_BOOST=auto` (1.5x rank) | outputs **byte-identical** to baseline — reconstruction fidelity is not binding on these tokens |
| block-coverage stranding | the `BLOCK COVERAGE` warning never fires in any run |
| `DKV_SPARSE_BIAS=0.0` (exact merge) | **much worse**, 0/9, outputs collapse to `'ZE'` — `auto` is helping and is the correct default |

Two of those are informative beyond being negatives. Byte-identical output under a
1.5x rank change says the needle's tokens are almost certainly already served
EXACTLY (as residuals), so the loss is not low-rank error on the needle itself.
And the failure reproduces at **2k**, where `DKV_COMPRESSED_MIN_CTX=8192` means
compressed DECODE never engages — so the token is lost on the PREFILL side, in
the smallest and cheapest case to debug. Start there, not at 32k.

Residual selection is already MLX-parity (joint absolute `sqrt(eK^2+eV^2)`, one
index set, `lowrank.py:1625`), and `compute_boost_multipliers` is a faithful port
of MLX's `is_core`/`is_prose`/segment logic — verified line by line, so neither is
worth re-auditing.

### Qwen2.5-1.5B-Instruct is NOT a working configuration — and the defect is a RACE

`--long` on Qwen2.5-1.5B-Instruct is **0/9 at HEAD and 0/9 with both fixes**, with
3 DISTINCT OUTPUTS AT TEMPERATURE 0 in 8 of 9 cases and degenerating text
(`'ZE{[]) [Z [Z [   [Z []));'`). The prefill router is not the cause: SP-TRACE
shows the needle's block kept at rank 0-2 of 120 in every layer of `32k@0.9`.

The dense control settles who owns it. Same model, same harness, DKV disengaged:
**4/6 recall and EVERY case deterministic** (1 distinct output across 3 runs), and
its two misses are near-miss spellings — `'ZEBA-4471-QUARTZ'`, one letter — i.e.
the model's own limit, not corruption. So:

* the **nondeterminism is DKV's** (dense is deterministic, DKV is not),
* it is **pre-existing** (identical at HEAD), and
* it is a **different defect class** from the prefill routing fixed above —
  fixing routing cannot fix a race, which is why 0/9 did not move.

**FIXED.** `write_blocks_batched` did not clear a recycled slot's stratified
group. Determinism on Qwen2.5-1.5B went **1/9 -> 9/9**; the garbage outputs are
gone and every case now returns one stable answer. Qwen3.5-2B is unaffected
(still 9/9 / 9/9). Details below; the history is kept because the eliminated
hypotheses are what make the next slot bug cheap.

*The bug.* `write_block` clears `U_sem`, `U_sem_scale`, `U_fact` and sets
`n_semantic` on every write. `write_blocks_batched` -- whose docstring claims it
"Mirrors write_block() field-for-field" -- cleared none of them, nor the fact
anchors. So a recycled slot inherited the previous occupant's SEMANTIC SPLIT, and
the reconstruction divided the new block at the old block's boundary, reading
`U_sem`/`U_fact` bytes belonging to another block.

It is not inert on CUDA: `_needs_legacy_slots` is HARDCODED `True`
(`native_block_pool.py:121`, with `not (_is_cuda_dev and _gpu_compress)`
commented out on the line directly above), so those tensors ARE allocated on
exactly the path that uses the batched writer when `gpu_compress` is on.

*Why it hid.* A FRESH slot is zeros from `torch.zeros`, so the first prompt in a
process is always clean and every single-prompt test passes. Only a RECYCLED slot
carries the stale split. `test_write_blocks_batched_parity` cannot catch it
either: it writes both pools from clean, where "cleared" and "never written" are
indistinguishable. The recycling IS the test --
`test_write_blocks_batched_clears_slot.py` pins it and fails without the fix.

*What is still wrong on 1.5B, and is NOT this bug.* Recall is 1/9 (was 0/9).
The outputs are now clean near-misses -- `ZEBR4471QUARTZ` (drops the A),
`ZEBA-4471-QUARTZ` (drops the R) -- not corruption. Dense on the same model is
only 4/6 and misses the same way, so 1.5B's ceiling is well below 9/9 regardless
of DKV. Parity for this model means matching DENSE, not matching the 2B.

*Reproduce in ~3 min, not ~10.* The bug needs TWO DIFFERENT prompts in ONE
process — a single case in a fresh process is deterministic and correct on this
model, so any one-case repro is blind to it. `validate_cuda_dkv.py` seeds `random`
ONCE and draws all cases from that stream; reseeding per case yields byte-identical
prompts that the caches answer trivially. Two cases (2k@0.0 then 2k@0.5) is enough
and reproduces the validator's exact outputs. Structure: case 1 deterministic,
case 2 rep 1 CORRECT, reps 2-3 garbage.

*The mechanism, established by measurement.* `DKV_NO_SLOT_REUSE=1`
(`native_block_pool.free_block`, diagnostic-only, leaks slots — never a fix)
makes case 2 fully deterministic AND correct. So the corruption is a **stale
block -> pool-slot mapping surviving slot recycling** — the same class as the
`_decode_block_cache` bug whose fix is documented in `clear_session`, which that
comment says left 1 of 7 failures unaccounted for. This is very likely that one.

*Eliminated by measurement — do not re-test these:*

| hypothesis | how it was excluded |
|---|---|
| background mutators / async SVD | `DKV_DETERMINISTIC=1` — race unchanged |
| decode cache | `DKV_DECODE_CACHE=0` — race unchanged |
| tiered eviction (LRU by wall-clock) | `DKV_TIER_ENABLED=0` — race unchanged |
| block prefetch engine | `DKV_BLOCK_PREFETCH=0` — race unchanged |
| pool exhaustion | 1.5B pool is BIGGER than 2B's (95k vs 88k tokens) |
| prefix-cache reuse | "Reusing KV cache" never printed in any run |
| prefill routing | no SP-TRACE at 2k — the router does not even engage, yet it races |
| SRL `_route_layer_cache` step-counter reset | added the guard; race unchanged; reverted |

*Where to look next.* Both manager-level caches ARE cleared correctly
(`_decode_block_cache` filtered by `key[0]`, `decode_workspace` popped), so the
surviving holder is elsewhere. `native_block_pool` maintains `self.version[slot]`,
incremented on every `write_block` — and NOTHING reads it. A reader that captured
a slot id could be validated against it. Note also that CUDA writes
`_sp_pinned_blocks[session_id]` at `hf_dkv_wrapper.py:974` and never reads or
clears it, while MLX reads it, resets it to `()`, AND deletes the entry
(`mlx_dkv_wrapper.py:4995-5541) — dead state on CUDA today, but a real parity gap.

The instrument that found the last one was a WRITE MAP vs ROUTE TRACE diff
(writer slot vs reader slot per layer/anchor per generation); rebuild that rather
than guessing at more flags, which is what the table above cost.

**Measured 2x2** (Qwen2.5-0.5B-Instruct, `ACTIVE_RUNTIME/tests/test_niah.py`,
8k NIAH, deterministic; all 4k cases pass in every cell):

| config | 8k@0.1 | 8k@0.5 | 8k@0.9 |
|---|---|---|---|
| HEAD (neither fix) | pass | **FAIL** | **FAIL** |
| router fix only | pass | **FAIL** | **FAIL** |
| RoPE fix only | **FAIL** | **FAIL** | pass |
| **both** | pass | pass | pass |

Read this carefully: **neither fix alone recovers anything**, and the RoPE fix
alone MOVES the failure rather than removing it. They are complementary — the
router decides which blocks the prompt is read against, the rotation decides
whether those blocks' keys mean anything. Fixing one while the other still
corrupts prefill is indistinguishable from fixing nothing, which is the same
shape as the "8 rounds of byte-identical decode fixes" this document opens with.

COVERAGE, stated plainly: this is Qwen2.5-0.5B at 8k on an RTX 4070, the same
failure class (deep needle, sparse-prefill regime, temp 0) but NOT the
Qwen3.5-2B/32k configuration §1 describes. `validate_cuda_dkv.py --long` has not
been re-run — that model is not on this box. Sparsity was confirmed live, not
assumed: `DKV_SP_TRACE_TOKEN` reports `k_eff=8` of `12` routable, so this is not
§3's "accidentally reproduced `DKV_SPARSE_PREFILL=0`" trap.

---

## 1. WHERE THINGS STAND

`python colab/validate_cuda_dkv.py --long` (Qwen3.5-2B, 9 NIAH cases):

| config | result |
|---|---|
| default | **8/9** — `32k@depth0.9` gives `'None'`, deterministic at temp 0 |
| `DKV_SPARSE_PREFILL=0` | **9/9** ✅ |
| MLX reference (`mlx_needle_parity.py --long`, on a Mac) | 9/9 |
| dense control (same weights, DKV disengaged) | 9/9 |

So the failure is DKV's, not the model's or the prompt's, and it is **in prefill**.

`DKV_SPARSE_PREFILL=0` is a *diagnostic*, not the fix — it restores O(L²) prefill
and discards the entire point of sparse prefill. **Your job is to make sparse
prefill correct, i.e. match MLX.**

---

## 2. THE ROOT CAUSE

CUDA runs block-sparse attention **during prefill** (`DKV_SPARSE_PREFILL` default
`"1"`, `ACTIVE_RUNTIME/runtime/dkv_attention.py:412`). Each chunk attends only its
routed blocks plus a recency window. When prefill routing misses the needle's
block, **the model's own hidden states never absorb that sentence.** The query at
decode is then not "degraded by noise" — it is the query of a model that
effectively never read the needle.

This is why ~8 rounds of decode-side fixes produced byte-identical output: they
were all downstream of a defect that had already happened.

### The specific divergence — `dkv_attention.py` ~line 497

```python
anchor_ks = torch.stack([b.anchor_kv[0, 0] for _, b in valid], dim=0)  # [nb,H_kv,D]
q_repr    = chunk_q[0].mean(dim=(0, 1)).float()                        # [D]
scores    = torch.einsum("nhd,d->nh", anchor_ks, q_repr).mean(dim=1)   # [nb]
top_idx   = torch.topk(scores, k=k_eff).indices.tolist()
```

Two departures from MLX:

**(a) ANCHOR-ONLY SCORING.** An anchor is a single token — the block's first.
A needle at within-block offset 232 of 257 contributes *nothing* to it, so this
router is structurally blind to content buried deep in a block. MLX scores
`_block_relevance_residual` = **anchor + the block's top-R exact residual keys**
(`mlx_dkv_wrapper.py:1156`), and the residuals are precisely each block's
highest-error / most distinctive tokens — which is what a random code is.

This is the SAME bug already fixed on the decode side in `e48cc31`
(`route_blocks_relevance` scored bare `q·rk` on anchor-relative residuals without
adding `s_anc` back, so `maximum(s_anc, q·rk) → s_anc` always ⇒ anchor-only).
**Prefill never received that fix.**

**(b) MEAN-POOLED QUERY.** `chunk_q.mean(dim=(0,1))` collapses all heads *and* up
to 1024 chunk tokens into one `[D]` vector. Retrieval is head-specialised;
averaging a retrieval head with seven others erases exactly the signal that finds
a needle. MLX scores per-head and reduces with max.

### The fix — SUPERSEDED BY §0.5, kept for the reasoning

This section said: make prefill call `route_blocks_relevance` (the decode
router). **Do not do that** — see §0.5(2). MLX's prefill router is
`_block_relevance_minmax`, the decode router's residual term is gated off for 3D
queries and would blow up memory if it were not, and during a fresh prefill there
are no compressed blocks to score residuals from in the first place.

What shipped instead (`_prefill_block_key_boxes` + `_sparse_prefill_relevance` in
`dkv_attention.py`): per-block min/max over the block's exact keys, scored
per-head against every chunk token, max-reduced over both axes — MLX `:1098`
transcribed. Compressed blocks (2nd-turn prefill only) fall back to anchor + the
anchor-relative EXACT residuals, which is where the `_exact_keys_enabled` warning
below still applies and is honoured.

One deliberate departure, and it is an identity rather than an approximation:
MLX's elementwise `sum_d max(q_d·min_d, q_d·max_d)` materialises `[H_q, L, nb, D]`
(~4 GB at 32k). Since `max(a,b) == (a+b)/2 + |a-b|/2` and `max_d >= min_d`, the
same bound is `q·mid + |q|·half` — two GEMMs, largest tensor `[H_kv, gpk·L, nb]`.
Same numbers, same selection; `tests/test_sparse_prefill_router.py` pins it
against the literal MLX form.

### Parameters are NOT the divergence — don't waste a run there

`DKV_SPARSE_PREFILL_MIN` 2048, `..._WINDOW` 1024, `..._KMIN` 8 already claim MLX
parity, and `..._FRAC` is 0.25 vs MLX's 0.05 — CUDA attends **strictly more**
blocks than MLX. The block *count* is not the problem. The block *scoring* is.

---

## 3. VERIFICATION PLAN (pre-decide the reading)

1. Baseline on the new box first: `python colab/validate_cuda_dkv.py --long`
   must reproduce **8/9** with `fallback_count=0`. This suite has been
   invalidated before by config drift (a validator that didn't apply
   `BEST_DECODE_DEFAULTS`; an env reset that bumped `transformers` to a version
   the track can't use). Do not read any new number until the baseline matches.
2. `DKV_SPARSE_PREFILL=0` must give **9/9**. Confirms the diagnosis transferred.
3. After the router fix, default config must give **9/9** — *and* prefill must
   still be sparse. Check the sparse path actually ran; if your change silently
   makes `k_eff >= len(routable)` the function returns all blocks and you have
   accidentally reproduced `=0` while believing you fixed routing.
4. Re-check throughput. The point of sparse prefill is speed; a "fix" that
   attends everything is the workaround wearing a disguise.

---

## 4. WHAT IS ALREADY PROVEN CORRECT — do not re-investigate

Each of these was established by measurement this session. Re-testing them is
how you lose a day.

* **The stored key is EXACT.** `colab/probe_residual_values.py`:
  `anchors_K + residual_K_values` vs `RoPE(k_proj(h),pos)` = **cos 1.0000**,
  rel_err 3e-4, at every layer, *identically on the failing 0.9 and passing 0.5*.
* **Decode routing works** — the needle's block ranks 0–1 of 16, its row is
  unmasked, its offset resolves correctly.
* **Sparse-half math is MLX-equivalent** — `delta_s + s_anc` with residual twins
  masked, `1/√D` applied equivalently, padding masks matching.
* **The merge is not the cause** — the MLX partition plus a genuinely adaptive
  `auto` moved nothing; remat (a single unbiased softmax with no merge at all)
  fails identically; MLX passes 9/9 at its own `0.0` default.
* **Also eliminated:** residual capacity (128, already MLX-equal) and selection,
  attend-all, TF32, the RoPE clamp (instrumented — never fires), and the rotation
  convention (documented A/B in `pool_stores_rotated_k`'s docstring:
  `32k@0.9 unchanged, still 0/3` both ways).
* **Per-layer output cosine vs dense is a DEAD observable** — the *passing* case
  measures a worse cosine (0.276) than the failing one (0.291).

---

## 5. TRAPS THAT MADE EXPERIMENTS VACUOUS

1. **`DKV_RESIDUAL_EXACT_ROPE` is dead on the remat path by construction.**
   `do_rot = (... and not pool_stores_rotated_k())`, while `_remat_attend`
   *declines* unless `pool_stores_rotated_k()` is true. Toggling the flag is
   byte-identical — that is evidence the flag is dead, **not** evidence about
   rotation.
2. **The rotation A/B was already in the source** —
   `triton_fused_decode.py:237-310`. Read docstrings before designing a run.
3. **A `tl.constexpr` declared on a Triton kernel but not passed at the call site**
   raises `TypeError`, which the `try/except` swallows into a silent PyTorch
   fallback. Production then runs something different from what you are testing.
   This already happened once with `S_MAX`. Always confirm `fallback_count=0`.
4. **Swallowed exceptions generally.** `except Exception: # SRL failure is
   non-fatal` hid a router that never ran for SEVEN consecutive "fixes."
   **When N changes in a row do nothing, verify the code RUNS before making an
   (N+1)th change.**

---

## 6. PROBE DISCIPLINE (learned the hard way, twice each)

* **Compare at decode step 0, not the last step.** By the last step the two runs
  have emitted different answers, so their hidden states differ *because* of the
  wrong answer. That is an effect, not a cause. Step 0 has identical token history
  in both runs by construction.
* **Align hidden-state comparisons on ABSOLUTE POSITION, never list index.** The
  runs emit different token counts once they diverge.
* **Layer 0's input is the token embedding and MUST read cos 1.0000.** A `0.0319`
  there is misalignment, not a discovery. *When a measurement violates an
  invariant, the measurement is wrong until proven otherwise.*
* **Always read the failing depth against the passing 0.5 control.** A number
  identical in both explains nothing. This trap was hit twice.
* **State coverage.** "The block is routed" from a probe that prints once is a
  statement about token 0, not about the token that produces the answer.

---

## 7. TOOLS

| file | what it answers |
|---|---|
| `colab/validate_cuda_dkv.py --long` | the 9-case NIAH suite; `--dense` for the control |
| `colab/probe_query_vs_dense.py` | DKV's q vs dense's q against the same exact key. **This is what found the bug.** `--mode dkv`, then `--mode dense`; `--mode compare` re-analyses caches with no GPU |
| `colab/probe_residual_values.py` | is the stored key correct? (`--depth 0.5` = passing control) |
| `colab/probe_needle_block.py` | is the needle selected into its block's residual set? |
| `colab/mlx_cuda_parity.py` | side-by-side MLX/CUDA harness — this found the dead router that reading missed 7 times |
| `ACTIVE_RUNTIME/tests/` | CPU tests; run before any GPU turn |
| `ACTIVE_RUNTIME/tests/test_niah.py` | 6 NIAH cases on Qwen2.5-0.5B (cached locally, no download) — the cheapest end-to-end signal that exists, ~80s |
| `ACTIVE_RUNTIME/tests/test_sparse_prefill_router.py` | prefill router vs the literal MLX formula; the buried-needle regression also asserts the OLD router FAILS it, so it cannot silently pass on both sides |
| `DKV_SP_TRACE_TOKEN=<abs token idx>` | prefill counterpart of `DKV_ROUTE_TRACE_TOKEN`: per layer, does that token's block survive routing, at what RANK, against what CUTOFF, and out of how many candidates. Says so explicitly when the token is not yet ingested at that chunk, instead of resolving to the last block and reading like an answer |

Windows note: `pytest -s` writes through a cp1252 console and dies on the `→` in
`hf_dkv_wrapper._trim_python_heap`'s print. `PYTHONIOENCODING=utf-8` fixes it;
the traceback is not about your change.

Known rough edge: `probe_query_vs_dense.py --mode compare` throws
`TypeError: iteration over a 0-d tensor` on caches written before the
position-alignment change. Delete `/tmp/dkv_qprobe_*.pt` and re-run both passes.

---

## 8. OPEN ITEMS

* **Re-run `validate_cuda_dkv.py --long` on Qwen3.5-2B/32k.** Both §0.5 fixes are
  in, and 8k NIAH went 4/6 -> 6/6 on Qwen2.5-0.5B, but the model in §1's table has
  not been touched since. Pre-decided reading: 9/9 with `fallback_count=0` and
  `DKV_SP_TRACE_TOKEN` still showing `k_eff < nb` means both defects were the
  whole story; 8/9 with the needle's block RANKED but dropped means K is too small
  (a parameter, `DKV_SPARSE_PREFILL_FRAC`); 8/9 with it kept means a third defect
  downstream of routing.
* **Controlled prefill-throughput measurement.** Wall clock on the 3 8k cases was
  47.0s (HEAD) vs 48.7s (fixed), but the two runs emit different text once one of
  them starts answering correctly, so that number is not a throughput result. The
  RoPE fix strictly REMOVES work (no gather + rotate of dense history per chunk
  per layer); the router adds a cached min/max plus two GEMMs in place of one
  einsum. Measure it properly before quoting a number.
* **`_apply_rope_single` at `dkv_attention.py:3344`** — same double-rotation
  shape, but it is inside the MPS `_validate_this_step` branch, not production
  CUDA, so it was left alone. Fix it if that validation path is ever trusted.
* **`ingest_streaming` frame is inconsistent across prefill paths.** The
  chunked-sparse path (the one that fails --long) captures via `_ingest_k`, but
  INCREMENTAL prefill (`dkv_attention.py:3526`) passes raw `unrot_key_states` and
  `finalize_contiguous_prefill` inverse-RoPEs before calling `capture_prefill_kv`.
  Under the rotated-pool default those two write the pool in the OPPOSITE frame
  from the first. Not exercised by 1st-turn NIAH; a 2nd-turn session is where it
  would show. Route every capture site through `_ingest_k`.
* **`tests/test_niah.py` is only meaningful RUN ALONE.** Run as part of the full
  suite it fails all three 8k cases — *identically at HEAD and with both §0.5
  fixes in*, so it is cross-test state contamination, not a regression, and the
  full-suite NIAH result cannot discriminate anything. Ruled out by measurement:
  `DKV_SRL_THRESHOLD=5` (leaked by `test_failure_cases.py`, and named in
  `test_niah`'s own docstring as having broken 8000/0.1 before) and
  `DKV_MLA_LATENT=1` (leaked by `test_ktransformers_features.py`) — 6/6 with each
  forced. Leading unverified hypothesis: the pool budget is computed from FREE
  VRAM at init (`ceiling: 50% of 8.6 GB free VRAM` in the logs), so earlier tests
  holding memory change `max_blocks`, block sizing and therefore routing. Same
  full-suite-only pattern hits `test_triton_combined`. Whoever needs the suite
  green should fix the isolation, and until then A/B the FILE, never the suite.
* `tests/test_facter_retention.py::test_localized_vertical_factual_retrieval` is
  FLAKY, unrelated to any of this: unseeded `torch.randn`, ~1-in-5 failures on the
  relaxed-threshold fallback assertion. Seed it before it costs someone a
  bisect.
* **Prefill router alignment** — section 2. Done; see §0.5.
* **64k** — untested. Depth 0.9 there puts the needle in the same relative
  position, so the same failure is *expected* but unverified. Re-check after the
  fix. Note the routed-row count does not grow with context (K=16 regardless), so
  "more context = more competitors" is NOT the mechanism.
* **`DKV_RESIDUALS_IN_DENSE`** (commits `53d928d`, `6dab025`) — real MLX parity:
  exact residual rows belong in the DENSE half (`mlx_dkv_wrapper.py:1031`), and
  with them in the sparse half `DKV_SPARSE_BIAS=auto` had the wrong sign and was
  pinned near +2.0, then disabled outright. Verified safe (no regression at bias
  0.0 or `auto`) but it does **not** fix `32k@0.9`. Default OFF. Ship it only on
  its own merits, with its own measurement.
* **`DKV_REMAT_CACHE`** — stays default OFF. Its old speed numbers came from a
  broken 1.5B build; on the 2B it measured 21.1 vs 20.1 ms/token, inside noise.
* Prefill throughput work generally — the reason sparse prefill exists.

---

## 9. ARCHITECTURAL NOTE — why this class of bug exists on CUDA and not MLX

MLX compresses as a **side-effect of the model's own forward pass**:
`keys_rot = self.rope(keys, offset)` (`:4565`), ordinary attention with those
keys, then `manager.ingest_streaming(keys_rot, ...)` (`:4613`). Its hidden states
are structurally identical to dense's, and there is no separate prefill
implementation that *can* drift.

CUDA **replaces the attention implementation** (`DKV Attention Interception
Applied`) in prefill as well as decode. That is what allows prefill to produce
different hidden states than dense — the precondition for this entire bug.

If you find yourself repeatedly patching prefill to behave like dense, the
deeper fix is to make CUDA's ingest a side-effect of normal attention the way
MLX's is, rather than a substitute for it. That is a large change; do not start
it without agreement.
