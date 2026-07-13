
# Relational binding — root-cause program

(The original RC1-RC8 report file was not committed; the RC1-RC8 fix program
status lives in the session memory `project_relational_binding_progress` —
all eight implemented in both impls as of 2026-06-14, RC3/RC5/RC8 pending CLI
validation. This file restarts the record at the compression layer.)

## 2026-07-12 — Layer 1 closed: owner-capture at the compression layer

The RC1-RC8 program treats binding at the GENERATION layer. A controlled
probe (`benchmarks/binding_probe.py`: 6 planted entity→value pairs at 8k, three
scored outcomes per fact — correct / swapped / missing) found a lower layer was
broken first: residual capture kept fact VALUES exact (digits are is_core) but
entity NAMES are title-case → is_prose → never captured, surviving only as
rank-r reconstruction. The decoder then emits real values bound to CORRUPTED
names ("Okazaki"→"Okinawa"/"Okapi", "Brancusi"→"Bruckner"/"Brancos"):
- MLX list-all: dense 5/6 vs compressed 1/6 (no swaps — the names are corrupted,
  not exchanged); entity→value 4/4 (query carries the name) vs value→entity 1/4.
- Fix: `DIFFKV_RESIDUAL_OWNER_CAPTURE` (default ON) — for each core segment walk
  left to the nearest capitalized non-function word, expand to the full surface
  run (subword continuations + multi-word names), boost those rows into the
  exact-residual set; plus a budget floor (boosted_count+4) so the adaptive
  8/16 cap can't evict the owner. `_apply_owner_capture` in
  mlx_diffkv_wrapper.py (both capture sites) + the mirrored block in lowrank.cpp.
- After: MLX list-all 6/6 (beats dense), corruption class ELIMINATED both
  runtimes; native OC0 "Brancos/Okla/Pierre/Rochester" → OC1 all six names
  surface-perfect. Recall gates all green (MLX NIAH 4k-32k exact incl. 16k/0.9
  displacement cell, MN 3/3; native 6-cell 6/6, MN 3/3, margins 12.4/13.1).
  Cost: native synthesis 16k cell 26.7→10.0 (knife-edge metric, 8k identical
  at 26.7 incl. linkage; controlled — owner=0 on same binary reads 26.7).

**What remains is layer 2 — true association swaps**: with names AND values both
exact, native still swapped adjacent pairs (Halvorsen↔Okazaki), and dense MLX
makes the same class of error on value→entity (2/4 with swaps). That is the
decoder's own binding limit — RC5/RC8 generation-path territory (or
neighborhood co-retrieval), NOT capture. The torch/CUDA path lacks the boost
machinery entirely and still has the layer-1 disease — see CUDA_TRITON_AUDIT.md
C10.

## 2026-07-12 (later) — Layer 2: enumeration-order association, coverage scaffold

With owner-capture in, the cross-runtime probe found the next layer precisely:
- Native retrieval binding is EXACT in both directions (fwd 3/3, rev 3/3 —
  native rev beats MLX, whose rev swaps 2/4 even DENSE → that residue is the
  4-bit model's own margin, not DiffKV).
- But native list-all ENUMERATION transposed adjacent names (3/6; values walked
  document order, names didn't) — dense native enumerates correctly, so the
  compressed context was missing the positional scaffold through filler that
  ordered scanning needs.

Fix: stride-stratified residual coverage (existing knob) — measured dose-response
(binding list-all / multi-needle 16k): 0 → 3/6 + 3/3; 0.125 → 4/6 + 3/3;
**0.25 → 5/6 + 3/3 (native default now)**; 0.5 → 6/6 + 0/3 (multi-needle
suffix displacement: "OMEGA-7741-BETA" — needle stems exact, Greek suffixes
scrambled). Two structural bugs found and fixed on the way:
1. **Coverage rows were ordered FIRST in the residual arrays** (+1e12 selection
   bonus) — but the residual head doubles as the block's ROUTING signature
   (decode relevance reads the first route_residuals rows). Coverage now gets
   selected but APPENDED after the ranked rows, both runtimes.
2. **Coverage-vs-floor eviction**: the boosted-row budget floor now reserves
   room for the scaffold too (need/(1-cov_frac); margin env
   DIFFKV_RESIDUAL_FLOOR_MARGIN, default 4). Also aligned the quota semantics
   (fraction of block budget) across runtimes, ASCII→Latin-1 uppercase parity
   in the C++ owner walk.

Final native default config (owner ON + coverage 0.25): NIAH 6/6, multi-needle
3/3, margins 12.48/14.26 (baseline 12.62/14.32), synthesis **26.7/26.7 = full
baseline both ctx** (the owner-capture 16k cost is erased), binding list-all
5/6 (from 3/6), zero name corruption. MLX default coverage stays 0 (its
enumeration was already 6/6; flip only with fresh MLX measurements).

Remaining layer-2 tail: one association slot (Ellsworth took Halvorsen's value
at 8k list-all) and the 4-bit MLX rev-lookup swaps — both are the decoder's own
capability envelope (dense exhibits the same class). Next lever if pursued:
RC5/RC8 end-to-end validation or neighborhood co-retrieval at decode.

## 2026-07-12 (later) — RC5/RC8 end-to-end validation: the honest verdict

The RC1-RC8 program's generation-path validators (RC5 comparison sequencing,
RC8 foreign-token license) were "implemented, never validated end-to-end." Done
now, with a comparison/REV probe (`benchmarks/binding_probe.py` cmp_q/rev_q) and
a DENSE control. Findings:

1. **The probes never exercised RC5/RC8.** The binding/synthesis harnesses do
   raw `np.argmax(output.logits)` (MLX) or the native CLI. RC8's logit license
   lives in `batch_engine._sample` (MLX) — bypassed by argmax — and was
   dead-commented in native (main.cpp). The whole RC stack is gated on the
   factual store (`current_step_factual_sequences`), default OFF. So every prior
   binding number was measured on a path where RC5/RC8 are inert. The "remaining
   tail = decoder envelope" claim was therefore unproven — until now.

2. **RC5/RC8 target failures that are already fixed.** 2-entity comparisons
   (RC5's design target — the swap-prone Okazaki/Halvorsen pair) and forward
   lookups bind CORRECTLY on the default path (owner-capture + coverage-0.25),
   in dense, compressed, AND compressed+factual. There is no interleave
   inversion left for RC5 to sequence away.

3. **The live remaining failure is value→entity REV, and it is a BASE-MODEL
   limit.** REV ("which facility processes 4382?") is 3/6 on the compressed
   default (swaps collapse to a neighbor/attractor: 4382→Okazaki, 8617→
   Halvorsen, 5248→Halvorsen). **DENSE is byte-identical — same 3/6, same three
   swaps.** Compression is bit-faithful to the model here; there is nothing in
   the KV/compression/routing/capture layer to fix. RC8 does not target REV
   anyway (no entity is locked when the entity is the unknown answer).

4. **RC5/RC8's prerequisite is net-negative and makes output worse.** Turning
   the factual store on (to feed RC5/RC8) derails multi-entity generation into
   filler-copying ("…10s with massive datasets and GPU compute. The history of
   artificial intelligence…"), and REV factual-on stays 3/6 but adds derail.
   RC8=1 live reproduces the original disable reason (Bug 🅗): on the
   Okazaki/Halvorsen comparison it SUPPRESSED "Halvorsen" (foreign while locked
   to Okazaki) → "Okazaki processes 7156. The facility processes 2903." — the
   second entity's own name licensed away.

**Decision (evidence-directed): RC5/RC8 stay OPT-IN, and the cross-runtime
divergence is resolved.** RC8 is unified behind `DIFFKV_RC8_LICENSE` (default
OFF) in BOTH runtimes — uncommented+gated in native (main.cpp), wrapped in MLX
(batch_engine.py). Default OFF changes nothing (it only ever fired under the
default-OFF factual store) and is now consistent + runnable for anyone A/B-ing
comparison-heavy factual workloads. RC5's comparison lock is already inert by
default (needs the factual store). Verified: RC5/RC8 unit suite 22/22, native
default list-all unchanged 5/6, recall gates green.

Bottom line for the "which noun owned which number" thesis: retention and the
capture-layer binding are SOLVED (owner-capture + coverage). The residual
value→entity swaps are Qwen2.5-1.5B's own reverse-lookup ceiling, proven by the
dense control — not something DiffKV can or should paper over with generation
gates that regress other cases.

## 2026-07-13 — Layer 3 closed: TABLE binding (capture) + the serving sampler was a second, independent killer

Report (2026-07-12, NAT-style paper via the active CLI): concepts and values
survive but re-attach to the wrong rows/metrics — 83.2 migrating 7x7→3x3, a
fabricated 4x4 row, real imgs/sec throughputs replaced by invented "G/s".
Controlled reproduction (`benchmarks/table_probe2.py`: 6-row × 3-metric
kernel-size table STRADDLING a 256-token block boundary + a second
ablation-path table, planted in real paper text at 16k; scored per row
correct / mixed / swap / fab / miss): dense 6/6, compressed **3/6** (cross-
table value migration + a vanished row) — DiffKV-specific, reproduced.

**Root cause A — capture (compression side).** Tables break both existing
capture rules at once: (1) the digit cells are all is_core, so a block
holding a table plus technical filler carries MORE boosted rows than
residual slots (measured 181 boosted / 128 slots) and the err-ranked cut
drops table fragments structure-blind; (2) header/unit/row-name cells
('Kernel', 'imgs', '/sec', 'Swin-T baseline') are prose → never boosted →
rank-r smear (hence fabricated units and lost row names — Table B decoded
as bare values: "variant: 78.6"). Fix: **table capture**
(`DIFFKV_RESIDUAL_TABLE_CAPTURE`, default ON; priority
`DIFFKV_RESIDUAL_TABLE_PRIORITY`=4): every token on a table-like LINE
(>= 2 standalone '|'/'&' separators + shape guard: line-initial separator,
LaTeX `\\` terminator, or density >= 1/12 — rejects prose with inline |x−y|
math, which otherwise marked 19 false-positive blocks) gets the core boost ×
priority; native also skips the coverage quota for saturated table blocks.
Implemented in `mlx_diffkv_wrapper.py` (both capture sites, + debug dial
`DIFFKV_DBG_TABLE`), `lowrank.cpp`, and — closing audit item C10 — the
torch/CUDA batched path via the new shared
`native_core/compression/residual_capture.py` (token boost + owner capture +
table capture; CPU-tested end-to-end in `tests/test_residual_capture.py`).
Measured: MLX 16k list-all 3/6 → **6/6 == dense**; native 4/6 (row-shift
swap + missing row) → **6/6**.

**Root cause B — the serving sampler (decode side, INVISIBLE to raw-argmax
probes).** Through the CLI the table still died (header + EMPTY '| | | |'
cells; the model even claimed the table "is not provided in your original
document") while the identical compressed state read 6/6 via raw argmax.
Cause: the default repetition penalty 1.15 over the last 64 generated
tokens — after one emitted row, every digit/pipe/space of the table is
penalized and the argmax slides to wrong-but-unpenalized tokens. This
mechanism needs no compression: the dense CLI failed identically. Fixes
(all four sampler sites: batch_engine `_sample_gpu_jit` call site, the MLX
wrapper decode loop — the live one on macOS —, the HF wrapper loop, native
main.cpp):
  1. numeric exemption — digit-bearing and '|'/'&' tokens are never
     repetition-penalized (digits carry semantics, not fluency);
  2. table-line suspension — while the current output line (plus the line
     above, so row starts count) contains >= 2 separators, the penalty is
     suspended entirely (verbatim-reproduction mode). Digits-only exemption
     was measured INSUFFICIENT: with glue (spaces, 'x', '.') still
     penalized, rows came out as empty pipe skeletons, then as Table B's
     values pouring into Table A's shape once row bodies were freed but row
     KEYS still penalized — the two-line lookback fixed the row starts.
Both are suspended while a repetition LOOP is detected, so the escalated
1.3×/256 recovery still breaks digit loops ("7741-7741");
`DIFFKV_REP_PENALTY_PROTECT_NUMERIC=0` restores the old behavior.

**End-to-end (the user's exact invocation** — `serving/cli.py --preset mid
--serving-mode balanced`, 12k paper + both tables, temp 0): compressed
0/6 → **5 correct + 1 mixed**, byte-matching the dense CLI control (same
single 3x3-cell bleed) — i.e. DiffKV == dense through the full serving path;
the residue is the 4-bit 1.5B model's own envelope.

Gates (all green, both runtimes): MLX NIAH 16k d0.5+d0.9 exact, MN 3/3,
synthesis 8k 6.7 == same-day capture-off control, binding_probe list-all 6/6
zero swaps, v1 small-table 4/4; native NIAH 6-cell sweep 6/6 (re-run after
the sampler change), margins 12.4769/14.2615 (== documented baseline),
synthesis 26.7 == control, native table probe 6/6 at the DEFAULT 1.15
sampler. NIAH/synthesis/binding prompts contain no standalone separators, so
the capture change is a provable no-op there (unit-tested).

Remaining tail: Table B row-name reversal (value→name) fabricates under
BOTH dense and compressed ("Core Vector Machine" — an entity from the
filler paper) — base-model ceiling, not DiffKV. CUDA GPU cert for the
C10 port still owed (see CUDA_TRITON_AUDIT.md C10/C11).
