
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
