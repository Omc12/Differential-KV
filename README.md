# Differential KV Cache — Research Prototype

> **Status:** Phase 1 — Offline KV Simulation & Benchmarking  
> **Goal:** Validate whether adaptive anchor-based KV reconstruction reduces KV memory movement enough to justify reconstruction overhead.

---

## What This Is

A **controlled research laboratory** for an experimental transformer KV-cache memory architecture.

This is **NOT**:
- A production inference engine
- Triton/CUDA optimized
- Integrated into any serving stack
- A full transformer implementation

This is:
- A PyTorch simulation framework
- A compression/reconstruction benchmark
- A systems hypothesis validator

---

## Core Idea

Traditional KV-cache stores every token's full KV state in FP16/BF16.

**Differential KV Cache** stores:
- **Anchors**: full KV snapshots at strategic intervals
- **Deltas**: INT8-quantized residuals between anchors and each token
- **Anchor Index**: lightweight lookup mapping token ranges → nearest anchor

Reconstruction happens **on-demand**, only for attention-requested regions.

---

## Project Structure

```
differential-kv/
├── anchor_logic/        # Anchor placement strategies (periodic + adaptive)
├── compression/         # Delta encoding, INT8 quantization
├── reconstruction/      # KV reconstruction from anchor + delta chain
├── benchmarks/          # Benchmark runners vs FP16/FP8/INT8 baselines
├── profiling/           # Memory movement, compute time, locality stats
├── visualization/       # Plots: compression ratio, error, crossover curves
├── experiments/         # High-level experiment scripts
└── tests/               # Unit tests for each component
```

---

## Phase 1 — Key Experiments

| Experiment | Question |
|---|---|
| `exp_crossover.py` | At what context length does Differential KV win? |
| `exp_compression.py` | What compression ratios are achievable? |
| `exp_reconstruction.py` | How much does reconstruction overhead cost? |
| `exp_anchor_density.py` | How anchor frequency affects compression vs error |

---

## Quick Start

```bash
pip install -r requirements.txt

# Run the crossover experiment (most important)
python experiments/exp_crossover.py

# Run full benchmark suite
python benchmarks/run_all.py

# Generate all visualization plots
python visualization/plot_all.py
```

---

## Key Metrics

- **Compression Ratio**: bytes_stored / bytes_original
- **Reconstruction Error**: L2 norm of (reconstructed - original)
- **Memory Movement**: bytes moved during attention serving
- **Reconstruction Latency**: time to reconstruct requested token range
- **Tokens/sec Approximation**: effective throughput estimate

---

## The Defining Question

> **At what context length does Differential KV become worthwhile?**

This crossover point is the central finding of Phase 1.

---

## Future Phases

- **Phase 2**: Triton kernels + fused reconstruction
- **Phase 3**: FlashAttention integration + tiled attention
- **Phase 4**: Serving integration + production benchmarks
