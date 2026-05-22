"""
experiments/exp_reconstruction.py

Experiment: How much does reconstruction overhead cost?

Tests:
  1. Single-token reconstruction latency
  2. Grouped reconstruction (contiguous window) latency
  3. Reconstruction vs window size: latency as window grows
  4. Reconstruction reuse: does repeated queries to same region amortize?
  5. Reconstruction depth: how anchor distance affects error

Key question:
  "Is reconstruction overhead manageable enough to justify memory savings?"
"""

import sys
import json
import time
import random
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy, AdaptiveAnchorStrategy
from reconstruction.reconstructor import KVReconstructor
from profiling.profiler import ReconstructionProfiler


NUM_HEADS  = 32
HEAD_DIM   = 128
SEQ_LEN    = 32768
STRATEGY   = PeriodicAnchorStrategy(interval=64)


def test_single_vs_grouped(kv, manager, recon):
    """Compare per-token vs grouped reconstruction latency."""
    results = {}
    random.seed(42)
    N = 50

    # Single-token
    single_times = []
    for _ in range(N):
        idx = random.randint(1, SEQ_LEN - 1)
        t0 = time.perf_counter()
        _ = recon.reconstruct_token(idx)
        single_times.append((time.perf_counter() - t0) * 1000)
    results["single_token_mean_ms"] = round(sum(single_times)/N, 4)
    results["single_token_p95_ms"]  = round(sorted(single_times)[int(0.95*N)], 4)

    # Grouped windows of various sizes
    for window in [16, 32, 64, 128, 256, 512]:
        times = []
        for _ in range(N):
            start = random.randint(0, SEQ_LEN - window - 1)
            end   = start + window - 1
            t0    = time.perf_counter()
            _     = recon.reconstruct_range(start, end)
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = round(sum(times)/N, 4)
        per_tok = round(mean_ms / window, 5)
        results[f"grouped_{window}_mean_ms"]  = mean_ms
        results[f"grouped_{window}_per_tok_ms"] = per_tok

    return results


def test_error_vs_anchor_distance(kv, num_heads, head_dim):
    """How does reconstruction error grow as distance from anchor increases?"""
    manager = AnchorManager(strategy=PeriodicAnchorStrategy(interval=256))
    manager.compress(kv)
    recon = KVReconstructor(manager)

    # Find first anchor, then measure error at distances 1, 8, 16, 32, 64, 128, 255
    first_anchor = manager.index_list[0]
    second_anchor = manager.index_list[1] if len(manager.index_list) > 1 else 256

    distances = [1, 4, 8, 16, 32, 64, 128, min(255, second_anchor - first_anchor - 1)]
    rows = []
    for d in distances:
        tok = first_anchor + d
        if tok >= kv.shape[0]:
            break
        err = recon.measure_error(kv, tok, tok)
        rows.append({
            "distance_from_anchor": d,
            "mean_relative_error": round(err["mean_relative"], 6),
            "max_l2": round(err["max_l2"], 6),
        })
    return rows


def test_reconstruction_throughput(kv, manager, recon):
    """Measure tokens-per-second across different workloads."""
    results = {}
    random.seed(0)

    # Sequential full-sequence reconstruction (simulate prefill decode)
    t0 = time.perf_counter()
    batch = 512
    for start in range(0, min(SEQ_LEN, 8192), batch):
        end = min(start + batch - 1, SEQ_LEN - 1)
        recon.reconstruct_range(start, end)
    elapsed = time.perf_counter() - t0
    toks_reconstructed = min(SEQ_LEN, 8192)
    results["sequential_tok_per_sec"] = round(toks_reconstructed / elapsed)

    # Random access reconstruction (simulate attention-style access)
    t0 = time.perf_counter()
    N_rand = 200
    window = 64
    total_toks = 0
    for _ in range(N_rand):
        start = random.randint(0, SEQ_LEN - window - 1)
        end   = start + window - 1
        recon.reconstruct_range(start, end)
        total_toks += window
    elapsed = time.perf_counter() - t0
    results["random_access_tok_per_sec"] = round(total_toks / elapsed)

    return results


def main():
    output_dir = Path("results/reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = KVGenerator(num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42)
    kv  = gen.generate(SEQ_LEN, mode="mixed")

    print(f"\n{'='*65}")
    print(f"  RECONSTRUCTION OVERHEAD EXPERIMENT")
    print(f"  seq_len={SEQ_LEN:,} | strategy={STRATEGY}")
    print(f"{'='*65}\n")

    manager = AnchorManager(strategy=STRATEGY)
    stats   = manager.compress(kv)
    recon   = KVReconstructor(manager)

    print(f"  Compression ratio : {stats.compression_ratio:.3f}x")
    print(f"  Anchor density    : {stats.anchor_density:.3%}")
    print(f"  Num anchors       : {stats.num_anchors}")

    all_results = {}

    # 1. Single vs grouped
    print("\n[1] Single vs Grouped Reconstruction Latency")
    svg = test_single_vs_grouped(kv, manager, recon)
    all_results["latency"] = svg
    for k, v in svg.items():
        print(f"    {k:<35}: {v}")

    # 2. Error vs anchor distance
    print("\n[2] Reconstruction Error vs Distance from Anchor (interval=256)")
    err_rows = test_error_vs_anchor_distance(kv, NUM_HEADS, HEAD_DIM)
    all_results["error_vs_distance"] = err_rows
    print(f"    {'Distance':>10} | {'Mean Rel Error':>15} | {'Max L2':>10}")
    print(f"    {'-'*10}-+-{'-'*15}-+-{'-'*10}")
    for r in err_rows:
        print(f"    {r['distance_from_anchor']:>10} | {r['mean_relative_error']:>15.6f} | {r['max_l2']:>10.6f}")

    # 3. Throughput
    print("\n[3] Reconstruction Throughput")
    tput = test_reconstruction_throughput(kv, manager, recon)
    all_results["throughput"] = tput
    for k, v in tput.items():
        print(f"    {k:<35}: {v:,} tok/sec")

    out_path = output_dir / "reconstruction_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results saved → {out_path}")


if __name__ == "__main__":
    main()
