"""
visualization/plot_phase5c_results.py

Generates plots for Phase 5C real residency validation.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_residency_data(results_file: str, output_dir: str):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return
        
    with open(results_file, "r") as f:
        data = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    modes = list(data.keys())
    # Find all contexts present in any mode
    all_contexts = sorted(list(set([r["context_length"] for m in modes for r in data[m]])))
    
    # 1. Real KV Residency
    plt.figure(figsize=(10, 6))
    for mode in modes:
        ctxs = [r["context_length"] for r in data[mode]]
        res = [r["active_mb"] for r in data[mode]]
        plt.plot(ctxs, res, marker='o', label=mode)
    plt.title("Real GPU KV Residency (Active Tensors)")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("VRAM (MB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "real_kv_residency.png"))
    
    # 2. Compression Ratio (Real)
    plt.figure(figsize=(10, 6))
    if "fp16" in data:
        # Create a map for fp16 baseline
        fp16_map = {r["context_length"]: r["active_mb"] for r in data["fp16"]}
        for mode in modes:
            if mode == "fp16": continue
            ctxs = [r["context_length"] for r in data[mode] if r["context_length"] in fp16_map]
            ratios = [fp16_map[r["context_length"]] / r["active_mb"] for r in data[mode] if r["context_length"] in fp16_map]
            plt.plot(ctxs, ratios, marker='s', label=mode)
        plt.title("Real Compression Ratio (FP16 Active / Mode Active)")
        plt.xlabel("Context Length (tokens)")
        plt.ylabel("Ratio (x)")
        plt.axhline(y=1.0, color='k', linestyle='--')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "compression_ratio_real.png"))
    
    # 3. Allocator Breakdown (Active vs Reserved) for LowRank R16
    lr_mode = "lowrank_r16"
    if lr_mode in data:
        plt.figure(figsize=(10, 6))
        ctxs = [r["context_length"] for r in data[lr_mode]]
        active = [r["active_mb"] for r in data[lr_mode]]
        reserved = [r["reserved_mb"] for r in data[lr_mode]]
        
        plt.bar(ctxs, reserved, width=400, label='Reserved (Allocator Cache)', alpha=0.3)
        plt.bar(ctxs, active, width=400, label='Active (Tensors)')
        plt.title(f"Allocator Breakdown: {lr_mode}")
        plt.xlabel("Context Length (tokens)")
        plt.ylabel("VRAM (MB)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "allocator_breakdown.png"))
        
    # 4. Fragmentation Analysis
    plt.figure(figsize=(10, 6))
    for mode in modes:
        ctxs = [r["context_length"] for r in data[mode]]
        frag = [r["fragmentation_mb"] for r in data[mode]]
        plt.plot(ctxs, frag, marker='x', label=mode)
    plt.title("Allocator Fragmentation Overhead")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Fragmentation (MB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "fragmentation_analysis.png"))

    print(f"Phase 5C Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_residency_data("results/phase5c/real_kv_residency.json", "results/phase5c/plots/")
