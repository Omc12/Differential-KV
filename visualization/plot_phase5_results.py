"""
visualization/plot_phase5_results.py

Generates plots for Phase 5 systems validation.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(results_file: str, output_dir: str):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return
        
    with open(results_file, "r") as f:
        data = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    modes = list(data.keys())
    contexts = [r["context_length"] for r in data[modes[0]]]
    
    # 1. Throughput vs Context
    plt.figure(figsize=(10, 6))
    for mode in modes:
        thru = [r["tokens_per_sec"] for r in data[mode]]
        plt.plot(contexts, thru, marker='o', label=mode)
    plt.title("Throughput vs Context Length")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Tokens/sec")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "throughput_vs_context.png"))
    
    # 2. VRAM vs Context
    plt.figure(figsize=(10, 6))
    for mode in modes:
        vram = [r["vram_mb"] for r in data[mode]]
        plt.plot(contexts, vram, marker='s', label=mode)
    plt.title("VRAM Usage vs Context Length")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("VRAM (MB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "vram_vs_context.png"))
    
    # 3. Latency Breakdown (Stacked bar for 32k context)
    plt.figure(figsize=(10, 6))
    idx = -1 # Use largest context
    categories = modes
    prefill = [data[m][idx]["prefill_latency_sec"] for m in modes]
    recon = [data[m][idx]["recon_latency_ms"] / 1000 for m in modes]
    
    x = np.arange(len(modes))
    plt.bar(x, prefill, label='Prefill (sec)')
    plt.bar(x, recon, bottom=prefill, label='Reconstruction (sec)')
    plt.xticks(x, modes)
    plt.title(f"Latency Breakdown at {contexts[idx]} tokens")
    plt.ylabel("Seconds")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "latency_breakdown.png"))
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_results("results/phase5/benchmark_results.json", "results/phase5/plots/")
