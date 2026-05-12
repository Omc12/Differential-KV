"""
visualization/plot_phase5b_results.py

Generates plots for Phase 5B systems validation.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all(results_file: str, output_dir: str):
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
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "throughput_vs_context.png"))
    
    # 2. KV-only VRAM
    plt.figure(figsize=(10, 6))
    for mode in modes:
        kv_vram = [r["isolated_kv_mb"] for r in data[mode]]
        plt.plot(contexts, kv_vram, marker='s', label=mode)
    plt.title("Isolated KV Memory Residency")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("KV VRAM (MB)")
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "kv_only_vram.png"))
    
    # 3. Latency Breakdown
    plt.figure(figsize=(10, 6))
    idx = -1 # Largest context
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
    
    # 4. Reconstruction Cost (ms)
    plt.figure(figsize=(10, 6))
    for mode in modes:
        costs = [r["recon_latency_ms"] for r in data[mode]]
        plt.plot(contexts, costs, marker='x', label=mode)
    plt.title("Reconstruction Overhead (ms)")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Latency (ms)")
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "reconstruction_cost.png"))

    # 5. Bandwidth Simulation (Theoretical)
    plt.figure(figsize=(10, 6))
    # Using theoretical values for a H=32, D=128 model (Llama)
    H, D, R, B = 32, 128, 16, 64
    feat_dim = 2 * H * D
    fp16_b = [ctx * feat_dim * 2 / (1024**2) for ctx in contexts]
    diffkv_b = [( (ctx // B) * feat_dim * 2 + ctx * R * 2 + (ctx // B) * R * feat_dim * 4 ) / (1024**2) for ctx in contexts]
    
    plt.plot(contexts, fp16_b, 'r--', label="FP16 (Theoretical)")
    plt.plot(contexts, diffkv_b, 'g-', label="DiffKV (Theoretical)")
    plt.title("KV Fetch Traffic (Theoretical)")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Traffic (MB)")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "bandwidth_vs_context.png"))
    
    # 6. Scaling Efficiency (Reduction Ratio)
    plt.figure(figsize=(10, 6))
    ratios = [fp / dk for fp, dk in zip(fp16_b, diffkv_b)]
    plt.plot(contexts, ratios, 'b-o', label="Compression Ratio")
    plt.axhline(y=1.0, color='k', linestyle='--')
    plt.title("Scaling Efficiency (Reduction x)")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Ratio (FP16 / DiffKV)")
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "scaling_efficiency.png"))

    print(f"Phase 5B Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_all("results/phase5b/benchmark_results.json", "results/phase5b/plots/")
