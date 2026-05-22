"""
visualization/plot_runtime_results.py — Phase 4

Generates plots for Runtime Reality Validation.
- Throughput vs Context Length
- VRAM vs Context Length
- Crossover Analysis
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def generate_plots():
    data_path = "results/runtime_validation/benchmark_data.json"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Filter out OOM if any
    df = df[df['status'] != 'OOM'] if 'status' in df.columns else df
    
    # Ensure correct types
    df['context'] = df['context'].astype(int)
    df['tokens_per_sec'] = df['tokens_per_sec'].astype(float)
    df['vram_mb'] = df['vram_mb'].astype(float)
    df['avg_latency_ms'] = df['avg_latency_ms'].astype(float)

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", len(df['method'].unique()))

    # Plot 1: Throughput vs Context Length
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="context", y="tokens_per_sec", hue="method", marker="o", palette=palette)
    plt.title("KV Reconstruction Throughput (Tokens/sec) vs Context Length", fontsize=14)
    plt.xlabel("Context Length", fontsize=12)
    plt.ylabel("Throughput (tok/s)", fontsize=12)
    plt.xscale("log", base=2)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/runtime_validation/throughput_vs_context.png", dpi=300)
    plt.close()

    # Plot 2: VRAM vs Context Length
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="context", y="vram_mb", hue="method", marker="o", palette=palette)
    plt.title("KV Cache VRAM Usage (MB) vs Context Length", fontsize=14)
    plt.xlabel("Context Length", fontsize=12)
    plt.ylabel("VRAM (MB)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/runtime_validation/vram_vs_context.png", dpi=300)
    plt.close()

    # Plot 3: Latency Comparison (Bar chart for a specific context)
    max_ctx = df['context'].max()
    df_max = df[df['context'] == max_ctx]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_max, x="method", y="avg_latency_ms", palette=palette)
    plt.title(f"Reconstruction Latency (ms) at {max_ctx} tokens", fontsize=14)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Avg Latency (ms)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/runtime_validation/latency_comparison.png", dpi=300)
    plt.close()

    print("Plots generated in results/runtime_validation/")

if __name__ == "__main__":
    generate_plots()
