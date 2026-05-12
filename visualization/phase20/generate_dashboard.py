import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def generate_plots(results_dir, output_dir):
    print(f"\n>>> Generating Visualization Dashboard in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. GSM8K Accuracy vs Compression Ratio
    gsm8k_path = os.path.join(results_dir, "gsm8k_full.json")
    if os.path.exists(gsm8k_path):
        with open(gsm8k_path, "r") as f:
            data = json.load(f)
        
        plot_data = []
        for model, modes in data.items():
            for mode, m in modes.items():
                plot_data.append({
                    "Model": model,
                    "Mode": mode,
                    "Accuracy": m["accuracy"],
                    "Ratio": m["compression_ratio"]
                })
        
        df = pd.DataFrame(plot_data)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="Ratio", y="Accuracy", hue="Model", style="Mode", s=100)
        plt.title("GSM8K Accuracy vs. Compression Ratio")
        plt.savefig(os.path.join(output_dir, "gsm8k_pareto.png"))
        plt.close()

    # 2. Throughput scaling
    tp_path = os.path.join(results_dir, "throughput_metrics.json")
    if os.path.exists(tp_path):
        with open(tp_path, "r") as f:
            tp_data = json.load(f)
        
        plt.figure(figsize=(10, 6))
        for bs, ctxs in tp_data.items():
            x = []
            y = []
            for ctx, m in ctxs.items():
                x.append(int(ctx.replace("k", "")))
                y.append(m["throughput_tok_per_sec"])
            plt.plot(x, y, marker="o", label=f"BS={bs}")
        
        plt.xlabel("Context Length (k tokens)")
        plt.ylabel("Throughput (tokens/sec)")
        plt.title("Inference Throughput Scaling (LCG Mode)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "throughput_scaling.png"))
        plt.close()

    # 3. Baseline Comparison Bar Chart
    comp_path = os.path.join(results_dir, "baseline_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            comp_data = json.load(f)
        
        names = list(comp_data.keys())
        ratios = [comp_data[n]["compression_ratio"] for n in names]
        errors = [comp_data[n].get("mean_error", 0) for n in names]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        sns.barplot(x=names, y=ratios, alpha=0.6, color="b", ax=ax1, label="Compression Ratio")
        sns.lineplot(x=names, y=errors, marker="s", color="r", ax=ax2, label="Mean Error")
        
        ax1.set_ylabel("Compression Ratio (x)")
        ax2.set_ylabel("Mean Relative Error")
        plt.title("Differential KV vs. Baselines")
        plt.savefig(os.path.join(output_dir, "baseline_comparison.png"))
        plt.close()

    print(f"Plots generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="phase20/results")
    parser.add_argument("--output", type=str, default="phase20/plots")
    args = parser.parse_args()
    
    generate_plots(args.results, args.output)
