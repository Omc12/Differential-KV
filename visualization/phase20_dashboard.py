import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def generate_plots(results_dir, output_dir):
    print(f"\n>>> Generating Visualization Dashboard in {output_dir}")
    os.makedirs(output_dir, exist_ok=True); sns.set_theme(style="whitegrid")
    gsm8k_path = os.path.join(results_dir, "gsm8k_full.json")
    if os.path.exists(gsm8k_path):
        with open(gsm8k_path, "r") as f: data = json.load(f)
        plot_data = []
        for model, modes in data.items():
            for mode, m in modes.items(): plot_data.append({"Model": model, "Mode": mode, "Accuracy": m["accuracy"], "Ratio": m["compression_ratio"]})
        df = pd.DataFrame(plot_data); plt.figure(figsize=(10, 6)); sns.scatterplot(data=df, x="Ratio", y="Accuracy", hue="Model", s=100)
        plt.title("GSM8K Accuracy vs. Compression Ratio"); plt.savefig(os.path.join(output_dir, "gsm8k_pareto.png")); plt.close()
    print(f"Plots generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/phase20")
    parser.add_argument("--output", type=str, default="visualization/phase20")
    args = parser.parse_args()
    generate_plots(args.results, args.output)
