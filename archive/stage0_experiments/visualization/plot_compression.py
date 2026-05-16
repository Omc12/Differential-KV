"""
visualization/plot_compression.py

Plots compression experiment results.

Generates:
  1. Compression ratio grouped by KV mode (bar chart)
  2. Compression vs anchor density scatter
  3. Delta norm distribution violin plots
  4. Error vs compression ratio Pareto frontier
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PALETTE = {
    "Periodic-32":   "#3498DB",
    "Periodic-64":   "#2980B9",
    "Periodic-128":  "#1A5276",
    "Adaptive":      "#8E44AD",
}

MODE_COLORS = {
    "gaussian":    "#E74C3C",
    "smooth":      "#27AE60",
    "mixed":       "#F39C12",
    "real_approx": "#3498DB",
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0D1117",
        "axes.facecolor":   "#161B22",
        "axes.edgecolor":   "#30363D",
        "axes.labelcolor":  "#C9D1D9",
        "axes.titlecolor":  "#F0F6FC",
        "axes.grid":        True,
        "grid.color":       "#21262D",
        "grid.linewidth":   0.8,
        "xtick.color":      "#8B949E",
        "ytick.color":      "#8B949E",
        "text.color":       "#C9D1D9",
        "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D",
        "legend.labelcolor":"#C9D1D9",
        "font.size":        10,
        "axes.titlesize":   12,
    })


def plot_compression(data, output_dir: Path):
    setup_style()

    modes      = list(dict.fromkeys(r["mode"] for r in data))
    strategies = list(dict.fromkeys(r["strategy"] for r in data))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Compression Ratio & Quality — All KV Modes × Strategies",
                 fontsize=14, color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: Grouped bar chart — ratio by mode ─────────────────────────────
    ax = axes[0]
    ax.set_title("Compression Ratio by Mode & Strategy")
    n_modes = len(modes)
    n_strat = len(strategies)
    bar_w   = 0.18
    x_base  = np.arange(n_modes)

    for s_idx, strategy in enumerate(strategies):
        ratios = []
        for mode in modes:
            row = next((r for r in data if r["mode"] == mode and r["strategy"] == strategy), None)
            ratios.append(row["compression_ratio"] if row else 0)

        offset = (s_idx - n_strat / 2 + 0.5) * bar_w
        bars = ax.bar(x_base + offset, ratios, width=bar_w,
                      label=strategy, color=PALETTE.get(strategy, "#AAAAAA"),
                      alpha=0.85, edgecolor="#0D1117", linewidth=0.5)

    ax.axhline(y=2.0, color="#27AE60", linestyle="--", linewidth=1.2,
               alpha=0.7, label="2x (FP8 equiv)")
    ax.set_xticks(x_base)
    ax.set_xticklabels(modes, rotation=15)
    ax.set_ylabel("Compression Ratio")
    ax.legend(fontsize=8)

    # ── Plot 2: Pareto frontier — error vs ratio ──────────────────────────────
    ax = axes[1]
    ax.set_title("Error vs Compression Ratio (Pareto)")
    ax.set_xlabel("Compression Ratio (higher = better)")
    ax.set_ylabel("Mean Relative Reconstruction Error (lower = better)")

    for mode in modes:
        rows = [r for r in data if r["mode"] == mode]
        ratios = [r["compression_ratio"]  for r in rows]
        errors = [r["mean_recon_error"]   for r in rows]
        strats = [r["strategy"]           for r in rows]

        ax.scatter(ratios, errors,
                   color=MODE_COLORS.get(mode, "#AAAAAA"),
                   s=80, alpha=0.8, label=mode, edgecolors="#0D1117", linewidth=0.5)

        # Label each point with strategy
        for ratio, error, strat in zip(ratios, errors, strats):
            short = strat.replace("Periodic-", "P").replace("Adaptive", "A")
            ax.annotate(short, (ratio, error),
                        textcoords="offset points", xytext=(4, 2),
                        fontsize=7, color="#8B949E")

    ax.set_yscale("log")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "compression_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Compression plot saved → {out_path}")
    return out_path


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_compression(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/compression/compression_results.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
