"""
visualization/plot_layer_analysis.py — Phase 2 Task 2

Plots layer-wise compressibility analysis:
  1. Compression ratio heatmap (layer x strategy)
  2. Layer ranking scatter (smoothness vs ratio)
  3. Delta norm heatmap (layer x token position)
  4. Strategy recommendation distribution
"""

import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0D1117", "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D", "axes.labelcolor": "#C9D1D9",
        "axes.titlecolor": "#F0F6FC", "axes.grid": True,
        "grid.color": "#21262D", "grid.linewidth": 0.6,
        "xtick.color": "#8B949E", "ytick.color": "#8B949E",
        "text.color": "#C9D1D9", "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D", "legend.labelcolor": "#C9D1D9",
        "font.size": 9, "axes.titlesize": 11,
    })


def plot_layer_analysis(data: dict, output_dir: Path):
    setup_style()

    strategy_names = list(data.keys())
    if not strategy_names:
        print("[WARN] No data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Layer-wise Compressibility Analysis", fontsize=15,
                 color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: Compression ratio per layer (all strategies) ─────────────
    ax = axes[0, 0]
    ax.set_title("Compression Ratio per Layer")
    colors = ["#3498DB", "#27AE60", "#F39C12", "#8E44AD", "#E74C3C"]

    for c_idx, strat in enumerate(strategy_names):
        profiles = data[strat].get("profiles", [])
        if not profiles:
            continue
        profiles_sorted = sorted(profiles, key=lambda p: p["layer_idx"])
        layers = [p["layer_idx"] for p in profiles_sorted]
        ratios = [p["compression_ratio"] for p in profiles_sorted]
        ax.plot(layers, ratios, label=strat, color=colors[c_idx % len(colors)],
                linewidth=2, marker="o", markersize=3)

    ax.axhline(2.0, color="#8B949E", ls="--", lw=1, alpha=0.5, label="FP8 baseline")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Compression Ratio")
    ax.legend(fontsize=8)

    # ── Plot 2: Anchor density per layer ──────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Anchor Density per Layer")

    for c_idx, strat in enumerate(strategy_names):
        profiles = data[strat].get("profiles", [])
        if not profiles:
            continue
        profiles_sorted = sorted(profiles, key=lambda p: p["layer_idx"])
        layers   = [p["layer_idx"] for p in profiles_sorted]
        densities = [p["anchor_density"] for p in profiles_sorted]
        ax.plot(layers, densities, label=strat, color=colors[c_idx % len(colors)],
                linewidth=2, marker="s", markersize=3)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Anchor Density")
    ax.legend(fontsize=8)

    # ── Plot 3: Smoothness vs compressibility scatter ─────────────────────
    ax = axes[1, 0]
    ax.set_title("Smoothness vs Compression Ratio (Periodic-64)")
    strat = "Periodic-64" if "Periodic-64" in data else strategy_names[0]
    profiles = data[strat].get("profiles", [])

    if profiles:
        smoothness = [p["smoothness_score"]   for p in profiles]
        ratios     = [p["compression_ratio"]  for p in profiles]
        errors     = [p["mean_recon_error"]   for p in profiles]
        layers     = [p["layer_idx"]          for p in profiles]

        sc = ax.scatter(smoothness, ratios, c=layers, cmap="plasma",
                        s=80, alpha=0.85, edgecolors="#0D1117", linewidth=0.4)
        plt.colorbar(sc, ax=ax, label="Layer Index")
        ax.set_xlabel("Smoothness Score (lower = smoother)")
        ax.set_ylabel("Compression Ratio")
        # Annotate a few layers
        for p in profiles[::4]:
            ax.annotate(f"L{p['layer_idx']}", (p["smoothness_score"],
                        p["compression_ratio"]),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=6, color="#8B949E")

    # ── Plot 4: Reconstruction error per layer ────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Reconstruction Error per Layer")

    for c_idx, strat in enumerate(strategy_names):
        profiles = data[strat].get("profiles", [])
        if not profiles:
            continue
        profiles_sorted = sorted(profiles, key=lambda p: p["layer_idx"])
        layers = [p["layer_idx"]        for p in profiles_sorted]
        errors = [p["mean_recon_error"] for p in profiles_sorted]
        ax.plot(layers, errors, label=strat, color=colors[c_idx % len(colors)],
                linewidth=2, marker="^", markersize=3)

    ax.axhline(0.01, color="#F39C12", ls="--", lw=1, alpha=0.7, label="1% limit")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Relative Error")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "layer_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Layer analysis plot saved -> {out_path}")
    return out_path


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_layer_analysis(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/layer_analysis/layer_profiles.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
