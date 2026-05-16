"""
visualization/plot_anchor_density.py

Plots anchor density sweep results.

Shows: for each KV mode, how compression ratio and error change
as anchor interval increases from 8 to 512.

The "sweet spot" curve is the most important output here.
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


def plot_anchor_density(data, output_dir: Path):
    setup_style()

    # Group by mode
    by_mode = defaultdict(list)
    for row in data:
        by_mode[row["mode"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Anchor Density Sweep — Compression vs Error Tradeoff",
                 fontsize=14, color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: Compression ratio vs interval ─────────────────────────────────
    ax = axes[0]
    ax.set_title("Compression Ratio vs Anchor Interval")
    ax.axhline(y=2.0, color="#8B949E", linestyle="--", linewidth=1,
               alpha=0.5, label="FP8 (2x)")

    for mode, rows in by_mode.items():
        rows_sorted = sorted(rows, key=lambda r: r["interval"])
        intervals = [r["interval"]          for r in rows_sorted]
        ratios    = [r["compression_ratio"] for r in rows_sorted]
        ax.plot(intervals, ratios, label=mode, color=MODE_COLORS.get(mode, "#AAAAAA"),
                linewidth=2, marker="o", markersize=5)

    ax.set_xlabel("Anchor Interval (tokens)")
    ax.set_ylabel("Compression Ratio")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)

    # ── Plot 2: Error vs interval ─────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Reconstruction Error vs Anchor Interval")
    ax.axhline(y=0.01, color="#F39C12", linestyle="--", linewidth=1,
               alpha=0.7, label="1% error threshold")

    for mode, rows in by_mode.items():
        rows_sorted = sorted(rows, key=lambda r: r["interval"])
        intervals = [r["interval"]    for r in rows_sorted]
        errors    = [r["mean_error"]  for r in rows_sorted]
        ax.plot(intervals, errors, label=mode, color=MODE_COLORS.get(mode, "#AAAAAA"),
                linewidth=2, marker="o", markersize=5)

    ax.set_xlabel("Anchor Interval (tokens)")
    ax.set_ylabel("Mean Relative Reconstruction Error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "anchor_density_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Anchor density plot saved → {out_path}")
    return out_path


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_anchor_density(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/anchor_density/periodic_sweep.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
