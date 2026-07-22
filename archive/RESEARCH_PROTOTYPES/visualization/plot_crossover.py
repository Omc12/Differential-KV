"""
visualization/plot_crossover.py

Plots the defining crossover experiment result.

Generates:
  1. Memory bytes vs context length (log scale) — all strategies
  2. Compression ratio vs context length
  3. Bandwidth savings vs INT8 — where does DKV win?
  4. Reconstruction error vs context length

Usage:
    python visualization/plot_crossover.py --input results/crossover/crossover_mixed.json
    python visualization/plot_crossover.py --input results/crossover/crossover_mixed.json --mode smooth
"""

import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "FP16":          "#E74C3C",
    "FP8":           "#F39C12",
    "INT8":          "#27AE60",
    "DKV-P32":    "#3498DB",
    "DKV-P64":    "#2980B9",
    "DKV-P128":   "#1A5276",
    "DKV-Adapt":  "#8E44AD",
}
LINESTYLES = {
    "FP16":          "--",
    "FP8":           "--",
    "INT8":          "--",
    "DKV-P32":    "-",
    "DKV-P64":    "-",
    "DKV-P128":   "-",
    "DKV-Adapt":  "-",
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0D1117",
        "axes.facecolor":    "#161B22",
        "axes.edgecolor":    "#30363D",
        "axes.labelcolor":   "#C9D1D9",
        "axes.titlecolor":   "#F0F6FC",
        "axes.grid":         True,
        "grid.color":        "#21262D",
        "grid.linewidth":    0.8,
        "xtick.color":       "#8B949E",
        "ytick.color":       "#8B949E",
        "text.color":        "#C9D1D9",
        "legend.facecolor":  "#161B22",
        "legend.edgecolor":  "#30363D",
        "legend.labelcolor": "#C9D1D9",
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })


def human_bytes(n):
    if n >= 1024**3: return f"{n/1024**3:.1f} GB"
    if n >= 1024**2: return f"{n/1024**2:.1f} MB"
    if n >= 1024:    return f"{n/1024:.1f} KB"
    return f"{n} B"


def load_data(path: str):
    with open(path) as f:
        return json.load(f)


def plot_crossover(data, output_dir: Path):
    setup_style()

    seq_lens = [r["seq_len"] for r in data]
    x = np.array(seq_lens)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Differential KV Cache — Crossover Analysis", fontsize=16,
                 color="#F0F6FC", fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: Memory bytes vs seq_len ──────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("KV Memory Usage vs Context Length")

    configs = {
        "FP16":         [r["FP16_bytes"] for r in data],
        "FP8":          [r["FP8_bytes"] for r in data],
        "INT8":         [r["INT8_bytes"] for r in data],
        "DKV-P64":   [r.get("DKV-P64_bytes", 0) for r in data],
        "DKV-P128":  [r.get("DKV-P128_bytes", 0) for r in data],
        "DKV-Adapt": [r.get("DKV-Adapt_bytes", 0) for r in data],
    }
    for label, ys in configs.items():
        ax.plot(x, np.array(ys) / 1024**2, label=label,
                color=PALETTE.get(label, "#AAAAAA"),
                linestyle=LINESTYLES.get(label, "-"),
                linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Memory (MB) — single layer")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8)

    # ── Plot 2: Compression ratio ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Compression Ratio vs Context Length")
    ax.axhline(y=2.0, color="#27AE60", linestyle="--", linewidth=1, alpha=0.6, label="FP8 target (2x)")
    ax.axhline(y=1.0, color="#8B949E", linestyle=":", linewidth=1, alpha=0.5, label="No compression")

    for label in ["DKV-P32", "DKV-P64", "DKV-P128", "DKV-Adapt"]:
        key = f"{label}_ratio"
        ys  = [r.get(key, 1.0) for r in data]
        ax.plot(x, ys, label=label, color=PALETTE.get(label, "#AAAAAA"),
                linestyle="-", linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Compression Ratio (higher = better)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8)

    # ── Plot 3: Bandwidth saving vs INT8 ─────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Bandwidth Saving vs INT8-Naive")
    ax.axhline(y=0.0, color="#E74C3C", linestyle="--", linewidth=1.5,
               alpha=0.8, label="Break-even with INT8")

    for label in ["DKV-P32", "DKV-P64", "DKV-P128", "DKV-Adapt"]:
        key = f"{label}_bw_saving_vs_int8"
        ys  = [r.get(key, 0.0) for r in data]
        ax.plot(x, np.array(ys) * 100, label=label,
                color=PALETTE.get(label, "#AAAAAA"),
                linestyle="-", linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Bandwidth Saving vs INT8 (%)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8)
    ax.fill_between(x, 0, 100, alpha=0.05, color="#27AE60")   # DKV winning region

    # ── Plot 4: Reconstruction error ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Reconstruction Error vs Context Length")
    ax.axhline(y=0.01, color="#F39C12", linestyle="--", linewidth=1,
               alpha=0.7, label="1% error threshold")

    for label in ["DKV-P32", "DKV-P64", "DKV-P128", "DKV-Adapt"]:
        key = f"{label}_error"
        ys  = [r.get(key, 0.0) for r in data]
        ax.plot(x, ys, label=label, color=PALETTE.get(label, "#AAAAAA"),
                linestyle="-", linewidth=2, marker="o", markersize=4)

    # Also plot INT8/FP8 error
    int8_errs = [r.get("INT8_error", 0.0) for r in data]
    fp8_errs  = [r.get("FP8_error",  0.0) for r in data]
    ax.plot(x, int8_errs, label="INT8-naive", color=PALETTE["INT8"],
            linestyle="--", linewidth=1.5, marker="s", markersize=3)
    ax.plot(x, fp8_errs,  label="FP8-sim",   color=PALETTE["FP8"],
            linestyle="--", linewidth=1.5, marker="s", markersize=3)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Mean Relative L2 Error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "crossover_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Crossover plot saved → {out_path}")
    return out_path


def main(args):
    data = load_data(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_crossover(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/crossover/crossover_mixed.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
