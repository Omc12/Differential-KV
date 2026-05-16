"""
visualization/plot_delta_analysis.py — Phase 2 Task 4

Plots delta distribution statistics:
  1. RMS distribution histograms per KV mode
  2. SVD energy retention curves (low-rank approximability)
  3. Sparsity comparison bar chart
  4. Kurtosis & entropy per mode
"""

import sys
import json
import argparse
from pathlib import Path

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
        "figure.facecolor": "#0D1117", "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D", "axes.labelcolor": "#C9D1D9",
        "axes.titlecolor": "#F0F6FC", "axes.grid": True,
        "grid.color": "#21262D", "grid.linewidth": 0.7,
        "xtick.color": "#8B949E", "ytick.color": "#8B949E",
        "text.color": "#C9D1D9", "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D", "legend.labelcolor": "#C9D1D9",
        "font.size": 9, "axes.titlesize": 11,
    })


def plot_delta_analysis(data: dict, output_dir: Path):
    setup_style()

    modes = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Delta Distribution Analysis — All KV Modes", fontsize=15,
                 color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: SVD energy retention curves ───────────────────────────────
    ax = axes[0, 0]
    ax.set_title("SVD Energy Retention (Low-Rank Approximability)")

    for mode in modes:
        d = data[mode]
        lr = d.get("lowrank", {})
        ranks, energies = [], []
        for rk in [1, 2, 4, 8, 16, 32]:
            key = f"rank_{rk}"
            if key in lr and isinstance(lr[key], dict):
                ranks.append(rk)
                energies.append(lr[key]["energy_retained"])
        if ranks:
            ax.plot(ranks, energies, label=mode, color=MODE_COLORS.get(mode, "#AAA"),
                    linewidth=2, marker="o", markersize=5)

    ax.axhline(0.9, color="#8B949E", ls="--", lw=1, alpha=0.6, label="90% threshold")
    ax.axhline(0.7, color="#555",    ls=":",  lw=1, alpha=0.5, label="70% threshold")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative Energy Retained")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)

    # ── Plot 2: Low-rank reconstruction error curves ──────────────────────
    ax = axes[0, 1]
    ax.set_title("Low-Rank Reconstruction Error vs Rank")

    for mode in modes:
        d  = data[mode]
        lr = d.get("lowrank", {})
        ranks, errors = [], []
        for rk in [1, 2, 4, 8, 16, 32]:
            key = f"rank_{rk}"
            if key in lr and isinstance(lr[key], dict):
                ranks.append(rk)
                errors.append(lr[key]["recon_error"])
        if ranks:
            ax.plot(ranks, errors, label=mode, color=MODE_COLORS.get(mode, "#AAA"),
                    linewidth=2, marker="s", markersize=5)

    ax.axhline(0.05, color="#F39C12", ls="--", lw=1, alpha=0.7, label="5% error limit")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative Reconstruction Error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # ── Plot 3: Sparsity & entropy comparison ─────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Sparsity & Entropy by Mode")

    x = np.arange(len(modes))
    w = 0.25
    sp01  = [data[m]["distribution"].get("sparsity_01",  0) for m in modes]
    sp001 = [data[m]["distribution"].get("sparsity_001", 0) for m in modes]
    entr  = [data[m]["distribution"].get("mean_entropy_bits", 0) / 10 for m in modes]  # scale

    bars1 = ax.bar(x - w, sp01,  width=w, label="Sparsity (<0.1)",  alpha=0.85,
                   color="#3498DB", edgecolor="#0D1117", linewidth=0.4)
    bars2 = ax.bar(x,     sp001, width=w, label="Sparsity (<0.01)", alpha=0.85,
                   color="#27AE60", edgecolor="#0D1117", linewidth=0.4)
    bars3 = ax.bar(x + w, entr,  width=w, label="Entropy/10 (bits)", alpha=0.85,
                   color="#F39C12", edgecolor="#0D1117", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15)
    ax.set_ylabel("Fraction / Scaled Entropy")
    ax.legend(fontsize=8)

    # ── Plot 4: Kurtosis & autocorrelation ────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Kurtosis & Temporal Autocorrelation")

    kurt_vals  = [data[m]["distribution"].get("kurtosis", 0) for m in modes]
    autocorr   = [data[m]["temporal"].get("autocorrelation_lag1", 0) for m in modes]
    smoothness = [data[m]["temporal"].get("mean_consecutive_rms", 0) for m in modes]

    x2 = np.arange(len(modes))
    ax2 = ax.twinx()
    ax2.set_ylabel("Temporal Autocorrelation", color="#8E44AD")
    ax2.tick_params(axis="y", labelcolor="#8E44AD")

    ax.bar(x2, kurt_vals, width=0.4, label="Kurtosis (excess)", alpha=0.8,
           color="#E74C3C", edgecolor="#0D1117", linewidth=0.4)
    ax2.plot(x2, autocorr, "o--", color="#8E44AD", linewidth=2,
             markersize=8, label="Autocorr lag-1")

    ax.axhline(0, color="#8B949E", ls=":", lw=1, alpha=0.5)
    ax.set_xticks(x2)
    ax.set_xticklabels(modes, rotation=15)
    ax.set_ylabel("Excess Kurtosis")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = output_dir / "delta_distribution_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Delta analysis plot saved -> {out_path}")
    return out_path


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_delta_analysis(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/delta_analysis/delta_statistics.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
