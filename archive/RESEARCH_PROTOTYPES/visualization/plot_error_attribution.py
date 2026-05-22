"""
visualization/plot_error_attribution.py — Phase 2.5 Objective 4

Plots error attribution study results:
  1. Stacked bar: chain vs quantization error components
  2. Spacing sweep curves (error vs anchor interval)
  3. Adaptive vs periodic error delta
  4. Error dominance pie chart
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
        "grid.color": "#21262D", "grid.linewidth": 0.6,
        "xtick.color": "#8B949E", "ytick.color": "#8B949E",
        "text.color": "#C9D1D9", "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D",
        "font.size": 9, "axes.titlesize": 11,
    })


def plot_error_attribution(data: dict, output_dir: Path):
    setup_style()
    modes = list(data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Error Attribution Study", fontsize=14,
                 color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: Stacked bar — component breakdown ─────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Error Component Breakdown (interval=64)")
    x  = np.arange(len(modes))
    w  = 0.35
    chain_errs = [data[m]["attribution"]["chain_error_contribution"] for m in modes]
    quant_errs = [data[m]["attribution"]["quantization_contribution"] for m in modes]
    inter_errs = [max(0, data[m]["attribution"]["interaction_term"]) for m in modes]

    ax.bar(x, chain_errs, width=w, label="Chain Error",
           color="#3498DB", alpha=0.85, edgecolor="#0D1117")
    ax.bar(x, quant_errs, width=w, bottom=chain_errs,
           label="Quantization Error", color="#F39C12", alpha=0.85, edgecolor="#0D1117")
    ax.bar(x, inter_errs, width=w,
           bottom=[c+q for c,q in zip(chain_errs, quant_errs)],
           label="Interaction", color="#E74C3C", alpha=0.75, edgecolor="#0D1117")

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, fontsize=8)
    ax.set_ylabel("Reconstruction Error")
    ax.legend(fontsize=8)

    # ── Plot 2: Spacing sweep curves ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Reconstruction Error vs Anchor Interval")
    for mode in modes:
        sweep   = data[mode].get("spacing_sweep", {})
        ivs     = sorted([int(k) for k in sweep.keys()])
        errs    = [sweep[str(i)] for i in ivs]
        ax.plot(ivs, errs, label=mode, color=MODE_COLORS.get(mode, "#AAA"),
                linewidth=2, marker="o", markersize=5)

    ax.set_xlabel("Anchor Interval (tokens)")
    ax.set_ylabel("Reconstruction Error")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # ── Plot 3: Adaptive vs periodic policy error delta ───────────────────────
    ax = axes[1, 0]
    ax.set_title("EMA Adaptive Error Delta vs Periodic (+ = worse)")
    adap_deltas = [data[m]["adaptive_vs_periodic"]["adaptive_delta"] for m in modes]
    colors_bar  = ["#27AE60" if d <= 0 else "#E74C3C" for d in adap_deltas]
    ax.bar(modes, adap_deltas, color=colors_bar, alpha=0.85,
           edgecolor="#0D1117", linewidth=0.5)
    ax.axhline(0, color="#8B949E", ls="--", lw=1)
    ax.set_ylabel("Error Delta (EMA - Periodic)")
    ax.set_xticklabels(modes, rotation=15, fontsize=8)

    for i, (mode, d) in enumerate(zip(modes, adap_deltas)):
        adap = data[mode]["adaptive_vs_periodic"]
        density_txt = (f"P:{adap['periodic_density']:.3f} "
                       f"E:{adap['ema_density']:.3f}")
        ax.text(i, d + (0.0001 if d >= 0 else -0.0003), density_txt,
                ha="center", fontsize=6, color="#C9D1D9")

    # ── Plot 4: Full DiffKV vs FP16 chain comparison ─────────────────────────
    ax = axes[1, 1]
    ax.set_title("Full DiffKV vs Components (mixed mode)")
    mode = "mixed" if "mixed" in data else modes[0]
    d    = data[mode]
    labels  = ["FP16\nChain Only", "INT8\nNo Chain", "Full\nDiffKV"]
    vals    = [d["fp16_chain"], d["int8_no_chain"], d["full_diffkv"]]
    colors2 = ["#8E44AD", "#3498DB", "#E74C3C"]
    bars = ax.bar(labels, vals, color=colors2, alpha=0.85,
                  edgecolor="#0D1117", linewidth=0.5)
    ax.set_ylabel("Mean Relative Reconstruction Error")
    ax.set_title(f"Error Components ({mode} KV)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005,
                f"{v:.5f}", ha="center", fontsize=8, color="#C9D1D9")

    plt.tight_layout()
    out_path = output_dir / "error_attribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[OK] Error attribution plot -> {out_path}")


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_error_attribution(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/error_attribution/error_attribution.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
