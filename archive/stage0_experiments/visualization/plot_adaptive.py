"""
visualization/plot_adaptive.py — Phase 2 Task 1

Plots adaptive policy sweep results:
  1. Compression ratio vs anchor density scatter (Pareto view)
  2. Error vs compression ratio per mode
  3. Anchor density comparison: all policies, all modes (bar chart)
  4. Policy trigger reason breakdown (stacked bar)
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
    "Periodic":     "#8B949E",
    "AbsNorm":      "#3498DB",
    "RelChange":    "#27AE60",
    "Rolling":      "#F39C12",
    "EMA":          "#8E44AD",
    "LayerNorm":    "#E74C3C",
}
MODE_COLORS = {
    "gaussian":    "#E74C3C",
    "smooth":      "#27AE60",
    "mixed":       "#F39C12",
    "real_approx": "#3498DB",
}

def _policy_family(label):
    for family in ["Periodic", "AbsNorm", "RelChange", "Rolling", "EMA", "LayerNorm"]:
        if label.startswith(family):
            return family
    return "Other"

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0D1117", "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D", "axes.labelcolor": "#C9D1D9",
        "axes.titlecolor": "#F0F6FC", "axes.grid": True,
        "grid.color": "#21262D", "grid.linewidth": 0.8,
        "xtick.color": "#8B949E", "ytick.color": "#8B949E",
        "text.color": "#C9D1D9", "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D", "legend.labelcolor": "#C9D1D9",
        "font.size": 10, "axes.titlesize": 12,
    })


def plot_adaptive(data, output_dir: Path):
    setup_style()

    modes = list(dict.fromkeys(r["mode"] for r in data))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Phase 2 — Adaptive Policy Analysis", fontsize=15,
                 color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: Error vs Ratio scatter (all modes) ─────────────────────────
    ax = axes[0, 0]
    ax.set_title("Error vs Compression Ratio — All Policies & Modes")
    plotted_families = set()

    for r in data:
        family = _policy_family(r["policy"])
        color  = PALETTE.get(family, "#AAAAAA")
        marker = {"gaussian": "o", "smooth": "s",
                  "mixed": "^", "real_approx": "D"}.get(r["mode"], "o")
        label  = family if family not in plotted_families else None
        plotted_families.add(family)
        ax.scatter(r["compression_ratio"], r["mean_recon_error"],
                   c=color, marker=marker, s=60, alpha=0.75, label=label,
                   edgecolors="#0D1117", linewidth=0.4)

    ax.axhline(0.01,  color="#F39C12", ls="--", lw=1, alpha=0.7, label="1% error limit")
    ax.axvline(2.0,   color="#27AE60", ls=":",  lw=1, alpha=0.5, label="FP8 equiv (2x)")
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Mean Relative Error")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)

    # ── Plot 2: Anchor density per policy family (violin per mode) ────────
    ax = axes[0, 1]
    ax.set_title("Anchor Density by Policy Family")

    families = ["Periodic", "AbsNorm", "RelChange", "Rolling", "EMA", "LayerNorm"]
    x_base = np.arange(len(families))
    bar_w  = 0.18

    for m_idx, mode in enumerate(modes):
        densities = []
        for fam in families:
            fam_data = [r["anchor_density"] for r in data
                        if r["mode"] == mode and _policy_family(r["policy"]) == fam]
            densities.append(np.mean(fam_data) if fam_data else 0)

        offset = (m_idx - len(modes)/2 + 0.5) * bar_w
        ax.bar(x_base + offset, densities, width=bar_w,
               label=mode, color=MODE_COLORS.get(mode, "#AAA"),
               alpha=0.8, edgecolor="#0D1117", linewidth=0.4)

    ax.axhline(0.02, color="#8B949E", ls=":", lw=1, alpha=0.6, label="2% target")
    ax.set_xticks(x_base)
    ax.set_xticklabels(families, rotation=20, fontsize=8)
    ax.set_ylabel("Mean Anchor Density")
    ax.legend(fontsize=8)

    # ── Plot 3: Compression ratio per mode, best policy of each family ────
    ax = axes[1, 0]
    ax.set_title("Best Compression Ratio per Family (by mode)")

    for m_idx, mode in enumerate(modes):
        best_ratios = []
        for fam in families:
            fam_data = [r for r in data
                        if r["mode"] == mode and _policy_family(r["policy"]) == fam
                        and r["mean_recon_error"] < 0.02]  # error budget
            if fam_data:
                best_ratios.append(max(r["compression_ratio"] for r in fam_data))
            else:
                best_ratios.append(0)
        offset = (m_idx - len(modes)/2 + 0.5) * bar_w
        ax.bar(x_base + offset, best_ratios, width=bar_w,
               label=mode, color=MODE_COLORS.get(mode, "#AAA"),
               alpha=0.8, edgecolor="#0D1117", linewidth=0.4)

    ax.axhline(2.0, color="#27AE60", ls="--", lw=1, alpha=0.6, label="FP8 line")
    ax.set_xticks(x_base)
    ax.set_xticklabels(families, rotation=20, fontsize=8)
    ax.set_ylabel("Best Compression Ratio (err < 2%)")
    ax.legend(fontsize=8)

    # ── Plot 4: Error budget tradeoff — smooth mode focus ────────────────
    ax = axes[1, 1]
    ax.set_title("Error vs Density Tradeoff (smooth mode)")
    smooth = [r for r in data if r["mode"] == "smooth"]

    for fam in families:
        fam_rows = [r for r in smooth if _policy_family(r["policy"]) == fam]
        if not fam_rows:
            continue
        xs = [r["anchor_density"]   for r in fam_rows]
        ys = [r["mean_recon_error"] for r in fam_rows]
        ax.scatter(xs, ys, c=PALETTE.get(fam, "#AAA"), s=70, label=fam,
                   edgecolors="#0D1117", linewidth=0.4, alpha=0.85)
        # Connect with line (sorted by density)
        pairs = sorted(zip(xs, ys))
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                color=PALETTE.get(fam, "#AAA"), lw=1, alpha=0.4)

    ax.axhline(0.01, color="#F39C12", ls="--", lw=1, alpha=0.7, label="1% limit")
    ax.set_xlabel("Anchor Density (lower = better)")
    ax.set_ylabel("Reconstruction Error")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "adaptive_policy_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Plot saved -> {out_path}")
    return out_path


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_adaptive(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/adaptive_policies/policy_sweep.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
