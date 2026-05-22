"""
visualization/plot_lowrank.py — Phase 2.5 Objective 3

Plots low-rank feasibility analysis:
  1. SVD energy retention curves per KV mode
  2. Subspace drift over time
  3. Per-head rank-1 energy heatmap across layers
  4. Compute cost comparison: INT8 vs LoRA at various ranks
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


def plot_lowrank(data: dict, output_dir: Path):
    setup_style()

    modes  = [k for k in data.keys() if k != "layer_analysis"]
    layers = data.get("layer_analysis", {})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Low-Rank Feasibility Analysis", fontsize=14,
                 color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: SVD energy retention ─────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("SVD Energy Retention by Rank")
    for mode in modes:
        profile = data[mode]["profile"]
        energy  = profile.get("energy_at_rank", {})
        ranks   = [int(k) for k in energy.keys()]
        vals    = [energy[str(r)] for r in ranks]
        ax.plot(ranks, vals, label=mode, color=MODE_COLORS.get(mode, "#AAA"),
                linewidth=2, marker="o", markersize=5)

    ax.axhline(0.90, color="#F39C12", ls="--", lw=1, alpha=0.7, label="90% line")
    ax.axhline(0.99, color="#E74C3C", ls=":",  lw=1, alpha=0.7, label="99% line")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative Energy Retained")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)

    # ── Plot 2: Subspace drift summary ────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Mean Subspace Drift by KV Mode")
    drifts     = [data[m]["profile"].get("mean_subspace_drift", 0) for m in modes]
    drift_stds = [data[m]["profile"].get("std_subspace_drift", 0) for m in modes]
    colors     = [MODE_COLORS.get(m, "#AAA") for m in modes]
    ax.bar(modes, drifts, yerr=drift_stds, color=colors, alpha=0.85,
           edgecolor="#0D1117", linewidth=0.5, capsize=5)
    ax.axhline(0.3, color="#F39C12", ls="--", lw=1, alpha=0.7,
               label="0.3 = 'stable' threshold")
    ax.set_ylabel("Mean Subspace Drift (0=stable, 1=random)")
    ax.legend(fontsize=8)

    # ── Plot 3: Per-layer rank for 90% energy ─────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Rank Required for 90% Energy per Layer")
    if layers:
        layer_idxs = [int(k) for k in sorted(layers.keys(), key=int)]
        ranks90    = [layers[str(li)]["rank_for_90pct"] for li in layer_idxs]
        drifts_l   = [layers[str(li)]["mean_drift"] for li in layer_idxs]

        ax2 = ax.twinx()
        ax2.set_ylabel("Mean Subspace Drift", color="#E74C3C")
        ax2.tick_params(axis="y", labelcolor="#E74C3C")

        ax.bar(layer_idxs, ranks90, color="#3498DB", alpha=0.7, label="Rank@90%")
        ax2.plot(layer_idxs, drifts_l, "o--", color="#E74C3C", lw=2,
                 markersize=5, label="Drift")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Rank for 90% Energy", color="#3498DB")
        ax.tick_params(axis="y", labelcolor="#3498DB")
        ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No layer data", ha="center", va="center",
                transform=ax.transAxes, color="#8B949E")

    # ── Plot 4: Compute cost — INT8 vs LoRA ──────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Storage: INT8 vs LoRA-Delta by Rank (mixed mode)")

    if "mixed" in data:
        costs = data["mixed"].get("compute_costs", {})
        ranks = []
        int8_mb, lr_mb = [], []
        for rk_key, cost in costs.items():
            r = int(rk_key.split("_")[1])
            ranks.append(r)
            int8_mb.append(cost["int8_bytes"] / 1024**2)
            lr_mb.append(cost["lowrank_bytes"] / 1024**2)

        x = np.arange(len(ranks))
        w = 0.35
        ax.bar(x - w/2, int8_mb, width=w, label="INT8 Deltas", color="#3498DB",
               alpha=0.8, edgecolor="#0D1117")
        ax.bar(x + w/2, lr_mb,  width=w, label="LoRA Deltas", color="#27AE60",
               alpha=0.8, edgecolor="#0D1117")
        ax.set_xticks(x)
        ax.set_xticklabels([f"r={r}" for r in ranks], fontsize=8)
        ax.set_ylabel("Storage (MB)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "lowrank_feasibility.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[OK] Low-rank plot -> {out_path}")


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_lowrank(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/lowrank_feasibility/lowrank_feasibility.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
