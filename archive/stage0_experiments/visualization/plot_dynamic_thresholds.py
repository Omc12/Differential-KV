"""
visualization/plot_dynamic_thresholds.py — Task 3

Plots the temporal evolution of dynamic thresholds.
Shows how the threshold reacts to changes in KV delta RMS.
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


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0D1117", "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D", "axes.labelcolor": "#C9D1D9",
        "axes.titlecolor": "#F0F6FC", "axes.grid": True,
        "grid.color": "#21262D", "grid.linewidth": 0.5,
        "xtick.color": "#8B949E", "ytick.color": "#8B949E",
        "text.color": "#C9D1D9", "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D", "legend.labelcolor": "#C9D1D9",
        "font.size": 10,
    })


def plot_traces(traces: dict, output_dir: Path):
    setup_style()

    # Create one subplot per policy
    n = len(traces)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1: axes = [axes]

    fig.suptitle("Dynamic Threshold Evolution & Anchor Triggering", fontsize=16,
                 color="#F0F6FC", fontweight="bold", y=0.95)

    colors = ["#3498DB", "#F39C12", "#27AE60", "#E74C3C"]

    for i, (label, data) in enumerate(traces.items()):
        ax = axes[i]
        rms = data["rms"]
        thresh = data["threshold"]
        anchors = data["anchors"]

        x = np.arange(len(rms))

        # Plot signal
        ax.plot(x, rms, color="#8B949E", alpha=0.4, label="Delta RMS", linewidth=1)

        # Plot threshold (pad/broadcast if needed)
        if len(thresh) == 1:
            ax.axhline(thresh[0], color=colors[i % len(colors)], ls="--",
                       label=f"Threshold ({label})", linewidth=2)
        else:
            # Handle cases where history might be slightly shorter than rms
            tx = np.arange(len(thresh))
            ax.plot(tx, thresh, color=colors[i % len(colors)], linewidth=2,
                    label=f"Threshold ({label})")

        # Plot anchor triggers
        ax.scatter(anchors, [0] * len(anchors), marker="|", s=100,
                   color="#F0F6FC", label="Anchor Trigger", zorder=5)

        ax.set_title(f"Policy: {label}", loc="left", color=colors[i % len(colors)],
                     fontweight="bold")
        ax.set_ylabel("RMS / Threshold")
        ax.set_ylim(0, max(max(rms), max(thresh)) * 1.1)
        ax.legend(loc="upper right", fontsize=8, ncol=3)

        # Add stats annotation
        stats = data.get("stats", {})
        trigger_summary = stats.get("trigger_counts", {})
        text = f"Anchors: {len(anchors)} | Ratio: {len(anchors)/len(rms):.1%}"
        if "final_threshold" in stats:
            text += f" | Final Thresh: {stats['final_threshold']:.3f}"
        if "current_percentile" in stats:
            text += f" | Pct: {stats['current_percentile']}%"

        ax.text(0.01, 0.05, text, transform=ax.transAxes, fontsize=9,
                color="#C9D1D9", bbox=dict(facecolor="#0D1117", alpha=0.8,
                                            edgecolor=colors[i % len(colors)],
                                            boxstyle="round,pad=0.5"))

    axes[-1].set_xlabel("Token Position")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    out_path = output_dir / "dynamic_threshold_evolution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[OK] Dynamic threshold plot saved -> {out_path}")


def main(args):
    with open(args.input) as f:
        traces = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_traces(traces, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/dynamic_thresholds/threshold_traces.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
