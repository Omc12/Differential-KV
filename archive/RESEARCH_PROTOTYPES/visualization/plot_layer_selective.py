"""
visualization/plot_layer_selective.py — Phase 2.5 Objective 2

Plots layer-selective compression analysis:
  1. Per-layer compression heatmap (mode x selector)
  2. Per-layer error heatmap
  3. Selector comparison bar chart
  4. Layer contribution to total compression savings
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
        "axes.titlecolor": "#F0F6FC", "axes.grid": False,
        "xtick.color": "#8B949E", "ytick.color": "#8B949E",
        "text.color": "#C9D1D9", "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D", "legend.labelcolor": "#C9D1D9",
        "font.size": 8, "axes.titlesize": 10,
    })


def plot_layer_selective(data: dict, output_dir: Path):
    setup_style()

    # Use "mixed" mode for primary analysis
    mode = "mixed" if "mixed" in data else list(data.keys())[0]
    mode_data = data[mode]
    selector_names = list(mode_data.keys())

    # Build error heatmap matrix: [selectors x layers]
    max_layers = 0
    for sel_data in mode_data.values():
        layers = [int(k) for k in sel_data.get("per_layer", {}).keys()]
        max_layers = max(max_layers, max(layers) + 1) if layers else max_layers

    error_matrix = np.full((len(selector_names), max_layers), np.nan)
    ratio_matrix = np.full((len(selector_names), max_layers), np.nan)

    for si, sel_name in enumerate(selector_names):
        pl = mode_data[sel_name].get("per_layer", {})
        for li_str, stats in pl.items():
            li = int(li_str)
            if li < max_layers:
                error_matrix[si, li] = stats["mean_error"]
                ratio_matrix[si, li] = stats["compression_ratio"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Layer-Selective Compression — KV Mode: {mode}",
                 fontsize=13, color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Heatmap 1: Reconstruction error ──────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Reconstruction Error Heatmap (layer x selector)")
    im = ax.imshow(error_matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=min(0.05, np.nanmax(error_matrix)))
    plt.colorbar(im, ax=ax, label="Mean Relative Error")
    ax.set_yticks(range(len(selector_names)))
    ax.set_yticklabels(selector_names, fontsize=7)
    ax.set_xlabel("Layer Index")

    # ── Heatmap 2: Compression ratio ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Compression Ratio Heatmap")
    im2 = ax.imshow(ratio_matrix, cmap="YlGn", aspect="auto",
                    vmin=1.0, vmax=np.nanmax(ratio_matrix))
    plt.colorbar(im2, ax=ax, label="Compression Ratio")
    ax.set_yticks(range(len(selector_names)))
    ax.set_yticklabels(selector_names, fontsize=7)
    ax.set_xlabel("Layer Index")

    # ── Bar chart: Overall tradeoff ───────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Overall Compression vs Mean Error")
    colors = plt.cm.Set2(np.linspace(0, 1, len(selector_names)))

    totals = [mode_data[s].get("totals", {}) for s in selector_names]
    ratios = [t.get("mean_compression_ratio", 1.0) for t in totals]
    errors = [t.get("mean_error", 0) for t in totals]

    for i, (sel_name, ratio, error) in enumerate(zip(selector_names, ratios, errors)):
        ax.scatter(ratio, error, c=[colors[i]], s=120, label=sel_name,
                   edgecolors="#0D1117", linewidth=0.5)
        ax.annotate(sel_name.replace("_", "\n"), (ratio, error),
                    textcoords="offset points", xytext=(3, 3),
                    fontsize=6, color=colors[i])

    ax.set_xlabel("Mean Compression Ratio")
    ax.set_ylabel("Mean Reconstruction Error")
    ax.legend(fontsize=6, loc="upper right")

    # ── Bar chart: Per-layer error for best vs worst selector ─────────────────
    ax = axes[1, 1]
    ax.set_title("Per-Layer Error: Progressive vs Uniform Periodic")
    x = np.arange(min(max_layers, 32))
    w = 0.4

    for si, sel_name in enumerate(["uniform_periodic", "progressive"]):
        if sel_name not in mode_data:
            continue
        pl = mode_data[sel_name].get("per_layer", {})
        errs = [pl.get(str(li), {}).get("mean_error", 0) for li in range(min(max_layers, 32))]
        offset = (si - 0.5) * w
        color = "#3498DB" if "periodic" in sel_name else "#27AE60"
        ax.bar(x + offset, errs, width=w, label=sel_name, color=color,
               alpha=0.8, edgecolor="#0D1117", linewidth=0.3)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Reconstruction Error")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "layer_selective_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[OK] Layer selective plot -> {out_path}")


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_layer_selective(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/layer_selective/layer_selective_results.json")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
