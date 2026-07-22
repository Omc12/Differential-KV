"""
visualization/plot_quality_eval.py — Phase 2.5 Objective 1

Plots behavioral quality evaluation results:
  1. Perplexity delta (vs baseline) per strategy and text type
  2. Token agreement bar chart
  3. KL divergence heatmap
  4. Compression ratio vs quality tradeoff scatter
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

STRATEGY_COLORS = {
    "baseline_fp16": "#8B949E",
    "periodic_64":   "#3498DB",
    "periodic_128":  "#85C1E9",
    "ema_balanced":  "#27AE60",
    "rolling_k3":    "#F39C12",
}


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


def plot_quality(data: dict, output_dir: Path):
    setup_style()

    summary   = data.get("summary", {})
    ppl_rows  = data.get("perplexity", [])
    drift_rows = data.get("drift", [])

    base_ppl  = summary.get("baseline_fp16", {}).get("perplexity")
    strategies = [k for k in summary if k != "baseline_fp16"]
    if not strategies:
        print("[WARN] No strategy data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"DKV Quality Evaluation — {data.get('model_name', '?')}",
                 fontsize=14, color="#F0F6FC", fontweight="bold")
    fig.patch.set_facecolor("#0D1117")

    # ── Plot 1: Perplexity delta per strategy ─────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Perplexity Delta vs FP16 Baseline")

    ppls  = [summary[s].get("perplexity", 0) or 0 for s in strategies]
    deltas = [round(p - (base_ppl or 0), 4) for p in ppls]
    colors = [STRATEGY_COLORS.get(s, "#E74C3C") for s in strategies]

    bars = ax.bar(strategies, deltas, color=colors, alpha=0.85,
                  edgecolor="#0D1117", linewidth=0.5)
    ax.axhline(0, color="#8B949E", ls="--", lw=1)
    ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Perplexity Delta (lower is better)")
    for bar, v in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:+.3f}", ha="center", fontsize=7, color="#C9D1D9")

    # ── Plot 2: Token agreement ───────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Token Agreement with FP16 Baseline")

    agrees = [summary[s].get("token_agreement") or 0 for s in strategies]
    ax.bar(strategies, agrees, color=colors, alpha=0.85,
           edgecolor="#0D1117", linewidth=0.5)
    ax.axhline(0.95, color="#27AE60", ls="--", lw=1, alpha=0.7, label="95% target")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Token Agreement (higher is better)")
    ax.legend(fontsize=8)

    # ── Plot 3: KL divergence ────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Mean KL Divergence from Baseline Logits")

    kl_vals = [summary[s].get("mean_kl_div") or 0 for s in strategies]
    ax.bar(strategies, kl_vals, color=colors, alpha=0.85,
           edgecolor="#0D1117", linewidth=0.5)
    ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("KL Divergence (lower is better)")
    ax.set_yscale("symlog", linthresh=0.001)

    # ── Plot 4: Compression ratio vs perplexity delta (Pareto) ───────────────
    ax = axes[1, 1]
    ax.set_title("Compression Ratio vs Perplexity Delta")

    for s in strategies:
        m = summary[s]
        ratio = m.get("compression_ratio") or 1.0
        delta = round((m.get("perplexity") or (base_ppl or 0)) - (base_ppl or 0), 4)
        color = STRATEGY_COLORS.get(s, "#E74C3C")
        ax.scatter(ratio, delta, c=color, s=120, label=s,
                   edgecolors="#0D1117", linewidth=0.5, alpha=0.9)
        ax.annotate(s, (ratio, delta), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color=color)

    ax.axhline(0, color="#8B949E", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Compression Ratio (higher = better memory)")
    ax.set_ylabel("Perplexity Delta (lower = better quality)")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    out_path = output_dir / f"quality_eval_{data.get('model_name','model')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[OK] Quality eval plot -> {out_path}")


def main(args):
    # Find report files
    input_path = Path(args.input)
    if input_path.is_dir():
        files = list(input_path.glob("*_quality_report.json"))
    else:
        files = [input_path]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        plot_quality(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/quality_eval/")
    parser.add_argument("--output", default="results/plots/")
    args = parser.parse_args()
    main(args)
