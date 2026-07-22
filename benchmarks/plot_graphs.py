#!/usr/bin/env python3
"""
Generate publication-quality figures from results_latest.json.

Outputs (in benchmarks/results/):
  fig_memory.png     — peak memory vs context length
  fig_decode_tps.png — decode throughput vs context length
  fig_prefill.png    — prefill time vs context length
  fig_combined.png   — all three panels side-by-side
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "results_latest.json")
OUT_DIR = os.path.join(HERE, "results")

ENGINES = ["active", "dense", "normal_dense"]
COLORS = {
    "active": "#1565C0",
    "dense": "#2e7d32",
    "normal_dense": "#d35400",
}
LABELS = {
    "active": "DiffKV Active (MLX int4, compressed KV)",
    "dense": "Optimized Dense (mlx_lm int4, full KV)",
    "normal_dense": "Standard PyTorch Dense (AutoModelForCausalLM)",
}
MARKERS = {
    "active": "o",
    "dense": "^",
    "normal_dense": "v",
}

RAM_CAP_GB = 7.2
RAM_TOTAL_GB = 8.6

CTX_LABELS = {4096: "4k", 8192: "8k", 16384: "16k",
               32768: "32k", 65536: "64k", 131072: "128k"}


def is_trunc(r):
    return (r.get("engine") == "ollama"
            and r.get("status") == "ok"
            and (r.get("gen_tokens") or 0) <= 1)


def load():
    blob = json.load(open(RESULTS))
    by = {(r["engine"], r["ctx_target"]): r for r in blob["results"]}
    contexts = blob["meta"]["contexts"]
    return by, contexts, blob["meta"]


def _xticks(contexts):
    return list(range(len(contexts))), [CTX_LABELS[c] for c in contexts]


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 2.2,
        "lines.markersize": 8,
        "legend.fontsize": 9.5,
        "legend.framealpha": 0.85,
        "figure.dpi": 150,
    })


def plot_memory(ax, by, contexts):
    """Two lines per engine, same colour:
         solid  = KV-cache footprint (kv_mem_gb) — what the method bounds
         dashed = peak process RAM (peak_mem_gb) — what decides whether it fits
    so the figure separates the resident KV state (compressed) from the transient
    peak that drives PyTorch dense into the host's memory ceiling."""
    xs, xlabs = _xticks(contexts)
    # dashed-peak label sits indented under its engine's solid KV label; the
    # short parenthetical keeps each string unique for the legend lookup
    PEAK_LABELS = {
        "active":       "  └ peak process RAM (DiffKV)",
        "dense":        "  └ peak process RAM (Dense)",
        "normal_dense": "  └ peak process RAM (PyTorch)",
    }

    for e in ENGINES:
        # ── solid: measured KV-cache footprint (only where the run completed) ──
        kx = [i for i, c in enumerate(contexts)
              if by.get((e, c)) and by[(e, c)]["status"] == "ok"
              and by[(e, c)].get("kv_mem_gb") is not None]
        ky = [by[(e, contexts[i])]["kv_mem_gb"] for i in kx]
        if kx:
            ax.plot(kx, ky, color=COLORS[e], marker=MARKERS[e], label=LABELS[e])

        # ── dashed: measured peak RAM (through OK and OOM cells alike; the peak
        #    at kill time is a real measurement) ──
        px = [i for i, c in enumerate(contexts)
              if by.get((e, c)) and by[(e, c)]["status"] in ("ok", "oom")
              and by[(e, c)].get("peak_mem_gb") is not None]
        py = [by[(e, contexts[i])]["peak_mem_gb"] for i in px]
        if px:
            ax.plot(px, py, color=COLORS[e], marker=MARKERS[e], linestyle="--",
                    linewidth=1.8, markerfacecolor="white", markersize=6,
                    alpha=0.95, label=PEAK_LABELS[e])

        # ── OOM cells: red X on the peak line + a single "OOM" annotation ──
        oom_pts = [(i, by[(e, contexts[i])]["peak_mem_gb"])
                   for i, c in enumerate(contexts)
                   if by.get((e, c)) and by[(e, c)]["status"] == "oom"
                   and by[(e, c)].get("peak_mem_gb") is not None]
        for j, (i, y) in enumerate(oom_pts):
            ax.plot(i, y, "x", color=COLORS[e], markersize=11,
                    markeredgewidth=2.5, zorder=6)
            if j == 0:
                ax.annotate("OOM", xy=(i, y), xytext=(7, -13),
                            textcoords="offset points", ha="left", va="top",
                            fontsize=8.5, fontweight="bold", color=COLORS[e])

    # System-RAM ceiling: the line the PyTorch peak RAM slams into at OOM
    ax.axhline(RAM_TOTAL_GB, color="#555", linestyle="-.", linewidth=1.2,
               label=f"System RAM ({RAM_TOTAL_GB} GB)")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabs)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Memory vs context length\n"
                 "(solid = KV cache · dashed = peak process RAM)")
    ax.set_ylim(0, RAM_TOTAL_GB * 1.06)
    # group each engine's dashed peak line directly under its solid KV line
    handles, labels = ax.get_legend_handles_labels()
    lut = {l: h for l, h in zip(labels, handles)}
    order = []
    for e in ENGINES:
        order += [LABELS[e], PEAK_LABELS[e]]
    order.append(f"System RAM ({RAM_TOTAL_GB} GB)")
    oh = [lut[k] for k in order if k in lut]
    ol = [k for k in order if k in lut]
    ax.legend(oh, ol, loc="upper left", ncol=1, fontsize=8.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f GB"))


def plot_decode_tps(ax, by, contexts):
    xs, xlabs = _xticks(contexts)
    for e in ENGINES:
        px, py = [], []
        for i, c in enumerate(contexts):
            r = by.get((e, c))
            if r and r["status"] == "ok" and not is_trunc(r):
                v = r.get("decode_tps")
                if v and v < 1e5:
                    px.append(i)
                    py.append(v)
        if px:
            ax.plot(px, py, color=COLORS[e], marker=MARKERS[e], label=LABELS[e])

        # OOM / fail markers
        for i, c in enumerate(contexts):
            r = by.get((e, c))
            if not r:
                continue
            if r["status"] == "oom":
                # draw a downward arrow at the last valid decode_tps for that engine
                last_valid = max(
                    [j for j, cc in enumerate(contexts)
                     if by.get((e, cc)) and by[(e, cc)]["status"] == "ok"
                     and not is_trunc(by[(e, cc)])
                     and (by[(e, cc)].get("decode_tps") or 0) < 1e5],
                    default=None
                )
                if last_valid is not None:
                    last_tps = by[(e, contexts[last_valid])]["decode_tps"]
                    ax.annotate("OOM", xy=(last_valid, last_tps),
                                xytext=(last_valid + 0.15, last_tps * 1.08),
                                fontsize=7.5, color=COLORS[e],
                                arrowprops=dict(arrowstyle="->", color=COLORS[e],
                                                lw=1.2))
                break
        # trunc annotation for ollama
        if e == "ollama":
            for i, c in enumerate(contexts):
                r = by.get((e, c))
                if r and is_trunc(r):
                    ax.annotate("trunc\n@32k", xy=(i, 5), fontsize=7,
                                color=COLORS[e], ha="center",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          fc="white", ec=COLORS[e], alpha=0.7))
                    break

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabs)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Decode throughput vs context length\n"
                 "(Standard PyTorch dense OOMs at 16k+)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f tok/s"))


def plot_prefill(ax, by, contexts):
    xs, xlabs = _xticks(contexts)
    for e in ENGINES:
        px, py = [], []
        for i, c in enumerate(contexts):
            r = by.get((e, c))
            if r and r["status"] == "ok" and not is_trunc(r):
                v = r.get("prefill_s")
                if v is not None:
                    px.append(i)
                    py.append(v)
        if px:
            ax.plot(px, py, color=COLORS[e], marker=MARKERS[e], label=LABELS[e])

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabs)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Prefill time (s)")
    ax.set_title("Prefill time vs context length\n"
                 "(Standard PyTorch dense OOMs at 16k+)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f s"))


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {os.path.relpath(path, os.path.dirname(HERE))}")
    return path


def main():
    _style()
    by, contexts, meta = load()

    host_line = (f"{meta['model']} · {meta['chip']}, "
                 f"{meta['ram_gb']:.1f} GB unified memory · greedy, 128 gen tok")

    # ── Individual figures ──────────────────────────────────────────────────
    for plot_fn, fname, figsize in [
        (plot_memory,     "fig_memory.png",     (8, 5)),
        (plot_decode_tps, "fig_decode_tps.png", (8, 5)),
        (plot_prefill,    "fig_prefill.png",    (8, 5)),
    ]:
        fig, ax = plt.subplots(figsize=figsize)
        plot_fn(ax, by, contexts)
        fig.suptitle(host_line, fontsize=9, color="#555", y=1.01)
        fig.tight_layout()
        save(fig, fname)
        plt.close(fig)

    # ── Combined 3-panel figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    plot_memory(axes[0], by, contexts)
    plot_decode_tps(axes[1], by, contexts)
    plot_prefill(axes[2], by, contexts)
    # remove per-panel legends, add a single shared legend below
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels and not ll.startswith("_"):
                handles.append(hh)
                labels.append(ll)
        ax.get_legend().remove()

    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.07), fontsize=9.5, framealpha=0.9)
    fig.suptitle(
        "DiffKV vs Optimized Dense vs Standard PyTorch — long-context benchmark\n"
        f"{host_line}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig_combined.png")
    plt.close(fig)

    print("done.")


if __name__ == "__main__":
    main()
