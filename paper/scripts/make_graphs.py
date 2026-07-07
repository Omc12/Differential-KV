#!/usr/bin/env python3
"""Measured-data graphs for the DiffKV paper — DeepSeek-report aesthetics (style v2).

All numbers come from paper/scripts/data.py (clean measured JSON) or are derived
from the runtime dimensions in the code. Charts carry no in-figure titles (the
LaTeX captions do); text is black; palette = blues + emerald.

  g_perf      two panels: (a) prefill time, (b) decode throughput — incl. 64K reach
              where the dense baseline OOMs (marker, no fabricated value)
  g1_kv_footprint   analytic KV-state growth vs the bounded pool
  g6_residual_tradeoff  residual budget: compression ratio bars + decode tok/s line
  g7_decode_ablation    compressed vs exact decode over the same store
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S           # noqa: E402
import data as D            # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402

S.apply_rc()
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG, exist_ok=True)
GB = 1e9

_prim = D.load_primary()
CTX_BOTH = [c for c in D.CONTEXTS if c in _prim["active"] and c in _prim["dense"]]
CTX_ACT = [c for c in D.CONTEXTS if c in _prim["active"]]
DENSE_OOM_CTX = [c for c in D.CONTEXTS
                 if c not in _prim["dense"] and D.cell_status("dense", c) == "oom"]


def _xk(c):
    return f"{c//1024}K"


# ── g_perf · (a) prefill, (b) decode — the headline two-panel ────────────────
def g_perf():
    a_pf = [_prim["active"][c]["prefill_s"] for c in CTX_ACT]
    d_pf = [_prim["dense"][c]["prefill_s"] for c in CTX_BOTH]
    a_tp = [_prim["active"][c]["decode_tps"] for c in CTX_ACT]
    d_tp = [_prim["dense"][c]["decode_tps"] for c in CTX_BOTH]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.85))

    # (a) prefill
    ax1.plot(CTX_BOTH, d_pf, "-o", color=S.C_DENSE_LN, label="Dense (full KV)",
             markerfacecolor="white")
    ax1.plot(CTX_ACT, a_pf, "-o", color=S.BLUE, label="DiffKV")
    ax1.set_yscale("log")
    ax1.set_ylabel("Prefill time (s)")
    S.context_ticks(ax1, CTX_ACT, "Context length")
    # dense OOM at 64K: ✗ marker in the empty region under the DiffKV endpoint
    # (no implied value, no fabricated guide line)
    for c in DENSE_OOM_CTX:
        y = d_pf[-1] * 2.0
        ax1.plot([c], [y], marker="x", ms=7.5, mew=1.9,
                 color=S.OOM_RED, ls="none", zorder=5)
        ax1.annotate("Dense: OOM", (c, y), textcoords="offset points",
                     xytext=(-7, 0), ha="right", va="center", fontsize=7.8,
                     color=S.BLACK, fontweight="bold")
    S.legend(ax1, loc="upper left")
    S.box_axes(ax1)

    # (b) decode
    ax2.plot(CTX_BOTH, d_tp, "-o", color=S.C_DENSE_LN, label="Dense (full KV)",
             markerfacecolor="white")
    ax2.plot(CTX_ACT, a_tp, "-o", color=S.BLUE, label="DiffKV")
    ax2.set_ylabel("Decode throughput (tokens/s)")
    ax2.set_ylim(0, max(d_tp) * 1.18)
    # reserved series slot for the CUDA fused decode path (future work; no data)
    ax2.plot([], [], "s", color="white", markeredgecolor=S.GRAY_D,
             label="CUDA fused decode (future work)")
    for c in DENSE_OOM_CTX:
        y = d_tp[-1] * 0.88
        ax2.plot([c], [y], marker="x", ms=7.5, mew=1.9,
                 color=S.OOM_RED, ls="none", zorder=5)
        ax2.annotate("Dense: OOM", (c, y),
                     textcoords="offset points", xytext=(-7, 0), ha="right",
                     va="center", fontsize=7.8, color=S.BLACK, fontweight="bold")
    S.context_ticks(ax2, CTX_ACT, "Context length")
    S.legend(ax2, loc="upper right")
    S.box_axes(ax2)

    fig.text(0.27, -0.04, "(a) Prefilling", ha="center", fontsize=9, color=S.BLACK)
    fig.text(0.77, -0.04, "(b) Decoding", ha="center", fontsize=9, color=S.BLACK)
    fig.tight_layout(w_pad=2.4)
    S.finalize(fig, os.path.join(FIG, "g_perf.png"))


# ── g1 · KV-state footprint (analytic, bounded pool vs growing dense) ────────
def g1_kv_footprint():
    ctxs = D.CONTEXTS
    d128 = [D.analytic_footprint(c, 128) for c in ctxs]
    d64 = [D.analytic_footprint(c, 64) for c in ctxs]
    dense = [f["dense_bytes"] / GB for f in d128]
    s128 = [f["store_bytes"] / GB for f in d128]
    s64 = [f["store_bytes"] / GB for f in d64]
    cap = (D.DIMS["n_layers"] * D.DIMS["max_blocks"] * D.block_budget(128)["total"]
           + D.DIMS["n_layers"] * 768 * D.DIMS["kv_heads"] * D.DIMS["head_dim"] * 4) / GB

    fig, ax = plt.subplots(figsize=(4.9, 3.05))
    ax.plot(ctxs, dense, "-o", color=S.C_DENSE_LN, label="Dense full KV",
            markerfacecolor="white")
    ax.plot(ctxs, s128, "-o", color=S.BLUE, label="DiffKV store ($R{=}128$, default)")
    ax.plot(ctxs, s64, "--s", color=S.BLUE_L, label="DiffKV store ($R{=}64$ preset)",
            markerfacecolor="white", markeredgecolor=S.BLUE)
    ax.axhline(cap, color=S.EMERALD, lw=1.3, ls=(0, (4, 3)),
               label=f"Pool capacity bound ({cap:.2f} GB)")
    ax.fill_between(ctxs, s128, dense, color=S.BLUE_XL, alpha=0.55, zorder=0)
    for c, u, dn in [(ctxs[-1], s128[-1], dense[-1])]:
        ax.annotate(f"{dn/u:.2f}$\\times$", (c, (u * dn) ** 0.5), fontsize=8.5,
                    color=S.BLACK, fontweight="bold", ha="right",
                    textcoords="offset points", xytext=(-2, 0))
    S.context_ticks(ax, ctxs, "Context length")
    ax.set_ylabel("KV-cache state (GB, all layers)")
    ax.set_ylim(0, max(dense) * 1.12)
    S.legend(ax, loc="upper left")
    S.box_axes(ax)
    S.finalize(fig, os.path.join(FIG, "g1_kv_footprint.png"))


# ── g6 · residual budget: ratio bars + decode line, needle row ────────────────
def g6_residual_tradeoff():
    rows = D.load_residual_sweep()
    R = [r["max_residual"] for r in rows]
    ratio = [r["kv"]["ratio_used_vs_dense"] for r in rows]
    tps = [r["decode_tps"] for r in rows]
    needle = [r["needle_found"] for r in rows]
    x = np.arange(len(R))

    fig, ax = plt.subplots(figsize=(4.9, 3.05))
    bars = ax.bar(x, ratio, 0.58, color=S.BLUE, edgecolor=S.BLACK, linewidth=0.7,
                  label="Compression ratio (left)")
    # value labels INSIDE the bars (white, bold) — immune to line/legend collisions
    for r, v in zip(bars, ratio):
        ax.annotate(f"{v:.2f}$\\times$",
                    (r.get_x() + r.get_width() / 2, v - 0.07),
                    ha="center", va="top", fontsize=7.6, color="white",
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in R])
    ax.set_xlabel("Residual budget $R$ (exact tokens per block)")
    ax.set_ylabel("Compression ratio vs dense")
    # ranges chosen so the tok/s line rides ABOVE every bar top (no collisions)
    ax.set_ylim(0, max(ratio) * 1.45)

    ax2 = ax.twinx()
    ax2.plot(x, tps, "-o", color=S.EMERALD, label="Decode tok/s (right)",
             markerfacecolor="white")
    ax2.set_ylabel("Decode throughput (tokens/s)")
    ax2.set_ylim(0, max(tps) * 1.30)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    lg = ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.6)
    lg.get_frame().set_edgecolor(S.LEGEND_EC); lg.get_frame().set_linewidth(0.7)
    S.box_axes(ax); S.box_axes(ax2)
    S.finalize(fig, os.path.join(FIG, "g6_residual_tradeoff.png"))


# ── g7 · compressed vs exact decode over the same store ──────────────────────
def g7_decode_ablation():
    comp = D.modes_by("compressed")
    exact = D.modes_by("exact")
    ctx = [c for c in D.CONTEXTS if c in comp and c in exact]
    c_tps = [comp[c]["decode_tps"] for c in ctx]
    e_tps = [exact[c]["decode_tps"] for c in ctx]
    x = np.arange(len(ctx)); w = 0.36

    fig, ax = plt.subplots(figsize=(4.9, 2.95))
    b1 = ax.bar(x - w / 2, e_tps, w, color=S.BLUE_L, edgecolor=S.BLACK,
                linewidth=0.7, label="Exact decode (upper bound)")
    b2 = ax.bar(x + w / 2, c_tps, w, color=S.BLUE, edgecolor=S.BLACK,
                linewidth=0.7, hatch="//", label="Compressed sparse decode")
    S.bar_value_labels(ax, b1); S.bar_value_labels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels([_xk(c) for c in ctx])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_ylim(0, max(e_tps) * 1.22)
    S.legend(ax, loc="upper right", fontsize=7.6)
    S.box_axes(ax)
    S.finalize(fig, os.path.join(FIG, "g7_decode_ablation.png"))


if __name__ == "__main__":
    g_perf()
    g1_kv_footprint()
    g6_residual_tradeoff()
    g7_decode_ablation()
    print("\ngraphs done ->", FIG)
