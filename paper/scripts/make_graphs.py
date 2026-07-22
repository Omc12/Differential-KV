#!/usr/bin/env python3
"""Measured-data graphs for the DiffKV paper — DeepSeek-report aesthetics (style v2).

All numbers come from paper/scripts/data.py (clean measured JSON) or are derived
from the runtime dimensions in the code. Charts carry no in-figure titles (the
LaTeX captions do); text is black; palette = blues + emerald.

  g_perf      two panels: (a) prefill time, (b) decode throughput — three engines
              (DiffKV, optimized mlx_lm dense, standard PyTorch dense); the naive
              PyTorch baseline OOMs at 16K+ (✗ markers, no fabricated value)
  g1_kv_footprint   analytic KV-state growth vs the bounded pool
  g6_residual_tradeoff  residual budget: compression ratio bars + decode tok/s line
  g7_decode_ablation    DiffKV compressed vs dense full-KV decode (primary sweep)
  g8_latency_breakdown  E11 per-step decode latency split (extruded 3-D pie)
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
CTX_PT = [c for c in D.CONTEXTS if c in _prim["normal_dense"]]   # Standard PyTorch, OK cells
PT_OOM_CTX = [c for c in D.CONTEXTS
              if c not in _prim["normal_dense"]
              and D.cell_status("normal_dense", c) in ("oom", "error")]
C_PT = "#6B5CA5"   # muted purple — Standard PyTorch (naive full KV) baseline


def _xk(c):
    return f"{c//1024}K"


# ── g_perf · (a) prefill, (b) decode — the headline two-panel ────────────────
def g_perf():
    a_pf = [_prim["active"][c]["prefill_s"] for c in CTX_ACT]
    d_pf = [_prim["dense"][c]["prefill_s"] for c in CTX_BOTH]
    p_pf = [_prim["normal_dense"][c]["prefill_s"] for c in CTX_PT]
    a_tp = [_prim["active"][c]["decode_tps"] for c in CTX_ACT]
    d_tp = [_prim["dense"][c]["decode_tps"] for c in CTX_BOTH]
    p_tp = [_prim["normal_dense"][c]["decode_tps"] for c in CTX_PT]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.85))

    # (a) prefill — three engines; Standard PyTorch OOMs at 16K+
    ax1.plot(CTX_BOTH, d_pf, "-o", color=S.C_DENSE_LN,
             label="Optimized dense (mlx_lm)", markerfacecolor="white")
    ax1.plot(CTX_PT, p_pf, "-^", color=C_PT, label="Standard PyTorch dense",
             markerfacecolor="white", markeredgecolor=C_PT)
    ax1.plot(CTX_ACT, a_pf, "-o", color=S.BLUE, label="DiffKV")
    ax1.set_yscale("log")
    ax1.set_ylabel("Prefill time (s)")
    S.context_ticks(ax1, CTX_ACT, "Context length")
    # PyTorch OOM at 16K+: ✗ markers just above its last (fast, unchunked) prefill
    for i, c in enumerate(PT_OOM_CTX):
        y = p_pf[-1] * 1.7
        ax1.plot([c], [y], marker="x", ms=7.5, mew=1.9,
                 color=S.OOM_RED, ls="none", zorder=5)
        if i == 0:
            ax1.annotate("PyTorch: OOM", (c, y), textcoords="offset points",
                         xytext=(6, 2), ha="left", va="bottom", fontsize=7.6,
                         color=S.BLACK, fontweight="bold")
    S.legend(ax1, loc="upper left")
    S.box_axes(ax1)

    # (b) decode — three engines
    ax2.plot(CTX_BOTH, d_tp, "-o", color=S.C_DENSE_LN,
             label="Optimized dense (mlx_lm)", markerfacecolor="white")
    ax2.plot(CTX_PT, p_tp, "-^", color=C_PT, label="Standard PyTorch dense",
             markerfacecolor="white", markeredgecolor=C_PT)
    ax2.plot(CTX_ACT, a_tp, "-o", color=S.BLUE, label="DiffKV")
    ax2.set_ylabel("Decode throughput (tokens/s)")
    ax2.set_ylim(0, max(d_tp) * 1.18)
    for i, c in enumerate(PT_OOM_CTX):
        y = p_tp[-1]
        ax2.plot([c], [y], marker="x", ms=7.5, mew=1.9,
                 color=S.OOM_RED, ls="none", zorder=5)
        if i == 0:
            ax2.annotate("PyTorch: OOM", (c, y), textcoords="offset points",
                         xytext=(6, 6), ha="left", va="bottom", fontsize=7.6,
                         color=S.BLACK, fontweight="bold")
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
    bars = ax.bar(x, ratio, 0.40, color=S.BLUE, edgecolor=S.BLACK, linewidth=0.7,
                  label="Compression ratio (left)")
    # value labels INSIDE the bars (white, normal weight) — adjusted for thinner bars
    for r, v in zip(bars, ratio):
        ax.annotate(f"{v:.2f}$\\times$",
                    (r.get_x() + r.get_width() / 2, v - 0.07),
                    ha="center", va="top", fontsize=6.6, color="white")
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


# ── g7 · decode throughput of the three engines (primary sweep) ──────────────
def g7_decode_ablation():
    # Three-engine decode-throughput comparison (shipping config, 4k..64k):
    # optimized mlx_lm dense, DiffKV compressed sparse, and the raw PyTorch dense
    # (which OOMs at 16k+). Complements the same-store compressed-vs-exact
    # mechanism ablation in Table 4 (t6_decode_ablation).
    prim = D.load_primary()
    ctx = [c for c in D.CONTEXTS if c in prim["active"] and c in prim["dense"]]
    e_tps = [prim["dense"][c]["decode_tps"] for c in ctx]     # optimized dense
    c_tps = [prim["active"][c]["decode_tps"] for c in ctx]    # DiffKV compressed
    p_tps = [prim["normal_dense"][c]["decode_tps"] if c in prim["normal_dense"]
             else None for c in ctx]                          # raw PyTorch dense
    x = np.arange(len(ctx)); w = 0.26

    fig, ax = plt.subplots(figsize=(4.9, 2.95))
    b1 = ax.bar(x - w, e_tps, w, color=S.BLUE_L, edgecolor=S.BLACK,
                linewidth=0.7, label="Optimized dense (full KV)")
    b2 = ax.bar(x, c_tps, w, color=S.BLUE, edgecolor=S.BLACK,
                linewidth=0.7, hatch="//", label="DiffKV compressed sparse")
    p_x = [x[i] + w for i, v in enumerate(p_tps) if v is not None]
    p_y = [v for v in p_tps if v is not None]
    b3 = ax.bar(p_x, p_y, w, color=C_PT, edgecolor=S.BLACK, linewidth=0.7,
                label="Standard PyTorch dense (raw full KV)")

    # value labels above each bar — small font + the 3-bar spacing (centres w
    # apart) keeps adjacent labels (e.g. 20.2 vs 21.4 at 64k) from colliding
    def _labels(bars, fs=6.0):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 1.6), ha="center",
                        va="bottom", fontsize=fs, color=S.BLACK)
    _labels(b1); _labels(b2); _labels(b3)

    # OOM tick where the raw PyTorch dense did not run (16k+)
    for i, v in enumerate(p_tps):
        if v is None:
            ax.text(x[i] + w, 1.8, "OOM", rotation=90, ha="center", va="bottom",
                    fontsize=5.6, color=S.OOM_RED, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([_xk(c) for c in ctx])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_ylim(0, max(e_tps) * 1.24)
    S.legend(ax, loc="upper right", fontsize=7.0)
    S.box_axes(ax)
    S.finalize(fig, os.path.join(FIG, "g7_decode_ablation.png"))


# ── g8 · E11 decode-latency breakdown (extruded 3-D pie) ────────────────────
def g8_latency_breakdown():
    """E11 decode-step breakdown as an extruded 3-D pie.

    Same visual grammar as F6: orthographic squash, thin black strokes, the
    blue/emerald semantic palette. Top faces carry the percentages; the
    right-hand key carries names, ms values, and the takeaway.
    """
    from matplotlib.patches import Polygon, Rectangle
    BLACK, BLUE, BLUE_D, BLUE_L, EMERALD, GRAY_D, LEGEND_EC, WHITE = (
        S.BLACK, S.BLUE, S.BLUE_D, S.BLUE_L, S.EMERALD, S.GRAY_D,
        S.LEGEND_EC, S.WHITE)

    parts = [
        ("low-rank SVD scoring",           35, 378, BLUE,    WHITE),
        ("residual attention",             28, 302, EMERALD, WHITE),
        ("cache merge",                    15, 162, BLUE_D,  WHITE),
        ("SRL query routing",              12, 130, GRAY_D,  WHITE),
        ("fused-buffer materialisation",   10, 108, BLUE_L,  BLACK),
    ]

    def darken(hexcol, f=0.68):
        h = hexcol.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))

    fig = plt.figure(figsize=(7.6, 3.75))
    ax = fig.add_axes([0.015, 0.03, 0.50, 0.94])
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.82, 1.88); ax.set_ylim(-1.42, 0.98)

    SQ, DEPTH, R = 0.55, 0.22, 1.0

    # slice angle ranges — clockwise from 12 o'clock, like the F6-era charts
    edges, acc = [], 90.0
    for _, pct, _, _, _ in parts:
        edges.append((acc, acc - pct * 3.6))
        acc -= pct * 3.6

    def arc(th_hi, th_lo, n=None):
        n = n or max(12, int((th_hi - th_lo) / 2.5))
        return np.linspace(th_lo, th_hi, n)

    # rims first (they sit under the top faces), front half only (sin < 0)
    for (th1, th2), (_, _, _, c, _) in zip(edges, parts):
        ts = arc(th1, th2, 120)
        mask = np.sin(np.radians(ts)) < -1e-9
        if not mask.any():
            continue
        runs = np.split(np.where(mask)[0], np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
        for run in runs:
            tt = ts[run]
            if len(tt) < 2:
                continue
            top = [(R * np.cos(np.radians(t)), SQ * R * np.sin(np.radians(t))) for t in tt]
            bot = [(x, y - DEPTH) for x, y in reversed(top)]
            ax.add_patch(Polygon(top + bot, closed=True, fc=darken(c), ec=BLACK,
                                 lw=0.4, zorder=2))

    # top faces
    for (th1, th2), (_, pct, _, c, tc) in zip(edges, parts):
        ts = arc(th1, th2)
        pts = [(0.0, 0.0)] + [(R * np.cos(np.radians(t)), SQ * R * np.sin(np.radians(t)))
                              for t in ts]
        ax.add_patch(Polygon(pts, closed=True, fc=c, ec=BLACK, lw=0.45, zorder=3))

    # ── elbow leaders: label in the white margin, horizontal run, then a
    #    vertical drop (or rise) straight onto the slice ──
    #    (x_anchor, y_touch) = where the vertical meets the slice;
    #    y_h = the horizontal run's height; x_text = label position
    leaders = [
        ("35%", 0.891,  0.300,  0.62,  1.42, "left"),    # upper right, drops down
        ("10%", -0.309, 0.575,  0.86, -0.72, "right"),   # top, drops down
        ("12%", -0.845, 0.345,  0.62, -1.42, "right"),   # upper left, drops down
        ("15%", -0.960, -0.425, -0.68, -1.42, "right"),  # lower left, rises to the rim
        ("28%", 0.063,  -0.820, -1.10,  0.55, "left"),   # bottom, rises to the rim
    ]
    for txt, xa, y_end, y_h, x_text, ha in leaders:
        x_line = x_text - 0.05 if ha == "left" else x_text + 0.05
        ax.plot([x_line, xa], [y_h, y_h], color=BLACK, lw=0.7, zorder=5)
        ax.plot([xa, xa], [y_h, y_end], color=BLACK, lw=0.7, zorder=5)
        ax.text(x_text, y_h, txt, ha=ha, va="center", fontsize=8.6,
                fontweight="bold", color=BLACK, zorder=6)

    ax.text(0.03, -1.32, "one decode step  ≈ 1,080 ms   (instrumented, 16k)",
            ha="center", va="center", fontsize=8.4, color=BLACK)

    # ── right-hand key: swatch · component · ms ──
    ax2 = fig.add_axes([0.545, 0.02, 0.445, 0.96]); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    y = 0.955
    for name, pct, ms, c, _ in parts:
        ax2.add_patch(Rectangle((0.005, y - 0.030), 0.052, 0.060, fc=c, ec=BLACK, lw=0.5,
                                transform=ax2.transAxes, clip_on=False))
        ax2.text(0.080, y + 0.014, name, ha="left", va="center", fontsize=8.0, color=BLACK)
        ax2.text(0.080, y - 0.040, f"{pct}%   ·   ≈{ms} ms", ha="left", va="center",
                 fontsize=7.2, color=BLACK)
        y -= 0.132
    ax2.plot([0.005, 0.985], [0.275, 0.275], color=LEGEND_EC, lw=0.8,
             transform=ax2.transAxes, clip_on=False)
    ax2.text(0.005, 0.215,
             "63% of the step is the per-token low-rank\n"
             "contraction $q\\cdot V_K$ plus the residual attend —\n"
             "not buffer materialisation. Collapsing the two\n"
             "into one fused launch is the open optimisation.",
             ha="left", va="top", fontsize=7.2, color=BLACK, linespacing=1.5,
             fontstyle="italic")

    S.finalize(fig, os.path.join(FIG, "g8_latency_breakdown.png"))


if __name__ == "__main__":
    g_perf()
    g1_kv_footprint()
    g6_residual_tradeoff()
    g7_decode_ablation()
    g8_latency_breakdown()
    print("\ngraphs done ->", FIG)
