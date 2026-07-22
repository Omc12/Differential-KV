#!/usr/bin/env python3
"""Architecture / dataflow diagrams for the DKV paper — DeepSeek-report style.

Visual language (matched to DeepSeek-V3.2 Figures 2 and 4):
  flat rounded rectangles, thin black strokes, light fills; medium blue with
  white text for "hot" compute; emerald tints reserved for the EXACT path
  (residuals / recency); tiny italic annotations; no in-figure titles (the
  LaTeX caption carries the title); text otherwise always black.

Structural numbers match the code at the measured config (rank 32, R=128,
block 256, window 1,024, pool 256, top-K 16).

  F1  system architecture        F2  compression pipeline
  F4  cache lifecycle            F5a routed sparse decode attention
  F5b fused decode buffer        F6  3D memory architecture
"""
import os
import sys
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import (BLACK, BLUE, BLUE_D, BLUE_L, BLUE_XL, EMERALD, EMER_L,
                   GRAY, GRAY_XL, GRAY_D, LEGEND_EC, WHITE)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

style.apply_rc()
plt.rcParams.update({"axes.grid": False})
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)


# ── drawing helpers ──────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, fc=GRAY_XL, ec=BLACK, tc=BLACK, fs=8.4, lw=0.9,
        bold=False, rounding=0.035, style_="round"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"{style_},pad=0.008,rounding_size={rounding}"
                       if style_ == "round" else f"{style_},pad=0.008",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, zorder=4, fontweight="bold" if bold else "normal",
                linespacing=1.3)
    return (x, y, w, h)


def chip(ax, x, y, w, h, title, body, fc=WHITE, ec=EMERALD, tc=BLACK,
         fs_t=8.0, fs_b=6.9, lw=0.9):
    """Titled chip: bold heading on top, small body below."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.035",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.235, title, ha="center", va="center",
            fontsize=fs_t, color=tc, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + (h - 0.42) / 2, body, ha="center", va="center",
            fontsize=fs_b, color=tc, zorder=4, linespacing=1.35)


def band(ax, x, y, w, h, fc=GRAY_XL, ec=LEGEND_EC, lw=0.8, ls="-", dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, zorder=1,
                       linestyle=(0, (4, 3)) if dashed else "-")
    ax.add_patch(p)


def arrow(ax, p0, p1, color=BLACK, lw=1.0, rad=0.0, ls="-", style_="-|>", z=2, **kwargs):
    a = FancyArrowPatch(p0, p1, arrowstyle=style_, mutation_scale=10, color=color,
                        lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=z,
                        linestyle=ls, **kwargs)
    ax.add_patch(a)


def note(ax, x, y, text, fs=7.0, ha="center", va="center", style_="italic"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=BLACK,
            fontstyle=style_, zorder=5)


def clean(ax, xmax=10, ymax=10):
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")


# ── F1 · system architecture ─────────────────────────────────────────────────
def f1_architecture():
    fig, ax = plt.subplots(figsize=(8.6, 5.9)); clean(ax, 10, 9.6)

    # serving band
    band(ax, 0.3, 8.35, 9.4, 1.05, fc=GRAY_XL)
    note(ax, 0.42, 9.28, "Serving", fs=7.2, ha="left")
    box(ax, 0.75, 8.52, 2.5, 0.62, "OpenAI-compatible\nAPI gateway", fc=WHITE, fs=7.8)
    box(ax, 3.75, 8.52, 2.5, 0.62, "Continuous-batching\ndecode engine", fc=WHITE, fs=7.8)
    box(ax, 6.75, 8.52, 2.5, 0.62, "Session manager\n(residency, lifecycle)", fc=WHITE, fs=7.8)
    arrow(ax, (3.25, 8.83), (3.75, 8.83)); arrow(ax, (6.25, 8.83), (6.75, 8.83))

    box(ax, 1.6, 7.35, 6.8, 0.62,
        "MLXDKVWrapper  —  generate(): chunked prefill + decode loop",
        fc=BLUE, tc=WHITE, fs=8.6, bold=True)
    arrow(ax, (5.0, 8.35), (5.0, 8.0))

    box(ax, 1.6, 6.42, 6.8, 0.56,
        "MLXQwenModel  —  native KV cache (prefill only) + prefill→decode release",
        fc=WHITE, fs=8.2)
    arrow(ax, (5.0, 7.35), (5.0, 7.01))

    box(ax, 3.3, 5.42, 3.4, 0.62, "Patched Qwen2 attention", fc=BLUE_XL, fs=8.6, bold=True)
    arrow(ax, (5.0, 6.42), (5.0, 6.07))

    # branch labels
    arrow(ax, (3.9, 5.42), (2.7, 4.85), rad=-0.15)
    arrow(ax, (6.1, 5.42), (7.3, 4.85), rad=0.15)
    note(ax, 2.35, 5.22, "prefill  ($L>1$)", fs=7.6)
    note(ax, 7.65, 5.22, "decode  ($L=1$)", fs=7.6)

    # prefill column
    band(ax, 0.45, 2.6, 4.25, 2.2, fc="#F7F8FA")
    note(ax, 0.7, 4.62, "Prefill path", fs=7.4, ha="left")
    box(ax, 0.7, 3.75, 3.75, 0.68, "block-sparse causal attention\nover the native cache", fc=WHITE, fs=7.8)
    box(ax, 0.7, 2.78, 3.75, 0.68, "capture K/V  →  stream-compress\nblocks leaving the window", fc=WHITE, fs=7.8)
    arrow(ax, (2.58, 3.75), (2.58, 3.46))

    # decode column
    band(ax, 5.3, 2.6, 4.25, 2.2, fc="#F7F8FA")
    note(ax, 5.55, 4.62, "Decode path", fs=7.4, ha="left")
    box(ax, 5.55, 3.75, 3.75, 0.68, "ingest token into dense window\n(+ flush eligible block)", fc=WHITE, fs=7.8)
    box(ax, 5.55, 2.78, 3.75, 0.68, "route top-$K$  →  fused low-rank +\nexact-residual + recency attention", fc=BLUE_XL, fs=7.8)
    arrow(ax, (7.42, 3.75), (7.42, 3.46))

    # store
    box(ax, 0.55, 0.5, 8.9, 1.4,
        "MLXKVBlockManager — per-session KV store  (×28 layers)\n"
        "dense recency window  [$H_{kv}$, 1,280, $d$]  fp16   (newest 1,024 tokens)\n"
        "+  compressed pool  {$U$, $V_K$, $V_V$, anchors, residuals, min/max}  × 256 blocks",
        fc=GRAY_XL, fs=7.5)
    arrow(ax, (2.58, 2.78), (3.4, 1.92), rad=-0.1)
    arrow(ax, (7.42, 2.78), (6.6, 1.92), rad=0.1)
    note(ax, 2.35, 2.32, "write", fs=6.8, ha="right")
    note(ax, 7.65, 2.32, "read (no decompress)", fs=6.8, ha="left")

    style.finalize(fig, os.path.join(FIG, "f1_architecture.png"))


# ── F2 · compression pipeline (multi-signal residual selection) ─────────────
def f2_compression():
    fig, ax = plt.subplots(figsize=(8.6, 6.31)); clean(ax, 10, 10.45)

    # ── stage 1: block → deltas → normalize → SVD ──
    y = 9.00
    bx = 0.45
    for i in range(8):
        c = BLUE_L if i == 0 else (EMER_L if i in (5, 6) else BLUE_XL)
        ax.add_patch(Rectangle((bx + i * 0.33, y), 0.29, 0.62, fc=c, ec=BLACK,
                               lw=0.8, zorder=3))
    note(ax, bx + 0.145, y - 0.22, "anchor\n(token 0)", fs=6.6, va="top", style_="normal")
    note(ax, bx + 1.32, y + 0.94, "$K,V$ block  [$H_{kv}$, 256, $d$]", fs=7.8, style_="normal")
    note(ax, bx + 2.15, y - 0.22, "outlier tokens", fs=6.6, va="top")

    arrow(ax, (3.2, y + 0.31), (3.62, y + 0.31), lw=1.1)
    box(ax, 3.66, y - 0.06, 1.5, 0.74, "$\\Delta K = K - a_k$\n$\\Delta V = V - a_v$", fc=WHITE, fs=7.8)
    arrow(ax, (5.16, y + 0.31), (5.55, y + 0.31), lw=1.1)
    box(ax, 5.6, y - 0.06, 1.75, 0.74, "rescale $V$ to $K$ RMS,\nrow-normalize, concat", fc=WHITE, fs=6.8)
    arrow(ax, (7.35, y + 0.31), (7.74, y + 0.31), lw=1.1)
    box(ax, 7.82, y - 0.06, 1.7, 0.74, "randomized\ntruncated SVD\n(rank 32, seeded)",
        fc=BLUE, tc=WHITE, fs=6.8, bold=True)

    # ── stage 2: low-rank outputs on an orthogonal bus ──
    yo, y_bus = 6.60, 7.68
    ax.plot([8.67, 8.67], [y - 0.06, y_bus], color=BLACK, lw=0.9, zorder=2)
    note(ax, 8.90, 8.24, "$U\\,\\Sigma\\,V^{\\top}$", fs=7.6, ha="left", style_="normal")

    outs = [("$U$  [255, 32]", "coefficients"),
            ("$V_K$  [2, 32, 128]", "key basis"),
            ("$V_V$  [2, 32, 128]", "value basis"),
            ("$a_k, a_v$", "anchors (exact)"),
            ("$k_{\\min}, k_{\\max}$", "router stats")]
    ox, cw = 0.45, 1.72
    cx_first = ox + cw / 2
    cx_last = ox + 4 * (cw + 0.12) + cw / 2
    ax.plot([cx_first, cx_last], [y_bus, y_bus], color=BLACK, lw=0.9, zorder=2)
    for i, (t, sub) in enumerate(outs):
        cx_i = ox + i * (cw + 0.12) + cw / 2
        box(ax, ox + i * (cw + 0.12), yo, cw, 0.78, f"{t}\n{sub}", fc=BLUE_XL, fs=7.0)
        arrow(ax, (cx_i, y_bus), (cx_i, yo + 0.78), lw=0.9, shrinkA=0, shrinkB=0)

    # ── stage 3: base error signal ──
    ye = 5.10
    box(ax, 1.15, ye, 7.7, 0.78,
        "base signal — per-token joint reconstruction error\n"
        "$e_i = \\| \\Delta K_i - \\widehat{\\Delta K}_i \\|^2 "
        "+ g^2 \\| \\Delta V_i - \\widehat{\\Delta V}_i \\|^2$",
        fc=WHITE, ec=EMERALD, fs=7.6)
    arrow(ax, (1.31, yo), (1.31, ye + 0.80), color=EMERALD, lw=1.1, ls=(0, (4, 3)))
    note(ax, 1.52, (yo + ye + 0.78) / 2, "reconstruct deltas from $U, V$;\nmeasure per-token error",
         fs=6.6, ha="left")

    # ── stage 4: the three IDF-weighted boost signals ──
    yb = 2.72
    band(ax, 0.45, yb, 9.1, 2.05, fc="#F5FBF8", ec=EMER_L)
    note(ax, 0.72, yb + 1.83, "IDF-weighted boost signals — inflate priority before the cut  (all default on)",
         fs=7.4, ha="left", style_="normal")
    cw2, gp = 2.75, 0.26
    chips = [
        ("owner-capture",
         "walk $\\leq 12$ tokens left of a\nhigh-error value; boost the\nnearest capitalized word\n"
         "→ keeps entity $\\mathit{names}$"),
        ("edge-capture",
         "cosine-collision test flags a\nconnective whose low-rank row\ncollides with a neighbour's key\n"
         "→ keeps $\\mathit{relations}$"),
        ("coverage bonus",
         "uniform bonus across block\npositions, so one hot segment\ncannot take all $R$ slots\n"
         "→ keeps $\\mathit{spread}$"),
    ]
    for i, (t, b) in enumerate(chips):
        chip(ax, 0.70 + i * (cw2 + gp), yb + 0.16, cw2, 1.50, t, b, fc=WHITE, ec=EMERALD)
    arrow(ax, (5.0, ye), (5.0, yb + 2.05), color=EMERALD, lw=1.1)

    # ── stage 5: cut → exact residuals ──
    yr = 1.30
    box(ax, 0.90, yr, 3.90, 0.80,
        "priority rank  →  keep the top-$R$ rows\n(default $R{=}128$)",
        fc=WHITE, ec=EMERALD, fs=7.6)
    arrow(ax, (2.85, yb), (2.85, yr + 0.82), color=EMERALD, lw=1.1)
    arrow(ax, (4.80, yr + 0.40), (5.32, yr + 0.40), color=EMERALD, lw=1.1)
    box(ax, 5.40, yr, 4.10, 0.80,
        "exact residuals   $R_K, R_V$  [$R$, $H_{kv}$, $d$]  fp16",
        fc=EMER_L, ec=EMERALD, fs=7.8, bold=True)
    note(ax, 7.45, yr - 0.22, "attended exactly at decode — verbatim recall", fs=6.8, va="top")

    note(ax, 5.0, 0.28,
         "stored block ≈ 178 KiB  (50 KiB low-rank + 128 KiB residuals)  vs  256 KiB dense  →  1.44× "
         "smaller   ·   $R{=}64$ preset → 114 KiB, 2.25×",
         fs=7.2, style_="normal")

    style.finalize(fig, os.path.join(FIG, "f2_compression.png"))


# ── F4 · cache lifecycle ─────────────────────────────────────────────────────
def f4_lifecycle():
    fig, ax = plt.subplots(figsize=(8.6, 4.9)); clean(ax, 10, 8.2)

    # PREFILL band
    band(ax, 0.3, 5.5, 9.4, 2.35, fc="#F7F8FA")
    note(ax, 0.55, 7.6, "Prefill  (chunk = 512 tokens, repeated)", fs=7.6, ha="left", style_="normal")
    box(ax, 0.6, 6.35, 2.0, 0.85, "chunk forward\n(block-sparse\ncausal SDPA)", fc=WHITE, fs=7.4)
    box(ax, 3.05, 6.35, 2.0, 0.85, "capture chunk\n$K,V$ → dense\nbuffer", fc=WHITE, fs=7.4)
    box(ax, 5.5, 6.35, 2.2, 0.85, "window overflow?\nflush + compress\noldest block", fc=BLUE_XL, fs=7.4)
    box(ax, 8.15, 6.35, 1.3, 0.85, "next\nchunk", fc=WHITE, ec=GRAY_D, tc=GRAY_D, fs=7.4)
    arrow(ax, (2.6, 6.78), (3.05, 6.78)); arrow(ax, (5.05, 6.78), (5.5, 6.78))
    arrow(ax, (7.7, 6.78), (8.15, 6.78))
    # orthogonal dashed return path, routed UNDER the row (no box crossings)
    ax.plot([8.8, 8.8, 1.6, 1.6], [6.35, 5.98, 5.98, 6.15], color=GRAY_D,
            lw=0.9, ls=(0, (2, 2)), zorder=2)
    arrow(ax, (1.6, 6.15), (1.6, 6.33), color=GRAY_D, lw=0.9, ls=(0, (2, 2)))
    note(ax, 5.0, 5.74, "loop until the prompt is consumed", fs=6.8)

    # boundary
    box(ax, 1.05, 4.28, 7.9, 0.82,
        "prefill → decode boundary:   mx.eval  ·  mx.clear_cache  ·  resolve decode policy\n"
        "compressed decode → drop the native prefill cache (footprint = DKV store only)",
        fc=BLUE, tc=WHITE, fs=7.2, bold=True)
    arrow(ax, (5.0, 5.5), (5.0, 5.12), lw=1.2)

    # DECODE band
    band(ax, 0.3, 0.7, 9.4, 3.1, fc="#F7F8FA")
    note(ax, 0.55, 3.52, "Decode  (per token, per layer)", fs=7.6, ha="left", style_="normal")
    arrow(ax, (5.0, 4.28), (5.0, 3.82), lw=1.2)
    box(ax, 0.6, 1.95, 2.0, 1.0, "ingest new token\n$K,V$ → dense\nwindow (+ self)", fc=WHITE, fs=7.4)
    box(ax, 3.05, 1.95, 2.0, 1.0, "flush eligible\nblock if window\n> $W{+}B$", fc=WHITE, fs=7.4)
    box(ax, 5.5, 1.95, 2.2, 1.0, "route top-$K$ →\nfused sparse +\nexact attention", fc=BLUE_XL, fs=7.4, bold=True)
    box(ax, 8.15, 1.95, 1.3, 1.0, "logits →\nsample", fc=WHITE, fs=7.4)
    arrow(ax, (2.6, 2.45), (3.05, 2.45)); arrow(ax, (5.05, 2.45), (5.5, 2.45))
    arrow(ax, (7.7, 2.45), (8.15, 2.45))
    ax.plot([8.8, 8.8, 1.6, 1.6], [1.95, 1.58, 1.58, 1.75], color=GRAY_D,
            lw=0.9, ls=(0, (2, 2)), zorder=2)
    arrow(ax, (1.6, 1.75), (1.6, 1.93), color=GRAY_D, lw=0.9, ls=(0, (2, 2)))
    note(ax, 5.0, 1.32, "loop for each generated token", fs=6.8)

    style.finalize(fig, os.path.join(FIG, "f4_lifecycle.png"))


# ── F5a · routed sparse decode attention (semantics) ────────────────────────
def f5a_decode_semantics():
    fig, ax = plt.subplots(figsize=(8.6, 5.05)); clean(ax, 10, 8.70)

    box(ax, 4.15, 7.95, 1.7, 0.55, "query $q$  [$H$, $d$]", fc=BLUE_D, tc=WHITE, fs=8.2, bold=True)
    arrow(ax, (5.0, 7.95), (5.0, 7.56), lw=1.2)

    box(ax, 2.00, 6.82, 6.0, 0.68,
        "router:  $\\rho_b = \\max(\\,q{\\cdot}a_k,\\ \\max_j\\, q{\\cdot}R_{K,j})\\cdot$scale"
        "   →   keep top-$K{=}16$ blocks", fc=GRAY_XL, fs=7.8)
    note(ax, 8.28, 7.16, "exact $q{\\cdot}k$ over anchor\n+ residual keys", fs=6.6, ha="left")
    arrow(ax, (3.1, 6.82), (2.60, 6.22), rad=-0.12, lw=1.1)
    arrow(ax, (6.9, 6.82), (7.40, 6.22), rad=0.12, lw=1.1)

    # low-rank branch (blue)
    band(ax, 0.35, 1.62, 4.55, 4.52, fc="#F6F9FE", ec=BLUE_L)
    note(ax, 0.60, 5.88, "Low-rank branch  (selected blocks)", fs=7.6, ha="left", style_="normal")
    lrb = [("anchor score   $s_{anc} = (q\\cdot a_k)\\cdot$scale", WHITE, BLACK, False),
           ("project query   $\\tilde q = (q\\,V_K^{\\top})\\cdot$scale $\\in \\mathbb{R}^{32}$", WHITE, BLACK, False),
           ("delta scores   $\\delta s = s\\,(\\tilde q\\,U^{\\top}) + s_{anc}$", BLUE, WHITE, True),
           ("per-block softmax  →  $w$, lse$_{sp}$", WHITE, BLACK, False),
           ("$O_{sp} = \\Sigma w\\,a_v + (\\Sigma w\\,U)\\,s\\,V_V$", WHITE, BLACK, False)]
    ys = [5.00, 4.22, 3.44, 2.66, 1.88]
    for (t, fc, tc, bd), y0 in zip(lrb, ys):
        box(ax, 0.62, y0, 4.00, 0.56, t, fc=fc, tc=tc, fs=7.6, bold=bd)
    for y0 in ys[:-1]:
        arrow(ax, (2.62, y0), (2.62, y0 - 0.22), lw=0.9)
    note(ax, 0.60, 1.34, "$O(K\\,r\\,B)$ — never forms $\\hat K$", fs=6.9, ha="left")

    # exact branch (emerald)
    band(ax, 5.10, 1.62, 4.55, 4.52, fc="#F5FBF8", ec=EMER_L)
    note(ax, 5.35, 5.88, "Exact branch  (residuals + recency)", fs=7.6, ha="left", style_="normal")
    box(ax, 5.38, 4.76, 4.00, 0.80,
        "concat selected blocks' exact residuals\n$R_K, R_V$  ⊕  dense recency window",
        fc=WHITE, ec=EMERALD, fs=7.6)
    box(ax, 5.38, 3.52, 4.00, 0.80,
        "exact attention over the\naugmented $\\tilde K, \\tilde V$  (full fp16)",
        fc=EMERALD, tc=WHITE, fs=7.6, bold=True)
    box(ax, 5.38, 2.66, 4.00, 0.56, "softmax  →  $O_{dn}$, lse$_{dn}$", fc=WHITE, fs=7.6)
    arrow(ax, (7.38, 4.76), (7.38, 4.34), lw=0.9)
    arrow(ax, (7.38, 3.52), (7.38, 3.24), lw=0.9)
    note(ax, 9.42, 2.12, "verbatim recall of\nneedle tokens", fs=6.9, ha="right")

    box(ax, 2.85, 0.50, 4.30, 0.94,
        "flash-style LSE merge\n$m = \\max($lse$_{sp}$, lse$_{dn})$;   "
        "$o = (e^{\\cdot}O_{sp} + e^{\\cdot}O_{dn})\\,/\\,\\Sigma$",
        fc=BLUE_D, tc=WHITE, fs=7.4, bold=True)
    arrow(ax, (2.62, 1.88), (3.30, 1.44), color=BLUE, rad=-0.15, lw=1.3)
    arrow(ax, (7.38, 2.66), (6.70, 1.44), color=EMERALD, rad=0.15, lw=1.3)
    note(ax, 5.0, 0.26, "NaN/inf-guarded: an empty compressed set contributes exactly zero weight", fs=6.7)

    style.finalize(fig, os.path.join(FIG, "f5a_decode_attention.png"))


# ── F5b · fused decode buffer (runtime path) ────────────────────────────────
def f5b_fused_buffer():
    # Wide/short figure: the two explanatory notes live in the LaTeX caption
    # rather than in-figure, so the remaining type survives the down-scale to
    # \linewidth at a readable size.
    fig, ax = plt.subplots(figsize=(7.6, 2.32)); clean(ax, 10, 4.40)

    note(ax, 9.85, 4.20, "persistent — re-written every $N{=}16$ tokens", fs=7.4, ha="right")

    # geometry of the buffer columns (sources sit directly above their own run)
    x0, x1 = 0.45, 9.05
    NSEG = 15
    sw = (x1 - x0) / NSEG
    LR, RES, WIN, NEW = range(0, 6), range(6, 9), range(9, 13), 13
    MASKED = (1, 4)                       # low-rank twins of residual rows
    cx_lr = x0 + 3.0 * sw
    cx_res = x0 + 7.5 * sw
    cx_win = x0 + 11.0 * sw
    cx_new = x0 + (NEW + 0.5) * sw

    # sources — filled to match the strip run they feed
    ysrc, hsrc = 3.10, 0.78
    box(ax, cx_lr - 1.65, ysrc, 3.30, hsrc,
        "selected blocks materialised\n$\\hat K = a_k + s\\,U V_K$,   $\\hat V = a_v + s\\,U V_V$",
        fc=BLUE_XL, ec=BLUE_D, fs=7.4)
    box(ax, cx_res - 0.80, ysrc, 1.60, hsrc,
        "their exact\nresiduals $R_K, R_V$", fc=EMER_L, ec=EMERALD, fs=7.4)
    box(ax, cx_win - 1.06, ysrc, 2.12, hsrc,
        "dense recency window\n1,024 tokens, fp16", fc=BLUE_L, ec=BLUE_D, fs=7.4)

    # the persistent buffer strip — the fixed additive mask is shown in-place
    ystrip, hstrip = 1.78, 0.82
    for i in range(NSEG):
        blocked = i in MASKED or i > NEW
        c = (GRAY if blocked else
             BLUE_XL if i in LR else EMER_L if i in RES else
             BLUE_L if i in WIN else WHITE)
        ax.add_patch(Rectangle((x0 + i * sw, ystrip), sw, hstrip, fc=c,
                               ec=BLACK, lw=0.8,
                               ls=(0, (2, 2)) if i == NEW else "-", zorder=3))
        if blocked:
            note(ax, x0 + (i + 0.5) * sw, ystrip + hstrip / 2, "$-\\infty$", fs=6.8, style_="normal")
    note(ax, cx_new, ystrip + hstrip / 2, "+1", fs=7.2, style_="normal")
    for cx in (cx_lr, cx_res, cx_win):
        arrow(ax, (cx, ysrc), (cx, ystrip + hstrip), lw=0.95, shrinkA=0, shrinkB=0)

    # per-token fast path
    ypt, hpt = 0.32, 0.78
    box(ax, 0.55, ypt, 3.85, hpt, "one scaled_dot_product_attention\nover exact-length views",
        fc=BLUE_D, tc=WHITE, fs=7.2, bold=True)
    box(ax, 4.65, ypt, 1.30, hpt, "attention\noutput $o$", fc=WHITE, fs=7.4)
    box(ax, 6.20, ypt, 3.10, hpt, "each step: append the new token's\n$K,V$ as one in-place row",
        fc=WHITE, fs=7.0)
    # read arrow drops from an unmasked column
    arrow(ax, (x0 + 3.5 * sw, ystrip), (x0 + 3.5 * sw, ypt + hpt), lw=1.0)
    arrow(ax, (4.40, ypt + hpt / 2), (4.65, ypt + hpt / 2), lw=1.0)
    # append routes around the right edge, so it crosses nothing
    ax.plot([9.30, 9.62, 9.62, cx_new], [ypt + hpt / 2, ypt + hpt / 2, 2.90, 2.90],
            color=BLACK, lw=1.0, zorder=2)
    arrow(ax, (cx_new, 2.90), (cx_new, ystrip + hstrip), lw=1.0, shrinkA=0, shrinkB=0)

    style.finalize(fig, os.path.join(FIG, "f5b_fused_buffer.png"))


# ── F6 · 3D memory architecture ──────────────────────────────────────────────
def f6_memory_3d():
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch, FancyArrowPatch
    from matplotlib.lines import Line2D

    # 3D Arrow helper class for matplotlib 3D sorting compatibility
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0,0), (0,0), *args, **kwargs)
            self._adds = (xs, ys, zs)

        def do_3d_projection(self, renderer=None):
            xs, ys, zs = self._adds
            xs_2d, ys_2d, zs_2d = proj3d.proj_transform(xs, ys, zs, self.axes.get_proj())
            self.set_positions((xs_2d[0], ys_2d[0]), (xs_2d[1], ys_2d[1]))
            return np.min(zs_2d)

    fig = plt.figure(figsize=(9.2, 5.8))
    ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    ax.set_proj_type("ortho")

    def cuboid(o, size, color, alpha=1.0, ec=BLACK, lw=0.5, z=1):
        x, y, zc = o; dx, dy, dz = size
        pts = np.array([
            [x,    y,    zc],   [x+dx, y,    zc],
            [x+dx, y+dy, zc],   [x,    y+dy, zc],
            [x,    y,    zc+dz],[x+dx, y,    zc+dz],
            [x+dx, y+dy, zc+dz],[x,    y+dy, zc+dz],
        ])
        faces = [[pts[j] for j in f] for f in (
            [0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4])]
        pc = Poly3DCollection(faces, facecolors=color, edgecolors=ec,
                              linewidths=lw, alpha=alpha)
        pc.set_zsort("max"); pc.set_zorder(z)
        ax.add_collection3d(pc)

    # ── 1. Vertical Layer Stack (Left side) ──
    n_layers = 5
    slab_w, slab_d, slab_h = 2.4, 2.2, 0.22
    slab_x, slab_y = 0.5, 0.8
    z_coords = [0.0, 1.1, 2.2, 3.3, 4.4]

    for i, z_val in enumerate(z_coords):
        color = BLUE_XL if i < n_layers - 1 else BLUE_L
        cuboid((slab_x, slab_y, z_val), (slab_w, slab_d, slab_h),
               color, alpha=0.9, ec=GRAY_D, lw=0.6, z=10+i)

    # ── 2. Detailed single-layer view (Right side) ──
    x0 = 5.2
    z_lr = 1.8; z_res = 3.6
    bw = 0.65; gap = 0.25; bdepth = 2.2
    n_used = 9; n_ghost = 3
    pool_y = 0.8

    for b in range(n_used + n_ghost):
        x = x0 + b*(bw+gap); used = b < n_used
        al = 0.95 if used else 0.16
        ag = 0.82 if used else 0.12
        ec = BLACK if used else GRAY_D
        lw = 0.65 if used else 0.35
        cuboid((x, pool_y, 0.0),  (bw, bdepth, z_lr),  BLUE,
               alpha=al, ec=ec, lw=lw, z=30+b)
        cuboid((x, pool_y, z_lr), (bw, bdepth, z_res), EMERALD,
               alpha=ag, ec=ec, lw=lw, z=30+b)

    # Dense recency window slab (front, segmented array)
    win_y = -2.2
    win_w = (bw+gap)*7.0
    win_h = 1.0
    cuboid((x0, win_y, 0.0), (win_w, 1.8, win_h), BLUE_L, alpha=0.92, ec=BLACK, lw=0.75, z=80)
    
    # Draw segments on the recency window to show it is a token buffer (array of blocks)
    for seg_idx in range(1, 7):
        xs = x0 + seg_idx * (win_w / 7.0)
        ax.plot([xs, xs], [win_y, win_y], [0.0, win_h], color=BLACK, lw=0.5, zorder=85)
        ax.plot([xs, xs], [win_y, win_y + 1.8], [win_h, win_h], color=BLACK, lw=0.5, zorder=85)

    # ── 3. Flush & zoom connector lines (entirely solid) ──
    ax.plot([5.12, 2.6], [-1.3, -1.3], [0.9, 0.9], color=BLACK, lw=1.1, zorder=90)
    ax.plot([2.6,  2.6], [-1.3,  0.8], [0.9, 0.9], color=BLACK, lw=1.1, zorder=90)
    ax.plot([2.6,  2.9], [ 0.8,  0.8], [0.9, 0.9], color=BLACK, lw=1.1, zorder=90)
    ax.plot([2.9,  4.7], [ 0.8,  0.8], [0.9, 0.9], color=BLACK, lw=1.1, zorder=90)







    # Zoom-in connector lines: bound the expanded detail view from the layer stack.
    # Both lines use the FRONT face (y = slab_y = pool_y = 0.8) so they stay on the
    # viewer-facing side and are not occluded by the block geometry.
    #   Top line:    top-front-right of TOP layer    → top-front-left of CBP
    #   Bottom line: bottom-front-right of BOT layer → top-front-left of DRW
    #                (DRW is the actual bottom of the expanded view)
    stack_top_r = (slab_x + slab_w, slab_y, z_coords[-1] + slab_h)   # (2.9, 0.8, 4.62)
    stack_bot_r = (slab_x + slab_w, slab_y, z_coords[0])              # (2.9, 0.8, 0.0)
    cbp_top_l   = (x0,              pool_y,  z_lr + z_res)             # (5.2, 0.8, 5.4)
    drw_top_l   = (x0,              win_y,   win_h)                    # (5.2,-2.2, 1.0)

    ax.plot([stack_top_r[0], cbp_top_l[0]], [stack_top_r[1], cbp_top_l[1]], [stack_top_r[2], cbp_top_l[2]],
            color=GRAY_D, lw=1.0, ls="--", zorder=6)
    ax.plot([stack_bot_r[0], drw_top_l[0]], [stack_bot_r[1], drw_top_l[1]], [stack_bot_r[2], drw_top_l[2]],
            color=GRAY_D, lw=1.0, ls="--", zorder=6)

    # ── limits & camera ──
    x_max = x0 + (n_used+n_ghost)*(bw+gap) + 0.5
    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(-2.8, 4.0)
    ax.set_zlim(-0.2, z_lr+z_res+0.8)
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()
    ax.set_box_aspect((13.5, 9.0, 5.0), zoom=1.16)


    # Helper to project 3D point to 2D figure coordinates (0.0 to 1.0)
    def project_3d_to_fig(x, y, z):
        x_p, y_p, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        disp_coord = ax.transData.transform((x_p, y_p))
        fig_coord = fig.transFigure.inverted().transform(disp_coord)
        return fig_coord

    # Helper to project 3D point to screen pixels
    def project_3d_to_pixels(x, y, z):
        x_p, y_p, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        return ax.transData.transform((x_p, y_p))

    # Draw canvas to get projection matrices
    fig.canvas.draw()

    # Get target coordinates for arrows dynamically
    target_recency   = project_3d_to_fig(x0 + win_w/2, win_y + 0.9, win_h/2)
    target_overflow  = project_3d_to_fig(x0, win_y, 0.0) # bottom-left corner of the window block

    # Helper for 2D figure-level arrows
    def draw_arrow(p0, p1, style="->", color=BLACK, lw=0.85):
        a = FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle=style,
                            mutation_scale=8, color=color, lw=lw, zorder=99)
        fig.add_artist(a)

    # ── 2D labels ──

    # Project coordinates for CBP brackets
    p_used_left = project_3d_to_fig(x0, pool_y + bdepth, z_lr + z_res)
    p_used_right = project_3d_to_fig(x0 + n_used*(bw+gap) - gap, pool_y + bdepth, z_lr + z_res)
    p_free_left = project_3d_to_fig(x0 + n_used*(bw+gap), pool_y + bdepth, z_lr + z_res)
    p_free_right = project_3d_to_fig(x0 + (n_used+n_ghost)*(bw+gap) - gap, pool_y + bdepth, z_lr + z_res)

    # Arrowhead for the DRW→CBP connector.
    # The 3D line was stopped at (4.7, 0.8, 0.9); we project that point as the arrow tail
    # and project the true target (5.10, 0.8, 0.9) as the arrow tip, so the triangle
    # perfectly caps the line with the correct perspective-accurate direction.
    p_line_end  = project_3d_to_fig(4.7,  0.8, 0.9)   # where the 3D line was stopped
    p_arrow_tip = project_3d_to_fig(5.10, 0.8, 0.9)   # true CBP left-face target
    arrow2d = FancyArrowPatch(p_line_end, p_arrow_tip, transform=fig.transFigure, arrowstyle="-|>",
                              mutation_scale=8, color=BLACK, lw=1.1, zorder=99)
    fig.add_artist(arrow2d)





    
    dy = 0.024
    dt = 0.012

    # Calculate angles of the slanted brackets directly in display pixels
    px_used_left = project_3d_to_pixels(x0, pool_y + bdepth, z_lr + z_res)
    px_used_right = project_3d_to_pixels(x0 + n_used*(bw+gap) - gap, pool_y + bdepth, z_lr + z_res)
    px_free_left = project_3d_to_pixels(x0 + n_used*(bw+gap), pool_y + bdepth, z_lr + z_res)
    px_free_right = project_3d_to_pixels(x0 + (n_used+n_ghost)*(bw+gap) - gap, pool_y + bdepth, z_lr + z_res)

    angle_used = math.degrees(math.atan2(px_used_right[1] - px_used_left[1], px_used_right[0] - px_used_left[0]))
    angle_free = math.degrees(math.atan2(px_free_right[1] - px_free_left[1], px_free_right[0] - px_free_left[0]))

    # Pool title (centered dynamically above the pool and rotated parallel to brackets)
    fig.text((p_used_left[0] + p_free_right[0])/2, (p_used_left[1] + p_free_right[1])/2 + dy + 0.038, "compressed block pool — 256 slots (bounded)",
             ha="center", va="bottom", fontsize=8.8, color=BLACK,
             rotation=angle_used, rotation_mode="anchor")


    # used blocks bracket (slanted)
    fig.add_artist(Line2D([p_used_left[0], p_used_right[0]], [p_used_left[1] + dy, p_used_right[1] + dy], transform=fig.transFigure, color=BLACK, lw=0.85))
    fig.add_artist(Line2D([p_used_left[0], p_used_left[0]], [p_used_left[1] + dy - dt, p_used_left[1] + dy], transform=fig.transFigure, color=BLACK, lw=0.85))
    fig.add_artist(Line2D([p_used_right[0], p_used_right[0]], [p_used_right[1] + dy - dt, p_used_right[1] + dy], transform=fig.transFigure, color=BLACK, lw=0.85))
    fig.text((p_used_left[0] + p_used_right[0])/2, (p_used_left[1] + p_used_right[1])/2 + dy + 0.008, "used", 
             ha="center", va="bottom", fontsize=7.3, color=BLACK, rotation=angle_used, rotation_mode="anchor")

    # free (zeroed) bracket (slanted)
    fig.add_artist(Line2D([p_free_left[0], p_free_right[0]], [p_free_left[1] + dy, p_free_right[1] + dy], transform=fig.transFigure, color=GRAY_D, lw=0.85))
    fig.add_artist(Line2D([p_free_left[0], p_free_left[0]], [p_free_left[1] + dy - dt, p_free_left[1] + dy], transform=fig.transFigure, color=GRAY_D, lw=0.85))
    fig.add_artist(Line2D([p_free_right[0], p_free_right[0]], [p_free_right[1] + dy - dt, p_free_right[1] + dy], transform=fig.transFigure, color=GRAY_D, lw=0.85))
    fig.text((p_free_left[0] + p_free_right[0])/2, (p_free_left[1] + p_free_right[1])/2 + dy + 0.008, "free (zeroed)", 
             ha="center", va="bottom", fontsize=7.3, color=GRAY_D, rotation=angle_free, rotation_mode="anchor")

    # Layers stack bracket on the left
    target_stack_top = project_3d_to_fig(slab_x, slab_y + slab_d/2, z_coords[-1] + slab_h)
    target_stack_bot = project_3d_to_fig(slab_x, slab_y + slab_d/2, z_coords[0])
    bx_x = target_stack_bot[0] - 0.075

    fig.add_artist(Line2D([bx_x, bx_x], [target_stack_bot[1], target_stack_top[1]], transform=fig.transFigure, color=BLACK, lw=0.85))
    fig.add_artist(Line2D([bx_x, bx_x + 0.015], [target_stack_top[1], target_stack_top[1]], transform=fig.transFigure, color=BLACK, lw=0.85))
    fig.add_artist(Line2D([bx_x, bx_x + 0.015], [target_stack_bot[1], target_stack_bot[1]], transform=fig.transFigure, color=BLACK, lw=0.85))
    
    fig.text(bx_x - 0.015, (target_stack_bot[1] + target_stack_top[1])/2, "layers 0 to 27\n(replicated stack)",
             ha="right", va="center", fontsize=8.0, color=BLACK, linespacing=1.2)

    # recency window label — below the slab; arrow ends at the bottom face of the DRW (z=0)
    # to avoid piercing through the slab.
    drw_bottom_center = project_3d_to_fig(x0 + win_w/2, win_y + 0.9, 0.0)
    label_y_top = 0.135   # top of the label block (arrow tail starts here)
    fig.text(target_recency[0], 0.065, "dense recency window — 1,024 exact fp16 tokens\n(acts as a sliding FIFO queue)",
             ha="center", va="bottom", fontsize=8.1, color=BLACK, linespacing=1.3)
    draw_arrow((target_recency[0], label_y_top), (drw_bottom_center[0], drw_bottom_center[1] - 0.005))

    # overflow label — centered above its arrow, rotated to match the arrow angle
    arrow_start_x = target_overflow[0] - 0.175
    arrow_start_y = target_overflow[1] + 0.055
    arrow_end_x   = target_overflow[0] - 0.012
    arrow_end_y   = target_overflow[1] + 0.005
    # Midpoint of the arrow (in figure coords) and angle for label rotation
    mid_x = (arrow_start_x + arrow_end_x) / 2
    mid_y = (arrow_start_y + arrow_end_y) / 2
    # Convert figure-coord deltas to display pixels for angle calculation
    px0 = fig.transFigure.transform((arrow_start_x, arrow_start_y))
    px1 = fig.transFigure.transform((arrow_end_x, arrow_end_y))
    arrow_angle = math.degrees(math.atan2(px1[1] - px0[1], px1[0] - px0[0]))
    fig.text(mid_x, mid_y + 0.012, "overflow \u2192 flush + compress",
             ha="center", va="bottom", fontsize=7.4, color=BLACK,
             rotation=arrow_angle, rotation_mode="anchor")
    draw_arrow((arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y))

    # Dummy spacer to force bbox_inches="tight" to add padding on the right edge of the cropped figure
    fig.text(p_free_right[0] + 0.20, p_free_right[1], " ", transform=fig.transFigure)


    # ── legend (upper-left, vertical) ──
    leg = [
        Patch(fc=BLUE,    ec=BLACK, label=r"low-rank core  $U,V_K,V_V$, anchors, min/max  (≈50 KiB)"),
        Patch(fc=EMERALD, ec=BLACK, label=r"exact residuals  $R_K,R_V$  (≈128 KiB,  $R{=}128$)"),
        Patch(fc=BLUE_L,  ec=BLACK, label="dense recency window (fp16)"),
    ]
    lg = fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.01, 0.995),
                    ncol=1, fontsize=7.3, frameon=True,
                    handlelength=1.4, borderpad=0.55, labelspacing=0.5)
    lg.get_frame().set_edgecolor(LEGEND_EC)
    lg.get_frame().set_linewidth(0.7)
    lg.get_frame().set_alpha(0.93)

    out = os.path.join(FIG, "f6_memory_3d")
    fig.savefig(out+".png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out+".pdf",          bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", out+".png")



if __name__ == "__main__":
    f1_architecture()
    f2_compression()
    f4_lifecycle()
    f5a_decode_semantics()
    f5b_fused_buffer()
    f6_memory_3d()
    print("diagrams done")
