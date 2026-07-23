#!/usr/bin/env python3
"""DeepSeek-V3.2 Style Architecture & Sequence Diagrams for Differential-KV.

Redesigned to mirror the visual elegance of DeepSeek-V3.2 Figures 2 & 4:
  - Vector/tensor pill arrays (q, k, v, c, u hidden states)
  - Operator badges (concatenate, RoPE, Top-k, LSE merge)
  - Precise color themes:
      * Blue (#1E40AF / #2563EB / #DBEAFE): Core Low-Rank compute & attention
      * Emerald (#047857 / #059669 / #A7F3D0): Exact residuals, recency & router
      * Neutral (#F8F9FA / #EDEFF2 / #111111): Base structures, hidden states
      * Rose (#FEF2F2 / #EF4444): Outliers / high-error flags
  - Clean vector routing & 300 DPI rendering into paper/figures_preview/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import (BLACK, BLUE, BLUE_D, BLUE_L, BLUE_XL, EMERALD, EMER_L,
                   GRAY, GRAY_XL, GRAY_D, LEGEND_EC, WHITE, OOM_RED)

style.apply_rc()
plt.rcParams.update({"axes.grid": False})

PREVIEW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures_preview")
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Color tokens for DeepSeek-V3.2 aesthetic
DS_BLUE_DARK = "#1E40AF"
DS_BLUE_MED  = "#2563EB"
DS_BLUE_LIGHT= "#DBEAFE"
DS_BLUE_BG   = "#EFF6FF"

DS_EMER_DARK = "#047857"
DS_EMER_MED  = "#059669"
DS_EMER_LIGHT= "#A7F3D0"
DS_EMER_BG   = "#ECFDF5"

DS_GRAY_BG   = "#F8F9FA"
DS_GRAY_BOX  = "#EDEFF2"
DS_GRAY_STROKE="#C4C9D0"

DS_ROSE_BG   = "#FEF2F2"
DS_ROSE_MED  = "#EF4444"


def clean(ax, xmax=10, ymax=10):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")


def box(ax, x, y, w, h, text="", fc=DS_GRAY_BOX, ec=BLACK, tc=BLACK, fs=8.2, lw=0.85,
        bold=False, rounding=0.03, style_="round", align="center", zorder=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"{style_},pad=0.005,rounding_size={rounding}"
                       if style_ == "round" else f"{style_},pad=0.005",
                       fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2 if align == "center" else x + 0.15,
                y + h / 2, text,
                ha=align, va="center", fontsize=fs, color=tc, zorder=zorder+1,
                fontweight="bold" if bold else "normal", linespacing=1.25)
    return (x, y, w, h)


def pill_array(ax, x, y, w, h, count=5, labels=None, fc=WHITE, ec=BLACK, tc=BLACK, fs=6.8, show_dots=True):
    p_w = (w - (count - 1) * 0.04) / count
    for i in range(count):
        px = x + i * (p_w + 0.04)
        if show_dots and i == count // 2:
            ax.text(px + p_w / 2, y + h / 2, "···", ha="center", va="center", fontsize=fs + 1, color=tc, zorder=5)
        else:
            p = FancyBboxPatch((px, y), p_w, h, boxstyle="round,pad=0.002,rounding_size=0.025",
                               fc=fc, ec=ec, lw=0.75, zorder=4)
            ax.add_patch(p)
            if labels and i < len(labels) and labels[i]:
                ax.text(px + p_w / 2, y + h / 2, labels[i], ha="center", va="center",
                        fontsize=fs, color=tc, zorder=5)


def badge(ax, cx, cy, text, fc=WHITE, ec=BLACK, tc=BLACK, fs=6.8, shape="round", pad_x=0.25, pad_y=0.14):
    tw = len(text) * 0.08 + pad_x
    th = pad_y * 2
    p = FancyBboxPatch((cx - tw / 2, cy - th / 2), tw, th,
                       boxstyle="round,pad=0.003,rounding_size=0.025",
                       fc=fc, ec=ec, lw=0.8, zorder=6)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=7)


def diamond(ax, cx, cy, hw, hh, text, fc=WHITE, ec=BLACK, tc=BLACK, fs=7.2, lw=0.85):
    p = Polygon([(cx - hw, cy), (cx, cy + hh), (cx + hw, cy), (cx, cy - hh)],
                closed=True, fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4, linespacing=1.25)


def arrow(ax, p0, p1, color=BLACK, lw=0.95, rad=0.0, ls="-", style_="-|>", z=3, **kwargs):
    a = FancyArrowPatch(p0, p1, arrowstyle=style_, mutation_scale=9, color=color,
                        lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=z,
                        linestyle=ls, **kwargs)
    ax.add_patch(a)


def band(ax, x, y, w, h, title="", fc=DS_GRAY_BG, ec=DS_GRAY_STROKE, lw=0.8, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.04",
                       fc=fc, ec=ec, lw=lw, zorder=1,
                       linestyle=(0, (4, 3)) if dashed else "-")
    ax.add_patch(p)
    if title:
        ax.text(x + 0.2, y + h - 0.18, title, ha="left", va="center",
                fontsize=7.8, color=BLACK, fontweight="bold", zorder=2)


# ── REDESIGNED F5a: ROUTED SPARSE DECODE ATTENTION (DeepSeek Fig 2 Twin) ────
def redraw_f5a_decode_attention():
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    clean(ax, 10, 10)

    # Top Query Input Box + Pill Array
    box(ax, 3.8, 8.8, 2.4, 0.75, "", fc=DS_BLUE_MED, ec=DS_BLUE_DARK, lw=1.0)
    ax.text(5.0, 9.25, r"Query $q \in \mathbb{R}^{H \times d}$", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=WHITE)
    pill_array(ax, 4.0, 8.88, 2.0, 0.22, count=5, fc=WHITE, ec=DS_BLUE_DARK, tc=DS_BLUE_DARK, fs=6.5)

    arrow(ax, (5.0, 8.8), (5.0, 8.25), lw=1.1)

    # Router Component Container
    box(ax, 1.8, 7.35, 6.4, 0.85, fc=DS_GRAY_BOX, ec=BLACK, lw=0.9)
    ax.text(5.0, 7.88, r"Sparse Router:  $\rho_b = \max(q \cdot a_k, \max_j q \cdot R_{K,j}) \cdot \text{scale}$",
            ha="center", va="center", fontsize=8.0, color=BLACK)
    badge(ax, 5.0, 7.52, r"keep top-$K = 16$ blocks", fc=DS_EMER_BG, ec=DS_EMER_MED, tc=DS_EMER_DARK, fs=7.2, pad_x=0.3)
    ax.text(8.35, 7.75, r"exact $q \cdot k$ over anchor" "\n+ residual keys", ha="left", va="center",
            fontsize=6.8, fontstyle="italic", color=BLACK)

    # Split Arrows
    arrow(ax, (3.2, 7.35), (2.6, 6.6), rad=-0.1, lw=1.1)
    arrow(ax, (6.8, 7.35), (7.4, 6.6), rad=0.1, lw=1.1)

    # Left Branch: Low-Rank Attention (Blue Theme)
    band(ax, 0.35, 1.35, 4.45, 5.25, title="Low-Rank Branch (Selected Blocks)", fc=DS_BLUE_BG, ec=DS_BLUE_MED)
    
    lr_steps = [
        (r"1. Anchor Score:  $s_{\text{anc}} = (q \cdot a_k) \cdot \text{scale}$", WHITE, BLACK, False),
        (r"2. Project Query:  $\tilde{q} = (q V_K^{\top}) \cdot \text{scale} \in \mathbb{R}^{32}$", WHITE, BLACK, False),
        (r"3. Delta Scores:  $\delta s = s(\tilde{q} U^{\top}) + s_{\text{anc}}$", DS_BLUE_MED, WHITE, True),
        (r"4. Per-Block Softmax  $\rightarrow  w, \text{lse}_{\text{sp}}$", WHITE, BLACK, False),
        (r"5. Output:  $O_{\text{sp}} = \sum w a_v + (\sum w U) s V_V$", WHITE, BLACK, False),
    ]
    ys = [5.2, 4.4, 3.6, 2.8, 2.0]
    for (txt, fc_, tc_, bd_), y0 in zip(lr_steps, ys):
        box(ax, 0.55, y0, 4.05, 0.60, txt, fc=fc_, tc=tc_, fs=7.5, bold=bd_, align="center")
    for i in range(len(ys) - 1):
        arrow(ax, (2.57, ys[i]), (2.57, ys[i+1] + 0.60), lw=0.9)

    ax.text(0.60, 1.55, r"$O(K \cdot r \cdot B)$ compute — never materializes dense $\hat{K}$",
            ha="left", va="center", fontsize=7.0, fontstyle="italic", color=BLACK)

    # Right Branch: Exact Branch (Emerald Theme)
    band(ax, 5.20, 1.35, 4.45, 5.25, title="Exact Branch (Residuals + Recency)", fc=DS_EMER_BG, ec=DS_EMER_MED)

    box(ax, 5.40, 4.95, 4.05, 0.85,
        "Concat selected blocks' exact residuals\n"
        r"$R_K, R_V \in \mathbb{R}^{R \times d}$  $\oplus$  Dense Recency Window",
        fc=WHITE, ec=DS_EMER_MED, fs=7.5)
    
    arrow(ax, (7.42, 4.95), (7.42, 4.2), lw=0.95)

    box(ax, 5.40, 3.35, 4.05, 0.85,
        "Exact Attention over augmented\n"
        r"tensors $\tilde{K}, \tilde{V}$  (full fp16 fidelity)",
        fc=DS_EMER_MED, tc=WHITE, fs=7.8, bold=True)
    
    arrow(ax, (7.42, 3.35), (7.42, 2.6), lw=0.95)

    box(ax, 5.40, 2.0, 4.05, 0.6, r"Softmax  $\rightarrow  O_{\text{dn}}, \text{lse}_{\text{dn}}$",
        fc=WHITE, ec=DS_EMER_MED, fs=7.6)

    ax.text(9.50, 1.55, "verbatim recall of\nneedle tokens", ha="right", va="center",
            fontsize=7.0, fontstyle="italic", color=BLACK)

    # Bottom Flash LSE Merge Engine (Royal Blue Container)
    box(ax, 2.7, 0.35, 4.6, 0.95, "", fc=DS_BLUE_DARK, ec=BLACK, lw=1.0)
    ax.text(5.0, 0.98, "Flash-Style LSE Merge Engine", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=WHITE)
    ax.text(5.0, 0.62,
            r"$m = \max(\text{lse}_{\text{sp}}, \text{lse}_{\text{dn}}), \quad o = \frac{e^{\text{lse}_{\text{sp}}-m} O_{\text{sp}} + e^{\text{lse}_{\text{dn}}-m} O_{\text{dn}}}{e^{\text{lse}_{\text{sp}}-m} + e^{\text{lse}_{\text{dn}}-m}}$",
            ha="center", va="center", fontsize=7.2, color=WHITE)

    arrow(ax, (2.57, 2.6), (3.35, 1.30), color=DS_BLUE_MED, rad=-0.15, lw=1.2)
    arrow(ax, (7.42, 2.4), (6.65, 1.30), color=DS_EMER_MED, rad=0.15, lw=1.2)

    badge(ax, 5.0, 0.12, "NaN / inf-guarded: empty compressed set contributes zero weight",
          fc=WHITE, ec=DS_GRAY_STROKE, tc=BLACK, fs=6.5, pad_x=0.2)

    style.finalize(fig, os.path.join(PREVIEW_DIR, "f5a_decode_attention.png"))


# ── REDESIGNED F2: COMPRESSION & RESIDUAL SELECTION PIPELINE ──────────────────
def redraw_f2_compression():
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    clean(ax, 10, 10.2)

    y1 = 9.0
    bx = 0.45
    for i in range(8):
        c = DS_BLUE_LIGHT if i == 0 else (DS_EMER_LIGHT if i in (5, 6) else DS_GRAY_BOX)
        ax.add_patch(Rectangle((bx + i * 0.32, y1), 0.28, 0.65, fc=c, ec=BLACK, lw=0.8, zorder=3))
        if i == 0:
            ax.text(bx + i * 0.32 + 0.14, y1 + 0.32, "$a_k$", ha="center", va="center", fontsize=6.8, color=BLACK)

    ax.text(bx + 0.14, y1 - 0.22, "Anchor\n(token 0)", ha="center", va="top", fontsize=6.6, color=BLACK)
    ax.text(bx + 1.28, y1 + 0.92, r"$K,V$ block  [$H_{\text{kv}}, 256, d$]", ha="center", va="center", fontsize=7.8, fontweight="bold")
    ax.text(bx + 1.8, y1 - 0.22, "Outlier / relation\ntokens", ha="center", va="top", fontsize=6.6, color=BLACK)

    arrow(ax, (3.15, y1 + 0.325), (3.6, y1 + 0.325), lw=1.1)
    box(ax, 3.65, y1 - 0.05, 1.5, 0.75, r"$\Delta K = K - a_k$" "\n" r"$\Delta V = V - a_v$", fc=WHITE, fs=7.8)
    
    arrow(ax, (5.15, y1 + 0.325), (5.55, y1 + 0.325), lw=1.1)
    box(ax, 5.6, y1 - 0.05, 1.8, 0.75, "Rescale $V$ to $K$ RMS,\nrow-normalize, concat", fc=WHITE, fs=7.0)
    
    arrow(ax, (7.4, y1 + 0.325), (7.8, y1 + 0.325), lw=1.1)
    box(ax, 7.85, y1 - 0.05, 1.75, 0.75, "Randomized\nTruncated SVD\n(rank 32, seeded)",
        fc=DS_BLUE_MED, tc=WHITE, fs=7.2, bold=True)

    y_bus = 7.7
    yo = 6.6
    ax.plot([8.72, 8.72], [y1 - 0.05, y_bus], color=BLACK, lw=0.95, zorder=2)
    ax.text(8.95, 8.25, r"$U \Sigma V^{\top}$", ha="left", va="center", fontsize=7.8)

    outs = [
        ("$U$ [255, 32]", "coefficients"),
        (r"$V_K$ [2, 32, 128]", "key basis"),
        (r"$V_V$ [2, 32, 128]", "value basis"),
        ("$a_k, a_v$", "anchors (exact)"),
        (r"$k_{\min}, k_{\max}$", "router stats")
    ]
    ox, cw = 0.45, 1.72
    cx_first = ox + cw / 2
    cx_last = ox + 4 * (cw + 0.12) + cw / 2
    ax.plot([cx_first, cx_last], [y_bus, y_bus], color=BLACK, lw=0.95, zorder=2)
    for i, (t, sub) in enumerate(outs):
        cx_i = ox + i * (cw + 0.12) + cw / 2
        box(ax, ox + i * (cw + 0.12), yo, cw, 0.78, f"{t}\n{sub}", fc=DS_BLUE_BG, ec=DS_BLUE_MED, fs=7.0)
        arrow(ax, (cx_i, y_bus), (cx_i, yo + 0.78), lw=0.9, shrinkA=0, shrinkB=0)

    ye = 5.1
    box(ax, 1.15, ye, 7.7, 0.8,
        "Base Signal — Per-token joint reconstruction error\n"
        r"$e_i = \| \Delta K_i - \widehat{\Delta K}_i \|^2 + g^2 \| \Delta V_i - \widehat{\Delta V}_i \|^2$",
        fc=WHITE, ec=DS_EMER_MED, fs=7.6)
    arrow(ax, (1.31, yo), (1.31, ye + 0.80), color=DS_EMER_MED, lw=1.1, ls=(0, (4, 3)))
    ax.text(1.5, (yo + ye + 0.8) / 2, "Reconstruct deltas from $U,V$;\nmeasure per-token residual",
            ha="left", va="center", fontsize=6.8, fontstyle="italic", color=BLACK)

    yb = 2.65
    band(ax, 0.45, yb, 9.1, 2.15, title="IDF-Weighted Boost Signals — inflate priority before the cut (all default on)",
         fc=DS_EMER_BG, ec=DS_EMER_MED)
    
    chips = [
        ("owner-capture",
         "walk $\\leq 12$ tokens left of a\nhigh-error value; boost the\nnearest capitalized word\n"
         "→ keeps entity names"),
        ("edge-capture",
         "cosine-collision test flags a\nconnective whose low-rank row\ncollides with a neighbour's key\n"
         "→ keeps relations"),
        ("coverage bonus",
         "uniform bonus across block\npositions, so one hot segment\ncannot take all $R$ slots\n"
         "→ keeps spread"),
    ]
    cw2, gp = 2.75, 0.26
    for i, (t, b) in enumerate(chips):
        px = 0.70 + i * (cw2 + gp)
        py = yb + 0.18
        box(ax, px, py, cw2, 1.55, "", fc=WHITE, ec=DS_EMER_MED, fs=7.0)
        ax.text(px + cw2/2, py + 1.3, t, ha="center", va="center", fontsize=7.8, fontweight="bold", color=BLACK)
        ax.text(px + cw2/2, py + 0.65, b, ha="center", va="center", fontsize=6.8, linespacing=1.3)

    arrow(ax, (5.0, ye), (5.0, yb + 2.15), color=DS_EMER_MED, lw=1.15)

    yr = 1.2
    box(ax, 0.9, yr, 3.9, 0.82,
        "Priority Rank  →  keep top-$R$ rows\n(default $R = 128$)",
        fc=WHITE, ec=DS_EMER_MED, fs=7.6)
    arrow(ax, (2.85, yb), (2.85, yr + 0.82), color=DS_EMER_MED, lw=1.1)
    arrow(ax, (4.8, yr + 0.41), (5.32, yr + 0.41), color=DS_EMER_MED, lw=1.1)
    
    box(ax, 5.4, yr, 4.1, 0.82,
        r"Exact Residuals   $R_K, R_V \in \mathbb{R}^{R \times H_{\text{kv}} \times d}$ fp16",
        fc=DS_EMER_LIGHT, ec=DS_EMER_MED, fs=7.8, bold=True)
    ax.text(7.45, yr - 0.22, "attended exactly at decode — verbatim needle recall",
            ha="center", va="top", fontsize=6.8, fontstyle="italic")

    ax.text(5.0, 0.35,
            r"stored block $\approx 178$ KiB ($50$ KiB low-rank + $128$ KiB residuals) vs $256$ KiB dense  $\rightarrow$  1.44$\times$ smaller   ·   $R=64$ preset $\rightarrow$ 114 KiB (2.25$\times$)",
            ha="center", va="center", fontsize=7.2, color=BLACK)

    style.finalize(fig, os.path.join(PREVIEW_DIR, "f2_compression.png"))


# ── REDESIGNED F1: SYSTEM ARCHITECTURE ───────────────────────────────────────
def redraw_f1_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    clean(ax, 10, 9.6)

    band(ax, 0.3, 8.0, 9.4, 1.3, title="Serving Engine Integration", fc=DS_GRAY_BG, ec=DS_GRAY_STROKE)
    box(ax, 0.65, 8.18, 2.6, 0.65, "OpenAI-compatible\nAPI gateway", fc=WHITE, fs=7.8)
    box(ax, 3.7, 8.18, 2.6, 0.65, "Continuous-batching\ndecode engine", fc=WHITE, fs=7.8)
    box(ax, 6.75, 8.18, 2.6, 0.65, "Session Manager\n(residency & lifecycle)", fc=WHITE, fs=7.8)
    arrow(ax, (3.25, 8.5), (3.7, 8.5))
    arrow(ax, (6.3, 8.5), (6.75, 8.5))

    box(ax, 1.6, 7.0, 6.8, 0.65,
        "MLXDKVWrapper  —  generate(): chunked prefill + decode loop",
        fc=DS_BLUE_MED, tc=WHITE, fs=8.6, bold=True)
    arrow(ax, (5.0, 8.0), (5.0, 7.65), lw=1.1)

    box(ax, 1.6, 6.05, 6.8, 0.6,
        "MLXQwenModel  —  native KV cache (prefill only) + prefill→decode release",
        fc=WHITE, fs=8.2)
    arrow(ax, (5.0, 7.0), (5.0, 6.65), lw=1.1)

    box(ax, 3.2, 5.05, 3.6, 0.65, "Patched Qwen2 Attention", fc=DS_BLUE_BG, ec=DS_BLUE_MED, fs=8.6, bold=True)
    arrow(ax, (5.0, 6.05), (5.0, 5.7), lw=1.1)

    arrow(ax, (3.8, 5.05), (2.6, 4.45), rad=-0.12, lw=1.1)
    arrow(ax, (6.2, 5.05), (7.4, 4.45), rad=0.12, lw=1.1)
    ax.text(2.35, 4.85, "prefill  ($L > 1$)", ha="center", va="center", fontsize=7.6, fontstyle="italic")
    ax.text(7.65, 4.85, "decode  ($L = 1$)", ha="center", va="center", fontsize=7.6, fontstyle="italic")

    band(ax, 0.45, 2.25, 4.25, 2.2, title="Prefill Path", fc=DS_GRAY_BG)
    box(ax, 0.7, 3.4, 3.75, 0.7, "Block-sparse causal attention\nover the native cache", fc=WHITE, fs=7.8)
    box(ax, 0.7, 2.45, 3.75, 0.7, "Capture K/V  →  Stream-compress\nblocks leaving the window", fc=WHITE, fs=7.8)
    arrow(ax, (2.58, 3.4), (2.58, 3.15), lw=0.95)

    band(ax, 5.3, 2.25, 4.25, 2.2, title="Decode Path", fc=DS_GRAY_BG)
    box(ax, 5.55, 3.4, 3.75, 0.7, "Ingest token into dense window\n(+ flush eligible block)", fc=WHITE, fs=7.8)
    box(ax, 5.55, 2.45, 3.75, 0.7, "Route top-$K$  →  Fused low-rank +\nexact-residual + recency attention",
        fc=DS_BLUE_BG, ec=DS_BLUE_MED, fs=7.8, bold=True)
    arrow(ax, (7.42, 3.4), (7.42, 3.15), lw=0.95)

    box(ax, 0.55, 0.35, 8.9, 1.45,
        "MLXKVBlockManager — per-session KV store  (×28 layers)\n"
        "dense recency window  [$H_{\\text{kv}}$, 1,280, $d$] fp16  (newest 1,024 tokens)\n"
        "+ compressed pool  {$U$, $V_K$, $V_V$, anchors, residuals, min/max} × 256 blocks",
        fc=DS_GRAY_BOX, fs=7.6)
    
    arrow(ax, (2.58, 2.45), (3.4, 1.8), rad=-0.1, lw=1.1)
    arrow(ax, (7.42, 2.45), (6.6, 1.8), rad=0.1, lw=1.1)
    ax.text(2.35, 2.05, "write", ha="right", va="center", fontsize=7.0, fontstyle="italic")
    ax.text(7.65, 2.05, "read (no decompress)", ha="left", va="center", fontsize=7.0, fontstyle="italic")

    style.finalize(fig, os.path.join(PREVIEW_DIR, "f1_architecture.png"))


# ── REDESIGNED F4: CACHE LIFECYCLE ───────────────────────────────────────────
def redraw_f4_lifecycle():
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    clean(ax, 10, 8.5)

    band(ax, 0.3, 5.6, 9.4, 2.25, title="Prefill Stage (chunk = 512 tokens, repeated)", fc=DS_GRAY_BG)
    box(ax, 0.6, 6.4, 2.0, 0.85, "Chunk Forward\n(block-sparse\ncausal SDPA)", fc=WHITE, fs=7.4)
    box(ax, 3.05, 6.4, 2.0, 0.85, "Capture Chunk\n$K,V$  →  Dense\nbuffer", fc=WHITE, fs=7.4)
    box(ax, 5.5, 6.4, 2.2, 0.85, "Window Overflow?\nFlush & compress\noldest block", fc=DS_BLUE_BG, ec=DS_BLUE_MED, fs=7.4, bold=True)
    box(ax, 8.15, 6.4, 1.3, 0.85, "Next\nChunk", fc=WHITE, ec=GRAY_D, tc=GRAY_D, fs=7.4)
    
    arrow(ax, (2.6, 6.825), (3.05, 6.825))
    arrow(ax, (5.05, 6.825), (5.5, 6.825))
    arrow(ax, (7.7, 6.825), (8.15, 6.825))

    ax.plot([8.8, 8.8, 1.6, 1.6], [6.4, 6.05, 6.05, 6.22], color=GRAY_D, lw=0.9, ls=(0, (2, 2)), zorder=2)
    arrow(ax, (1.6, 6.22), (1.6, 6.38), color=GRAY_D, lw=0.9, ls=(0, (2, 2)))
    ax.text(5.0, 5.82, "loop until full prompt is consumed", ha="center", va="center", fontsize=6.8, fontstyle="italic")

    box(ax, 1.05, 4.35, 7.9, 0.82,
        "prefill → decode boundary:   mx.eval  ·  mx.clear_cache  ·  resolve decode policy\n"
        "compressed decode  →  drop native prefill cache (footprint = DKV store only)",
        fc=DS_BLUE_MED, tc=WHITE, fs=7.4, bold=True)
    arrow(ax, (5.0, 5.6), (5.0, 5.17), lw=1.2)

    band(ax, 0.3, 0.65, 9.4, 3.2, title="Decode Stage (per token, per layer)", fc=DS_GRAY_BG)
    arrow(ax, (5.0, 4.35), (5.0, 3.85), lw=1.2)
    
    box(ax, 0.6, 1.95, 2.0, 1.0, "Ingest New Token\n$K,V$  →  Dense\nwindow (+ self)", fc=WHITE, fs=7.4)
    box(ax, 3.05, 1.95, 2.0, 1.0, "Flush Eligible\nblock if window\n$> W+B$", fc=WHITE, fs=7.4)
    box(ax, 5.5, 1.95, 2.2, 1.0, "Route Top-$K$  →\nFused sparse +\nexact attention", fc=DS_BLUE_BG, ec=DS_BLUE_MED, fs=7.4, bold=True)
    box(ax, 8.15, 1.95, 1.3, 1.0, "Logits  →\nSample", fc=WHITE, fs=7.4)
    
    arrow(ax, (2.6, 2.45), (3.05, 2.45))
    arrow(ax, (5.05, 2.45), (5.5, 2.45))
    arrow(ax, (7.7, 2.45), (8.15, 2.45))

    ax.plot([8.8, 8.8, 1.6, 1.6], [1.95, 1.58, 1.58, 1.77], color=GRAY_D, lw=0.9, ls=(0, (2, 2)), zorder=2)
    arrow(ax, (1.6, 1.77), (1.6, 1.93), color=GRAY_D, lw=0.9, ls=(0, (2, 2)))
    ax.text(5.0, 1.32, "loop for each generated token until EOS", ha="center", va="center", fontsize=6.8, fontstyle="italic")

    style.finalize(fig, os.path.join(PREVIEW_DIR, "f4_lifecycle.png"))


# ── REDESIGNED F5b: FUSED DECODE BUFFER STRIP ────────────────────────────────
def redraw_f5b_fused_buffer():
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    clean(ax, 10, 4.8)

    x0, x1 = 0.45, 9.05
    NSEG = 15
    sw = (x1 - x0) / NSEG
    LR, RES, WIN, NEW = range(0, 6), range(6, 9), range(9, 13), 13
    MASKED = (1, 4)
    cx_lr = x0 + 3.0 * sw
    cx_res = x0 + 7.5 * sw
    cx_win = x0 + 11.0 * sw
    cx_new = x0 + (NEW + 0.5) * sw

    ysrc, hsrc = 3.2, 0.78
    box(ax, cx_lr - 1.65, ysrc, 3.3, hsrc,
        "Selected blocks materialized\n" r"$\hat{K} = a_k + s U V_K, \quad \hat{V} = a_v + s U V_V$",
        fc=DS_BLUE_BG, ec=DS_BLUE_MED, fs=7.2)
    box(ax, cx_res - 0.8, ysrc, 1.6, hsrc,
        "Their exact\nresiduals $R_K, R_V$", fc=DS_EMER_BG, ec=DS_EMER_MED, fs=7.2)
    box(ax, cx_win - 1.06, ysrc, 2.12, hsrc,
        "Dense Recency Window\n1,024 tokens, fp16", fc=DS_BLUE_LIGHT, ec=DS_BLUE_MED, fs=7.2)

    ystrip, hstrip = 1.85, 0.85
    for i in range(NSEG):
        blocked = i in MASKED or i > NEW
        c = (GRAY if blocked else
             DS_BLUE_BG if i in LR else DS_EMER_BG if i in RES else
             DS_BLUE_LIGHT if i in WIN else WHITE)
        ax.add_patch(Rectangle((x0 + i * sw, ystrip), sw, hstrip, fc=c,
                               ec=BLACK, lw=0.8,
                               ls=(0, (2, 2)) if i == NEW else "-", zorder=3))
        if blocked:
            ax.text(x0 + (i + 0.5) * sw, ystrip + hstrip / 2, r"$-\infty$", ha="center", va="center", fontsize=6.8)
    
    ax.text(cx_new, ystrip + hstrip / 2, "+1", ha="center", va="center", fontsize=7.2, fontweight="bold")
    for cx in (cx_lr, cx_res, cx_win):
        arrow(ax, (cx, ysrc), (cx, ystrip + hstrip), lw=0.95, shrinkA=0, shrinkB=0)

    ypt, hpt = 0.35, 0.8
    box(ax, 0.55, ypt, 3.85, hpt, "Single scaled_dot_product_attention\nover exact-length views",
        fc=DS_BLUE_DARK, tc=WHITE, fs=7.2, bold=True)
    box(ax, 4.65, ypt, 1.3, hpt, "Attention\noutput $o$", fc=WHITE, fs=7.4)
    box(ax, 6.2, ypt, 3.1, hpt, "Each step: append new token's\n$K,V$ as one in-place row",
        fc=WHITE, fs=7.0)

    arrow(ax, (x0 + 3.5 * sw, ystrip), (x0 + 3.5 * sw, ypt + hpt), lw=1.0)
    arrow(ax, (4.4, ypt + hpt / 2), (4.65, ypt + hpt / 2), lw=1.0)
    
    ax.plot([9.3, 9.62, 9.62, cx_new], [ypt + hpt / 2, ypt + hpt / 2, 3.0, 3.0], color=BLACK, lw=1.0, zorder=2)
    arrow(ax, (cx_new, 3.0), (cx_new, ystrip + hstrip), lw=1.0, shrinkA=0, shrinkB=0)

    style.finalize(fig, os.path.join(PREVIEW_DIR, "f5b_fused_buffer.png"))


# ── REDESIGNED F7: TRITON GPU DISPATCH FLOW ──────────────────────────────────
def redraw_f7_triton_dispatch():
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    clean(ax, 10, 9.4)

    CX = 2.98                       
    LX, LW = 0.45, 5.05             

    box(ax, LX, 8.62, LW, 0.58,
        "native_triton_sparse_attn_decode($q$, blocks, routing)",
        fc=DS_BLUE_MED, tc=WHITE, fs=7.4, bold=True)
    arrow(ax, (CX, 8.62), (CX, 8.30), lw=1.1)
    
    diamond(ax, CX, 7.90, 1.22, 0.38, "HAS_TRITON ?", fs=7.6, fc=WHITE)
    ax.text(CX + 0.16, 7.31, "yes", fontsize=7.0, ha="left", fontstyle="normal")
    arrow(ax, (CX, 7.52), (CX, 7.18), lw=1.1)
    arrow(ax, (CX + 1.22, 7.90), (5.95, 7.90), lw=1.1)
    ax.text(4.58, 8.04, "no", fontsize=7.0, fontstyle="normal")

    band(ax, 5.72, 5.30, 3.93, 3.00, fc=DS_GRAY_BG, dashed=True)
    ax.text(5.9, 8.1, "Fallback Path", fontsize=7.6, ha="left", fontweight="bold")
    
    FCX = 7.68
    box(ax, 5.95, 7.42, 3.46, 0.56,
        "_pytorch_vectorized_sparse_attn_decode()", fc=WHITE, fs=7.2)
    arrow(ax, (FCX, 7.42), (FCX, 7.08), lw=1.0)
    
    diamond(ax, FCX, 6.68, 1.42, 0.38, "DKV_TRITON_STRICT = 1 ?", fs=6.9, fc=WHITE)
    arrow(ax, (7.34, 6.36), (6.86, 6.04), rad=-0.10, lw=1.0)
    arrow(ax, (8.02, 6.36), (8.50, 6.04), rad=0.10, lw=1.0)
    ax.text(6.84, 6.30, "yes", fontsize=6.9)
    ax.text(8.54, 6.30, "no", fontsize=6.9)
    
    box(ax, 5.95, 5.38, 1.62, 0.64, "hard error\n(re-raise)", fc=DS_ROSE_BG, ec=DS_ROSE_MED, tc=DS_ROSE_MED, fs=7.2, bold=True)
    box(ax, 7.79, 5.38, 1.62, 0.64, "PyTorch output\n+ telemetry log", fc=WHITE, fs=7.2)

    box(ax, LX, 6.42, LW, 0.72,
        "@triton.autotune  over  (S_MAX, BLOCKS_PER_CHUNK, num_warps)\n"
        "first five real-shape calls benchmarked → fastest config locked in",
        fc=WHITE, fs=6.8)
    arrow(ax, (CX, 6.42), (CX, 6.12), lw=1.1)
    
    box(ax, LX, 5.44, LW, 0.60,
        "_fused_sparse_decode_kernel  —  one launch grid",
        fc=DS_BLUE_MED, tc=WHITE, fs=8.0, bold=True)

    band(ax, LX, 0.8, LW, 4.45, title="Triton GPU Kernel Execution Steps", fc=DS_BLUE_BG, ec=DS_BLUE_MED)
    rows = [
        (r"1. Anchor Score:  $s_{\text{anc}} = (q \cdot a_k) \cdot \text{scale}$", WHITE, BLACK, False),
        (r"2. Delta Projection:  $\tilde{q} = q V_K^{\top} \in \mathbb{R}^{32}$", WHITE, BLACK, False),
        (r"3. Token Scores:  $\delta s_i = \tilde{q} \cdot U_i$  (all $i$ in block)", WHITE, BLACK, False),
        (r"4. Scatter $q \cdot R_K$ into score at res_K_positions", DS_EMER_BG, DS_EMER_DARK, True),
        (r"5. Scatter $p \cdot R_V$ into numerator at res_V_positions", DS_EMER_BG, DS_EMER_DARK, True),
        (r"6. Online Softmax  $\rightarrow$  accumulate $O$, lse", WHITE, BLACK, False),
    ]
    ys = [4.34, 3.75, 3.16, 2.57, 1.98, 1.39]
    for (t, fc, tc, bd), y0 in zip(rows, ys):
        box(ax, LX + 0.20, y0, LW - 0.40, 0.48, t, fc=fc,
            ec=DS_EMER_MED if fc is DS_EMER_BG else BLACK, tc=tc, fs=6.9, bold=bd)
    for y0 in ys[:-1]:
        arrow(ax, (CX, y0), (CX, y0 - 0.11), lw=0.85, shrinkA=0, shrinkB=0)
        
    ax.text(LX + 0.20, 0.95, r"$O(K \cdot r \cdot B)$ compute — block is never expanded to dense $\hat{K}$",
            fontsize=6.9, ha="left", fontstyle="italic")

    style.finalize(fig, os.path.join(PREVIEW_DIR, "f7_triton_dispatch.png"))


# ── REDESIGNED F9: MULTI-BACKEND DISPATCH ───────────────────────────────────
def redraw_f9_backend_dispatch():
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    clean(ax, 10, 9.4)

    box(ax, 2.5, 8.3, 5.0, 0.75,
        "DKV Unified Engine Dispatch Router\n"
        "select_attention_backend(device, batch_size, seq_len)",
        fc=DS_BLUE_DARK, tc=WHITE, fs=8.2, bold=True)
    
    arrow(ax, (5.0, 8.3), (5.0, 7.7), lw=1.1)

    diamond(ax, 5.0, 7.2, 1.8, 0.45, "Target Backend?", fc=WHITE, fs=8.0)

    arrow(ax, (3.2, 7.2), (2.0, 6.2), rad=-0.1, lw=1.1)
    arrow(ax, (5.0, 6.75), (5.0, 6.2), lw=1.1)
    arrow(ax, (6.8, 7.2), (8.0, 6.2), rad=0.1, lw=1.1)

    ax.text(2.3, 6.95, "Apple Silicon (MLX)", ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(5.0, 6.55, "NVIDIA GPU (CUDA)", ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(7.7, 6.95, "CPU / Fallback", ha="center", va="center", fontsize=7.2, fontweight="bold")

    band(ax, 0.35, 1.2, 2.9, 4.8, title="MLX Engine (Metal)", fc=DS_BLUE_BG, ec=DS_BLUE_MED)
    box(ax, 0.5, 4.8, 2.6, 0.7, "mlx.core Metal JIT\nVectorized GEMM", fc=WHITE, fs=7.4)
    box(ax, 0.5, 3.6, 2.6, 0.7, "Fused Low-Rank +\nResidual Attention", fc=DS_BLUE_BG, ec=DS_BLUE_MED, fs=7.4, bold=True)
    box(ax, 0.5, 2.4, 2.6, 0.7, "In-place Block Store\nUnified Memory", fc=WHITE, fs=7.4)
    arrow(ax, (1.8, 4.8), (1.8, 4.3))
    arrow(ax, (1.8, 3.6), (1.8, 3.1))

    band(ax, 3.55, 1.2, 2.9, 4.8, title="CUDA / Triton Engine", fc=DS_EMER_BG, ec=DS_EMER_MED)
    box(ax, 3.7, 4.8, 2.6, 0.7, "Triton Fused Kernel\nAutotuned Grid Launch", fc=WHITE, fs=7.4)
    box(ax, 3.7, 3.6, 2.6, 0.7, "Warp-Level Reduction\nTensor Core FP16", fc=DS_EMER_BG, ec=DS_EMER_MED, fs=7.4, bold=True)
    box(ax, 3.7, 2.4, 2.6, 0.7, "Paged KV Pool &\nExact Residual Cache", fc=WHITE, fs=7.4)
    arrow(ax, (5.0, 4.8), (5.0, 4.3))
    arrow(ax, (5.0, 3.6), (5.0, 3.1))

    band(ax, 6.75, 1.2, 2.9, 4.8, title="PyTorch CPU Fallback", fc=DS_GRAY_BG, ec=DS_GRAY_STROKE)
    box(ax, 6.9, 4.8, 2.6, 0.7, "PyTorch Vectorized\nReference SDPA", fc=WHITE, fs=7.4)
    box(ax, 6.9, 3.6, 2.6, 0.7, "Full Correctness\nTelemetry Verification", fc=WHITE, fs=7.4)
    box(ax, 6.9, 2.4, 2.6, 0.7, "Strict Execution\nSafety Checks", fc=WHITE, fs=7.4)
    arrow(ax, (8.2, 4.8), (8.2, 4.3))
    arrow(ax, (8.2, 3.6), (8.2, 3.1))

    box(ax, 1.2, 0.25, 7.6, 0.75, r"Unified Output Tensor  $O \in \mathbb{R}^{B \times 1 \times H \times d}$   (Verbatim Recall Preserved)",
        fc=DS_BLUE_DARK, tc=WHITE, fs=8.2, bold=True)
    
    arrow(ax, (1.8, 1.2), (2.8, 1.0), rad=-0.1, color=DS_BLUE_MED, lw=1.1)
    arrow(ax, (5.0, 1.2), (5.0, 1.0), color=DS_EMER_MED, lw=1.1)
    arrow(ax, (8.2, 1.2), (7.2, 1.0), rad=0.1, color=GRAY_D, lw=1.1)

    style.finalize(fig, os.path.join(PREVIEW_DIR, "f9_backend_dispatch.png"))


if __name__ == "__main__":
    print("Generating redesigned DeepSeek-V3.2 style diagrams in paper/figures_preview/ ...")
    redraw_f5a_decode_attention()
    redraw_f2_compression()
    redraw_f1_architecture()
    redraw_f4_lifecycle()
    redraw_f5b_fused_buffer()
    redraw_f7_triton_dispatch()
    redraw_f9_backend_dispatch()
    print("All preview diagrams successfully generated!")
