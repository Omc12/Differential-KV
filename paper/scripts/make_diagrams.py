#!/usr/bin/env python3
"""Architecture / dataflow diagrams for the DiffKV paper — DeepSeek-report style.

Visual language (matched to DeepSeek-V3.2 Figures 2 and 4):
  flat rounded rectangles, thin black strokes, light fills; medium blue with
  white text for "hot" compute; emerald tints reserved for the EXACT path
  (residuals / recency); tiny italic annotations; no in-figure titles (the
  LaTeX caption carries the title); text otherwise always black.

Structural numbers match the code at the measured config (rank 32, R=128,
block 256, window 768, pool 256, top-K 16).

  F1  system architecture        F2  compression pipeline
  F4  cache lifecycle            F5  fused routed decode attention
  F6  3D memory architecture
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import (BLACK, BLUE, BLUE_D, BLUE_L, BLUE_XL, EMERALD, EMER_L,
                   GRAY_XL, GRAY_D, LEGEND_EC, WHITE)
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


def band(ax, x, y, w, h, fc=GRAY_XL, ec=LEGEND_EC, lw=0.8, ls="-", dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, zorder=1,
                       linestyle=(0, (4, 3)) if dashed else "-")
    ax.add_patch(p)


def arrow(ax, p0, p1, color=BLACK, lw=1.0, rad=0.0, ls="-", style_="-|>", z=2):
    a = FancyArrowPatch(p0, p1, arrowstyle=style_, mutation_scale=10, color=color,
                        lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=z,
                        linestyle=ls)
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
        "MLXDiffKVWrapper  —  generate(): chunked prefill + decode loop",
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
        "dense recency window  [$H_{kv}$, 768, $d$]  fp16\n"
        "+  compressed pool  {$U$, $V_K$, $V_V$, anchors, residuals, min/max}  × 256 blocks",
        fc=GRAY_XL, fs=7.5)
    arrow(ax, (2.58, 2.78), (3.4, 1.92), rad=-0.1)
    arrow(ax, (7.42, 2.78), (6.6, 1.92), rad=0.1)
    note(ax, 2.35, 2.32, "write", fs=6.8, ha="right")
    note(ax, 7.65, 2.32, "read (no decompress)", fs=6.8, ha="left")

    style.finalize(fig, os.path.join(FIG, "f1_architecture.png"))


# ── F2 · compression pipeline ────────────────────────────────────────────────
def f2_compression():
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); clean(ax, 10, 7.2)

    y = 5.6
    # token block
    bx = 0.45
    for i in range(8):
        c = BLUE_L if i == 0 else (EMER_L if i in (5, 6) else BLUE_XL)
        ax.add_patch(Rectangle((bx + i * 0.33, y), 0.29, 0.62, fc=c, ec=BLACK,
                               lw=0.8, zorder=3))
    note(ax, bx + 0.145, y - 0.24, "anchor\n(token 0)", fs=6.6, va="top", style_="normal")
    note(ax, bx + 1.32, y + 0.92, "$K,V$ block  [$H_{kv}$, 256, $d$]", fs=7.8, style_="normal")
    note(ax, bx + 2.15, y - 0.24, "outlier tokens", fs=6.6, va="top")

    arrow(ax, (3.2, y + 0.31), (3.62, y + 0.31), lw=1.1)
    box(ax, 3.66, y - 0.06, 1.5, 0.74, "$\\Delta K = K - a_k$\n$\\Delta V = V - a_v$", fc=WHITE, fs=7.8)
    arrow(ax, (5.16, y + 0.31), (5.55, y + 0.31), lw=1.1)
    box(ax, 5.6, y - 0.06, 1.75, 0.74, "rescale $V$ to $K$ RMS,\nrow-normalize, concat", fc=WHITE, fs=7.4)
    arrow(ax, (7.35, y + 0.31), (7.74, y + 0.31), lw=1.1)
    box(ax, 7.78, y - 0.06, 1.7, 0.74, "randomized\ntruncated SVD\n(rank 32, seeded)", fc=BLUE, tc=WHITE, fs=7.4, bold=True)

    # low-rank outputs
    yo = 3.1
    arrow(ax, (8.63, y - 0.06), (8.63, yo + 0.85), lw=1.1)
    note(ax, 8.86, 4.35, "$U\\,\\Sigma\\,V^{\\top}$", fs=7.6, ha="left", style_="normal")
    outs = [("$U$  [255, 32]", "coefficients"),
            ("$V_K$  [2, 32, 128]", "key basis"),
            ("$V_V$  [2, 32, 128]", "value basis"),
            ("$a_k, a_v$", "anchors (exact)"),
            ("$k_{\\min}, k_{\\max}$", "router stats")]
    ox, cw = 0.45, 1.72
    for i, (t, sub) in enumerate(outs):
        box(ax, ox + i * (cw + 0.12), yo, cw, 0.78, f"{t}\n{sub}", fc=BLUE_XL, fs=7.0)
        arrow(ax, (8.63, yo + 0.9), (ox + i * (cw + 0.12) + cw / 2, yo + 0.80),
              lw=0.7, style_="-")

    # residual branch (EXACT path → emerald): the selection signal is the
    # reconstruction error of the low-rank factors, so the arrow leaves the
    # OUTPUT row (U, V bases), not the router stats.
    yr = 1.15
    box(ax, 0.45, yr, 4.6, 0.85,
        "rank per-token joint reconstruction error\n→ keep top-$R$ worst rows (default $R{=}128$)",
        fc=WHITE, ec=EMERALD, fs=7.6)
    arrow(ax, (2.3, yo - 0.02), (2.3, yr + 0.87), color=EMERALD,
          lw=1.1, ls=(0, (4, 3)))
    note(ax, 2.5, (yo + yr + 0.85) / 2, "reconstruct deltas from $U, V$;\nmeasure per-token error",
         fs=6.6, ha="left")
    arrow(ax, (5.05, yr + 0.42), (5.5, yr + 0.42), color=EMERALD, lw=1.1)
    box(ax, 5.55, yr, 3.9, 0.85,
        "exact residuals   $R_K, R_V$  [$R$, $H_{kv}$, $d$]  fp16",
        fc=EMER_L, ec=EMERALD, fs=7.8, bold=True)
    note(ax, 7.5, yr - 0.28, "attended exactly at decode — verbatim recall", fs=6.8, va="top")

    # byte budget line
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
        "compressed decode → drop the native prefill cache (footprint = DiffKV store only)",
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


# ── F5 · fused routed decode attention ───────────────────────────────────────
def f5_decode():
    fig, ax = plt.subplots(figsize=(8.6, 5.7)); clean(ax, 10, 9.4)

    box(ax, 4.15, 8.55, 1.7, 0.55, "query $q$  [$H$, $d$]", fc=BLUE_D, tc=WHITE, fs=8.2, bold=True)
    arrow(ax, (5.0, 8.55), (5.0, 8.15), lw=1.2)

    box(ax, 2.15, 7.4, 5.7, 0.7,
        "router:  $\\rho_b = \\max(\\,q{\\cdot}a_k,\\ \\max_j\\, q{\\cdot}R_{K,j})\\cdot$scale   →   keep top-$K{=}16$ blocks",
        fc=GRAY_XL, fs=7.8)
    note(ax, 8.35, 7.75, "exact $q{\\cdot}k$ over anchor\n+ residual keys", fs=6.6, ha="left")
    arrow(ax, (3.1, 7.4), (2.55, 6.75), rad=-0.12, lw=1.1)
    arrow(ax, (6.9, 7.4), (7.45, 6.75), rad=0.12, lw=1.1)

    # low-rank branch (blue)
    band(ax, 0.35, 2.1, 4.55, 4.65, fc="#F6F9FE", ec=BLUE_L)
    note(ax, 0.6, 6.5, "Low-rank branch  (selected blocks)", fs=7.6, ha="left", style_="normal")
    box(ax, 0.62, 5.55, 4.0, 0.6, "anchor score   $s_{anc} = (q\\cdot a_k)\\cdot$scale", fc=WHITE, fs=7.6)
    box(ax, 0.62, 4.72, 4.0, 0.6, "project query   $\\tilde q = (q\\,V_K^{\\top})\\cdot$scale $\\in \\mathbb{R}^{32}$", fc=WHITE, fs=7.6)
    box(ax, 0.62, 3.89, 4.0, 0.6, "delta scores   $\\delta s = s\\,(\\tilde q\\,U^{\\top}) + s_{anc}$", fc=BLUE, tc=WHITE, fs=7.6, bold=True)
    box(ax, 0.62, 3.06, 4.0, 0.6, "per-block softmax  →  $w$, lse$_{sp}$", fc=WHITE, fs=7.6)
    box(ax, 0.62, 2.23, 4.0, 0.6, "$O_{sp} = \\Sigma w\\,a_v + (\\Sigma w\\,U)\\,s\\,V_V$", fc=WHITE, fs=7.6)
    for y0 in (5.55, 4.72, 3.89, 3.06):
        arrow(ax, (2.62, y0), (2.62, y0 - 0.23), lw=0.9)
    note(ax, 1.6, 1.78, "$O(K\\,r\\,B)$ — never forms $\\hat K$", fs=6.8, ha="left")

    # exact branch (emerald)
    band(ax, 5.1, 2.1, 4.55, 4.65, fc="#F5FBF8", ec=EMER_L)
    note(ax, 5.35, 6.5, "Exact branch  (residuals + recency)", fs=7.6, ha="left", style_="normal")
    box(ax, 5.38, 5.25, 4.0, 0.85, "concat selected blocks' exact residuals\n$R_K, R_V$  ⊕  dense recency window", fc=WHITE, ec=EMERALD, fs=7.6)
    box(ax, 5.38, 4.0, 4.0, 0.85, "exact attention over the\naugmented $\\tilde K, \\tilde V$  (full fp16)", fc=EMERALD, tc=WHITE, fs=7.6, bold=True)
    box(ax, 5.38, 3.06, 4.0, 0.6, "softmax  →  $O_{dn}$, lse$_{dn}$", fc=WHITE, fs=7.6)
    arrow(ax, (7.38, 5.25), (7.38, 4.88), lw=0.9); arrow(ax, (7.38, 4.0), (7.38, 3.69), lw=0.9)
    note(ax, 8.4, 2.5, "verbatim recall of\nneedle tokens", fs=6.8)

    # merge
    box(ax, 3.3, 0.5, 3.4, 1.0,
        "flash-style LSE merge\n$m = \\max($lse$_{sp}$, lse$_{dn})$\n$o = (e^{\\cdot}O_{sp} + e^{\\cdot}O_{dn})\\,/\\,\\Sigma$",
        fc=BLUE_D, tc=WHITE, fs=7.4, bold=True)
    arrow(ax, (2.62, 2.23), (3.6, 1.5), color=BLUE, rad=-0.15, lw=1.3)
    arrow(ax, (7.38, 3.06), (6.4, 1.5), color=EMERALD, rad=0.15, lw=1.3)
    note(ax, 5.0, 0.16, "NaN/inf-guarded: an empty compressed set contributes exactly zero weight", fs=6.6)

    style.finalize(fig, os.path.join(FIG, "f5_decode_attention.png"))


# ── F6 · 3D memory architecture ──────────────────────────────────────────────
def f6_memory_3d():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(8.8, 5.4))
    ax = fig.add_axes([0.0, -0.06, 1.0, 1.02], projection="3d")
    ax.set_proj_type("ortho")

    def cuboid(o, size, color, alpha=1.0, ec=BLACK, lw=0.5, z=1):
        x, y, zc = o; dx, dy, dz = size
        pts = np.array([[x, y, zc], [x+dx, y, zc], [x+dx, y+dy, zc], [x, y+dy, zc],
                        [x, y, zc+dz], [x+dx, y, zc+dz], [x+dx, y+dy, zc+dz], [x, y+dy, zc+dz]])
        faces = [[pts[j] for j in f] for f in
                 ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6],
                  [1, 2, 6, 5], [0, 3, 7, 4])]
        pc = Poly3DCollection(faces, facecolors=color, edgecolors=ec,
                              linewidths=lw, alpha=alpha)
        pc.set_zsort("max"); pc.set_zorder(z)
        ax.add_collection3d(pc)

    # layer stack (left, receding)
    n_shown = 5
    for i in range(n_shown):
        yy = 3.1 * i
        cuboid((0.0, yy, 0.0), (2.0, 2.4, 0.16), BLUE_XL, alpha=0.9,
               ec=GRAY_D, lw=0.5, z=i)
    # (layer labels are placed in 2D figure space below — 3D projection of text
    #  anchors is unreliable across zoom changes)

    # block pool (right): blue low-rank base + emerald residual cap
    x0 = 5.0
    z_lr, z_res = 1.7, 4.4
    bw, gap, depth = 0.72, 0.30, 2.0
    n_used, n_ghost = 8, 3
    for b in range(n_used + n_ghost):
        x = x0 + b * (bw + gap)
        used = b < n_used
        cuboid((x, 0.0, 0.0), (bw, depth, z_lr), BLUE,
               alpha=(0.95 if used else 0.20), ec=(BLACK if used else GRAY_D), lw=0.6, z=30 + b)
        cuboid((x, 0.0, z_lr), (bw, depth, z_res), EMERALD,
               alpha=(0.80 if used else 0.16), ec=(BLACK if used else GRAY_D), lw=0.6, z=30 + b)

    # dense recency window slab (front)
    cuboid((x0, -3.0, 0.0), ((bw + gap) * 6.4, 1.7, 1.0), BLUE_L, alpha=0.95,
           ec=BLACK, lw=0.7, z=80)

    # flush line: window slab -> first pool block (kept off the bar faces)
    ax.plot([x0 + 0.9, x0 + 0.35], [-1.3, -0.05], [0.75, 0.75], color=BLACK,
            lw=1.3, zorder=90)

    ax.set_xlim(0, x0 + (n_used + n_ghost) * (bw + gap) + 0.4)
    ax.set_ylim(-3.2, 3.1 * (n_shown - 1) + 2.5)
    ax.set_zlim(0, z_lr + z_res + 0.4)
    ax.view_init(elev=23, azim=-60)
    ax.set_axis_off()
    ax.set_box_aspect((12.5, 11.0, 4.0), zoom=1.52)

    # callouts — all in 2D figure space, positions verified against the render
    fig.text(0.085, 0.47, "28-layer stack\n(store replicated\nper layer)", ha="center",
             va="center", fontsize=8.2, color=BLACK, linespacing=1.35)
    fig.text(0.245, 0.615, "layer 0", ha="right", fontsize=7.5, color=BLACK)
    fig.text(0.155, 0.335, "layer 27", ha="right", fontsize=7.5, color=BLACK)
    fig.text(0.60, 0.775, "compressed block pool — 256 slots (bounded)", ha="center",
             fontsize=8.6, color=BLACK)
    fig.text(0.475, 0.695, "used blocks", ha="center", fontsize=7.5, color=BLACK)
    fig.text(0.715, 0.585, "free (zeroed)", ha="center", fontsize=7.4, color=GRAY_D)
    fig.text(0.50, 0.055, "dense recency window — 768 exact fp16 tokens", ha="center",
             fontsize=8.2, color=BLACK)
    fig.text(0.125, 0.245, "overflow →\nflush + compress", ha="center", fontsize=7.4,
             color=BLACK, linespacing=1.3)

    # legend strip under the top edge
    leg = [Patch(fc=BLUE, ec=BLACK, label="low-rank core  $U, V_K, V_V$, anchors, min/max  (≈50 KiB)"),
           Patch(fc=EMERALD, ec=BLACK, label="exact residuals  $R_K, R_V$  (≈128 KiB, $R{=}128$)"),
           Patch(fc=BLUE_L, ec=BLACK, label="dense recency window (fp16)")]
    lg = fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                    ncol=3, fontsize=7.2, frameon=True, columnspacing=1.2,
                    handlelength=1.3, borderpad=0.5)
    lg.get_frame().set_edgecolor(LEGEND_EC); lg.get_frame().set_linewidth(0.7)

    fig.savefig(os.path.join(FIG, "f6_memory_3d.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(FIG, "f6_memory_3d.pdf"),
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", os.path.join(FIG, "f6_memory_3d.png"))


if __name__ == "__main__":
    f1_architecture()
    f2_compression()
    f4_lifecycle()
    f5_decode()
    f6_memory_3d()
    print("diagrams done")
