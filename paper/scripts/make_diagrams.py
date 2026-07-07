#!/usr/bin/env python3
"""Original publication-quality architecture / dataflow diagrams for the paper.

Pure matplotlib vector drawing -> PNG + PDF, one coherent DeepSeek-style visual
identity (style.py). All structural numbers match the code
(MLXKVBlockManager: block_size=256, rank=16, kv_heads=2, head_dim=128,
recency_window=512, max_residual=128 default, max_blocks=256, topk_blocks=16).

  F1  System architecture (serving stack -> MLX core -> patched attention)
  F2  Differential compression pipeline (anchor + low-rank + residuals)
  F3  Per-session KV store memory layout (2D)
  F4  Cache lifecycle & execution flow (prefill -> boundary -> decode)
  F5  Fused routed decode attention dataflow (low-rank + exact -> LSE merge)
  F6  3D memory architecture (exploded layered block pool)   [3D]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import (INK, BLUE, SKY, MIST, SLATE, AMBER, GOOD, BAD, GRID,
                   PANEL, MUTED, WHITE)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

style.apply_rc()
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)


def box(ax, x, y, w, h, text, fc=WHITE, ec=INK, tc=INK, fs=10, lw=1.6, bold=False,
        round=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={round}",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, fontweight="bold" if bold else "normal",
            linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=INK, lw=1.8, style_="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style_, mutation_scale=14, color=color,
                        lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=2,
                        linestyle=ls)
    ax.add_patch(a)


def title(ax, t, sub=None):
    ax.text(0.5, 0.985, t, transform=ax.transAxes, ha="center", va="top",
            fontsize=15, fontweight="bold", color=INK)
    if sub:
        ax.text(0.5, 0.938, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=10.0, color=MUTED)


def clean(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


# ───────────────────────── F1: system architecture ──────────────────────────
def f1_architecture():
    fig, ax = plt.subplots(figsize=(11, 7.2)); clean(ax)
    title(ax, "DiffKV Active Runtime — System Architecture",
          "Serving stack drives the MLX inference core; the patched attention layer routes prefill vs. decode")

    box(ax, 0.3, 8.05, 9.4, 0.5, "", fc=PANEL, ec=GRID, lw=1)
    ax.text(0.5, 8.3, "Serving", fontsize=9, color=MUTED, fontstyle="italic", va="center")
    box(ax, 1.6, 8.08, 2.2, 0.44, "OpenAI-compatible\nAPI gateway", fc=WHITE, fs=8.4)
    box(ax, 4.0, 8.08, 2.0, 0.44, "Continuous-batching\ndecode engine", fc=WHITE, fs=8.4)
    box(ax, 6.2, 8.08, 2.3, 0.44, "Session manager\n(residency, lifecycle)", fc=WHITE, fs=8.4)
    arrow(ax, (3.8, 8.3), (4.0, 8.3)); arrow(ax, (6.0, 8.3), (6.2, 8.3))

    box(ax, 1.4, 6.85, 7.2, 0.7,
        "MLXDiffKVWrapper  ·  generate(): chunked prefill  +  greedy / sampled decode loop",
        fc=INK, ec=INK, tc=WHITE, fs=9.5, bold=True)
    arrow(ax, (5.0, 8.05), (5.0, 7.57), lw=2.0)

    box(ax, 1.4, 5.75, 7.2, 0.62,
        "MLXQwenModel  ·  per-session native KVCache (prefill)  +  prefill→decode memory release",
        fc=WHITE, fs=9.2)
    arrow(ax, (5.0, 6.85), (5.0, 6.37), lw=2.0)

    box(ax, 3.0, 4.55, 4.0, 0.7, "Patched Qwen2 Attention\n(monkey-patched __call__)",
        fc=BLUE, ec=INK, tc="white", fs=9.6, bold=True)
    arrow(ax, (5.0, 5.75), (5.0, 5.25), lw=2.0)

    arrow(ax, (4.0, 4.55), (2.6, 3.75), color=INK, rad=-0.15)
    arrow(ax, (6.0, 4.55), (7.4, 3.75), color=INK, rad=0.15)
    ax.text(2.6, 4.2, "L > 1   PREFILL", fontsize=8.5, color=INK, ha="center", fontweight="bold")
    ax.text(7.4, 4.2, "L == 1   DECODE", fontsize=8.5, color=INK, ha="center", fontweight="bold")

    box(ax, 0.7, 2.95, 3.7, 0.7, "sparse causal SDPA over native cache\n→ capture K/V into dense buffer",
        fc=PANEL, fs=8.6)
    box(ax, 0.7, 2.0, 3.7, 0.7, "streaming flush+compress oldest block\n(anchor + low-rank SVD + exact residuals)",
        fc=PANEL, fs=8.2)
    arrow(ax, (2.55, 2.95), (2.55, 2.7))

    box(ax, 5.6, 2.95, 3.7, 0.7, "ingest current token → dense buffer\n(+ self), flush eligible block",
        fc=PANEL, fs=8.6)
    box(ax, 5.6, 2.0, 3.7, 0.7, "route top-K → fused low-rank +\nexact-residual + dense attention",
        fc=PANEL, fs=8.3)
    arrow(ax, (7.45, 2.95), (7.45, 2.7))

    box(ax, 1.9, 0.5, 6.2, 1.0,
        "MLXKVBlockManager — per-session KV store\n"
        "Dense recency window [H$_{kv}$, 768, d] fp16   +   Compressed pool "
        "{U, V$_K$, V$_V$, anchors, residuals, min/max} × 256",
        fc=INK, ec=INK, tc=WHITE, fs=8.6, bold=True)
    arrow(ax, (2.55, 2.0), (3.2, 1.5), color=MUTED, rad=-0.1)
    arrow(ax, (7.45, 2.0), (6.8, 1.5), color=MUTED, rad=0.1)
    arrow(ax, (4.2, 2.35), (4.2, 1.5), color=MUTED, ls=":", style_="-|>")
    ax.text(4.5, 1.78, "read (no decompress)", fontsize=7.6, color=MUTED, rotation=90, va="center")

    style.finalize(fig, os.path.join(FIG, "f1_architecture.png"))


# ───────────────────────── F2: compression pipeline ─────────────────────────
def f2_compression():
    fig, ax = plt.subplots(figsize=(11, 6.4)); clean(ax)
    ax.set_ylim(0, 9)
    title(ax, "Differential KV Compression — one 256-token block",
          "Anchor + low-rank joint SVD delta (shared structure)  +  top-error exact residuals (the outliers SVD discards)")

    y = 6.4
    bx = 0.4
    for i in range(8):
        c = AMBER if i == 0 else (BAD if i in (5, 6) else SKY)
        ax.add_patch(Rectangle((bx + i * 0.34, y), 0.30, 0.7, fc=c, ec=INK, lw=1.0, zorder=3))
    ax.text(bx + 0.15, y - 0.30, "anchor\n(tok 0)", fontsize=7.2, color=INK, ha="center", va="top")
    ax.text(bx + 8 * 0.34 / 2 + 0.5, y + 1.0, "K, V block  [H$_{kv}$, 256, d]", fontsize=8.8, color=INK, ha="center")
    ax.text(bx + 4 * 0.34, y - 0.92, "256 tokens  (anchor + 255 deltas)", fontsize=7.4, color=MUTED, ha="center")

    arrow(ax, (3.35, y + 0.35), (3.85, y + 0.35), lw=2.0)
    box(ax, 3.9, y - 0.05, 1.6, 0.8, "ΔK = K − a$_k$\nΔV = V − a$_v$", fc=PANEL, fs=8.4)
    arrow(ax, (5.5, y + 0.35), (5.95, y + 0.35), lw=2.0)
    box(ax, 6.0, y - 0.05, 1.7, 0.8, "V-rescale to K RMS\nrow L2-norm, concat", fc=PANEL, fs=8.2)
    arrow(ax, (7.7, y + 0.35), (8.15, y + 0.35), lw=2.0)
    box(ax, 8.2, y - 0.05, 1.5, 0.8, "randomized\ntrunc. SVD\n(rank 16, seeded)", fc=BLUE, tc="white", fs=8.0, bold=True)

    yo = 3.7
    arrow(ax, (8.95, y - 0.05), (8.95, yo + 0.95), rad=0.0, lw=2.0)
    ax.text(9.28, 5.2, "U Σ Vᵀ", fontsize=8.2, color=INK, va="center")
    outs = [
        ("U [255,16]", "coeffs", BLUE),
        ("V$_K$", "K basis", INK),
        ("V$_V$", "V basis", INK),
        ("a$_k$, a$_v$", "anchors", AMBER),
        ("k$_{min}$, k$_{max}$", "router", SLATE),
    ]
    ox = 0.6
    for i, (t, sub, c) in enumerate(outs):
        box(ax, ox + i * 1.55, yo, 1.4, 0.95, t + "\n" + sub, fc=WHITE, ec=c, tc=INK, fs=8.0)
    for i in range(len(outs)):
        arrow(ax, (8.95, yo + 1.55), (ox + i * 1.55 + 0.7, yo + 0.95), color=MUTED, lw=0.9, style_="-")

    yr = 1.95
    box(ax, 0.6, yr, 4.5, 0.95,
        "measure per-token joint recon. error\n→ keep top-$R$ highest-error tokens (default $R{=}128$)",
        fc=PANEL, ec=BAD, fs=8.4)
    arrow(ax, (8.95, yo - 0.05), (5.6, yr + 0.95), color=BAD, rad=0.18, lw=1.6, ls="--")
    arrow(ax, (5.1, yr + 0.47), (5.55, yr + 0.47), lw=1.8, color=BAD)
    box(ax, 5.6, yr, 4.0, 0.95, "exact residuals\nR$_K$, R$_V$  [$R$, H$_{kv}$, d]  fp16", fc=WHITE, ec=BAD, tc=BAD, fs=8.6, bold=True)

    ax.text(5.0, 1.02, "stored block  ≈ 154 KiB  (26 KiB low-rank  +  128 KiB residuals, $R{=}128$)   vs   256 KiB dense   →   1.66× smaller",
            fontsize=9.2, color=INK, ha="center", fontweight="bold")
    ax.text(5.0, 0.5, "$R{=}64$ memory preset → 90 KiB, 2.85×.   Low-rank recon (implicit): $\\hat K_i \\approx a_k + s\\,(U_i V_K)$;  residuals attended EXACTLY at decode.",
            fontsize=7.8, color=MUTED, ha="center")

    style.finalize(fig, os.path.join(FIG, "f2_compression.png"))


# ───────────────────────── F3: session memory layout ────────────────────────
def f3_memory_layout():
    fig, ax = plt.subplots(figsize=(11, 6.0)); clean(ax)
    ax.set_ylim(0, 9)
    title(ax, "Per-Session KV Store — memory layout (one layer of 28)",
          "Bounded dense recency window + fixed compressed pool — the state that does NOT grow unboundedly with context")

    ax.text(0.5, 7.7, "Compressed pool  (fixed M = 256 blocks  ·  ≤ 65 536 tokens)",
            fontsize=10.5, color=INK, fontweight="bold")
    px, py, pw = 0.5, 5.0, 9.0
    box(ax, px, py, pw, 2.4, "", fc=PANEL, ec=GRID, lw=1)
    cols = ["U\n[255,16]", "V$_K$\n[2,16,128]", "V$_V$\n[2,16,128]", "a$_k$,a$_v$\n[2,128]",
            "R$_K$,R$_V$\n[128,2,128]", "k$_{min}$\nk$_{max}$", "scale\nseq_len"]
    cw = 1.18
    for i, c in enumerate(cols):
        fc = BLUE if i == 0 else (INK if i in (1, 2) else (AMBER if i == 3 else (BAD if i == 4 else WHITE)))
        tc = "white" if fc in (INK, BLUE, AMBER, BAD) else INK
        box(ax, px + 0.25 + i * cw, py + 1.3, cw - 0.18, 0.9, c, fc=fc, tc=tc, fs=7.6)
    ax.text(px + pw / 2, py + 0.6, "block 0   block 1   …   block k   ·   (k+1 … 255 pre-allocated, zeroed)",
            fontsize=8.6, color=MUTED, ha="center")
    ax.text(px + pw + 0.0, py + 2.2, "≈154 KiB\n/block", fontsize=7.6, color=INK, ha="left", va="top")
    ax.text(px + 0.25 + 4 * cw + (cw - 0.18) / 2, py + 1.25, "exact", fontsize=6.8, color=BAD,
            ha="center", va="top")

    ax.text(0.5, 4.3, "Dense recency window  (sliding, capacity W+B = 768 tokens, uncompressed fp16)",
            fontsize=10.5, color=INK, fontweight="bold")
    dx, dy, dw = 0.5, 2.5, 9.0
    box(ax, dx, dy, dw, 1.3, "", fc=PANEL, ec=GRID, lw=1)
    n = 24
    for i in range(n):
        filled = i < 18
        ax.add_patch(Rectangle((dx + 0.2 + i * ((dw - 0.4) / n), dy + 0.35), (dw - 0.4) / n - 0.04, 0.6,
                               fc=SKY if filled else WHITE, ec=INK, lw=0.8))
    ax.text(dx + dw * 0.45, dy + 0.15, "dense_len (live)", fontsize=8.0, color=INK, ha="center", va="top")
    ax.text(dx + dw * 0.86, dy + 0.15, "free", fontsize=8.0, color=MUTED, ha="center", va="top")

    arrow(ax, (dx + 0.6, dy + 1.3), (px + 0.8, py), color=BAD, rad=-0.25, lw=2.0)
    ax.text(1.0, 4.75, "overflow → flush+compress\noldest block", fontsize=8.2, color=BAD, ha="left")

    box(ax, 0.5, 0.5, 9.0, 1.4,
        "Footprint per layer ≈  dense 768·H$_{kv}$·d·2·2 B  +  used_blocks · 154 KiB.\n"
        "The dense window is constant; the compressed term grows at ≈ 1/1.66 the dense-KV rate (default $R$),\n"
        "and the pool is capped at 256 blocks → the bounded memory slope.",
        fc=WHITE, ec=INK, fs=9.0)

    style.finalize(fig, os.path.join(FIG, "f3_memory_layout.png"))


# ───────────────────────── F4: cache lifecycle / flow ───────────────────────
def f4_lifecycle():
    fig, ax = plt.subplots(figsize=(11, 6.2)); clean(ax)
    ax.set_ylim(0, 9)
    title(ax, "Cache Lifecycle & Execution Flow",
          "Chunked prefill (sparse) → streaming compression → memory release → fused sparse decode")

    ax.add_patch(Rectangle((0.3, 5.2), 9.4, 3.1, fc=PANEL, ec=GRID, lw=1, zorder=0))
    ax.text(0.5, 8.1, "PREFILL  (chunk = 512 tokens, repeated)", fontsize=10.5, color=INK, fontweight="bold")
    box(ax, 0.6, 6.9, 2.0, 0.9, "chunk forward\n(native KVCache)\nsparse causal SDPA", fc=WHITE, fs=8.2)
    box(ax, 3.0, 6.9, 2.0, 0.9, "capture chunk\nK/V → dense buffer", fc=WHITE, fs=8.2)
    box(ax, 5.4, 6.9, 2.1, 0.9, "dense full?\nflush+compress\noldest block", fc=WHITE, fs=8.2)
    box(ax, 7.9, 6.9, 1.6, 0.9, "next chunk", fc=WHITE, ec=MUTED, tc=MUTED, fs=8.2)
    arrow(ax, (2.6, 7.35), (3.0, 7.35)); arrow(ax, (5.0, 7.35), (5.4, 7.35))
    arrow(ax, (7.5, 7.35), (7.9, 7.35))
    arrow(ax, (8.7, 6.9), (1.6, 6.2), color=MUTED, rad=0.25, ls=":", style_="-|>")
    ax.text(5.0, 6.25, "loop until prompt consumed", fontsize=8.0, color=MUTED, ha="center")

    box(ax, 1.7, 4.05, 6.6, 0.8,
        "PREFILL → DECODE boundary:  mx.eval · mx.clear_cache · resolve decode policy\n"
        "(compressed → drop native KVCache; decode footprint = DiffKV store only)",
        fc=BLUE, ec=INK, tc="white", fs=8.6, bold=True)
    arrow(ax, (5.0, 5.2), (5.0, 4.85), lw=2.2)

    ax.add_patch(Rectangle((0.3, 0.5), 9.4, 3.0, fc=PANEL, ec=GRID, lw=1, zorder=0))
    ax.text(0.5, 3.3, "DECODE  (per token, per layer)", fontsize=10.5, color=INK, fontweight="bold")
    arrow(ax, (5.0, 4.05), (5.0, 3.5), lw=2.2)
    box(ax, 0.6, 1.7, 2.0, 1.0, "ingest current\ntoken → dense\nbuffer (+ self)", fc=WHITE, fs=8.2)
    box(ax, 3.0, 1.7, 2.1, 1.0, "flush eligible\nblock if dense\n> W+B", fc=WHITE, fs=8.2)
    box(ax, 5.5, 1.7, 2.2, 1.0, "route top-K →\nlow-rank + exact\nresidual + dense", fc=BLUE, tc="white", fs=8.0, bold=True)
    box(ax, 8.0, 1.7, 1.4, 1.0, "logits →\nsample", fc=WHITE, fs=8.2)
    arrow(ax, (2.6, 2.2), (3.0, 2.2)); arrow(ax, (5.1, 2.2), (5.5, 2.2)); arrow(ax, (7.7, 2.2), (8.0, 2.2))
    arrow(ax, (8.7, 1.7), (1.6, 1.0), color=MUTED, rad=0.25, ls=":", style_="-|>")
    ax.text(5.0, 1.0, "loop for each new token", fontsize=8.0, color=MUTED, ha="center")

    style.finalize(fig, os.path.join(FIG, "f4_lifecycle.png"))


# ───────────────────────── F5: fused decode attention ───────────────────────
def f5_decode():
    fig, ax = plt.subplots(figsize=(11, 7.0)); clean(ax)
    ax.set_ylim(0, 10)
    title(ax, "Fused Routed Decode Attention   (compute_decode_attention_static, mx.compile)",
          "Route top-K by EXACT residual-key score → low-rank scoring (no decompression) + exact residuals + recency → LSE merge")

    box(ax, 4.1, 9.0, 1.8, 0.6, "query q  [H, d]", fc=INK, tc="white", fs=9.0, bold=True)
    arrow(ax, (5.0, 9.0), (5.0, 8.5), lw=2.0)

    box(ax, 2.3, 7.7, 5.4, 0.75,
        "Router:  ρ$_b$ = max( q·a$_k$ ,  max$_j$ q·R$_{K,j}$ )·scale   →   keep top-K = 16 blocks",
        fc=AMBER, ec=INK, tc="white", fs=8.6, bold=True)
    ax.text(8.0, 8.05, "exact q·k over\nanchor + residual keys", fontsize=7.4, color=MUTED, ha="left", va="center")
    arrow(ax, (3.0, 7.7), (2.6, 6.9), rad=-0.12); arrow(ax, (7.0, 7.7), (7.4, 6.9), rad=0.12)

    ax.add_patch(Rectangle((0.3, 1.9), 4.7, 5.0, fc=PANEL, ec=GRID, lw=1, zorder=0))
    ax.text(0.5, 6.75, "Low-rank branch  (selected blocks)", fontsize=9.8, color=INK, fontweight="bold")
    box(ax, 0.6, 5.7, 4.1, 0.7, "anchor score   s$_{anc}$ = (q·a$_k$)·scale", fc=WHITE, fs=8.4)
    box(ax, 0.6, 4.8, 4.1, 0.7, "project query   q̃ = (q·V$_K$)·scale ∈ ℝ¹⁶", fc=WHITE, fs=8.4)
    box(ax, 0.6, 3.9, 4.1, 0.7, "delta scores   δs = (q̃·Uᵀ)·s + s$_{anc}$", fc=BLUE, tc="white", fs=8.4, bold=True)
    box(ax, 0.6, 3.0, 4.1, 0.7, "per-block softmax → w, lse$_{sparse}$", fc=WHITE, fs=8.4)
    box(ax, 0.6, 2.1, 4.1, 0.7, "O$_{sp}$ = Σw·a$_v$ + (Σw·U)·s·V$_V$", fc=WHITE, fs=8.4)
    for y0, y1 in [(5.7, 5.5), (4.8, 4.6), (3.9, 3.7), (3.0, 2.8)]:
        arrow(ax, (2.65, y0), (2.65, y1), lw=1.4)
    ax.text(2.65, 1.55, "O(K·r·B) — no O(B·d) decompression", fontsize=7.6, color=BAD, ha="center")

    ax.add_patch(Rectangle((5.0, 1.9), 4.7, 5.0, fc=PANEL, ec=GRID, lw=1, zorder=0))
    ax.text(5.2, 6.75, "Exact branch  (residuals + recency)", fontsize=9.8, color=INK, fontweight="bold")
    box(ax, 5.3, 5.3, 4.1, 0.9, "concat  selected blocks' exact\nresiduals R$_K$,R$_V$  ⊕  dense window", fc=WHITE, ec=BAD, fs=8.4)
    box(ax, 5.3, 3.9, 4.1, 0.9, "exact attention over\naugmented K̃, Ṽ  (full fp16)", fc=BAD, tc="white", fs=8.4, bold=True)
    box(ax, 5.3, 3.0, 4.1, 0.7, "softmax → out$_{dn}$, lse$_{dn}$", fc=WHITE, fs=8.4)
    arrow(ax, (7.35, 5.3), (7.35, 4.8), lw=1.4); arrow(ax, (7.35, 3.9), (7.35, 3.7), lw=1.4)
    ax.text(7.35, 2.65, "verbatim recall of needle tokens", fontsize=7.6, color=BAD, ha="center")

    box(ax, 3.2, 0.55, 3.6, 0.95,
        "flash-style LSE merge\nm = max(lse$_{sp}$, lse$_{dn}$)\nout = (e$^{…}$O$_{sp}$ + e$^{…}$O$_{dn}$)/Σ",
        fc=INK, tc="white", fs=8.4, bold=True)
    arrow(ax, (2.65, 2.1), (3.6, 1.5), color=BLUE, rad=-0.15, lw=2.0)
    arrow(ax, (7.35, 3.0), (6.4, 1.5), color=BAD, rad=0.15, lw=2.0)
    ax.text(5.0, 0.22, "NaN/inf-guarded so an empty compressed set contributes exactly zero weight",
            fontsize=7.6, color=MUTED, ha="center")

    style.finalize(fig, os.path.join(FIG, "f5_decode_attention.png"))


# ───────────────────────── F6: 3D memory architecture ───────────────────────
def f6_memory_3d():
    """Clean axonometric view of the per-session store.

    Left: the 28-layer stack as receding plates (the store is replicated per
    layer). Right: one layer's compressed block pool as a row of stacked cuboids
    — a short blue low-rank base + a tall red residual cap — with the dense
    recency window as a distinct slab in front. All descriptive text lives in the
    empty margins (a 2D legend + corner callouts), never over the 3D cluster.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(11.6, 7.2))
    ax = fig.add_axes([0.0, -0.04, 1.0, 0.86], projection="3d")
    ax.set_proj_type("ortho")

    def cuboid(o, size, color, alpha=1.0, ec=INK, lw=0.6, z=1):
        x, y, zc = o; dx, dy, dz = size
        pts = np.array([[x, y, zc], [x+dx, y, zc], [x+dx, y+dy, zc], [x, y+dy, zc],
                        [x, y, zc+dz], [x+dx, y, zc+dz], [x+dx, y+dy, zc+dz], [x, y+dy, zc+dz]])
        faces = [[pts[j] for j in f] for f in
                 ([0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4])]
        pc = Poly3DCollection(faces, facecolors=color, edgecolors=ec, linewidths=lw, alpha=alpha)
        pc.set_zsort("max"); pc.set_zorder(z)
        ax.add_collection3d(pc)

    # ── 1) layer stack: receding plates on the LEFT (y large = far/back) ──────
    n_shown = 5
    for i in range(n_shown):
        yy = 3.1 * i
        tint = 0.90 - 0.10 * (n_shown - 1 - i) / n_shown
        col = (tint - 0.08, tint - 0.02, 0.99)
        cuboid((0.0, yy, 0.0), (2.0, 2.4, 0.16), col, alpha=0.9, ec=SLATE, lw=0.6, z=i)
    ax.text(1.0, 3.1 * (n_shown - 1) + 1.9, 0.0, "layer 0", color=MUTED, fontsize=8.5,
            ha="center", zdir=None)
    ax.text(1.0, -1.7, 0.0, "layer 27", color=MUTED, fontsize=8.5, ha="center")

    # ── 2) one layer's compressed block pool: stacked cuboids on the RIGHT ────
    #    heights proportional to the per-block byte budget (R=128):
    #    low-rank 26 KiB, residuals 128 KiB.
    x0 = 4.2
    z_lr, z_res = 0.9, 4.4
    bw, gap, depth = 0.72, 0.30, 2.0
    y_pool = 0.0
    n_used, n_ghost = 8, 3
    for b in range(n_used + n_ghost):
        x = x0 + b * (bw + gap)
        used = b < n_used
        cuboid((x, y_pool, 0.0), (bw, depth, z_lr), BLUE,
               alpha=(0.97 if used else 0.22), ec=(INK if used else SLATE), lw=0.7, z=30 + b)
        cuboid((x, y_pool, z_lr), (bw, depth, z_res), BAD,
               alpha=(0.90 if used else 0.18), ec=(INK if used else SLATE), lw=0.7, z=30 + b)

    # ── 3) dense recency window: one distinct slab in FRONT of the pool ───────
    cuboid((x0, -3.0, 0.0), ((bw + gap) * 6.4, 1.7, 1.0), SKY, alpha=0.92, ec=INK, lw=0.9, z=80)

    # ── 4) flush arrow: dense window -> pool ──────────────────────────────────
    ax.plot([x0 + 1.6, x0 + 1.6], [-1.3, -0.05], [0.5, 0.5], color=BAD, lw=2.2, zorder=90)

    ax.set_xlim(0, x0 + (n_used + n_ghost) * (bw + gap) + 0.4)
    ax.set_ylim(-3.2, 3.1 * (n_shown - 1) + 2.5)
    ax.set_zlim(0, z_lr + z_res + 0.4)
    ax.view_init(elev=23, azim=-60)
    ax.set_axis_off()
    ax.set_box_aspect((11.5, 11.0, 4.0), zoom=1.42)

    # ── in-scene callouts, placed in EMPTY regions only ───────────────────────
    fig.text(0.115, 0.50, "28-layer stack\n($\\times$28 — store\nreplicated per layer)", ha="center",
             va="center", fontsize=9.2, color=INK, fontweight="bold", linespacing=1.4)
    fig.text(0.66, 0.775, "Compressed block pool  ·  256 blocks (bounded)", ha="center",
             fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.92, 0.52, "used\nblocks", ha="center", va="center", fontsize=8.6, color=INK)
    fig.text(0.965, 0.40, "free\n(zeroed)", ha="center", va="center", fontsize=8.4, color=MUTED)
    fig.text(0.55, 0.11, "Dense recency window — 768 exact fp16 tokens", ha="center",
             fontsize=9.6, color=INK, fontweight="bold")
    fig.text(0.375, 0.30, "overflow →\nflush + compress", ha="center", fontsize=8.4,
             color=BAD, fontweight="bold", linespacing=1.3)

    # ── clean 2D legend strip, in the empty band under the subtitle ───────────
    leg = [Patch(fc=BLUE, ec=INK, label="Low-rank core  U, V$_K$, V$_V$, anchors, min/max  ($\\approx$26 KiB)"),
           Patch(fc=BAD, ec=INK, label="Exact residuals  R$_K$, R$_V$  ($\\approx$128 KiB, $R{=}128$)"),
           Patch(fc=SKY, ec=INK, label="Dense recency window  (uncompressed fp16)")]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, fontsize=8.6, frameon=False, columnspacing=1.6, handlelength=1.4)

    fig.text(0.5, 0.975, "Per-Session KV Store — 3D Memory Architecture",
             ha="center", fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.935,
             "State is replicated across 28 layers, organized per block, and pool-bounded; "
             "exact residuals dominate each block's byte budget at the default setting",
             ha="center", fontsize=10.0, color=MUTED)
    style.watermark(fig)
    fig.savefig(os.path.join(FIG, "f6_memory_3d.png"), dpi=220)
    fig.savefig(os.path.join(FIG, "f6_memory_3d.pdf"))
    plt.close(fig)
    print("wrote", os.path.join(FIG, "f6_memory_3d.png"))


if __name__ == "__main__":
    f1_architecture()
    f2_compression()
    f3_memory_layout()
    f4_lifecycle()
    f5_decode()
    f6_memory_3d()
    print("diagrams done")
