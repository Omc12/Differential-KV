#!/usr/bin/env python3
"""Generate original publication-quality architecture / dataflow diagrams (F1-F5).

Pure matplotlib vector drawing -> PNG + PDF, consistent visual identity (style.py).
Run: python paper/scripts/make_diagrams.py   (outputs to paper/figures/)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import INK, ACCENT, ACCENT2, GOOD, BAD, GRID, PANEL, MUTED, WHITE
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

style.apply_rc()
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)


def box(ax, x, y, w, h, text, fc=WHITE, ec=INK, tc=INK, fs=10, lw=1.6, bold=False,
        round=0.02, align="center", style_="round"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={round}",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, fontweight="bold" if bold else "normal",
            linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=INK, lw=1.8, style_="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style_, mutation_scale=14,
                        color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                        zorder=2, linestyle=ls)
    ax.add_patch(a)


def title(ax, t, sub=None):
    ax.text(0.5, 0.985, t, transform=ax.transAxes, ha="center", va="top",
            fontsize=15, fontweight="bold", color=INK)
    if sub:
        ax.text(0.5, 0.94, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color=MUTED)


def clean(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


# ───────────────────────── F1: system architecture ──────────────────────────
def f1_architecture():
    fig, ax = plt.subplots(figsize=(11, 7.2)); clean(ax)
    title(ax, "DiffKV Active Runtime — System Architecture",
          "Serving stack (top) drives the MLX inference core (bottom); the patched attention routes prefill vs. decode")

    # serving lane
    box(ax, 0.3, 8.05, 9.4, 0.5, "", fc=PANEL, ec=GRID, lw=1)
    ax.text(0.5, 8.3, "Serving", fontsize=9, color=MUTED, fontstyle="italic", va="center")
    box(ax, 1.6, 8.08, 2.2, 0.44, "OpenAI-compatible\nAPI gateway (FastAPI)", fc=WHITE, fs=8.4)
    box(ax, 4.0, 8.08, 2.0, 0.44, "Continuous-batching\ndecode engine", fc=WHITE, fs=8.4)
    box(ax, 6.2, 8.08, 2.3, 0.44, "Session manager\n(LRU residency, lifecycle)", fc=WHITE, fs=8.4)
    arrow(ax, (3.8, 8.3), (4.0, 8.3)); arrow(ax, (6.0, 8.3), (6.2, 8.3))

    # wrapper
    box(ax, 1.4, 6.85, 7.2, 0.7,
        "MLXDiffKVWrapper  ·  generate() / chunked prefill + greedy/sampled decode loop",
        fc=INK, ec=INK, tc=WHITE, fs=9.5, bold=True)
    arrow(ax, (5.0, 8.05), (5.0, 7.57), lw=2.0)

    # patched model
    box(ax, 1.4, 5.75, 7.2, 0.62,
        "MLXQwenModel  ·  per-session native KVCache (prefill) + prefill→decode memory release",
        fc=WHITE, fs=9.2)
    arrow(ax, (5.0, 6.85), (5.0, 6.37), lw=2.0)

    # patched attention (router)
    box(ax, 3.0, 4.55, 4.0, 0.7, "Patched Qwen2 Attention\n(monkey-patched __call__)",
        fc=ACCENT, ec=INK, tc="white", fs=9.6, bold=True)
    arrow(ax, (5.0, 5.75), (5.0, 5.25), lw=2.0)

    # two routes
    arrow(ax, (4.0, 4.55), (2.6, 3.75), color=INK, rad=-0.15)
    arrow(ax, (6.0, 4.55), (7.4, 3.75), color=INK, rad=0.15)
    ax.text(2.6, 4.2, "L>1  PREFILL", fontsize=8.5, color=INK, ha="center", fontweight="bold")
    ax.text(7.4, 4.2, "L==1  DECODE", fontsize=8.5, color=INK, ha="center", fontweight="bold")

    # prefill column
    box(ax, 0.7, 2.95, 3.7, 0.7, "Exact causal SDPA over native cache\n→ capture K/V into dense buffer",
        fc=PANEL, fs=8.6)
    box(ax, 0.7, 2.0, 3.7, 0.7, "Streaming flush+compress oldest block\n(anchor + low-rank SVD delta)",
        fc=PANEL, fs=8.6)
    arrow(ax, (2.55, 2.95), (2.55, 2.7))

    # decode column
    box(ax, 5.6, 2.95, 3.7, 0.7, "Ingest current token → dense buffer\n+ flush eligible block",
        fc=PANEL, fs=8.6)
    box(ax, 5.6, 2.0, 3.7, 0.7, "Fused compressed+dense attention\n(mx.compile, low-rank scoring)",
        fc=PANEL, fs=8.6)
    arrow(ax, (7.45, 2.95), (7.45, 2.7))

    # KV store (shared)
    box(ax, 1.9, 0.5, 6.2, 1.0,
        "MLXKVBlockManager — per-session KV store\n"
        "Dense recency window [H_kv, 768, d] fp16   +   Compressed pool {U, V_K, V_V, anchors, scale}×256 blocks",
        fc=INK, ec=INK, tc=WHITE, fs=9.0, bold=True)
    arrow(ax, (2.55, 2.0), (3.2, 1.5), color=MUTED, rad=-0.1)
    arrow(ax, (7.45, 2.0), (6.8, 1.5), color=MUTED, rad=0.1)
    arrow(ax, (4.2, 2.35), (4.2, 1.5), color=MUTED, ls=":", style_="-|>")
    ax.text(4.5, 1.75, "read (no decompress)", fontsize=7.6, color=MUTED, rotation=90, va="center")

    style.finalize(fig, os.path.join(FIG, "f1_architecture.png"))


# ───────────────────────── F2: compression pipeline ─────────────────────────
def f2_compression():
    fig, ax = plt.subplots(figsize=(11, 5.4)); clean(ax)
    ax.set_ylim(0, 8)
    title(ax, "Differential KV Compression — one 256-token block",
          "Anchor + per-token delta → joint K/V randomized truncated SVD (rank 16) → compact basis")

    y = 4.3
    # block of tokens
    bx = 0.4
    for i in range(8):
        c = ACCENT2 if i == 0 else ACCENT
        Rectangle  # noqa
        ax.add_patch(Rectangle((bx + i*0.34, y), 0.30, 0.7, fc=c, ec=INK, lw=1.0, zorder=3))
    ax.text(bx + 0.15, y - 0.32, "anchor\n(tok 0)", fontsize=7.4, color=INK, ha="center", va="top")
    ax.text(bx + 8*0.34/2 + 0.5, y + 1.05, "K,V block  [H_kv, 256, d]", fontsize=8.8, color=INK, ha="center")
    ax.text(bx + 4*0.34, y - 0.95, "256 tokens (… ×31 more …)", fontsize=7.6, color=MUTED, ha="center")

    arrow(ax, (3.35, y + 0.35), (3.95, y + 0.35), lw=2.0)

    # delta
    box(ax, 4.0, y - 0.05, 1.7, 0.8, "ΔK = K − a_k\nΔV = V − a_v", fc=PANEL, fs=8.6)
    arrow(ax, (5.7, y + 0.35), (6.2, y + 0.35), lw=2.0)

    # normalize + concat
    box(ax, 6.25, y - 0.05, 1.7, 0.8, "row L2-normalize\nconcat[ΔK|ΔV]", fc=PANEL, fs=8.6)
    arrow(ax, (7.95, y + 0.35), (8.45, y + 0.35), lw=2.0)

    # rSVD
    box(ax, 8.5, y - 0.05, 1.4, 0.8, "randomized\ntrunc. SVD", fc=ACCENT, tc="white", fs=8.6, bold=True)

    # outputs row
    yo = 1.7
    arrow(ax, (9.2, y - 0.05), (9.2, yo + 0.95), rad=0.0, lw=2.0)
    ax.text(9.55, 3.0, "U Σ Vᵀ", fontsize=8.4, color=INK, va="center")
    outs = [
        ("U  [255, 16]", "per-token\ncoefficients", ACCENT),
        ("V_K [H_kv,16,d]", "K basis", INK),
        ("V_V [H_kv,16,d]", "V basis", INK),
        ("anchor a_k,a_v", "[H_kv, d]", ACCENT2),
        ("scale s", "fp32", MUTED),
    ]
    ox = 0.7
    for i, (t, sub, c) in enumerate(outs):
        box(ax, ox + i*1.85, yo, 1.65, 0.95, t + "\n" + sub, fc=WHITE, ec=c, tc=INK, fs=8.2)
    # brace text
    ax.text(5.0, 0.95, "stored compressed block  ≈ 25 KiB   vs   256 KiB dense   →   ≈ 10.3× smaller",
            fontsize=9.6, color=INK, ha="center", fontweight="bold")
    ax.text(5.0, 0.45, "reconstruction (implicit):  K̂_i ≈ a_k + s·(U_i · V_K)     "
                       "— never materialized as dense KV", fontsize=8.4, color=MUTED, ha="center")
    for i in range(len(outs)):
        arrow(ax, (9.2, yo + 1.6), (ox + i*1.85 + 0.8, yo + 0.95), color=MUTED, lw=1.0, rad=0.0, style_="-")

    style.finalize(fig, os.path.join(FIG, "f2_compression.png"))


# ───────────────────────── F3: session memory layout ────────────────────────
def f3_memory_layout():
    fig, ax = plt.subplots(figsize=(11, 6.0)); clean(ax)
    ax.set_ylim(0, 9)
    title(ax, "Per-Session KV Store — memory layout (one layer of 28)",
          "Bounded dense recency window + fixed compressed pool — the size that does NOT grow with context")

    # compressed pool
    ax.text(0.5, 7.7, "Compressed pool  (fixed M = 256 blocks  ·  ≤ 65 536 tokens)",
            fontsize=10.5, color=INK, fontweight="bold")
    px, py, pw = 0.5, 5.0, 9.0
    box(ax, px, py, pw, 2.4, "", fc=PANEL, ec=GRID, lw=1)
    cols = ["U\n[255,16]", "V_K\n[2,16,128]", "V_V\n[2,16,128]", "a_k\n[2,128]", "a_v\n[2,128]", "scale", "seq_len"]
    cw = 1.18
    for i, c in enumerate(cols):
        fc = ACCENT if i == 0 else (INK if i in (1, 2) else (ACCENT2 if i in (3, 4) else WHITE))
        tc = "white" if fc in (INK, ACCENT, ACCENT2) else INK
        box(ax, px + 0.25 + i*cw, py + 1.3, cw - 0.18, 0.9, c, fc=fc, tc=tc, fs=8.0)
    ax.text(px + pw/2, py + 0.6, "block 0   block 1   …   block k   ·   (k+1 … 255 pre-allocated, zeroed)",
            fontsize=8.6, color=MUTED, ha="center")
    ax.text(px + pw + 0.0, py + 2.2, "≈25 KiB\n/block", fontsize=7.6, color=INK, ha="left", va="top")

    # dense window
    ax.text(0.5, 4.3, "Dense recency window  (sliding, capacity W+B = 768 tokens, uncompressed fp16)",
            fontsize=10.5, color=INK, fontweight="bold")
    dx, dy, dw = 0.5, 2.5, 9.0
    box(ax, dx, dy, dw, 1.3, "", fc=PANEL, ec=GRID, lw=1)
    n = 24
    for i in range(n):
        filled = i < 18
        ax.add_patch(Rectangle((dx + 0.2 + i*((dw-0.4)/n), dy + 0.35), (dw-0.4)/n - 0.04, 0.6,
                               fc=ACCENT if filled else WHITE, ec=INK, lw=0.8))
    ax.text(dx + dw*0.45, dy + 0.15, "dense_len (live)", fontsize=8.0, color=INK, ha="center", va="top")
    ax.text(dx + dw*0.86, dy + 0.15, "free", fontsize=8.0, color=MUTED, ha="center", va="top")

    # flush arrow
    arrow(ax, (dx + 0.6, dy + 1.3), (px + 0.8, py), color=BAD, rad=-0.25, lw=2.0)
    ax.text(1.0, 4.75, "overflow → flush+compress\noldest block", fontsize=8.2, color=BAD, ha="left")

    # footprint note
    box(ax, 0.5, 0.5, 9.0, 1.4,
        "Footprint per layer ≈  dense 768·H_kv·d·2·2 B  +  used_blocks·25 KiB.\n"
        "Dense term is constant; compressed term grows at ≈ 1/10 the dense-KV rate → the flat memory slope.",
        fc=WHITE, ec=INK, fs=9.2)

    style.finalize(fig, os.path.join(FIG, "f3_memory_layout.png"))


# ───────────────────────── F4: cache lifecycle / flow ───────────────────────
def f4_lifecycle():
    fig, ax = plt.subplots(figsize=(11, 6.2)); clean(ax)
    ax.set_ylim(0, 9)
    title(ax, "Cache Lifecycle & Execution Flow",
          "Chunked prefill (exact) → streaming compression → memory release → fused sparse decode")

    # PREFILL band
    ax.add_patch(Rectangle((0.3, 5.2), 9.4, 3.1, fc=PANEL, ec=GRID, lw=1, zorder=0))
    ax.text(0.5, 8.1, "PREFILL  (chunk = 512 tokens, repeated)", fontsize=10.5, color=INK, fontweight="bold")
    box(ax, 0.6, 6.9, 2.0, 0.9, "chunk forward\n(native KVCache)\nexact causal SDPA", fc=WHITE, fs=8.2)
    box(ax, 3.0, 6.9, 2.0, 0.9, "capture chunk\nK/V → dense buffer", fc=WHITE, fs=8.2)
    box(ax, 5.4, 6.9, 2.1, 0.9, "dense full?\nflush+compress\noldest block", fc=WHITE, fs=8.2)
    box(ax, 7.9, 6.9, 1.6, 0.9, "next chunk", fc=WHITE, ec=MUTED, tc=MUTED, fs=8.2)
    arrow(ax, (2.6, 7.35), (3.0, 7.35)); arrow(ax, (5.0, 7.35), (5.4, 7.35))
    arrow(ax, (7.5, 7.35), (7.9, 7.35))
    arrow(ax, (8.7, 6.9), (1.6, 6.2), color=MUTED, rad=0.25, ls=":", style_="-|>")
    ax.text(5.0, 6.25, "loop until prompt consumed", fontsize=8.0, color=MUTED, ha="center")

    # BOUNDARY
    box(ax, 2.0, 4.05, 6.0, 0.8,
        "PREFILL → DECODE boundary:  mx.eval · mx.clear_cache · drop native prefill KVCache\n"
        "→ release peak GQA activations; decode footprint = DiffKV store only",
        fc=ACCENT, ec=INK, tc="white", fs=8.8, bold=True)
    arrow(ax, (5.0, 5.2), (5.0, 4.85), lw=2.2)

    # DECODE band
    ax.add_patch(Rectangle((0.3, 0.5), 9.4, 3.0, fc=PANEL, ec=GRID, lw=1, zorder=0))
    ax.text(0.5, 3.3, "DECODE  (per token, per layer)", fontsize=10.5, color=INK, fontweight="bold")
    arrow(ax, (5.0, 4.05), (5.0, 3.5), lw=2.2)
    box(ax, 0.6, 1.7, 2.0, 1.0, "ingest current\ntoken → dense\nbuffer (+self)", fc=WHITE, fs=8.2)
    box(ax, 3.0, 1.7, 2.1, 1.0, "flush eligible\nblock if dense\n> W+B", fc=WHITE, fs=8.2)
    box(ax, 5.5, 1.7, 2.2, 1.0, "fused attention:\nlow-rank sparse\n+ dense, LSE merge", fc=ACCENT, tc="white", fs=8.2, bold=True)
    box(ax, 8.0, 1.7, 1.4, 1.0, "logits →\nsample", fc=WHITE, fs=8.2)
    arrow(ax, (2.6, 2.2), (3.0, 2.2)); arrow(ax, (5.1, 2.2), (5.5, 2.2)); arrow(ax, (7.7, 2.2), (8.0, 2.2))
    arrow(ax, (8.7, 1.7), (1.6, 1.0), color=MUTED, rad=0.25, ls=":", style_="-|>")
    ax.text(5.0, 1.0, "loop for each new token", fontsize=8.0, color=MUTED, ha="center")

    style.finalize(fig, os.path.join(FIG, "f4_lifecycle.png"))


# ───────────────────────── F5: fused decode attention ───────────────────────
def f5_decode():
    fig, ax = plt.subplots(figsize=(11, 6.4)); clean(ax)
    ax.set_ylim(0, 9)
    title(ax, "Fused Sparse Decode Attention  (compute_decode_attention_static, mx.compile)",
          "Query scored in low-rank space — KV is never decompressed; two branches merged by log-sum-exp")

    # query
    box(ax, 4.1, 7.9, 1.8, 0.7, "query q  [H, d]", fc=INK, tc="white", fs=9.2, bold=True)
    arrow(ax, (4.6, 7.9), (2.7, 7.1), rad=-0.12); arrow(ax, (5.4, 7.9), (7.4, 7.1), rad=0.12)

    # SPARSE branch
    ax.add_patch(Rectangle((0.3, 1.9), 4.7, 5.0, fc=PANEL, ec=GRID, lw=1, zorder=0))
    ax.text(0.5, 6.75, "Sparse / compressed branch", fontsize=10.2, color=INK, fontweight="bold")
    box(ax, 0.6, 5.7, 4.1, 0.7, "anchor score   s_anc = (q·a_k)·scale", fc=WHITE, fs=8.6)
    box(ax, 0.6, 4.8, 4.1, 0.7, "project query   q̃ = (q·V_K)·scale  ∈ ℝ¹⁶", fc=WHITE, fs=8.6)
    box(ax, 0.6, 3.9, 4.1, 0.7, "delta scores   δs = (q̃·Uᵀ)·s + s_anc", fc=ACCENT, tc="white", fs=8.6, bold=True)
    box(ax, 0.6, 3.0, 4.1, 0.7, "mask (seq_len, num_blocks) · per-block softmax", fc=WHITE, fs=8.6)
    box(ax, 0.6, 2.1, 4.1, 0.7, "value   O = Σw·a_v + (Σw·U)·s·V_V  →  lse_sparse", fc=WHITE, fs=8.6)
    for y0, y1 in [(5.7, 5.5), (4.8, 4.6), (3.9, 3.7), (3.0, 2.8)]:
        arrow(ax, (2.65, y0), (2.65, y1), lw=1.5)
    ax.text(2.65, 1.55, "O(r·B) per block — no O(B·d) decompression", fontsize=7.8, color=BAD, ha="center")

    # DENSE branch
    ax.add_patch(Rectangle((5.0, 1.9), 4.7, 5.0, fc=PANEL, ec=GRID, lw=1, zorder=0))
    ax.text(5.2, 6.75, "Dense recency branch", fontsize=10.2, color=INK, fontweight="bold")
    box(ax, 5.3, 4.8, 4.1, 0.9, "exact attention over\ndense window K,V  (≤768 tok)", fc=WHITE, fs=8.6)
    box(ax, 5.3, 3.4, 4.1, 0.9, "softmax · weighted sum\n→ out_dense, lse_dense", fc=WHITE, fs=8.6)
    arrow(ax, (7.35, 4.8), (7.35, 4.3), lw=1.5)

    # MERGE
    box(ax, 3.2, 0.55, 3.6, 0.95,
        "flash-style LSE merge\nw = softmax([lse_sparse, lse_dense])\nout = w_s·O_sparse + w_d·O_dense",
        fc=INK, tc="white", fs=8.6, bold=True)
    arrow(ax, (2.65, 2.1), (3.6, 1.5), color=ACCENT, rad=-0.15, lw=2.0)
    arrow(ax, (7.35, 3.4), (6.4, 1.5), color=ACCENT, rad=0.15, lw=2.0)
    ax.text(5.0, 0.25, "NaN/inf-guarded so a fully-masked branch contributes exactly zero weight",
            fontsize=7.8, color=MUTED, ha="center")

    style.finalize(fig, os.path.join(FIG, "f5_decode_attention.png"))


if __name__ == "__main__":
    f1_architecture()
    f2_compression()
    f3_memory_layout()
    f4_lifecycle()
    f5_decode()
    print("diagrams done")
