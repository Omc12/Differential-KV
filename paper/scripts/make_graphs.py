#!/usr/bin/env python3
"""Regenerate every measured-data graph for the DiffKV paper.

All numbers come from paper/scripts/data.py (measured JSON) or are derived
analytically from the runtime dimensions in the code. Nothing is hand-typed.
Consistent DeepSeek-style blue identity from paper/scripts/style.py.

Graphs:
  G1  KV-state footprint vs context      (analytic; two presets vs dense)
  G2  Peak allocator memory vs context   (measured mx_peak, active vs dense)
  G3  Decode throughput vs context       (measured, active vs dense; +CUDA slot)
  G4  Prefill time vs context            (measured; sparse-prefill crossover)
  G5  Combined 4-panel system dashboard  (measured)
  G6  Residual accuracy/memory tradeoff  (measured residual sweep @16k)
  G7  Compressed-vs-exact decode ablation(measured; isolates decode cost)
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
CTX = D.CONTEXTS
GB = 1e9


def _xk(ctx):
    return f"{ctx//1024}k"


# ── G1 · KV-state footprint (analytic, bounded pool vs growing dense) ────────
def g1_kv_footprint():
    modes = D.modes_by("compressed")
    used64 = [modes[c]["kv"]["store_used_bytes"] / GB for c in CTX]
    dense  = [modes[c]["kv"]["dense_full_bytes"] / GB for c in CTX]
    nblk   = [modes[c]["kv"]["num_blocks_layer0"] for c in CTX]
    alloc  = modes[CTX[0]]["kv"]["store_alloc_bytes"] / GB
    extra_per_block = (128 - 64) * (2 * D.DIMS["kv_heads"] * D.DIMS["head_dim"] * 2)
    used128 = [(modes[c]["kv"]["store_used_bytes"]
                + D.DIMS["n_layers"] * nb * extra_per_block) / GB
               for c, nb in zip(CTX, nblk)]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(CTX, dense, "-o", color=S.C_DENSE, label="Dense full KV", zorder=3)
    ax.plot(CTX, used128, "-o", color=S.BLUE,
            label="DiffKV store  ($R{=}128$, default)", zorder=4)
    ax.plot(CTX, used64, "--o", color=S.SKY,
            label="DiffKV store  ($R{=}64$, memory preset)", zorder=4,
            markerfacecolor="white")
    ax.axhline(alloc, color=S.AMBER, lw=1.6, ls=":",
               label=f"Bounded pool cap ({alloc:.2f} GB, 256 blocks)")
    ax.fill_between(CTX, used128, dense, color=S.MIST, alpha=0.5, zorder=1)

    for c, u, dn in [(CTX[0], used128[0], dense[0]), (CTX[-1], used128[-1], dense[-1])]:
        ax.annotate(f"{dn/u:.2f}$\\times$", (c, (u*dn)**0.5), fontsize=9.5,
                    color=S.INK, fontweight="bold", ha="center")
    S.context_ticks(ax, CTX)
    ax.set_ylabel("KV-cache state (GB, all layers)")
    ax.set_title("KV-state footprint: bounded compressed store vs. growing dense cache")
    ax.set_ylim(0, max(dense) * 1.12)
    ax.legend(loc="upper left")
    S.style_axes(ax)
    S.finalize(fig, os.path.join(FIG, "g1_kv_footprint.png"))


# ── G2 · Peak allocator memory (measured) ────────────────────────────────────
def g2_mx_peak():
    prim = D.load_primary()
    a = [prim["active"][c]["mx_peak_gb"] for c in CTX]
    d = [prim["dense"][c]["mx_peak_gb"] for c in CTX]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(CTX, a, "-o", color=S.BLUE, label="DiffKV (active)")
    ax.plot(CTX, d, "-o", color=S.C_DENSE, label="Dense full KV")
    S.annotate_points(ax, CTX, a, "{:.2f}", dy=9, color=S.BLUE)
    S.annotate_points(ax, CTX, d, "{:.2f}", dy=-16, color=S.C_DENSE)
    S.context_ticks(ax, CTX)
    ax.set_ylabel("MLX allocator peak (GB)")
    ax.set_title("Measured allocator peak — a wash at 1.5B / 8\\,GB (peak set in prefill)")
    ax.set_ylim(0, max(a + d) * 1.16)
    ax.legend(loc="upper left")
    S.style_axes(ax)
    S.finalize(fig, os.path.join(FIG, "g2_mx_peak.png"))


# ── G3 · Decode throughput (measured) + CUDA placeholder slot ────────────────
def g3_decode_tps():
    prim = D.load_primary()
    a = [prim["active"][c]["decode_tps"] for c in CTX]
    d = [prim["dense"][c]["decode_tps"] for c in CTX]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(CTX, d, "-o", color=S.C_DENSE, label="Dense full KV")
    ax.plot(CTX, a, "-o", color=S.BLUE, label="DiffKV sparse decode")
    ax.fill_between(CTX, a, d, color=S.MIST, alpha=0.45)
    S.annotate_points(ax, CTX, a, "{:.1f}", dy=-16, color=S.BLUE)
    S.annotate_points(ax, CTX, d, "{:.1f}", dy=9, color=S.C_DENSE)
    ax.plot([], [], "s", color=S.C_CUDA, markerfacecolor="white",
            markeredgecolor=S.SLATE, label="CUDA fused decode (future work)")
    S.context_ticks(ax, CTX)
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_title("Decode throughput — the honest cost of sparse reconstruction")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(d) * 1.12)
    S.style_axes(ax)
    S.finalize(fig, os.path.join(FIG, "g3_decode_tps.png"))


# ── G4 · Prefill time (measured) with sparse-prefill crossover ───────────────
def g4_prefill():
    prim = D.load_primary()
    a = [prim["active"][c]["prefill_s"] for c in CTX]
    d = [prim["dense"][c]["prefill_s"] for c in CTX]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(CTX, d, "-o", color=S.C_DENSE, label="Dense (full attention, $O(L^2)$)")
    ax.plot(CTX, a, "-o", color=S.BLUE, label="DiffKV (sparse prefill, $O(L\\!\\cdot\\!K)$)")
    ax.set_yscale("log")
    ax.axvspan(32768, 65536, color=S.MIST, alpha=0.5, zorder=0)
    ax.annotate(f"64k: {a[-1]:.0f}s vs {d[-1]:.0f}s\n({d[-1]/a[-1]:.2f}$\\times$ faster)",
                (65536, a[-1]), textcoords="offset points", xytext=(-6, 18),
                ha="right", fontsize=9.5, color=S.INK, fontweight="bold")
    S.context_ticks(ax, CTX)
    ax.set_ylabel("Prefill time (s, log scale)")
    ax.set_title("Prefill scaling: sparse prefill crosses over to win at long context")
    ax.legend(loc="upper left")
    S.style_axes(ax)
    S.finalize(fig, os.path.join(FIG, "g4_prefill.png"))


# ── G5 · Combined system dashboard (measured) ────────────────────────────────
def g5_combined():
    prim = D.load_primary()
    a_pf = [prim["active"][c]["prefill_s"] for c in CTX]
    d_pf = [prim["dense"][c]["prefill_s"] for c in CTX]
    a_tp = [prim["active"][c]["decode_tps"] for c in CTX]
    d_tp = [prim["dense"][c]["decode_tps"] for c in CTX]
    a_mx = [prim["active"][c]["mx_peak_gb"] for c in CTX]
    d_mx = [prim["dense"][c]["mx_peak_gb"] for c in CTX]
    modes = D.modes_by("compressed")
    used = [modes[c]["kv"]["store_used_bytes"] / GB for c in CTX]
    dfull = [modes[c]["kv"]["dense_full_bytes"] / GB for c in CTX]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(CTX, d_pf, "-o", color=S.C_DENSE, label="Dense")
    ax1.plot(CTX, a_pf, "-o", color=S.BLUE, label="DiffKV")
    ax1.set_yscale("log"); ax1.set_ylabel("Prefill (s)")
    ax1.set_title("(a) Prefill time", loc="left")
    ax1.axvspan(32768, 65536, color=S.MIST, alpha=0.5, zorder=0)

    ax2.plot(CTX, d_tp, "-o", color=S.C_DENSE, label="Dense")
    ax2.plot(CTX, a_tp, "-o", color=S.BLUE, label="DiffKV")
    ax2.set_ylabel("Decode (tok/s)")
    ax2.set_title("(b) Decode throughput", loc="left")

    ax3.plot(CTX, d_mx, "-o", color=S.C_DENSE, label="Dense")
    ax3.plot(CTX, a_mx, "-o", color=S.BLUE, label="DiffKV")
    ax3.set_ylabel("Allocator peak (GB)")
    ax3.set_title("(c) Peak memory (measured)", loc="left")

    ax4.plot(CTX, dfull, "-o", color=S.C_DENSE, label="Dense full KV")
    ax4.plot(CTX, used, "-o", color=S.BLUE, label="DiffKV store ($R{=}64$)")
    ax4.fill_between(CTX, used, dfull, color=S.MIST, alpha=0.5)
    ax4.set_ylabel("KV state (GB)")
    ax4.set_title("(d) KV-state footprint (analytic)", loc="left")

    for ax in (ax1, ax2, ax3, ax4):
        S.context_ticks(ax, CTX); S.style_axes(ax); ax.legend(loc="best")
    fig.suptitle("DiffKV vs. dense — Qwen2.5-1.5B (int4), Apple M3 / 8.6\\,GB, greedy gen=128",
                 fontsize=13.5, fontweight="bold", color=S.INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    S.finalize(fig, os.path.join(FIG, "g5_combined.png"))


# ── G6 · Residual accuracy/memory tradeoff (measured) ────────────────────────
def g6_residual_tradeoff():
    rows = D.load_residual_sweep()
    R = [r["max_residual"] for r in rows]
    ratio = [r["kv"]["ratio_used_vs_dense"] for r in rows]
    tps = [r["decode_tps"] for r in rows]
    needle = [r["needle_found"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    first_ok = next(i for i, ok in enumerate(needle) if ok)
    ax.axvspan(min(R) - 2, R[first_ok] - 0.5, color="#FBE9EC", alpha=0.8, zorder=0)
    ax.axvspan(R[first_ok] - 0.5, max(R) + 4, color="#E8F5EE", alpha=0.8, zorder=0)
    ax.plot(R, ratio, "-o", color=S.BLUE, label="Compression ratio (vs dense)", zorder=4)
    ax.set_xlabel("Residual budget $R$ (exact tokens per block)")
    ax.set_ylabel("Compression ratio  ($\\times$ vs dense)", color=S.BLUE)
    ax.tick_params(axis="y", labelcolor=S.BLUE)
    for x, y, ok in zip(R, ratio, needle):
        ax.annotate(("✓" if ok else "✗"), (x, y),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=12, fontweight="bold",
                    color=(S.GOOD if ok else S.BAD))
    ax.annotate("needle lost", (min(R) + 1, max(ratio) * 0.80), color=S.BAD,
                fontsize=9.5, fontweight="bold")
    ax.annotate("exact recall", (R[first_ok] + 6, max(ratio) * 0.80), color=S.GOOD,
                fontsize=9.5, fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(R, tps, "--s", color=S.SLATE, label="Decode tok/s", markerfacecolor="white")
    ax2.set_ylabel("Decode throughput (tok/s)", color=S.SLATE)
    ax2.tick_params(axis="y", labelcolor=S.SLATE)
    ax2.grid(False)
    ax.set_title("Residual budget: the accuracy $\\leftrightarrow$ memory $\\leftrightarrow$ speed knob (16k)")
    ax.set_xticks(R)
    S.style_axes(ax)
    ax2.spines["top"].set_visible(False)
    S.finalize(fig, os.path.join(FIG, "g6_residual_tradeoff.png"))


# ── G7 · Compressed-vs-exact decode ablation (measured) ──────────────────────
def g7_decode_ablation():
    comp = D.modes_by("compressed")
    exact = D.modes_by("exact")
    ctx = [c for c in CTX if c in comp and c in exact]
    c_tps = [comp[c]["decode_tps"] for c in ctx]
    e_tps = [exact[c]["decode_tps"] for c in ctx]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(ctx)); w = 0.38
    b1 = ax.bar(x - w/2, e_tps, w, color=S.SKY, label="Exact decode over store (upper bound)")
    b2 = ax.bar(x + w/2, c_tps, w, color=S.BLUE, label="Compressed sparse decode")
    for bars, vals in ((b1, e_tps), (b2, c_tps)):
        for r, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (r.get_x() + r.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.5, color=S.INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([_xk(c) for c in ctx])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Decode cost of compression, same store ($R{=}64$) — both recover the needle")
    ax.legend(loc="upper right")
    S.style_axes(ax)
    S.finalize(fig, os.path.join(FIG, "g7_decode_ablation.png"))


if __name__ == "__main__":
    g1_kv_footprint()
    g2_mx_peak()
    g3_decode_tps()
    g4_prefill()
    g5_combined()
    g6_residual_tradeoff()
    g7_decode_ablation()
    print("\nAll graphs regenerated into", FIG)
