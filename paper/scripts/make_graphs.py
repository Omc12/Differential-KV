#!/usr/bin/env python3
"""Generate publication-quality DATA graphs (G1-G5) from measured JSON.

Inputs:
  paper/generated/active_modes_sweep.json   (measure_active.py: compressed + exact)
  benchmarks/results/PAPER_dense_sweep.json  (run_bench.py: dense full-KV)

Outputs -> paper/figures/g*.png + .pdf
Only measured cells are plotted; OOM/missing cells are annotated, never invented.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import INK, ACCENT, ACCENT2, GOOD, BAD, MUTED, PANEL
import matplotlib.pyplot as plt
import numpy as np

style.apply_rc()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "..", "figures")
GEN = os.path.join(HERE, "..", "generated")
os.makedirs(FIG, exist_ok=True)

ACTIVE_JSON = os.path.join(REPO, "paper/generated/active_modes_sweep.json")
DENSE_JSON = os.path.join(REPO, "benchmarks/results/PAPER_dense_sweep.json")


def load_active():
    if not os.path.exists(ACTIVE_JSON):
        return {}
    d = json.load(open(ACTIVE_JSON))["results"]
    out = {"compressed": {}, "exact": {}}
    for r in d:
        if r.get("status") == "error" or "decode_tps" not in r:
            continue
        out.setdefault(r["mode"], {})[r["ctx"]] = r
    return out


def load_dense():
    if not os.path.exists(DENSE_JSON):
        return {}
    d = json.load(open(DENSE_JSON))["results"]
    out = {}
    for r in d:
        if r.get("status") == "ok":
            out[r["ctx_target"]] = r
    return out


def series(dct, ctxs, key, sub=None):
    xs, ys = [], []
    for c in ctxs:
        r = dct.get(c)
        if r is None:
            continue
        v = r[sub][key] if sub else r.get(key)
        if v is None:
            continue
        xs.append(c); ys.append(v)
    return xs, ys


CTXS = [4096, 8192, 16384, 32768, 65536]


def g1_kv_footprint(A, D):
    """Headline: DiffKV KV-state footprint is bounded; dense full-KV grows linearly."""
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    comp = A.get("compressed", {})
    # store used + store alloc (bounded) — same store regardless of decode mode
    xs, used = series(comp, CTXS, "store_used_bytes", sub="kv")
    _, alloc = series(comp, CTXS, "store_alloc_bytes", sub="kv")
    _, densef = series(comp, CTXS, "dense_full_bytes", sub="kv")
    used = [v/1e9 for v in used]; alloc = [v/1e9 for v in alloc]; densef = [v/1e9 for v in densef]
    if xs:
        ax.plot(xs, densef, "-o", color=ACCENT2, label="Dense full KV-cache (analytic)")
        ax.plot(xs, used, "-o", color=INK, label="DiffKV store, occupied")
        ax.plot(xs, alloc, "--", color=ACCENT, label="DiffKV store, bounded capacity")
    ax.set_yscale("log"); style.context_ticks(ax, xs or CTXS)
    ax.set_ylabel("KV-cache state (GB, all 28 layers)")
    ax.set_title("KV-state footprint vs context length")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04, "compression is in the KV state:\nbounded pool vs linear growth",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTED)
    style.finalize(fig, os.path.join(FIG, "g1_kv_footprint.png"))


def g2_mx_peak(A, D):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    comp = A.get("compressed", {}); exa = A.get("exact", {})
    xs1, y1 = series(comp, CTXS, "mx_decode_peak_gb")
    xs2, y2 = series(exa, CTXS, "mx_peak_gb")
    dx = sorted(D); dy = [D[c]["mx_peak_gb"] for c in dx]
    if xs1: ax.plot(xs1, y1, "-o", color=INK, label="DiffKV compressed (decode-phase peak)")
    if xs2: ax.plot(xs2, y2, "-s", color=ACCENT, label="DiffKV exact decode (global peak)")
    if dx: ax.plot(dx, dy, "-^", color=ACCENT2, label="Dense full KV (global peak)")
    # dense OOM marker at 64k
    if 65536 not in D and dx:
        ax.scatter([65536], [max(dy)*1.02], marker="x", s=120, color=BAD, zorder=5)
        ax.annotate("dense OOM", (65536, max(dy)*1.02), color=BAD, fontsize=9,
                    ha="center", va="bottom")
    style.context_ticks(ax, CTXS)
    ax.set_ylabel("MLX allocator peak (GB)")
    ax.set_title("Allocator peak memory vs context")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04, "peak is weights+prefill-dominated;\ncompression's win is reach, not peak",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTED)
    style.finalize(fig, os.path.join(FIG, "g2_mx_peak.png"))


def g3_decode_tps(A, D):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    comp = A.get("compressed", {}); exa = A.get("exact", {})
    xs1, y1 = series(comp, CTXS, "decode_tps")
    xs2, y2 = series(exa, CTXS, "decode_tps")
    dx = sorted(D); dy = [D[c]["decode_tps"] for c in dx]
    if dx: ax.plot(dx, dy, "-^", color=ACCENT2, label="Dense full KV")
    if xs2: ax.plot(xs2, y2, "-s", color=ACCENT, label="DiffKV exact decode (ablation)")
    if xs1: ax.plot(xs1, y1, "-o", color=INK, label="DiffKV compressed decode")
    style.context_ticks(ax, CTXS)
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Decode throughput vs context")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.04, "CUDA/Triton fused decode: future work\n(placeholder — not yet measured)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, color=MUTED, style="italic")
    style.finalize(fig, os.path.join(FIG, "g3_decode_tps.png"))


def g4_prefill(A, D):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    comp = A.get("compressed", {})
    xs1, y1 = series(comp, CTXS, "prefill_s")
    dx = sorted(D); dy = [D[c]["prefill_s"] for c in dx]
    if dx: ax.plot(dx, dy, "-^", color=ACCENT2, label="Dense (mlx_lm)")
    if xs1: ax.plot(xs1, y1, "-o", color=INK, label="DiffKV active (chunked + compress)")
    style.context_ticks(ax, CTXS); ax.set_yscale("log")
    ax.set_ylabel("Prefill time (s)")
    ax.set_title("Prefill time vs context")
    ax.legend(loc="upper left")
    style.finalize(fig, os.path.join(FIG, "g4_prefill.png"))


def g5_combined(A, D):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    comp = A.get("compressed", {}); exa = A.get("exact", {})
    dx = sorted(D)
    # panel 1: KV footprint
    ax = axes[0]
    xs, used = series(comp, CTXS, "store_used_bytes", sub="kv")
    _, alloc = series(comp, CTXS, "store_alloc_bytes", sub="kv")
    _, densef = series(comp, CTXS, "dense_full_bytes", sub="kv")
    if xs:
        ax.plot(xs, [v/1e9 for v in densef], "-o", color=ACCENT2, label="Dense full KV")
        ax.plot(xs, [v/1e9 for v in used], "-o", color=INK, label="DiffKV store (used)")
        ax.plot(xs, [v/1e9 for v in alloc], "--", color=ACCENT, label="DiffKV store (cap)")
    ax.set_yscale("log"); style.context_ticks(ax, xs or CTXS)
    ax.set_ylabel("KV state (GB)"); ax.set_title("(a) KV-state footprint"); ax.legend(fontsize=8.5)
    # panel 2: decode tps
    ax = axes[1]
    if dx: ax.plot(dx, [D[c]["decode_tps"] for c in dx], "-^", color=ACCENT2, label="Dense")
    x2, y2 = series(exa, CTXS, "decode_tps");  ax.plot(x2, y2, "-s", color=ACCENT, label="Exact (abl.)")
    x1, y1 = series(comp, CTXS, "decode_tps"); ax.plot(x1, y1, "-o", color=INK, label="Compressed")
    style.context_ticks(ax, CTXS); ax.set_ylabel("tok/s"); ax.set_title("(b) Decode throughput"); ax.legend(fontsize=8.5)
    # panel 3: prefill
    ax = axes[2]
    if dx: ax.plot(dx, [D[c]["prefill_s"] for c in dx], "-^", color=ACCENT2, label="Dense")
    x1, y1 = series(comp, CTXS, "prefill_s"); ax.plot(x1, y1, "-o", color=INK, label="DiffKV")
    style.context_ticks(ax, CTXS); ax.set_yscale("log"); ax.set_ylabel("s"); ax.set_title("(c) Prefill time"); ax.legend(fontsize=8.5)
    fig.suptitle("DiffKV active runtime — Qwen2.5-1.5B int4, Apple M3 (8.6 GB)", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    style.finalize(fig, os.path.join(FIG, "g5_combined.png"))


if __name__ == "__main__":
    A = load_active(); D = load_dense()
    print("active modes:", {k: sorted(v) for k, v in A.items()})
    print("dense ctxs:", sorted(D))
    g1_kv_footprint(A, D); g2_mx_peak(A, D); g3_decode_tps(A, D)
    g4_prefill(A, D); g5_combined(A, D)
    print("graphs done")
