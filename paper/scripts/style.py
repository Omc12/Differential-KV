"""Shared publication visual identity for all DiffKV paper figures.

White background, dark-blue accents, light-blue highlights, restrained gray grid,
consistent typography. Pure matplotlib (no system LaTeX dependency; mathtext only).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

# ── palette ───────────────────────────────────────────────────────────────
INK      = "#1F3A5F"   # dark blue — primary lines, headings, strokes
ACCENT   = "#4F8FD0"   # light blue — highlights, secondary series
ACCENT2  = "#E1A140"   # amber — tertiary / contrast series (dense baseline)
GOOD     = "#2E8B57"   # green — success (needle found)
BAD      = "#C0392B"   # red — failure / OOM
GRID     = "#D9DEE6"   # light gray grid
PANEL    = "#F4F7FB"   # very light blue panel fill
MUTED    = "#6B7785"   # muted gray text
WHITE    = "#FFFFFF"

# Sequential accents for multi-series
SERIES = [INK, ACCENT, ACCENT2, GOOD, "#7B5EA7"]


def apply_rc():
    plt.rcParams.update({
        "figure.facecolor":  WHITE,
        "axes.facecolor":    WHITE,
        "savefig.facecolor": WHITE,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11.5,
        "axes.labelcolor":   INK,
        "axes.edgecolor":    "#9AA7B6",
        "axes.linewidth":    1.0,
        "axes.titlecolor":   INK,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        GRID,
        "grid.linewidth":    0.9,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.frameon":    False,
        "legend.fontsize":   10,
        "lines.linewidth":   2.4,
        "lines.markersize":  7,
        "figure.dpi":        150,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
    })


def context_ticks(ax, contexts):
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([f"{c//1024}k" for c in contexts])
    ax.set_xlabel("Context length (tokens)")


def finalize(fig, path, also_pdf=True):
    fig.savefig(path, dpi=200)
    if also_pdf and path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)
    print("wrote", path)
