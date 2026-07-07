"""Shared publication visual identity for all DiffKV paper figures.

DeepSeek-style scientific identity: white background, a deep-navy ink, a
signature blue for the DiffKV series, light-blue highlights, a muted slate for
the dense baseline, restrained gridlines, generous whitespace, consistent
typography. Pure matplotlib (Agg, mathtext only — no system LaTeX dependency).

Every figure in the paper imports from here so the whole document reads as one
coherent visual system.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401
from matplotlib.patches import FancyBboxPatch  # noqa: F401

# ── palette ─────────────────────────────────────────────────────────────────
INK      = "#0B2E5E"   # deep navy — primary strokes, headings, DiffKV series
NAVY     = "#0B2E5E"   # alias
BLUE     = "#2F6BE0"   # signature blue — DiffKV highlight / primary series
SKY      = "#6FA8F5"   # light blue — secondary / fills
MIST     = "#CFE0FA"   # very light blue — area fills, soft highlights
SLATE    = "#8A97A8"   # muted slate — dense baseline series
AMBER    = "#E0A03A"   # amber — tertiary contrast series (e.g. exact ablation)
GOOD     = "#2E9E6B"   # green — success (needle recovered)
BAD      = "#D1495B"   # red — failure / OOM
GRID     = "#E4E9F1"   # very light gray-blue grid
PANEL    = "#F5F8FD"   # panel fill
MUTED    = "#5C6B7E"   # muted gray text
EDGE     = "#B7C2D2"   # axis edge
WHITE    = "#FFFFFF"

# Backward-compatible aliases (older scripts used these names).
ACCENT  = BLUE
ACCENT2 = AMBER

# Ordered accents for multi-series plots.
SERIES = [BLUE, SLATE, AMBER, GOOD, "#7B5EA7", SKY]

# Semantic role → colour (used consistently across every figure).
C_DIFFKV = BLUE     # the DiffKV active runtime
C_DENSE  = SLATE    # the dense full-KV baseline
C_EXACT  = AMBER    # exact-decode-over-compressed-store ablation
C_CUDA   = "#B7C2D2"  # reserved CUDA placeholder (drawn hollow/hatched)


def apply_rc():
    plt.rcParams.update({
        "figure.facecolor":  WHITE,
        "axes.facecolor":    WHITE,
        "savefig.facecolor": WHITE,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "mathtext.fontset":  "dejavusans",
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11.5,
        "axes.labelweight":  "medium",
        "axes.labelcolor":   INK,
        "axes.edgecolor":    EDGE,
        "axes.linewidth":    1.1,
        "axes.titlecolor":   INK,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        GRID,
        "grid.linewidth":    1.0,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.frameon":    False,
        "legend.fontsize":   10,
        "lines.linewidth":   2.6,
        "lines.markersize":  7,
        "lines.markeredgewidth": 1.4,
        "figure.dpi":        150,
        "savefig.dpi":       220,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.06,
    })


def style_axes(ax):
    """Apply the shared spine/tick treatment to an axis."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
        ax.spines[s].set_linewidth(1.1)
    ax.tick_params(length=0)
    return ax


def context_ticks(ax, contexts, label="Context length (tokens)"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([f"{c//1024}k" for c in contexts])
    ax.set_xlabel(label)


def annotate_points(ax, xs, ys, fmt="{:.1f}", dy=10, color=INK, fontsize=8.5, ha="center"):
    for x, y in zip(xs, ys):
        if y is None:
            continue
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=(0, dy), ha=ha, fontsize=fontsize, color=color,
                    fontweight="bold")


def watermark(fig, text="DiffKV · Active Runtime (MLX)"):
    fig.text(0.995, 0.005, text, ha="right", va="bottom",
             fontsize=7.5, color=SLATE, alpha=0.8)


def finalize(fig, path, also_pdf=True, mark=True):
    if mark:
        watermark(fig)
    fig.savefig(path, dpi=220)
    if also_pdf and path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)
    print("wrote", path)
