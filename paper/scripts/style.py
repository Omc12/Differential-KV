"""DKV paper visual identity v2 — DeepSeek-report aesthetics.

Principles (matched to the DeepSeek-V3.2 report):
  * charts carry NO in-figure title — the LaTeX caption does that job;
  * full thin box spines on all four sides; framed legend with a hairline border;
  * text is ALWAYS black; colour is reserved for shapes/lines only;
  * palette = blues (primary) + emerald (secondary accent) + neutral grays;
  * subtle dotted y-grid only; bold black value labels; generous whitespace.

Every figure imports from here so the document reads as one visual system.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── palette ─────────────────────────────────────────────────────────────────
BLACK    = "#111111"   # all text + box strokes
BLUE     = "#3D6FD1"   # primary series (DKV)
BLUE_D   = "#274F9E"   # darker blue (lines, emphasis)
BLUE_L   = "#A9C4EF"   # light blue fill
BLUE_XL  = "#DDE7F8"   # very light blue (panel fills)
EMERALD  = "#2E9E77"   # secondary accent (exact/upper-bound, cap lines)
EMER_L   = "#BFE3D4"   # light emerald fill
GRAY     = "#C4C9D0"   # dense-baseline bars / neutral series
GRAY_D   = "#7A828C"   # neutral lines
GRAY_XL  = "#EDEFF2"   # neutral light fill (diagram boxes)
GRID     = "#E4E7EB"   # y-grid
LEGEND_EC= "#B9BEC6"   # legend frame hairline
WHITE    = "#FFFFFF"
OOM_RED  = "#C0392B"   # only for failure markers (OOM ✗)

# semantic roles
C_DKV = BLUE
C_DENSE  = GRAY
C_DENSE_LN = GRAY_D    # dense as a line
C_EXACT  = EMERALD

# ── compatibility aliases (older scripts) ───────────────────────────────────
INK = BLACK; NAVY = BLUE_D; SKY = BLUE_L; MIST = BLUE_XL; SLATE = GRAY_D
AMBER = EMERALD; GOOD = EMERALD; BAD = OOM_RED; PANEL = GRAY_XL
MUTED = "#444444"; EDGE = BLACK; ACCENT = BLUE; ACCENT2 = EMERALD
SERIES = [BLUE, GRAY_D, EMERALD, BLUE_L, "#6B5CA5"]


def apply_rc():
    plt.rcParams.update({
        "figure.facecolor":  WHITE,
        "axes.facecolor":    WHITE,
        "savefig.facecolor": WHITE,
        "font.family":       "DejaVu Sans",
        "font.size":         9.0,
        "mathtext.fontset":  "dejavusans",
        "text.color":        BLACK,
        "axes.titlesize":    9.5,          # rarely used — captions carry titles
        "axes.labelsize":    9.0,
        "axes.labelcolor":   BLACK,
        "axes.edgecolor":    BLACK,
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "axes.axisbelow":    True,
        # full box, DeepSeek-style
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "grid.color":        GRID,
        "grid.linewidth":    0.7,
        "grid.linestyle":    (0, (1, 3)),   # fine dots
        "xtick.color":       BLACK,
        "ytick.color":       BLACK,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon":    True,
        "legend.edgecolor":  LEGEND_EC,
        "legend.framealpha": 1.0,
        "legend.fancybox":   False,
        "legend.fontsize":   8.0,
        "legend.borderpad":  0.5,
        "legend.labelspacing": 0.35,
        "legend.handlelength": 1.6,
        "lines.linewidth":   1.7,
        "lines.markersize":  4.5,
        "lines.markeredgewidth": 0.9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.03,
        "hatch.linewidth":   0.7,
        "hatch.color":       WHITE,
    })


def box_axes(ax):
    """Thin full-box spines (all four), black."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(BLACK)
        s.set_linewidth(0.8)
    ax.tick_params(color=BLACK)
    return ax


def legend(ax, **kw):
    """Framed hairline legend, DeepSeek style."""
    kw.setdefault("frameon", True)
    lg = ax.legend(**kw)
    lg.get_frame().set_edgecolor(LEGEND_EC)
    lg.get_frame().set_linewidth(0.7)
    return lg


def bar_value_labels(ax, bars, fmt="{:.1f}", dy=2, fontsize=6.8):
    for r in bars:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    va="bottom", fontsize=fontsize, color=BLACK)


def context_ticks(ax, contexts, label="Context length (tokens)"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([f"{c//1024}K" for c in contexts])
    ax.minorticks_off()
    ax.set_xlabel(label)


def finalize(fig, path, also_pdf=True):
    fig.savefig(path, dpi=300)
    if also_pdf and path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)
    print("wrote", path)
