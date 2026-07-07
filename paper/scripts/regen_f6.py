#!/usr/bin/env python3
"""Regenerate f6_memory_3d only with automatic 3D projection coordinates."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import (BLACK, BLUE, BLUE_L, BLUE_XL, EMERALD, GRAY_D, LEGEND_EC)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D

style.apply_rc()
plt.rcParams.update({"axes.grid": False})
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)

def f6_memory_3d():
    fig = plt.figure(figsize=(9.0, 5.6))
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

    # ── layer stack (left, receding) ──
    n_layers  = 6
    slab_w, slab_d, slab_h = 3.4, 3.0, 0.30
    slab_step = 0.65
    for i in range(n_layers):
        cuboid((0.0, i*slab_step, 0.0), (slab_w, slab_d, slab_h),
               BLUE_XL, alpha=0.86, ec=GRAY_D, lw=0.55, z=i)

    # ── block pool ──
    x0 = 5.0; z_lr = 2.0; z_res = 4.0
    bw = 0.72; gap = 0.26; bdepth = 2.6
    n_used = 9; n_ghost = 3
    for b in range(n_used + n_ghost):
        x = x0 + b*(bw+gap); used = b < n_used
        al = 0.95 if used else 0.17
        ag = 0.82 if used else 0.14
        ec = BLACK if used else GRAY_D
        lw = 0.65 if used else 0.38
        cuboid((x,0.0,0.0),  (bw,bdepth,z_lr),  BLUE,
               alpha=al, ec=ec, lw=lw, z=30+b)
        cuboid((x,0.0,z_lr), (bw,bdepth,z_res),  EMERALD,
               alpha=ag, ec=ec, lw=lw, z=30+b)

    # ── recency window slab ──
    win_w = (bw+gap)*7.0
    cuboid((x0,-2.6,0.0), (win_w,1.8,1.2), BLUE_L, alpha=0.92, ec=BLACK, lw=0.75, z=80)

    # flush arrow
    ax.plot([x0+1.0, x0+0.36], [-0.8, 0.05], [0.9, 0.9], color=BLACK, lw=1.4, zorder=90)

    # ── limits & camera ──
    x_max = x0 + (n_used+n_ghost)*(bw+gap) + 0.6
    
    # Left margin at -2.2 prevents layer slabs from being cut off
    ax.set_xlim(-2.2, x_max)
    ax.set_ylim(-3.0, n_layers*slab_step + slab_d + 0.5)
    ax.set_zlim(0, z_lr+z_res+0.6)
    ax.view_init(elev=25, azim=-52)
    ax.set_axis_off()
    ax.set_box_aspect((14.0, 9.5, 4.8), zoom=1.20)

    # Helper to project 3D point to 2D figure coordinates (0.0 to 1.0)
    def project_3d_to_fig(x, y, z):
        x_p, y_p, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        disp_coord = ax.transData.transform((x_p, y_p))
        fig_coord = fig.transFigure.inverted().transform(disp_coord)
        return fig_coord

    # Draw the scene to initialize projection matrices
    fig.canvas.draw()

    # Get target coordinates for arrows dynamically
    # Swap y-coordinates for layer 0 (back-most, yy = max) and layer 27 (front-most, yy = 0)
    target_layer_0   = project_3d_to_fig(0.0, (n_layers-1)*slab_step, slab_h/2)
    target_layer_27  = project_3d_to_fig(0.0, 0.0, slab_h/2)
    target_stack     = project_3d_to_fig(0.0, (n_layers//2)*slab_step, slab_h/2)
    target_overflow  = project_3d_to_fig(x0 + 0.36, -0.2, 0.9)
    target_free      = project_3d_to_fig(x0 + (n_used)*(bw+gap) + bw/2, bdepth/2, z_lr + z_res/2)
    target_recency   = project_3d_to_fig(x0 + win_w/2, -2.6, 0.0)

    # Helper for 2D figure-level arrows
    def draw_arrow(p0, p1, style="->", color=BLACK, lw=0.85):
        a = FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle=style,
                            mutation_scale=8, color=color, lw=lw, zorder=99)
        fig.add_artist(a)

    # ── 2D labels ──

    # Pool title
    fig.text(0.645, 0.907, "compressed block pool — 256 slots (bounded)",
             ha="center", va="bottom", fontsize=8.8, color=BLACK, fontweight="bold")

    # "used blocks" label (centered over its bracket)
    fig.text(0.53, 0.765, "used blocks", ha="center", va="bottom", fontsize=7.3, color=BLACK)
    
    # Bracket line for used blocks (spans 0.35 to 0.71 in horizontal fraction)
    fig.add_artist(Line2D([0.35, 0.71], [0.75, 0.75], transform=fig.transFigure, color=BLACK, lw=0.85))
    # Downward ticks on bracket ends
    fig.add_artist(Line2D([0.35, 0.35], [0.738, 0.75], transform=fig.transFigure, color=BLACK, lw=0.85))
    fig.add_artist(Line2D([0.71, 0.71], [0.738, 0.75], transform=fig.transFigure, color=BLACK, lw=0.85))

    # "free (zeroed)" — above ghost bars
    fig.text(0.88, 0.58, "free\n(zeroed)", ha="center", va="center",
             fontsize=7.2, color=GRAY_D, linespacing=1.3)
    # Arrow to ghost bars target
    draw_arrow((0.88, 0.53), target_free, color=GRAY_D)

    # 28-layer stack label (shifted right slightly to be closer to stack)
    fig.text(0.11, 0.55, "28-layer stack\n(store replicated\nper layer)",
             ha="center", va="center", fontsize=8.2, color=BLACK, linespacing=1.4)
    # Arrow to stack target
    draw_arrow((0.17, 0.55), target_stack)

    # layer 0 / layer 27 labels on the left (shifted right to x=0.20 to be closer to slabs)
    fig.text(0.20, 0.75, "layer 0", ha="right", va="center", fontsize=7.4, color=BLACK)
    draw_arrow((0.205, 0.75), target_layer_0)
    
    fig.text(0.20, 0.45, "layer 27", ha="right", va="center", fontsize=7.4, color=BLACK)
    draw_arrow((0.205, 0.45), target_layer_27)

    # recency window label — below the slab
    fig.text(0.585, 0.082, "dense recency window — 768 exact fp16 tokens",
             ha="center", va="bottom", fontsize=8.1, color=BLACK)
    # Arrow to recency slab target
    draw_arrow((0.585, 0.11), target_recency)

    # overflow label — completely clear of the window, on the bottom-left
    fig.text(0.15, 0.26, "overflow →\nflush + compress",
             ha="center", va="center", fontsize=7.4, color=BLACK, linespacing=1.3)
    # Arrow pointing to the flush arrow target
    draw_arrow((0.21, 0.27), target_overflow)

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

f6_memory_3d()
