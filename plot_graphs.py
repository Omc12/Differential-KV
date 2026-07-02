import os
import matplotlib.pyplot as plt
import numpy as np

# Output Directory — override with PLOT_OUT_DIR; defaults to ./plots next to this file.
out_dir = os.environ.get(
    "PLOT_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
os.makedirs(out_dir, exist_ok=True)

# Colors
DENSE_COLOR = "#FF007F"  # Electric Magenta
DIFFKV_COLOR = "#00F5D4"  # Electric Cyan/Teal
GRID_COLOR = "#E5E5E5"
TEXT_COLOR = "#2B2D42"
ACCENT_COLOR = "#7209B7"  # Deep Purple

# Matplotlib global settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'axes.edgecolor': TEXT_COLOR,
    'axes.linewidth': 0.8,
    'grid.color': GRID_COLOR,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
})

contexts = [1024, 2048, 4096, 8192]
contexts_str = ["1K", "2K", "4K", "8K"]

# ==========================================
# 1. LATENCY & THROUGHPUT GRAPH
# ==========================================
prefill_dense = [0.048, 0.953, 2.856, None]
prefill_diffkv = [0.040, 0.908, 9.584, 42.452]

tps_dense = [31.6, 24.1, 8.1, None]
tps_diffkv = [37.9, 27.3, 6.6, 4.2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: Prefill Latency
ax1.plot(contexts[:3], prefill_dense[:3], 'o-', label='Dense (Baseline)', color=DENSE_COLOR, linewidth=2.5, markersize=8)
ax1.plot(contexts, prefill_diffkv, 's-', label='DiffKV (Compressed)', color=DIFFKV_COLOR, linewidth=2.5, markersize=8)
# Mark OOM
ax1.plot(8192, 10, 'x', color=DENSE_COLOR, markersize=10, markeredgewidth=2.5)
ax1.text(8192, 12, "Dense\nOOM", color=DENSE_COLOR, ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xscale('log', base=2)
ax1.set_xticks(contexts)
ax1.set_xticklabels(contexts_str)
ax1.set_xlabel('Context Length (tokens)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_ylabel('Prefill Latency (seconds)', fontsize=11, fontweight='bold')
ax1.set_title('Prefill Latency (Lower is Better)', fontsize=12, fontweight='bold', pad=10)
ax1.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR)
ax1.grid(True, which="both", ls="--")

# Right: Decode TPS
ax2.plot(contexts[:3], tps_dense[:3], 'o-', label='Dense (Baseline)', color=DENSE_COLOR, linewidth=2.5, markersize=8)
ax2.plot(contexts, tps_diffkv, 's-', label='DiffKV (Zero-Sync)', color=DIFFKV_COLOR, linewidth=2.5, markersize=8)
# Mark OOM
ax2.plot(8192, 0.5, 'x', color=DENSE_COLOR, markersize=10, markeredgewidth=2.5)
ax2.text(8192, 1.5, "Dense\nOOM", color=DENSE_COLOR, ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xscale('log', base=2)
ax2.set_xticks(contexts)
ax2.set_xticklabels(contexts_str)
ax2.set_xlabel('Context Length (tokens)', fontsize=11, fontweight='bold', labelpad=8)
ax2.set_ylabel('Decode Throughput (tokens/sec)', fontsize=11, fontweight='bold')
ax2.set_title('Decode Throughput (Higher is Better)', fontsize=12, fontweight='bold', pad=10)
ax2.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR)
ax2.grid(True, which="both", ls="--")

plt.suptitle('Latency & Decode Throughput vs. Context Length', y=0.98, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'graph1_latency.png'), dpi=200)
plt.close()

# ==========================================
# 2. QUALITY (FIDELITY) GRAPH
# ==========================================
cos_sim_diffkv = [0.9993, 0.9991, 0.9993, 0.9994]

plt.figure(figsize=(8, 5))
plt.axhline(y=1.0, color=DENSE_COLOR, linestyle='-', linewidth=2.0, label='Dense Baseline (100% Fidelity)')
plt.plot(contexts, cos_sim_diffkv, 's-', color=DIFFKV_COLOR, linewidth=2.5, markersize=8, label='DiffKV Compressed')

plt.xscale('log', base=2)
plt.xticks(contexts, contexts_str)
plt.xlabel('Context Length (tokens)', fontsize=11, fontweight='bold', labelpad=8)
plt.ylabel('Cosine Similarity (Keys & Values)', fontsize=11, fontweight='bold')
plt.title('DiffKV Cosine Similarity to Dense Baseline', fontsize=13, fontweight='bold', pad=15)
plt.ylim(0.9980, 1.0005)
plt.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR, loc='lower right')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'graph2_quality.png'), dpi=200)
plt.close()

# ==========================================
# 3. MEMORY FOOTPRINT GRAPH
# ==========================================
peak_dense_mb = [1254.1, 1561.3, 2459.1, None]
peak_diffkv_mb = [958.8, 969.7, 1166.2, 1434.4]

kv_dense_mb = [309.0, 321.0, 642.0, None]
kv_diffkv_mb = [13.0, 24.0, 306.0, 882.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: Peak System Memory
ax1.plot(contexts[:3], [x/1024.0 for x in peak_dense_mb[:3]], 'o-', label='Dense (Baseline)', color=DENSE_COLOR, linewidth=2.5, markersize=8)
ax1.plot(contexts, [x/1024.0 for x in peak_diffkv_mb], 's-', label='DiffKV (Compressed)', color=DIFFKV_COLOR, linewidth=2.5, markersize=8)
# Mark OOM
ax1.plot(8192, 4.0, 'x', color=DENSE_COLOR, markersize=10, markeredgewidth=2.5)
ax1.text(8192, 4.2, "Dense\nOOM", color=DENSE_COLOR, ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xscale('log', base=2)
ax1.set_xticks(contexts)
ax1.set_xticklabels(contexts_str)
ax1.set_xlabel('Context Length (tokens)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_ylabel('Peak VRAM Footprint (GB)', fontsize=11, fontweight='bold')
ax1.set_title('Peak VRAM (Model + KV Cache)', fontsize=12, fontweight='bold', pad=10)
ax1.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR)
ax1.grid(True, which="both", ls="--")

# Right: KV Cache Memory
ax2.plot(contexts[:3], kv_dense_mb[:3], 'o-', label='Dense (Baseline)', color=DENSE_COLOR, linewidth=2.5, markersize=8)
ax2.plot(contexts, kv_diffkv_mb, 's-', label='DiffKV (Compressed)', color=DIFFKV_COLOR, linewidth=2.5, markersize=8)
# Mark OOM
ax2.plot(8192, 100, 'x', color=DENSE_COLOR, markersize=10, markeredgewidth=2.5)
ax2.text(8192, 150, "Dense\nOOM", color=DENSE_COLOR, ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xscale('log', base=2)
ax2.set_xticks(contexts)
ax2.set_xticklabels(contexts_str)
ax2.set_xlabel('Context Length (tokens)', fontsize=11, fontweight='bold', labelpad=8)
ax2.set_ylabel('KV Cache Memory Footprint (MB)', fontsize=11, fontweight='bold')
ax2.set_title('Isolated KV Cache VRAM Footprint', fontsize=12, fontweight='bold', pad=10)
ax2.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR)
ax2.grid(True, which="both", ls="--")

plt.suptitle('Memory Footprint Comparison vs. Context Length', y=0.98, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'graph3_memory.png'), dpi=200)
plt.close()

# ==========================================
# 4. RANK ABLATION STUDY GRAPH
# ==========================================
ranks = [8, 16, 32, 64]
ab_cos = [0.9909, 0.9991, 0.9997, 0.9997]
ab_kv_mb = [12, 24, 48, 96]  # Theoretical KV Cache Memory Footprint for R=[8,16,32,64] at 2048

fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

color = ACCENT_COLOR
ax1.set_xlabel('Compression Rank ($R$)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_ylabel('Output Quality (Cosine Similarity)', color=color, fontsize=11, fontweight='bold')
line1 = ax1.plot(ranks, ab_cos, 'o-', color=color, linewidth=2.5, markersize=8, label='Quality (Cosine Sim)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(ranks)
ax1.set_ylim(0.9880, 1.0010)
ax1.grid(True, which="both", ls="--")

ax2 = ax1.twinx()  
color = "#2E86DE"  # Medium Blue
ax2.set_ylabel('KV Cache VRAM Footprint (MB)', color=color, fontsize=11, fontweight='bold')
line2 = ax2.plot(ranks, ab_kv_mb, 's--', color=color, linewidth=2.0, markersize=8, label='KV Cache VRAM (MB)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 110)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', frameon=True, facecolor='white', edgecolor=GRID_COLOR)

plt.title('Ablation Study: Compression Rank ($R$) vs. Quality vs. Memory', fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig(os.path.join(out_dir, 'graph4_ablation.png'), dpi=200)
plt.close()

print("All plots generated successfully.")
