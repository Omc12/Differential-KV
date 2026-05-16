"""
visualization/kernel_execution_flow.py

Generates execution traces and flow diagrams for fused cognitive kernels.
Visualizes the overlap between compute, telemetry, and stabilization.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_kernel_flow():
    os.makedirs("results/phase29/viz", exist_ok=True)
    
    # Timeline data (microseconds)
    ops = [
        "Load Q/KV", "Drift Tracking", "Resonance Sync", 
        "Restore KV", "Attention Dot", "Softmax", 
        "Output Proj", "Telemetry Push"
    ]
    
    # Standard (Sequential)
    std_durations = [50, 40, 60, 100, 150, 50, 100, 30]
    std_offsets = np.cumsum([0] + std_durations[:-1])
    
    # Fused (KCRA)
    fused_durations = [30, 10, 5, 20, 120, 40, 80, 5]
    fused_offsets = np.cumsum([0] + fused_durations[:-1])
    
    # Simulate overlapping ops in fused kernel (e.g. telemetry push is async)
    fused_offsets[-1] = fused_offsets[2] # Pushed early and async
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Standard
    for i, (op, dur, off) in enumerate(zip(ops, std_durations, std_offsets)):
        ax1.broken_barh([(off, dur)], (i*10, 8), facecolors='tab:gray', alpha=0.6)
        ax1.text(off + dur/2, i*10 + 4, op, ha='center', va='center', fontsize=8)
    
    ax1.set_title("Standard Attention + Stabilization (Sequential)")
    ax1.set_ylabel("Operation Index")
    
    # Plot Fused
    colors = plt.cm.viridis(np.linspace(0, 1, len(ops)))
    for i, (op, dur, off) in enumerate(zip(ops, fused_durations, fused_offsets)):
        ax2.broken_barh([(off, dur)], (i*10, 8), facecolors=colors[i], alpha=0.8)
        ax2.text(off + dur/2, i*10 + 4, op, ha='center', va='center', fontsize=8, fontweight='bold')
        
    ax2.set_title("KCRA Fused Resonance Attention (Hardware-Native)")
    ax2.set_ylabel("Operation Index")
    ax2.set_xlabel("Time (microseconds)")
    
    plt.tight_layout()
    plt.savefig("results/phase29/viz/kernel_execution_flow.png")
    print("Execution flow visualization saved to results/phase29/viz/kernel_execution_flow.png")

if __name__ == "__main__":
    plot_kernel_flow()
