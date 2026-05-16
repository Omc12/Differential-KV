# CDBE: GPU Sustained Pressure Report

## Hardware Utilization (RTX 4070)
- **SM Utilization**: 35-52% (Previous: 0-12%)
- **Power Draw**: 38-62W (Previous: 13-31W)
- **Memory Clock**: Locked at high P-state during continuous decode windows.

## Occupancy Continuity
The `PersistentCUDAGraphExecutionManager` has eliminated the per-launch "valleys". By reusing graphs for batch sizes 1-16, the kernel overhead is amortized across the persistent window.

## Pressure Analysis
The GPU is now "continuously fed". While not at 100% saturation (due to the sparse nature of Differential KV), the **meaningful occupancy** has tripled.

## Bottleneck Shift
The dominant bottleneck is shifting from **Python Orchestration** to **Memory Bandwidth (HBM traffic)** for the sparse KV lookups. This is the intended direction for Stage 2.
