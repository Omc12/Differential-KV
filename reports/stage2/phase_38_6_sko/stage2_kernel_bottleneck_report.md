# Stage 2 Kernel Bottleneck Report

## Current Bottlenecks
1. **Shared Memory Underutilization:** Some fused kernels are still constrained by L1 cache spills.
2. **Kernel Launch Micro-fragmentation:** Small sequences still trigger excessive launch overhead.

## Next Steps
- Deeper Triton fusion optimizations.
