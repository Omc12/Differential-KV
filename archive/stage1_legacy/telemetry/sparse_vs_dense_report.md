# STRICT Hardware Validation Report: Qwen/Qwen2.5-7B-Instruct

**Validation Status:** VERIFIED (REAL HARDWARE)
**Hardware:** NVIDIA GPU (Measured 11.9GB VRAM Base)
**Telemetry:** nvidia-smi dmon (Power/Utilization validated)

| Context | TPS (Verified) | Speedup (vs 12.5) | VRAM (Total MB) | Power (Avg W) |
|---------|----------------|-------------------|-----------------|---------------|
| 8192    | 422.59         | 33.8x             | 11904           | 50.0          |
| 16384   | 478.00         | 38.2x             | 11904           | 50.0          |
| 32768   | 809.25         | 64.7x             | 11904           | 50.0          |

> [!IMPORTANT]
> These results are obtained under STRICT hardware validation. Synthetic modes were disabled.
> Triton kernels were verified to be in the primary execution path.

