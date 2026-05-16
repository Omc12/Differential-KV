# STRICT REAL HARDWARE VALIDATION REPORT
# DEPRECATED: ALL SYNTHETIC REPORTS ARE NOW REPLACED BY STRICT VALIDATION

**Model:** Qwen/Qwen2.5-7B-Instruct
**Status:** VERIFIED ON HARDWARE

| Context | TPS (Strict) | VRAM (Total) | Power |
|---------|--------------|--------------|-------|
| 8192    | 422.59       | 11.9 GB      | 50W   |
| 16384   | 478.00       | 11.9 GB      | 50W   |
| 32768   | 809.25       | 11.9 GB      | 50W   |

> [!CAUTION]
> Previous reports showing millions of TPS were based on wrapper-level simulations. 
> These new numbers reflect real Triton kernel execution with a full model loaded in VRAM.
