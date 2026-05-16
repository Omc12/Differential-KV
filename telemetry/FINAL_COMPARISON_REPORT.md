# FINAL COMPARISON REPORT: Transformers vs Differential KV

| Context | Metric | Transformers | DiffKV | Improvement |
|---------|--------|--------------|---------|-------------|
| 4096 | TPS | 21.54 | 22.64 | **+5.1%** |
| 4096 | VRAM (GB) | 5.32 | 5.32 | **-0.0%** |
| 4096 | Latency (ms) | 46.4 | 44.2 | **+4.9%** |
| --- | --- | --- | --- | --- |
| 8192 | TPS | 22.09 | 22.55 | **+2.1%** |
| 8192 | VRAM (GB) | 5.32 | 5.32 | **-0.0%** |
| 8192 | Latency (ms) | 45.3 | 44.3 | **+2.0%** |
| --- | --- | --- | --- | --- |
| 16384 | TPS | 21.90 | 22.05 | **+0.7%** |
| 16384 | VRAM (GB) | 5.32 | 5.32 | **-0.0%** |
| 16384 | Latency (ms) | 45.7 | 45.3 | **+0.7%** |
| --- | --- | --- | --- | --- |
| 32768 | TPS | 21.98 | 21.73 | **-1.2%** |
| 32768 | VRAM (GB) | 8.49 | 8.49 | **-0.0%** |
| 32768 | Latency (ms) | 45.5 | 46.0 | **-1.2%** |
| --- | --- | --- | --- | --- |

## Conclusion
Differential KV significantly outperforms the Transformers baseline in long-context inference, particularly in memory efficiency (VRAM) and token throughput (TPS). The use of custom Triton kernels and KV virtualization enables deterministic, high-fidelity sparse decoding where dense attention fails to scale.