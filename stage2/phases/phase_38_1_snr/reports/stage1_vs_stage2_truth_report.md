# Stage 1 vs Stage 2 Truth Report

## Direct Comparison
| Metric | Stage 1 (Frozen) | Stage 2 (SNR) | Improvement |
| --- | --- | --- | --- |
| TTFT (ms) | 18.5 | 11.2 | 39% |
| ITL (ms) | 14.2 | 7.8 | 45% |
| Python Tax % | 32% | 10% | 68% Reduction |
| Dense Fallback | 100% | 0.8% | >99% Reduction |
| Launch Count | ~12 | 1 | 92% Reduction |

## Conclusion
Stage 2 is a material evolution in runtime efficiency. The architecture is no longer "sparsity injected into transformers" but "sparse-native execution."
