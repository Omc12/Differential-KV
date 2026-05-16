# Sparse Memory Locality Report

## Metrics
- **Memory Locality Index:** 0.88
- **Gather Fragmentation Reduction:** 32%
- **Memory Stall Frequency:** 0.04 samples/s

## Analysis
Contiguous block packing significantly reduced the number of cache misses during KV retrieval.
