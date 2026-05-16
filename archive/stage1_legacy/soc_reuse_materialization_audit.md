# SOC Reuse & Materialization Audit: Differential KV

## 1. Materially Contributing Systems
| System | Contribution | Status |
| :--- | :--- | :--- |
| **SML (Sparse MLP)** | Real neuron routing; reduces FFN compute. | Highly active. |
| **ATC (Token Collapse)** | Real sequence compression; reduces token-level load. | Highly active. |
| **FRM (Residency)** | Forces model weights to stay on GPU. | Highly active. |
| **SEM (Economics)** | Hard active-window limits for KV residency. | Highly active. |

## 2. Fragmented Sparse Kernels
- **Triton FFN Kernels**: Currently launched per token/step; small launch overhead relative to compute.
- **Gather/Scatter Kernels**: Launched frequently for token collapse; needs fusion with following GEMMs.
- **Sparse Attention**: Multiple kernels for scoring, filtering, and compute; lacks launch consolidation.

## 3. Occupancy-Inefficient Launches
- **Small Microbatches**: Individual token decodes in low-concurrency scenarios result in low SM occupancy.
- **Scattered Token Routes**: Sparse paths create fragmented memory access; needs coalescence.

## 4. Launch-Bound Paths
- **Routing Controllers**: Host-side logic for token/neuron routing adds overhead before each kernel launch.
- **Telemetry Hooks**: Frequent sampling of GPU state adds small gaps between kernels.

## 5. Dominant Dense Paths
- **Logits & Sampling**: Still run as separate dense kernels; high potential for fusion with last sparse block.
- **Tokenizer/Embeddings**: Dense by nature, but currently non-overlapping with sparse compute.

---
**Verdict**: Differential KV has achieved real sparsity, but it is **operationally fragmented**. SOC must focus on **fusion** and **batch-level consolidation** to drive sustained hardware occupancy.
