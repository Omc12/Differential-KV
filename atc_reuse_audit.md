# ATC Reuse Audit: Differential KV Sparse Systems

## 1. Reusable Systems
| System | Phase | Status | Reuse Strategy |
| :--- | :--- | :--- | :--- |
| **Aggressive KV Materializer** | 30.0 (SEM) | Active | Extend to handle collapsed token eviction. |
| **Sustained Decode Engine** | 30.1 (SHM) | Active | Use for continuous ATC hardware testing. |
| **Benchmark Mode Classifier** | 30.2 (BIC) | Active | Maintain PRODUCTION class for ATC. |
| **Sparse MLP Router** | 31.0 (SML) | Active | Coordinate with token-level survival. |
| **Sparse Flop Accountant** | 30.0 (SEM) | Active | Track collapsed token FLOP savings. |
| **Runtime Density Profiler** | 30.0 (SEM) | Active | Track ATC impact on runtime dominance. |
| **Triton Dispatch Infra** | 30.1 (SHM) | Active | Use for token gather/scatter kernels. |

## 2. Partially Materialized Systems
- **Triton Kernels**: Currently focus on attention/MLP; needs gather/scatter for sequence compression.
- **Participation Controller**: Needs to account for total active sequence length, not just sparsity ratios.
- **KV Virtualization**: Needs a mechanism to "drop" collapsed tokens from the active window without full eviction.

## 3. Disconnected Systems
- **CRMP Activation**: Needs to ensure ATC is only active in high-capability modes.
- **Sparse Schedulers**: Often assume a fixed sequence length; must adapt to dynamic collapse.

## 4. Telemetry Gaps
- **Active Token Ratio**: Currently we track neuron/attention ratios; needs top-level sequence ratio.
- **Effective Sequence Length**: New metric required for hardware-visible scaling.

## 5. Dense Fallback Risks
- **HF Forward**: Most likely to fallback to dense if sequence indices become fragmented.
- **Logits/Sampling**: Currently always dense; must skip collapsed tokens.

---
**Verdict**: Differential KV infrastructure is 85% ready for ATC. The primary task is **token-level routing** and **sequence compression** integration.
