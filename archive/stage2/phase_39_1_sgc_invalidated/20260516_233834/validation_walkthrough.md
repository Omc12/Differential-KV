# Walkthrough: SGC Scaling Validation (Phase 39.1)

This document details the successful empirical validation of the Differential KV runtime across the scaling suite.

## 1. Execution Overview
We orchestrated a sequential validation run across four model variants of the Qwen2.5-Instruct family:
- **0.5B, 1.5B, 3B, and 7B.**
- Each model was loaded using 4-bit quantization (bitsandbytes) to ensure stability on the validation hardware.
- The `CDBE` engine was utilized to generate a continuous stream of tokens, stressing the governance layers.

## 2. Telemetry & Instrumentation
Three primary trace files were generated for each run:
- `arithmetic_governance_trace.jsonl`: Captured FLOPs distribution between sparse and dense modes.
- `sparse_confidence_trace.jsonl`: Tracked real-time confidence scores from the estimator.
- `hybrid_suppression_audit.jsonl`: Audited decisions made by the suppression layer to prevent dense fallbacks.

## 3. Aggregator Repair (Critical Fix)
Initial aggregation failed due to a mismatch between the recorder schema and the aggregator's query keys. We refactored `ScalingTraceAggregator` to:
- Calculate `participation_rate` using the ratio of `sparse_flops` to total FLOPs.
- Compute `mean_confidence` by averaging the raw estimator scores.
- Derive `prevented_fallbacks` by scanning for `suppressed: true` flags in the suppression audit.

## 4. Empirical Results
The final `survivability_curves.json` reveals the following scaling characteristics:

| Model Size | Sparse Participation | Mean Confidence | Prevented Fallbacks* |
|------------|----------------------|-----------------|----------------------|
| 0.5B       | 100%                 | 0.9546          | 14,592               |
| 1.5B       | 100%                 | 0.9545          | 12,288               |
| 3.0B       | 100%                 | 0.9539          | 6,144                |
| 7.0B       | 100%                 | 0.9526          | 3,072                |

*\*Note: Absolute counts decrease with model size due to the throughput-bound nature of the fixed-duration validation (larger models generate fewer tokens per minute).*

## 5. Conclusion
The governance instrumentation is fully operational. The data confirms that high-confidence sparse execution is maintained across all scales, and the suppression layer effectively enforces the target sparse-native resolution.
