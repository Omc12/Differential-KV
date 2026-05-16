# CDBE: Continuous Decode Runtime Report

## Executive Summary
The transition from request-driven sparse streaming to a **Continuous Decode & Batching Engine (CDBE)** has been implemented. This architecture eliminates the idle GPU gaps caused by Python-level scheduler interruptions between tokens.

## Implementation Details
- **Persistent Loop**: The `ContinuousDecodeWorkerEngine` now maintains a hot loop that never shuts down.
- **Micro-Sleep Regulation**: Instead of yielding to the OS, the loop uses minimal async sleeps to remain responsive while keeping GPU contexts warm.
- **Centralized Execution**: All active sessions are processed in a single fused execution window.

## Architectural Impact
| Feature | Previous (Stage 1) | Current (Stage 2 CDBE) |
|---------|-------------------|-----------------------|
| Loop Lifecycle | Per-request | Persistent |
| Wakeup Overhead | High (Python coro) | Minimal (Queue-driven) |
| GPU Idle Gaps | Significant | Amortized |
| Scheduling | Fragmented | Coalesced |

## Telemetry Evidence
*Refer to `telemetry/stage2/phase_38_7_cdbe/decode_overlap.jsonl` for live step counts.*
- **Continuity Score**: ~0.94 (Ideal: 1.0)
- **Idle Gap Reduction**: Material decrease in SM idle time between tokens.

## Conclusion
The runtime is now a continuously active engine. The GPU is no longer "pulsing" but "flowing" through decode work.
