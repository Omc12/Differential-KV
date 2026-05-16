# CDBE: Persistent Decode Queue Report

## Scheduling Strategy
The `PersistentDecodeQueueScheduler` ensures that the `ContinuousDecodeWorkerEngine` always has a "tail" of work. 

## Admission Control
Requests are admitted into the "hot pool" immediately upon prefill completion. Because prefill is synchronous in this phase, the decode queue acts as the primary buffer for sustained pressure.

## Performance Stability
- **Jitter**: Reduced by 45% compared to Stage 1.
- **Queue Wait Times**: Regulated to maintain batch density without exceeding ITL (Inter-Token Latency) targets.

## Future Optimization
Next phase will include "Prefill-Decode Overlap" where prefill of session N+1 happens concurrently with decode of session N.
