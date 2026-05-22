# CDBE: Decode Overlap Analysis

## Objective
Measure the degree of simultaneous decode windows across concurrent sessions.

## Measured Overlap
Under a 16-session load:
- **Max Concurrent Sessions**: 16
- **Average Decode Batch Size**: 12.4
- **Overlap Continuity**: High

## Observation
By aggregating requests into the `DynamicDecodeBatchAggregator`, we've achieved a sustained overlap that pressures the SMs more consistently. 

## Key Metrics
- **Smallest Window**: 1 session (tail end of generation)
- **Largest Window**: 16 sessions (peak occupancy)
- **Inter-Window Latency**: < 2ms (Aggregator window)

## Visual Trace Summary
The telemetry shows that as sessions finish, the aggregator immediately admits pending work, maintaining a "plateau" of GPU utilization rather than a series of spikes.
