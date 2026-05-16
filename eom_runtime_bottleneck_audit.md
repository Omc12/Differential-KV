# EOM Runtime Bottleneck Audit: Phase 34.0

## 1. Executive Summary
Differential KV’s sparse runtime savings are materially real but currently "trapped" behind serving-layer inefficiencies and decode-stage serialization. While the Triton kernels reduce FLOPS and memory bandwidth requirements, the overhead of launching thousands of fragmented kernels and the sequential nature of the current autoregressive serving loop prevents these savings from translating into user-visible TPS gains.

## 2. Identified Bottlenecks

### B1: Decode Serialization (CRITICAL)
- **Problem**: The current inference engine processes concurrent requests sequentially within a microbatch.
- **Impact**: Concurrency 8 takes ~8x longer than Concurrency 1, resulting in zero throughput scaling.
- **Root Cause**: `real_runtime_executor` uses a Python `for` loop over sessions.

### B2: Launch Fragmentation (HIGH)
- **Problem**: Sparse KV reconstruction happens on a per-block, per-layer basis.
- **Impact**: For Qwen2.5-0.5B (24 layers) and 4096 context (64 blocks/layer), a single token generation triggers ~1,500 Triton kernel launches.
- **Root Cause**: Lack of kernel fusion across blocks and layers.

### B3: Synchronization Hotspots (HIGH)
- **Problem**: Implicit synchronizations between Python control flow and GPU kernel launches.
- **Impact**: CPU remains idle waiting for kernel completion before launching the next block.
- **Root Cause**: Lack of CUDA Graphs or asynchronous launch overlapping.

### B4: Scheduler Contention (MEDIUM)
- **Problem**: The `SparseRequestScheduler` uses a simple `PriorityQueue` with high-frequency wakeups.
- **Impact**: CPU overhead dominates at high concurrencies.
- **Root Cause**: Unoptimized `asyncio` loop frequency and lack of batch-aware scheduling.

### B5: Serving Overhead Dominance (MEDIUM)
- **Problem**: Serialization, tokenizer scheduling, and streaming flushes add significant wall-clock time.
- **Impact**: Up to 15-20% of end-to-end latency is non-model overhead.
- **Root Cause**: Inefficient word-based streaming and blocking serialization.

## 3. Occupancy Collapse Analysis
- **Observed**: GPU utilization is bursty and low during sparse reconstruction.
- **Mechanism**: Small, fragmented kernels fail to saturate SM occupancy.
- **Result**: Sparse savings are offset by low arithmetic intensity.

## 4. EOM Action Plan
- **Fusion**: Implement `DecodePipelineFusionEngine` to coalesce launches.
- **Recovery**: Use `OccupancyRecoveryController` to stabilize GPU work windows.
- **Minimization**: Streamline the API path in `ServingOverheadMinimizer`.
- **Priority**: Ensure the `SparseRuntimePrioritizer` keeps sparse paths active even under high serving pressure.
