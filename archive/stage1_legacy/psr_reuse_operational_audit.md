# PSR (Production Serving Realism) Reuse & Operational Audit

## 1. Production-Capable Serving Systems
The following systems are currently in place and serve as the foundation for PSR:
- **PSI (Production Serving Infrastructure)**: Core deployment modules and orchestration.
- **DPK (Deployment Packaging)**: Readiness for containerized and distributed environments.
- **OpenAI-Compatible API Gateway**: Initial REST implementation for industry compatibility.
- **Production Session Manager**: Basic tracking of concurrent request states.
- **Real Sparse Serving Runtime**: Materialized integration of Triton kernels into a serving loop.

## 2. Currently Excluded Serving Overheads
The following overheads are present in production but typically bypassed in current isolated benchmarks:
- **Streaming Response Serialization**: SSE (Server-Sent Events) framing and chunking logic.
- **API Entry/Exit Latency**: HTTP/TCP handshake, JSON serialization/deserialization.
- **Tokenizer/Sampling Latency**: Tokenization of prompt and sampling (greedy/top-p) of logits.
- **Asyncio Contention**: Event loop pressure when handling hundreds of concurrent streaming connections.

## 3. Sparse Paths Degrading Under Concurrency
- **KV Virtualization Fragmentation**: Under high concurrency, the virtualized KV address space may fragment, leading to inefficient memory access.
- **Triton Kernel Launch Overhead**: With small batch sizes (high concurrency, low per-user batch), launch latency becomes a significant fraction of total time.
- **Residency Swapping**: Concurrent requests with long contexts may force frequent eviction/re-materialization of KV blocks.

## 4. Unrealistic Scheduler Paths
- **Isolated Batching**: Current schedulers often assume batch-synchronous execution, whereas real serving is continuous.
- **Fixed Context Lengths**: Benchmarks use uniform context lengths; real serving involves a mix of 100-token and 100k-token requests.
- **No Starvation Handling**: Sparse optimizations may favor requests with higher "salience," potentially starving low-salience requests.

## 5. Telemetry Gaps
- **Streaming Jitter**: Metrics currently focus on throughput, ignoring the variance in inter-token latency (ITL).
- **Queue Wait Time**: Isolated benchmarks often assume the GPU is immediately available.
- **System-Wide TPS**: Focus is on per-user performance rather than total system capacity under load.
- **Resource Contention Cost**: The "hidden" cost of VRAM residency management and sparse metadata updates.
