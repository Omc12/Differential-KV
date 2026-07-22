# Phase 22 Native AsyncCompressor Design

## Why Pure Python Is Insufficient
The Python `threading.Thread` + `queue.Queue` model releases the GIL during `torch.linalg.svd` LAPACK calls but holds it during all orchestration: queue pops, dict updates, pool writes, and state transitions. Under 100+ concurrent sessions, this creates millisecond-scale GIL stalls on the main decode thread.

## Native Design: `DKVCompressorThread` (C++ Extension)

### 1. Thread Lifecycle
```cpp
// dkv_compressor.cpp
class DKVCompressorThread {
    std::thread worker_;
    std::atomic<bool> running_;
    SPSCRingBuffer<CompressJob> queue_;  // Lock-free Single-Producer Single-Consumer
    cudaStream_t compress_stream_;
    void run();  // Worker loop: pop job → cuSOLVER SVD → write to slab pool
public:
    void start();
    void stop();
    bool submit(CompressJob job);  // Returns false on overflow (non-blocking)
};
```

### 2. Lock-Free Job Queue
- Uses a bounded **SPSC ring buffer** (single producer: Python/main thread, single consumer: C++ worker).
- Zero mutex overhead on the hot-path `submit()` call.
- If the ring buffer is full (overload), `submit()` returns `false` immediately. The block remains in the Dense Pool until the queue drains.

### 3. GPU-Direct Write
After SVD, the worker writes compressed `U` and `V` tensors **directly to the slab pool** using `cudaMemcpyAsync()` on the dedicated `compress_stream_`. No Python tensor allocation occurs.

### 4. Graceful Overload Handling
When the queue is full:
- The block **stays in the Dense Pool** (safe fallback).
- No data loss, no stall.
- The next available queue slot triggers re-submission automatically.

### 5. Cancellation Safety
Each `CompressJob` carries a `session_id` and `block_id`. If a session disconnects while its block is mid-compression:
- The worker checks `is_session_alive(session_id)` before writing to the slab pool.
- If dead: discard result, mark block `Invalid`, free the Dense Pool slot.
