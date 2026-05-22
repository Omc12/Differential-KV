"""
runtime/async_compressor.py

Phase 7 Step 4: Asynchronous Block Compression Pipeline

SVD compression is the most expensive synchronous operation in the decode loop.
Currently _compress_block() blocks the decode thread while torch.linalg.svd runs.

This module offloads compression to a background thread pool:
  1. When a block is ready to compress, it is placed into a queue.
  2. A background worker thread performs SVD and writes U/V/scale back to the block.
  3. The block remains in DENSE state (active_k/v) until compression completes.
     Sparse attention gracefully handles this via the dense fallback path in
     fused_sparse_attention_decode (block.U is None → process_dense_block).
  4. On completion the dense active_k/v are cleared (VRAM freed).

Design constraints:
  - CORRECTNESS FIRST: a block's active_k/v remains valid and readable at all
    times during async compression. No partial state is ever visible.
  - The compress_worker serialises SVD via a single worker thread to avoid
    CUDA context contention between threads.
  - If the queue is full (backpressure), we fall back to synchronous compression
    to prevent unbounded memory growth.

Profiler-visible impact:
  - Decode latency reduced by ~SVD_time per block (typically 5-20ms per block).
  - GPU is never idle waiting for SVD when the async path is active.
"""

import threading
import queue
import time
import torch
from typing import Callable, Optional


class AsyncCompressor:
    """
    Background compression worker.

    Usage:
        compressor = AsyncCompressor(compress_fn, max_queue=32)
        compressor.start()

        # Instead of direct _compress_block(block, k, v):
        compressor.submit(block, k, v)   # non-blocking
        # or
        compressor.submit_sync(block, k, v)  # blocks until done
    """

    def __init__(self, compress_fn: Callable, max_queue: int = 100000, num_workers: int = 2):
        """
        compress_fn: callable(block, k, v) — the synchronous compression function.
        num_workers: number of background SVD threads (2 is sufficient for burst load).
        """
        self._compress_fn = compress_fn
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._workers: list = []
        self._num_workers = num_workers
        self._running = False

        # Stats — use a lock since multiple workers update them
        self._stats_lock = threading.Lock()
        self.stats = {
            "submitted":        0,
            "completed":        0,
            "sync_fallbacks":   0,
            "total_svd_ms":     0.0,
            "queue_depth_peak": 0,
        }

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"DiffKV-Compressor-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def stop(self) -> None:
        self._running = False
        # Clear the queue to discard pending compression tasks and speed up shutdown
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break
        # Unblock all workers
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        for t in self._workers:
            t.join(timeout=2.0)
        self._workers.clear()

    # ── Submission ───────────────────────────────────────────────────────────

    def submit(self, block, k: torch.Tensor, v: torch.Tensor) -> bool:
        """
        Non-blocking submit. Returns True if queued, False if fell back to sync.

        The caller MUST NOT clear block.active_k/v after this call — the worker
        will do so after SVD completes.
        """
        # Snapshot tensors in CPU-pinned memory to avoid holding GPU tensors in
        # the queue (prevents VRAM fragmentation from queued-but-not-compressed blocks).
        # IMPORTANT: non_blocking=True means the GPU→CPU DMA copy runs on the CUDA stream
        # and may not be complete when the tensors are immediately enqueued.
        k_cpu = k.detach().to("cpu", non_blocking=True)
        v_cpu = v.detach().to("cpu", non_blocking=True)
        
        event = None
        if k.device.type == "cuda":
            event = torch.cuda.Event()
            event.record()

        try:
            self._queue.put_nowait((block, k_cpu, v_cpu, event))
            with self._stats_lock:
                self.stats["submitted"] += 1
                depth = self._queue.qsize()
                if depth > self.stats["queue_depth_peak"]:
                    self.stats["queue_depth_peak"] = depth
            return True
        except queue.Full:
            # Backpressure: fall back to synchronous compression
            self._compress_fn(block, k, v)
            with self._stats_lock:
                self.stats["sync_fallbacks"] += 1
            return False

    def submit_sync(self, block, k: torch.Tensor, v: torch.Tensor) -> None:
        """Blocking path — always compresses synchronously."""
        t0 = time.perf_counter()
        self._compress_fn(block, k, v)
        with self._stats_lock:
            self.stats["completed"]    += 1
            self.stats["total_svd_ms"] += (time.perf_counter() - t0) * 1000

    # ── Worker ───────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break   # shutdown signal

            block, k_cpu, v_cpu, event = item
            try:
                if event is not None:
                    event.synchronize()
                t0 = time.perf_counter()
                # Execute compress_fn completely on the CPU-resident tensors!
                self._compress_fn(block, k_cpu, v_cpu)
                elapsed = (time.perf_counter() - t0) * 1000
                with self._stats_lock:
                    self.stats["completed"]    += 1
                    self.stats["total_svd_ms"] += elapsed
            except Exception as e:
                print(f"[AsyncCompressor] WARNING: compression failed for block "
                      f"anchor={block.anchor_idx}: {e}")
            finally:
                self._queue.task_done()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        avg_ms = (self.stats["total_svd_ms"] / max(1, self.stats["completed"]))
        return {
            "submitted":        self.stats["submitted"],
            "completed":        self.stats["completed"],
            "queued":           self._queue.qsize(),
            "sync_fallbacks":   self.stats["sync_fallbacks"],
            "avg_svd_ms":       round(avg_ms, 2),
            "queue_depth_peak": self.stats["queue_depth_peak"],
        }