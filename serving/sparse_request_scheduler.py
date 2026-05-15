import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import torch

@dataclass(order=True)
class SparseRequest:
    priority: int
    arrival_time: float
    session_id: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)
    future: asyncio.Future = field(default_factory=asyncio.Future, compare=False)

class SparseRequestScheduler:
    """
    Manages request prioritization, microbatching, and queue balancing.
    Optimizes for sparse inference execution.
    """
    def __init__(self, microbatch_size: int = 4, max_queue_size: int = 100):
        self.queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.microbatch_size = microbatch_size
        self.is_running = False
        self.processing_task = None
        self.stats = {
            "processed_requests": 0,
            "batches_executed": 0,
            "average_latency": 0.0
        }

    async def submit_request(self, session_id: str, payload: Dict[str, Any], priority: int = 10) -> Any:
        request = SparseRequest(
            priority=priority,
            arrival_time=time.time(),
            session_id=session_id,
            payload=payload
        )
        await self.queue.put(request)
        return await request.future

    async def start(self, runtime_executor_fn):
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_queue(runtime_executor_fn))

    async def stop(self):
        self.is_running = False
        if self.processing_task:
            await self.processing_task

    async def _process_queue(self, runtime_executor_fn):
        while self.is_running:
            if self.queue.empty():
                await asyncio.sleep(0.01)
                continue

            batch = []
            # Collect a microbatch
            try:
                # Get the first item
                request = await self.queue.get()
                batch.append(request)
                
                # Try to get more up to microbatch_size without blocking too long
                while len(batch) < self.microbatch_size and not self.queue.empty():
                    request = self.queue.get_nowait()
                    batch.append(request)
            except asyncio.QueueEmpty:
                pass

            if batch:
                await self._execute_batch(batch, runtime_executor_fn)

    async def _execute_batch(self, batch: List[SparseRequest], runtime_executor_fn):
        start_time = time.time()
        
        # In a real system, we'd group batch by session or shared sparse indices
        # For now, we execute them sequentially or in parallel depending on the runtime capability
        # Let's assume the runtime can handle a batch
        try:
            payloads = [req.payload for req in batch]
            session_ids = [req.session_id for req in batch]
            
            # Simulated execution
            results = await runtime_executor_fn(session_ids, payloads)
            
            for req, res in zip(batch, results):
                req.future.set_result(res)
                self.queue.task_done()
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
                self.queue.task_done()

        # Update stats
        duration = time.time() - start_time
        self.stats["processed_requests"] += len(batch)
        self.stats["batches_executed"] += 1
        # Simple moving average for latency
        self.stats["average_latency"] = (self.stats["average_latency"] * 0.9) + (duration / len(batch) * 0.1)

    def get_serving_metrics(self) -> Dict[str, Any]:
        return {
            "queue_depth": self.queue.qsize(),
            "processed_requests": self.stats["processed_requests"],
            "batches_executed": self.stats["batches_executed"],
            "average_latency_ms": self.stats["average_latency"] * 1000
        }
