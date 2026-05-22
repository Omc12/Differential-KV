import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

class DynamicDecodeBatchAggregator:
    """
    STAGE 2 CDBE: Dynamic Decode Batch Aggregator.
    Coalesces incoming decode requests into optimal batch windows.
    """
    def __init__(self, engine, min_batch_size: int = 1, max_batch_size: int = 32, aggregation_window_ms: float = 2.0):
        self.engine = engine
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.aggregation_window_ms = aggregation_window_ms
        self.logger = logging.getLogger("CDBEAggregator")
        
        self.pending_requests = []
        self._aggregator_task = None
        self._is_running = False

    async def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._aggregator_task = asyncio.create_task(self._aggregation_loop())
        self.logger.info("Dynamic Decode Batch Aggregator STARTED.")

    async def stop(self):
        self._is_running = False
        if self._aggregator_task:
            await self._aggregator_task
        self.logger.info("Dynamic Decode Batch Aggregator STOPPED.")

    async def submit_request(self, session_id: str, input_ids: Any, max_tokens: int) -> asyncio.Queue:
        """
        Submits a request to the aggregator and returns an output queue for tokens.
        """
        output_queue = asyncio.Queue()
        request = {
            "session_id": session_id,
            "input_ids": input_ids,
            "max_tokens": max_tokens,
            "output_queue": output_queue,
            "arrival_ts": time.time()
        }
        self.pending_requests.append(request)
        return output_queue

    async def _aggregation_loop(self):
        while self._is_running:
            if not self.pending_requests:
                await asyncio.sleep(0.001)
                continue
            
            # Wait for a small window to collect more requests or if we already have enough
            start_wait = time.time()
            while len(self.pending_requests) < self.max_batch_size:
                elapsed = (time.time() - start_wait) * 1000
                if elapsed >= self.aggregation_window_ms:
                    break
                await asyncio.sleep(0.0005)
            
            # Batching Logic
            batch_to_submit = self.pending_requests[:self.max_batch_size]
            self.pending_requests = self.pending_requests[self.max_batch_size:]
            
            if batch_to_submit:
                for req in batch_to_submit:
                    # Direct injection into the continuous engine
                    self.engine.add_session(
                        req["session_id"], 
                        req["input_ids"], 
                        req["max_tokens"], 
                        req["output_queue"]
                    )
                
                self.logger.debug(f"Aggregated {len(batch_to_submit)} sessions into worker pool.")
            
            await asyncio.sleep(0)

    def get_batch_metrics(self) -> Dict[str, Any]:
        return {
            "pending_count": len(self.pending_requests),
            "window_ms": self.aggregation_window_ms
        }
