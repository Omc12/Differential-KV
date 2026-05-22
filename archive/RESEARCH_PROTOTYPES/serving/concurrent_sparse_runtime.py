import time
import asyncio
from typing import List, Dict, Any
from .multi_user_sparse_scheduler import MultiUserSparseScheduler
from .adaptive_request_balancer import AdaptiveRequestBalancer

class ConcurrentSparseRuntime:
    """
    Main orchestration engine for concurrent multi-user sparse serving.
    Handles request lifecycle and worker delegation.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduler = MultiUserSparseScheduler()
        self.balancer = AdaptiveRequestBalancer()
        self.is_active = False

    async def serve_request(self, user_id: str, prompt: str):
        """
        Entry point for a single user request.
        """
        start_time = time.perf_counter()
        
        # 1. Schedule request
        ticket = await self.scheduler.enqueue(user_id, prompt)
        
        # 2. Balance to worker
        worker = self.balancer.get_best_worker(ticket)
        
        # 3. Execute (Simulated)
        result = await worker.execute(ticket)
        
        end_time = time.perf_counter()
        result['latency'] = end_time - start_time
        return result

    def get_status(self):
        return {
            "active_users": self.scheduler.get_active_count(),
            "queue_depth": self.scheduler.get_queue_depth(),
            "avg_latency": self.balancer.get_avg_latency()
        }
