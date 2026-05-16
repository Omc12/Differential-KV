import asyncio
from typing import Dict, Any

class MultiUserSparseScheduler:
    """
    Sparse-aware batching and scheduling for multi-user workloads.
    Prioritizes context-reuse and retrieval stability.
    """
    def __init__(self):
        self.queue = asyncio.Queue()
        self.active_requests = {}

    async def enqueue(self, user_id: str, prompt: str):
        ticket = {
            "user_id": user_id,
            "prompt_len": len(prompt),
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.queue.put(ticket)
        self.active_requests[user_id] = ticket
        return ticket

    def get_active_count(self):
        return len(self.active_requests)

    def get_queue_depth(self):
        return self.queue.qsize()

    async def get_next_batch(self, batch_size: int = 4):
        batch = []
        for _ in range(min(batch_size, self.get_queue_depth())):
            batch.append(await self.queue.get())
        return batch
