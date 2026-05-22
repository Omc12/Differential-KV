import asyncio
from typing import List, Any

class ServingOverheadMinimizer:
    """
    EOM MODULE 3: Reduces non-model costs (serialization, streaming, queueing).
    Ensures sparse savings are not lost in the serving layer.
    """
    def __init__(self, flush_interval_tokens: int = 4):
        self.flush_interval_tokens = flush_interval_tokens
        self.pending_tokens = []
        
    async def optimize_stream(self, token_gen: Any):
        """
        Batches streaming updates to reduce serialization and network overhead.
        """
        count = 0
        async for token in token_gen:
            self.pending_tokens.append(token)
            count += 1
            
            if count >= self.flush_interval_tokens:
                yield "".join(self.pending_tokens)
                self.pending_tokens = []
                count = 0
                
        if self.pending_tokens:
            yield "".join(self.pending_tokens)

    def reduce_queue_contention(self, queue: asyncio.Queue):
        """
        Optimizes queue wakeup frequency.
        """
        # In a real system, we'd use a more efficient semaphore or batch-aware pull
        pass
