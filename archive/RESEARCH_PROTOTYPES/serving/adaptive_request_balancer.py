import random
import asyncio

class AdaptiveRequestBalancer:
    """
    Adaptive load balancing across sparse worker nodes.
    Accounts for VRAM pressure and retrieval hotsets.
    """
    def __init__(self, worker_count: int = 4):
        self.workers = [WorkerMock(i) for i in range(worker_count)]

    def get_best_worker(self, ticket: dict):
        # Simplistic least-loaded balancing
        return min(self.workers, key=lambda w: w.load)

    def get_avg_latency(self):
        latencies = [w.avg_latency for w in self.workers if w.avg_latency > 0]
        return sum(latencies) / len(latencies) if latencies else 0.0

class WorkerMock:
    def __init__(self, id: int):
        self.id = id
        self.load = 0
        self.avg_latency = 0.0
        self.processed = 0

    async def execute(self, ticket: dict):
        self.load += 1
        start = asyncio.get_event_loop().time()
        
        # Simulate execution
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        latency = asyncio.get_event_loop().time() - start
        self.avg_latency = (self.avg_latency * self.processed + latency) / (self.processed + 1)
        self.processed += 1
        self.load -= 1
        
        return {
            "worker_id": self.id,
            "tps": 50.0 + random.uniform(-5, 5),
            "status": "SUCCESS"
        }
