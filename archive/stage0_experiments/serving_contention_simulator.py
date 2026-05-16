import random
import asyncio
from typing import List, Dict, Any

class ServingContentionSimulator:
    """
    PSR System 4: Serving Contention Simulator.
    Simulates burst traffic, uneven contexts, and scheduler contention to stress the system.
    """
    def __init__(self):
        self.contention_active = False

    async def trigger_burst(self, orchestrator, burst_size: int):
        """Injects a sudden burst of requests into the orchestrator."""
        print(f"!!! PSR: Triggering burst of {burst_size} requests !!!")
        tasks = []
        for i in range(burst_size):
            user_id = f"burst-user-{i}-{random.randint(0, 1000)}"
            tasks.append(asyncio.create_task(orchestrator.simulate_user_session(user_id, "short_chat")))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def simulate_uneven_contexts(self, orchestrator, num_users: int):
        """Simulates a mix of extremely long and extremely short contexts."""
        tasks = []
        for i in range(num_users):
            user_id = f"uneven-user-{i}"
            # 20% long context, 80% short chat
            workload = "long_context" if random.random() < 0.2 else "short_chat"
            tasks.append(asyncio.create_task(orchestrator.simulate_user_session(user_id, workload)))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def inject_residency_pressure(self, runtime_executor):
        """Forces the KV cache to its limits by submitting high-memory requests."""
        # This would involve calling the runtime directly with large KV requirements
        pass

    def get_contention_scenarios(self):
        return [
            {"name": "steady_state", "concurrency": 4, "burst": 0},
            {"name": "high_concurrency", "concurrency": 16, "burst": 0},
            {"name": "burst_impact", "concurrency": 4, "burst": 12},
            {"name": "uneven_mix", "concurrency": 8, "uneven": True}
        ]
