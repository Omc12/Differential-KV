import asyncio
import random
import time
from typing import List, Dict, Any
from real_multiuser_serving_orchestrator import RealMultiUserServingOrchestrator

class RealUserPressureSimulator:
    """
    HSM System 4: Real User Pressure Simulator.
    Generates heavy serving load with long-running users and staggered arrivals.
    """
    def __init__(self, orchestrator: RealMultiUserServingOrchestrator):
        self.orchestrator = orchestrator
        self.is_running = False

    async def generate_heavy_load(self, target_concurrency: int, duration_secs: int):
        """Generates a sustained heavy load by staggering user arrivals."""
        print(f"[HSM] Generating heavy serving pressure: target_concurrency={target_concurrency}")
        self.is_running = True
        
        # Staggered arrival to simulate real-world ramping
        arrival_tasks = []
        for i in range(target_concurrency):
            arrival_delay = random.uniform(0, 1.0) # Reduced stagger for test
            task = asyncio.create_task(self._delayed_user_start(f"heavy-user-{i}", arrival_delay))
            arrival_tasks.append(task)
            
        # Run the simulator loop
        start_ts = time.time()
        while time.time() - start_ts < duration_secs and self.is_running:
            await asyncio.sleep(1.0)
            
        self.is_running = False
        self.orchestrator.is_running = False # Prevent deadlock
        await asyncio.gather(*arrival_tasks, return_exceptions=True)

    async def _delayed_user_start(self, user_id: str, delay: float):
        await asyncio.sleep(delay)
        if self.is_running:
            # Randomly select heavy workloads
            workload = random.choice(["long_context", "code_gen", "summarization"])
            await self.orchestrator.simulate_user_session(user_id, workload)

    def stop(self):
        self.is_running = False
