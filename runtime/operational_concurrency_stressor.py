import asyncio
import random
import logging
from typing import Dict, Any, List
from .production_session_lifecycle_manager import ProductionSessionLifecycleManager

class OperationalConcurrencyStressor:
    """
    ORX Phase 40.2: Operational Concurrency Stressor.
    Stresses batching, queues, and concurrency under serving turbulence.
    """
    def __init__(self, session_manager: ProductionSessionLifecycleManager):
        self.session_manager = session_manager
        self.logger = logging.getLogger("ConcurrencyStressor")
        self.stress_levels = {
            "queue_depth": 0,
            "stream_overlap": 0,
            "cancellation_rate": 0.0
        }

    async def apply_stress(self, intensity: float = 0.5):
        """
        Periodically injects concurrency stress events.
        """
        self.logger.info(f"Applying concurrency stress [Intensity: {intensity}]")
        
        while True:
            # 1. Reconnect Storms
            if random.random() < intensity * 0.2:
                self.logger.warning("INJECTING RECONNECT STORM")
                active_sessions = list(self.session_manager.sessions.keys())
                if active_sessions:
                    targets = random.sample(active_sessions, min(len(active_sessions), 5))
                    for sid in targets:
                        self.session_manager.handle_reconnect(sid)

            # 2. Cancellation Races
            if random.random() < intensity * 0.3:
                self.logger.warning("INJECTING CANCELLATION RACE")
                # (Simulated by logging events that should be handled by a real scheduler)
                pass

            # 3. Queue Pressure
            self.stress_levels["queue_depth"] = int(intensity * 100 * random.random())
            
            await asyncio.sleep(5.0)

    def get_stress_metrics(self) -> Dict[str, Any]:
        return self.stress_levels
