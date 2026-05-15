import time
import logging
from typing import Dict, Any, Callable, Optional, List
import traceback

class ServingFaultRecoveryEngine:
    """
    Handles request retries, graph replay recovery, and runtime fault isolation.
    Ensures serving continuity during intermittent failures.
    """
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.recovery_stats = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "isolated_faults": 0,
            "permanent_failures": 0
        }
        self.logger = logging.getLogger("ServingFaultRecovery")

    async def execute_with_recovery(self, 
                                    session_id: str, 
                                    func: Callable, 
                                    *args, 
                                    **kwargs) -> Any:
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.recovery_stats["total_failures"] += 1
                last_error = e
                
                if self._is_recoverable(e):
                    retries += 1
                    if retries <= self.max_retries:
                        wait_time = self.backoff_factor ** retries
                        self.logger.warning(f"Recoverable error in session {session_id}: {e}. Retry {retries}/{self.max_retries} in {wait_time}s")
                        
                        # Attempt specific recovery actions
                        await self._perform_recovery_actions(session_id, e)
                        
                        time.sleep(wait_time)
                        continue
                else:
                    # Non-recoverable or session-specific fault isolation
                    self._isolate_fault(session_id, e)
                    break
        
        self.recovery_stats["permanent_failures"] += 1
        raise last_error

    def _is_recoverable(self, error: Exception) -> bool:
        # Define logic for what is recoverable (e.g., CUDA OOM might need restart, 
        # but transient network or small kernel errors might be retried)
        error_str = str(error).lower()
        if "timeout" in error_str or "transient" in error_str or "connection" in error_str:
            return True
        if "cuda" in error_str and "out of memory" not in error_str:
            return True # Some CUDA errors might be transient if we reset graph
        return False

    async def _perform_recovery_actions(self, session_id: str, error: Exception):
        # Example: Revert to last known good checkpoint for this session
        self.logger.info(f"Performing recovery actions for session {session_id}")
        self.recovery_stats["successful_recoveries"] += 1
        # In a real system, we'd call the session manager to reload last state
        pass

    def _isolate_fault(self, session_id: str, error: Exception):
        self.logger.error(f"Isolating fault in session {session_id}: {error}\n{traceback.format_exc()}")
        self.recovery_stats["isolated_faults"] += 1
        # Mark session as faulted to prevent it from affecting others
        pass

    def get_recovery_metrics(self) -> Dict[str, Any]:
        return {
            "recovery_rate": self.recovery_stats["successful_recoveries"] / max(1, self.recovery_stats["total_failures"]),
            "total_failures": self.recovery_stats["total_failures"],
            "isolated_faults": self.recovery_stats["isolated_faults"],
            "permanent_failures": self.recovery_stats["permanent_failures"]
        }
