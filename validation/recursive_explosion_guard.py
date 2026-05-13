import time
from typing import Dict, Any

class RecursiveExplosionGuard:
    """
    Prevents uncontrolled recursion by monitoring time and stack depth during loops.
    Fails the execution if thresholds are exceeded.
    """
    def __init__(self, time_limit_sec: float = 10.0, max_depth: int = 20):
        self.time_limit_sec = time_limit_sec
        self.max_depth = max_depth
        self.start_time = 0

    def start_monitoring(self):
        self.start_time = time.time()

    def check_safety(self, current_depth: int):
        """Checks if current recursion depth or execution time exceeds limits."""
        elapsed = time.time() - self.start_time
        if elapsed > self.time_limit_sec:
            raise TimeoutError(f"Recursive explosion detected: Time limit {self.time_limit_sec}s exceeded.")
        
        if current_depth > self.max_depth:
            raise RecursionError(f"Recursive explosion detected: Depth limit {self.max_depth} exceeded.")

    def get_guard_stats(self, current_depth: int) -> Dict[str, Any]:
        return {
            "elapsed_time": time.time() - self.start_time,
            "current_depth": current_depth,
            "safety_margin_time": self.time_limit_sec - (time.time() - self.start_time),
            "safety_margin_depth": self.max_depth - current_depth
        }
