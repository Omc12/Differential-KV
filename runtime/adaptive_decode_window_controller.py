import time
from typing import Dict, List, Any

class AdaptiveDecodeWindowController:
    """
    STAGE 2 DQO: Adaptive Decode Window Controller.
    Dynamically tunes decode window sizes based on queue pressure and throughput.
    """
    def __init__(self, 
                 min_window_size: int = 1, 
                 max_window_size: int = 128,
                 target_latency_ms: float = 50.0):
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.target_latency_ms = target_latency_ms
        
        self.current_window_size = min_window_size
        self.last_adjustment_ts = time.time()
        
    def adjust_window(self, queue_depth: int, overlap_count: int, current_latency_ms: float, throughput_tps: float) -> int:
        """
        Logic:
        1. If queue is deep or overlap is high, and latency is within target, increase window size.
        2. If latency exceeds target, decrease window size.
        3. If both are low, decay towards min window size.
        """
        now = time.time()
        if now - self.last_adjustment_ts < 0.1: # 100ms cooldown
            return self.current_window_size
            
        pressure = max(queue_depth, overlap_count)
            
        if current_latency_ms > self.target_latency_ms:
            # Latency pressure: scale down
            self.current_window_size = max(self.min_window_size, int(self.current_window_size * 0.8))
        elif pressure > self.current_window_size:
            # High pressure and safe latency: scale up
            self.current_window_size = min(self.max_window_size, int(self.current_window_size * 1.5) + 1)
        elif pressure < self.current_window_size // 2:
            # Low pressure: decay
            self.current_window_size = max(self.min_window_size, self.current_window_size - 1)
            
        self.last_adjustment_ts = now
        return self.current_window_size

    def get_config(self) -> Dict[str, Any]:
        return {
            "current_window_size": self.current_window_size,
            "min_window_size": self.min_window_size,
            "max_window_size": self.max_window_size,
            "target_latency_ms": self.target_latency_ms
        }
