import torch
from typing import Dict, Any

class SinkPriorityScheduler:
    """
    Manages allocation and density of attention sinks based on sequence length and pressure.
    """
    def __init__(self, base_sink_size: int = 4, max_sink_size: int = 32):
        self.base_sink_size = base_sink_size
        self.max_sink_size = max_sink_size
        self.current_sink_size = base_sink_size

    def adjust_sink_density(self, context_length: int, kv_pressure: float):
        """
        Dynamically adjust sink size based on context length and KV cache pressure.
        Higher pressure or longer context may require more stable sinks.
        """
        if context_length > 32768: # 32k
            self.current_sink_size = min(self.max_sink_size, self.base_sink_size * 2)
        
        if kv_pressure > 0.8:
            self.current_sink_size = min(self.max_sink_size, self.current_sink_size + 4)
        elif kv_pressure < 0.4:
            self.current_sink_size = max(self.base_sink_size, self.current_sink_size - 1)

        return self.current_sink_size

    def get_config(self) -> Dict[str, Any]:
        return {
            "current_sink_size": self.current_sink_size,
            "base_sink_size": self.base_sink_size,
            "max_sink_size": self.max_sink_size
        }
