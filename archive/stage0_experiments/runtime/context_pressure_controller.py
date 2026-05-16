import torch

class ContextPressureController:
    """
    Monitors KV cache usage and triggers pruning/eviction when memory pressure exceeds limits.
    """
    def __init__(self, vram_limit_gb: float = 24.0, safety_margin: float = 0.1):
        self.vram_limit_bytes = vram_limit_gb * 1024**3
        self.safety_threshold = self.vram_limit_bytes * (1.0 - safety_margin)

    def get_current_pressure(self) -> float:
        """
        Returns VRAM pressure ratio.
        """
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated()
        return allocated / self.vram_limit_bytes

    def should_trigger_pruning(self, current_seq_len: int) -> bool:
        """
        Decides if pruning is necessary based on pressure and context length.
        """
        pressure = self.get_current_pressure()
        
        # Trigger if pressure is high OR if we are reaching context limits
        if pressure > 0.85:
            return True
        if current_seq_len > 100000: # 100k
            return True
            
        return False

    def get_pruning_aggressive_factor(self) -> float:
        """
        Returns how aggressively to prune [0, 1].
        """
        pressure = self.get_current_pressure()
        if pressure < 0.5:
            return 0.2
        if pressure < 0.8:
            return 0.5
        return 0.8
