import torch
import time
from typing import List, Dict, Any

class KVGrowthTracker:
    """
    Tracks KV cache growth and memory efficiency over long sequences.
    Provides measurable metrics for pruning effectiveness.
    """
    def __init__(self):
        self.history: List[Dict[str, float]] = []

    def log_step(self, step: int, seq_len: int, kv_elements: int):
        """
        Logs KV cache statistics for a single step.
        """
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024**2
        else:
            vram_mb = 0.0
            
        self.history.append({
            "step": step,
            "seq_len": seq_len,
            "kv_elements": kv_elements,
            "vram_mb": vram_mb,
            "timestamp": time.time()
        })

    def get_growth_rate(self) -> float:
        """
        Calculates the average VRAM growth per 1000 tokens.
        """
        if len(self.history) < 2:
            return 0.0
        
        d_vram = self.history[-1]["vram_mb"] - self.history[0]["vram_mb"]
        d_tokens = self.history[-1]["seq_len"] - self.history[0]["seq_len"]
        
        if d_tokens == 0:
            return 0.0
            
        return (d_vram / d_tokens) * 1000

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_steps": len(self.history),
            "final_seq_len": self.history[-1]["seq_len"] if self.history else 0,
            "final_vram_mb": self.history[-1]["vram_mb"] if self.history else 0,
            "avg_growth_rate_mb_per_1k": self.get_growth_rate()
        }
