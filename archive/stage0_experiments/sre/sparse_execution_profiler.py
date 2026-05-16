
import torch
import time
from typing import Dict, Any, List

class SparseExecutionProfiler:
    """
    PHASE 22.0: SRE - Sparse Execution Profiler.
    Tracks execution density and sparse runtime analytics.
    """
    def __init__(self):
        self.telemetry_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        # Cumulative stats
        self.total_tokens_processed = 0
        self.total_compute_steps = 0
        self.active_elements_sum = 0
        
    def record_step(self, 
                    active_mask: torch.Tensor, 
                    total_elements: int, 
                    symbolic_continuity: float):
        """
        Records a single execution step's density and health.
        """
        active_count = active_mask.sum().item() if isinstance(active_mask, torch.Tensor) else active_mask
        ratio = active_count / total_elements if total_elements > 0 else 0
        
        step_metrics = {
            "timestamp": time.time() - self.start_time,
            "active_compute_ratio": ratio,
            "symbolic_continuity": symbolic_continuity,
            "execution_entropy": self._calculate_mask_entropy(active_mask) if isinstance(active_mask, torch.Tensor) else 0.0
        }
        
        self.telemetry_history.append(step_metrics)
        
        # Update cumulative
        self.total_compute_steps += 1
        self.active_elements_sum += active_count
        self.total_tokens_processed += 1 # Assuming 1 token per step for profiling

        if len(self.telemetry_history) > 1000:
            self.telemetry_history.pop(0)

    def _calculate_mask_entropy(self, mask: torch.Tensor) -> float:
        """
        Measures the 'health' of the sparse mask distribution.
        Low entropy might indicate 'sparse collapse' (too uniform or too empty).
        """
        if mask.dtype != torch.float:
            mask = mask.float()
        
        probs = mask / (mask.sum() + 1e-9)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        return entropy.item()

    def get_sparse_analytics(self) -> Dict[str, Any]:
        """
        Aggregates profiling data for validation.
        """
        if not self.telemetry_history:
            return {}
            
        avg_ratio = sum(m["active_compute_ratio"] for m in self.telemetry_history) / len(self.telemetry_history)
        avg_continuity = sum(m["symbolic_continuity"] for m in self.telemetry_history) / len(self.telemetry_history)
        
        # Calculate efficiency gain (theoretical)
        # Assuming 1.0 is full dense compute
        efficiency_gain = (1.0 - avg_ratio) * 100.0 # Percentage
        
        return {
            "active_compute_ratio": avg_ratio,
            "sparse_efficiency_gain_pct": efficiency_gain,
            "symbolic_continuity_avg": avg_continuity,
            "execution_entropy_health": self.telemetry_history[-1]["execution_entropy"] if self.telemetry_history else 0,
            "total_steps": self.total_compute_steps
        }
