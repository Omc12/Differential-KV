import torch
from typing import Dict, Any

class SparseNeuronParticipationTracker:
    """
    Tracks real MLP sparsity during runtime.
    Measures active neurons and skipped FFN blocks.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.stats = {
            "total_neurons": 0,
            "active_neurons": 0,
            "total_blocks": 0,
            "active_blocks": 0,
            "skipped_mlp_flops": 0
        }

    def record_ffn_step(self, total_neurons: int, active_neurons: int, total_blocks: int, active_blocks: int):
        self.stats["total_neurons"] += total_neurons
        self.stats["active_neurons"] += active_neurons
        self.stats["total_blocks"] += total_blocks
        self.stats["active_blocks"] += active_blocks
        
        # Estimate FLOP reduction based on neuron participation
        sparsity_ratio = 1.0 - (active_neurons / total_neurons) if total_neurons > 0 else 0
        self.stats["skipped_mlp_flops"] += int(sparsity_ratio * 100) # Arbitrary unit for tracking

    def get_metrics(self) -> Dict[str, float]:
        total_n = self.stats["total_neurons"]
        active_n = self.stats["active_neurons"]
        total_b = self.stats["total_blocks"]
        active_b = self.stats["active_blocks"]

        return {
            "active_neuron_ratio": active_n / total_n if total_n > 0 else 1.0,
            "active_block_ratio": active_b / total_b if total_b > 0 else 1.0,
            "mlp_flop_reduction": (1.0 - (active_n / total_n)) * 100 if total_n > 0 else 0,
            "skipped_ffn_blocks": total_b - active_b
        }

# Global singleton
tracker = SparseNeuronParticipationTracker()
