import torch
import numpy as np
from typing import Dict, List

class DistributedEntropyMaps:
    """
    Generates resonance evolution maps and manifold entropy flows.
    Visualizes cognitive energy evolution across GPUs.
    """
    def __init__(self, n_gpus: int, d_model: int):
        self.n_gpus = n_gpus
        self.d_model = d_model
        self.entropy_history = [[] for _ in range(n_gpus)]
        
    def log_gpu_entropy(self, gpu_id: int, entropy_map: torch.Tensor):
        """
        entropy_map: (seq_len,) tensor of entropy values.
        """
        self.entropy_history[gpu_id].append(entropy_map.cpu().numpy())
        
    def generate_resonance_map(self) -> np.ndarray:
        """
        Aggregates entropy into a global heatmap.
        """
        # Average entropy over time for each GPU
        avg_entropy = np.zeros(self.n_gpus)
        for i in range(self.n_gpus):
            if self.entropy_history[i]:
                avg_entropy[i] = np.mean([np.mean(h) for h in self.entropy_history[i]])
                
        return avg_entropy

    def get_energy_evolution(self) -> Dict[int, List[float]]:
        energy_map = {}
        for i in range(self.n_gpus):
            energy_map[i] = [float(np.mean(h)) for h in self.entropy_history[i]]
        return energy_map
