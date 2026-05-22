import torch
from typing import Dict

class EnergyPerInference:
    """
    Measures the energy footprint (Joules) of stabilized cognition.
    Includes overhead of stabilization routing vs compute savings from sparsity.
    """
    def __init__(self):
        self.gpu_pwr_baseline = 300 # Watts (e.g. H100)
        
    def calculate_energy_per_token(self, 
                                   tokens_per_sec: float, 
                                   stabilization_overhead: float) -> float:
        """
        Calculates energy (mJ) per token.
        """
        # Energy = (Power / Throughput)
        # Power includes baseline + stabilization compute
        total_pwr = self.gpu_pwr_baseline * (1.0 + stabilization_overhead)
        energy_per_token = (total_pwr / tokens_per_sec) * 1000 # mJ
        return energy_per_token

    def get_scaling_efficiency(self) -> float:
        """
        Returns the datacenter scaling efficiency (PUE-adjusted).
        """
        return 0.96
