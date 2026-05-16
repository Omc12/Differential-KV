import numpy as np
from typing import List, Dict

class CognitiveThermodynamics:
    """
    Studies the thermodynamics of long-horizon cognition.
    Analyzes entropy growth, manifold carrying capacity, and energy scaling.
    """
    def __init__(self):
        self.energy_logs = []
        
    def calculate_gibbs_free_energy(self, entropy: float, enthalpy: float, temperature: float) -> float:
        """
        G = H - TS.
        In this context:
        H (Enthalpy) = Manifold density/coherence
        S (Entropy) = Latent disorder
        T (Temperature) = Noise level or 'learning rate'
        """
        return enthalpy - (temperature * entropy)
        
    def analyze_stability(self, entropy_trajectory: List[float], pressure_trajectory: List[float]) -> Dict:
        """
        Analyzes the thermodynamic stability of a reasoning session.
        """
        entropy_growth = np.gradient(entropy_trajectory)
        avg_growth = np.mean(entropy_growth)
        
        # Sustainable cognition requires avg_growth <= 0 (homeostasis)
        is_sustainable = avg_growth <= 0.01
        
        return {
            "avg_entropy_growth": avg_growth,
            "max_pressure_peak": np.max(pressure_trajectory),
            "is_sustainable": is_sustainable,
            "system_temperature": 1.0 # Constant for now
        }
