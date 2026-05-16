import torch

class LayerStabilityMonitor:
    """
    Monitors the stability of attention across layers to detect potential collapse.
    """

    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.last_entropy = None

    def check_stability(self, current_entropy):
        """
        Detects sudden drops or spikes in attention entropy that might 
        indicate numerical instability or 'resonance lock' (though we don't call it that now).
        """
        if self.last_entropy is None:
            self.last_entropy = current_entropy
            return True, "Stable (Initial)"
            
        drift = abs(current_entropy - self.last_entropy)
        self.last_entropy = current_entropy
        
        if drift > self.threshold:
            return False, f"Unstable: Entropy drift {drift:.4f} exceeds threshold {self.threshold}"
            
        return True, "Stable"

if __name__ == "__main__":
    monitor = LayerStabilityMonitor(threshold=0.1)
    print(monitor.check_stability(0.5))
    print(monitor.check_stability(0.55))
    print(monitor.check_stability(0.7)) # Should be unstable
