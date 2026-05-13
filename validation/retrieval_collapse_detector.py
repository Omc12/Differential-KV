from runtime.retrieval_path_tracker import RetrievalPathTracker
import torch

class RetrievalCollapseDetector:
    """
    Automated watchdog for retrieval collapse.
    Triggers 'emergency density increase' if historical retrieval drops.
    """
    def __init__(self, threshold: float = 0.05):
        self.tracker = RetrievalPathTracker()
        self.threshold = threshold

    def check_and_remediate(self, attn_weights: torch.Tensor, current_density: float) -> float:
        """
        Monitor step and return adjusted density.
        """
        self.tracker.track_step(attn_weights)
        
        if self.tracker.is_collapsing(self.threshold):
            print("WARNING: Retrieval collapse detected! Increasing density.")
            return min(1.0, current_density * 1.5)
            
        return current_density

if __name__ == "__main__":
    detector = RetrievalCollapseDetector()
    # Simulate a collapse
    dummy_attn = torch.zeros(8, 1, 100)
    dummy_attn[..., -1] = 1.0 # All attention to most recent token (collapse)
    
    density = 0.2
    for _ in range(60): # Need enough steps for the window
        density = detector.check_and_remediate(dummy_attn, density)
        
    print(f"Final density after remediation: {density}")
