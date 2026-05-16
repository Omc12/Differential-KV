class ContextCollisionDetector:
    """
    Identifies interference between concurrent contexts.
    Detects if User A's retrieval is being corrupted by User B's noise.
    """
    def __init__(self, noise_threshold: float = 0.01):
        self.noise_threshold = noise_threshold

    def audit_interference(self, target_retrieval: float, concurrent_noise: float):
        if concurrent_noise > self.noise_threshold:
            print(f"WARNING: Context collision detected! Noise level: {concurrent_noise:.4f}")
            return True
        return False
