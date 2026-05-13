class VRAMFragmentationGrowth:
    """
    Specifically tracks GPU memory fragmentation growth.
    High fragmentation can lead to OOM even if enough free memory exists.
    """
    def __init__(self):
        self.frag_steps = []

    def log_step(self, fragmentation_ratio: float):
        self.frag_steps.append(fragmentation_ratio)

    def is_dangerous(self, threshold: float = 0.4) -> bool:
        if not self.frag_steps:
            return False
        return self.frag_steps[-1] > threshold
