import torch

class CacheReuseDestroyer:
    """
    PHASE 6H: Cache-Reuse Destroyer
    Forces the GPU/CPU caches to flush between benchmark runs.
    Ensures that 'cold' retrieval performance is honestly measured.
    """
    def __init__(self):
        # Create a huge dummy tensor to flush L2/L3 caches
        self.flush_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')

    def flush(self):
        """Zeroes out caches by reading/writing to a large buffer."""
        self.flush_buffer.fill_(1)
        torch.cuda.synchronize()
        self.flush_buffer.fill_(0)
        torch.cuda.synchronize()
