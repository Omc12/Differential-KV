import torch
import torch.nn.functional as F

class ContextGradientSmoother:
    """
    PHASE 19.0B: Context Gradient Smoother.
    Smoothes importance gradients to prevent sharp discontinuities that 
    cause transformer traversal failure.
    """
    def __init__(self, kernel_size: int = 15):
        self.kernel_size = kernel_size
        # Ensure kernel size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def smooth_importance(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Applies a Gaussian or moving average filter to importance scores.
        """
        if importance_scores.shape[1] < self.kernel_size:
            return importance_scores
            
        # Reshape for 1D convolution [batch, channels, length]
        x = importance_scores.unsqueeze(1)
        
        # Simple box blur kernel
        kernel = torch.ones((1, 1, self.kernel_size), device=importance_scores.device, dtype=importance_scores.dtype) / self.kernel_size
        
        # Reflect padding to maintain size
        padding = self.kernel_size // 2
        smoothed = F.conv1d(F.pad(x, (padding, padding), mode='reflect'), kernel)
        
        return smoothed.squeeze(1)
