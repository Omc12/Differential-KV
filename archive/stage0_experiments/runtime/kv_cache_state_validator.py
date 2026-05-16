import torch

class KVCacheStateValidator:
    """
    Validates the consistency and integrity of the KV cache during live inference.
    Detects drift between dense and sparse representations.
    """
    def __init__(self, threshold=1e-3):
        self.threshold = threshold

    def validate_reconstruction(self, original_kv, reconstructed_kv):
        """
        Compares original dense KV with reconstructed sparse KV.
        Returns error metrics and a boolean indicating if it's within tolerance.
        """
        mse = torch.mean((original_kv - reconstructed_kv) ** 2)
        max_err = torch.max(torch.abs(original_kv - reconstructed_kv))
        
        is_valid = mse < self.threshold
        
        return {
            "mse": mse.item(),
            "max_error": max_err.item(),
            "is_valid": is_valid
        }

    def check_nans(self, kv_tensor):
        """
        Checks for NaNs or Inf in the KV cache.
        """
        return torch.isnan(kv_tensor).any() or torch.isinf(kv_tensor).any()
