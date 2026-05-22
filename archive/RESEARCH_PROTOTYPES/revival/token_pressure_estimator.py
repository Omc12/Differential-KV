import torch

class TokenPressureEstimator:
    """
    Estimates cache pressure and memory limits.
    """

    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens

    def estimate_pressure(self, current_tokens):
        """
        Returns a pressure signal between 0 and 1.
        """
        pressure = current_tokens / self.max_tokens
        return min(1.0, max(0.0, pressure))

    def get_vram_usage(self, num_layers, num_heads, head_dim, current_tokens, dtype_bytes=2):
        """
        Estimates VRAM usage of the KV cache in bytes.
        KV cache size = 2 * layers * heads * seq_len * head_dim * bytes_per_param
        """
        usage = 2 * num_layers * num_heads * current_tokens * head_dim * dtype_bytes
        return usage

if __name__ == "__main__":
    estimator = TokenPressureEstimator(max_tokens=2048)
    print(f"Pressure at 1024 tokens: {estimator.estimate_pressure(1024):.2f}")
    print(f"Pressure at 3000 tokens: {estimator.estimate_pressure(3000):.2f}")
    
    vram = estimator.get_vram_usage(32, 32, 128, 2048)
    print(f"Estimated VRAM for 2048 tokens (32 layers, 32 heads, 128 dim): {vram / 1024**2:.2f} MB")
