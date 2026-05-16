import torch

class KVBandwidthCompressor:
    """
    On-GPU compressor for KV blocks to reduce memory bandwidth saturation.
    Uses simple FP8 or INT8 quantization for rapid transport.
    """
    def __init__(self, mode: str = "fp8"):
        self.mode = mode

    def compress(self, tensor: torch.Tensor) -> torch.Tensor:
        # Simulate compression to FP8/INT8
        # In a real kernel, this would be bit-packed
        if self.mode == "fp8":
            return tensor.to(torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.int8)
        return tensor.to(torch.int8)

    def decompress(self, compressed_tensor: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
        return compressed_tensor.to(original_dtype)

    def get_bandwidth_savings(self, original_size: int):
        # FP16 (2 bytes) -> FP8 (1 byte) = 50% savings
        if self.mode == "fp8":
            return 0.5
        return 0.0
