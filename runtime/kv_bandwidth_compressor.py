import torch
import torch.nn as nn

class KVBandwidthCompressor:
    """
    Reduces KV movement by applying low-bit quantization or projection 
    during VRAM-RAM transfer and inter-node movement.
    """
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio

    def compress(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compresses KV tensor for bandwidth-efficient movement.
        Uses simple FP8-like quantization for simulation.
        """
        # In a real system, this might involve bit-packing or learned projection
        scale = kv_tensor.abs().max() + 1e-6
        compressed = (kv_tensor / scale * 127).to(torch.int8)
        return compressed, scale

    def decompress(self, compressed_kv: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Restores KV tensor from compressed format.
        """
        return compressed_kv.to(torch.float16) * (scale / 127)
