"""
compression/quantization.py

INT8 symmetric per-tensor quantization for delta blocks.

Design notes:
- We use symmetric quantization: scale = max(|x|) / 127
- Each delta block stores: int8 tensor + float32 scale
- This is the simplest possible baseline; INT4 / block-wise can come later.
- All operations stay in PyTorch — no custom CUDA kernels yet.
"""

from dataclasses import dataclass
import torch


@dataclass
class QuantizedDelta:
    """
    A single INT8-quantized delta block.

    Attributes
    ----------
    data  : torch.Tensor [int8]  — quantized residual values
    scale : float                — dequantization scale factor
    shape : tuple                — original tensor shape (for reshape on decode)
    """
    data: torch.Tensor   # dtype=torch.int8
    scale: float
    shape: tuple

    def nbytes(self) -> int:
        """Bytes consumed by this quantized delta (int8 data + scale)."""
        return self.data.numel() * 1 + 4  # 1 byte/element + 4 bytes for float32 scale

    def compression_ratio_vs_fp16(self) -> float:
        """How much smaller is this vs storing the delta in FP16."""
        fp16_bytes = self.data.numel() * 2
        return fp16_bytes / self.nbytes()


def quantize_int8(x: torch.Tensor) -> QuantizedDelta:
    """
    Symmetric INT8 quantization of a delta tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input delta tensor (any shape, any float dtype).

    Returns
    -------
    QuantizedDelta
    """
    x_float = x.float()
    abs_max = x_float.abs().max().item()

    if abs_max < 1e-9:
        # Near-zero delta — store zeros
        scale = 1.0
    else:
        scale = abs_max / 127.0

    quantized = (x_float / scale).round().clamp(-127, 127).to(torch.int8)
    return QuantizedDelta(data=quantized, scale=scale, shape=tuple(x.shape))


def dequantize_int8(q: QuantizedDelta, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Reconstruct a float tensor from a QuantizedDelta.

    Parameters
    ----------
    q           : QuantizedDelta
    target_dtype: Output dtype (default FP16 to match baseline KV format).

    Returns
    -------
    torch.Tensor — reconstructed delta, same shape as original.
    """
    return (q.data.float() * q.scale).to(target_dtype).reshape(q.shape)
