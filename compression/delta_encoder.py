"""
compression/delta_encoder.py

DeltaEncoder: wraps quantization into a clean encode/decode API
used by the AnchorManager and benchmarks.

Handles:
  - encoding a raw delta (KV residual) → QuantizedDelta
  - decoding a QuantizedDelta → float tensor
  - batch encoding of a full sequence given anchor positions
"""

from typing import List, Tuple, Dict
import torch

from .quantization import quantize_int8, dequantize_int8, QuantizedDelta


class DeltaEncoder:
    """
    Stateless encoder/decoder for INT8 KV deltas.

    This class is intentionally thin — it delegates all math to
    quantize_int8 / dequantize_int8 and exists primarily to provide
    a clean interface for the reconstruction and benchmark layers.
    """

    def encode(self, kv: torch.Tensor, anchor_kv: torch.Tensor) -> QuantizedDelta:
        """
        Compute and quantize the delta: kv - anchor_kv.

        Parameters
        ----------
        kv        : torch.Tensor — token KV, shape [2, heads, head_dim]
        anchor_kv : torch.Tensor — preceding anchor KV, same shape

        Returns
        -------
        QuantizedDelta
        """
        delta = kv.float() - anchor_kv.float()
        return quantize_int8(delta)

    def decode(self, anchor_kv: torch.Tensor, q_delta: QuantizedDelta,
               dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        Reconstruct KV from anchor + quantized delta.

        Parameters
        ----------
        anchor_kv : torch.Tensor — anchor KV [2, heads, head_dim]
        q_delta   : QuantizedDelta
        dtype     : output dtype (default FP16)

        Returns
        -------
        torch.Tensor — reconstructed KV in requested dtype
        """
        delta = dequantize_int8(q_delta, target_dtype=torch.float32)
        reconstructed = anchor_kv.float() + delta
        return reconstructed.to(dtype)

    def encode_sequence(
        self,
        kv_sequence: torch.Tensor,
        anchor_positions: List[int],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, QuantizedDelta]]:
        """
        Encode a full KV sequence given known anchor positions.

        Returns
        -------
        anchor_table : dict[int, Tensor]   — full KV at anchor positions
        delta_table  : dict[int, QuantizedDelta] — quantized deltas elsewhere
        """
        seq_len = kv_sequence.shape[0]
        anchor_set = set(anchor_positions)

        anchor_table: Dict[int, torch.Tensor] = {}
        delta_table: Dict[int, QuantizedDelta] = {}

        last_anchor_kv = None
        last_anchor_idx = None

        for i in range(seq_len):
            kv = kv_sequence[i]
            if i in anchor_set:
                anchor_table[i] = kv.clone()
                last_anchor_kv = kv
                last_anchor_idx = i
            else:
                assert last_anchor_kv is not None, \
                    f"Token {i} has no preceding anchor. Token 0 must always be an anchor."
                q_delta = self.encode(kv, last_anchor_kv)
                delta_table[i] = q_delta

        return anchor_table, delta_table

    def reconstruction_error(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> float:
        """
        Compute L2 reconstruction error normalized by the original norm.

        Returns a scalar: ||original - reconstructed||_2 / ||original||_2
        """
        diff_norm = (original.float() - reconstructed.float()).norm().item()
        orig_norm = original.float().norm().item()
        return diff_norm / (orig_norm + 1e-9)

    def absolute_reconstruction_error(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> float:
        """L2 norm of absolute error (not normalized)."""
        return (original.float() - reconstructed.float()).norm().item()
