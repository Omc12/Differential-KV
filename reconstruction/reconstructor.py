"""
reconstruction/reconstructor.py

KVReconstructor: reconstructs KV tensors on-demand from anchor + delta storage.

Key responsibilities:
  - Single-token reconstruction: anchor_kv + dequantize(delta)
  - Grouped reconstruction: reconstruct a contiguous token range efficiently
    (one anchor lookup + batch dequantize, avoiding redundant anchor fetches)
  - Reconstruction error measurement
  - Timing instrumentation hooks (used by profiler)

Design principle: reconstruction should never traverse more than one
delta hop from the anchor. The AnchorManager guarantees this because
deltas are always stored relative to the IMMEDIATELY preceding anchor.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from anchor_logic.anchor_manager import AnchorManager
from compression.quantization import dequantize_int8


@dataclass
class ReconstructionResult:
    """Result of a grouped reconstruction call."""
    kv: torch.Tensor                   # [range_len, 2, heads, head_dim]
    token_start: int
    token_end: int                     # inclusive
    elapsed_ms: float = 0.0
    anchor_loads: int = 0              # how many distinct anchors were loaded
    delta_loads: int = 0               # how many deltas were dequantized
    bytes_read: int = 0                # estimated bytes touched

    @property
    def num_tokens(self) -> int:
        return self.token_end - self.token_start + 1


class KVReconstructor:
    """
    Reconstructs KV tensors from a compressed AnchorManager.

    Parameters
    ----------
    manager : AnchorManager
        A fully compressed anchor manager for one layer.
    target_dtype : torch.dtype
        Output dtype for reconstructed tensors (default FP16).
    """

    def __init__(self, manager: AnchorManager, target_dtype: torch.dtype = torch.float16):
        self.manager = manager
        self.dtype = target_dtype

    def reconstruct_token(self, token_idx: int) -> torch.Tensor:
        """
        Reconstruct KV for a single token.

        Returns
        -------
        torch.Tensor — shape [2, heads, head_dim]
        """
        if self.manager.is_anchor(token_idx):
            return self.manager.anchors[token_idx].to(self.dtype)

        anchor_idx, anchor_kv = self.manager.get_preceding_anchor(token_idx)
        q_delta = self.manager.get_delta(token_idx)

        if q_delta is None:
            raise KeyError(f"No delta found for token {token_idx} (not an anchor either)")

        delta = dequantize_int8(q_delta, target_dtype=torch.float32)
        return (anchor_kv.float() + delta).to(self.dtype)

    def reconstruct_range(
        self,
        token_start: int,
        token_end: int,          # inclusive
    ) -> ReconstructionResult:
        """
        Grouped reconstruction: reconstruct all tokens in [token_start, token_end].

        This is the key efficiency gain over token-by-token reconstruction:
        within a single anchor segment, we load the anchor once and
        dequantize all deltas in one pass.

        Parameters
        ----------
        token_start : int — first token index (inclusive)
        token_end   : int — last token index (inclusive)

        Returns
        -------
        ReconstructionResult
        """
        t0 = time.perf_counter()
        num_tokens = token_end - token_start + 1

        # Pre-allocate output tensor
        sample = self.manager.anchors[self.manager.index_list[0]]
        _, num_heads, head_dim = sample.shape
        out = torch.empty(num_tokens, 2, num_heads, head_dim, dtype=self.dtype)

        anchor_loads = 0
        delta_loads = 0
        bytes_read = 0

        # Group tokens by their preceding anchor to minimize anchor loads
        # Build segments: list of (anchor_idx, [token_indices])
        segments: Dict[int, List[int]] = {}
        for token_idx in range(token_start, token_end + 1):
            anchor_idx, _ = self.manager.get_preceding_anchor(token_idx)
            if anchor_idx not in segments:
                segments[anchor_idx] = []
            segments[anchor_idx].append(token_idx)

        for anchor_idx, token_list in segments.items():
            anchor_kv = self.manager.anchors[anchor_idx].float()
            anchor_bytes = anchor_kv.numel() * 4  # float32 intermediate
            bytes_read += anchor_bytes
            anchor_loads += 1

            for token_idx in token_list:
                out_idx = token_idx - token_start

                if self.manager.is_anchor(token_idx):
                    out[out_idx] = anchor_kv.to(self.dtype)
                    bytes_read += anchor_kv.numel() * 2  # FP16 read
                else:
                    q_delta = self.manager.get_delta(token_idx)
                    if q_delta is None:
                        raise KeyError(f"Missing delta for token {token_idx}")

                    delta = dequantize_int8(q_delta, target_dtype=torch.float32)
                    out[out_idx] = (anchor_kv + delta).to(self.dtype)
                    # bytes: int8 delta data + scale
                    bytes_read += q_delta.nbytes()
                    delta_loads += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ReconstructionResult(
            kv=out,
            token_start=token_start,
            token_end=token_end,
            elapsed_ms=elapsed_ms,
            anchor_loads=anchor_loads,
            delta_loads=delta_loads,
            bytes_read=bytes_read,
        )

    def measure_error(
        self, original: torch.Tensor, token_start: int, token_end: int
    ) -> Dict[str, float]:
        """
        Reconstruct a range and measure error against the original KV.

        Parameters
        ----------
        original    : torch.Tensor [seq_len, 2, heads, head_dim] — ground truth
        token_start : int
        token_end   : int (inclusive)

        Returns
        -------
        dict with keys: 'mean_l2', 'max_l2', 'mean_relative', 'max_relative'
        """
        result = self.reconstruct_range(token_start, token_end)
        recon = result.kv.float()
        orig = original[token_start:token_end + 1].float()

        diff = (orig - recon)
        l2_per_token = torch.linalg.vector_norm(diff, dim=(-1, -2, -3))   # [num_tokens]
        orig_norm_per_token = torch.linalg.vector_norm(orig, dim=(-1, -2, -3))

        rel_per_token = l2_per_token / (orig_norm_per_token + 1e-9)

        return {
            "mean_l2": l2_per_token.mean().item(),
            "max_l2": l2_per_token.max().item(),
            "mean_relative": rel_per_token.mean().item(),
            "max_relative": rel_per_token.max().item(),
        }
