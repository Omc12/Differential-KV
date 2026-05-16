"""
benchmarks/kv_generator.py

Synthetic KV tensor generator for offline simulation.

Generates KV tensors that mimic realistic transformer KV-cache distributions:

  - 'gaussian'    : i.i.d. normal — worst case for delta compression
  - 'smooth'      : slowly varying signal — best case (high locality)
  - 'mixed'       : mostly smooth with occasional sharp spikes
  - 'real_approx' : mimics empirical KV distributions from transformer
                    research (low-rank structure + outliers)

All outputs are FP16 tensors shaped [seq_len, 2, num_heads, head_dim].

This is the ground truth that everything else reconstructs from.
"""

import torch
import math
from typing import Literal

KVMode = Literal["gaussian", "smooth", "mixed", "real_approx", "multilingual", "retrieval"]


class KVGenerator:
    """
    Generates synthetic KV tensors for benchmarking.

    Parameters
    ----------
    num_heads : int
    head_dim  : int
    dtype     : torch.dtype (default FP16)
    seed      : int — for reproducibility
    """

    def __init__(
        self,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        seed: int = 42,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.seed = seed
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def generate(self, seq_len: int, mode: KVMode = "mixed") -> torch.Tensor:
        """
        Generate a KV sequence.

        Parameters
        ----------
        seq_len : int — number of tokens
        mode    : KVMode — distribution type

        Returns
        -------
        torch.Tensor — shape [seq_len, 2, num_heads, head_dim], dtype FP16
        """
        if mode == "gaussian":
            return self._gaussian(seq_len)
        elif mode == "smooth":
            return self._smooth(seq_len)
        elif mode == "mixed":
            return self._mixed(seq_len)
        elif mode == "real_approx":
            return self._real_approx(seq_len)
        elif mode == "multilingual":
            return self._multilingual(seq_len)
        elif mode == "retrieval":
            return self._retrieval(seq_len)
        else:
            raise ValueError(f"Unknown KV mode: {mode}")

    def _gaussian(self, seq_len: int) -> torch.Tensor:
        """Purely random — worst case for compression."""
        kv = torch.randn(seq_len, 2, self.num_heads, self.head_dim,
                         generator=self._rng)
        return kv.to(self.dtype)

    def _smooth(self, seq_len: int) -> torch.Tensor:
        """
        Slow sinusoidal drift — best case for delta compression.
        Simulates contexts where KV states evolve gradually.
        """
        t = torch.linspace(0, 4 * math.pi, seq_len)
        # Base signal: different frequency per head
        kv = torch.zeros(seq_len, 2, self.num_heads, self.head_dim)
        for h in range(self.num_heads):
            freq = 1.0 + h * 0.1
            signal = torch.sin(freq * t).unsqueeze(-1)  # [seq_len, 1]
            # Add small random noise
            noise = torch.randn(seq_len, self.head_dim, generator=self._rng) * 0.05
            kv[:, 0, h, :] = signal + noise  # K
            kv[:, 1, h, :] = signal * 0.9 + noise * 0.8  # V (slightly different)
        return kv.to(self.dtype)

    def _mixed(self, seq_len: int) -> torch.Tensor:
        """
        Mostly smooth with occasional sharp discontinuities.
        Most realistic for transformer attention patterns.
        'Spike' positions simulate topic/context shifts.
        """
        base = self._smooth(seq_len).float()
        # Add spikes at roughly every 256 tokens
        spike_interval = 256
        spike_magnitude = 3.0
        spike_positions = list(range(spike_interval, seq_len, spike_interval))
        for pos in spike_positions:
            end = min(pos + 8, seq_len)
            noise = torch.randn(end - pos, 2, self.num_heads, self.head_dim,
                                generator=self._rng)
            base[pos:end] += noise * spike_magnitude
        return base.to(self.dtype)

    def _real_approx(self, seq_len: int) -> torch.Tensor:
        """
        Approximates empirically observed KV distributions:
          - Low-rank dominant component (simulates attention sink / value clusters)
          - Heavy-tailed outliers on a subset of dimensions
          - Smooth drift over position
        
        Based on observations from: LLM.int8(), SnapKV, and H2O research.
        """
        kv = torch.zeros(seq_len, 2, self.num_heads, self.head_dim)

        # Low-rank component: rank-8 shared structure
        rank = 8
        U = torch.randn(self.num_heads * self.head_dim, rank, generator=self._rng)
        V = torch.randn(seq_len, rank, generator=self._rng) * 0.3  # low amplitude drift
        low_rank = (V @ U.T).reshape(seq_len, self.num_heads, self.head_dim)

        # Attention sink: first token dominates in keys
        sink_weight = torch.zeros(seq_len)
        sink_weight[0] = 5.0

        # Outlier dimensions (1% of dims have large activations)
        outlier_dims = torch.randperm(self.head_dim, generator=self._rng)[:max(1, self.head_dim // 100)]

        for kv_idx in range(2):
            kv[:, kv_idx, :, :] = low_rank
            # Outlier channels
            kv[:, kv_idx, :, outlier_dims] *= 4.0
            # Add small random noise
            kv[:, kv_idx, :, :] += torch.randn(
                seq_len, self.num_heads, self.head_dim, generator=self._rng) * 0.1

        return kv.to(self.dtype)

    def _multilingual(self, seq_len: int) -> torch.Tensor:
        """
        High cross-token variance, simulating switching between languages.
        Frequent but structured 'jumps' in KV space.
        """
        base = self._smooth(seq_len).float()
        curr = 0
        while curr < seq_len:
            jump = torch.randint(32, 64, (1,), generator=self._rng).item()
            end = min(curr + jump, seq_len)
            shift = torch.randn(1, 2, self.num_heads, self.head_dim, generator=self._rng) * 2.0
            base[curr:end] += shift
            curr = end
        return base.to(self.dtype)

    def _retrieval(self, seq_len: int) -> torch.Tensor:
        """
        Simulates retrieval-heavy prompts (needle-in-a-haystack).
        Sparse, extremely high-magnitude correlations to specific past tokens.
        """
        base = self._mixed(seq_len).float()
        n_needles = max(1, seq_len // 100)
        for _ in range(n_needles):
            idx = torch.randint(0, seq_len, (1,), generator=self._rng).item()
            base[idx] *= 10.0
        return base.to(self.dtype)

    def nbytes_fp16(self, seq_len: int) -> int:
        """Bytes to store this sequence in FP16."""
        return seq_len * 2 * self.num_heads * self.head_dim * 2

    def nbytes_int8(self, seq_len: int) -> int:
        """Bytes to store this sequence in INT8 (naive)."""
        return seq_len * 2 * self.num_heads * self.head_dim * 1
