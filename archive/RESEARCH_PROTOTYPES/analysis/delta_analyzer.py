"""
analysis/delta_analyzer.py — Task 4: Delta Distribution Analysis

Analyzes the statistical structure of KV delta blocks:
  - norm distribution & histogram
  - sparsity (fraction near-zero)
  - entropy estimates
  - singular value spectrum (low-rank approximability)
  - temporal smoothness (token-to-token correlation)
  - heavy-tail detection
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


@dataclass
class DeltaStats:
    """Full statistical profile of a delta block collection."""
    num_deltas: int = 0
    # Norm statistics
    rms_values: List[float] = field(default_factory=list)
    rel_change_values: List[float] = field(default_factory=list)
    # Sparsity
    sparsity_at_01: float = 0.0   # fraction with |val| < 0.1
    sparsity_at_001: float = 0.0
    # Entropy
    mean_entropy_bits: float = 0.0
    # Low-rank
    svd_energy_retention: List[float] = field(default_factory=list)  # by rank
    # Temporal
    token_correlation: float = 0.0
    # Heavy tail
    kurtosis: float = 0.0

    def summary(self) -> Dict:
        rms = self.rms_values
        return {
            "num_deltas":          self.num_deltas,
            "rms_mean":            round(float(np.mean(rms)), 5) if rms else 0,
            "rms_std":             round(float(np.std(rms)), 5)  if rms else 0,
            "rms_p50":             round(float(np.percentile(rms, 50)), 5) if rms else 0,
            "rms_p95":             round(float(np.percentile(rms, 95)), 5) if rms else 0,
            "rms_p99":             round(float(np.percentile(rms, 99)), 5) if rms else 0,
            "sparsity_01":         round(self.sparsity_at_01, 4),
            "sparsity_001":        round(self.sparsity_at_001, 4),
            "mean_entropy_bits":   round(self.mean_entropy_bits, 4),
            "token_correlation":   round(self.token_correlation, 4),
            "kurtosis":            round(self.kurtosis, 4),
            "svd_rank1_energy":    round(self.svd_energy_retention[0], 4) if self.svd_energy_retention else 0,
            "svd_rank4_energy":    round(self.svd_energy_retention[3], 4) if len(self.svd_energy_retention) > 3 else 0,
            "svd_rank8_energy":    round(self.svd_energy_retention[7], 4) if len(self.svd_energy_retention) > 7 else 0,
        }


class DeltaAnalyzer:
    """
    Compute deep statistical analysis of KV deltas.

    Usage
    -----
    analyzer = DeltaAnalyzer()
    stats = analyzer.analyze(kv_sequence, anchor_positions)
    print(stats.summary())
    """

    def analyze(
        self,
        kv_sequence: torch.Tensor,     # [seq_len, 2, heads, head_dim]
        anchor_positions: List[int],
    ) -> DeltaStats:
        """Full analysis pass over a compressed KV sequence."""
        stats = DeltaStats()
        anchor_set = set(anchor_positions)
        seq_len = kv_sequence.shape[0]

        deltas: List[torch.Tensor] = []
        last_anchor_kv = None
        last_anchor_idx = None

        for i in range(seq_len):
            kv = kv_sequence[i].float()
            if i in anchor_set:
                last_anchor_kv  = kv
                last_anchor_idx = i
            else:
                if last_anchor_kv is None:
                    continue
                delta = kv - last_anchor_kv
                deltas.append(delta)

        if not deltas:
            return stats

        stats.num_deltas = len(deltas)

        # ── RMS norms ──────────────────────────────────────────────────────────
        for i, d in enumerate(deltas):
            rms = (d.norm() / math.sqrt(d.numel())).item()
            stats.rms_values.append(rms)
            if last_anchor_kv is not None:
                anchor = kv_sequence[anchor_positions[
                    max(k for k in range(len(anchor_positions))
                        if anchor_positions[k] <= i + (anchor_positions[0]))
                ]].float()
                rel = (d.norm() / (anchor.norm() + 1e-9)).item()
                stats.rel_change_values.append(rel)

        # ── Sparsity ───────────────────────────────────────────────────────────
        all_vals = torch.cat([d.flatten() for d in deltas])
        stats.sparsity_at_01  = (all_vals.abs() < 0.1).float().mean().item()
        stats.sparsity_at_001 = (all_vals.abs() < 0.01).float().mean().item()

        # ── Kurtosis (heavy tails) ─────────────────────────────────────────────
        v = all_vals
        mean = v.mean()
        std  = v.std()
        if std > 1e-9:
            stats.kurtosis = ((((v - mean) / std) ** 4).mean() - 3).item()

        # ── Entropy estimate (histogram) ───────────────────────────────────────
        entropies = []
        for d in deltas[:min(200, len(deltas))]:   # sample for speed
            flat = d.flatten().numpy()
            hist, _ = np.histogram(flat, bins=64, density=True)
            hist = hist[hist > 0]
            dx = np.ptp(flat) / 64 + 1e-9
            entropy = -float(np.sum(hist * np.log2(hist + 1e-12) * dx))
            entropies.append(max(0.0, entropy))
        stats.mean_entropy_bits = float(np.mean(entropies)) if entropies else 0.0

        # ── Token-to-token correlation ─────────────────────────────────────────
        if len(deltas) >= 2:
            rms_arr = np.array(stats.rms_values)
            if rms_arr.std() > 1e-9:
                corr = np.corrcoef(rms_arr[:-1], rms_arr[1:])[0, 1]
                stats.token_correlation = float(corr)

        # ── SVD energy retention ───────────────────────────────────────────────
        # Flatten each delta to a vector, stack → matrix, compute SVD
        sample_deltas = deltas[:min(500, len(deltas))]
        delta_matrix  = torch.stack([d.flatten() for d in sample_deltas])  # [N, D]
        try:
            _, sv, _ = torch.linalg.svd(delta_matrix, full_matrices=False)
            total_energy = (sv ** 2).sum().item()
            cumsum = (sv ** 2).cumsum(0)
            max_rank = min(32, len(sv))
            stats.svd_energy_retention = [
                (cumsum[r].item() / (total_energy + 1e-9))
                for r in range(max_rank)
            ]
        except Exception:
            stats.svd_energy_retention = [0.0] * 32

        return stats

    def analyze_temporal_smoothness(
        self, kv_sequence: torch.Tensor
    ) -> Dict:
        """
        Measure token-to-token KV smoothness directly.
        Returns statistics on consecutive-token deltas.
        """
        seq_len = kv_sequence.shape[0]
        consec_rms = []
        for i in range(1, seq_len):
            diff = (kv_sequence[i].float() - kv_sequence[i-1].float())
            rms  = (diff.norm() / math.sqrt(diff.numel())).item()
            consec_rms.append(rms)

        arr = np.array(consec_rms)
        return {
            "mean_consecutive_rms":   round(float(arr.mean()), 5),
            "std_consecutive_rms":    round(float(arr.std()),  5),
            "p10_consecutive_rms":    round(float(np.percentile(arr, 10)), 5),
            "p90_consecutive_rms":    round(float(np.percentile(arr, 90)), 5),
            "p99_consecutive_rms":    round(float(np.percentile(arr, 99)), 5),
            "autocorrelation_lag1":   round(float(np.corrcoef(arr[:-1], arr[1:])[0,1])
                                            if len(arr) > 1 else 0.0, 4),
        }
