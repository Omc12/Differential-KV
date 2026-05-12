"""
analysis/layer_analyzer.py — Task 2: Layer-wise Compressibility Analysis

Analyzes compression behavior separately per transformer layer.

For each layer measures:
  - delta norm distribution
  - reconstruction error
  - compression ratio
  - anchor density
  - adaptive trigger frequency
  - local smoothness statistics

Produces layer rankings by compressibility so different layers
can be assigned different compression strategies.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy
from anchor_logic.adaptive_policies import EMAPolicy
from reconstruction.reconstructor import KVReconstructor


@dataclass
class LayerProfile:
    """Compression profile for a single transformer layer."""
    layer_idx: int
    seq_len: int
    compression_ratio: float
    anchor_density: float
    mean_recon_error: float
    max_recon_error: float
    mean_delta_rms: float
    std_delta_rms: float
    p95_delta_rms: float
    smoothness_score: float     # lower = smoother = more compressible
    compressibility_rank: int = 0

    def to_dict(self) -> Dict:
        return {
            "layer_idx":          self.layer_idx,
            "compression_ratio":  round(self.compression_ratio, 4),
            "anchor_density":     round(self.anchor_density, 4),
            "mean_recon_error":   round(self.mean_recon_error, 6),
            "max_recon_error":    round(self.max_recon_error, 6),
            "mean_delta_rms":     round(self.mean_delta_rms, 5),
            "std_delta_rms":      round(self.std_delta_rms, 5),
            "p95_delta_rms":      round(self.p95_delta_rms, 5),
            "smoothness_score":   round(self.smoothness_score, 5),
            "compressibility_rank": self.compressibility_rank,
        }


class LayerAnalyzer:
    """
    Analyzes compressibility across all transformer layers.

    Accepts a 4D KV tensor [num_layers, seq_len, 2, heads, head_dim]
    or a dict mapping layer_idx → [seq_len, 2, heads, head_dim].

    Parameters
    ----------
    strategy : anchor strategy (default: Periodic-64)
    """

    def __init__(self, strategy=None):
        self.strategy = strategy or PeriodicAnchorStrategy(interval=64)

    def analyze_all_layers(
        self,
        kv_by_layer: Dict[int, torch.Tensor],
        num_recon_queries: int = 20,
    ) -> List[LayerProfile]:
        """
        Analyze every layer and return ranked profiles.

        Parameters
        ----------
        kv_by_layer        : dict[layer_idx → Tensor[seq_len, 2, H, D]]
        num_recon_queries  : number of random windows to measure error

        Returns
        -------
        List[LayerProfile] sorted by compressibility (best first)
        """
        import random
        random.seed(42)

        profiles: List[LayerProfile] = []

        for layer_idx, kv in kv_by_layer.items():
            seq_len = kv.shape[0]

            # Compress
            manager = AnchorManager(strategy=self.strategy)
            stats   = manager.compress(kv)
            recon   = KVReconstructor(manager)

            # Measure reconstruction error
            window   = min(64, seq_len // 4)
            errors   = []
            for _ in range(num_recon_queries):
                s = random.randint(0, max(0, seq_len - window - 1))
                e = min(s + window - 1, seq_len - 1)
                err = recon.measure_error(kv, s, e)
                errors.append(err["mean_relative"])

            mean_err = float(np.mean(errors)) if errors else 0.0
            max_err  = float(np.max(errors))  if errors else 0.0

            # Delta RMS statistics
            delta_rms_vals = []
            for i in range(1, min(seq_len, 1000)):
                if not manager.is_anchor(i):
                    anchor_idx, anchor_kv = manager.get_preceding_anchor(i)
                    diff = kv[i].float() - anchor_kv.float()
                    rms  = (diff.norm() / math.sqrt(diff.numel())).item()
                    delta_rms_vals.append(rms)

            rms_arr    = np.array(delta_rms_vals) if delta_rms_vals else np.array([0.0])
            mean_rms   = float(rms_arr.mean())
            std_rms    = float(rms_arr.std())
            p95_rms    = float(np.percentile(rms_arr, 95))

            # Smoothness: mean consecutive-token RMS change
            consec_rms = []
            for i in range(1, min(seq_len, 500)):
                diff = (kv[i].float() - kv[i-1].float())
                rms  = (diff.norm() / math.sqrt(diff.numel())).item()
                consec_rms.append(rms)
            smoothness = float(np.mean(consec_rms)) if consec_rms else 0.0

            profiles.append(LayerProfile(
                layer_idx=layer_idx,
                seq_len=seq_len,
                compression_ratio=stats.compression_ratio,
                anchor_density=stats.anchor_density,
                mean_recon_error=mean_err,
                max_recon_error=max_err,
                mean_delta_rms=mean_rms,
                std_delta_rms=std_rms,
                p95_delta_rms=p95_rms,
                smoothness_score=smoothness,
            ))

        # Rank by compressibility:
        # Higher compression_ratio + lower error = more compressible
        # Use a combined score: ratio / (error + 0.001)
        scored = sorted(
            profiles,
            key=lambda p: p.compression_ratio / (p.mean_recon_error + 0.001),
            reverse=True
        )
        for rank, p in enumerate(scored):
            p.compressibility_rank = rank

        return scored

    def generate_synthetic_layers(
        self,
        num_layers: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        seed: int = 42,
    ) -> Dict[int, torch.Tensor]:
        """
        Generate synthetic KV tensors with different compressibility
        per layer — simulates real transformer layer behavior:

        - Early layers: smooth, high compressibility
        - Middle layers: mixed
        - Late layers: more variable, lower compressibility
        """
        from benchmarks.kv_generator import KVGenerator

        result = {}
        modes  = []
        for i in range(num_layers):
            frac = i / max(num_layers - 1, 1)
            if frac < 0.25:
                modes.append("smooth")
            elif frac < 0.65:
                modes.append("mixed")
            else:
                modes.append("real_approx")

        for layer_idx, mode in enumerate(modes):
            gen = KVGenerator(num_heads=num_heads, head_dim=head_dim,
                               seed=seed + layer_idx)
            result[layer_idx] = gen.generate(seq_len, mode=mode)

        return result

    def recommend_strategies(self, profiles: List[LayerProfile]) -> Dict[int, str]:
        """
        Based on compressibility profiles, recommend compression strategy per layer.
        Returns dict[layer_idx → strategy_name]
        """
        recommendations = {}
        for p in profiles:
            if p.compression_ratio > 2.5 and p.mean_recon_error < 0.005:
                rec = "aggressive"       # high compression, low error → push harder
            elif p.compression_ratio > 1.8 and p.mean_recon_error < 0.02:
                rec = "balanced"         # good ratio, acceptable error
            elif p.mean_recon_error > 0.05:
                rec = "conservative"     # error already too high → back off
            else:
                rec = "periodic_64"      # default fallback
            recommendations[p.layer_idx] = rec
        return recommendations
