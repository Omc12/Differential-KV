"""
analysis/subspace_tracker.py — Phase 3 Stage C

Tracks temporal subspace stability and drift.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SubspaceMetrics:
    drift_rate: float
    turnover_rate: float
    overlap_matrix: np.ndarray
    persistence_curve: List[float]
    principal_angles: List[float]

class SubspaceTracker:
    def __init__(self, rank: int = 8):
        self.rank = rank

    def compute_subspace_similarity(self, V1: torch.Tensor, V2: torch.Tensor) -> float:
        """
        Computes similarity between two subspaces defined by their basis vectors (Vh).
        V1, V2: [rank, feat_dim]
        Uses mean cosine similarity of principal angles.
        """
        # Ensure float32 for stable SVD
        V1 = V1.float()
        V2 = V2.float()
        
        # Principal angles via SVD of V1 @ V2.T
        # If V1, V2 are orthonormal, singular values are cosines of principal angles.
        try:
            gram = V1 @ V2.T
            _, s, _ = torch.linalg.svd(gram)
            return s.mean().item()
        except Exception:
            return 0.0

    def analyze_temporal_drift(self, 
                               delta_windows: List[torch.Tensor], 
                               rank: Optional[int] = None) -> SubspaceMetrics:
        """
        Analyzes how the dominant subspace evolves over time.
        delta_windows: List of [window_size, feat_dim] tensors
        """
        r = rank or self.rank
        bases = []
        
        for win in delta_windows:
            if win.shape[0] < 2:
                bases.append(None)
                continue
            try:
                _, _, Vh = torch.linalg.svd(win.float(), full_matrices=False)
                bases.append(Vh[:r])
            except Exception:
                bases.append(None)

        n = len(bases)
        overlap = np.zeros((n, n))
        drifts = []
        
        for i in range(n):
            for j in range(n):
                if bases[i] is not None and bases[j] is not None:
                    sim = self.compute_subspace_similarity(bases[i], bases[j])
                    overlap[i, j] = sim
                else:
                    overlap[i, j] = 0.0
            
            if i > 0 and bases[i] is not None and bases[i-1] is not None:
                drifts.append(1.0 - overlap[i-1, i])

        # Persistence: similarity to the FIRST window's subspace over time
        persistence = []
        if bases[0] is not None:
            for i in range(n):
                if bases[i] is not None:
                    persistence.append(overlap[0, i])
                else:
                    persistence.append(0.0)

        return SubspaceMetrics(
            drift_rate=float(np.mean(drifts)) if drifts else 0.0,
            turnover_rate=float(np.std(drifts)) if drifts else 0.0,
            overlap_matrix=overlap,
            persistence_curve=persistence,
            principal_angles=[] # Placeholder
        )

    def cross_domain_similarity(self, 
                                domain_deltas: Dict[str, torch.Tensor],
                                rank: int = 8) -> Dict[str, Dict[str, float]]:
        """Compare subspaces across different prompt domains."""
        domain_bases = {}
        for domain, deltas in domain_deltas.items():
            try:
                _, _, Vh = torch.linalg.svd(deltas.float(), full_matrices=False)
                domain_bases[domain] = Vh[:rank]
            except Exception:
                continue
        
        results = {}
        domains = list(domain_bases.keys())
        for d1 in domains:
            results[d1] = {}
            for d2 in domains:
                sim = self.compute_subspace_similarity(domain_bases[d1], domain_bases[d2])
                results[d1][d2] = sim
        return results
