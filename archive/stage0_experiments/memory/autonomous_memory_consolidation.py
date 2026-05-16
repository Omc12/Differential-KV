import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np

class AutonomousMemoryConsolidation:
    """
    Handles recursive memory consolidation and attractor merging.
    Optimizes the manifold storage by extracting long-term motifs.
    """
    def __init__(self, compression_ratio: float = 0.5, merge_threshold: float = 0.9):
        self.compression_ratio = compression_ratio
        self.merge_threshold = merge_threshold
        self.archived_motifs = []

    def consolidate_manifolds(self, manifolds: torch.Tensor) -> torch.Tensor:
        """
        Consolidates a set of manifolds by merging redundant attractors.
        """
        # Compute similarity matrix between attractors
        # For simplicity, treat each point in the manifold as an attractor candidate
        if manifolds.dim() == 2:
            manifolds = manifolds.unsqueeze(0)
            
        batch_size, n, d = manifolds.shape
        consolidated_list = []
        
        for b in range(batch_size):
            m = manifolds[b]
            # Normalize for cosine similarity
            m_norm = torch.nn.functional.normalize(m, dim=-1)
            sim = torch.mm(m_norm, m_norm.t())
            
            # Find groups of similar attractors
            merged = torch.zeros(n, dtype=torch.bool)
            new_m = []
            
            for i in range(n):
                if merged[i]:
                    continue
                
                # Find all neighbors within threshold
                neighbors = sim[i] > self.merge_threshold
                merged[neighbors] = True
                
                # Average the neighbors to form a single stable attractor
                stable_attractor = m[neighbors].mean(dim=0)
                new_m.append(stable_attractor)
            
            consolidated_list.append(torch.stack(new_m))
            
        # Note: Resulting manifolds might have different sizes, 
        # return the first one or pad them if needed.
        return consolidated_list[0]

    def extract_long_term_motifs(self, manifolds: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """
        Extracts dominant reasoning motifs from the manifold.
        These are the 'most stable' patterns that survive consolidation.
        """
        # Use SVD to find the most significant components
        u, s, v = torch.pca_lowrank(manifolds, q=top_k)
        self.archived_motifs.append(v)
        return v

    def compute_redundancy_score(self, manifolds: torch.Tensor) -> float:
        """
        Measures how much redundant information is in the manifold.
        """
        m_norm = torch.nn.functional.normalize(manifolds, dim=-1)
        sim = torch.mm(m_norm, m_norm.t())
        # Average off-diagonal similarity
        n = sim.shape[0]
        redundancy = (sim.sum() - n) / (n * (n - 1) + 1e-6)
        return redundancy.item()
