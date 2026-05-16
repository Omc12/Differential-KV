"""
analysis/lowrank_analyzer.py — Phase 2.5 Objective 3

Low-rank feasibility analysis for KV deltas.

Measures:
  1. Rank stability over sequence time
  2. Rank stability across layers
  3. Head-wise low-rank structure
  4. Temporal subspace drift (do dominant directions change?)
  5. Reconstruction error vs rank curves
  6. Compute-cost estimates for LoRA-style compression

NOTE: This is EXPLORATORY. We do NOT implement LoRA compression here.
      We only determine whether it is stable and worthwhile.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


@dataclass
class LowRankProfile:
    """Low-rank properties for a single KV sequence."""
    seq_len: int
    num_heads: int
    head_dim: int
    # Energy retention by rank
    energy_at_rank: Dict[int, float] = field(default_factory=dict)
    # Reconstruction error by rank
    error_at_rank: Dict[int, float] = field(default_factory=dict)
    # Subspace drift (cosine similarity of top-k singular vectors across windows)
    subspace_drift: List[float] = field(default_factory=list)
    # Per-head rank stability (mean energy in top-1 singular value)
    per_head_rank1_energy: List[float] = field(default_factory=list)
    # Estimated compute saving vs INT8
    rank_for_90pct_energy: int = 0
    rank_for_99pct_energy: int = 0

    def summary(self) -> Dict:
        return {
            "seq_len":               self.seq_len,
            "energy_at_rank":        {str(k): round(v, 5) for k, v in self.energy_at_rank.items()},
            "error_at_rank":         {str(k): round(v, 5) for k, v in self.error_at_rank.items()},
            "mean_subspace_drift":   round(float(np.mean(self.subspace_drift)), 5)
                                     if self.subspace_drift else 0.0,
            "std_subspace_drift":    round(float(np.std(self.subspace_drift)), 5)
                                     if self.subspace_drift else 0.0,
            "rank_for_90pct":        self.rank_for_90pct_energy,
            "rank_for_99pct":        self.rank_for_99pct_energy,
            "per_head_rank1_energy": [round(v, 4) for v in self.per_head_rank1_energy],
            "mean_head_rank1":       round(float(np.mean(self.per_head_rank1_energy)), 4)
                                     if self.per_head_rank1_energy else 0.0,
        }


class LowRankAnalyzer:
    """
    Analyzes whether KV deltas are low-rank and if so, how stable that structure is.

    This determines whether LoRA-style delta compression is worth pursuing.
    """

    TEST_RANKS = [1, 2, 4, 8, 16, 32, 64]

    def analyze(
        self,
        kv_sequence: torch.Tensor,   # [seq_len, 2, heads, dim]
        anchor_positions: List[int],
        window_size: int = 128,       # tokens per temporal window for drift analysis
    ) -> LowRankProfile:
        """Full low-rank analysis on a KV sequence."""
        seq_len, _, num_heads, head_dim = kv_sequence.shape

        profile = LowRankProfile(
            seq_len=seq_len, num_heads=num_heads, head_dim=head_dim
        )

        # ── 1. Build delta matrix ────────────────────────────────────────────
        anchor_set = set(anchor_positions)
        deltas: List[torch.Tensor] = []
        last_anchor_kv = None

        for i in range(seq_len):
            kv_i = kv_sequence[i].float()
            if i in anchor_set:
                last_anchor_kv = kv_i
            else:
                if last_anchor_kv is not None:
                    deltas.append((kv_i - last_anchor_kv).flatten())

        if len(deltas) < 2:
            return profile

        # ── 2. Global SVD energy / error ────────────────────────────────────
        D = torch.stack(deltas[:min(len(deltas), 1000)])  # [N, D]
        try:
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            total_energy = (S**2).sum().item()

            # Find rank for 90% and 99% energy retention
            cumsum = (S**2).cumsum(0)
            for r, ce in enumerate(cumsum.tolist()):
                if profile.rank_for_90pct_energy == 0 and ce / total_energy >= 0.90:
                    profile.rank_for_90pct_energy = r + 1
                if profile.rank_for_99pct_energy == 0 and ce / total_energy >= 0.99:
                    profile.rank_for_99pct_energy = r + 1
                if profile.rank_for_90pct_energy > 0 and profile.rank_for_99pct_energy > 0:
                    break

            for rank in self.TEST_RANKS:
                if rank <= len(S):
                    retained = (S[:rank]**2).sum().item() / (total_energy + 1e-9)
                    profile.energy_at_rank[rank] = retained
                    approx = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
                    err = (D - approx).norm() / (D.norm() + 1e-9)
                    profile.error_at_rank[rank] = err.item()

        except Exception as e:
            print(f"  [LowRankAnalyzer] SVD failed: {e}")

        # ── 3. Temporal subspace drift ───────────────────────────────────────
        # Split deltas into temporal windows, compare top-k subspace alignment
        windows = [
            deltas[i:i + window_size]
            for i in range(0, len(deltas) - window_size, window_size // 2)
        ]
        prev_Vh = None
        for win in windows[:20]:   # cap at 20 windows
            if len(win) < 4:
                continue
            W = torch.stack(win)
            try:
                _, _, Vh_w = torch.linalg.svd(W, full_matrices=False)
                if prev_Vh is not None:
                    top_k = min(4, Vh_w.shape[0], prev_Vh.shape[0])
                    V1 = Vh_w[:top_k].float()
                    V2 = prev_Vh[:top_k].float()
                    # Subspace similarity: principal angles
                    gram = (V1 @ V2.T).abs()
                    sim  = gram.diagonal().mean().item()
                    profile.subspace_drift.append(1.0 - sim)  # drift = 1 - similarity
                prev_Vh = Vh_w
            except Exception:
                pass

        # ── 4. Per-head analysis ─────────────────────────────────────────────
        # For each attention head, build its own delta matrix and check rank-1 energy
        for h in range(min(num_heads, 16)):   # cap at 16 heads
            head_deltas = []
            last_anchor_kv = None
            for i in range(seq_len):
                kv_i = kv_sequence[i].float()
                if i in anchor_set:
                    last_anchor_kv = kv_i
                else:
                    if last_anchor_kv is not None:
                        # K-head delta: [head_dim], V-head delta: [head_dim]
                        k_delta = (kv_i[0, h, :] - last_anchor_kv[0, h, :])
                        v_delta = (kv_i[1, h, :] - last_anchor_kv[1, h, :])
                        head_deltas.append(torch.cat([k_delta, v_delta]))

            if len(head_deltas) < 4:
                profile.per_head_rank1_energy.append(0.0)
                continue

            Hmat = torch.stack(head_deltas[:500])
            try:
                _, Sh, _ = torch.linalg.svd(Hmat, full_matrices=False)
                rank1_energy = (Sh[0]**2 / (Sh**2).sum()).item()
                profile.per_head_rank1_energy.append(rank1_energy)
            except Exception:
                profile.per_head_rank1_energy.append(0.0)

        return profile

    def estimate_lowrank_compute_cost(
        self, seq_len: int, num_heads: int, head_dim: int, rank: int
    ) -> Dict:
        """
        Estimate theoretical compute costs for LoRA-style delta storage.

        INT8 delta: seq_len * 2 * heads * head_dim bytes
        LoRA delta: rank * (2 * heads * head_dim) + seq_len * rank bytes

        Returns dict with size comparison and break-even sequence length.
        """
        # INT8 size (current baseline)
        int8_bytes = seq_len * 2 * num_heads * head_dim  # 1 byte per element

        # Low-rank storage: basis (rank x D) + coords (N x rank)
        dim = 2 * num_heads * head_dim
        lr_basis_bytes = rank * dim * 4       # float32 basis
        lr_coords_bytes = seq_len * rank * 2  # float16 coords
        lr_total = lr_basis_bytes + lr_coords_bytes

        # Break-even: when is LoRA smaller than INT8?
        # seq * 1 > rank * dim * 4 + seq * rank * 2
        # seq * (1 - rank*2) > rank * dim * 4
        # seq > rank * dim * 4 / (1 - rank*2)  [requires rank < 0.5 = 1 byte / 2 bytes]
        if rank * 2 < 1:
            breakeven = int(rank * dim * 4 / (1 - rank * 2))
        else:
            breakeven = None  # LoRA is always larger for this rank

        return {
            "rank":            rank,
            "int8_bytes":      int8_bytes,
            "lowrank_bytes":   lr_total,
            "ratio":           round(int8_bytes / (lr_total + 1e-9), 3),
            "breakeven_seq":   breakeven,
            "lr_smaller":      lr_total < int8_bytes,
        }
