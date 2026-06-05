"""
native_core/srl/chunk_graph.py

Block-to-block similarity graph for neighborhood expansion.

When the ANN search returns top-5 semantic matches, their neighbors
in this graph are also included. This catches semantically related
blocks that may have drifted slightly in descriptor space (e.g., the
same topic discussed in different linguistic registers).

Graph construction:
  - K_semantic=6 nearest neighbors by cosine similarity (from pairwise matmul)
  - K_temporal=2 adjacent blocks in timeline (i-1, i+1)
  - Total: 8 neighbors per block → [N, 8] int32

Build cost:
  [N, DESC_DIM] @ [DESC_DIM, N] → [N, N] pairwise similarity
  For N=781, DESC_DIM=64: ~39M ops — ~0.02s on GPU, ~0.1s on CPU.
  Built once per prefill.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass
class ChunkGraph:
    """
    Adjacency structure for the block pool.

    neighbors[i] contains up to MAX_NEIGHBORS=8 row indices into the
    SemanticIndex.slot_ids array (i.e., indices into the slot list, NOT
    raw pool slot IDs). Use semantic_index.slot_ids[neighbors[i]] to
    get pool slot IDs.
    """
    neighbors: torch.Tensor   # [N, MAX_NEIGHBORS] int32, CPU-resident


def build_chunk_graph(
    desc_matrix: torch.Tensor,   # [N, DESC_DIM] float16, L2 normalized (from SemanticIndex)
    slot_ids:    torch.Tensor,   # [N] int32 pool slot IDs (from SemanticIndex)
    K_semantic:  int = 6,
    K_temporal:  int = 2,
) -> ChunkGraph:
    """
    Build block-to-block similarity graph.

    Args:
        desc_matrix: L2-normalized descriptor matrix from build_semantic_index()
        slot_ids:    pool slot IDs (same order as desc_matrix rows)
        K_semantic:  number of semantic nearest neighbors per block
        K_temporal:  number of temporal (adjacent-block) neighbors per block

    Returns:
        ChunkGraph with neighbors[i] = row indices of i's neighbors
    """
    N = desc_matrix.shape[0]
    MAX_NEIGHBORS = K_semantic + K_temporal

    if N == 0:
        return ChunkGraph(neighbors=torch.zeros((0, MAX_NEIGHBORS), dtype=torch.int32))

    if N == 1:
        return ChunkGraph(neighbors=torch.zeros((1, MAX_NEIGHBORS), dtype=torch.int32))

    # ── Pairwise cosine similarity ────────────────────────────────────────
    # desc_matrix is already L2-normalized → dot product = cosine similarity
    # Cast to float32 for numerical stability
    desc_f32 = desc_matrix.float()
    sim = desc_f32 @ desc_f32.T                          # [N, N]
    sim.fill_diagonal_(-1.0)                              # exclude self

    # ── Top-K semantic neighbors ──────────────────────────────────────────
    k_sem = min(K_semantic, N - 1)
    if k_sem > 0:
        _, sem_idx = torch.topk(sim, k=k_sem, dim=1, largest=True, sorted=True)
        # sem_idx: [N, k_sem]
    else:
        sem_idx = torch.zeros((N, 0), dtype=torch.long, device=desc_matrix.device)

    # ── Temporal neighbors (prev/next in chronological order) ─────────────
    t_neighbors = torch.zeros(N, K_temporal, dtype=torch.long, device=desc_matrix.device)
    indices = torch.arange(N, device=desc_matrix.device)
    if K_temporal >= 1:
        prev_idx = (indices - 1).clamp(min=0)
        t_neighbors[:, 0] = prev_idx
    if K_temporal >= 2:
        next_idx = (indices + 1).clamp(max=N - 1)
        t_neighbors[:, 1] = next_idx

    # ── Combine semantic + temporal ───────────────────────────────────────
    if sem_idx.shape[1] > 0:
        all_neighbors = torch.cat([sem_idx, t_neighbors], dim=1)   # [N, K_sem + K_temp]
    else:
        all_neighbors = t_neighbors

    # Pad to MAX_NEIGHBORS if needed
    if all_neighbors.shape[1] < MAX_NEIGHBORS:
        pad = torch.zeros(N, MAX_NEIGHBORS - all_neighbors.shape[1], dtype=torch.long, device=desc_matrix.device)
        all_neighbors = torch.cat([all_neighbors, pad], dim=1)

    # Move to CPU (graph is small, accessed by Python during routing)
    return ChunkGraph(neighbors=all_neighbors.int().cpu())
