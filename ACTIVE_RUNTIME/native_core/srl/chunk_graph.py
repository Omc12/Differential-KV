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
from typing import Optional

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
    weights:   Optional[torch.Tensor] = None   # [N, MAX_NEIGHBORS] float32, CPU-resident
    parent_landmarks: Optional[torch.Tensor] = None  # [L] int32 CPU-resident pool slot IDs of chunk parents
    parent_to_children: Optional[dict] = None        # parent slot ID -> list of child slot IDs
    slot_to_parent: Optional[dict] = None            # slot ID -> parent slot ID


def build_chunk_graph(
    desc_matrix:  torch.Tensor,   # [N, DESC_DIM] float16, L2 normalized (from SemanticIndex)
    slot_ids:     torch.Tensor,   # [N] int32 pool slot IDs (from SemanticIndex)
    K_semantic:   int = 6,
    K_temporal:   int = 2,
    inv_index:    Optional["InvertedTokenIndex"] = None,
    overlap_threshold: float = 0.15, # threshold on relative keyword overlap (15%)
    blocks:       Optional[list] = None, # Optional list of KVBlock objects in chronological order
) -> ChunkGraph:
    """
    Build block-to-block similarity graph.

    Args:
        desc_matrix: L2-normalized descriptor matrix from build_semantic_index()
        slot_ids:    pool slot IDs (same order as desc_matrix rows)
        K_semantic:  number of semantic nearest neighbors per block
        K_temporal:  number of temporal (adjacent-block) neighbors per block
        inv_index:   Optional MergedTokenDictionary (InvertedTokenIndex) containing vocabularies
        overlap_threshold: Percentage threshold (0.0 to 1.0) of shared keywords
        blocks:      Optional list of KVBlock/StreamingKVBlock objects

    Returns:
        ChunkGraph with neighbors[i] = row indices of i's neighbors, padded with -1
    """
    N = desc_matrix.shape[0]

    # Initialize hierarchical fields
    parent_landmarks_tensor = torch.zeros((0,), dtype=torch.int32)
    parent_to_children = {}
    slot_to_parent = {}

    if blocks is not None and len(blocks) > 0:
        # Group blocks by chunk (prefill chunk size defaults to 512 tokens)
        chunk_size = 512
        chunk_groups = {}
        for b in blocks:
            if getattr(b, "pool_idx", None) is not None:
                c_idx = b.anchor_idx // chunk_size
                chunk_groups.setdefault(c_idx, []).append(b.pool_idx)
                
        parent_landmarks_list = []
        for c_idx in sorted(chunk_groups.keys()):
            group_slots = chunk_groups[c_idx]
            if group_slots:
                parent = group_slots[0]
                parent_landmarks_list.append(parent)
                parent_to_children[parent] = group_slots[1:]
                for s in group_slots:
                    slot_to_parent[s] = parent
                    
        parent_landmarks_tensor = torch.tensor(parent_landmarks_list, dtype=torch.int32)

    if N == 0:
        return ChunkGraph(
            neighbors=torch.zeros((0, 8), dtype=torch.int32),
            weights=torch.zeros((0, 8), dtype=torch.float32),
            parent_landmarks=parent_landmarks_tensor,
            parent_to_children=parent_to_children,
            slot_to_parent=slot_to_parent
        )

    if N == 1:
        return ChunkGraph(
            neighbors=torch.zeros((1, 8), dtype=torch.int32),
            weights=torch.zeros((1, 8), dtype=torch.float32),
            parent_landmarks=parent_landmarks_tensor,
            parent_to_children=parent_to_children,
            slot_to_parent=slot_to_parent
        )

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
        sem_list = sem_idx.tolist()
    else:
        sem_list = [[] for _ in range(N)]

    # ── Temporal neighbors (prev/next in chronological order) ─────────────
    t_neighbors = [[] for _ in range(N)]
    for i in range(N):
        if K_temporal >= 1:
            t_neighbors[i].append(max(0, i - 1))
        if K_temporal >= 2:
            t_neighbors[i].append(min(N - 1, i + 1))

    # ── Lexical / Keyword Overlap Neighbors ───────────────────────────────
    lex_neighbors = [[] for _ in range(N)]
    vocabs = []
    if inv_index is not None and getattr(inv_index, "chunk_vocabularies", None):
        import os as _local_os
        slot_list = slot_ids.tolist()
        for slot in slot_list:
            if slot in inv_index.chunk_vocabularies:
                vocabs.append(set(inv_index.chunk_vocabularies[slot].keys()))
            else:
                vocabs.append(set())

        for i in range(N):
            w_i = vocabs[i]
            len_w_i = len(w_i)
            if len_w_i == 0:
                continue
            candidates_i = []
            for j in range(N):
                if i == j:
                    continue
                w_j = vocabs[j]
                overlap = len(w_i & w_j)
                relative_score = overlap / len_w_i
                if relative_score >= overlap_threshold:
                    candidates_i.append((j, relative_score))
            
            # Connect all blocks exceeding the threshold (unlimited degree web)
            lex_neighbors[i] = [c[0] for c in candidates_i]

    # ── Merge and Deduplicate Per Block (Variable Degree Web) ─────────────
    merged_neighbors = []
    merged_weights = []
    max_degree = 0
    for i in range(N):
        # Merge semantic, temporal, and lexical connections
        # Filter out self-loops
        candidates = sem_list[i] + t_neighbors[i] + lexical_slots if 'lexical_slots' in locals() else sem_list[i] + t_neighbors[i] + lex_neighbors[i]
        # Wait, the original code had:
        # candidates = sem_list[i] + t_neighbors[i] + lex_neighbors[i]
        # let's keep it as:
        candidates = sem_list[i] + t_neighbors[i] + lex_neighbors[i]
        unique_neighbors = list(dict.fromkeys(c for c in candidates if c != i))
        merged_neighbors.append(unique_neighbors)
        max_degree = max(max_degree, len(unique_neighbors))

        # Calculate directed weights for neighbors of block i
        weights_i = []
        for j in unique_neighbors:
            if j >= N or j < 0:
                print(f"[DiffKV DEBUG] Out of bounds neighbor: N={N} i={i} j={j} sem_list[i]={sem_list[i]} t_neighbors[i]={t_neighbors[i]} lex_neighbors[i]={lex_neighbors[i]}", flush=True)
                j = min(N - 1, max(0, j))
            sim_score = max(0.0, float(sim[i, j]))
            
            # Lexical score
            lex_score = 0.0
            if vocabs:
                w_i = vocabs[i]
                w_j = vocabs[j]
                if len(w_i) > 0:
                    lex_score = len(w_i & w_j) / len(w_i)
            
            # Temporal boost
            temporal_boost = 0.2 if abs(i - j) == 1 else 0.0
            
            weight = 0.5 * sim_score + 0.5 * lex_score + temporal_boost
            weights_i.append(max(1e-5, weight))
        merged_weights.append(weights_i)

    # Pad all neighbor lists to max_degree with -1 (out-of-bounds sentinel)
    # Ensure a minimum degree of at least 1 to prevent empty tensor shape issues
    max_degree = max(1, max_degree)
    pad_value = -1
    
    neighbors_tensor = torch.full((N, max_degree), pad_value, dtype=torch.int32)
    weights_tensor = torch.zeros((N, max_degree), dtype=torch.float32)
    for i, neighbors_i in enumerate(merged_neighbors):
        if neighbors_i:
            neighbors_tensor[i, :len(neighbors_i)] = torch.tensor(neighbors_i, dtype=torch.int32)
            weights_tensor[i, :len(neighbors_i)] = torch.tensor(merged_weights[i], dtype=torch.float32)

    return ChunkGraph(
        neighbors=neighbors_tensor.cpu(),
        weights=weights_tensor.cpu(),
        parent_landmarks=parent_landmarks_tensor,
        parent_to_children=parent_to_children,
        slot_to_parent=slot_to_parent
    )


