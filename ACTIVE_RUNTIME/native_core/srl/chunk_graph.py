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
    parent_to_children_tensor: Optional[torch.Tensor] = None  # [max_slot + 1, max_children] int32, CPU-resident, padded with -1
    slot_to_parent_tensor: Optional[torch.Tensor] = None      # [max_slot + 1] int32, CPU-resident, padded with -1

    # Concentric zoning fields
    role_mapping_tensor: Optional[torch.Tensor] = None         # [max_slot + 1] int32 CPU-resident role (0=outer, 1=around, 2=center)
    cluster_centers_tensor: Optional[torch.Tensor] = None      # [C] int32 CPU-resident pool slot IDs of cluster centers
    slot_to_center_tensor: Optional[torch.Tensor] = None       # [max_slot + 1] int32 CPU-resident pool slot IDs of center

    # Backwards compatibility properties
    @property
    def parent_to_children(self) -> dict:
        if self.parent_to_children_tensor is None:
            return {}
        d = {}
        for parent in range(self.parent_to_children_tensor.shape[0]):
            row = self.parent_to_children_tensor[parent]
            children = row[row != -1].tolist()
            if children:
                d[parent] = children
        return d

    @property
    def slot_to_parent(self) -> dict:
        if self.slot_to_parent_tensor is None:
            return {}
        d = {}
        for slot in range(self.slot_to_parent_tensor.shape[0]):
            parent = int(self.slot_to_parent_tensor[slot].item())
            if parent != -1:
                d[slot] = parent
        return d

    @property
    def role_mapping(self) -> dict:
        if self.role_mapping_tensor is None:
            return {}
        d = {}
        role_names = {0: "outer", 1: "around", 2: "center"}
        for slot in range(self.role_mapping_tensor.shape[0]):
            role_val = int(self.role_mapping_tensor[slot].item())
            if role_val != -1:
                d[slot] = role_names.get(role_val, "outer")
        return d

    @property
    def cluster_centers(self) -> list:
        if self.cluster_centers_tensor is None:
            return []
        return self.cluster_centers_tensor.tolist()

    @property
    def slot_to_center(self) -> dict:
        if self.slot_to_center_tensor is None:
            return {}
        d = {}
        for slot in range(self.slot_to_center_tensor.shape[0]):
            center = int(self.slot_to_center_tensor[slot].item())
            if center != -1:
                d[slot] = center
        return d


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

    # Find max slot ID dynamically
    max_slot = 0
    if slot_ids.numel() > 0:
        max_slot = max(max_slot, int(slot_ids.max().item()))
    if blocks:
        for b in blocks:
            if getattr(b, "pool_idx", None) is not None:
                max_slot = max(max_slot, int(b.pool_idx))

    # Initialize hierarchical fields
    parent_landmarks_tensor = torch.zeros((0,), dtype=torch.int32)
    parent_to_children_tensor = torch.full((max_slot + 1, 1), -1, dtype=torch.int32)
    slot_to_parent_tensor = torch.full((max_slot + 1,), -1, dtype=torch.int32)

    if blocks is not None and len(blocks) > 0:
        # Group blocks by chunk (prefill chunk size defaults to 512 tokens)
        chunk_size = 512
        chunk_groups = {}
        for b in blocks:
            if getattr(b, "pool_idx", None) is not None:
                c_idx = b.anchor_idx // chunk_size
                chunk_groups.setdefault(c_idx, []).append(b.pool_idx)
                
        # Find maximum children count for sizing
        max_children = 0
        for group_slots in chunk_groups.values():
            if len(group_slots) > 1:
                max_children = max(max_children, len(group_slots) - 1)
        max_children = max(1, max_children)
        
        parent_to_children_tensor = torch.full((max_slot + 1, max_children), -1, dtype=torch.int32)
        
        parent_landmarks_list = []
        for c_idx in sorted(chunk_groups.keys()):
            group_slots = chunk_groups[c_idx]
            if group_slots:
                parent = group_slots[0]
                parent_landmarks_list.append(parent)
                children = group_slots[1:]
                if children:
                    parent_to_children_tensor[parent, :len(children)] = torch.tensor(children, dtype=torch.int32)
                for s in group_slots:
                    slot_to_parent_tensor[s] = parent
                    
        parent_landmarks_tensor = torch.tensor(parent_landmarks_list, dtype=torch.int32)

    # Initialize concentric tensors for edge cases
    role_mapping_tensor = torch.full((max_slot + 1,), -1, dtype=torch.int32)
    cluster_centers_tensor = torch.zeros((0,), dtype=torch.int32)
    slot_to_center_tensor = torch.full((max_slot + 1,), -1, dtype=torch.int32)

    if N == 0:
        return ChunkGraph(
            neighbors=torch.zeros((0, 8), dtype=torch.int32),
            weights=torch.zeros((0, 8), dtype=torch.float32),
            parent_landmarks=parent_landmarks_tensor,
            parent_to_children_tensor=parent_to_children_tensor,
            slot_to_parent_tensor=slot_to_parent_tensor,
            role_mapping_tensor=role_mapping_tensor,
            cluster_centers_tensor=cluster_centers_tensor,
            slot_to_center_tensor=slot_to_center_tensor
        )

    if N == 1:
        slot_s = int(slot_ids[0].item())
        if slot_s < len(role_mapping_tensor):
            role_mapping_tensor[slot_s] = 2  # Center
            slot_to_center_tensor[slot_s] = slot_s
        cluster_centers_tensor = torch.tensor([slot_s], dtype=torch.int32)
        return ChunkGraph(
            neighbors=torch.zeros((1, 8), dtype=torch.int32),
            weights=torch.zeros((1, 8), dtype=torch.float32),
            parent_landmarks=parent_landmarks_tensor,
            parent_to_children_tensor=parent_to_children_tensor,
            slot_to_parent_tensor=slot_to_parent_tensor,
            role_mapping_tensor=role_mapping_tensor,
            cluster_centers_tensor=cluster_centers_tensor,
            slot_to_center_tensor=slot_to_center_tensor
        )

    # Cast to float32 for pairwise similarity dot products
    desc_f32 = desc_matrix.float()
    sim = desc_f32 @ desc_f32.T
    sim.fill_diagonal_(-1.0)

    # ── Lexical setup ──
    vocabs = []
    if inv_index is not None and getattr(inv_index, "chunk_vocabularies", None):
        slot_list = slot_ids.tolist()
        for slot in slot_list:
            if slot in inv_index.chunk_vocabularies:
                vocabs.append(set(inv_index.chunk_vocabularies[slot].keys()))
            else:
                vocabs.append(set())

    # ── Handshake Hunting Protocol ──────────────────────────────────────────
    # Simulate sequential/chronological creation of chunks.
    # The newly created ones are targeted by earlier ones (hunting), which
    # send a request with their position/relevance, and the targeted chunk
    # sends back its ID to form a bidirectional connection.
    handshake_neighbors = [[] for _ in range(N)]
    handshake_weights = [[] for _ in range(N)]

    for i in range(N):
        slot_i = int(slot_ids[i].item())
        
        # Earlier chunks j < i are hunting for matching chunks
        for j in range(i):
            slot_j = int(slot_ids[j].item())
            
            sim_score = max(0.0, float(sim[j, i]))
            
            # Asymmetric lexical scores
            lex_score_i_to_j = 0.0
            lex_score_j_to_i = 0.0
            if vocabs:
                w_i = vocabs[i]
                w_j = vocabs[j]
                if len(w_i) > 0:
                    lex_score_i_to_j = len(w_i & w_j) / len(w_i)
                if len(w_j) > 0:
                    lex_score_j_to_i = len(w_i & w_j) / len(w_j)
            
            temporal_boost = 0.2 if abs(i - j) == 1 else 0.0
            
            weight_i_to_j = 0.5 * sim_score + 0.5 * lex_score_i_to_j + temporal_boost
            weight_j_to_i = 0.5 * sim_score + 0.5 * lex_score_j_to_i + temporal_boost
            
            # Handshake criteria:
            is_semantic_match = (sim_score >= 0.3)
            # Match is valid if either direction meets lexical threshold
            is_lexical_match = (lex_score_i_to_j >= overlap_threshold) or (lex_score_j_to_i >= overlap_threshold)
            is_temporal_match = (abs(i - j) == 1)
            
            if is_semantic_match or is_lexical_match or is_temporal_match:
                # Handshake complete:
                # j targets i (sends request), i returns ID (connects)
                handshake_neighbors[j].append(i)
                handshake_weights[j].append(max(1e-5, weight_j_to_i))
                
                handshake_neighbors[i].append(j)
                handshake_weights[i].append(max(1e-5, weight_i_to_j))

    # For structural robustness and backwards compatibility, ensure top-k semantic
    # neighbors are also included in the handshake connections.
    k_sem = min(K_semantic, N - 1)
    if k_sem > 0:
        _, sem_idx = torch.topk(sim, k=k_sem, dim=1, largest=True, sorted=True)
        sem_list = sem_idx.tolist()
    else:
        sem_list = [[] for _ in range(N)]

    for i in range(N):
        for j in sem_list[i]:
            if j not in handshake_neighbors[i]:
                sim_score = max(0.0, float(sim[i, j]))
                lex_score_i_to_j = 0.0
                if vocabs:
                    w_i = vocabs[i]
                    w_j = vocabs[j]
                    if len(w_i) > 0:
                        lex_score_i_to_j = len(w_i & w_j) / len(w_i)
                temporal_boost = 0.2 if abs(i - j) == 1 else 0.0
                weight_i_to_j = 0.5 * sim_score + 0.5 * lex_score_i_to_j + temporal_boost
                
                handshake_neighbors[i].append(j)
                handshake_weights[i].append(max(1e-5, weight_i_to_j))
                
                if i not in handshake_neighbors[j]:
                    lex_score_j_to_i = 0.0
                    if vocabs:
                        w_i = vocabs[i]
                        w_j = vocabs[j]
                        if len(w_j) > 0:
                            lex_score_j_to_i = len(w_i & w_j) / len(w_j)
                    weight_j_to_i = 0.5 * sim_score + 0.5 * lex_score_j_to_i + temporal_boost
                    
                    handshake_neighbors[j].append(i)
                    handshake_weights[j].append(max(1e-5, weight_j_to_i))

    # Deduplicate connections per block
    merged_neighbors = []
    merged_weights = []
    max_degree = 0
    for i in range(N):
        unique_nb = []
        unique_wt = []
        seen_nb = set()
        for idx, nb in enumerate(handshake_neighbors[i]):
            if nb != i and nb not in seen_nb:
                unique_nb.append(nb)
                unique_wt.append(handshake_weights[i][idx])
                seen_nb.add(nb)
        merged_neighbors.append(unique_nb)
        merged_weights.append(unique_wt)
        max_degree = max(max_degree, len(unique_nb))

    # Pad all neighbor lists to max_degree with -1 (out-of-bounds sentinel)
    max_degree = max(1, max_degree)
    pad_value = -1
    
    neighbors_tensor = torch.full((N, max_degree), pad_value, dtype=torch.int32)
    weights_tensor = torch.zeros((N, max_degree), dtype=torch.float32)
    for i, neighbors_i in enumerate(merged_neighbors):
        if neighbors_i:
            neighbors_tensor[i, :len(neighbors_i)] = torch.tensor(neighbors_i, dtype=torch.int32)
            weights_tensor[i, :len(neighbors_i)] = torch.tensor(merged_weights[i], dtype=torch.float32)

    # ── Concentric Cluster Relevance Zoning ─────────────────────────────────
    role_mapping_tensor = torch.full((max_slot + 1,), -1, dtype=torch.int32)
    slot_to_center_tensor = torch.full((max_slot + 1,), -1, dtype=torch.int32)
    
    cluster_centers_list = []
    if len(parent_landmarks_tensor) > 0:
        cluster_centers_list = parent_landmarks_tensor.tolist()
    else:
        # Fallback when blocks is not provided: group chunks in windows of 4
        slot_list = slot_ids.tolist()
        for idx in range(0, len(slot_list), 4):
            cluster_centers_list.append(slot_list[idx])
            
    cluster_centers_tensor = torch.tensor(cluster_centers_list, dtype=torch.int32)
    
    # Map center nodes
    for c in cluster_centers_list:
        if c < len(role_mapping_tensor):
            role_mapping_tensor[c] = 2  # Center
            slot_to_center_tensor[c] = c

    # Map remaining nodes
    slot_list = slot_ids.tolist()
    center_set = set(cluster_centers_list)
    for i, s in enumerate(slot_list):
        if s in center_set:
            continue
            
        c = -1
        if s < len(slot_to_parent_tensor) and slot_to_parent_tensor[s] != -1:
            c = int(slot_to_parent_tensor[s].item())
        else:
            best_center = -1
            best_dist = 999999
            for center_candidate in cluster_centers_list:
                dist = abs(s - center_candidate)
                if dist < best_dist:
                    best_dist = dist
                    best_center = center_candidate
            c = best_center
            
        if c != -1:
            slot_to_center_tensor[s] = c
            
            # Check proximity to center
            row_s = i
            row_c = -1
            for idx, slot_candidate in enumerate(slot_list):
                if slot_candidate == c:
                    row_c = idx
                    break
            
            is_around = False
            if row_c != -1:
                similarity_sc = float(torch.dot(desc_matrix[row_s].float(), desc_matrix[row_c].float()))
                is_direct_neighbor = (row_s in merged_neighbors[row_c]) or (row_c in merged_neighbors[row_s])
                
                # Around if similarity >= 0.35 or directly connected
                if similarity_sc >= 0.35 or is_direct_neighbor:
                    is_around = True
                    
            role_mapping_tensor[s] = 1 if is_around else 0  # 1 = Around, 0 = Outer

    return ChunkGraph(
        neighbors=neighbors_tensor.cpu(),
        weights=weights_tensor.cpu(),
        parent_landmarks=parent_landmarks_tensor,
        parent_to_children_tensor=parent_to_children_tensor,
        slot_to_parent_tensor=slot_to_parent_tensor,
        role_mapping_tensor=role_mapping_tensor.cpu(),
        cluster_centers_tensor=cluster_centers_tensor.cpu(),
        slot_to_center_tensor=slot_to_center_tensor.cpu()
    )


