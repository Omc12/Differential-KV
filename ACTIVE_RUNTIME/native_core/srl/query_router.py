"""
native_core/srl/query_router.py

QueryRouter: selects the ~50 most relevant pool slots per decode step.

Entry point: route_query() — called once per decode step at layer 0.
The result (selected_slots tensor) is cached on SessionSRLState and
reused by layers 1–27 of the same decode step.

Algorithm (budget K ≈ 50):
  Step 1:  Adaptive K selection (query complexity → K_min..K_max)
  Step 2:  Compute query descriptor [64-dim]
  Step 3:  Semantic ANN search → top 50% of K
  Step 4:  Lexical inverted index → top 15% of K
  Step 5:  Chunk graph expansion → top 15% of K
  Step 6:  Recency window → top 20% of K
  Step 7:  Always include sink blocks (block 0, special tokens)
  Step 8:  Merge, deduplicate, cap at K
  Step 9:  Level-1 anchor screening (rerank by cheap dot product)

Total cost: ~0.5ms for 781 blocks on a modern GPU.
"""

from __future__ import annotations
import sys
import os
# Add the build directory containing diffkv_core.so to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.abspath(os.path.join(_script_dir, "../diffkv_core"))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch

from native_core.srl.chunk_descriptor import compute_query_descriptor
from native_core.srl.inverted_index import lookup as inverted_lookup, lookup_occurrences

if TYPE_CHECKING:
    from native_core.srl.session_srl_state import SessionSRLState


# ── Config (overridable via env vars) ─────────────────────────────────────────
_DEFAULT_K_MIN     = int(os.environ.get("DIFFKV_SRL_K_MIN",     "20"))
_DEFAULT_K_MAX     = int(os.environ.get("DIFFKV_SRL_K_MAX",     "200"))
_SEM_FRAC          = float(os.environ.get("DIFFKV_SRL_SEM_FRAC",  "0.50"))
_LEX_FRAC          = float(os.environ.get("DIFFKV_SRL_LEX_FRAC",  "0.15"))
_GRAPH_FRAC        = float(os.environ.get("DIFFKV_SRL_GRAPH_FRAC", "0.15"))
_RECENCY_FRAC      = float(os.environ.get("DIFFKV_SRL_REC_FRAC",  "0.20"))
_ROUTING_THRESHOLD = int(os.environ.get("DIFFKV_SRL_THRESHOLD",  "50"))
# High-Quality Mode (cross-runtime toggle; mirrors native src/main.cpp +
# mlx_diffkv_wrapper.py). When OFF (default = fast bounded-K), the router keeps
# only the semantic + lexical + recency + sink channels — the same channels that
# preserved NIAH 6/6 + multi-fact recall in native & MLX fast mode. When ON, it
# also runs the dynamic graph routing: 2-hop chunk-graph expansion + dynamic /
# prompt anchor neighborhood expansion (best synthesis fidelity). CUDA decode is
# already bounded-K (adaptive_k 20..200), so this toggle controls the graph
# channels, not an attend-all switch. NOTE: this path is GPU-only and could not be
# validated on the Mac dev machine — see CUDA_TRITON_AUDIT.md GPU cert checklist.
_HIGH_QUALITY_ROUTING = os.environ.get(
    "DIFFKV_HIGH_QUALITY_ROUTING", "0").strip().lower() not in ("0", "", "false", "off", "auto")
# Topic-switch: if the best semantic match falls below this cosine similarity,
# the query is treated as a new topic and stale rare-lexical seeds are suppressed.
_TOPIC_SWITCH_THRESHOLD = float(os.environ.get("DIFFKV_SRL_TOPIC_SWITCH_THRESHOLD", "0.30"))


# ── Level-1 Anchor Screening ──────────────────────────────────────────────────

def two_level_gate(
    Q:         torch.Tensor,         # [H, D] current query (all heads)
    pool,                            # NativeBlockPool
    slot_ids:  torch.Tensor,         # [M] candidate pool slot IDs (int32/int64)
    scale:     float,
    k_pass:    int,                  # how many to keep
    srl_state: Optional["SessionSRLState"] = None,
) -> torch.Tensor:                   # [k_pass] pool slot IDs
    """
    Cheap anchor-only dot-product screening with recency-decay scoring.

    Loads only pool.anchors_K (small, often in L2 cache) and computes
    a mean-query dot product to rerank the candidate set.

    Cost: M × D multiplications. For M=100, D=128: ~13K ops — sub-ms.
    """
    N = slot_ids.shape[0]
    if N <= k_pass:
        return slot_ids

    if srl_state is not None:
        age_penalty_factor = getattr(srl_state, "srl_age_penalty", 0.01)
    else:
        age_penalty_factor = float(os.environ.get("DIFFKV_SRL_AGE_PENALTY", "0.0"))
    try:
        import diffkv_core as _dkv_core
        if getattr(_dkv_core, "HAS_SRL_ROUTER", False) and (srl_state is None or age_penalty_factor == 0.0):
            return _dkv_core.anchor_screen(Q, pool.anchors_K, slot_ids, scale, k_pass)
    except Exception:
        pass

    # GQA Gating: match query heads to corresponding KV heads and take the max dot product
    H = Q.shape[0]
    slot_long = slot_ids.long()
    anc_K = pool.anchors_K[slot_long].float()           # [M, kv_heads, D]
    kv_heads = anc_K.shape[1]
    group_size = max(1, H // kv_heads)

    # Expand anc_K to [M, H, D] by repeating along the kv_heads dimension
    anc_K_expanded = anc_K.repeat_interleave(group_size, dim=1)
    if anc_K_expanded.shape[1] < H:
        pad_size = H - anc_K_expanded.shape[1]
        last_head = anc_K_expanded[:, -1:, :]
        anc_K_expanded = torch.cat([anc_K_expanded, last_head.repeat(1, pad_size, 1)], dim=1)
    elif anc_K_expanded.shape[1] > H:
        anc_K_expanded = anc_K_expanded[:, :H, :]

    # Compute dot product per query head: [M, H]
    scores_per_head = (anc_K_expanded * Q.unsqueeze(0).float()).sum(dim=-1) * scale  # [M, H]
    anchor_scores = scores_per_head.max(dim=1).values                               # [M]

    # Apply chronological age penalty to prevent old concepts from staying over-active
    if srl_state is not None and srl_state.ordered_slot_ids:
        ordered_slots = srl_state.ordered_slot_ids
        n_total = len(ordered_slots)
        
        # Build dictionary for O(1) age lookup
        slot_to_idx = {slot_id: idx for idx, slot_id in enumerate(ordered_slots)}
        
        if age_penalty_factor > 0.0:
            penalties = []
            for slot_id in slot_ids.tolist():
                idx = slot_to_idx.get(slot_id, n_total - 1)
                age = n_total - 1 - idx
                penalties.append(age * age_penalty_factor)
                
            penalties_t = torch.tensor(penalties, dtype=anchor_scores.dtype, device=anchor_scores.device)
            anchor_scores = anchor_scores - penalties_t

        # Apply slot reinforcement strength boost
        if srl_state is not None and getattr(srl_state, "slot_activation_strength", None):
            boosts = []
            for slot_id in slot_ids.tolist():
                strength = srl_state.slot_activation_strength.get(slot_id, 1.0)
                boosts.append((strength - 1.0) * 0.1)
            boosts_t = torch.tensor(boosts, dtype=anchor_scores.dtype, device=anchor_scores.device)
            anchor_scores = anchor_scores + boosts_t

    # ── Edge-aware routing propagation (Python fallback) ──────────────────────
    er_on = os.environ.get("DIFFKV_EDGE_ROUTING", "1").strip().lower() not in ("0", "", "false", "off", "auto")
    if er_on and N >= 3:
        try:
            er_beta = float(os.environ.get("DIFFKV_EDGE_ROUTE_BETA", "0.25"))
        except ValueError:
            er_beta = 0.25
        try:
            er_maxnb = int(os.environ.get("DIFFKV_EDGE_ROUTE_MAXNB", "512"))
        except ValueError:
            er_maxnb = 512

        if N <= er_maxnb:
            # anc_K shape is [N, kv_heads, D]
            # Flatten to [N, kv_heads * D]
            akf = anc_K.reshape(N, -1).float()
            # Normalize row-wise to get unit vector signatures
            akn = akf / (torch.norm(akf, dim=-1, keepdim=True) + 1e-6)
            # Cosine similarity matrix A: [N, N]
            A = torch.matmul(akn, akn.t())
            # Self-loops removed, keep positive edges only
            A = torch.clamp(A - torch.eye(N, device=A.device), min=0.0)
            # Row normalize A
            A = A / (torch.sum(A, dim=-1, keepdim=True) + 1e-6)
            # Propagate relevance
            prop = torch.mv(A, anchor_scores.float())
            anchor_scores = anchor_scores + er_beta * prop

    k_keep = min(k_pass, N)
    top_idx = torch.topk(anchor_scores, k=k_keep, largest=True, sorted=True).indices
    return slot_ids[top_idx]


# ── Adaptive K Selection ───────────────────────────────────────────────────────

def adaptive_k(
    q_desc:     torch.Tensor,         # [DESC_DIM] float32, normalized
    srl_state:  "SessionSRLState",
    N_total:    int,                  # total blocks in session
) -> int:
    """
    Estimate query complexity from entropy of semantic score distribution,
    then map to a K value in [k_min, k_max].

    Simple query ("hi") → entropy near 0 → K ≈ k_min (20)
    Complex query ("explain the algorithm") → higher entropy → K up to k_max
    """
    k_max = min(srl_state.k_max, N_total)
    k_min = min(max(srl_state.k_min, int(0.15 * N_total)), k_max)

    if N_total <= k_min:
        return N_total

    # Compute softmax-entropy over semantic scores.
    # OPT: Keep everything on GPU until the very last scalar conversion.
    # Previously: entropy.item() + max_ent float + C_active.item() = 3 CUDA syncs.
    # Now: a single .item() at the end to read final K.
    if q_desc.device.type == "mps":
        scores = srl_state.semantic_index.desc_matrix.float() @ q_desc.float()
    else:
        scores = srl_state.semantic_index.desc_matrix.float() @ q_desc   # [N]
    probs  = torch.softmax(scores * 5.0, dim=0)                       # temperature-scaled
    # Clamp log to avoid -inf; compute entropy entirely on GPU
    log_probs = probs.clamp(min=1e-10).log()
    entropy_t = -(probs * log_probs).sum()                            # GPU scalar tensor, no sync
    max_ent   = math.log(max(N_total, 2))                             # pure Python, no GPU

    # Normalize to [0, 1] on GPU; clamp for safety
    complexity_t = (entropy_t / max_ent).clamp(min=0.0, max=1.0)     # GPU tensor

    # Scale K linearly with complexity on GPU; apply floor to mimic Python's int() cast
    k_range = k_max - k_min
    k_raw_t = torch.floor(k_min + k_range * complexity_t)             # GPU tensor

    # Apply adaptive multiplier (boosted when miss rate is high)
    k_scaled_t = torch.floor(k_raw_t * srl_state.k_multiplier)        # GPU scalar

    # Calculate C_active (number of active clusters) — GPU-resident throughout
    C_active = 1
    chunk_graph = srl_state.chunk_graph
    if chunk_graph is not None and getattr(chunk_graph, "parent_landmarks", None) is not None and chunk_graph.parent_landmarks.numel() > 0:
        parent_slots = chunk_graph.parent_landmarks
        slot_ids = srl_state.semantic_index.slot_ids
        desc_matrix = srl_state.semantic_index.desc_matrix
        
        slot_to_idx = {int(slot): idx for idx, slot in enumerate(slot_ids.tolist())}
        parent_idxs = [slot_to_idx[p] for p in parent_slots.tolist() if p in slot_to_idx]
        if parent_idxs:
            parent_descs = desc_matrix[parent_idxs].to(q_desc.device, dtype=q_desc.dtype)
            parent_scores = parent_descs @ q_desc
            S_max_t = parent_scores.max()                              # GPU tensor, no sync
            theta_active_t = torch.clamp(S_max_t * 0.85, min=0.30)
            C_active = int((parent_scores >= theta_active_t).sum().item())  # single .item() here
            C_active = max(1, C_active)

    k_final_t = torch.floor(k_scaled_t * (1.0 + 0.35 * math.log(max(C_active, 1))))

    # Single .item() to materialize the final K integer (unavoidable for loop bound)
    k_final = int(k_final_t.item())
    return max(k_min, min(k_max, k_final))


# ── Full Query Router ──────────────────────────────────────────────────────────

def route_query(
    Q:          torch.Tensor,       # [H, D] current query (all query heads)
    srl_state:  "SessionSRLState",
    pool,                           # NativeBlockPool
    scale:      float,              # attention scale = 1/sqrt(head_dim)
    layer_idx:  int,
    query_tokens: Optional[List[int]] = None,
) -> torch.Tensor:                  # [K_selected] pool slot IDs (int32), on Q.device
    """
    Select K most relevant pool slots for the current step.

    Returns pool slot IDs as a 1-D int32 tensor on Q.device.
    """
    N = srl_state.n_active_blocks()

    if N == 0:
        return torch.tensor([], dtype=torch.int32, device=Q.device)

    # ── Step 1: Adaptive K ────────────────────────────────────────────────
    try:
        import diffkv_core as _dkv_core
        if getattr(_dkv_core, "HAS_SRL_ROUTER", False):
            q_desc = _dkv_core.compute_query_desc(Q, pool.W_proj)
        else:
            q_desc = compute_query_descriptor(Q, pool.W_proj)
    except ImportError:
        q_desc = compute_query_descriptor(Q, pool.W_proj)
    K      = adaptive_k(q_desc, srl_state, N)

    # ── Step 1.5: Lexical inverted index lookup (IDF & Term Coverage Boost) ──
    # FIX (Bug 1): Prioritize current_query_tokens for lexical lookup.
    # Only use recent_generated_tokens (previous-turn output) as a weak fallback
    # to avoid contaminating the inverted-index search with stale vocabulary.
    k_lexical   = max(1, int(K * _LEX_FRAC))
    if query_tokens is not None:
        recent_toks = query_tokens
    else:
        current_q_toks = getattr(srl_state, "current_query_tokens", [])
        if current_q_toks:
            # Use only the incoming query tokens — exclude previous-turn outputs.
            recent_toks = current_q_toks[-128:]
        else:
            # Fallback: no explicit query tokens available — use a small tail of
            # recent generated tokens so we don't completely lose lexical signal.
            recent_toks = srl_state.recent_generated_tokens[-16:]

    from collections import defaultdict
    inv_index = srl_state.inverted_index
    all_abs_positions = []
    
    # Track matching positions and IDF values
    for tok in recent_toks:
        if tok in inv_index.occurrences:
            for slot, abs_pos, rel_pos in inv_index.occurrences[tok]:
                all_abs_positions.append(abs_pos)
                
    rare_lex_slots = []
    if all_abs_positions:
        L = max(all_abs_positions)
        slot_scores = defaultdict(float)
        slot_matched_toks = defaultdict(set)
        decay_factor = float(os.environ.get("DIFFKV_SRL_DECAY_FACTOR", "1.0"))
        
        for tok in recent_toks:
            if tok in inv_index.occurrences:
                # Retrieve precomputed IDF score (rare tokens have higher values)
                idf_val = inv_index.idf.get(tok, 1.0)
                for slot, abs_pos, rel_pos in inv_index.occurrences[tok]:
                    slot_scores[slot] += idf_val * (decay_factor ** (L - abs_pos))
                    slot_matched_toks[slot].add(tok)
                    
        # Apply Query Term Coverage boost: scale score by (unique_matches ** 2)
        for slot in list(slot_scores.keys()):
            n_unique = len(slot_matched_toks[slot])
            slot_scores[slot] *= (n_unique ** 2)
            
        sorted_lex_slots = sorted(slot_scores.keys(), key=lambda s: slot_scores[s], reverse=True)
        lexical_slots = sorted_lex_slots[:k_lexical]

        # Extract rare keyword matches (IDF >= 2.0)
        rare_slots_with_scores = defaultdict(float)
        for tok in recent_toks:
            if tok in inv_index.occurrences:
                idf_val = inv_index.idf.get(tok, 1.0)
                if idf_val >= 2.0:
                    for slot, abs_pos, rel_pos in inv_index.occurrences[tok]:
                        rare_slots_with_scores[slot] += idf_val * (decay_factor ** (L - abs_pos))
        if rare_slots_with_scores:
            rare_lex_slots = sorted(rare_slots_with_scores.keys(), key=lambda s: rare_slots_with_scores[s], reverse=True)
    else:
        lexical_slots = []

    # ── Step 2: Semantic ANN search (Hierarchical Graph-based Routing) ────
    k_semantic = max(1, int(K * _SEM_FRAC))
    chunk_graph = srl_state.chunk_graph
    
    concentric_routed = False
    semantic_slots = []
    
    if (getattr(chunk_graph, "cluster_centers_tensor", None) is not None 
            and chunk_graph.cluster_centers_tensor.numel() > 0
            and getattr(chunk_graph, "role_mapping_tensor", None) is not None):
        
        # 1. Score all cluster centers (centroids)
        centers = chunk_graph.cluster_centers_tensor
        center_rows = srl_state.semantic_index.slot_to_row_vec(centers)
        valid_mask = center_rows >= 0
        centers = centers[valid_mask]
        center_rows = center_rows[valid_mask]
        
        if center_rows.numel() > 0:
            center_desc = srl_state.semantic_index.desc_matrix[center_rows.to(srl_state.semantic_index.desc_matrix.device)]
            q16 = q_desc.half().to(center_desc.device)
            if q16.device.type == "mps":
                scores_centers = center_desc.float() @ q16.float()
            else:
                scores_centers = center_desc @ q16
                
            # Select cluster centers with similarity score >= max(0.30, 0.85 * S_max)
            s_max_centroid = float(scores_centers.max().item()) if scores_centers.numel() > 0 else 0.0
            threshold = max(0.30, 0.85 * s_max_centroid)
            active_mask = scores_centers >= threshold
            active_indices = torch.where(active_mask)[0]
            if active_indices.numel() == 0:
                top_center_idx = torch.topk(scores_centers, k=1, largest=True, sorted=True).indices
                selected_centers = centers.to(top_center_idx.device)[top_center_idx].tolist()
            else:
                selected_centers = centers[active_indices.to(centers.device)].tolist()

            # Map matched lexical_slots and rare_lex_slots to parent landmark IDs (prime nodes)
            lexical_parents = []
            if getattr(chunk_graph, "slot_to_parent_tensor", None) is not None:
                for s in (lexical_slots + rare_lex_slots):
                    if s < chunk_graph.slot_to_parent_tensor.shape[0]:
                        p = int(chunk_graph.slot_to_parent_tensor[s].item())
                        if p != -1:
                            lexical_parents.append(p)
            for p in lexical_parents:
                if p not in selected_centers:
                    selected_centers.append(p)

            # Retrieve 1-hop prime neighbors from chunk_graph.prime_neighbors
            all_selected_prime_nodes = set(selected_centers)
            prime_neighbors_tensor = chunk_graph.prime_neighbors
            if prime_neighbors_tensor is not None:
                for c in selected_centers:
                    if c < prime_neighbors_tensor.shape[0]:
                        neighbors = prime_neighbors_tensor[c].tolist()
                        for nb in neighbors:
                            if nb != -1:
                                all_selected_prime_nodes.add(nb)

            # 2. Gather slots associated with these centers/neighbors, grouped by concentric role (Center, Around, Outer) and center association
            role_map = chunk_graph.role_mapping_tensor
            slot_to_center = chunk_graph.slot_to_center_tensor
            
            centers_order = []
            for c in selected_centers:
                if c not in centers_order:
                    centers_order.append(c)
            for c in all_selected_prime_nodes:
                if c not in centers_order:
                    centers_order.append(c)

            around_by_center = {c: [] for c in centers_order}
            outer_by_center = {c: [] for c in centers_order}
            
            slot_ids_cpu = srl_state.semantic_index.slot_ids.cpu().tolist()
            for s in slot_ids_cpu:
                if s in all_selected_prime_nodes:
                    continue
                # check if this slot is associated with one of the selected centers or neighbors
                if s < len(slot_to_center):
                    assoc_c = int(slot_to_center[s].item())
                    if assoc_c in all_selected_prime_nodes:
                        role = int(role_map[s].item())
                        if role == 1:
                            around_by_center.setdefault(assoc_c, []).append(s)
                        elif role == 0:
                            outer_by_center.setdefault(assoc_c, []).append(s)
                            
            # Interleave around/outer slots per centroid
            around_lists = [around_by_center.get(c, []) for c in centers_order]
            around_slots_interleaved = []
            max_len_around = max((len(l) for l in around_lists), default=0)
            for step in range(max_len_around):
                for l in around_lists:
                    if step < len(l):
                        around_slots_interleaved.append(l[step])

            outer_lists = [outer_by_center.get(c, []) for c in centers_order]
            outer_slots_interleaved = []
            max_len_outer = max((len(l) for l in outer_lists), default=0)
            for step in range(max_len_outer):
                for l in outer_lists:
                    if step < len(l):
                        outer_slots_interleaved.append(l[step])

            # 3. Build prioritized list: Center -> Around -> Outer (interleaved)
            prioritized_slots = []
            seen = set()
            
            # High relevance: Center (both the scored ones and their neighbors)
            for c in centers_order:
                if c not in seen:
                    prioritized_slots.append(c)
                    seen.add(c)
                    
            # Mid relevance: Around
            for s in around_slots_interleaved:
                if s not in seen:
                    prioritized_slots.append(s)
                    seen.add(s)
                    
            # Low relevance: Outer
            for s in outer_slots_interleaved:
                if s not in seen:
                    prioritized_slots.append(s)
                    seen.add(s)
                    
            semantic_slots = prioritized_slots[:k_semantic]
            semantic_slots_t = torch.tensor(semantic_slots, dtype=torch.int32, device=Q.device)
            concentric_routed = True

    if not concentric_routed:
        if (getattr(chunk_graph, "parent_landmarks", None) is not None 
                and chunk_graph.parent_landmarks.numel() > 0):
            # 1. Score parent landmark blocks
            parent_slots = chunk_graph.parent_landmarks
            parent_rows = srl_state.semantic_index.slot_to_row_vec(parent_slots)
            valid_mask = parent_rows >= 0
            parent_slots = parent_slots[valid_mask]
            parent_rows = parent_rows[valid_mask]
            
            if parent_rows.numel() > 0:
                parent_desc = srl_state.semantic_index.desc_matrix[parent_rows.to(srl_state.semantic_index.desc_matrix.device)]
                q16 = q_desc.half().to(parent_desc.device)
                if q16.device.type == "mps":
                    scores_parent = parent_desc.float() @ q16.float()
                else:
                    scores_parent = parent_desc @ q16
                    
                # Select top landmark parents
                k_parent = max(1, min(k_semantic // 8, parent_slots.numel()))
                top_parent_idx = torch.topk(scores_parent, k=k_parent, largest=True, sorted=True).indices
                selected_parents = parent_slots.to(top_parent_idx.device)[top_parent_idx].tolist()
                
                # Map matched lexical_slots and rare_lex_slots to parent landmark IDs (prime nodes)
                lexical_parents = []
                if getattr(chunk_graph, "slot_to_parent_tensor", None) is not None:
                    for s in (lexical_slots + rare_lex_slots):
                        if s < chunk_graph.slot_to_parent_tensor.shape[0]:
                            p = int(chunk_graph.slot_to_parent_tensor[s].item())
                            if p != -1:
                                lexical_parents.append(p)
                for p in lexical_parents:
                    if p not in selected_parents:
                        selected_parents.append(p)
                
                # 2. Gather children blocks for the selected parent landmarks (interleaved)
                children_lists = []
                if getattr(chunk_graph, "parent_to_children_tensor", None) is not None:
                    for p in selected_parents:
                        if p < chunk_graph.parent_to_children_tensor.shape[0]:
                            row = chunk_graph.parent_to_children_tensor[p]
                            children = row[row != -1].tolist()
                            children_lists.append(children)
                
                children_interleaved = []
                max_children_len = max((len(l) for l in children_lists), default=0)
                for step in range(max_children_len):
                    for l in children_lists:
                        if step < len(l):
                            children_interleaved.append(l[step])
                    
                hierarchical_slots = []
                seen = set()
                for parent in selected_parents:
                    if parent not in seen:
                        hierarchical_slots.append(parent)
                        seen.add(parent)
                for child in children_interleaved:
                    if child not in seen:
                        hierarchical_slots.append(child)
                        seen.add(child)
                        
                semantic_slots_t = torch.tensor(hierarchical_slots, dtype=torch.int32, device=Q.device)
            else:
                semantic_slots_t = srl_state.semantic_index.search(q_desc, k=k_semantic)
        else:
            semantic_slots_t = srl_state.semantic_index.search(q_desc, k=k_semantic)
            
        semantic_slots = semantic_slots_t.tolist()

    # ── Topic-switch detection ────────────────────────────────────────────
    # If the best semantic match is weak, this query is likely a new topic.
    # Suppress stale rare-lexical graph seeds to avoid over-anchoring.
    desc_matrix = srl_state.semantic_index.desc_matrix
    q16 = q_desc.half().to(desc_matrix.device)
    if q16.device.type == "mps":
        sem_scores = desc_matrix.float() @ q16.float()
    else:
        sem_scores = desc_matrix @ q16
    sem_scores_cpu = torch.clamp(sem_scores.float().cpu(), min=0.0)

    _is_topic_switch = False
    if semantic_slots_t.numel() > 0:
        top_slot_row = srl_state.semantic_index.slot_to_idx(int(semantic_slots_t[0]))
        if top_slot_row >= 0:
            top_score = float(sem_scores_cpu[top_slot_row])
            _is_topic_switch = top_score < _TOPIC_SWITCH_THRESHOLD



    # ── Step 4: Chunk graph neighborhood expansion (vectorized) ──────────
    k_graph   = max(1, int(K * _GRAPH_FRAC))
    idx_map   = srl_state.semantic_index
    neighbors = srl_state.chunk_graph.neighbors   # [N, MAX_DEGREE] CPU int32
    weights   = getattr(srl_state.chunk_graph, "weights", None)
    
    # Fallback to ones if weights are not yet populated (e.g. legacy test runs)
    if weights is None:
        weights = torch.ones_like(neighbors, dtype=torch.float32)

    # Seed expansion with top semantic + (rare/lex if not a topic switch).
    # Dynamically scale seeds based on actual fractions to match model capacity.
    n_sem_seeds = max(1, k_semantic)
    n_rare_seeds = max(2, int(2 * k_lexical))
    n_lex_seeds = max(1, k_lexical)

    top_sem_slots = semantic_slots_t[:n_sem_seeds].tolist() if semantic_slots_t.numel() > 0 else []
    if _is_topic_switch:
        # Topic switch: anchor graph expansion only on fresh semantic matches.
        seeds = list(dict.fromkeys(top_sem_slots))
    else:
        seeds = list(dict.fromkeys(top_sem_slots + rare_lex_slots[:n_rare_seeds] + lexical_slots[:n_lex_seeds]))

    graph_slots: List[int] = []
    if seeds and neighbors.shape[0] > 0:
        seeds_tensor = torch.tensor(seeds, dtype=torch.int32)
        row_ids = idx_map.slot_to_row_vec(seeds_tensor)                     # [M]
        valid_rows = row_ids[row_ids >= 0]                                  # filter misses

        if valid_rows.numel() > 0:
            N_idx = idx_map.slot_ids.shape[0]
            
            # Initial activations at seeds: A0
            A0 = torch.zeros(N_idx, dtype=torch.float32)
            A0[valid_rows] = sem_scores_cpu[valid_rows]
            
            # Split seeds by segment ID
            row_segs = []
            for r in range(N_idx):
                slot_id = int(idx_map.slot_ids[r].item())
                row_segs.append(srl_state.segment_ids.get(slot_id, 0))
            row_segs_tensor = torch.tensor(row_segs, dtype=torch.int32)

            valid_rows_list = valid_rows.tolist()
            rows_seg1 = []
            rows_seg2 = []
            rows_seg0 = []
            for r in valid_rows_list:
                slot_id = int(idx_map.slot_ids[r].item())
                seg = srl_state.segment_ids.get(slot_id, 0)
                if seg == 1:
                    rows_seg1.append(r)
                elif seg == 2:
                    rows_seg2.append(r)
                else:
                    rows_seg0.append(r)

            current_query_segment_id = getattr(srl_state, "current_query_segment_id", 0)
            forbidden_for_0 = None
            if current_query_segment_id == 1:
                rows_seg2 = []
                forbidden_for_0 = 2
            elif current_query_segment_id == 2:
                rows_seg1 = []
                forbidden_for_0 = 1

            valid_rows_1 = torch.tensor(rows_seg1, dtype=torch.long)
            valid_rows_2 = torch.tensor(rows_seg2, dtype=torch.long)
            valid_rows_0 = torch.tensor(rows_seg0, dtype=torch.long)

            # Retention scaling (query similarity target-damping)
            # Retention(q, j) = graph_hop_decay * sem_scores_cpu[j]
            graph_hop_decay = getattr(srl_state, "graph_hop_decay", 0.5)
            retention = graph_hop_decay * sem_scores_cpu

            # Precompute slot reinforcement strengths
            strengths_cpu = torch.ones(N_idx, dtype=torch.float32)
            if srl_state is not None and getattr(srl_state, "slot_activation_strength", None):
                for r in range(N_idx):
                    slot_id = int(idx_map.slot_ids[r].item())
                    strengths_cpu[r] = srl_state.slot_activation_strength.get(slot_id, 1.0)
            
            # Calculate node degrees for transition dilution (degree damping)
            # degree(i) is number of valid (>= 0) neighbors for node i
            degrees = (neighbors >= 0).sum(dim=1, keepdim=True).float()
            damping = 1.0 + torch.log(1.0 + degrees)  # [N_idx, 1]
            
            def propagate_for_group(group_rows, forbidden_segment):
                A1_g = torch.zeros(N_idx, dtype=torch.float32)
                A2_g = torch.zeros(N_idx, dtype=torch.float32)
                if group_rows.numel() == 0:
                    return A1_g, A2_g

                A0_g = torch.zeros(N_idx, dtype=torch.float32)
                A0_g[group_rows] = sem_scores_cpu[group_rows]

                # 1-hop propagation
                seed_neighbors = neighbors[group_rows]  # [M_group, MAX_DEGREE]
                seed_weights = weights[group_rows]      # [M_group, MAX_DEGREE]
                
                # Dilute transition weights out of seed nodes
                damped_seed_weights = seed_weights / damping[group_rows]
                
                seed_A0 = A0_g[group_rows].unsqueeze(1)    # [M_group, 1]
                propagated_signals_1 = seed_A0 * damped_seed_weights  # [M_group, MAX_DEGREE]
                
                flat_neighbors_1 = seed_neighbors.view(-1).to(torch.int64)
                flat_signals_1 = propagated_signals_1.view(-1)
                
                valid_mask_1 = (flat_neighbors_1 >= 0) & (flat_neighbors_1 < N_idx)
                A1_g.scatter_add_(0, flat_neighbors_1[valid_mask_1], flat_signals_1[valid_mask_1])
                A1_g = A1_g * retention * strengths_cpu

                if forbidden_segment is not None:
                    forbidden_mask = (row_segs_tensor == forbidden_segment)
                    A1_g[forbidden_mask] = 0.0
                
                # 2-hop propagation
                active_rows = torch.where(A1_g > 0.0)[0]
                if active_rows.numel() > 0:
                    active_neighbors = neighbors[active_rows]
                    active_weights = weights[active_rows]
                    
                    # Dilute transition weights out of active nodes
                    damped_active_weights = active_weights / damping[active_rows]
                    
                    active_A1 = A1_g[active_rows].unsqueeze(1)
                    propagated_signals_2 = active_A1 * damped_active_weights
                    
                    flat_neighbors_2 = active_neighbors.view(-1).to(torch.int64)
                    flat_signals_2 = propagated_signals_2.view(-1)
                    
                    valid_mask_2 = (flat_neighbors_2 >= 0) & (flat_neighbors_2 < N_idx)
                    A2_g.scatter_add_(0, flat_neighbors_2[valid_mask_2], flat_signals_2[valid_mask_2])
                    A2_g = A2_g * retention * strengths_cpu

                    if forbidden_segment is not None:
                        A2_g[forbidden_mask] = 0.0
                
                return A1_g, A2_g

            A1_1, A2_1 = propagate_for_group(valid_rows_1, forbidden_segment=2)
            A1_2, A2_2 = propagate_for_group(valid_rows_2, forbidden_segment=1)
            A1_0, A2_0 = propagate_for_group(valid_rows_0, forbidden_segment=forbidden_for_0)

            A1 = A1_1 + A1_2 + A1_0
            A2 = A2_1 + A2_2 + A2_0
            graph_scores = A1 + A2
            
            # Exclude seed nodes from the new neighbors selection
            graph_scores_for_selection = graph_scores.clone()
            graph_scores_for_selection[valid_rows] = -1e9
            graph_scores_for_selection[graph_scores <= 0.0] = -1e9
            
            # Select top min(k_graph, 20) neighbors based on propagated activation
            k_g = max(k_graph, 20)
            top_val, top_nb_idx = torch.topk(graph_scores_for_selection, k=min(k_g, N_idx), largest=True, sorted=True)
            valid_nb_mask = top_val > -1e9
            valid_nb_idx = top_nb_idx[valid_nb_mask]
            
            if valid_nb_idx.numel() > 0:
                slot_ids_cpu = idx_map.slot_ids.cpu()
                graph_slots_t = slot_ids_cpu[valid_nb_idx]
                graph_slots = graph_slots_t.tolist()
            else:
                graph_slots = []
        else:
            graph_slots = []
    else:
        graph_slots = []

    # ── Step 5: Recency window ────────────────────────────────────────────
    k_recent     = max(1, int(K * _RECENCY_FRAC))
    recent_slots = srl_state.ordered_slot_ids[-k_recent:]

    # ── Step 6: Sink blocks (always include: block 0 + special tokens) ────
    sink = srl_state.sink_blocks

    # ── Step 7: Merge and deduplicate (do NOT slice to K here to enable Level-1 gate reranking) ──
    # FIX (Bug 3): Semantic slots take priority over rare_lex (which may be
    # stale). On topic switch, rare_lex is zeroed out above; on same-topic
    # queries it still contributes but doesn't displace semantic hits.
    # Order: sink → semantic → rare_lex → graph → lexical → recent
    # ── Step 6.5: Dynamic Anchors expansion ──
    dynamic_routed_slots = []
    if getattr(srl_state, "dynamic_anchors", None):
        dynamic_routed_slots = list(srl_state.expand_neighborhood(set(srl_state.dynamic_anchors)))

    # ── Step 6.6: Prompt Anchors expansion ──
    prompt_routed_slots = []
    if getattr(srl_state, "prompt_anchors", None):
        block_size = srl_state.inverted_index.block_size if (srl_state.inverted_index is not None and hasattr(srl_state.inverted_index, "block_size")) else 256
        prompt_anchor_slots = []
        ordered_anchors = getattr(srl_state, "ordered_anchor_idxs", None)
        if ordered_anchors and len(ordered_anchors) == len(srl_state.ordered_slot_ids):
            for idx in srl_state.prompt_anchors:
                matched_block_idx = -1
                for b_idx, start in enumerate(ordered_anchors):
                    if start <= idx < start + block_size:
                        matched_block_idx = b_idx
                        break
                if matched_block_idx != -1:
                    prompt_anchor_slots.append(srl_state.ordered_slot_ids[matched_block_idx])
        else:
            for idx in srl_state.prompt_anchors:
                block_idx = idx // block_size
                if block_idx < len(srl_state.ordered_slot_ids):
                    prompt_anchor_slots.append(srl_state.ordered_slot_ids[block_idx])
        if prompt_anchor_slots:
            prompt_routed_slots = list(srl_state.expand_neighborhood(set(prompt_anchor_slots)))

    # ── High-Quality Mode gate ────────────────────────────────────────────
    # The 2-hop chunk-graph expansion + dynamic/prompt anchor neighborhood
    # expansion are the "dynamic graph routing" — HQ-only, mirroring native's
    # route_decode_slots sections 3–4.5. In fast bounded-K mode (default) drop
    # them; semantic + lexical + recency + sink remain (recall-validated on
    # native & MLX). Cross-runtime toggle: DIFFKV_HIGH_QUALITY_ROUTING=1.
    if not _HIGH_QUALITY_ROUTING:
        graph_slots = []
        dynamic_routed_slots = []
        prompt_routed_slots = []

    combined = list(dict.fromkeys(
        sink + semantic_slots + rare_lex_slots + graph_slots + lexical_slots + recent_slots + dynamic_routed_slots + prompt_routed_slots
    ))

    # Apply Structured Attention Segmenting filtering
    curr_seg = getattr(srl_state, "current_query_segment_id", 0)
    if curr_seg != 0 and getattr(srl_state, "segment_ids", None):
        segment_ids = srl_state.segment_ids
        sink_set = set(sink)
        filtered_combined = []
        for slot in combined:
            if slot in sink_set:
                filtered_combined.append(slot)
                continue
            seg_id = segment_ids.get(slot, 0)
            if seg_id == 0 or seg_id == curr_seg:
                filtered_combined.append(slot)
        combined = filtered_combined

    if curr_seg == 0 and N - len(combined) <= 2:
        combined = srl_state.ordered_slot_ids

    if not combined:
        # Fallback: use all ordered slots (shouldn't happen in practice)
        combined = srl_state.ordered_slot_ids[:K]

    # ── Step 8: Level-1 anchor reranking over candidates (preserving sink blocks) ──
    # We separate sink blocks so they are never filtered out (critical for attention stability).
    # Reranking is applied only to non-sink candidates.
    sink_set = set(sink)
    non_sink_candidates = [s for s in combined if s not in sink_set]
    
    k_non_sink = max(0, K - len(sink))
    non_sink_tensor = torch.tensor(non_sink_candidates, dtype=torch.long, device=Q.device)
    
    if non_sink_tensor.shape[0] > k_non_sink:
        if hasattr(pool, "anchors_K") and k_non_sink > 0:
            filtered_non_sink = two_level_gate(Q, pool, non_sink_tensor, scale, k_pass=k_non_sink, srl_state=srl_state)
        else:
            filtered_non_sink = non_sink_tensor[:k_non_sink]
    else:
        filtered_non_sink = non_sink_tensor
        
    sink_tensor = torch.tensor(sink, dtype=torch.int32, device=Q.device)
    combined_tensor = torch.cat([sink_tensor, filtered_non_sink.to(torch.int32)])

    # Update slot reinforcement/activation strength for the routed slots using an EMA.
    # OPT: Amortize the O(N_slots) decay loop — run only every DECAY_EVERY steps.
    # The decay is multiplicative (0.99^16 ≈ 0.851) so infrequent application is
    # still correct; no slot escapes eventual decay.
    _DECAY_EVERY = 16
    if srl_state is not None and getattr(srl_state, "slot_activation_strength", None) is not None:
        selected_slots_set = set(combined_tensor.tolist())
        alpha_boost = 0.05
        decay_rate = 0.99
        
        # Initialize strength for slots not yet seen and boost selected slots
        for slot in selected_slots_set:
            if slot not in srl_state.slot_activation_strength:
                srl_state.slot_activation_strength[slot] = 1.0
            srl_state.slot_activation_strength[slot] += alpha_boost

        # Lazy decay: run only every DECAY_EVERY steps to cut O(N_slots) Python
        # work from every token to every DECAY_EVERY tokens (~16× reduction).
        _step = getattr(srl_state, "current_step_count", 0)
        if _step % _DECAY_EVERY == 0:
            # Batch decay factor for N applications: rate^N = 0.99^16 ≈ 0.851
            batch_decay = decay_rate ** _DECAY_EVERY
            for slot in list(srl_state.slot_activation_strength.keys()):
                if slot not in selected_slots_set:
                    srl_state.slot_activation_strength[slot] *= batch_decay
                    if srl_state.slot_activation_strength[slot] < 1.0:
                        srl_state.slot_activation_strength[slot] = 1.0

    srl_state.current_step_count += 1
    return combined_tensor.to(torch.int32)


# ── Static Shape Routing ──────────────────────────────────────────────────────
K_FIXED = int(os.environ.get("DIFFKV_SRL_K_FIXED", "64"))

def route_query_fixed_k(
    Q:          torch.Tensor,
    srl_state:  "SessionSRLState",
    pool,
    scale:      float,
    layer_idx:  int,
    query_tokens: Optional[List[int]] = None,
) -> torch.Tensor:
    """Return at most K_FIXED pool slot IDs, ranked best-first, always distinct.

    This used to pad the selection up to exactly K_FIXED by repeating
    ``selected[-1]``, to hand the decode kernel a static block count.  The
    padding is unsound: the caller feeds this list straight into
    ``block_indices`` (diffkv_attention.py:676), and the sparse kernel scores
    every entry independently, so a repeated block enters the softmax once per
    copy.  Padding repeats the *lowest-ranked* selected block, so that block was
    the one whose weight got inflated.

    Concretely, at the 13.4K NAT prompt (49 blocks) with adaptive_k picking
    K=20: the kernel received 64 entries, 44 of them copies of the 20th-ranked
    block — its attention weight inflated ~45x, while the dispatch did MORE work
    (64) than simply attending every block (49).  That is why routing measured
    no faster than not routing.

    MPS/MLX never hit this (it returns above), and MLX's own router selects a
    fixed K=16 *distinct* blocks.  Match that semantic: truncate when we have
    more than K_FIXED, otherwise return what we have.  The count is still stable
    for a given block count, and it can never exceed the number of live blocks.
    """
    selected = route_query(Q, srl_state, pool, scale, layer_idx, query_tokens)

    if Q.device.type == "mps":
        return selected

    if selected.numel() == 0:
        # No routing signal — an empty selection means "no opinion".  Returning
        # K_FIXED copies of slot 0 would force the kernel onto block 0 alone.
        return selected

    if selected.numel() > K_FIXED:
        return selected[:K_FIXED]

    return selected
