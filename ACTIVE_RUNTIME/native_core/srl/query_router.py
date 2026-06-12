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
# Topic-switch: if the best semantic match falls below this cosine similarity,
# the query is treated as a new topic and stale rare-lexical seeds are suppressed.
_TOPIC_SWITCH_THRESHOLD = float(os.environ.get("DIFFKV_SRL_TOPIC_SWITCH_THRESHOLD", "0.25"))


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
        age_penalty_factor = float(os.environ.get("DIFFKV_SRL_AGE_PENALTY", "0.01"))
    try:
        import diffkv_core as _dkv_core
        if getattr(_dkv_core, "HAS_SRL_ROUTER", False) and (srl_state is None or age_penalty_factor == 0.0):
            return _dkv_core.anchor_screen(Q, pool.anchors_K, slot_ids, scale, k_pass)
    except Exception:
        pass

    # GQA: average query over heads → [D]
    q_mean = Q.float().mean(dim=0)                      # [D]

    # Gather anchor keys for candidate slots: [M, kv_heads, D]
    slot_long = slot_ids.long()
    anc_K = pool.anchors_K[slot_long].float()           # [M, kv_heads, D]
    anc_flat = anc_K.mean(dim=1)                        # [M, D] average over kv heads

    if q_mean.device.type == "mps":
        anchor_scores = (anc_flat.float() @ q_mean.float()) * scale
    else:
        anchor_scores = (anc_flat @ q_mean) * scale         # [M]

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

    # Compute softmax-entropy over semantic scores
    if q_desc.device.type == "mps":
        scores = srl_state.semantic_index.desc_matrix.float() @ q_desc.float()
    else:
        scores = srl_state.semantic_index.desc_matrix.float() @ q_desc   # [N]
    probs  = torch.softmax(scores * 5.0, dim=0)                       # temperature-scaled
    # Clamp log to avoid -inf
    log_probs = probs.clamp(min=1e-10).log()
    entropy   = -(probs * log_probs).sum().item()
    max_ent   = math.log(N_total)

    # Normalize to [0, 1]
    complexity = min(entropy / max(max_ent, 1e-8), 1.0)

    # Scale K linearly with complexity
    k_raw = int(k_min + (k_max - k_min) * complexity)

    # Apply adaptive multiplier (boosted when miss rate is high)
    k_scaled = int(k_raw * srl_state.k_multiplier)

    return max(k_min, min(k_max, k_scaled))


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
                
            # Select top cluster centers based on semantic similarity
            k_centers = max(1, min(k_semantic // 8, centers.numel()))
            top_center_idx = torch.topk(scores_centers, k=k_centers, largest=True, sorted=True).indices
            selected_centers = centers.to(top_center_idx.device)[top_center_idx].tolist()
            
            # 2. Gather slots associated with these centers, grouped by concentric role (Center, Around, Outer)
            role_map = chunk_graph.role_mapping_tensor
            slot_to_center = chunk_graph.slot_to_center_tensor
            
            around_slots = []
            outer_slots = []
            
            selected_centers_set = set(selected_centers)
            slot_ids_cpu = srl_state.semantic_index.slot_ids.cpu().tolist()
            for s in slot_ids_cpu:
                if s in selected_centers_set:
                    continue
                # check if this slot is associated with one of the selected centers
                if s < len(slot_to_center):
                    assoc_c = int(slot_to_center[s].item())
                    if assoc_c in selected_centers_set:
                        role = int(role_map[s].item())
                        if role == 1:
                            around_slots.append(s)
                        elif role == 0:
                            outer_slots.append(s)
                            
            # 3. Build prioritized list: Center -> Around -> Outer
            prioritized_slots = []
            seen = set()
            
            # High relevance: Center
            for c in selected_centers:
                if c not in seen:
                    prioritized_slots.append(c)
                    seen.add(c)
                    
            # Mid relevance: Around
            for s in around_slots:
                if s not in seen:
                    prioritized_slots.append(s)
                    seen.add(s)
                    
            # Low relevance: Outer
            for s in outer_slots:
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
                
                # 2. Gather children blocks for the selected parent landmarks
                selected_parents_t = torch.tensor(selected_parents, dtype=torch.int32, device=chunk_graph.parent_to_children_tensor.device)
                valid_parents_t = selected_parents_t[selected_parents_t < chunk_graph.parent_to_children_tensor.shape[0]]
                
                if valid_parents_t.numel() > 0:
                    children_tensor = chunk_graph.parent_to_children_tensor[valid_parents_t.long()]
                    children_flat = children_tensor.flatten()
                    valid_children = children_flat[children_flat != -1].tolist()
                else:
                    valid_children = []
                    
                hierarchical_slots = []
                seen = set()
                for parent in selected_parents:
                    if parent not in seen:
                        hierarchical_slots.append(parent)
                        seen.add(parent)
                for child in valid_children:
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

    # ── Step 3: Lexical inverted index lookup (IDF & Term Coverage Boost) ──
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
        decay_factor = float(os.environ.get("DIFFKV_SRL_DECAY_FACTOR", "0.999"))
        
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
            
            # Retention scaling (query similarity target-damping)
            # Retention(q, j) = graph_hop_decay * sem_scores_cpu[j]
            graph_hop_decay = getattr(srl_state, "graph_hop_decay", 0.5)
            retention = graph_hop_decay * sem_scores_cpu
            
            # Calculate node degrees for transition dilution (degree damping)
            # degree(i) is number of valid (>= 0) neighbors for node i
            degrees = (neighbors >= 0).sum(dim=1, keepdim=True).float()
            damping = 1.0 + torch.log(1.0 + degrees)  # [N_idx, 1]
            
            # 1-hop propagation: A0 -> A1
            A1 = torch.zeros(N_idx, dtype=torch.float32)
            seed_neighbors = neighbors[valid_rows]  # [M_seeds, MAX_DEGREE]
            seed_weights = weights[valid_rows]      # [M_seeds, MAX_DEGREE]
            
            # Dilute transition weights out of seed nodes
            damped_seed_weights = seed_weights / damping[valid_rows]
            
            seed_A0 = A0[valid_rows].unsqueeze(1)    # [M_seeds, 1]
            propagated_signals_1 = seed_A0 * damped_seed_weights  # [M_seeds, MAX_DEGREE]
            
            flat_neighbors_1 = seed_neighbors.view(-1).to(torch.int64)
            flat_signals_1 = propagated_signals_1.view(-1)
            
            valid_mask_1 = (flat_neighbors_1 >= 0) & (flat_neighbors_1 < N_idx)
            A1.scatter_add_(0, flat_neighbors_1[valid_mask_1], flat_signals_1[valid_mask_1])
            A1 = A1 * retention
            
            # 2-hop propagation: A1 -> A2
            A2 = torch.zeros(N_idx, dtype=torch.float32)
            active_rows = torch.where(A1 > 0.0)[0]
            if active_rows.numel() > 0:
                active_neighbors = neighbors[active_rows]
                active_weights = weights[active_rows]
                
                # Dilute transition weights out of active nodes
                damped_active_weights = active_weights / damping[active_rows]
                
                active_A1 = A1[active_rows].unsqueeze(1)
                propagated_signals_2 = active_A1 * damped_active_weights
                
                flat_neighbors_2 = active_neighbors.view(-1).to(torch.int64)
                flat_signals_2 = propagated_signals_2.view(-1)
                
                valid_mask_2 = (flat_neighbors_2 >= 0) & (flat_neighbors_2 < N_idx)
                A2.scatter_add_(0, flat_neighbors_2[valid_mask_2], flat_signals_2[valid_mask_2])
                A2 = A2 * retention
                
            # Total graph score
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
    combined = list(dict.fromkeys(
        sink + semantic_slots + rare_lex_slots + graph_slots + lexical_slots + recent_slots
    ))

    if N - len(combined) <= 2:
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
    """Always returns exactly K_FIXED pool slot IDs, padded with duplicates if necessary."""
    selected = route_query(Q, srl_state, pool, scale, layer_idx, query_tokens)
    
    if Q.device.type == "mps":
        return selected

    if selected.numel() == 0:
        return torch.zeros(K_FIXED, dtype=torch.int32, device=Q.device)
        
    if selected.numel() >= K_FIXED:
        return selected[:K_FIXED]
        
    pad_count = K_FIXED - selected.numel()
    pad = selected[-1:].expand(pad_count)
    return torch.cat([selected, pad])
