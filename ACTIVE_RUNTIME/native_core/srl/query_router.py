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
# Add the build directory containing dkv_core.so to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.abspath(os.path.join(_script_dir, "../dkv_core"))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch

from native_core.srl.chunk_descriptor import compute_query_descriptor


def _rope_partial(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE honouring partial rotary (rotary_dim may be < head_dim).

    Rotates the leading `rotary_dim` slice and passes the remainder through
    unrotated. Reduces EXACTLY to `x*cos + rotate_half(x)*sin` when
    rotary_dim >= head_dim, so full-rotary models are bit-identical.

    Mirrors triton_fused_decode._partial_rope_apply; kept local so the router has
    no import dependency on the Triton module (which is optional on non-CUDA).
    """
    rot = cos.shape[-1]
    D = x.shape[-1]
    if rot >= D:
        h = D // 2
        return x * cos + torch.cat([-x[..., h:], x[..., :h]], dim=-1) * sin
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    h = rot // 2
    rotated = x_rot * cos + torch.cat([-x_rot[..., h:], x_rot[..., :h]], dim=-1) * sin
    return torch.cat([rotated, x_pass], dim=-1)
from native_core.srl.inverted_index import lookup as inverted_lookup, lookup_occurrences

if TYPE_CHECKING:
    from native_core.srl.session_srl_state import SessionSRLState


# ── Config (overridable via env vars) ─────────────────────────────────────────
_DEFAULT_K_MIN     = int(os.environ.get("DKV_SRL_K_MIN",     "20"))
_DEFAULT_K_MAX     = int(os.environ.get("DKV_SRL_K_MAX",     "200"))
_SEM_FRAC          = float(os.environ.get("DKV_SRL_SEM_FRAC",  "0.50"))
_LEX_FRAC          = float(os.environ.get("DKV_SRL_LEX_FRAC",  "0.15"))
_GRAPH_FRAC        = float(os.environ.get("DKV_SRL_GRAPH_FRAC", "0.15"))
_RECENCY_FRAC      = float(os.environ.get("DKV_SRL_REC_FRAC",  "0.20"))
_ROUTING_THRESHOLD = int(os.environ.get("DKV_SRL_THRESHOLD",  "50"))
# High-Quality Mode (cross-runtime toggle; mirrors native src/main.cpp +
# mlx_dkv_wrapper.py). When OFF (default = fast bounded-K), the router keeps
# only the semantic + lexical + recency + sink channels — the same channels that
# preserved NIAH 6/6 + multi-fact recall in native & MLX fast mode. When ON, it
# also runs the dynamic graph routing: 2-hop chunk-graph expansion + dynamic /
# prompt anchor neighborhood expansion (best synthesis fidelity). CUDA decode is
# already bounded-K (adaptive_k 20..200), so this toggle controls the graph
# channels, not an attend-all switch. NOTE: this path is GPU-only and could not be
# validated on the Mac dev machine — see CUDA_TRITON_AUDIT.md GPU cert checklist.
#
# NAME COLLISION WARNING: MLX's OWN `DKV_HIGH_QUALITY_ROUTING` flag (in
# mlx_dkv_wrapper.py) does something different — it forces attend-ALL (bypasses
# top-K block pruning entirely) as a ground-truth comparison mode. The two
# flags share a name but not semantics; do not assume setting this one on CUDA
# gets you MLX's attend-all behavior. CUDA's actual attend-all equivalent is
# `DKV_TOPK_BLOCKS=0` (see `if topk <= 0` below in route_blocks_relevance,
# which returns every block unfiltered) — that is the flag to reach for when
# you want CUDA's ground-truth / exhaustive-attention comparison mode.
_HIGH_QUALITY_ROUTING = os.environ.get(
    "DKV_HIGH_QUALITY_ROUTING", "0").strip().lower() not in ("0", "", "false", "off", "auto")
# Topic-switch: if the best semantic match falls below this cosine similarity,
# the query is treated as a new topic and stale rare-lexical seeds are suppressed.
_TOPIC_SWITCH_THRESHOLD = float(os.environ.get("DKV_SRL_TOPIC_SWITCH_THRESHOLD", "0.30"))


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
        age_penalty_factor = float(os.environ.get("DKV_SRL_AGE_PENALTY", "0.0"))
    try:
        import dkv_core as _dkv_core
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
    er_on = os.environ.get("DKV_EDGE_ROUTING", "1").strip().lower() not in ("0", "", "false", "off", "auto")
    if er_on and N >= 3:
        try:
            er_beta = float(os.environ.get("DKV_EDGE_ROUTE_BETA", "0.25"))
        except ValueError:
            er_beta = 0.25
        try:
            er_maxnb = int(os.environ.get("DKV_EDGE_ROUTE_MAXNB", "512"))
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
        import dkv_core as _dkv_core
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
        decay_factor = float(os.environ.get("DKV_SRL_DECAY_FACTOR", "1.0"))
        
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
    # native & MLX). Cross-runtime toggle: DKV_HIGH_QUALITY_ROUTING=1.
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
K_FIXED = int(os.environ.get("DKV_SRL_K_FIXED", "64"))


def route_blocks_relevance(
    Q:              torch.Tensor,   # [H, D] current ROTATED query (all query heads)
    pool,                           # NativeBlockPool
    block_indices:  torch.Tensor,   # [N] candidate pool slot IDs for this layer
    anchor_indices: torch.Tensor,   # [N] absolute anchor positions (same order)
    scale:          float,
    cos:            Optional[torch.Tensor] = None,
    sin:            Optional[torch.Tensor] = None,
    srl_state:      Optional[Any] = None,
) -> torch.Tensor:                  # [K<=N] selected slot IDs, best-first, distinct
    """MLX-parity block router: rank blocks by exact q·k relevance, take top-K.

    Direct port of mlx_dkv_wrapper._block_relevance_residual: a block's
    relevance is the max over query heads of max(q·anchor, max over its stored
    residual keys of q·k), fp16 products with fp32 accumulation, then plain
    top-K (DKV_TOPK_BLOCKS; default = pool.routing_topk_default = 4096 // block_size
    = 64 on CUDA; K = max(topk, topk_frac·N, k_min)).
    """
    N = block_indices.numel()
    # DKV_TOPK_BLOCKS: an explicit env value always wins. When unset, use the
    # block_size-derived default the manager stamped on the pool (MLX routed-token-
    # budget parity: 4096 // block_size = 64 on CUDA's block_size=64). The old flat
    # "16" covered only 16*64=1024 tokens and dropped a distant needle's block from
    # the top-K, breaking deep-context retrieval unless the user knew to set K=64.
    _pool_default = int(getattr(pool, "routing_topk_default", 16) or 16)

    # Derive K from the block span the pool ACTUALLY wrote, so the routed TOKEN
    # budget stays at MLX's 4096 regardless of how wide the blocks came out.
    #
    # K on its own is not the comparable quantity — K * block_span is. MLX has a
    # FIXED block_size of 256, so its K=16 always means 4096 routed tokens. This
    # side blocks ADAPTIVELY: kv_runtime_manager's own pool-sizing code says the
    # "REAL mean block size (adaptive schedule mean ~= 32 for short contexts)"
    # and uses max(32, min(micro_block_size, 64)), while routing_topk_default is
    # computed 180 lines later from max(micro_block_size, 257). Both cannot be
    # right, and the anchors settle it: a Qwen3.5-2B 8k run logs block anchors at
    # 0, 1235, 1300, 1365, 2405, 2470 ... — spaced 65, i.e. 64 tokens + 1 anchor,
    # not 257. So K=16 was routing ~1040 tokens where MLX routes 4096, a 4x
    # coverage shortfall that grows worse the longer the context.
    #
    # Using the observed span keeps BOTH regimes right: 256-token blocks still
    # give 4096//257 -> 16 (unchanged, and the 16k measurement behind the current
    # default stays valid), while 64-token blocks now give 64 for the same token
    # budget. Falls back to the pool's static default before the first write.
    _span = int(getattr(pool, "observed_block_span", 0) or 0)
    if _span > 0:
        _pool_default = max(16, 4096 // max(1, _span + 1))   # +1 for the anchor row

    _topk_env = os.environ.get("DKV_TOPK_BLOCKS")
    if _topk_env is None or _topk_env.strip() == "":
        topk = _pool_default
    else:
        try:
            topk = int(_topk_env)
        except ValueError:
            topk = _pool_default
    try:
        topk_frac = float(os.environ.get("DKV_TOPK_FRAC", "0.0"))
    except ValueError:
        topk_frac = 0.0
    if topk <= 0:
        # This is CUDA's attend-all / ground-truth comparison mode (MLX's
        # equivalent is its own, differently-named DKV_HIGH_QUALITY_ROUTING —
        # see the name-collision warning near this file's top).
        return block_indices  # routing disabled — attend every block
    
    k_eff = max(topk, int(topk_frac * N))
    if N <= k_eff:
        return block_indices

    # DKV_ROUTER_ROPE (default 1 = original behaviour). NOTE: MLX is NOT a
    # raw-key router — MLX captures keys POST-RoPE (mlx_dkv_wrapper.py:4448
    # keys_rot = rope(keys)) and its _block_relevance_residual scores
    # q_rot · k_rot, i.e. it is ALSO position-aware. So CUDA's per-key rotation
    # here is architecturally correct parity, NOT the divergence — do not assume
    # removing it "matches MLX". This flag exists only as an A/B knob: setting
    # DKV_ROUTER_ROPE=0 makes routing SELECTION content-only (no relative-
    # position decay), a heuristic that could keep a distant needle's block in
    # the top-K better than the decayed score — worth measuring, not a proven fix.
    _route_rope = os.environ.get("DKV_ROUTER_ROPE", "1") == "1"

    # Under DKV_ROTATED_POOL the pool holds POST-RoPE keys (MLX's convention),
    # so rotating here would rotate them a second time. MLX's own router does no
    # rotation at all for exactly this reason: _block_relevance_residual takes
    # comp_anc_k / comp_res_k already in their true frames and just scores q.k.
    # This is the single source of truth for the convention -- see
    # triton_fused_decode.pool_stores_rotated_k.
    try:
        from native_core.sparse_decode.triton_fused_decode import pool_stores_rotated_k
        if pool_stores_rotated_k():
            _route_rope = False
    except Exception:                                            # noqa: BLE001
        pass

    # Promote Q to 3D if it is 2D
    is_3d = (Q.dim() == 3)
    if is_3d:
        H, L, D = Q.shape
    else:
        H, D = Q.shape
        L = 1
        Q = Q.unsqueeze(1)  # [H, 1, D]

    slots_long = block_indices.long()
    anc = pool.anchors_K[slots_long].clone()                    # [N, H_kv, D] fp16

    if _route_rope and anchor_indices is not None and cos is not None and sin is not None:
        # Reshape by cos's OWN last dim (rotary_dim), not the key's head_dim.
        #
        # cos/sin come from rotary_emb as [1, L, rotary_dim]. Reshaping them to
        # (-1, head_dim) is only valid when rotary_dim == head_dim. On a partial-
        # rotary model (Qwen3.5-2B: head_dim 256, rotary_dim 64) it RAISES —
        #   RuntimeError: shape '[-1, 256]' is invalid for input of size 697792
        # — at every context length tested (2822, 10903, 32942 tokens; L*64 is
        # not a multiple of 256 in any of them). The call site in dkv_attention.py
        # wraps this in `except Exception: # SRL failure is non-fatal` with the
        # message behind DKV_SRL_VERBOSE, so the router simply never ran on that
        # model and decode silently fell back to attending every block.
        #
        # That is why seven consecutive changes to this function produced
        # byte-identical needle results on Qwen3.5-2B: none of them executed.
        # Qwen2.5-1.5B is full-rotary (128 == 128) so the reshape happened to be
        # valid there, which is why the two models behaved so differently.
        #
        # Even where the reshape does not raise it is wrong: it packs
        # head_dim/rotary_dim consecutive POSITIONS into one row, so the table is
        # that many times too short and every deeper anchor clamps to the last row.
        _rot_dim = cos.shape[-1]
        cos_flat = cos.reshape(-1, _rot_dim)
        sin_flat = sin.reshape(-1, _rot_dim)
        # .long(): index tensors must be long/int/byte/bool. anchor_indices is
        # int32 (metadata) which is valid, but coerce defensively so a narrower
        # int dtype (e.g. int16) can never raise "tensors used as indices...".
        anc_pos = anchor_indices.long().clamp(min=0, max=cos_flat.shape[0] - 1)
        cos_anc = cos_flat[anc_pos].to(device=anc.device, dtype=anc.dtype).unsqueeze(1)  # [N, 1, rot]
        sin_anc = sin_flat[anc_pos].to(device=anc.device, dtype=anc.dtype).unsqueeze(1)  # [N, 1, rot]
        # Partial-rotary aware: rotate the leading rotary_dim slice, pass the
        # tail through. Reduces to the original full-width form when
        # rotary_dim == head_dim, so full-rotary models are bit-identical.
        anc = _rope_partial(anc, cos_anc, sin_anc)
    else:
        cos_anc = sin_anc = None

    H_kv = anc.shape[1]
    gpk = max(1, H // H_kv)

    # Reshape Q to [H_kv, gpk * L, D]
    q_reshaped = Q.reshape(H_kv, gpk * L, D)            # [H_kv, gpk * L, D]

    # Permute anchors to [H_kv, D, N]
    anc_permuted = anc.permute(1, 2, 0)                 # [H_kv, D, N]

    # Compute anchor scores using batched matrix multiplication (BMM)
    # [H_kv, gpk * L, D] @ [H_kv, D, N] -> [H_kv, gpk * L, N]
    s_anc = torch.bmm(q_reshaped.float(), anc_permuted.float()) * scale
    s_anc = s_anc.reshape(H_kv, gpk, L, N)

    res_scores = None
    res_k = getattr(pool, "residual_K_values", None)
    res_pos = getattr(pool, "residual_K_positions", None)
    route_prefill_res = (os.environ.get("DKV_ROUTE_PREFILL_RESID", "0") == "1")
    if res_k is not None and res_pos is not None and res_k.numel() > 0 and (not is_3d or route_prefill_res):
        try:
            r_route = int(os.environ.get("DKV_ROUTE_RESIDUALS", "0"))
        except ValueError:
            r_route = 0
        R_all = res_k.shape[1]
        # Route on the top-64 residuals, which is what MLX actually does:
        #
        #     _rr = int(os.environ.get("DKV_ROUTE_RESIDUALS", "0"))
        #     self.route_residuals = _rr if _rr > 0 else min(64, self.max_residual)
        #     R_route = min(self.route_residuals, self.max_residual)
        #                       (mlx_dkv_wrapper.py:1756-1757 and :4022)
        #
        # THE COMMENT THIS REPLACES CLAIMED THE OPPOSITE. It said "score against
        # EVERY stored residual, as MLX does" and removed a cap of 64 as a bug.
        # MLX has never scored against every residual: 64 is its default, chosen
        # by a documented sweep recorded at :1750-1755 -- "R=16/32 -> NIAH ok but
        # SYNTHESIS breaks ... R=64 -> NIAH AND synthesis both pass ... the sweet
        # spot; R=128 -> 6.9 tps". The cap was removed on a misreading of the
        # reference, so this is a regression being undone, not a new policy.
        #
        # The removal's reasoning was also wrong about what entries 64..127 are.
        # It argued the cap "silently hid HALF of every block's exact keys from
        # routing -- and residual keys are precisely the verbatim content
        # (digits, codes) that routing most needs to see". But the residual array
        # is stored RANKED BEST-FIRST WITH THE COVERAGE SCAFFOLD APPENDED -- that
        # ordering is load-bearing precisely because the head of the array IS the
        # routing signature. So the tail is mostly evenly-spaced coverage rows:
        # low-error, deliberately non-distinctive filler positions. They are not
        # extra verbatim content.
        #
        # Feeding them to a MAX can only add noise. res_scores is
        # s_res.max(dim=-1), so one generic filler token that happens to match
        # the query lifts its whole block's relevance above a block that actually
        # holds the answer. That costs nothing when routing is comfortable and
        # everything when it is marginal -- which is exactly the regime the
        # remaining failures live in (8k@0.5 at 16 of ~42 blocks, 32k@0.9 at
        # 16 of ~128).
        #
        # DKV_ROUTE_RESIDUALS>0 still sets it explicitly, same as MLX.
        R = min(R_all, r_route) if r_route > 0 else min(64, R_all)
        rk = res_k[slots_long, :R].clone()                      # [N, R, H_kv, D]
        rvalid = (res_pos[slots_long, :R] >= 0)         # [N, R]

        if _route_rope and cos is not None and sin is not None:
            # Same fix as the anchor rotation above: reshape by cos's own
            # rotary_dim, never by the key's head_dim.
            _rot_dim = cos.shape[-1]
            cos_flat = cos.reshape(-1, _rot_dim)
            sin_flat = sin.reshape(-1, _rot_dim)
            # .long(): pool.residual_K_positions is int16 (native_block_pool.py),
            # which PyTorch REJECTS as an index dtype ("tensors used as indices
            # must be long, int, byte or bool tensors"). This is the crash that
            # took down SRL-gated decode routing. Coerce before indexing cos/sin.
            # ABSOLUTE position, not the within-block offset. residual_K_positions
            # stores an index into the block's ACTIVE-token array (0..T_active-1);
            # the anchor holds block-local slot 0, so the residual's true position
            # is anchor_idx + 1 + offset (streaming_sparse_ingest.py:1206/1216,
            # dkv_attention.py:1385, dkv_decode.metal:153).
            #
            # This used the raw offset, so EVERY residual key in the pool was
            # rotated as if it lived in the first 256 tokens of the sequence. A
            # residual in the block at anchor 5911 was rotated at position ~137.
            # The anchor above is rotated at its TRUE position, so the two halves
            # of the relevance score were computed in different rotational frames
            # and then combined with torch.maximum — meaning the residual term,
            # the one that exists to keep a buried code's block in the top-K,
            # contributed a number with no defined relationship to q. It could
            # promote an arbitrary block and demote the needle's.
            #
            # MLX never hits this: it captures keys POST-RoPE at their true
            # positions (mlx_dkv_wrapper.py:4448) and _block_relevance_residual
            # does no rotation at all. CUDA stores UNROTATED K by design
            # (dkv_backend.py:40) and re-rotates on read, so read sites are
            # exactly where the position convention has to be re-derived — and
            # both CUDA read sites had it wrong.
            _res_off = res_pos[slots_long, :R].long().clamp(min=0)
            if anchor_indices is not None:
                _anc_abs = anchor_indices.long().to(_res_off.device).view(-1, 1)
                res_p = _anc_abs + 1 + _res_off
            else:
                res_p = _res_off
            res_p = res_p.clamp(min=0, max=cos_flat.shape[0] - 1)
            cos_res = cos_flat[res_p].to(device=rk.device, dtype=rk.dtype).unsqueeze(2)  # [N, R, 1, rot]
            sin_res = sin_flat[res_p].to(device=rk.device, dtype=rk.dtype).unsqueeze(2)  # [N, R, 1, rot]
            rk = _rope_partial(rk, cos_res, sin_res)

        # Permute rk to [H_kv, D, N * R]
        rk_permuted = rk.permute(2, 3, 0, 1).reshape(H_kv, D, N * R)

        # BMM: [H_kv, gpk * L, D] @ [H_kv, D, N * R] -> [H_kv, gpk * L, N * R]
        s_res = torch.bmm(q_reshaped.float(), rk_permuted.float()) * scale
        s_res = s_res.reshape(H_kv, gpk, L, N, R)

        # ADD THE ANCHOR TERM BACK. This is the whole residual router.
        #
        # The two runtimes store a residual in DIFFERENT FRAMES, and only one of
        # them can be scored directly:
        #
        #   MLX   res_k_active = mx.take(block_k_t, top_k_indices + 1, axis=0)
        #         (mlx_dkv_wrapper.py:3875) -- the token's ABSOLUTE rotated key.
        #         _block_relevance_residual then scores q . comp_res_k, which IS
        #         that token's true attention score.
        #
        #   CUDA  res_K_vals = delta_K[fact_positions_K]
        #         (compression/lowrank.py:673, and the batched path at :1645)
        #         -- ANCHOR-RELATIVE, i.e. (k_exact - k_anchor). The decode
        #         kernels know this and reconstruct with s_anchor + q.rk
        #         (triton_fused_decode.py:725). This function did not: it scored
        #         the bare delta.
        #
        # So CUDA's residual term was the true score MINUS s_anc. Because an
        # anchor is a full key vector and a residual is a small delta, |q.rk| is
        # normally far below |q.anc|, so
        #
        #     torch.maximum(s_anc, q.rk)  ->  s_anc,  for essentially every block
        #
        # and the router collapsed into an ANCHOR-ONLY router. That silently
        # deletes the entire reason the residual router exists -- MLX's own
        # docstring: "Unlike a min/max box or an SVD low-rank score -- both cheap
        # summaries that by construction miss low-energy outliers -- the
        # residuals ARE the block's outlier tokens (a buried passcode is exactly
        # such a token)." A buried code contributes nothing to its block's anchor,
        # so an anchor-only router ranks the needle's block by its PROSE, which is
        # indistinguishable from every other filler block.
        #
        # It also explains the depth asymmetry directly: at depth 0.0 the needle
        # sits in the sink/system-prompt block, which top-K keeps on anchor score
        # alone; deeper needles have nothing keeping them in.
        #
        # Under the correction form (DKV_RESIDUAL_EXACT_KEYS=0) rk is
        # (exact - recon), and the true key is anchor + recon + rk -- recon is a
        # rank-R product this function deliberately never builds, so no exact
        # score is available there. Left as-is rather than made "less wrong" on a
        # non-default path with no measurement behind it.
        try:
            from native_core.compression.lowrank import _exact_keys_enabled
            _res_is_anchor_relative = bool(_exact_keys_enabled(rk.device))
        except Exception:                                        # noqa: BLE001
            _res_is_anchor_relative = False
        if _res_is_anchor_relative:
            s_res = s_res + s_anc.unsqueeze(-1)

        # Apply validity mask
        s_res = s_res.masked_fill(~rvalid.view(1, 1, 1, N, R), float("-inf"))
        res_scores = s_res.max(dim=-1).values           # [H_kv, gpk, L, N]

    if res_scores is not None:
        relevance = torch.maximum(s_anc, res_scores)
    else:
        relevance = s_anc

    # relevance has shape [H_kv, gpk, L, N]
    # Max-reduce over L (dim=1) and H (dim=0 after reshape)
    relevance = relevance.reshape(H_kv * gpk, L, N).max(dim=1).values.max(dim=0).values   # [N]

    # ── Edge-aware routing propagation (MLX parity) ──────────────────────────
    # MLX applies this to the SAME relevance vector, between scoring and top-K:
    #
    #     relevance = _block_relevance_residual(q, ak, rk[:, :R_route], ...)
    #     if _er_on:
    #         relevance = _edge_propagate_relevance(relevance, ak, beta, max_nb)
    #     sel = mx.argsort(relevance)[-k_eff:]
    #                                       (mlx_dkv_wrapper.py:4024-4030)
    #
    # This function -- the one production actually routes with -- went straight
    # from the max-reduce to torch.topk. DKV_EDGE_ROUTING defaults to "1" on
    # BOTH runtimes, so the flag read as enabled while the production path never
    # consulted it. There IS an edge implementation in this file already, but it
    # lives in a different routing function, so route_blocks_relevance never saw
    # it. Same family as DKV_COMPRESSED_MIN_CTX having no CUDA reader.
    #
    # What it does, in MLX's words: the router scores each block INDEPENDENTLY,
    # so it has no notion of which blocks are CONNECTED. When what the query
    # matches and what actually answers it live in different blocks, top-K can
    # take the first and drop the second. This diffuses relevance one hop over a
    # block-to-block similarity graph built from the anchor keys:
    #
    #     relevance <- relevance + beta * (A_hat . relevance)
    #
    # A_hat = row-normalised positive anchor-cosine adjacency, self-loops
    # removed. k_eff is unchanged, so the same number of blocks is attended and
    # decode throughput is flat; the NxN graph is tiny and this runs once per
    # routing interval, not per token.
    _er_on = os.environ.get("DKV_EDGE_ROUTING", "1").strip().lower() not in (
        "0", "", "false", "off")
    try:
        _er_beta = float(os.environ.get("DKV_EDGE_ROUTE_BETA", "0.25"))
    except ValueError:
        _er_beta = 0.25
    try:
        _er_maxnb = int(os.environ.get("DKV_EDGE_ROUTE_MAXNB", "512"))
    except ValueError:
        _er_maxnb = 512
    if _er_on and _er_beta > 0.0 and 3 <= N <= _er_maxnb:
        # anc is [N, H_kv, D] and already in whatever rotational frame the pool
        # uses, which is what MLX passes too (its `ak` is comp_anc_k).
        _akf = anc.reshape(N, -1).float()
        _akn = _akf / (_akf.norm(dim=-1, keepdim=True) + 1e-6)
        _A = torch.matmul(_akn, _akn.t())                                  # cosine
        _A = torch.clamp(_A - torch.eye(N, device=_A.device, dtype=_A.dtype), min=0.0)
        _A = _A / (_A.sum(dim=-1, keepdim=True) + 1e-6)                    # row-normalised
        _prop = torch.matmul(_A, relevance.float())
        relevance = relevance + _er_beta * _prop.to(relevance.dtype)

    # Plain top-K, matching MLX's _block_relevance_residual exactly (this
    # function's own docstring). An earlier version force-included a "sink"
    # block (lowest anchor_indices) regardless of relevance, unconditionally
    # spending one of the k_eff slots on it -- undocumented, MLX has no such
    # forcing, and a sibling flag using this same function (DKV_DECODE_PRUNE_K)
    # is confirmed on A100 to drop answer-critical blocks at matched K, which a
    # forced non-relevance slot is a plausible contributor to. Removed to match
    # the documented "direct port" contract.
    sel = torch.topk(relevance, k=k_eff).indices
    return block_indices[sel].to(torch.int32)

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
    ``block_indices`` (dkv_attention.py:676), and the sparse kernel scores
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
