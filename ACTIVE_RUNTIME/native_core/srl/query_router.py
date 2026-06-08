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


# ── Level-1 Anchor Screening ──────────────────────────────────────────────────

def two_level_gate(
    Q:         torch.Tensor,         # [H, D] current query (all heads)
    pool,                            # NativeBlockPool
    slot_ids:  torch.Tensor,         # [M] candidate pool slot IDs (int32/int64)
    scale:     float,
    k_pass:    int,                  # how many to keep
) -> torch.Tensor:                   # [k_pass] pool slot IDs
    """
    Cheap anchor-only dot-product screening.

    Loads only pool.anchors_K (small, often in L2 cache) and computes
    a mean-query dot product to rerank the candidate set.

    Cost: M × D multiplications. For M=100, D=128: ~13K ops — sub-ms.
    """
    N = slot_ids.shape[0]
    if N <= k_pass:
        return slot_ids

    try:
        import diffkv_core as _dkv_core
        if getattr(_dkv_core, "HAS_SRL_ROUTER", False):
            return _dkv_core.anchor_screen(Q, pool.anchors_K, slot_ids, scale, k_pass)
    except ImportError:
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

    # ── Step 2: Semantic ANN search ───────────────────────────────────────
    k_semantic = max(1, int(K * _SEM_FRAC))
    semantic_slots_t = srl_state.semantic_index.search(q_desc, k=k_semantic)
    semantic_slots   = semantic_slots_t.tolist()

    # ── Step 3: Lexical inverted index lookup (IDF & Term Coverage Boost) ──
    k_lexical   = max(1, int(K * _LEX_FRAC))
    if query_tokens is not None:
        recent_toks = query_tokens
    else:
        recent_toks = srl_state.recent_generated_tokens[-32:] + getattr(srl_state, "current_query_tokens", [])[-128:]

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
    graph_slots_t: Optional[torch.Tensor] = None
    idx_map   = srl_state.semantic_index
    neighbors = srl_state.chunk_graph.neighbors   # [N, MAX_DEGREE] CPU int32

    # Seed expansion with top-5 semantic, top-10 rare lexical, and top-5 lexical matches
    top_sem_slots = semantic_slots_t[:5].tolist() if semantic_slots_t.numel() > 0 else []
    seeds = list(dict.fromkeys(top_sem_slots + rare_lex_slots[:10] + lexical_slots[:5]))

    if seeds and neighbors.shape[0] > 0:
        seeds_tensor = torch.tensor(seeds, dtype=torch.int32)
        row_ids = idx_map.slot_to_row_vec(seeds_tensor)                     # [M]
        valid_rows = row_ids[row_ids >= 0]                                  # filter misses

        if valid_rows.numel() > 0:
            # Gather all neighbor row indices in one tensor op: [M_valid, MAX_DEGREE]
            nb_rows = neighbors[valid_rows]
            nb_flat = nb_rows.view(-1).to(torch.int64)                       # [M_valid*MAX_DEGREE]
            N_idx   = idx_map.slot_ids.shape[0]
            nb_valid = nb_flat[(nb_flat >= 0) & (nb_flat < N_idx)]
            # Translate row indices → pool slot IDs
            if nb_valid.numel() > 0:
                slot_ids_cpu = idx_map.slot_ids.cpu()
                graph_slots_t = slot_ids_cpu[nb_valid]                       # [M_valid*MAX_DEGREE] int32

    graph_slots: List[int]
    if graph_slots_t is not None and graph_slots_t.numel() > 0:
        # Deduplicate preserving order
        graph_slots_t = torch.unique(graph_slots_t)[:max(k_graph, 20)]
        graph_slots   = graph_slots_t.tolist()
    else:
        graph_slots = []

    # ── Step 5: Recency window ────────────────────────────────────────────
    k_recent     = max(1, int(K * _RECENCY_FRAC))
    recent_slots = srl_state.ordered_slot_ids[-k_recent:]

    # ── Step 6: Sink blocks (always include: block 0 + special tokens) ────
    sink = srl_state.sink_blocks

    # ── Step 7: Merge and deduplicate (order: sink → rare_lex → graph → semantic → lexical → recent) ─
    combined = list(dict.fromkeys(
        sink + rare_lex_slots + graph_slots + semantic_slots + lexical_slots + recent_slots
    ))[:K]

    if N - len(combined) <= 2:
        combined = srl_state.ordered_slot_ids

    if not combined:
        # Fallback: use all ordered slots (shouldn't happen in practice)
        combined = srl_state.ordered_slot_ids[:K]

    # ── Step 8: Level-1 anchor reranking over combined set ────────────────
    combined_tensor = torch.tensor(combined, dtype=torch.long, device=Q.device)
    if combined_tensor.shape[0] > K:
        if hasattr(pool, "anchors_K"):
            combined_tensor = two_level_gate(Q, pool, combined_tensor, scale, k_pass=K)
        else:
            combined_tensor = combined_tensor[:K]

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
