"""
native_core/srl/session_srl_state.py

Per-session SRL state: holds all routing indexes and running stats.
One instance per session, attached to KVRuntimeManager._session_srl[session_id].

Memory footprint for a 25K-token session (781 blocks):
  desc_matrix:   781 × 64 × 2 B  =  100 KB  (GPU)
  chunk_graph:   781 × 8  × 4 B  =   25 KB  (CPU/GPU)
  inverted_index: ~5000 terms × avg 12 blocks × 4 B = 240 KB  (CPU)
  ordered_slot_ids: 781 × 4 B    =    3 KB  (CPU)
  Total: ~368 KB  (vs ~400 MB compressed KV pool → <0.1% overhead)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch

from native_core.srl.semantic_index import SemanticIndex
from native_core.srl.chunk_graph import ChunkGraph
from native_core.srl.inverted_index import InvertedTokenIndex


@dataclass
class SessionSRLState:
    """All routing indexes for one session."""

    semantic_index:     SemanticIndex
    chunk_graph:        ChunkGraph
    inverted_index:     InvertedTokenIndex

    # Pool slot IDs in chronological order (one per block, layer-0 reference)
    ordered_slot_ids:   List[int]

    # Pool slot IDs that are always included (block 0, special-token blocks)
    sink_blocks:        List[int]

    # Adaptive-K running state
    recent_miss_rate:   float = 0.0
    k_multiplier:       float = 1.0
    call_count:         int   = 0

    # Rolling window of recently generated token IDs for lexical lookup
    recent_generated_tokens: List[int] = field(default_factory=list)

    ordered_anchor_idxs: List[int] = field(default_factory=list)
    cached_len: int = 0

    # Structured Attention Segmenting (SAS) fields
    segment_ids: Dict[int, int] = field(default_factory=dict)
    current_query_segment_id: int = 0
    concept_tok_1: int = -1
    concept_tok_2: int = -1

    # Eagle-Guided Query Anchoring & Dynamic Routing (EQA-DR) fields
    prompt_eagle_scores: Optional[torch.Tensor] = None
    prompt_anchors: List[int] = field(default_factory=list)
    recent_decode_keys: List[torch.Tensor] = field(default_factory=list)
    generated_token_slots: List[int] = field(default_factory=list)
    dynamic_anchors: List[int] = field(default_factory=list)

    # Per-decode-step cached slot selection (shared across all 28 layers)
    # Set by layer 0, consumed by layers 1–27
    current_step_slots: Optional[torch.Tensor] = None
    current_step_count: int = 0   # decode step counter for validation
    last_prefill_q: Optional[torch.Tensor] = None
    current_step_factual_tokens: set = field(default_factory=set)
    current_step_factual_sequences: List[List[int]] = field(default_factory=list)
    current_step_max_similarity: float = 0.0
    vsl_active_candidates: List[List[int]] = field(default_factory=list)
    vsl_consecutive_helpers: int = 0
    # First decode-step query vector stored as an anchor.
    # Used to blend with current query during factual store lookups so that
    # semantic drift (query vector accumulating generated text) cannot pull
    # retrieval away from the original question topic.
    factual_anchor_q: Optional[torch.Tensor] = None

    # Entity-subgraph tracking for relationship binding.
    # current_entity_id: document position (start_idx) of the prime entry for
    #   the entity currently being generated about. -1 = no context established.
    #   Persists across decode steps within one response; reset at generation start.
    # current_step_sequence_entity_ids: parallel to current_step_factual_sequences;
    #   entity_id (prime start_idx) for each sequence. -1 = unknown.
    # current_step_sequence_is_prime: parallel list; True if the sequence is a prime entry.
    current_entity_id: int = -1
    # Dual-entity mode for comparison questions (e.g., "Compare EP2 and EP3").
    # When active, both entities' factual sequences are available for generation
    # but cross-entity contamination is prevented via strict VSL filtering.
    dual_entity_mode: bool = False
    dual_entity_ids: List[int] = field(default_factory=list)  # [entity_id_1, entity_id_2]
    # RC5 — explicit comparison mode: when ≥2 entities are being compared we do
    # NOT let them interleave; we lock generation to one entity at a time and
    # advance only once it has been substantively covered.  comparison_entities
    # is the ordered block sequence; comparison_active_idx the current block;
    # comparison_covered the entities already produced.
    comparison_entities: List[int] = field(default_factory=list)
    comparison_active_idx: int = 0
    comparison_covered: set = field(default_factory=set)
    current_step_sequence_entity_ids: List[int] = field(default_factory=list)
    current_step_sequence_is_prime: List[bool] = field(default_factory=list)
    # current_step_sequence_prefixes: parallel list; the source tokens preceding
    # each sequence's span (RC2 quote-grounded connectives). [] for triples.
    current_step_sequence_prefixes: List[List[int]] = field(default_factory=list)

    # Per-layer ordered slots (all layers share pool slots → same list for all)
    # Maps layer_idx → ordered list of pool slot IDs (currently identical for every layer)
    layer_slot_ids: Dict[int, List[int]] = field(default_factory=dict)

    # ── Config (all overridable via SRL_CONFIG or env vars) ──────────────────
    k_min:             int   = 20
    k_max:             int   = 200
    k_semantic_frac:   float = 0.50
    k_lexical_frac:    float = 0.15
    k_graph_frac:      float = 0.15
    k_recency_frac:    float = 0.20
    routing_threshold: int   = 2
    overlap_threshold: float = 0.15
    graph_hop_decay:   float = 0.5
    srl_age_penalty:   float = 0.01

    def n_active_blocks(self) -> int:
        """Number of compressed blocks in this session."""
        return len(self.ordered_slot_ids)

    def update_generated_tokens(self, token_id: int, maxlen: int = 64) -> None:
        """Append a newly generated token ID to the rolling window."""
        self.recent_generated_tokens.append(token_id)
        if len(self.recent_generated_tokens) > maxlen:
            self.recent_generated_tokens = self.recent_generated_tokens[-maxlen:]

    def update_miss_rate(self, attention_weights_k: torch.Tensor) -> None:
        """
        Update the exponential moving average miss-rate signal.

        A flat attention distribution over the selected blocks (max_weight ≈ 1/K)
        suggests the truly-relevant block wasn't selected → high miss signal.
        A peaked distribution (one block dominated) → low miss signal.
        """
        K = attention_weights_k.shape[0]
        if K == 0:
            return
        max_w = float(attention_weights_k.max())
        expected_max = 1.0 / K
        denom = (1.0 - expected_max) + 1e-8
        miss_signal = max(0.0, min(1.0, 1.0 - (max_w - expected_max) / denom))

        alpha = 0.05  # EMA smoothing
        self.recent_miss_rate = (1.0 - alpha) * self.recent_miss_rate + alpha * miss_signal

        # Boost K multiplier when miss rate is high
        if self.recent_miss_rate > 0.4:
            self.k_multiplier = min(self.k_multiplier * 1.2, 3.0)
        else:
            self.k_multiplier = max(self.k_multiplier * 0.99, 1.0)

    def expand_neighborhood(self, seed_slots: Set[int]) -> Set[int]:
        expanded = set(seed_slots)
        if self.chunk_graph is None or self.semantic_index is None:
            return expanded
            
        slot_to_row = {int(slot): idx for idx, slot in enumerate(self.semantic_index.slot_ids.tolist())}
        neighbors_tensor = self.chunk_graph.neighbors
        
        for slot in list(seed_slots):
            if slot in slot_to_row:
                row_idx = slot_to_row[slot]
                if row_idx < neighbors_tensor.shape[0]:
                    nb_rows = neighbors_tensor[row_idx].tolist()
                    for nb_row in nb_rows:
                        if nb_row >= 0 and nb_row < self.semantic_index.slot_ids.shape[0]:
                            nb_slot = int(self.semantic_index.slot_ids[nb_row].item())
                            expanded.add(nb_slot)
        return expanded

    def update_query_segment(self, token_id: int) -> None:
        # Look at the last 12 generated tokens
        recent_window = self.recent_generated_tokens[-12:]
        seg1_score = 0
        seg2_score = 0
        
        for tid in recent_window:
            if self.inverted_index is not None and tid in self.inverted_index.occurrences:
                for slot, _, _ in self.inverted_index.occurrences[tid]:
                    seg = self.segment_ids.get(slot, 0)
                    if seg == 1:
                        seg1_score += 1
                    elif seg == 2:
                        seg2_score += 1
                        
        if seg1_score > seg2_score + 2:
            self.current_query_segment_id = 1
        elif seg2_score > seg1_score + 2:
            self.current_query_segment_id = 2
        else:
            self.current_query_segment_id = 0

    def update_dynamic_anchors(self, stop_token_ids: Set[int]) -> None:
        L_g = len(self.recent_decode_keys)
        if L_g < 2 or L_g % 4 != 0:
            return
            
        keys_tensor = torch.stack(self.recent_decode_keys) # [L_g, head_dim]
        sim = torch.matmul(keys_tensor, keys_tensor.T) / math.sqrt(keys_tensor.shape[1])
        
        # Apply causal mask
        mask = torch.triu(torch.ones(L_g, L_g, device=sim.device), diagonal=1).T
        sim = sim.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(sim, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        R_gen = attn.sum(dim=0).cpu().tolist()
        
        self.dynamic_anchors = []
        for i in range(L_g):
            if R_gen[i] > 1.5:
                if i < len(self.recent_generated_tokens):
                    tid = self.recent_generated_tokens[i]
                    if tid not in stop_token_ids:
                        if i < len(self.generated_token_slots):
                            self.dynamic_anchors.append(self.generated_token_slots[i])

    def setup_sas_and_eqa(self, token_ids: torch.Tensor, stop_token_ids: Set[int], tokenizer: Optional[Any] = None) -> None:
        if self.prompt_eagle_scores is None or token_ids is None:
            return
            
        total_seq_len = token_ids.numel()
        valid_candidates = []
        for i in range(total_seq_len):
            tid = int(token_ids[i].item())
            if tid not in stop_token_ids:
                score = float(self.prompt_eagle_scores[i].item())
                
                # Check for known comparative keywords and boost their scores
                try:
                    word = ""
                    if tokenizer is not None:
                        word = tokenizer.decode([tid]).strip().lower()
                except Exception:
                    pass
                
                if word in {"1", "2", "3", "ep2", "ep3", "hermitian", "diabolic", "diabolical", "conical", "branch", "exceptional", "symmetric", "eigenvalue", "eigenvalues", "eigenvector", "eigenvectors", "codimension", "topology", "monodromy", "loop", "left", "right"}:
                    score += 5.0
                    
                valid_candidates.append((i, score, tid))
                
        # Sort by score descending
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        self.prompt_anchors = [i for i, _, _ in valid_candidates[:3]]
        
        # Extract unique token IDs
        candidate_tids = []
        seen_tids = set()
        for _, _, tid in valid_candidates:
            if tid not in seen_tids:
                candidate_tids.append(tid)
                seen_tids.add(tid)
                
        # Find concepts with low Jaccard overlap
        def get_slots(tid):
            slots = set()
            if self.inverted_index is not None and tid in self.inverted_index.occurrences:
                for slot, _, _ in self.inverted_index.occurrences[tid]:
                    slots.add(slot)
            return slots
            
        def jaccard(s1, s2):
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)
            
        concept_tok_1 = -1
        concept_tok_2 = -1
        
        first_idx = -1
        for idx, tid in enumerate(candidate_tids):
            s = get_slots(tid)
            if len(s) >= 1:
                concept_tok_1 = tid
                first_idx = idx
                break
                
        if first_idx != -1:
            s1 = get_slots(concept_tok_1)
            for j in range(first_idx + 1, len(candidate_tids)):
                tid2 = candidate_tids[j]
                s2 = get_slots(tid2)
                if len(s2) >= 1:
                    j_val = jaccard(s1, s2)
                    if j_val <= 0.2:
                        concept_tok_2 = tid2
                        break
            if concept_tok_2 == -1:
                # Fallback to minimum Jaccard overlap
                min_j = 1.0
                best_tid2 = -1
                for j in range(first_idx + 1, len(candidate_tids)):
                    tid2 = candidate_tids[j]
                    s2 = get_slots(tid2)
                    if len(s2) >= 1:
                        j_val = jaccard(s1, s2)
                        if j_val < min_j:
                            min_j = j_val
                            best_tid2 = tid2
                concept_tok_2 = best_tid2
                
        self.concept_tok_1 = concept_tok_1
        self.concept_tok_2 = concept_tok_2
        
        # Map segments
        self.segment_ids = {slot: 0 for slot in self.ordered_slot_ids}
        slots_1 = set()
        slots_2 = set()
        
        if self.concept_tok_1 != -1:
            slots_1 = get_slots(self.concept_tok_1)
        if self.concept_tok_2 != -1:
            slots_2 = get_slots(self.concept_tok_2)
            
        # Neighborhood Expansion
        expanded_1 = self.expand_neighborhood(slots_1)
        expanded_2 = self.expand_neighborhood(slots_2)
        
        for slot in self.ordered_slot_ids:
            in_1 = slot in expanded_1
            in_2 = slot in expanded_2
            if in_1 and in_2:
                self.segment_ids[slot] = 0
            elif in_1:
                self.segment_ids[slot] = 1
            elif in_2:
                self.segment_ids[slot] = 2
            else:
                self.segment_ids[slot] = 0
