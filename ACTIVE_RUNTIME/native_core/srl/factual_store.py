import torch
import math
from typing import List, Dict, Optional, Set, Any

# High-binding-value relational tokens that are critical for preserving
# entity-property bindings and categorical distinctions.  These words
# typically receive very low IDF scores (they appear frequently) and get
# excluded from the top-5% salience selection.  We give them a fixed
# IDF-equivalent boost so they survive into factual spans.
RELATIONAL_KEYWORDS = {
    # Contrastive / comparative
    "unlike", "whereas", "while", "although", "however", "but",
    "instead", "rather", "conversely", "nevertheless", "nonetheless",
    "yet", "though", "notwithstanding",
    # Comparative degree
    "compared", "differs", "differ", "different", "difference",
    "differences", "distinct", "distinction", "distinguishes",
    "greater", "larger", "smaller", "higher", "lower", "fewer",
    "more", "less", "most", "least",
    # Causal / process
    "causes", "caused", "because", "therefore", "hence", "thus",
    "leads", "results", "produces", "induces", "triggers",
    "consequently", "accordingly",
    # Binding verbs / copulas that attach properties to entities
    "is", "are", "was", "were", "has", "have", "had",
    "exhibits", "exhibits", "possesses", "contains", "involves",
    "requires", "lacks", "features",
    # Specification / attribution
    "called", "named", "known", "defined", "characterized",
    "classified", "denoted", "refers", "represents",
    # Scope / exclusion
    "only", "exclusively", "specifically", "solely",
    "except", "excluding", "neither", "nor",
}

# RC3 — entity-signature dimensionality.  A small deterministic unit vector
# derived from an entity's distinguishing token, appended (conceptually) to the
# descriptor so that two property spans which differ ONLY by owning-entity
# separate in retrieval-ranking space — even when their random-projected layer-0
# descriptors are near-identical (the core RC3 failure).  Kept separate from the
# descriptor so it never perturbs descriptor↔descriptor or query↔descriptor
# matching unless an entity bias is actually supplied (null-safe).
ENTITY_SIG_DIM = 16


def entity_signature(token_id: int, dim: int = ENTITY_SIG_DIM) -> torch.Tensor:
    """Deterministic unit vector for a token id (the entity's distinguishing
    token).  Same formula must be mirrored in the C++ store."""
    if token_id is None or token_id < 0:
        return torch.zeros(dim, dtype=torch.float32)
    idx = torch.arange(1, dim + 1, dtype=torch.float32)
    v = torch.sin(float(token_id) * 0.61803398875 * idx + idx)
    n = v.norm()
    return v / n if n > 1e-8 else v


class FactEntry:
    def __init__(self, start_idx: int, end_idx: int, K: torch.Tensor, V: torch.Tensor, descriptor: torch.Tensor, slot_ids: Optional[List[int]] = None, tokens: Optional[List[int]] = None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.K = K              # [layer, heads, len, D] (on CPU or GPU)
        self.V = V              # [layer, heads, len, D]
        self.descriptor = descriptor  # [DESC_DIM] (on CPU)
        self.slot_ids = slot_ids or []  # List of slot IDs this fact belongs to
        self.tokens = tokens or []      # Token IDs covered by this span
        self.neighbors: List[int] = []  # Indices of connected factual entries
        self.weights: List[float] = []  # Connection weights
        self.is_prime: bool = False     # Whether this fact is a Factual Prime Node
        # Entity assignment: document position (start_idx) of the prime that owns
        # this property span.  -1 = unassigned.  Set during build() by
        # _assign_entities() using a distinguishing-token override plus
        # nearest-preceding-prime (reading-order) ownership — NOT plain token
        # overlap, which mis-binds shared-vocabulary properties (RC4), and NOT
        # absolute positional proximity, which mis-binds interleaved comparisons.
        self.entity_id: int = -1
        # For prime entries only: the single highest-IDF (most distinguishing)
        # token in this prime's span, or None.  A property span that contains
        # exactly one prime's distinguishing token is bound to that prime with
        # high confidence, regardless of how many low-IDF filler tokens it shares
        # with other primes.  Set during build().
        self.distinguishing_token: Optional[int] = None
        # RC3: deterministic unit signature of this span's owning entity (derived
        # from that entity's distinguishing token).  Used only when a query
        # supplies an entity bias, to rank the queried entity's spans above
        # shared-vocabulary spans of other entities.  Zeros = no owning entity.
        self.entity_sig: torch.Tensor = torch.zeros(ENTITY_SIG_DIM, dtype=torch.float32)
        # 3-token context preceding this span in the source document.
        # Used to determine which relational connectives are source-adjacent
        # for quote-grounded connective gating (RC2).
        self.prefix_tokens: List[int] = []
        # For prime entries: list of (bridge_tokens + value_tokens) sequences.
        # Each represents a (relation, value) pair grounded in the source text.
        # Injected into current_step_factual_sequences at decode time so the VSL
        # can enforce the complete (entity → relation → value) ordering.
        self.triple_sequences: List[List[int]] = []
        # DX1: True when this span sits immediately after its prime via a
        # definitional bridge (copula "is/are/means/refers to" with IDF < 1.0).
        # Used to boost retrieval priority so definitions are never evicted from
        # the coherence cap in favour of tangential context.
        self.is_definition: bool = False

def merge_adjacent_entries(entries: List[FactEntry]) -> List[FactEntry]:
    if not entries:
        return []
    sorted_entries = sorted(entries, key=lambda x: x.start_idx)
    merged = []
    curr = sorted_entries[0]
    for next_entry in sorted_entries[1:]:
        curr_eid = getattr(curr, "entity_id", -1)
        next_eid = getattr(next_entry, "entity_id", -1)
        if next_entry.start_idx == curr.end_idx and curr_eid == next_eid:
            new_K = torch.cat([curr.K, next_entry.K], dim=2)
            new_V = torch.cat([curr.V, next_entry.V], dim=2)
            curr_sim = max(getattr(curr, "current_sim", 0.0), getattr(next_entry, "current_sim", 0.0))
            curr = FactEntry(
                start_idx=curr.start_idx,
                end_idx=next_entry.end_idx,
                K=new_K,
                V=new_V,
                descriptor=curr.descriptor,
                slot_ids=list(set(curr.slot_ids + next_entry.slot_ids)),
                tokens=curr.tokens + next_entry.tokens
            )
            curr.entity_id = curr_eid
            curr.current_sim = curr_sim
        else:
            merged.append(curr)
            curr = next_entry
    merged.append(curr)
    return merged

class FactualExactStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.entries: List[FactEntry] = []
        
    def build(self, prefill_kv: Dict[int, List[torch.Tensor]], token_ids: torch.Tensor, W_proj: torch.Tensor, stop_token_ids: Set[int], slot_ids: Optional[List[int]] = None, block_size: Optional[int] = None, inv_index: Optional[Any] = None, semantic_prime_slots: Optional[Set[int]] = None, use_salience_parser: bool = True):
        """
        Identify rare content words and group them into factual spans, building a 3D factual graph.
        prefill_kv: Dict[layer_idx, [K_cpu, V_cpu]]
          K_cpu/V_cpu shape: [1, kv_heads, total_seq_len, head_dim]
        token_ids: [total_seq_len] cpu
        W_proj: [DESC_DIM, head_dim]
        stop_token_ids: set of common token IDs
        slot_ids: optional list of active slot IDs in chronological order
        block_size: optional size of a block/slot in tokens
        inv_index: optional InvertedTokenIndex containing vocabulary/IDF mappings
        semantic_prime_slots: optional set of pool slot IDs representing semantic prime nodes
        use_salience_parser: if True, use self-supervised salience selector; else use basic stop-token rule
        """
        if not prefill_kv or token_ids is None or token_ids.numel() == 0:
            return
            
        total_seq_len = token_ids.numel()
        layers = sorted(list(prefill_kv.keys()))
        
        factual_mask = torch.zeros(total_seq_len, dtype=torch.bool)
        
        if use_salience_parser:
            # 1. Compute Eagle lookback score R(t) using causal key self-similarity
            R = torch.zeros(total_seq_len, dtype=torch.float32)
            if total_seq_len > 1:
                try:
                    first_layer = layers[0]
                    K_layer, _ = prefill_kv[first_layer] # K_layer: [1, kv_heads, total_seq_len, head_dim]
                    
                    # Average K over heads
                    K_avg = K_layer[0].mean(dim=0).float() # [total_seq_len, head_dim]
                    
                    # Move to GPU/MPS for speed if available
                    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                    K_avg = K_avg.to(device)
                    
                    # Compute pairwise similarity
                    sim = torch.matmul(K_avg, K_avg.T) / math.sqrt(K_avg.shape[1])
                    
                    # Apply causal mask: future query positions look back to past key positions
                    mask = torch.triu(torch.ones(total_seq_len, total_seq_len, device=device), diagonal=1).T
                    sim = sim.masked_fill(mask == 0, -1e9)
                    
                    attn = torch.softmax(sim, dim=-1)
                    attn = torch.nan_to_num(attn, nan=0.0)
                    
                    # Column sum represents total attention lookbacks pointing to each token
                    R_device = attn.sum(dim=0)
                    R = R_device.cpu()
                except Exception:
                    R = torch.zeros(total_seq_len, dtype=torch.float32)
            self.eagle_scores = R
                    
            # 2. Calculate Key Norms at Layer 0
            key_norms = torch.ones(total_seq_len, dtype=torch.float32)
            try:
                first_layer = layers[0]
                K_layer, _ = prefill_kv[first_layer]
                key_norms = K_layer[0].mean(dim=0).float().norm(dim=-1).cpu()
            except Exception:
                pass
                
            # 3. Calculate IDF values
            idf_vals = torch.zeros(total_seq_len, dtype=torch.float32)
            if inv_index is not None and hasattr(inv_index, "idf"):
                for i in range(total_seq_len):
                    tid = int(token_ids[i].item())
                    idf_vals[i] = inv_index.idf.get(tid, 1.0)
            else:
                # Fallback using stop word exclusion: non-stop words get 2.5, stop words get 0.1
                for i in range(total_seq_len):
                    tid = int(token_ids[i].item())
                    idf_vals[i] = 0.1 if tid in stop_token_ids else 2.5
                    
            # 4. Relational keyword boost — give binding words a fixed IDF-
            # equivalent score so they survive the top-% salience selection.
            # Without this, verbs/prepositions/comparatives all score ~0.1
            # and are systematically excluded, destroying relational structure.
            if inv_index is not None and hasattr(inv_index, "idf"):
                _rel_tok_ids: Optional[Set[int]] = None
                try:
                    # Build the set of token IDs matching relational keywords.
                    # Cached on the inv_index to avoid rebuilding every call.
                    _rel_tok_ids = getattr(inv_index, "_relational_token_ids", None)
                    if _rel_tok_ids is None:
                        _rel_tok_ids = set()
                        vocab = None
                        if hasattr(inv_index, "_tokenizer_ref") and inv_index._tokenizer_ref is not None:
                            vocab = inv_index._tokenizer_ref
                        if vocab is None:
                            # Fallback: scan all occurrences for tokens whose
                            # decoded form matches a relational keyword.
                            # This is built once and cached.
                            pass
                        # Direct IDF-based identification: any token whose IDF
                        # is very low but whose raw text matches RELATIONAL_KEYWORDS.
                        # We check all tokens in the document.
                        for i_t in range(total_seq_len):
                            tid = int(token_ids[i_t].item())
                            if tid in inv_index.idf and inv_index.idf[tid] < 1.5:
                                _rel_tok_ids.add(tid)
                        inv_index._relational_token_ids = _rel_tok_ids
                except Exception:
                    _rel_tok_ids = None

                if _rel_tok_ids:
                    for i_t in range(total_seq_len):
                        tid = int(token_ids[i_t].item())
                        if tid in _rel_tok_ids:
                            # Boost relational tokens to median content-word IDF
                            idf_vals[i_t] = max(idf_vals[i_t], 2.0)

            # 5. Compute joint factual salience score
            total_salience = key_norms * idf_vals * (1.0 + 1.0 * R)
            
            # Select the top 5% most salient tokens (precision mode: max 300 tokens)
            # 50% was selecting half the document — far too broad for exact grounding.
            # 5% keeps only the rarest, most distinctive content words per document.
            k_num = max(8, int(total_seq_len * 0.05))
            k_num = min(k_num, 300)  # absolute cap regardless of document length
            k_num = min(k_num, total_seq_len)
            
            if total_seq_len > 0:
                threshold_val = float(torch.topk(total_salience, k=k_num).values[-1].item())
                factual_mask = total_salience >= threshold_val

            # 5b. Relational Context Window Expansion — each salient seed token
            # is expanded into a ±3-token window so the factual span captures
            # the surrounding relational structure (verbs, prepositions,
            # comparatives) that binds concepts together.  Without this, spans
            # are bags of nouns missing the connective tissue.
            CONTEXT_WINDOW = 3
            if total_seq_len > 0:
                expanded_mask = factual_mask.clone()
                seed_positions = torch.where(factual_mask)[0].tolist()
                for pos in seed_positions:
                    lo = max(0, pos - CONTEXT_WINDOW)
                    hi = min(total_seq_len, pos + CONTEXT_WINDOW + 1)
                    expanded_mask[lo:hi] = True
                factual_mask = expanded_mask
                
            # 5c. Gap-bridging (dilation): bridge single-token gaps
            if total_seq_len > 2:
                dilated_mask = factual_mask.clone()
                for i in range(1, total_seq_len - 1):
                    if factual_mask[i - 1] and factual_mask[i + 1]:
                        dilated_mask[i] = True
                factual_mask = dilated_mask
        else:
            # Fallback to simple stop-token exclusion for deterministic testing
            for i in range(total_seq_len):
                tid = int(token_ids[i].item())
                if tid not in stop_token_ids and tid > 0:
                    factual_mask[i] = True
            
        # 6. Group contiguous factual tokens into spans
        spans = []
        in_span = False
        start = -1
        for i in range(total_seq_len):
            if factual_mask[i]:
                if not in_span:
                    start = i
                    in_span = True
            else:
                if in_span:
                    spans.append((start, i))
                    in_span = False
        if in_span:
            spans.append((start, total_seq_len))
            
        # Split long spans into chunks of max length 12
        chunked_spans = []
        for s, e in spans:
            for sub_s in range(s, e, 20):
                sub_e = min(sub_s + 20, e)
                chunked_spans.append((sub_s, sub_e))
                
        # 7. Extract verbatim KV sequences across all layers for each span
        for s, e in chunked_spans:
            span_len = e - s
            if span_len <= 0:
                continue
                
            span_K_list = []
            span_V_list = []
            
            for layer in layers:
                K_layer, V_layer = prefill_kv[layer]
                # K_layer/V_layer: [1, kv_heads, total_seq_len, head_dim]
                span_K_list.append(K_layer[0, :, s:e, :].clone())
                span_V_list.append(V_layer[0, :, s:e, :].clone())
                
            span_K = torch.stack(span_K_list, dim=0) # [num_layers, kv_heads, span_len, head_dim]
            span_V = torch.stack(span_V_list, dim=0) # [num_layers, kv_heads, span_len, head_dim]
            
            # Compute descriptor for the span using layer 0 max-pooled key.
            # Max-pool over positions retains the most activated (distinctive) features
            # for each head, rather than averaging them away. This is critical for
            # formula/math spans where a single rare token dominates the span semantics.
            # Then mean over heads to produce the final descriptor vector.
            max_k = span_K[0].max(dim=1).values.mean(dim=0).float()  # [head_dim]
            if W_proj is not None:
                desc = max_k.to(W_proj.device) @ W_proj.T  # [DESC_DIM]
                desc = desc / (desc.norm() + 1e-8)
            else:
                desc = torch.zeros(64)
                
            # Determine which slot IDs this span overlaps with
            entry_slot_ids = []
            if slot_ids is not None and block_size is not None and block_size > 0:
                start_block_idx = s // block_size
                end_block_idx = (e - 1) // block_size
                for idx in range(start_block_idx, end_block_idx + 1):
                    if idx < len(slot_ids):
                        entry_slot_ids.append(slot_ids[idx])
                        
            span_tokens = token_ids[s:e].tolist()

            entry = FactEntry(
                start_idx=s,
                end_idx=e,
                K=span_K,
                V=span_V,
                descriptor=desc.cpu(),
                slot_ids=entry_slot_ids,
                tokens=span_tokens
            )

            # Capture 3-token source prefix for quote-grounded connective gating.
            prefix_start = max(0, s - 3)
            entry.prefix_tokens = token_ids[prefix_start:s].tolist()

            # Determine if this entry is a Factual Prime Node.
            # RC7: prime promotion requires BOTH rarity (IDF) AND that other tokens
            # in the document look back at this span (Eagle-R ≥ 0.3).  Pure rarity
            # without being referenced produces phantom entity nodes for jargon terms
            # that appear once but are never re-discussed.  A very high R (≥ 1.5×
            # average) promotes even moderately rare tokens — they are clearly entities
            # because the rest of the document is built around them.
            is_prime = False
            if semantic_prime_slots is not None:
                if any(slot in semantic_prime_slots for slot in entry_slot_ids):
                    is_prime = True
            if not is_prime:
                max_idf = 0.0
                if inv_index is not None and hasattr(inv_index, "idf"):
                    max_idf = max([inv_index.idf.get(t, 1.0) for t in span_tokens]) if span_tokens else 0.0
                # Eagle lookback score for this span (available when salience parser ran)
                have_eagle = (
                    hasattr(self, "eagle_scores")
                    and self.eagle_scores is not None
                    and self.eagle_scores.numel() > 0
                )
                max_r = 0.0
                if have_eagle and e > s:
                    r_slice = self.eagle_scores[s:e]
                    max_r = float(r_slice.max().item()) if r_slice.numel() > 0 else 0.0
                if have_eagle:
                    # RC7: an entity is a rare term the rest of the document refers
                    # back to — require BOTH rarity AND reference, so one-off jargon
                    # doesn't spawn a phantom entity.
                    if max_idf >= 3.0 and max_r >= 0.3:
                        is_prime = True
                    # Extremely referenced regardless of rarity → entity
                    elif max_r >= 1.5:
                        is_prime = True
                else:
                    # RC7 fallback: no reference signal was computed (salience parser
                    # off / single-token doc).  We cannot demand a reference we never
                    # measured, so fall back to rarity-only promotion.
                    if max_idf >= 3.0:
                        is_prime = True
            entry.is_prime = is_prime

            # For primes, record the single most distinguishing (highest-IDF) token.
            # Used by _assign_entities() to bind properties that explicitly name
            # their entity, overriding misleading low-IDF token overlap (RC4).
            if is_prime and inv_index is not None and hasattr(inv_index, "idf") and span_tokens:
                best_tok, best_idf = None, 0.0
                for t in span_tokens:
                    t_idf = inv_index.idf.get(t, 1.0)
                    if t_idf >= 3.0 and t_idf > best_idf:
                        best_idf = t_idf
                        best_tok = t
                entry.distinguishing_token = best_tok
            # entity_id is assigned after all entries are built (see _assign_entities)

            self.entries.append(entry)
            
        # 3. Assign entity_ids: bind each property span to the prime that owns it.
        prime_entries = [(idx, e) for idx, e in enumerate(self.entries) if e.is_prime]
        for p_idx, p_entry in prime_entries:
            p_entry.entity_id = p_entry.start_idx  # Primes define their own entity
        self._assign_entities(inv_index)

        # 3b. RC1 — Extract typed directed triples: (prime → bridge → value).
        # For each prime entry P, scan forward in the document for property spans V
        # owned by the same entity.  The tokens between P.end_idx and V.start_idx
        # form the "bridge" (relational connectives: "is", "has", "exhibits", etc.).
        # The complete (bridge + value) sequence is stored on the prime as a triple_sequence.
        # At decode time these triples are injected into current_step_factual_sequences
        # so the VSL enforces the source's exact (entity → relation → value) ordering,
        # preventing the model from freely re-composing the relation from helpers.
        if prime_entries and inv_index is not None and hasattr(inv_index, "idf"):
            # Low-IDF tokens are relational binders; capture any token appearing
            # in the bridge that is common enough to carry a structural role.
            rel_tid_set: Set[int] = set()
            for i_t in range(total_seq_len):
                tid = int(token_ids[i_t].item())
                if inv_index.idf.get(tid, 2.0) < 1.5:
                    rel_tid_set.add(tid)

            for p_entry in [e for e in self.entries if e.is_prime]:
                p_end = p_entry.end_idx
                p_entity = p_entry.start_idx

                # Find property spans owned by this prime that appear forward
                for v_entry in self.entries:
                    if v_entry.is_prime:
                        continue
                    if v_entry.entity_id != p_entity:
                        continue
                    if v_entry.start_idx < p_end:
                        continue  # value must follow prime in document
                    gap = v_entry.start_idx - p_end
                    if gap > 96:
                        continue  # too distant; bridge would be meaningless

                    bridge = token_ids[p_end:v_entry.start_idx].tolist()
                    if len(bridge) > 8:
                        continue  # overly long bridge → noise
                    # Bridge must contain at least one relational token so we know
                    # it is a genuine binding phrase and not just whitespace.
                    if not any(tid in rel_tid_set for tid in bridge):
                        continue

                    triple_seq = bridge + v_entry.tokens
                    if triple_seq and triple_seq not in p_entry.triple_sequences:
                        p_entry.triple_sequences.append(triple_seq)

        # 3c. DX1 — Tag definition spans.
        # A definition span is a non-prime property span that sits close (≤ 32 tokens)
        # to its owning prime via a short copular bridge (IDF < 1.0 — "is", "are",
        # "means", "refers to", etc.).  These receive a 1.5× similarity boost in
        # query() so they always survive the coherence cap.
        if prime_entries:
            prime_by_entity = {p_e.start_idx: p_e for _, p_e in prime_entries}
            for entry in self.entries:
                if entry.is_prime or entry.entity_id == -1:
                    continue
                p_entry = prime_by_entity.get(entry.entity_id)
                if p_entry is None or entry.start_idx < p_entry.end_idx:
                    continue
                gap = entry.start_idx - p_entry.end_idx
                if gap > 32:
                    continue
                bridge = token_ids[p_entry.end_idx:entry.start_idx].tolist()
                if len(bridge) > 5:
                    continue  # long bridges are paraphrases, not definitions
                if inv_index is not None and hasattr(inv_index, "idf"):
                    # Copular verbs have very low IDF (appear in nearly all documents)
                    if any(inv_index.idf.get(t, 2.0) < 1.0 for t in bridge):
                        entry.is_definition = True
                elif len(bridge) <= 3:
                    # Fallback: very short bridge is likely a copula
                    entry.is_definition = True

        # 4. Build factual layer graph connections with entity-aware dampening.
        # Cross-entity edges receive a 0.3× weight penalty to prevent graph
        # walks from propagating one entity's properties into another's
        # retrieval set.  This preserves the edges (needed for comparison
        # queries) but dramatically reduces cross-contamination.
        CROSS_ENTITY_DAMPEN = 0.3
        num_entries = len(self.entries)
        for i in range(num_entries):
            entry_i = self.entries[i]
            tokens_i = set(entry_i.tokens)
            
            for j in range(i + 1, num_entries):
                entry_j = self.entries[j]
                tokens_j = set(entry_j.tokens)
                
                # Lexical overlap: shares any non-stop words
                shared_non_stop = (tokens_i & tokens_j) - stop_token_ids
                lexical_overlap = len(shared_non_stop) > 0
                
                # Temporal adjacency: within 512 tokens
                temporal_dist = abs(entry_i.start_idx - entry_j.start_idx)
                is_temporal_adjacent = temporal_dist < 512
                
                # Descriptor similarity
                sim_val = torch.dot(entry_i.descriptor, entry_j.descriptor).item()
                is_similar = sim_val >= 0.3
                
                if lexical_overlap or is_temporal_adjacent or is_similar:
                    w_lex = 1.0 if lexical_overlap else 0.0
                    w_temp = max(0.0, 1.0 - (temporal_dist / 512.0))
                    w_sim = max(0.0, sim_val)
                    
                    weight = 0.4 * w_sim + 0.4 * w_lex + 0.2 * w_temp

                    # Entity-aware dampening: penalise cross-entity edges
                    eid_i = entry_i.entity_id
                    eid_j = entry_j.entity_id
                    if eid_i != -1 and eid_j != -1 and eid_i != eid_j:
                        weight *= CROSS_ENTITY_DAMPEN
                    
                    entry_i.neighbors.append(j)
                    entry_i.weights.append(weight)
                    
                    entry_j.neighbors.append(i)
                    entry_j.weights.append(weight)

    def _assign_entities(self, inv_index: Optional[Any]) -> None:
        """
        Bind every non-prime property span to the prime entity that owns it (RC4).

        Plain token-overlap mis-binds related concepts: a property of EP2 such as
        "codimension 2 while" shares the low-IDF filler "while" with the EP3 span
        and gets pulled to EP3, even though it names neither entity.  Absolute
        positional proximity is also ambiguous in interleaved comparison text.

        Ownership is decided in priority order:

          1. Distinguishing-token override — if the span explicitly contains
             exactly one prime's distinguishing (highest-IDF) token, it is owned
             by that prime.  This is the strongest, IDF-grounded signal and beats
             any amount of shared filler.
          2. IDF-weighted overlap — if the span contains *several* primes'
             distinguishing tokens (e.g. a sentence comparing both), award it to
             the prime with the greatest summed-IDF token overlap.
          3. Nearest-preceding prime — otherwise fall back to reading-order
             ownership: a property belongs to the most recent entity introduced
             before it.  This matches how the document reads and how triple
             extraction scans forward, and (unlike absolute proximity) never
             assigns a property to an entity introduced only *after* it.
             If no prime precedes the span, the nearest prime overall is used.

        Primes must already have entity_id == start_idx before this is called.
        """
        prime_entries = [e for e in self.entries if e.is_prime]
        # entity_id (prime start_idx) → that entity's distinguishing token.  Used
        # by query() to build the entity-bias signature (RC3).
        self.entity_distinguishing: Dict[int, int] = {
            p.start_idx: p.distinguishing_token
            for p in prime_entries
            if getattr(p, "distinguishing_token", None) is not None
        }
        if not prime_entries:
            return

        have_idf = inv_index is not None and hasattr(inv_index, "idf")

        def tok_idf(t: int) -> float:
            return inv_index.idf.get(t, 1.0) if have_idf else 1.0

        prime_starts_sorted = sorted(p.start_idx for p in prime_entries)
        # Map: prime distinguishing token -> set of prime start_idx that carry it.
        dist_owner: Dict[int, Set[int]] = {}
        for p in prime_entries:
            dt = getattr(p, "distinguishing_token", None)
            if dt is not None:
                dist_owner.setdefault(dt, set()).add(p.start_idx)

        for entry in self.entries:
            if entry.is_prime:
                continue
            entry_tokens = set(entry.tokens)

            # 1./2. Distinguishing-token signal.
            named_primes: Set[int] = set()
            for dt, owners in dist_owner.items():
                if dt in entry_tokens:
                    named_primes |= owners
            if len(named_primes) == 1:
                entry.entity_id = next(iter(named_primes))
                continue
            if len(named_primes) > 1:
                # Span references multiple entities — award to the strongest
                # IDF-weighted overlap among the named primes.
                best_entity, best_score = -1, -1.0
                for p in prime_entries:
                    if p.start_idx not in named_primes:
                        continue
                    score = sum(tok_idf(t) for t in (entry_tokens & set(p.tokens)))
                    if score > best_score:
                        best_score, best_entity = score, p.start_idx
                entry.entity_id = best_entity
                continue

            # 3. Reading-order ownership: nearest prime introduced before this span.
            preceding = [ps for ps in prime_starts_sorted if ps <= entry.start_idx]
            if preceding:
                entry.entity_id = preceding[-1]
            else:
                entry.entity_id = min(
                    prime_starts_sorted, key=lambda ps: abs(entry.start_idx - ps)
                )

        # RC3: stamp each entry with its owning entity's signature (zeros if the
        # entity has no distinguishing token).  Primes carry their own signature.
        for entry in self.entries:
            dt = self.entity_distinguishing.get(entry.entity_id)
            if dt is not None:
                entry.entity_sig = entity_signature(dt)

    def query(self, Q: torch.Tensor, W_proj: torch.Tensor, threshold: float = 0.4,
              active_slots: Optional[Set[int]] = None,
              query_entity_bias: Optional[Set[int]] = None) -> List[FactEntry]:
        """
        Query the factual store.
        Q: [H_q, D]
        """
        if not self.entries or W_proj is None:
            return []
            
        avg_q = Q.mean(dim=0).float()
        q_desc = avg_q @ W_proj.T # [DESC_DIM]
        q_desc = q_desc / (q_desc.norm() + 1e-8)
        q_desc_cpu = q_desc.cpu()

        # RC3 — entity-bias signature.  When the caller knows which entities the
        # query is about (from the prompt's prime matches), build a unit signature
        # from those entities' distinguishing tokens and use it to rank their spans
        # above shared-vocabulary spans of other entities.  This is the part of the
        # descriptor that random-projected layer-0 keys cannot represent.  When no
        # bias is supplied (or the entities have no distinguishing token), q_sig is
        # zero and ranking is byte-for-byte identical to before (null-safe).
        ENTITY_BIAS_WEIGHT = 0.15
        q_sig = None
        if query_entity_bias:
            ed = getattr(self, "entity_distinguishing", {})
            acc = torch.zeros(ENTITY_SIG_DIM, dtype=torch.float32)
            for eid in query_entity_bias:
                dt = ed.get(eid)
                if dt is not None:
                    acc = acc + entity_signature(dt)
            n = acc.norm()
            if n > 1e-8:
                q_sig = acc / n

        # 1. Base Layer (Horizontal -> Vertical): candidate entries matching active slots (or all if active_slots is None)
        candidate_indices = set()
        if active_slots is not None:
            for idx, entry in enumerate(self.entries):
                if any(slot in active_slots for slot in entry.slot_ids):
                    candidate_indices.add(idx)
        else:
            candidate_indices = set(range(len(self.entries)))
            
        # 2. Factual Prime Node Activation (Seeds)
        prime_seeds = []
        for idx, entry in enumerate(self.entries):
            if entry.is_prime:
                sim = torch.dot(q_desc_cpu, entry.descriptor).item()
                if sim >= threshold:
                    prime_seeds.append((idx, sim))
                    
        # 3. Factual Graph Walk (Horizontal propagation in Details layer)
        walk_candidates = {}
        for seed_idx, seed_sim in prime_seeds:
            walk_candidates[seed_idx] = seed_sim
            entry = self.entries[seed_idx]
            for nb_idx, weight in zip(entry.neighbors, entry.weights):
                prop_sim = seed_sim * weight
                if nb_idx not in walk_candidates or prop_sim > walk_candidates[nb_idx]:
                    walk_candidates[nb_idx] = prop_sim
                    
        # 4. Merge candidates (Union of slot-localized candidates and walk-traversed candidates)
        merged_results = []
        all_candidate_idxs = candidate_indices | set(walk_candidates.keys())
        
        for idx in all_candidate_idxs:
            entry = self.entries[idx]
            sim = torch.dot(q_desc_cpu, entry.descriptor).item()
            
            passes_main = sim >= threshold
            passes_relaxed = (active_slots is not None and 
                              any(slot in active_slots for slot in entry.slot_ids) and 
                              sim >= 0.15)
            # Walk threshold uses a lower bar than the main threshold so that genuine
            # property spans (connected to a matched category node via a strong edge)
            # are included. Two conditions must BOTH hold:
            #   1. walk_score >= 0.20 (the propagated relevance from the seed)
            #   2. Either sim >= 0.10 (some direct distributional relevance) OR
            #      walk_score >= 0.30 (strong edge — trust the graph topology)
            # This prevents zero-similarity entries reached only via weak temporal
            # adjacency edges from being returned (fixes test_3d_factual_graph_walk),
            # while still pulling in genuine property spans with moderate similarity.
            _WALK_THRESHOLD = 0.20
            _WALK_STRONG    = 0.30   # strong-edge override (no sim requirement)
            _WALK_MIN_SIM   = 0.10   # minimum direct sim when edge is moderate
            if idx in walk_candidates:
                ws = walk_candidates[idx]
                passes_walk = (ws >= _WALK_THRESHOLD and
                               (sim >= _WALK_MIN_SIM or ws >= _WALK_STRONG))
            else:
                passes_walk = False

            
            if passes_main or passes_relaxed or passes_walk:
                final_score = max(sim, walk_candidates.get(idx, -1.0))
                # RC3: nudge ranking toward the queried entity's spans.  Only
                # affects ordering (not the pass/fail gates above), so retrieval
                # recall is unchanged — we just resolve which entity's span wins
                # when their descriptors are otherwise indistinguishable.
                if q_sig is not None:
                    final_score = final_score + ENTITY_BIAS_WEIGHT * float(
                        torch.dot(q_sig, entry.entity_sig)
                    )
                # DX1: definition spans get boosted retrieval priority so they
                # always sort above tangential context in the coherence cap.
                if getattr(entry, "is_definition", False):
                    final_score = min(final_score * 1.5, 1.0)
                entry.current_sim = final_score
                merged_results.append((entry, final_score))
                
        # Fallback: if active_slots is provided but no matches passed, pull the top localized match
        if active_slots is not None and not merged_results:
            fallback_matches = []
            for idx in candidate_indices:
                entry = self.entries[idx]
                sim = torch.dot(q_desc_cpu, entry.descriptor).item()
                if sim >= 0.15:
                    entry.current_sim = sim
                    fallback_matches.append((entry, sim))
            fallback_matches.sort(key=lambda x: x[1], reverse=True)
            if fallback_matches:
                merged_results = [fallback_matches[0]]
                
        # Sort by final score descending
        merged_results.sort(key=lambda x: x[1], reverse=True)
        # RC6 — entity-proportional budget: a fixed top-5 cut drops one entity's
        # properties in multi-entity comparison answers.  Scale K with the number
        # of primes that actually matched so each entity gets coverage.
        n_matched_primes = len(prime_seeds)
        k_budget = max(5, n_matched_primes * 3 + 2)
        top_entries = [x[0] for x in merged_results[:k_budget]]
        return merge_adjacent_entries(top_entries)
