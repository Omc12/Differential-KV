import torch
import math
from typing import List, Dict, Optional, Set, Any

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

def merge_adjacent_entries(entries: List[FactEntry]) -> List[FactEntry]:
    if not entries:
        return []
    sorted_entries = sorted(entries, key=lambda x: x.start_idx)
    merged = []
    curr = sorted_entries[0]
    for next_entry in sorted_entries[1:]:
        if next_entry.start_idx == curr.end_idx:
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
                    
            # 4. Compute joint factual salience score
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

                
            # 5. Gap-bridging (dilation): bridge single-token gaps
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
            
            # Determine if this entry is a Factual Prime Node
            is_prime = False
            if semantic_prime_slots is not None:
                if any(slot in semantic_prime_slots for slot in entry_slot_ids):
                    is_prime = True
            if not is_prime and inv_index is not None and hasattr(inv_index, "idf"):
                max_idf = max([inv_index.idf.get(t, 1.0) for t in span_tokens]) if span_tokens else 0.0
                if max_idf >= 3.0:
                    is_prime = True
            entry.is_prime = is_prime
            
            self.entries.append(entry)
            
        # 3. Build factual layer graph connections
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
                    
                    entry_i.neighbors.append(j)
                    entry_i.weights.append(weight)
                    
                    entry_j.neighbors.append(i)
                    entry_j.weights.append(weight)
            
    def query(self, Q: torch.Tensor, W_proj: torch.Tensor, threshold: float = 0.4, active_slots: Optional[Set[int]] = None) -> List[FactEntry]:
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
        # Top 5 only — with 5% selection, each entry is a tight, high-precision span.
        # Returning more than 5 at this precision level floods the VSL with too many candidates.
        top_entries = [x[0] for x in merged_results[:5]]
        return merge_adjacent_entries(top_entries)
