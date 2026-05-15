
import torch
import time
from typing import Optional, List, Dict, Any
from runtime.sabeaf_resolver import SABEAFResolver
from hsha import (
    SymbolicHubRegistry, 
    SymbolicObjectEncoder, 
    ContextualRecallRouter, 
    HubRecallInjector, 
    TopologyIntegrityMap
)

class HSHAResolver(SABEAFResolver):
    """
    PHASE 21.0: HSHA (Hybrid Symbolic Hub Architecture).
    Implements external symbolic persistence hubs for exact symbolic survival.
    Targets "Symbolic Dilution" beyond transformer-native attention limits.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        # HSHA Core Modules
        self.hub_registry = SymbolicHubRegistry()
        self.encoder = SymbolicObjectEncoder(tokenizer)
        self.recall_router = ContextualRecallRouter(self.hub_registry)
        self.injector = HubRecallInjector(tokenizer)
        self.integrity_map = TopologyIntegrityMap(self.encoder.delimiter_ids)
        
        # HSHA Runtime State
        self.active_hubs: List[str] = []
        self.current_hub_id: Optional[str] = None
        self.hub_token_idx = 0
        self.context_tokens = []
        self.last_recall_time = 0
        
        # Metrics & Telemetry
        self.hsha_metrics = {
            "hub_utilization": 0,
            "recall_precision": 0.0,
            "false_recall_rate": 0.0,
            "exact_match_count": 0,
            "total_hub_tokens": 0,
            "drift_risk_history": []
        }

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        """
        Standard resolve/prune + HSHA Hub Registration & Lineage Tracking.
        """
        # 1. Base SABEAF logic (Booster population)
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # 2. Extract and Register Symbolic Objects (21.0)
        # Leverage SABEAF's booster spans to identify symbolic candidates
        if hasattr(self, 'booster') and len(self.booster.active_spans) > 0:
            for start, end in self.booster.active_spans:
                # Map position to current chunk if applicable
                chunk_size = chunk_input_ids.shape[1]
                chunk_start_global = self.global_offset - chunk_size
                
                if start >= chunk_start_global:
                    rel_start = max(0, start - chunk_start_global)
                    rel_end = min(chunk_size, end - chunk_start_global)
                    hub_tokens = chunk_input_ids[0, rel_start:rel_end].tolist()
                    
                    if len(hub_tokens) > 6: # Heuristic: meaningful symbolic object length
                        topology = self.encoder.extract_topology(hub_tokens)
                        hub_id = self.hub_registry.register_hub(hub_tokens, topology)
                        if hub_id not in self.active_hubs:
                            print(f"[DEBUG] HSHA: Registered hub {hub_id} (len {len(hub_tokens)}) at global start {start}")
                            self.active_hubs.append(hub_id)
        
        # Maintain rolling context for query matching
        self.context_tokens.extend(chunk_input_ids[0].tolist())
        if len(self.context_tokens) > 1024: # Context window for recall routing
            self.context_tokens = self.context_tokens[-1024:]
            
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        PHASE 21.0: HSHA - Hybrid Recall Routing & Exact Symbolic Reinjection.
        """
        # 1. Base SABEAF Logic (Energy Focusing + Anchor Boosting)
        calibrated_logits = super().guide_decoder(logits, attention_weights)
        
        # 2. HSHA Contextual Recall Routing
        recall_scores = self.recall_router.route_recall(self.context_tokens, self.active_hubs)
        
        # 3. Hub Selection & Symbolic Reinjection
        best_hub_id = None
        max_score = 0.0
        for h_id, score in recall_scores.items():
            if score > max_score:
                max_score = score
                best_hub_id = h_id
        
        if best_hub_id and max_score > self.recall_router.recall_threshold:
            self.current_hub_id = best_hub_id
            hub_obj = self.hub_registry.get_object(best_hub_id)
            
            # Synchronize position in hub if starting recall
            if self.hub_token_idx == 0:
                # Find matching prefix in recent context to align index
                match_found = False
                for i in range(len(hub_obj.tokens) - 2):
                    if self.context_tokens[-3:] == hub_obj.tokens[i:i+3]:
                        self.hub_token_idx = i + 3
                        match_found = True
                        break
                if not match_found:
                    self.hub_token_idx = 0 # Fallback or keep searching
            
            # Inject Recall (Entropy-Safe)
            if self.hub_token_idx < len(hub_obj.tokens):
                calibrated_logits = self.injector.inject_recall(
                    calibrated_logits, hub_obj.tokens, max_score, self.hub_token_idx
                )
                self.hsha_metrics["hub_utilization"] += 1
            
            # 4. Soft Topology Restoration
            drift_risk = self.integrity_map.get_drift_risk()
            self.hsha_metrics["drift_risk_history"].append(drift_risk)
            
            if drift_risk > 0.4:
                # Stronger reinforcement of structural boundaries
                calibrated_logits = self.injector.restore_topology(
                    calibrated_logits, list(self.encoder.delimiter_ids), drift_risk
                )

        return calibrated_logits

    def record_generated_token(self, token_id: int, logits: torch.Tensor):
        """
        Phase 21.0: Exact Match & Integrity Tracking.
        """
        # Update local context tracking
        self.context_tokens.append(token_id)
        if len(self.context_tokens) > 1024:
            self.context_tokens.pop(0)
            
        # Integrity Verification
        is_exact_match = False
        if self.current_hub_id:
            hub_obj = self.hub_registry.get_object(self.current_hub_id)
            if hub_obj and self.hub_token_idx < len(hub_obj.tokens):
                expected = hub_obj.tokens[self.hub_token_idx]
                if token_id == expected:
                    is_exact_match = True
                    self.hub_token_idx += 1
                    self.hsha_metrics["exact_match_count"] += 1
                else:
                    # Mismatch detected: hub continuity broken
                    if self.hub_token_idx > 0:
                        self.hsha_metrics["false_recall_rate"] += 0.1 # Penalty
                    self.current_hub_id = None
                    self.hub_token_idx = 0
            else:
                # Hub exhausted or missing
                self.current_hub_id = None
                self.hub_token_idx = 0
                
        # Update Topology Integrity Map
        self.integrity_map.record_token(token_id, is_exact_match)
        
        # Base logic (SPS/SPSLRIF momentum updates)
        super().record_generated_token(token_id, logits)

    def get_hsha_stats(self) -> Dict[str, Any]:
        """Returns summarized metrics for validation."""
        total_recalls = self.hsha_metrics["hub_utilization"]
        exact_matches = self.hsha_metrics["exact_match_count"]
        
        return {
            "hub_utilization": total_recalls,
            "exact_match_rate": exact_matches / total_recalls if total_recalls > 0 else 0,
            "mean_drift_risk": sum(self.hsha_metrics["drift_risk_history"]) / len(self.hsha_metrics["drift_risk_history"]) if self.hsha_metrics["drift_risk_history"] else 0,
            "registered_hubs": len(self.active_hubs),
            "false_recall_rate": self.hsha_metrics["false_recall_rate"]
        }
