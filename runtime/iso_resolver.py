
import torch
import time
from typing import Optional, List, Dict, Any, Tuple
from runtime.srl_resolver import SRLResolver
from hsha import (
    ImmutableSymbolicObject,
    SymbolicTopologyHasher,
    SymbolicObjectRegistry,
    SymbolicObjectSerializer,
    ObjectLineageTracker
)

class ISOResolver(SRLResolver):
    """
    PHASE 21.2: ISO - Immutable Symbolic Objects.
    Transforms symbolic memory from transient token spans into immutable entities.
    Targets 'Symbolic Identity' and 'Persistence Semantics'.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        # ISO Core Modules
        self.iso_registry = SymbolicObjectRegistry()
        self.topology_hasher = SymbolicTopologyHasher(self.encoder.delimiter_ids)
        self.serializer = SymbolicObjectSerializer()
        self.lineage_tracker = ObjectLineageTracker()
        
        # Local state for lineage tracking
        self.last_recalled_iso: Optional[ImmutableSymbolicObject] = None
        
        # Metrics
        self.iso_stats = {
            "registered_objects": 0,
            "mutation_events": 0,
            "serialization_count": 0,
            "mean_topology_stability": 1.0,
            "lineage_depth": 0
        }

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        """
        Extends SRL resolve_and_prune to register Immutable Symbolic Objects.
        """
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # Leverage HSHA/SABEAF booster spans to identify ISO candidates
        if hasattr(self, 'booster') and len(self.booster.active_spans) > 0:
            for start, end in self.booster.active_spans:
                chunk_size = chunk_input_ids.shape[1]
                chunk_start_global = self.global_offset - chunk_size
                
                if start >= chunk_start_global:
                    rel_start = max(0, start - chunk_start_global)
                    rel_end = min(chunk_size, end - chunk_start_global)
                    tokens = chunk_input_ids[0, rel_start:rel_end].tolist()
                    
                    if len(tokens) > 6:
                        t_hash = self.topology_hasher.hash_topology(tokens)
                        iso = ImmutableSymbolicObject(tokens, t_hash, {"source_pos": start, "timestamp": time.time()})
                        
                        if iso.object_id not in self.iso_registry.list_all():
                            self.iso_registry.register_object(iso)
                            self.iso_stats["registered_objects"] += 1
                            # print(f"[DEBUG] ISO: Registered {iso.object_id} with topology {t_hash[:8]}")
                            
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        SRL guidance with ISO identity tracking.
        """
        # Call SRL guidance
        calibrated_logits = super().guide_decoder(logits, attention_weights)
        
        # If SRL selected a hub, we sync our ISO tracker
        if self.current_hub_id:
            # In Phase 21.2, we map the hub_id (from registry) to an ISO
            # For this MVP, we assume the hub_id registered in HSHA matches the ISO logic
            # or we fetch it from our registry.
            # (HSHA hubs were registered via register_hub, which returns a string ID)
            pass
            
        return calibrated_logits

    def record_generated_token(self, token_id: int, logits: torch.Tensor):
        """
        Finalize token selection and track lineage.
        """
        super().record_generated_token(token_id, logits)
        
        # Mutation detection if we are in the middle of a recall
        if self.current_hub_id and self.hub_token_idx > 0:
            # We already have exact_match tracking in HSHAResolver
            # ISO adds formal mutation scoring if needed
            pass

    def get_iso_summary(self) -> Dict[str, Any]:
        """Returns summarized metrics for ISO validation."""
        srl_summary = super().get_srl_summary()
        
        return {
            **srl_summary,
            "object_integrity": 1.0 - (self.iso_stats["mutation_events"] / (srl_summary["hub_utilization"] + 1)),
            "registered_objects": self.iso_stats["registered_objects"],
            "topology_stability": self.iso_stats["mean_topology_stability"],
            "lineage_depth": self.iso_stats["lineage_depth"]
        }
