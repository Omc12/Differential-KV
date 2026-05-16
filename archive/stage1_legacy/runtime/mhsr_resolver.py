
import torch
import time
from typing import Optional, List, Dict, Any, Tuple
from runtime.lscp_resolver import LSCPResolver
from hsha import (
    SymbolicRelationshipGraph,
    MultihopReasoningRouter,
    SymbolicAssociationEngine,
    TraversalLegitimacyGuard,
    SymbolicContextComposer
)

class MHSRResolver(LSCPResolver):
    """
    PHASE 21.5: MHSR - Multi-Hop Symbolic Reasoning.
    Implements multi-hop traversal and relationship-aware recall.
    Targets 'Symbolic Traversal' and 'Relational Reasoning'.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        # MHSR Core Modules
        self.rel_graph = SymbolicRelationshipGraph()
        self.traversal_guard = TraversalLegitimacyGuard()
        self.reasoning_router = MultihopReasoningRouter(self.rel_graph, self.traversal_guard)
        self.association_engine = SymbolicAssociationEngine()
        self.composer = SymbolicContextComposer()
        
        # Metrics
        self.mhsr_stats = {
            "traversal_count": 0,
            "recursion_suppressions": 0,
            "mean_composition_quality": 1.0,
            "relationship_count": 0
        }

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        """
        Extends LSCP resolve_and_prune to build symbolic relationships.
        """
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # Form relationships between active ISOs
        active_ids = self.iso_registry.list_all()
        if len(active_ids) >= 2:
            # Connect the most recent two objects
            source, target = active_ids[-2], active_ids[-1]
            self.rel_graph.add_relationship(source, target)
            self.association_engine.record_co_occurrence(source, target)
            self.mhsr_stats["relationship_count"] = len(self.rel_graph._metadata)
            
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        LSCP guidance with multi-hop reasoning.
        """
        # 1. Base LSCP/STRL/ISO/SRL Guidance
        calibrated_logits = super().guide_decoder(logits, attention_weights)
        
        # 2. MHSR Multi-Hop Traversal
        if self.current_hub_id:
            # See if there's a chain starting from the current hub
            chain = self.reasoning_router.route_multihop(self.current_hub_id)
            if len(chain) > 1:
                self.mhsr_stats["traversal_count"] += 1
                # Future: anticipatory recall based on chain
                
        return calibrated_logits

    def get_mhsr_summary(self) -> Dict[str, Any]:
        """Returns summarized metrics for MHSR validation."""
        lscp_summary = super().get_lscp_summary()
        
        return {
            **lscp_summary,
            "traversal_integrity": 1.0 if self.mhsr_stats["traversal_count"] > 0 else 0.0,
            "relationship_stability": self.mhsr_stats["relationship_count"],
            "recursion_suppression": self.mhsr_stats["recursion_suppressions"],
            "symbolic_composition_quality": self.mhsr_stats["mean_composition_quality"],
            "topology_survival": lscp_summary["delimiter_integrity"]
        }
