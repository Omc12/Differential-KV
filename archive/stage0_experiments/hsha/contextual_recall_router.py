
import torch
from typing import List, Dict, Optional

class ContextualRecallRouter:
    """
    PHASE 21.0: Manages the 'Recall Gate'.
    Decides when and which hub objects should be recalled based on current context.
    """
    def __init__(self, hub_registry):
        self.hub_registry = hub_registry
        self.recall_threshold = 0.6
        self.min_attention_mass = 0.02

    def score_recall_legitimacy(self, query_tokens: List[int], hub_tokens: List[int]) -> float:
        """
        Calculates how 'legitimate' it is to recall a hub object.
        Based on query match (prefix) and lineage continuity.
        """
        if not query_tokens or not hub_tokens:
            return 0.0
            
        # Check prefix match (soft)
        # We look for the hub tokens as a continuation of the query
        # Or if the query is a prefix of the hub tokens
        match_count = 0
        min_len = min(len(query_tokens), len(hub_tokens))
        
        # Check if query ends with the start of hub_tokens
        # We only check up to a small window for "triggering"
        trigger_window = min(4, min_len)
        for i in range(1, trigger_window + 1):
            if query_tokens[-i:] == hub_tokens[:i]:
                match_count = i
        
        return match_count / trigger_window if trigger_window > 0 else 0.0

    def route_recall(self, current_context_tokens: List[int], active_hub_ids: List[str]) -> Dict[str, float]:
        """
        Determines which hubs to boost based on current context.
        Returns a map of hub_id to recall_score.
        """
        recall_scores = {}
        for hub_id in active_hub_ids:
            hub_obj = self.hub_registry.get_object(hub_id)
            if not hub_obj:
                continue
                
            # Scoring logic
            legitimacy = self.score_recall_legitimacy(current_context_tokens, hub_obj.tokens)
            
            # Check lineage matching (conceptual)
            # If current context contains tokens from a parent hub, boost the child
            if hub_obj.lineage:
                for parent_id in hub_obj.lineage:
                    parent_obj = self.hub_registry.get_object(parent_id)
                    if parent_obj and any(t in current_context_tokens[-32:] for t in parent_obj.tokens):
                        legitimacy += 0.3 # Lineage boost
            
            recall_scores[hub_id] = min(1.0, legitimacy)
            
        return recall_scores
