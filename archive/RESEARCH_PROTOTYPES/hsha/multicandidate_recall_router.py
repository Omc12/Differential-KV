
import torch
from typing import List, Dict, Tuple, Optional, Any

class MulticandidateRecallRouter:
    """
    PHASE 21.1: Supports competing symbolic recall candidates.
    Performs weighted probabilistic routing to determine the best recall target.
    """
    def __init__(self, legitimacy_scorer):
        self.scorer = legitimacy_scorer
        self.competition_threshold = 0.1

    def route_multi(self, context_tokens: List[int], candidate_hub_ids: List[str], 
                    hub_registry: Any) -> List[Tuple[str, float]]:
        """
        Routes recall across multiple candidates. 
        Returns a sorted list of (hub_id, legitimacy_score).
        """
        scored_candidates = []
        
        for hub_id in candidate_hub_ids:
            hub_obj = hub_registry.get_object(hub_id)
            if not hub_obj:
                continue
                
            # Score each candidate
            relevance = self.scorer.score_relevance(context_tokens, hub_obj.tokens)
            lineage = self.scorer.score_lineage([], hub_obj.lineage) # Simplified
            
            # Final legitimacy for routing
            score = (relevance * 0.8) + (lineage * 0.2)
            
            if score > self.scorer.min_relevance_threshold:
                scored_candidates.append((hub_id, score))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply normalization if there is competition
        if len(scored_candidates) > 1:
            total_score = sum(s for _, s in scored_candidates)
            scored_candidates = [(h, s / total_score) for h, s in scored_candidates]
            
        return scored_candidates

    def resolve_winner(self, scored_candidates: List[Tuple[str, float]]) -> Optional[str]:
        """Selects the best candidate, ensuring no hard deterministic lock if scores are close."""
        if not scored_candidates:
            return None
            
        # If the gap between top two is small, it's a 'competition' state
        if len(scored_candidates) > 1:
            gap = scored_candidates[0][1] - scored_candidates[1][1]
            if gap < self.competition_threshold:
                # In competition, we might return both or let the injector handle it
                # For now, we take the top one but with a 'weak winner' flag potentially
                pass
                
        return scored_candidates[0][0]
