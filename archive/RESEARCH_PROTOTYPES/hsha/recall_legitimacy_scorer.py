
import torch
from typing import List, Dict, Optional

class RecallLegitimacyScorer:
    """
    PHASE 21.1: Evaluates how 'legitimate' a symbolic recall event is.
    Prevents brute-force replay by checking context, lineage, and confidence.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.min_relevance_threshold = 0.3
        self.neutral_lineage_score = 0.5
        
    def score_relevance(self, context_tokens: List[int], hub_tokens: List[int]) -> float:
        """
        Calculates contextual relevance based on prefix matches.
        Determines if the hub tokens are a natural continuation of the context.
        """
        if not context_tokens or not hub_tokens:
            return 0.0
            
        # We look for the longest suffix of context that matches a prefix of hub
        max_match = 0
        window = min(12, len(context_tokens), len(hub_tokens))
        
        for i in range(1, window + 1):
            if context_tokens[-i:] == hub_tokens[:i]:
                max_match = i
                
        # Normalize score
        return max_match / 8.0 # High relevance if we match 8+ tokens

    def score_lineage(self, active_lineage: List[str], hub_lineage: List[str]) -> float:
        """
        Checks for lineage consistency. 
        Objects in the same chain of thought/propagation are more legitimate.
        """
        if not active_lineage or not hub_lineage:
            return self.neutral_lineage_score
            
        common = set(active_lineage) & set(hub_lineage)
        if common:
            return 1.0
        return 0.2 # Penalty for lineage mismatch

    def estimate_confidence(self, relevance: float, lineage_score: float, 
                            drift_risk: float) -> float:
        """
        Consolidates signals into a single recall confidence estimate [0, 1].
        """
        # Confidence is high if relevance is high OR (relevance is moderate AND risk is high)
        confidence = (relevance * 0.7) + (lineage_score * 0.3)
        
        # Amplify confidence if the system is drifting (needs help)
        if drift_risk > 0.5:
            confidence *= 1.2
            
        return min(1.0, confidence)

    def calculate_injection_strength(self, confidence: float, entropy: float) -> float:
        """
        Controls the reinjection strength.
        Lower strength if entropy is already low to prevent deterministic collapse.
        """
        strength = confidence * 1.5 # Scale to reasonable logit boost
        
        # Entropy-aware dampening
        if entropy < 0.5:
            strength *= 0.5
            
        return min(2.5, strength)
