"""
anchor_logic/dynamic_rank_scheduler.py
Phase 15: Dynamic Rank Morphing
Allocates compute budget (rank) based on cognitive demand.
"""

import torch
from typing import Dict, Any, List

class DynamicRankScheduler:
    def __init__(self, base_rank: int = 4, max_rank: int = 32):
        self.base_rank = base_rank
        self.max_rank = max_rank
        self.rank_map = {
            "narrative": 4,
            "retrieval": 8,
            "reasoning": 16,
            "pivot": 32,
            "collapse_recovery": "FP16"
        }

    def determine_rank(self, context_type: str, stability_score: float) -> Any:
        """
        Returns the appropriate rank or format for the current step.
        """
        # Rule-based fallback if stability is very low
        if stability_score < 0.2:
            return self.rank_map["collapse_recovery"]
        
        # Stability-adjusted rank
        requested_rank = self.rank_map.get(context_type, self.base_rank)
        
        # If stability is declining but not yet collapsed, boost the rank
        if stability_score < 0.6 and requested_rank != "FP16":
            requested_rank = min(self.max_rank, requested_rank * 2)
            
        return requested_rank

    def classify_context(self, tokens: List[int], tokenizer) -> str:
        """
        Simple heuristic for context classification.
        In a full system, this could be a small classifier.
        """
        if not tokens: return "narrative"
        
        text = tokenizer.decode(tokens[-5:]).lower() # Last 5 tokens
        
        # Reasoning indicators
        reasoning_words = ["therefore", "because", "step", "calculate", "if", "then", "proof", "solve"]
        if any(w in text for w in reasoning_words):
            return "reasoning"
            
        # Retrieval indicators (e.g. quotes, long names, technical terms)
        # Placeholder logic
        if any(c.isupper() for c in text if c.isalpha()):
            return "retrieval"
            
        return "narrative"

if __name__ == "__main__":
    scheduler = DynamicRankScheduler()
    print("Normal Narrative:", scheduler.determine_rank("narrative", 0.95))
    print("Reasoning Step:", scheduler.determine_rank("reasoning", 0.90))
    print("Unstable Reasoning:", scheduler.determine_rank("reasoning", 0.40))
    print("Near Collapse:", scheduler.determine_rank("narrative", 0.15))
