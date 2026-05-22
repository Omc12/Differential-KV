"""
runtime/cognitive_priority_manager.py

Determines cognitive priority for tokens, trajectories, and anchors.
Signals: semantic rarity, retrieval sensitivity, CoT pivots, entropy spikes.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional

class CognitivePriorityManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_priority = 0.0
        self.max_priority = 1.0

    def calculate_token_priority(self, 
                                 token_id: int, 
                                 hidden_state: torch.Tensor, 
                                 attention_weights: torch.Tensor,
                                 is_cot: bool = False) -> float:
        """
        Computes priority for a single token.
        """
        # 1. Semantic Rarity (hypothetical - could use norm of hidden state or frequency)
        rarity_score = torch.norm(hidden_state, p=2).item() / 20.0 # Normalized
        
        # 2. Attention Leverage
        # How much does this token contribute to future tokens?
        # Here we look at the entropy of the attention row
        attn_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9)).item()
        # High leverage = low entropy (focused attention)
        leverage_score = np.exp(-attn_entropy) 
        
        # 3. CoT/Reasoning Pivot
        # If the model is in a reasoning chain, increase priority
        cot_bonus = 0.3 if is_cot else 0.0
        
        # 4. Entropy Spikes
        # (This would be calculated across steps, but here we use a placeholder)
        
        priority = (rarity_score * 0.4) + (leverage_score * 0.4) + cot_bonus
        return float(np.clip(priority, self.min_priority, self.max_priority))

    def identify_critical_trajectories(self, history: List[Dict[str, Any]]) -> List[int]:
        """
        Identifies indices of tokens that belong to critical reasoning trajectories.
        """
        critical_indices = []
        if not history:
            return critical_indices
            
        for i, step in enumerate(history):
            if step.get("curvature", 0) > 0.5 or step.get("acceleration", 0) > 0.8:
                critical_indices.append(i)
        return critical_indices

    def decide_eviction(self, anchors: List[Any], budget_limit: int) -> List[Any]:
        """
        Decides which anchors to keep and which to evict.
        """
        if len(anchors) <= budget_limit:
            return anchors
            
        # Sort by importance score (assumed to be on the anchor object)
        sorted_anchors = sorted(anchors, key=lambda x: x.importance_score, reverse=True)
        return sorted_anchors[:budget_limit]
