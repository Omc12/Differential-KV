"""
memory/anchor_entropy_optimizer.py

Phase 12C: Anchor Entropy Optimizer
Optimizes anchor selection by prioritizing tokens with high information 
entropy, ensuring maximum 'semantic information density' in sparse memory.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict
from anchor_logic.semantic_anchor_system import SemanticAnchor

class AnchorEntropyOptimizer:
    """
    Analyzes the entropy of token distributions to identify critical 
    transition points in the text.
    """
    def __init__(self, entropy_threshold: float = 2.5):
        self.entropy_threshold = entropy_threshold

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Computes Shannon entropy for a batch of logits."""
        probs = F.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    def filter_by_entropy(self, candidates: List[SemanticAnchor], logits_stream: List[torch.Tensor]) -> List[SemanticAnchor]:
        """
        Refines a candidate list by keeping only those that correspond to 
        high-entropy tokens.
        """
        refined = []
        for i, anchor in enumerate(candidates):
            if i < len(logits_stream):
                ent = self.compute_entropy(logits_stream[i]).item()
                if ent > self.entropy_threshold:
                    anchor.importance_score *= (ent / self.entropy_threshold)
                    anchor.reason += "_high_entropy"
                    refined.append(anchor)
                elif anchor.importance_score > 2.0: # Keep very important anchors regardless
                    refined.append(anchor)
        
        return refined
