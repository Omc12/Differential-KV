"""
anchor_logic/reasoning_anchors.py
Phase 14: Reasoning Anchors and Temporal Chains
Implements specialized anchors for cognitive trajectory preservation.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from .semantic_anchor_system import SemanticAnchor, SemanticAnchorMemory, AnchorSelectionPolicy

@dataclass
class ReasoningAnchor(SemanticAnchor):
    """
    Anchors critical for reasoning trajectories.
    """
    step_idx: int = 0
    reasoning_type: str = "logical_transition" # pivot, checkpoint, inference
    is_chained: bool = False
    next_anchor_pos: Optional[int] = None
    trajectory_context: Optional[torch.Tensor] = None # Compressed hidden state

class ReasoningAnchorMemory(SemanticAnchorMemory):
    """
    Extends SAM with support for temporal chains and reasoning anchors.
    """
    def __init__(self, max_anchors: int = 128, chain_length: int = 4):
        super().__init__(max_anchors=max_anchors)
        self.chain_length = chain_length
        self.current_chain: List[int] = [] # List of positions in the current chain
        
    def add_reasoning_anchor(self, anchor: ReasoningAnchor):
        self.add_anchor(anchor)
        if anchor.is_chained:
            self.current_chain.append(anchor.position)
            if len(self.current_chain) > 1:
                # Link to previous
                prev_pos = self.current_chain[-2]
                if prev_pos in self.anchors:
                    self.anchors[prev_pos].next_anchor_pos = anchor.position
            
            # Keep chain length under limit
            if len(self.current_chain) > self.chain_length:
                self.current_chain.pop(0)

class ChainOfThoughtPolicy(AnchorSelectionPolicy):
    """
    Selects anchors based on CoT markers and logical step indicators.
    """
    def __init__(self, tokenizer, keywords: List[str] = None):
        self.tokenizer = tokenizer
        self.keywords = keywords or ["step", "therefore", "thus", "because", "so", "first", "finally"]
        self.keyword_ids = set()
        for k in self.keywords:
            ids = tokenizer.encode(f" {k}", add_special_tokens=False)
            self.keyword_ids.update(ids)

    def select(self, tokens, kv_states, metrics) -> List[ReasoningAnchor]:
        selected = []
        for i, tid in enumerate(tokens):
            tid_val = tid.item() if hasattr(tid, 'item') else tid
            if tid_val in self.keyword_ids:
                selected.append(ReasoningAnchor(
                    token_id=tid_val,
                    position=i,
                    kv_exact=kv_states[i].clone(),
                    importance_score=2.5,
                    reason="cot_marker",
                    reasoning_type="pivot",
                    is_chained=True
                ))
        return selected

class HighEntropyTransitionPolicy(AnchorSelectionPolicy):
    """
    Selects anchors where attention entropy or KL spikes, indicating a logical leap.
    """
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def select(self, tokens, kv_states, metrics) -> List[ReasoningAnchor]:
        kl_divs = metrics.get('kl_divergences')
        if kl_divs is None: return []
        
        selected = []
        for i, kl in enumerate(kl_divs):
            if kl > self.threshold:
                selected.append(ReasoningAnchor(
                    token_id=tokens[i].item() if hasattr(tokens[i], 'item') else tokens[i],
                    position=i,
                    kv_exact=kv_states[i].clone(),
                    importance_score=kl.item() * 4.0,
                    reason="logical_leap",
                    reasoning_type="checkpoint"
                ))
        return selected

class TemporalAnchorChain:
    """
    Manager for linking anchors across time steps.
    Ensures that if one anchor in a chain is present, the next one is prioritized.
    """
    def __init__(self, window_size: int = 512):
        self.window_size = window_size
        self.active_chains: Dict[int, List[int]] = {} # head_pos -> list of anchor_pos
        
    def get_chain_mask(self, current_pos: int, anchors: Dict[int, ReasoningAnchor]) -> torch.Tensor:
        """Returns a mask of positions that should be anchored based on active chains."""
        mask = torch.zeros(current_pos)
        for pos, anchor in anchors.items():
            if anchor.is_chained and anchor.next_anchor_pos is not None:
                # If we are close to the next anchor, signal stabilization
                if 0 < anchor.next_anchor_pos - current_pos < 64:
                    mask[pos] = 1.0
        return mask

class TrajectoryStabilizer:
    """
    Handles dynamic reasoning repair by steering the latent manifold.
    """
    def __init__(self, memory: ReasoningAnchorMemory):
        self.memory = memory
        self.drift_history: List[float] = []

    def detect_reasoning_collapse(self, current_hidden: torch.Tensor, baseline_hidden: Optional[torch.Tensor]) -> bool:
        if baseline_hidden is None: return False
        drift = torch.norm(current_hidden - baseline_hidden, p=2).item()
        self.drift_history.append(drift)
        # Collapse detected if drift accelerates
        if len(self.drift_history) > 3:
            accel = self.drift_history[-1] - self.drift_history[-2]
            return accel > 0.5
        return False

    def steer_manifold(self, hidden_state: torch.Tensor, anchor_state: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """
        Gently pushes the hidden state back towards the anchored manifold.
        """
        return (1 - alpha) * hidden_state + alpha * anchor_state
