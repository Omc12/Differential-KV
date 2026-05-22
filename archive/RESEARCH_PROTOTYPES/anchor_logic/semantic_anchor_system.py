"""
anchor_logic/semantic_anchor_system.py

Phase 11: Semantic Anchor Memory System
Implements a lightweight side-channel memory to preserve exact semantic identity
in the presence of aggressive KV compression.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

@dataclass
class SemanticAnchor:
    """
    Represents a semantically critical token stored in the side-channel.
    Supports varying levels of detail (Ablation Task 2).
    """
    token_id: int
    position: int
    kv_exact: Optional[torch.Tensor] = None  # [2, heads, dim]
    importance_score: float = 1.0
    reason: str = "unknown"
    
    # Metadata-only fields (Ablation 1)
    metadata_only: bool = False
    
    # Partial KV fields (Ablation 2)
    selected_heads: Optional[List[int]] = None # List of head indices if partial
    
    positional_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SemanticAnchorMemory:
    """
    The Semantic Anchor Memory (SAM) subsystem.
    Manages storage, selection, and reinjection of semantic anchors.
    """
    def __init__(self, max_anchors: int = 128, budget_per_token: float = 0.1):
        self.max_anchors = max_anchors
        self.budget_per_token = budget_per_token
        self.anchors: Dict[int, SemanticAnchor] = {}  # pos -> anchor
        self.selection_history: List[Dict] = []
        
    def add_anchor(self, anchor: SemanticAnchor):
        if len(self.anchors) < self.max_anchors:
            self.anchors[anchor.position] = anchor
        else:
            if not self.anchors:
                 self.anchors[anchor.position] = anchor
                 return
            # Simple eviction policy: replace lowest importance
            min_pos = min(self.anchors.keys(), key=lambda k: self.anchors[k].importance_score)
            if anchor.importance_score > self.anchors[min_pos].importance_score:
                del self.anchors[min_pos]
                self.anchors[anchor.position] = anchor

    def get_anchors_at_positions(self, positions: List[int]) -> List[Optional[SemanticAnchor]]:
        return [self.anchors.get(p) for p in positions]

    def reset(self):
        self.anchors.clear()
        self.selection_history.clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        if not self.anchors:
            return {"num_anchors": 0, "size_bytes": 0}
        
        first_anchor = next(iter(self.anchors.values()))
        if first_anchor.kv_exact is not None:
             kv_size = first_anchor.kv_exact.numel() * 2 # FP16
        else:
             kv_size = 0
             
        # Estimate metadata size (very rough)
        metadata_size = 32 # IDs, positions, etc.
        anchor_size = metadata_size + kv_size
        total_size = len(self.anchors) * anchor_size
        
        return {
            "num_anchors": len(self.anchors),
            "size_bytes": total_size,
            "anchor_reasons": self._count_reasons()
        }

    def _count_reasons(self) -> Dict[str, int]:
        reasons = {}
        for a in self.anchors.values():
            reasons[a.reason] = reasons.get(a.reason, 0) + 1
        return reasons

class AnchorSelectionPolicy:
    """Base class for anchor selection policies."""
    def select(self, tokens, kv_states, metrics: Dict[str, torch.Tensor]) -> List[SemanticAnchor]:
        raise NotImplementedError

class AttentionPeakPolicy(AnchorSelectionPolicy):
    """Selects anchors based on attention peak weights."""
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        attn = metrics.get('attn_weights')
        if attn is None: return []
        # attn: [layers, heads, q_len, k_len]
        importance = attn.max(dim=-2)[0].mean(dim=(0, 1)) # [k_len]
        selected = []
        top_indices = torch.where(importance > self.threshold)[0]
        for idx in top_indices:
            pos = idx.item()
            selected.append(SemanticAnchor(
                token_id=tokens[pos].item() if hasattr(tokens[pos], 'item') else tokens[pos],
                position=pos,
                kv_exact=kv_states[pos].clone(),
                importance_score=importance[pos].item(),
                reason="attention_peak"
            ))
        return selected

class EntropyPolicy(AnchorSelectionPolicy):
    """Selects anchors based on entropy spikes."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        entropies = metrics.get('entropies')
        if entropies is None: return []
        diffs = torch.abs(entropies[1:] - entropies[:-1])
        selected = []
        for i in range(len(diffs)):
            if diffs[i] > self.threshold:
                selected.append(SemanticAnchor(
                    token_id=tokens[i+1].item() if hasattr(tokens[i+1], 'item') else tokens[i+1],
                    position=i+1,
                    kv_exact=kv_states[i+1].clone(),
                    importance_score=diffs[i].item(),
                    reason="entropy_spike"
                ))
        return selected

class RareTokenPolicy(AnchorSelectionPolicy):
    """Selects anchors based on token frequency."""
    def __init__(self, rare_token_ids: Optional[List[int]] = None):
        self.rare_token_ids = set(rare_token_ids) if rare_token_ids else set()

    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        selected = []
        for i, tid in enumerate(tokens):
            tid_val = tid.item() if hasattr(tid, 'item') else tid
            if tid_val in self.rare_token_ids or tid_val > 50000:
                selected.append(SemanticAnchor(
                    token_id=tid_val,
                    position=i,
                    kv_exact=kv_states[i].clone(),
                    importance_score=2.0,
                    reason="rare_token"
                ))
        return selected

class KLSensitivityPolicy(AnchorSelectionPolicy):
    """Selects anchors based on KL divergence sensitivity."""
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        kl_divs = metrics.get('kl_divergences')
        if kl_divs is None: return []
        selected = []
        for i, kl in enumerate(kl_divs):
            if kl > self.threshold:
                selected.append(SemanticAnchor(
                    token_id=tokens[i].item() if hasattr(tokens[i], 'item') else tokens[i],
                    position=i,
                    kv_exact=kv_states[i].clone(),
                    importance_score=kl.item() * 5.0,
                    reason="kl_sensitivity"
                ))
        return selected

class PositionAwarePolicy(AnchorSelectionPolicy):
    """Anchors based on position."""
    def __init__(self, interval: int = 128):
        self.interval = interval

    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        selected = []
        for i in range(len(tokens)):
            if i % self.interval == 0:
                selected.append(SemanticAnchor(
                    token_id=tokens[i].item() if hasattr(tokens[i], 'item') else tokens[i],
                    position=i,
                    kv_exact=kv_states[i].clone(),
                    importance_score=1.5,
                    reason="positional"
                ))
        return selected

class LearnedSelectionPolicy(AnchorSelectionPolicy):
    """Placeholder for a learned policy (Task 5)."""
    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        # In practice, this would use a small MLP or heuristic ensemble
        return []

class RetrievalGradientPolicy(AnchorSelectionPolicy):
    """Selects anchors based on gradients of retrieval loss (Task 5)."""
    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        grads = metrics.get('retrieval_gradients')
        if grads is None: return []
        selected = []
        # Select positions where grads are high (meaning KV state is critical for retrieval)
        threshold = torch.quantile(grads, 0.95)
        top_indices = torch.where(grads > threshold)[0]
        for idx in top_indices:
            pos = idx.item()
            selected.append(SemanticAnchor(
                token_id=tokens[pos].item() if hasattr(tokens[pos], 'item') else tokens[pos],
                position=pos,
                kv_exact=kv_states[pos].clone(),
                importance_score=grads[pos].item(),
                reason="retrieval_gradient"
            ))
        return selected

class PositionalSaliencyPolicy(AnchorSelectionPolicy):
    """Anchors based on positional saliency (e.g. beginning of document, sentence starts)."""
    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        saliency = metrics.get('positional_saliency')
        if saliency is None: return []
        selected = []
        top_indices = torch.where(saliency > 0.5)[0]
        for idx in top_indices:
            pos = idx.item()
            selected.append(SemanticAnchor(
                token_id=tokens[pos].item() if hasattr(tokens[pos], 'item') else tokens[pos],
                position=pos,
                kv_exact=kv_states[pos].clone(),
                importance_score=saliency[pos].item(),
                reason="positional_saliency"
            ))
        return selected

class AdaptiveOnlinePolicy(AnchorSelectionPolicy):
    """Dynamically adjusts thresholds based on budget."""
    def __init__(self, target_budget: float = 0.01):
        self.target_budget = target_budget
        self.current_threshold = 0.5

    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        # Logic to adjust threshold based on history
        return []

class HybridPolicy(AnchorSelectionPolicy):
    """Combines multiple policies."""
    def __init__(self, policies: List[AnchorSelectionPolicy]):
        self.policies = policies

    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        all_candidates = []
        for policy in self.policies:
            all_candidates.extend(policy.select(tokens, kv_states, metrics))
        by_pos = {}
        for cand in all_candidates:
            if cand.position not in by_pos or cand.importance_score > by_pos[cand.position].importance_score:
                by_pos[cand.position] = cand
        return list(by_pos.values())

class StreamingPolicy(AnchorSelectionPolicy):
    """Simple sliding window policy (StreamingLLM style baseline)."""
    def __init__(self, window_size: int = 256):
        self.window_size = window_size

    def select(self, tokens, kv_states, metrics) -> List[SemanticAnchor]:
        seq_len = len(tokens)
        selected = []
        # Keep first few tokens (attention sinks)
        for i in range(min(4, seq_len)):
             selected.append(SemanticAnchor(token_id=0, position=i, kv_exact=kv_states[i].clone()))
        # Keep recent tokens
        for i in range(max(0, seq_len - self.window_size), seq_len):
             selected.append(SemanticAnchor(token_id=0, position=i, kv_exact=kv_states[i].clone()))
        return selected

class SemanticReinjector:
    """
    Handles the reinjection of semantic anchors into the generation process.
    """
    def __init__(self, memory: SemanticAnchorMemory):
        self.memory = memory

    def apply_kv_substitution(self, reconstructed_kv: torch.Tensor, positions: List[int]) -> torch.Tensor:
        """
        Replaces reconstructed KV states with exact anchor states where available.
        Handles partial KV and metadata-only anchors (Ablation Task 2).
        """
        repaired_kv = reconstructed_kv.clone()
        for i, pos in enumerate(positions):
            anchor = self.memory.anchors.get(pos)
            if anchor is not None:
                if anchor.metadata_only:
                    # For metadata-only, we might boost attention but not replace KV
                    continue
                
                if anchor.kv_exact is not None:
                    if anchor.selected_heads is not None:
                        # Partial head substitution
                        for h_idx in anchor.selected_heads:
                            repaired_kv[i, :, h_idx] = anchor.kv_exact[:, h_idx].to(repaired_kv.device)
                    else:
                        # Full substitution
                        repaired_kv[i] = anchor.kv_exact.to(repaired_kv.device)
        return repaired_kv

    def apply_attention_biasing(self, attn_weights: torch.Tensor, positions: List[int], bias_strength: float = 2.0) -> torch.Tensor:
        """
        Biases attention weights towards anchored positions.
        attn_weights: [layers, heads, q_len, k_len]
        """
        biased_attn = attn_weights.clone()
        for i, pos in enumerate(positions):
            if pos in self.memory.anchors:
                # Add bias to the corresponding key position
                # Assuming k_len corresponds to the positions list
                biased_attn[..., :, i] += bias_strength
        return biased_attn

    def apply_anchor_boosting(self, logits: torch.Tensor, current_pos: int) -> torch.Tensor:
        """
        Boosts the probability of anchored tokens if they are likely to be retrieved (e.g. induction).
        """
        boosted_logits = logits.clone()
        for anchor in self.memory.anchors.values():
            if current_pos - anchor.position < 100:
                boosted_logits[..., anchor.token_id] += 0.5 * anchor.importance_score
        return boosted_logits
