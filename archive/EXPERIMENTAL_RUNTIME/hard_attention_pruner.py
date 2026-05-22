import torch
import torch.nn as nn
from typing import Optional, Tuple

class HardAttentionPruner:
    """
    Forces sparse attention participation via hard pruning and routing.
    """
    def __init__(self, top_k_ratio: float = 0.1, aggressive_mode: bool = False):
        self.top_k_ratio = top_k_ratio
        self.aggressive_mode = aggressive_mode
        self.metrics = {
            "attention_flop_reduction": 0.0,
            "active_attention_ratio": 1.0,
            "skipped_attention_blocks": 0,
            "dense_attention_bypass_rate": 0.0
        }

    def prune_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        curvature: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies top-k pruning to attention keys/values.
        """
        bsz, n_heads, seq_len, d = k.shape
        top_k = max(1, int(seq_len * self.top_k_ratio))
        
        if self.aggressive_mode:
            # Further reduce participation if aggressive
            top_k = max(1, int(top_k * 0.5))

        # Use dot product as importance proxy if curvature is missing
        if curvature is None:
            importance = torch.matmul(q, k.transpose(-2, -1)).squeeze(-2) # [bsz, n_heads, seq_len]
        else:
            # Merge curvature and resonance
            resonance = torch.matmul(q, k.transpose(-2, -1)).squeeze(-2)
            importance = resonance + 0.5 * curvature

        _, indices = torch.topk(importance, top_k, dim=-1)
        
        # Track metrics
        self.metrics["active_attention_ratio"] = top_k / seq_len
        self.metrics["attention_flop_reduction"] = (1.0 - (top_k / seq_len)) * 100
        
        # Gather sparse KV
        batch_idx = torch.arange(bsz, device=q.device).view(bsz, 1, 1)
        head_idx = torch.arange(n_heads, device=q.device).view(1, n_heads, 1)
        
        k_sparse = k[batch_idx, head_idx, indices, :]
        v_sparse = v[batch_idx, head_idx, indices, :]
        
        return k_sparse, v_sparse

    def get_hard_mask(self, seq_len: int, device: str) -> torch.Tensor:
        """
        Returns a hard attention mask for sparse routing.
        """
        mask = torch.zeros((seq_len, seq_len), device=device)
        # Block-sparse structure could be applied here
        return mask
