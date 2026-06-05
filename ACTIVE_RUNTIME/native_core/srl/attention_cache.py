import torch
from typing import Dict, Tuple, Optional

class AttentionScoreCache:
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        # session_id -> layer_idx -> (cached_query, cached_attn_out)
        self.cache: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def clear_session(self, session_id: str):
        self.cache.pop(session_id, None)

    def check_and_update(
        self,
        session_id: str,
        layer_idx: int,
        q: torch.Tensor,  # [1, H_q, 1, D]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Check if we can reuse the previous step's attention output.
        Returns:
            reuse_mask: [H_q] boolean tensor, True for heads that can be reused.
            cached_out: [1, H_q, 1, D] previous attention output.
        """
        # Ensure session is in cache
        if session_id not in self.cache:
            self.cache[session_id] = {}
            return None, None
            
        layer_cache = self.cache[session_id].get(layer_idx)
        if layer_cache is None:
            return None, None
            
        q_prev, attn_out_prev = layer_cache
        
        # Check shapes and devices
        if q.shape != q_prev.shape or attn_out_prev.device != q.device:
            return None, None
            
        # Compute cosine similarity per head
        H_q, D = q.shape[1], q.shape[3]
        q_flat = q.view(H_q, D)
        q_prev_flat = q_prev.view(H_q, D)
        
        q_norm = q_flat.norm(dim=-1, keepdim=True) + 1e-8
        q_prev_norm = q_prev_flat.norm(dim=-1, keepdim=True) + 1e-8
        
        cos_sim = (q_flat * q_prev_flat).sum(dim=-1, keepdim=True) / (q_norm * q_prev_norm)
        cos_sim = cos_sim.squeeze(-1)  # [H_q]
        
        reuse_mask = cos_sim >= self.threshold  # [H_q]
        
        return reuse_mask, attn_out_prev

    def save(self, session_id: str, layer_idx: int, q: torch.Tensor, attn_out: torch.Tensor):
        if session_id not in self.cache:
            self.cache[session_id] = {}
        # Clone tensors to prevent memory leaks or backward graph retention
        self.cache[session_id][layer_idx] = (q.detach().clone(), attn_out.detach().clone())
