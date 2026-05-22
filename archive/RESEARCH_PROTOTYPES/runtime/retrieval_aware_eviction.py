import torch
from typing import List, Set

class RetrievalAwareEviction:
    """
    KV eviction policy that prioritizes retrieval-critical tokens.
    Protects 'anchor tokens' (attention sinks, key semantic points).
    """
    def __init__(self, anchor_ratio: float = 0.1, recent_ratio: float = 0.2):
        self.anchor_ratio = anchor_ratio
        self.recent_ratio = recent_ratio

    def get_eviction_indices(self, attn_history: torch.Tensor, current_len: int, target_len: int) -> torch.Tensor:
        """
        Identify indices to EVICT.
        attn_history: [LAYERS, HEADS, K_LEN] (accumulated attention scores)
        """
        if current_len <= target_len:
            return torch.tensor([], dtype=torch.long)

        # 1. Protect Anchor Tokens: Top-k accumulated attention
        num_anchors = int(target_len * self.anchor_ratio)
        _, anchor_indices = torch.topk(attn_history.mean(dim=(0, 1)), num_anchors)
        
        # 2. Protect Recent Tokens
        num_recent = int(target_len * self.recent_ratio)
        recent_indices = torch.arange(current_len - num_recent, current_len, device=attn_history.device)
        
        protected_indices = torch.cat([anchor_indices, recent_indices]).unique()
        
        # 3. Candidate indices for eviction (everything not protected)
        all_indices = torch.arange(current_len, device=attn_history.device)
        
        # We need a mask of what to KEEP. 
        # But the tool asks for eviction indices.
        
        mask = torch.ones(current_len, dtype=torch.bool, device=attn_history.device)
        mask[protected_indices] = False
        
        eviction_candidates = all_indices[mask]
        
        # Prune the least important of the remaining
        num_to_evict = current_len - target_len
        evict_scores = attn_history.mean(dim=(0, 1))[eviction_candidates]
        _, sorted_evict_local_indices = torch.sort(evict_scores)
        
        final_eviction_indices = eviction_candidates[sorted_evict_local_indices[:num_to_evict]]
        
        return final_eviction_indices
