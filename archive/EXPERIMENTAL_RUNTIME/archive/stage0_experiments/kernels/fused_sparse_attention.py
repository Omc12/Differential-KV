import torch
import torch.nn.functional as F

class FusedSparseAttention:
    """
    Simulated fused sparse attention kernel.
    In a real production environment, this would be a CUDA/Triton kernel.
    Focuses on reducing IO bandwidth by only fetching required KV pairs.
    """
    @staticmethod
    def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Simulates the reduced IO of a fused kernel.
        q: [B, H, Q, D]
        k: [B, H, K, D]
        v: [B, H, K, D]
        mask: [B, H, Q, K] (Boolean sparse mask)
        """
        # Logic: In a real kernel, we wouldn't compute the full K*Q matrix.
        # Here we simulate this by applying the mask before the expensive MatMul
        # (or using torch.sparse, but dense masking is easier for simulation).
        
        # 1. Compute scores only where mask is True (simulated)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1)**0.5)
        
        # 2. Apply mask
        attn = attn.masked_fill(~mask, float("-inf"))
        
        # 3. Softmax
        probs = F.softmax(attn, dim=-1)
        
        # 4. Final aggregation
        return torch.matmul(probs, v)

    @staticmethod
    def get_io_savings(mask: torch.Tensor) -> float:
        """Returns the percentage of bandwidth saved by the sparse mask."""
        total_elements = mask.numel()
        active_elements = mask.sum().item()
        return 1.0 - (active_elements / total_elements)
