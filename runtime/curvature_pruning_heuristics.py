import torch

class CurvaturePruningHeuristics:
    """
    Uses the 'curvature' of the importance manifold (local variance/gradients)
    to identify safe-to-prune tokens.
    'Safe' means tokens where small perturbations in KV don't change attention much.
    """
    def __init__(self):
        pass

    def compute_safety_score(self, k: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        k: [batch, heads, seq_len, head_dim]
        attention_weights: [batch, heads, q_len, seq_len]
        """
        # A token is 'safe' to prune if its removal doesn't significantly alter the softmax distribution.
        # Heuristic: tokens with very low variance in their key vectors relative to others
        # OR tokens that have very low max-attention across all queries.
        
        max_attn, _ = torch.max(attention_weights, dim=-2) # [batch, heads, seq_len]
        
        # Lower max_attn = higher safety to prune
        safety_score = 1.0 - max_attn
        
        return safety_score
