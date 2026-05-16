import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

def stable_kl_divergence(log_p: torch.Tensor, p_target: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Mathematically sound and numerically stable KL Divergence.
    KL(P || Q) = sum(P * (log P - log Q))
    
    Args:
        log_p: Log-probabilities of the baseline (P). Shape [B, V] or [V]
        p_target: Probabilities of the compressed model (Q). Shape [B, V] or [V]
    """
    # Ensure fp32 for calculation
    log_p = log_p.float()
    p_target = p_target.float()
    
    # P = exp(log_p)
    p = log_p.exp()
    
    # Q = p_target (already probs)
    q = p_target.clamp(min=eps) # Avoid log(0)
    
    # KL(P || Q) = sum(P * (log P - log Q))
    # Note: F.kl_div(input, target) computes target * (log target - input)
    # So if target=P and input=log Q:
    # F.kl_div(log_q, p) = p * (log p - log q)
    
    log_q = q.log()
    
    # Use reduction='batchmean' if batch dimension exists
    # We'll do it manually to be absolutely sure
    kl = (p * (log_p - log_q)).sum(dim=-1)
    
    # Strictly non-negative
    kl = torch.clamp(kl, min=0.0)
    
    return kl.mean().item()

def stable_attention_entropy(attn_weights: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-10) -> torch.Tensor:
    """
    Stable entropy calculation for attention weights.
    H(P) = -sum(P * log P)
    
    Args:
        attn_weights: [batch, heads, seq_q, seq_k] probabilities
        mask: Optional mask
    """
    p = attn_weights.float()
    
    if mask is not None:
        # If mask is 0 for padded positions, ensure P is 0 there
        p = p * mask.float()
        # Re-normalize if necessary? Usually attn_weights are already softmaxed with mask
        
    log_p = torch.log(p + eps)
    entropy = -(p * log_p).sum(dim=-1) # [batch, heads, seq_q]
    
    # Handle NaNs (shouldn't happen with eps, but safety first)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    
    return entropy

def verify_retrieval(response: str, expected_answer: str) -> bool:
    """
    Strict retrieval verification.
    """
    if not response or not expected_answer:
        return False
        
    res = response.lower().strip()
    ans = expected_answer.lower().strip()
    
    # Check if answer is in response
    if ans in res:
        return True
        
    # Check for tokenization artifacts or slight variations
    # (e.g. "albatross-99" vs "albatross 99")
    res_clean = "".join(c for c in res if c.isalnum())
    ans_clean = "".join(c for c in ans if c.isalnum())
    
    if ans_clean and ans_clean in res_clean:
        return True
        
    return False

def stable_cosine_drift(h1: torch.Tensor, h2: torch.Tensor) -> float:
    """
    Measures cosine similarity drift between two hidden states.
    1.0 means identical, 0.0 means orthogonal.
    """
    h1 = h1.float().view(-1)
    h2 = h2.float().view(-1)
    return F.cosine_similarity(h1, h2, dim=0).item()

def logit_rank_correlation(logits1: torch.Tensor, logits2: torch.Tensor, top_n: int = 100) -> float:
    """
    Spearman rank correlation of the top-N logits.
    """
    from scipy.stats import spearmanr
    v1 = logits1.float().cpu().numpy().flatten()
    v2 = logits2.float().cpu().numpy().flatten()
    
    # Take top N of the first one to focus on relevant tokens
    indices = np.argsort(v1)[-top_n:]
    
    corr, _ = spearmanr(v1[indices], v2[indices])
    return float(corr)

def top_k_overlap(logits1: torch.Tensor, logits2: torch.Tensor, k: int = 10) -> float:
    """
    Percentage of tokens that appear in the top-k of both models.
    """
    idx1 = torch.topk(logits1, k, dim=-1).indices.view(-1).tolist()
    idx2 = torch.topk(logits2, k, dim=-1).indices.view(-1).tolist()
    
    set1 = set(idx1)
    set2 = set(idx2)
    
    overlap = len(set1.intersection(set2))
    return overlap / k
