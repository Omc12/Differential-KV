import torch
import torch.nn.functional as F

class TokenSampler:
    """
    Real token sampling for non-deterministic generation.
    Supports temperature, top-k, and top-p (nucleus) sampling.
    """
    def __init__(self, temperature=1.0, top_k=0, top_p=0.0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / max(self.temperature, 1e-5)
        
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
            
        if self.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[..., indices_to_remove] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
