import torch

class RetrievalSurvivalController:
    """
    Monitors retrieval 'health' and prevents catastrophic retrieval collapse.
    Acts as a safety valve for aggressive sparsity policies.
    """
    def __init__(self, survival_threshold: float = 0.9):
        self.survival_threshold = survival_threshold
        self.is_critical = False

    def check_health(self, attention_probs: torch.Tensor, retrieval_targets: torch.Tensor):
        """
        Validates if retrieval targets are receiving sufficient attention.
        """
        # attention_probs: [batch, heads, q_len, k_len]
        # retrieval_targets: [batch, num_targets]
        
        target_attention = attention_probs.gather(-1, retrieval_targets.unsqueeze(1).expand(-1, attention_probs.size(1), -1))
        avg_target_attn = target_attention.mean()
        
        if avg_target_attn < self.survival_threshold * (1.0 / attention_probs.size(-1)):
            self.is_critical = True
            return "COLLAPSE_WARNING"
            
        self.is_critical = False
        return "HEALTHY"

    def get_emergency_density(self):
        """
        Returns a density recommendation to recover from collapse.
        """
        if self.is_critical:
            return 0.5  # Forced 50% density to recover
        return None
