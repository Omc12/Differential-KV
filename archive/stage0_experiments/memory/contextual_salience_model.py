import torch
import torch.nn.functional as F

class ContextualSalienceModel:
    """
    PHASE 20.1A: Contextual Symbolic Salience Modeling.
    Evolves entropy-based detection into contextual importance estimation.
    Identifies 'meaningful' tokens even if they lack extreme entropy spikes.
    """
    def __init__(self, hidden_dim: int = 3584, window_size: int = 8):
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.historical_magnitudes = None # EMA of token magnitudes
        self.alpha = 0.1 # Smoothing factor for context drift

    def estimate_salience(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Calculates multi-component salience scores.
        Args:
            hidden_states: [batch, q_len, hidden_dim]
        Returns:
            salience_scores: [batch, q_len]
        """
        magnitudes = torch.norm(hidden_states, p=2, dim=-1) # [batch, q_len]
        q_len = magnitudes.shape[1]
        
        # 1. Local Contrast (Statistical Novelty)
        mean_mag = magnitudes.mean(dim=-1, keepdim=True)
        if q_len > 1:
            std_mag = magnitudes.std(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            std_mag = torch.full_like(mean_mag, 1e-6)
            
        local_z = (magnitudes - mean_mag) / std_mag
        
        # 2. Contextual Novelty (Drift from History)
        if self.historical_magnitudes is None:
            self.historical_magnitudes = mean_mag.detach()
            context_novelty = torch.zeros_like(magnitudes)
        else:
            # Shift in magnitude relative to historical baseline
            context_novelty = (magnitudes - self.historical_magnitudes).abs() / self.historical_magnitudes.clamp(min=1e-6)
            # Update history
            self.historical_magnitudes = (1 - self.alpha) * self.historical_magnitudes + self.alpha * mean_mag.detach()

        # 3. Structural Tension (Layer-wise variance proxy)
        max_mag = magnitudes.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        tension = magnitudes / max_mag

        # 4. Composite Salience
        # We combine local surprise, global novelty, and structural intensity
        salience = (0.4 * local_z.clamp(min=0)) + (0.4 * context_novelty) + (0.2 * tension)
        
        # FINAL SAFETY: Ensure no NaNs escape
        salience = torch.nan_to_num(salience, nan=0.0, posinf=1.0, neginf=0.0)
        
        return salience

    def detect_salient_tokens(self, salience_scores: torch.Tensor, base_threshold: float = 1.5) -> torch.Tensor:
        """
        Generates a binary mask for salient symbolic candidates.
        """
        return salience_scores > base_threshold
