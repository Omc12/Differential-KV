import torch

class AdaptiveThresholdScheduler:
    """
    PHASE 20.1D: Dynamically adjusts symbolic thresholds based on context density.
    Lowers threshold in clear contexts, raises it in noisy/repetitive contexts.
    """
    def __init__(self, base_threshold: float = 1.2):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.pressure_ema = 0.0

    def adjust_threshold(self, hidden_states: torch.Tensor) -> float:
        """
        Estimates 'Context Density' (repetitiveness/noise) and adjusts threshold.
        """
        # Repetitiveness proxy: token-wise cosine similarity variance
        magnitudes = torch.norm(hidden_states, p=2, dim=-1)
        q_len = magnitudes.shape[1]
        
        mean = magnitudes.mean()
        
        if q_len > 1:
            std = magnitudes.std().clamp(min=1e-6)
            # High STD relative to mean implies high contrast (clear context)
            # Low STD relative to mean implies uniformity (repetitive/noisy context)
            noise_pressure = 1.0 / (std / mean.clamp(min=1e-6))
        else:
            # For single tokens, we can't estimate density from variance.
            # Default to neutral pressure or maintain EMA.
            noise_pressure = torch.tensor(0.5, device=hidden_states.device)
        
        # Update Pressure EMA
        # Ensure noise_pressure is not NaN/Inf
        noise_val = torch.nan_to_num(noise_pressure, nan=0.5, posinf=1.0, neginf=0.0).item()
        
        # PHASE 20.4: Reduce sensitivity to pressure to prevent steering collapse
        self.pressure_ema = 0.95 * self.pressure_ema + 0.05 * noise_val
        
        # Threshold scales with pressure
        # PHASE 20.4: Dampen the scale to keep threshold within steering range
        self.current_threshold = self.base_threshold * (0.8 + 0.4 * self.pressure_ema)
        
        # FINAL SAFETY: Ensure threshold is valid and within reasonable bounds
        if torch.isnan(torch.tensor(self.current_threshold)):
            self.current_threshold = self.base_threshold
            
        return self.current_threshold
