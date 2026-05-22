import torch

class ContextualCompetitionSuppressor:
    """PHASE 19.7B: Suppresses competing contextual probability mass."""
    def suppress(self, probabilities: torch.Tensor, confidence: float) -> torch.Tensor:
        if confidence > 0.5:
            # Penalize tokens that are NOT the top choice if confidence is high
            top_val, _ = torch.max(probabilities, dim=-1)
            # Soft suppression: push others down
            probabilities = torch.where(probabilities < top_val * 0.5, probabilities * 0.8, probabilities)
        return probabilities
