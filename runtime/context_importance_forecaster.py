import torch

class ContextImportanceForecaster:
    """
    Forecasts which KV blocks will likely be needed for future retrieval.
    Prevents pruning of tokens that will be critical in the next 10-100 tokens.
    """
    def __init__(self, forecast_window: int = 32):
        self.forecast_window = forecast_window

    def forecast(self, historical_importance: torch.Tensor):
        """
        Predicts future importance scores based on historical trends.
        """
        # historical_importance: [batch, heads, seq_len, history_len]
        
        # Simple linear trend prediction
        # If importance is growing for a block, we protect it
        if historical_importance.dim() < 2:
            return torch.zeros_like(historical_importance)
            
        diff = historical_importance[..., -1] - historical_importance[..., -2]
        predicted_future = historical_importance[..., -1] + diff * self.forecast_window
        
        return torch.clamp(predicted_future, 0, 1)

    def get_protection_mask(self, forecast_scores: torch.Tensor, threshold: float = 0.8):
        """
        Returns a mask of blocks to protect from pruning.
        """
        return forecast_scores > threshold
