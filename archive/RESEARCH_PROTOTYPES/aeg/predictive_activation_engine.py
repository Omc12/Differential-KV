
import torch
from typing import List, Dict, Any

class PredictiveActivationEngine:
    """
    PHASE 22.1: AEG - Predictive Activation Engine.
    Anticipates compute demand based on symbolic trajectory forecasting.
    """
    def __init__(self, history_len: int = 8):
        self.history_len = history_len
        self.activation_history: List[torch.Tensor] = []
        self.prediction_accuracy = 0.0
        
    def forecast_activation(self) -> torch.Tensor:
        """
        Predicts next-step activation levels based on history.
        """
        if not self.activation_history:
            return torch.zeros(1) # Unknown
            
        # Simple linear extrapolation or average trend
        if len(self.activation_history) < 2:
            return self.activation_history[-1]
            
        # Delta-based prediction
        last = self.activation_history[-1]
        prev = self.activation_history[-2]
        delta = last - prev
        
        prediction = torch.clamp(last + delta * 0.5, 0, 1)
        return prediction

    def record_actual(self, actual_activation: torch.Tensor):
        """
        Records actual activation and updates accuracy metrics.
        """
        if self.activation_history:
            # Simple MSE-based accuracy
            pred = self.forecast_activation()
            if pred.shape == actual_activation.shape:
                error = torch.mean((pred - actual_activation)**2).item()
                accuracy = 1.0 / (1.0 + error)
                # Rolling accuracy
                self.prediction_accuracy = 0.9 * self.prediction_accuracy + 0.1 * accuracy
                
        self.activation_history.append(actual_activation)
        if len(self.activation_history) > self.history_len:
            self.activation_history.pop(0)

    def get_demand_forecast(self) -> float:
        """
        Aggregate compute demand forecast (0.0 to 1.0).
        """
        if not self.activation_history:
            return 0.5
        return self.forecast_activation().mean().item()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "predictive_accuracy": self.prediction_accuracy,
            "demand_forecast": self.get_demand_forecast()
        }
