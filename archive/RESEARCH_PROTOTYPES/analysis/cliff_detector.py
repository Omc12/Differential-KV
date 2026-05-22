"""
analysis/cliff_detector.py
Phase 10: Semantic Collapse Detector.
Predicts imminent semantic failure using multiple signals.
"""

import numpy as np
from typing import List, Dict, Any, Optional

class CliffDetector:
    def __init__(self, window_size: int = 5, kl_threshold: float = 1.5, drift_threshold: float = 0.5):
        self.window_size = window_size
        self.kl_threshold = kl_threshold
        self.drift_threshold = drift_threshold
        self.history = []

    def update(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Updates detector with new step metrics and returns collapse probability.
        metrics: { "kl": float, "entropy": float, "top_k_overlap": float, "cosine": float }
        """
        self.history.append(metrics)
        if len(self.history) > self.window_size * 3:
            self.history.pop(0)
            
        if len(self.history) < 2:
            return {"collapse_prob": 0.0, "confidence": 0.0, "status": "Stable"}

        # 1. KL Divergence Level
        kl = metrics.get("kl", 0.0)
        
        # 2. KL Acceleration (Velocity)
        kls = [h["kl"] for h in self.history]
        kl_velocity = kls[-1] - kls[-2]
        
        # 3. Entropy Instability (Variance in recent window)
        entropies = [h.get("entropy", 0.0) for h in self.history]
        entropy_var = np.var(entropies[-self.window_size:]) if len(entropies) >= self.window_size else 0.0
        
        # 4. Top-K overlap collapse
        overlap = metrics.get("top_k_overlap", 1.0)
        
        # 5. Cosine Drift
        cosine = metrics.get("cosine", 1.0)
        
        # Heuristic Collapse Probability (0.0 to 1.0)
        prob = 0.0
        if kl > self.kl_threshold: prob += 0.4
        if kl_velocity > 0.3: prob += 0.3
        if overlap < 0.3: prob += 0.4
        if entropy_var > 0.15: prob += 0.2
        if cosine < 0.7: prob += 0.3
        
        prob = min(1.0, prob)
        
        # Signal Trend
        trend = "Neutral"
        if kl_velocity > 0.1: trend = "Rising"
        elif kl_velocity < -0.1: trend = "Falling"
        
        status = "Stable"
        if prob > 0.35: status = "Warning"
        if prob > 0.75: status = "Critical"
        
        return {
            "collapse_prob": float(prob),
            "kl_velocity": float(kl_velocity),
            "entropy_instability": float(entropy_var),
            "trend": trend,
            "status": status,
            "metrics": metrics
        }

    def predict_cliff_timing(self) -> Optional[float]:
        """
        Extrapolates metrics to estimate steps until collapse (prob > 0.8).
        Very heuristic linear extrapolation of KL.
        """
        if len(self.history) < 5: return None
        
        kls = [h["kl"] for h in self.history]
        steps = np.arange(len(kls))
        
        # Linear fit to KL
        coeffs = np.polyfit(steps, kls, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        if slope <= 0: return None # No predicted collapse if KL is stable/falling
        
        # How many steps until kl reaches 3.0?
        steps_to_target = (3.0 - kls[-1]) / slope
        return float(max(0.0, steps_to_target))
