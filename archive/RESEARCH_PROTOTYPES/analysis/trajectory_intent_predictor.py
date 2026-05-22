"""
analysis/trajectory_intent_predictor.py
Phase 27: Adaptive Cognitive Routing (ACR)
Predicts future cognitive trajectory intent and instability.
"""

import torch
import numpy as np
from typing import Dict, List, Any

class TrajectoryIntentPredictor:
    def __init__(self, history_len: int = 20):
        self.history_len = history_len
        self.metric_history = []
        
    def update_history(self, metrics: Dict[str, Any]):
        self.metric_history.append(metrics)
        if len(self.metric_history) > self.history_len:
            self.metric_history.pop(0)
            
    def predict_intent(self) -> Dict[str, Any]:
        """
        Analyzes historical trends to predict future intent and risks.
        """
        if len(self.metric_history) < 5:
            return {"status": "insufficient_data"}
            
        # Extract trends
        drifts = [m.get("latent_drift", 0.0) for m in self.metric_history]
        depths = [m.get("recursion_depth", 0) for m in self.metric_history]
        branches = [m.get("branch_factor", 1.0) for m in self.metric_history]
        
        # 1. Predict Future Branch Explosion
        # Look for exponential growth in branch factor
        branch_trend = np.polyfit(range(len(branches)), branches, 1)[0]
        future_branch_explosion = branch_trend > 0.5
        
        # 2. Detect Recursion Loops
        # Look for periodic patterns in recursion depth or stationary high depth
        recursion_loop_detected = False
        if len(depths) >= 10:
            last_5 = depths[-5:]
            if all(d > 5 for d in last_5) and max(last_5) - min(last_5) <= 1:
                recursion_loop_detected = True
        
        # 3. Estimate Long-Horizon Depth
        avg_depth = np.mean(depths)
        depth_trend = np.polyfit(range(len(depths)), depths, 1)[0]
        predicted_depth = avg_depth + depth_trend * 50 # Predict 50 tokens ahead
        
        # 4. Estimate Geometry Preservation Requirements
        # Based on curvature and drift acceleration
        drift_accel = np.polyfit(range(len(drifts)), drifts, 2)[0] if len(drifts) >= 5 else 0.0
        geometry_preservation_req = "low"
        if drift_accel > 0.1 or np.mean(drifts) > 0.4:
            geometry_preservation_req = "high"
        elif np.mean(drifts) > 0.2:
            geometry_preservation_req = "medium"
            
        return {
            "future_branch_explosion": future_branch_explosion,
            "recursion_loop_detected": recursion_loop_detected,
            "predicted_horizon_depth": float(predicted_depth),
            "geometry_preservation_req": geometry_preservation_req,
            "drift_acceleration": float(drift_accel)
        }

if __name__ == "__main__":
    predictor = TrajectoryIntentPredictor()
    # Simulate some metrics
    for i in range(15):
        predictor.update_history({
            "latent_drift": 0.01 * i,
            "recursion_depth": 5 if i > 10 else i // 2,
            "branch_factor": 1.1 ** i
        })
    print(f"Prediction: {predictor.predict_intent()}")
