"""
analysis/trajectory_monitor.py
Phase 15: Active Cognitive Trajectory Repair (ACTR)
Implements high-fidelity trajectory monitoring for transformer reasoning manifolds.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from analysis.reasoning_manifold import ReasoningTrajectoryTracker

class CognitiveTrajectoryMonitor(ReasoningTrajectoryTracker):
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        # We don't necessarily want to load the model every time if we just want the logic,
        # but for Phase 15 we need the actual tracker capabilities.
        super().__init__(model_id, device)
        self.history = []
        self.prev_hidden_states = None
        self.prev_velocity = None

    def compute_advanced_metrics(self, current_hidden_states: List[torch.Tensor], 
                                 target_hidden_states: Optional[List[torch.Tensor]] = None,
                                 attentions: Optional[List[torch.Tensor]] = None):
        """
        Computes the Phase 15 metrics.
        current_hidden_states: List of [1, seq, dim] for each layer
        target_hidden_states: Optional baseline for drift calculation
        """
        metrics = {}
        
        # 1. Latent Velocity and Acceleration
        # Velocity v_t = h_t - h_{t-1}
        # Acceleration a_t = v_t - v_{t-1}
        
        # We care about the last token in the sequence for online monitoring
        current_last_hidden = [h[:, -1, :].float() for h in current_hidden_states]
        
        if self.prev_hidden_states is not None:
            velocities = [(curr - prev) for curr, prev in zip(current_last_hidden, self.prev_hidden_states)]
            velocity_norms = [torch.norm(v, p=2).item() for v in velocities]
            metrics["latent_velocity"] = np.mean(velocity_norms)
            
            if self.prev_velocity is not None:
                accelerations = [(curr_v - prev_v) for curr_v, prev_v in zip(velocities, self.prev_velocity)]
                accel_norms = [torch.norm(a, p=2).item() for a in accelerations]
                metrics["latent_acceleration"] = np.mean(accel_norms)
                
                # Trajectory Curvature: 1 - cosine similarity between successive velocity vectors
                curvatures = []
                for v_curr, v_prev in zip(velocities, self.prev_velocity):
                    # Avoid division by zero
                    v_curr_norm = torch.norm(v_curr, p=2)
                    v_prev_norm = torch.norm(v_prev, p=2)
                    if v_curr_norm > 1e-6 and v_prev_norm > 1e-6:
                        cos_sim = F.cosine_similarity(v_curr, v_prev, dim=-1).item()
                        curvatures.append(1.0 - cos_sim)
                    else:
                        curvatures.append(0.0)
                metrics["trajectory_curvature"] = np.mean(curvatures)
            else:
                metrics["latent_acceleration"] = 0.0
                metrics["trajectory_curvature"] = 0.0
                
            self.prev_velocity = velocities
        else:
            metrics["latent_velocity"] = 0.0
            metrics["latent_acceleration"] = 0.0
            metrics["trajectory_curvature"] = 0.0
            self.prev_velocity = None
            
        self.prev_hidden_states = current_last_hidden

        # 2. Drift and Alignment (if target is provided)
        if target_hidden_states:
            target_last_hidden = [h[:, -1, :].float() for h in target_hidden_states]
            drifts = [torch.norm(c - t, p=2).item() for c, t in zip(current_last_hidden, target_last_hidden)]
            alignments = [F.cosine_similarity(c, t, dim=-1).item() for c, t in zip(current_last_hidden, target_last_hidden)]
            metrics["hidden_drift"] = np.mean(drifts)
            metrics["cosine_alignment"] = np.mean(alignments)
        else:
            metrics["hidden_drift"] = 0.0
            metrics["cosine_alignment"] = 1.0

        # 3. Attention Metrics
        if attentions:
            entropies = []
            fragmentations = []
            
            for layer_attn in attentions:
                # layer_attn: [1, heads, q_len, k_len]
                q_attn = layer_attn[0, :, -1, :] # [heads, k_len]
                
                # Entropy
                ent = -torch.sum(q_attn * torch.log(q_attn + 1e-9), dim=-1).mean().item()
                entropies.append(ent)
                
                # Fragmentation: high entropy + high diffusion
                # Here: count heads that are "diffuse" (entropy > threshold)
                k = q_attn.shape[-1]
                max_ent = np.log(k)
                diffuse_heads = ((-torch.sum(q_attn * torch.log(q_attn + 1e-9), dim=-1)) > 0.5 * max_ent).float().mean().item()
                fragmentations.append(diffuse_heads)
                
            metrics["attention_entropy"] = np.mean(entropies)
            metrics["attention_fragmentation"] = np.mean(fragmentations)
        
        # 4. Phase 15 Specialized Metrics
        metrics["basin_escape_score"] = self._compute_basin_escape(metrics)
        metrics["cognitive_stability_score"] = self._compute_stability_score(metrics)
        
        return metrics

    def _compute_basin_escape(self, metrics: Dict) -> float:
        drift = metrics.get("hidden_drift", 0)
        accel = metrics.get("latent_acceleration", 0)
        curvature = metrics.get("trajectory_curvature", 0)
        
        # Empirical weights for escape detection
        score = (drift * 2.0) + (accel * 5.0) + (curvature * 1.5)
        return float(np.clip(score, 0, 1))

    def _compute_stability_score(self, metrics: Dict) -> float:
        escape = metrics.get("basin_escape_score", 0)
        frag = metrics.get("attention_fragmentation", 0)
        
        stability = 1.0 - (escape * 0.6 + frag * 0.4)
        return float(np.clip(stability, 0, 1))

    def monitor_step(self, hidden_states, target_hidden=None, attentions=None):
        metrics = self.compute_advanced_metrics(hidden_states, target_hidden, attentions)
        self.history.append(metrics)
        return metrics

if __name__ == "__main__":
    # Test with dummy data
    monitor = CognitiveTrajectoryMonitor()
    h1 = [torch.randn(1, 1, 768) for _ in range(12)]
    h2 = [h + 0.1 for h in h1]
    h3 = [h + 0.5 for h in h2]
    
    print("Step 1:", monitor.monitor_step(h1))
    print("Step 2:", monitor.monitor_step(h2))
    print("Step 3:", monitor.monitor_step(h3))
