"""
runtime/cognitive_state_engine.py

The "nervous system" of the Unified Cognitive Runtime.
Tracks latent drift, entropy, trajectory velocity, acceleration, curvature, 
and manifold stability to determine the "health" of the transformer's cognition.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class CognitiveHealthState:
    cognitive_health_score: float
    collapse_probability: float
    semantic_integrity: float
    manifold_stability: float
    latent_drift: float
    entropy: float
    velocity: float
    acceleration: float
    curvature: float
    attractor_distance: float
    topology_fragmentation: float

class CognitiveStateEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history: List[CognitiveHealthState] = []
        self.prev_hidden_states: Optional[List[torch.Tensor]] = None
        self.prev_velocity: Optional[List[torch.Tensor]] = None
        
        # Thresholds for various metrics
        self.collapse_threshold = config.get("collapse_threshold", 0.7)
        self.drift_threshold = config.get("drift_threshold", 0.5)

    def process_step(self, 
                     current_hidden_states: List[torch.Tensor], 
                     attentions: Optional[List[torch.Tensor]] = None,
                     target_hidden_states: Optional[List[torch.Tensor]] = None) -> CognitiveHealthState:
        """
        Analyzes the current step's hidden states and attentions to update cognitive health.
        """
        # 1. Latent Dynamics (Velocity, Acceleration, Curvature)
        # We focus on the last token's hidden states
        current_last_hidden = [h[:, -1, :].float() for h in current_hidden_states]
        
        velocity_norms = []
        accel_norms = []
        curvatures = []
        
        if self.prev_hidden_states is not None:
            velocities = [(curr - prev) for curr, prev in zip(current_last_hidden, self.prev_hidden_states)]
            velocity_norms = [torch.norm(v, p=2).item() for v in velocities]
            avg_velocity = np.mean(velocity_norms)
            
            if self.prev_velocity is not None:
                accelerations = [(curr_v - prev_v) for curr_v, prev_v in zip(velocities, self.prev_velocity)]
                accel_norms = [torch.norm(a, p=2).item() for a in accelerations]
                avg_accel = np.mean(accel_norms)
                
                for v_curr, v_prev in zip(velocities, self.prev_velocity):
                    v_curr_norm = torch.norm(v_curr, p=2)
                    v_prev_norm = torch.norm(v_prev, p=2)
                    if v_curr_norm > 1e-6 and v_prev_norm > 1e-6:
                        cos_sim = F.cosine_similarity(v_curr, v_prev, dim=-1).item()
                        curvatures.append(1.0 - cos_sim)
                    else:
                        curvatures.append(0.0)
                avg_curvature = np.mean(curvatures)
                self.prev_velocity = velocities
            else:
                avg_accel = 0.0
                avg_curvature = 0.0
                self.prev_velocity = velocities
        else:
            avg_velocity = 0.0
            avg_accel = 0.0
            avg_curvature = 0.0
            self.prev_velocity = None
            
        self.prev_hidden_states = current_last_hidden

        # 2. Drift and Alignment
        if target_hidden_states:
            target_last_hidden = [h[:, -1, :].float() for h in target_hidden_states]
            drifts = []
            for c, t in zip(current_last_hidden, target_last_hidden):
                d = torch.norm(c - t, p=2).item()
                t_norm = torch.norm(t, p=2).item()
                drifts.append(d / (t_norm + 1e-9)) # Normalized drift
            latent_drift = np.mean(drifts)
        else:
            latent_drift = 0.0

        # 3. Attention Metrics
        entropy = 0.0
        fragmentation = 0.0
        if attentions:
            entropies = []
            frags = []
            for layer_attn in attentions:
                # [1, heads, q_len, k_len] -> [heads, k_len]
                q_attn = layer_attn[0, :, -1, :]
                ent = -torch.sum(q_attn * torch.log(q_attn + 1e-9), dim=-1).mean().item()
                entropies.append(ent)
                
                k = q_attn.shape[-1]
                max_ent = np.log(k) if k > 1 else 1.0
                diffuse_heads = ((-torch.sum(q_attn * torch.log(q_attn + 1e-9), dim=-1)) > 0.5 * max_ent).float().mean().item()
                frags.append(diffuse_heads)
            entropy = np.mean(entropies)
            fragmentation = np.mean(frags)

        # 4. Integrated Health Scores
        # Simple heuristic model for prototype
        collapse_prob = np.clip((latent_drift * 1.5) + (avg_accel * 3.0) + (avg_curvature * 1.0) + (fragmentation * 0.5), 0, 1)
        health_score = 1.0 - collapse_prob
        
        # manifold stability is inversely proportional to curvature and fragmentation
        manifold_stability = np.clip(1.0 - (avg_curvature * 2.0 + fragmentation), 0, 1)
        
        # semantic integrity is high when drift is low
        semantic_integrity = np.clip(1.0 - (latent_drift * 2.0), 0, 1)
        
        # attractor distance (hypothetical)
        attractor_distance = avg_velocity # simplistic proxy

        state = CognitiveHealthState(
            cognitive_health_score=float(health_score),
            collapse_probability=float(collapse_prob),
            semantic_integrity=float(semantic_integrity),
            manifold_stability=float(manifold_stability),
            latent_drift=float(latent_drift),
            entropy=float(entropy),
            velocity=float(avg_velocity),
            acceleration=float(avg_accel),
            curvature=float(avg_curvature),
            attractor_distance=float(attractor_distance),
            topology_fragmentation=float(fragmentation)
        )
        
        self.history.append(state)
        return state

    def get_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        
        last = self.history[-1]
        return {
            "health": last.cognitive_health_score,
            "collapse_risk": last.collapse_probability,
            "drift": last.latent_drift,
            "stability": last.manifold_stability,
            "avg_health": np.mean([h.cognitive_health_score for h in self.history])
        }
