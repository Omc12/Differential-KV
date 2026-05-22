import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Any
from runtime.recursive_stability_controller import RecursiveStabilityController
from analysis.recursive_attractor_dynamics import RecursiveAttractorDynamics
from analysis.resonance_decay_analysis import ResonanceDecayAnalysis

class InfiniteHorizonReasoningSim:
    """
    Simulates ultra-long reasoning horizons to test Recursive Cognitive Resonance.
    """
    def __init__(self, 
                 d_model: int = 512, 
                 n_layers: int = 12,
                 max_steps: int = 1000):
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_steps = max_steps
        
        self.controller = RecursiveStabilityController(d_model, n_layers)
        self.dynamics = RecursiveAttractorDynamics()
        self.decay_analyzer = ResonanceDecayAnalysis(max_steps)
        
        # Simulated "Ideal" reasoning trajectory (a slow drift in latent space)
        self.ideal_trajectory = torch.randn(max_steps, d_model)
        for i in range(1, max_steps):
            self.ideal_trajectory[i] = 0.999 * self.ideal_trajectory[i-1] + 0.001 * torch.randn(d_model)

    def run_experiment(self, use_rcr: bool = True):
        print(f"Starting Infinite Horizon Reasoning (RCR={'ON' if use_rcr else 'OFF'})...")
        
        # State tracking
        # Initialize latents from the start of the ideal trajectory to avoid immediate collapse
        current_latents = [self.ideal_trajectory[0].detach().clone() for _ in range(self.n_layers)]
        history = [[] for _ in range(self.n_layers)]
        
        survival_steps = 0
        
        start_time = time.time()
        
        for step in range(self.max_steps):
            # Simulate one reasoning step per layer
            for l in range(self.n_layers):
                # Add noise and basin-escape force
                noise = torch.randn(self.d_model) * 0.05
                # Basin escape force: if state gets too far from ideal, it accelerates away
                dist = torch.norm(current_latents[l] - self.ideal_trajectory[step])
                if dist > 0.5:
                    escape_force = (current_latents[l] - self.ideal_trajectory[step]) * 0.1
                    current_latents[l] = current_latents[l] + escape_force
                
                current_latents[l] = current_latents[l] + noise
                
                # Apply Recursive Cognitive Resonance
                if use_rcr:
                    current_latents[l] = self.controller.stabilize_step(l, current_latents[l])
                
                history[l].append(current_latents[l].detach().clone())

            # Check for collapse (if any layer loses coherence with ideal trajectory)
            l_idx = self.n_layers // 2 # Check middle layer
            coherence = torch.nn.functional.cosine_similarity(
                current_latents[l_idx], self.ideal_trajectory[step], dim=0
            ).item()
            
            if use_rcr:
                self.dynamics.log_step(coherence, 0.05 if step % 16 == 0 else 0.0, 0.0) # Simplified logging
            
            if coherence < 0.3: # Collapse threshold
                print(f"Collapse detected at step {step}")
                break
            
            survival_steps += 1

        end_time = time.time()
        elapsed = end_time - start_time
        
        # Final analysis
        for l in range(self.n_layers):
            self.decay_analyzer.measure_decay(l, history[l])
            
        return {
            "survival_steps": survival_steps,
            "survival_ratio": survival_steps / self.max_steps,
            "throughput": self.max_steps / elapsed,
            "metrics": self.controller.get_metrics() if use_rcr else {}
        }

if __name__ == "__main__":
    sim = InfiniteHorizonReasoningSim(max_steps=500)
    
    print("--- Baseline (No RCR) ---")
    baseline = sim.run_experiment(use_rcr=False)
    print(f"Baseline Survival: {baseline['survival_steps']} steps")
    
    print("\n--- RCR Enabled ---")
    rcr_result = sim.run_experiment(use_rcr=True)
    print(f"RCR Survival: {rcr_result['survival_steps']} steps")
    
    gain = (rcr_result['survival_steps'] / (baseline['survival_steps'] + 1e-6))
    print(f"\nReasoning Survival Gain: {gain:.2f}x")
