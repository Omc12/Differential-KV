import torch
import numpy as np
import time
from typing import Dict, List
from runtime.global_sync_manager import GlobalSyncManager
from analysis.cross_layer_resonance import ResonanceMetrics

class ResonanceSurvivalEval:
    """
    Evaluates how long reasoning survives under aggressive compression 
    using Cross-Layer Geometric Resonance (CLGR).
    """
    def __init__(self, model_name: str = "Qwen2-1.5B", compression_ratios: List[int] = [8, 12, 16, 20]):
        self.model_name = model_name
        self.compression_ratios = compression_ratios
        self.results = {}
        
    def run_eval(self, num_layers: int = 24, hidden_dim: int = 1536):
        print(f"Starting Resonance Survival Evaluation for {self.model_name}...")
        
        for ratio in self.compression_ratios:
            print(f"  Testing Compression Ratio: {ratio}x")
            sync_manager = GlobalSyncManager(num_layers, hidden_dim)
            
            # Simulate a reasoning task over 100 steps
            survival_steps = 0
            total_steps = 100
            overhead_acc = []
            
            for step in range(total_steps):
                start_time = time.time()
                
                # Simulate hidden states with increasing noise based on compression ratio
                # At higher ratios, noise increases and resonance starts to fracture
                noise_scale = (ratio / 10.0) * (step / total_steps)
                
                # Base states (idealized resonance - highly correlated across layers)
                shared_base = torch.randn(1, hidden_dim)
                base_states = [shared_base + torch.randn(1, hidden_dim) * 0.05 for _ in range(num_layers)]
                
                # Add drift and noise
                states = []
                for i, s in enumerate(base_states):
                    # Drift propagates through layers, but CLGR tries to stabilize it
                    layer_noise = noise_scale * (1.0 + 0.02 * i)
                    states.append(s + torch.randn_like(s) * layer_noise)
                
                # Sync step
                metrics = sync_manager.step(states)
                
                # Measure overhead
                overhead = (time.time() - start_time) * 1000 # ms
                overhead_acc.append(overhead)
                
                if metrics.coherence_score > 0.4:
                    survival_steps += 1
                else:
                    # Collapse!
                    if step > 10: # Early collapse is possible at high ratios
                         break
            
            survival_rate = (survival_steps / total_steps) * 100
            self.results[ratio] = {
                "survival_rate": survival_rate,
                "avg_coherence": np.mean(sync_manager.sync_history),
                "overhead_ms": np.mean(overhead_acc)
            }
            print(f"    Survival Rate: {survival_rate:.2f}%")
            
        return self.results

if __name__ == "__main__":
    eval = ResonanceSurvivalEval()
    results = eval.run_eval()
    print("\nFinal Results:", results)
