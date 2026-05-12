"""
experiments/self_healing_reasoning.py
Phase 17: Self-Healing Trajectories
Investigates whether transformers can internally recover from reasoning collapse
without explicit external repair.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
from analysis.reasoning_manifold import ReasoningTrajectoryTracker
import os
import json

class SelfHealingInvestigator(ReasoningTrajectoryTracker):
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        super().__init__(model_id, device)

    def investigate_recovery(self, prompt: str, perturbation_step: int, noise_std: float = 0.5):
        """
        1. Run baseline.
        2. Run with a ONE-TIME perturbation at perturbation_step.
        3. Observe if it heals.
        """
        print(f"Investigating self-healing for: {prompt[:50]}...")
        
        # 1. Baseline
        ids_base, traj_base = self.run_generation(prompt, max_new_tokens=40)
        
        # 2. Perturbed Run
        def perturb_once_mod(l_idx, k, v):
            # We need to know which step we are at.
            # This is tricky with the current kv_modifier_fn interface because it doesn't pass the step.
            # I'll update run_generation in a local subclass if needed or use a closure.
            pass

        # Let's implement a more direct investigation loop
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids_p = input_ids.clone()
        past_key_values = None
        traj_p = []
        
        for i in range(40):
            # Apply perturbation only at the specific step
            if i == perturbation_step:
                print(f"Applying perturbation at step {i}...")
                if past_key_values:
                    # Modify the cache
                    new_cache = []
                    for k, v in past_key_values:
                        nk = k + torch.randn_like(k) * noise_std
                        nv = v + torch.randn_like(v) * noise_std
                        new_cache.append((nk, nv))
                    from transformers import DynamicCache
                    past_key_values = DynamicCache.from_legacy_cache(tuple(new_cache))

            outputs = self.model(
                input_ids=generated_ids_p[:, -1:] if i > 0 else generated_ids_p,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1:].argmax(dim=-1)
            generated_ids_p = torch.cat([generated_ids_p, next_token_id], dim=-1)
            
            # Capture hidden states for analysis
            # (Note: we'd need hooks for high-fidelity, but we can just use logits/tokens for basic healing check)
            traj_p.append(next_token_id.item())
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
        text_base = self.tokenizer.decode(ids_base[0])
        text_p = self.tokenizer.decode(generated_ids_p[0])
        
        # Calculate "Recovery Score"
        # How many of the remaining tokens match the baseline after some 're-alignment' window?
        # Or simply: does it eventually reach a similar semantic conclusion?
        
        return {
            "text_base": text_base,
            "text_p": text_p,
            "perturbation_step": perturbation_step,
            "recovered": text_base.split()[-5:] == text_p.split()[-5:] # Loose semantic check
        }

if __name__ == "__main__":
    investigator = SelfHealingInvestigator()
    prompt = "Question: If a train travels at 60 mph for 2 hours and then at 80 mph for 3 hours, what is the total distance traveled? Let's think step by step."
    
    result = investigator.investigate_recovery(prompt, perturbation_step=10, noise_std=0.2)
    
    print("\n--- SELF-HEALING RESULTS ---")
    print(f"Baseline: {result['text_base']}")
    print(f"Perturbed: {result['text_p']}")
    print(f"Recovered: {result['recovered']}")
    
    os.makedirs("results/phase17/data", exist_ok=True)
    with open("results/phase17/data/self_healing_results.json", "w") as f:
        json.dump(result, f, indent=4)
