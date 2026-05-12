"""
experiments/continual_compression_learning.py
Phase 17: Continual Compression Learning (CCL)
Runs long-duration training with repeated compression and perturbation to observe
if cognition gradually becomes more compression-resilient and stable.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import os
import json
from tqdm import tqdm

class ContinualCompressionLearner:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.resilience_history = []

    def measure_resilience(self, prompt: str, noise_std: float = 0.1):
        """
        Measures performance drop under noise.
        """
        # Baseline Loss
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs_base = self.model(**inputs, labels=inputs.input_ids)
            loss_base = outputs_base.loss.item()
            
            # Noisy Forward
            # (We'll simulate by adding noise to hidden states)
            def hook_fn(module, input, output):
                return output + torch.randn_like(output) * noise_std
                
            hooks = []
            for layer in self.model.model.layers:
                hooks.append(layer.register_forward_hook(hook_fn))
                
            outputs_noisy = self.model(**inputs, labels=inputs.input_ids)
            loss_noisy = outputs_noisy.loss.item()
            
            for h in hooks: h.remove()
            
        # Resilience = Loss Ratio
        return loss_base / (loss_noisy + 1e-9)

    def run_continual_training(self, task_sequence: List[str], noise_std=0.05):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        print(f"Starting Continual Compression Learning for {len(task_sequence)} steps...")
        
        for i, task_text in enumerate(tqdm(task_sequence)):
            # 1. Measure Resilience before training on this task
            resilience = self.measure_resilience(task_text, noise_std=noise_std)
            self.resilience_history.append(resilience)
            
            # 2. Train with compression pressure (as in Task 2)
            inputs = self.tokenizer(task_text, return_tensors="pt").to(self.device)
            
            # We train to minimize loss UNDER NOISE
            def hook_fn(module, input, output):
                return output + torch.randn_like(output) * noise_std
            
            hooks = []
            for layer in self.model.model.layers:
                hooks.append(layer.register_forward_hook(hook_fn))
                
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            for h in hooks: h.remove()
            
        return self.resilience_history

if __name__ == "__main__":
    learner = ContinualCompressionLearner()
    
    # Synthetic sequence of tasks
    tasks = [
        "The sky is blue.",
        "The grass is green.",
        "Fire is hot.",
        "Ice is cold.",
        "Water is wet.",
        "2 + 2 = 4.",
        "A square has four sides.",
        "A circle is round."
    ]
    
    history = learner.run_continual_training(tasks, noise_std=0.1)
    
    print("\nResilience History:", history)
    
    os.makedirs("results/phase17/data", exist_ok=True)
    with open("results/phase17/data/ccl_history.json", "w") as f:
        json.dump({"resilience": history}, f)
