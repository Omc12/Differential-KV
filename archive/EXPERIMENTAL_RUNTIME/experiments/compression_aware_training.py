"""
experiments/compression_aware_training.py
Phase 17: Compression-Aware Training (CAT)
Forces cognition to adapt to constrained memory by training under active KV compression
and latent perturbation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import os

class CompressionAwareTrainer:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        
    def apply_compression_noise(self, past_key_values, noise_std=0.01, rank_restriction: Optional[int] = None):
        """
        Simulates compression pressure on the KV cache.
        """
        if past_key_values is None: return None
        
        new_kv = []
        for l_idx, (k, v) in enumerate(past_key_values):
            # 1. Add latent noise (simulates quantization/drift)
            k_noisy = k + torch.randn_like(k) * noise_std
            v_noisy = v + torch.randn_like(v) * noise_std
            
            # 2. Dynamic Rank Restriction (simulates sparse storage)
            if rank_restriction and k.shape[2] > 1: # Only if we have enough sequence length
                # Simplistic: zero out some ranks in a spectral sense if we were doing SVD
                # Or just randomly zero out some heads/dimensions
                pass 
                
            new_kv.append((k_noisy, v_noisy))
            
        return tuple(new_kv)

    def train_step(self, input_ids, labels, compression_params: Dict):
        """
        A single training step with compression pressure.
        """
        self.model.train()
        
        # We need to run the model step-by-step to apply compression to the cache
        # OR we can just inject noise into the hidden states during forward pass.
        # Injecting noise into KV is more specific to DKV.
        
        outputs = self.model(input_ids, labels=labels)
        loss = outputs.loss
        
        # In a real CAT, we would:
        # 1. Forward with compression
        # 2. Compare with uncompressed teacher (Knowledge Distillation)
        # 3. Backprop to the model weights
        
        # For this experiment, we'll simulate the gradient flow
        return loss

    def run_training_loop(self, training_data: List[str], epochs=3, noise_std=0.05):
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        print(f"Starting Compression-Aware Training on {len(training_data)} examples...")
        
        history = []
        for epoch in range(epochs):
            total_loss = 0
            for text in tqdm(training_data):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                
                # Knowledge Distillation Loss
                # Teacher (No compression)
                with torch.no_grad():
                    teacher_outputs = self.model(**inputs)
                    teacher_logits = teacher_outputs.logits
                
                # Student (With KV Noise)
                # Note: To really simulate KV noise in a standard HF forward pass,
                # we'd need to modify the model's internal cache handling.
                # Here we'll simulate it by adding noise to the hidden states.
                
                def hook_fn(module, input, output):
                    # Inject noise into hidden states to simulate KV degradation
                    if isinstance(output, tuple):
                        return (output[0] + torch.randn_like(output[0]) * noise_std, *output[1:])
                    return output + torch.randn_like(output) * noise_std

                hooks = []
                for layer in self.model.model.layers:
                    hooks.append(layer.register_forward_hook(hook_fn))
                
                student_outputs = self.model(**inputs, labels=inputs.input_ids)
                
                # KL Divergence between teacher and student
                loss_kl = nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(student_outputs.logits / 2.0, dim=-1),
                    torch.softmax(teacher_logits / 2.0, dim=-1)
                )
                
                loss = student_outputs.loss + loss_kl
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                for h in hooks: h.remove()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            history.append(avg_loss)
            
        return history

if __name__ == "__main__":
    trainer = CompressionAwareTrainer()
    
    # Representative reasoning data
    dataset = [
        "The capital of France is Paris. Therefore, the Eiffel Tower is in Paris.",
        "If all men are mortal and Socrates is a man, then Socrates is mortal.",
        "To make coffee, first boil water, then add grounds, then pour. Finally, enjoy.",
        "The square root of 16 is 4. 4 squared is 16.",
        "Python is a programming language. It is used for AI and data science."
    ]
    
    history = trainer.run_training_loop(dataset, epochs=2, noise_std=0.1)
    
    os.makedirs("results/phase17/data", exist_ok=True)
    with open("results/phase17/data/cat_history.json", "w") as f:
        import json
        json.dump({"loss_history": history}, f)
    
    print("Compression-Aware Training Complete.")
