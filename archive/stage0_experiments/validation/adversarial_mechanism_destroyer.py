import torch
import random
import numpy as np
from validation.reset_environment import reset_environment

class AdversarialMechanismDestroyer:
    """
    Aggressively attempts to break cognitive mechanisms by introducing entropy,
    shuffling contexts, and performing hard resets between evaluations.
    """
    
    def __init__(self):
        self.reset_count = 0

    def purge(self):
        """Perform hard reset and CUDA purge."""
        reset_environment()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self.reset_count += 1
        print(f"Purge #{self.reset_count} completed.")

    def shuffle_context(self, prompt_chunks):
        """
        Randomly shuffles prompt chunks (if applicable) to ensure the 
        mechanism isn't just overfitting to a specific structure.
        """
        shuffled = prompt_chunks.copy()
        random.shuffle(shuffled)
        return "".join(shuffled)

    def inject_noise_to_cache(self, kv_cache, noise_level=1e-5):
        """
        Injects small amounts of noise into the KV cache to test robustness.
        """
        if kv_cache is None:
            return None
        
        new_cache = []
        for layer_kv in kv_cache:
            if layer_kv is None:
                new_cache.append(None)
                continue
            
            k, v = layer_kv
            k_noise = torch.randn_like(k) * noise_level
            v_noise = torch.randn_like(v) * noise_level
            new_cache.append((k + k_noise, v + v_noise))
            
        return new_cache

    def stress_test(self, model_forward_func, input_ids):
        """
        Executes a forward pass with randomized seed injection.
        """
        seed = random.randint(0, 1000000)
        torch.manual_seed(seed)
        return model_forward_func(input_ids)

if __name__ == "__main__":
    destroyer = AdversarialMechanismDestroyer()
    destroyer.purge()
    print("Adversarial Destroyer Ready.")
