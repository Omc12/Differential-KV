import torch
import gc
import random
import numpy as np
import os

def reset_environment():
    """
    Perform a hard reset of the environment to prevent leakage.
    """
    print("--- HARD RESET INITIATED ---")
    
    # 1. Clear Python GC
    gc.collect()
    
    # 2. Clear CUDA Memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print("CUDA cache cleared and synchronized.")
    
    # 3. Randomize Seeds (unless a fixed seed is explicitly provided for reproducibility of a single run)
    seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seeds randomized with: {seed}")
    
    # 4. Clear potential environment variables that might affect execution
    # (e.g., hidden state paths, cache directories)
    env_vars_to_clear = [
        "DIFFERENTIAL_KV_CACHE",
        "COGNITIVE_IDENTITY_PATH",
        "RESONANCE_LOG_DIR"
    ]
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
            print(f"Cleared environment variable: {var}")

    print("--- ENVIRONMENT RESET COMPLETE ---")

if __name__ == "__main__":
    reset_environment()
