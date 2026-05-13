import torch
import os
import psutil

def sanity_check():
    """
    Verify that the environment is in a clean state.
    """
    print("--- RUNTIME SANITY CHECK ---")
    
    # 1. Check VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"VRAM Allocated: {allocated / 1024**2:.2f} MB")
        print(f"VRAM Reserved: {reserved / 1024**2:.2f} MB")
        
        if allocated > 100 * 1024**2: # More than 100MB is suspicious for a "clean" state
            print("WARNING: Significant VRAM already allocated!")
            return False
    
    # 2. Check for leftover process-local cache files (if any known)
    # This is a placeholder for project-specific cache checks
    suspicious_dirs = ["session_checkpoints", "distributed_identities", "memory"]
    for d in suspicious_dirs:
        path = os.path.join(os.getcwd(), d)
        if os.path.exists(path) and any(os.scandir(path)):
            print(f"INFO: Data found in {d}/. Ensure this is expected or cleared.")

    # 3. Check memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Process RAM usage: {mem_info.rss / 1024**2:.2f} MB")

    print("--- SANITY CHECK PASSED ---")
    return True

if __name__ == "__main__":
    if not sanity_check():
        exit(1)
