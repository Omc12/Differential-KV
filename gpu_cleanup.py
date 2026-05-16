import torch
import gc

def cleanup():
    print("Forcefully clearing GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup()
