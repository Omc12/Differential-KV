import torch
import os

def verify():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Check if model directory exists in common locations
    # Usually in ~/.cache/huggingface/hub/
    cache_path = os.path.expanduser("~/.cache/huggingface/hub/")
    print(f"HF Cache exists: {os.path.exists(cache_path)}")

if __name__ == "__main__":
    verify()
