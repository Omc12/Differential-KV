import torch
from transformers import AutoTokenizer, AutoConfig
import os

def check_env():
    print("Checking CUDA...")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\nChecking Model Checkpoint...")
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    try:
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        print(f"Model config found: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        print(f"Tokenizer found: {model_id}")
    except Exception as e:
        print(f"Error checking model: {e}")

if __name__ == "__main__":
    check_env()
