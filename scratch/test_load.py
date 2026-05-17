import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("Checking available VRAM...")
print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

model_id = "Qwen/Qwen2.5-7B-Instruct"

try:
    print(f"Trying to load {model_id} in float16...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    print(f"Successfully loaded in {time.time() - start:.2f} seconds!")
    print(f"CUDA memory allocated after load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
except Exception as e:
    print(f"Failed to load: {e}")
