import os
import sys
import gc
import psutil

def print_mem(label):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 ** 2)
    vms = process.memory_info().vms / (1024 ** 2)
    
    import torch
    mps_alloc = 0.0
    mps_driver = 0.0
    if hasattr(torch, "mps") and torch.mps.is_available():
        try:
            mps_alloc = torch.mps.current_allocated_memory() / (1024 ** 2)
            mps_driver = torch.mps.driver_allocated_memory() / (1024 ** 2)
        except Exception:
            pass
            
    print(f"[{label}]")
    print(f"  CPU RSS (Resident):   {rss:8.2f} MB")
    print(f"  CPU VMS (Virtual):    {vms:8.2f} MB")
    print(f"  MPS PyTorch Alloc:    {mps_alloc:8.2f} MB")
    print(f"  MPS Driver Total:     {mps_driver:8.2f} MB")
    print("-" * 50)

print_mem("Baseline: before import torch")

import torch
print_mem("After import torch")

from transformers import AutoTokenizer, AutoModelForCausalLM
print_mem("After import AutoModelForCausalLM")

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print_mem("After loading tokenizer")

# Load model weights on CPU first
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
print_mem("After loading model on CPU")

# Move to MPS
model = model.to("mps")
print_mem("After moving model to MPS")

# Run a dummy inference of 512 tokens
input_ids = torch.randint(0, 1000, (1, 512), device="mps")
with torch.no_grad():
    out = model(input_ids)
print_mem("After running 512 token dummy inference")

# Clear cache
del out
gc.collect()
torch.mps.empty_cache()
print_mem("After empty_cache() and gc.collect()")

# Run another inference but delete intermediates
input_ids = torch.randint(0, 1000, (1, 1024), device="mps")
with torch.no_grad():
    out = model(input_ids)
del out
gc.collect()
torch.mps.empty_cache()
print_mem("After 1024 token inference + empty_cache() + gc.collect()")
