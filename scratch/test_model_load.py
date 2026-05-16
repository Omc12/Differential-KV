import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
try:
    print(f"Testing load for {model_id}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cuda",
        trust_remote_code=True
    )
    print(f"Successfully loaded {model_id} in {time.time()-start:.2f}s")
    
    # Simple test
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=5)
    print(f"Generated: {tokenizer.decode(outputs[0])}")
except Exception as e:
    print(f"Failed to load {model_id}: {e}")
