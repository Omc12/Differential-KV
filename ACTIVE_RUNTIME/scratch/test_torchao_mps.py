import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_, Int8WeightOnlyConfig

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

print("Applying torchao Int8WeightOnlyConfig...")
try:
    quantize_(model, Int8WeightOnlyConfig())
    print("Quantization applied successfully!")
except Exception as e:
    print("Quantization failed:", e)
    import sys; sys.exit(1)

print("Moving model to MPS...")
try:
    model = model.to("mps")
    print("Moved to MPS successfully!")
except Exception as e:
    print("Move to MPS failed:", e)
    import sys; sys.exit(1)

print("Running a token forward pass...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("mps")
with torch.no_grad():
    out = model(**inputs)
    logits = out.logits
    print("Forward pass logits shape:", logits.shape)
    has_nan = torch.isnan(logits).any().item()
    print("Logits contain NaNs:", has_nan)
