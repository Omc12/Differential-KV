# HuggingFace Integration Example
from integrations.huggingface_runtime_adapter import DiffKVHFAdapter
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"
model = DiffKVHFAdapter.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
