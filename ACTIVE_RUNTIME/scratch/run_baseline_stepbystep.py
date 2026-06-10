from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32, device_map="cpu")
model.eval()

prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

past_key_values = None
current_tokens = tokens

# Run prefill (first step)
with torch.no_grad():
    outputs = model(torch.tensor([current_tokens]), use_cache=True)
logits = outputs.logits[0, -1, :]
past_key_values = outputs.past_key_values

# Print prefill top predictions
val_p, idx_p = torch.topk(logits, k=5)
print("\nPrefill Phase Top predictions:")
for i in range(5):
    print(f"  {i}: \"{tokenizer.decode([idx_p[i].item()])}\" (id: {idx_p[i].item()}, logit: {val_p[i].item():.4f})")

# Step 0: Feed "The"
next_token = idx_p[0].item()
print(f"\nStep 0 Input: {tokenizer.decode([next_token])} (id: {next_token})")
with torch.no_grad():
    outputs = model(torch.tensor([[next_token]]), past_key_values=past_key_values, use_cache=True)
logits = outputs.logits[0, -1, :]
past_key_values = outputs.past_key_values

# Print Step 0 top predictions
val_p, idx_p = torch.topk(logits, k=5)
print("Step 0 Top predictions:")
for i in range(5):
    print(f"  {i}: \"{tokenizer.decode([idx_p[i].item()])}\" (id: {idx_p[i].item()}, logit: {val_p[i].item():.4f})")

# Step 1: Feed " capital"
next_token = idx_p[0].item()
print(f"\nStep 1 Input: {tokenizer.decode([next_token])} (id: {next_token})")
with torch.no_grad():
    outputs = model(torch.tensor([[next_token]]), past_key_values=past_key_values, use_cache=True)
logits = outputs.logits[0, -1, :]
past_key_values = outputs.past_key_values

# Print Step 1 top predictions
val_p, idx_p = torch.topk(logits, k=5)
print("Step 1 Top predictions:")
for i in range(5):
    print(f"  {i}: \"{tokenizer.decode([idx_p[i].item()])}\" (id: {idx_p[i].item()}, logit: {val_p[i].item():.4f})")
