import sys
sys.path.insert(0, ".")
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from native_core.kv_runtime_manager import KVRuntimeManager
from runtime.diffkv_attention import apply_diffkv_attention_patch

print("=== PHASE 4: LONG-CONTEXT STABILITY TEST (NEEDLE IN A HAYSTACK) ===")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
model.eval()

# Normal block size, Rank 16 (standard compression)
mgr = KVRuntimeManager(
    num_layers=model.config.num_hidden_layers,
    heads=model.config.num_key_value_heads,
    head_dim=model.config.hidden_size // model.config.num_attention_heads,
    device="cuda"
)
mgr.block_size = 64
mgr.rank = 16
apply_diffkv_attention_patch(model, mgr)

session_id = "long_context_test"

# Create a haystack
haystack = "The Roman Empire was a vast and powerful civilization. " * 500
needle = "The secret password to access the emperor's vault is 'JupiterAscends99'. "
haystack2 = "Economic factors played a huge role in the collapse. " * 500

prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{haystack}{needle}{haystack2}\n\nWhat is the secret password to access the emperor's vault?<|im_end|>\n<|im_start|>assistant\nThe secret password to access the emperor's vault is"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

model._diffkv_session_ids = [session_id]
position_ids = torch.arange(0, input_ids.shape[1], device="cuda").unsqueeze(0)

print(f"\n[1] Running Prefill ({input_ids.shape[1]} tokens)...")
t0 = time.time()
with torch.no_grad():
    out = model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
t1 = time.time()
print(f"Prefill done in {t1-t0:.2f}s")

next_token = out.logits[0, -1, :].argmax().item()
generated = [next_token]

print(f"[2] Running Decode...")
for step in range(25):
    seq_len = mgr.get_seq_len(session_id, layer_idx=0)
    
    decode_input = torch.tensor([[next_token]], device="cuda")
    attention_mask = torch.ones((1, seq_len + 1), dtype=torch.long, device="cuda")
    decode_position_ids = torch.tensor([[seq_len]], device="cuda")
    
    with torch.no_grad():
        out2 = model(
            input_ids=decode_input,
            attention_mask=attention_mask,
            position_ids=decode_position_ids,
            use_cache=True
        )
        
    next_token = out2.logits[0, -1, :].argmax().item()
    if next_token == tokenizer.eos_token_id or next_token == 151645: # im_end
        break
    generated.append(next_token)

print("\n--- GENERATED TEXT ---")
print(tokenizer.decode(generated))

print("\n--- PERFORMANCE & METRICS ---")
print(f"Total Blocks Compressed: {mgr.total_compressions}")
if mgr.total_compressions > 0:
    avg_cos_sim = mgr.total_cosine_sim / mgr.total_compressions
    avg_norm_drift = mgr.total_norm_drift / mgr.total_compressions
    print(f"Average Cosine Similarity: {avg_cos_sim:.4f} (1.0 is perfect)")
    print(f"Average Norm Drift: {avg_norm_drift:.4f} (0.0 is perfect)")

print(f"VRAM Saved: {mgr.vram_saved_bytes / 1024 / 1024:.2f} MB")
