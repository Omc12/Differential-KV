"""
Isolated diagnostic for the KVRuntimeManager + DiffKV Attention forward pass.
Runs a single forward pass through the model with patched attention and prints
exactly where it hangs or errors.
"""
import sys
sys.path.insert(0, ".")

import torch
import time

print("=== PHASE 3 STEP DIAGNOSTIC ===")

# 1. Load model
print("\n[1] Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
model.eval()
print(f"    Model loaded. Layers: {model.config.num_hidden_layers}")

# 2. Init KVRuntimeManager
print("\n[2] Init KVRuntimeManager...")
from runtime.kv_runtime_manager import KVRuntimeManager
mgr = KVRuntimeManager(
    num_layers=model.config.num_hidden_layers,
    heads=model.config.num_key_value_heads,
    head_dim=model.config.hidden_size // model.config.num_attention_heads,
    device="cuda"
)
print(f"    KVRuntimeManager ready. heads={mgr.heads}, head_dim={mgr.head_dim}")

# 3. Apply attention patch
print("\n[3] Applying diffkv_attention patch...")
from runtime.diffkv_attention import apply_diffkv_attention_patch
apply_diffkv_attention_patch(model, mgr)
print("    Patch applied.")

# 4. Run PREFILL
session_id = "test-session"
prompt = "Hello, my name is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
print(f"\n[4] Running PREFILL | prompt={repr(prompt)} | tokens={input_ids.shape[1]}")

model._diffkv_session_ids = [session_id]
position_ids = torch.arange(0, input_ids.shape[1], device="cuda").unsqueeze(0)

t0 = time.time()
with torch.no_grad():
    out = model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
t1 = time.time()
print(f"    PREFILL done in {t1-t0:.3f}s | logits shape: {out.logits.shape}")

# 5. Sample next token
next_token = out.logits[0, -1, :].argmax().item()
next_text = tokenizer.decode([next_token])
print(f"    Next token: {next_token} = '{next_text}'")

# 6. Check KVRuntimeManager state
seq_len = mgr.get_seq_len(session_id, layer_idx=0)
print(f"\n[5] KVRuntimeManager state:")
print(f"    Sequence length (layer 0): {seq_len}")
blocks = mgr.session_blocks[session_id][0]
print(f"    Blocks: {len(blocks)}")
for b_idx, block in enumerate(blocks):
    print(f"      Block {b_idx}: anchor_idx={block.anchor_idx}, U={block.U.shape if block.U is not None else None}, active_k={block.active_k.shape if block.active_k is not None else None}")

# 7. Run DECODE step
print(f"\n[6] Running DECODE step...")
decode_input = torch.tensor([[next_token]], device="cuda")
attention_mask = torch.ones((1, seq_len + 1), dtype=torch.long, device="cuda")
decode_position_ids = torch.tensor([[seq_len]], device="cuda")

model._diffkv_session_ids = [session_id]

t0 = time.time()
with torch.no_grad():
    out2 = model(
        input_ids=decode_input,
        attention_mask=attention_mask,
        position_ids=decode_position_ids,
        use_cache=True
    )
t1 = time.time()
print(f"    DECODE done in {t1-t0:.3f}s | logits shape: {out2.logits.shape}")

next_token2 = out2.logits[0, -1, :].argmax().item()
print(f"    Next token: {next_token2} = '{tokenizer.decode([next_token2])}'")

# 8. VRAM report
print(f"\n[7] VRAM Report:")
print(f"    KVRuntimeManager compressions: {mgr.total_compressions}")
print(f"    KVRuntimeManager VRAM saved: {mgr.vram_saved_bytes/1024:.1f} KB")
print(f"    GPU VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.3f} GB")

print("\n=== DIAGNOSTIC COMPLETE ===")
