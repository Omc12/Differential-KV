import sys
sys.path.insert(0, ".")
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.kv_runtime_manager import KVRuntimeManager
from runtime.diffkv_attention import apply_diffkv_attention_patch

print("=== PHASE 4: PROFILING SPARSE RECONSTRUCTION ===")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
model.eval()

mgr = KVRuntimeManager(
    num_layers=model.config.num_hidden_layers,
    heads=model.config.num_key_value_heads,
    head_dim=model.config.hidden_size // model.config.num_attention_heads,
    device="cuda"
)
mgr.block_size = 64
mgr.rank = 16
apply_diffkv_attention_patch(model, mgr)

session_id = "prof_session"
prompt = "Explain quantum physics. " * 50
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

model._diffkv_session_ids = [session_id]
position_ids = torch.arange(0, input_ids.shape[1], device="cuda").unsqueeze(0)

print(f"\n[1] Running Prefill ({input_ids.shape[1]} tokens)...")
with torch.no_grad():
    out = model(input_ids=input_ids, position_ids=position_ids, use_cache=True)

next_token = out.logits[0, -1, :].argmax().item()

decode_input = torch.tensor([[next_token]], device="cuda")
seq_len = mgr.get_seq_len(session_id, layer_idx=0)
attention_mask = torch.ones((1, seq_len + 1), dtype=torch.long, device="cuda")
decode_position_ids = torch.tensor([[seq_len]], device="cuda")

print(f"[2] Running Profiler on 5 Decode Steps...")

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for i in range(5):
        with torch.no_grad():
            _ = model(
                input_ids=decode_input,
                attention_mask=attention_mask,
                position_ids=decode_position_ids,
                use_cache=True
            )

prof.export_chrome_trace("phase4_reconstruction_trace.json")
print("Trace saved to phase4_reconstruction_trace.json")

print("\n--- KERNEL BREAKDOWN ---")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

print("\n--- MEMORY ALLOCATION CHURN ---")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
