import os
import sys
import cProfile
import pstats
import torch
import time

os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

def run_decode():
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Matches the benchmark settings
    wrapper = DiffKVHFWrapper(
        MODEL, 
        config={
            "rank": 32, 
            "micro_block_size": 256, 
            "serving_mode": "balanced",
        }, 
        device=device
    )
    
    session_id = "default"
    # Create a prompt of 2048 tokens
    prompt_ids = wrapper.tokenizer("hello " * 2048, return_tensors='pt').input_ids[0].tolist()
    # Truncate exactly to 2048
    prompt_ids = prompt_ids[:2048]
    
    print(f"Initializing session with prompt length {len(prompt_ids)}...")
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=len(prompt_ids))
    wrapper.model._diffkv_session_ids = [session_id]
    
    input_ids = torch.tensor([prompt_ids], device=device)
    
    with torch.no_grad():
        outputs = wrapper.model(input_ids=input_ids, use_cache=True)
        # Finalize compressed blocks
        wrapper.manager.compress_deferred_prefill_blocks(session_id)
        wrapper.manager.finalize_compressed_blocks()
        while getattr(wrapper.manager, "_pending_cpu_blocks", 0) > 0:
            wrapper.manager.finalize_compressed_blocks()
            time.sleep(0.002)
        
    past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    cur_pos = len(prompt_ids)
    
    print("Starting profiled decode steps at 2048 context...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Pre-allocate position cache to avoid slow GPU allocations in the loop
    pos_cache = torch.arange(cur_pos + 20, dtype=torch.long, device=device)

    # Run 10 decode steps
    for step in range(10):
        next_id = torch.argmax(logits, dim=-1)
        pos_tensor = pos_cache[cur_pos].view(1, 1)
        input_ids = next_id.unsqueeze(0)
        
        with torch.no_grad():
            outputs = wrapper.model(
                input_ids=input_ids,
                position_ids=pos_tensor,
                past_key_values=past_kv,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values
        cur_pos += 1
        
    profiler.disable()
    print("Finished profiled decode steps.")
    
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(35)
    
    wrapper.stop()

def main():
    run_decode()

if __name__ == "__main__":
    main()
