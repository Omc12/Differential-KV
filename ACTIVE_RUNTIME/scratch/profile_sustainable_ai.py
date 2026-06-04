import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from scratch.test_sustainable_ai_prompt import PROMPT

def main():
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Initializing model {MODEL}...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    # Enable telemetry to see VRAM
    os.environ["DIFFKV_TELEMETRY"] = "1"
    
    print("\nStarting generation process...")
    t_start = time.perf_counter()
    
    session_id = "default"
    inputs = wrapper.tokenizer(PROMPT, return_tensors='pt').to(device)
    prompt_ids = inputs.input_ids[0].tolist()
    
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=len(prompt_ids))
    wrapper.model._diffkv_session_ids = [session_id]
    
    # ── Chunked prefill ──────────────────────────────────────────────────
    PREFILL_CHUNK = 512
    total_new = len(prompt_ids)
    
    print(f"Total prompt length: {total_new} tokens. Running prefill in {PREFILL_CHUNK}-token chunks...")
    
    t_prefill_start = time.perf_counter()
    outputs = None
    
    chunk_idx = 0
    for chunk_start in range(0, total_new, PREFILL_CHUNK):
        chunk_end = min(chunk_start + PREFILL_CHUNK, total_new)
        chunk = prompt_ids[chunk_start:chunk_end]
        
        chunk_tensor = torch.tensor([chunk], dtype=torch.long, device=device)
        pos_tensor = torch.arange(
            chunk_start, chunk_start + len(chunk),
            dtype=torch.long, device=device
        ).unsqueeze(0)
        
        t_chunk_start = time.perf_counter()
        if hasattr(wrapper.manager, "finalize_compressed_blocks"):
            wrapper.manager.finalize_compressed_blocks()
            
        with torch.no_grad():
            outputs = wrapper.model(
                input_ids=chunk_tensor,
                position_ids=pos_tensor,
                use_cache=True,
            )
            
        if hasattr(wrapper.manager, "compress_prefill_kv"):
            wrapper.manager.compress_prefill_kv(session_id)
            
        t_chunk = time.perf_counter() - t_chunk_start
        print(f"  Chunk {chunk_idx:2d} ({chunk_start:4d} to {chunk_end:4d}): {t_chunk:.4f} seconds")
        chunk_idx += 1
        
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"Prefill forward passes complete in {t_prefill:.4f} seconds")
    
    # ── Post-prefill barrier ──
    print("\nWaiting at post-prefill compression barrier...")
    t_barrier_start = time.perf_counter()
    if hasattr(wrapper.manager, "finalize_compressed_blocks"):
        _barrier_deadline = time.monotonic() + 30.0
        while time.monotonic() < _barrier_deadline:
            pending = getattr(wrapper.manager, "_pending_cpu_blocks", 0)
            if pending <= 0:
                break
            wrapper.manager.finalize_compressed_blocks()
            time.sleep(0.002)
    t_barrier = time.perf_counter() - t_barrier_start
    print(f"Barrier wait complete in {t_barrier:.4f} seconds")
    
    past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    
    generated = prompt_ids.copy()
    cur_pos = len(prompt_ids)
    
    print("\nRunning decode steps...")
    decode_times = []
    for step in range(16):
        next_id = torch.argmax(logits, dim=-1)
        generated.append(next_id.item())
        
        pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        input_ids = next_id.unsqueeze(0)
        
        t_step_start = time.perf_counter()
        
        if hasattr(wrapper.manager, "finalize_compressed_blocks"):
            wrapper.manager.finalize_compressed_blocks()
            
        with torch.no_grad():
            outputs = wrapper.model(
                input_ids=input_ids,
                position_ids=pos_tensor,
                past_key_values=past_kv,
                use_cache=True,
            )
        t_step = time.perf_counter() - t_step_start
        decode_times.append(t_step)
        print(f"  Decode step {step:2d}: {t_step*1000:.2f} ms")
        
        logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values
        cur_pos += 1
        
    print(f"\nAverage decode step time: {sum(decode_times)/len(decode_times)*1000:.2f} ms")
    print(f"Total execution time: {time.perf_counter() - t_start:.2f} seconds")
    
    wrapper.stop()

if __name__ == "__main__":
    main()
