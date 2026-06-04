import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

PROMPT = "Write a short poem about the ocean."

def main():
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Initializing model {MODEL}...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    print("\nRunning prefill...")
    t0 = time.perf_counter()
    
    # Run the prompt through generation with step-by-step timing
    # We will manually implement the generation loop to print timings
    session_id = "default"
    inputs = wrapper.tokenizer(PROMPT, return_tensors='pt').to(device)
    prompt_ids = inputs.input_ids[0].tolist()
    
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=len(prompt_ids))
    wrapper.model._diffkv_session_ids = [session_id]
    
    input_ids = torch.tensor([prompt_ids], device=device)
    
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = wrapper.model(input_ids=input_ids, use_cache=True)
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"Prefill time: {t_prefill:.4f} seconds")
    
    # Wait for SVD if any
    if hasattr(wrapper.manager, "finalize_compressed_blocks"):
        wrapper.manager.finalize_compressed_blocks()
    
    past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    
    generated = prompt_ids.copy()
    cur_pos = len(prompt_ids)
    
    decode_times = []
    for step in range(16):
        next_id = torch.argmax(logits, dim=-1)
        generated.append(next_id.item())
        
        pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        input_ids = next_id.unsqueeze(0)
        
        t_step_start = time.perf_counter()
        with torch.no_grad():
            # Test with or without capture_to_graph if available
            is_mps = (device == "mps")
            if is_mps and hasattr(torch, "mps") and hasattr(torch.mps, "capture_to_graph"):
                with torch.mps.capture_to_graph():
                    outputs = wrapper.model(
                        input_ids=input_ids,
                        position_ids=pos_tensor,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
            else:
                outputs = wrapper.model(
                    input_ids=input_ids,
                    position_ids=pos_tensor,
                    past_key_values=past_kv,
                    use_cache=True,
                )
        t_step = time.perf_counter() - t_step_start
        decode_times.append(t_step)
        print(f"Decode step {step:2d}: {t_step*1000:.2f} ms")
        
        logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values
        cur_pos += 1
        
    print(f"\nAverage decode step time: {sum(decode_times)/len(decode_times)*1000:.2f} ms")
    wrapper.stop()

if __name__ == "__main__":
    main()
