import os
import sys
import torch
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Construct a 1500-token prompt
    prompt = "This is a test prompt to verify the correct application of post-RoPE anchors and log-sum-exp combinations. " * 70
    
    print("1. Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model_baseline = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
    )
    model_baseline.eval()
    
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = encoded.input_ids[0].tolist()
    prefill_len = len(prompt_ids)
    
    print(f"Prefill length: {prefill_len} tokens")
    
    print("Running baseline prefill...")
    with torch.no_grad():
        outputs_baseline = model_baseline(input_ids=encoded.input_ids, use_cache=True)
    past_b = outputs_baseline.past_key_values
    logits_b = outputs_baseline.logits[0, -1, :].cpu().float()
    
    print("\n2. Loading patched model...")
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 16},  # Small block size to force compression early
        device=device,
    )
    
    print("Running patched model prefill...")
    session_id = "default"
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=prefill_len)
    wrapper.model._diffkv_session_ids = [session_id]
    
    with torch.no_grad():
        outputs_patched = wrapper.model(input_ids=encoded.input_ids, use_cache=True)
        
        # Trigger actual SVD compression of history blocks!
        print("Triggering SVD compression for prefill history blocks...")
        wrapper.manager.compress_deferred_prefill_blocks(session_id)
        
        # Wait for compression threads to finish
        t_start = time.time()
        while getattr(wrapper.manager, "_pending_cpu_blocks", 0) > 0:
            time.sleep(0.05)
            wrapper.manager.finalize_compressed_blocks()
            if time.time() - t_start > 15.0:  # Safety timeout
                print("Timeout waiting for SVD compression.")
                break
        wrapper.manager.finalize_compressed_blocks()
        
    logits_p = outputs_patched.logits[0, -1, :].cpu().float()
    
    # Check block state
    summary = wrapper.manager.get_streaming_summary(session_id)
    print(f"Compressed blocks summary: {summary}")
    
    diff = (logits_p - logits_b).abs()
    print(f"\n--- Prefill/Step 0 comparison: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f} ---")
    
    print("\n============================================================")
    print("  Autoregressive Generation Comparison - 10 Steps")
    print("============================================================")
    
    cur_pos = prefill_len
    generated_b = []
    generated_p = []
    
    for step in range(10):
        next_id_b = logits_b.argmax().item()
        next_id_p = logits_p.argmax().item()
        
        generated_b.append(next_id_b)
        generated_p.append(next_id_p)
        
        diff = (logits_p - logits_b).abs()
        print(f"Step {step+1:02d} (pos={cur_pos}):")
        print(f"  Baseline: {next_id_b:<6} ({tokenizer.decode([next_id_b])!r})")
        print(f"  Patched : {next_id_p:<6} ({tokenizer.decode([next_id_p])!r})")
        print(f"  Max Diff: {diff.max().item():.4f}")
        
        if next_id_b != next_id_p:
            print("  [Note] Token generation diverged.")
            
        # Update baseline
        pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        input_ids_b = torch.tensor([[next_id_b]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs_b = model_baseline(input_ids=input_ids_b, position_ids=pos_tensor, past_key_values=past_b, use_cache=True)
            past_b = outputs_b.past_key_values
            logits_b = outputs_b.logits[0, -1, :].cpu().float()
            
        # Update patched
        input_ids_p = torch.tensor([[next_id_p]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs_p = wrapper.model(input_ids=input_ids_p, position_ids=pos_tensor, past_key_values=None, use_cache=True)
            logits_p = outputs_p.logits[0, -1, :].cpu().float()
            
        cur_pos += 1

    wrapper.stop()

if __name__ == "__main__":
    main()
