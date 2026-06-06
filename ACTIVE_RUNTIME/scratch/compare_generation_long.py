import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_user_document():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_user_prompt_two_turns.py")
    with open(path, "r") as f:
        code = f.read()
    # Extract the user_document block
    start_marker = 'user_document = """'
    end_marker = '"""'
    start_idx = code.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find user_document start in test_user_prompt_two_turns.py")
    start_idx += len(start_marker)
    end_idx = code.find(end_marker, start_idx)
    if end_idx == -1:
        raise ValueError("Could not find user_document end in test_user_prompt_two_turns.py")
    return code[start_idx:end_idx]

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    user_document = get_user_document()
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + user_document + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
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
    
    # -----------------------------------------------------------------
    # Load patched model
    # -----------------------------------------------------------------
    print("\n2. Loading patched model...")
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 256},
        device=device,
    )
    
    print("Running patched model prefill...")
    session_id = "default"
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=prefill_len)
    wrapper.model._diffkv_session_ids = [session_id]
    
    with torch.no_grad():
        outputs_patched = wrapper.model(input_ids=encoded.input_ids, use_cache=True)
        wrapper.manager.compress_prefill_kv(session_id)
        # Force finalize compressed blocks synchronously for test accuracy
        wrapper.manager.finalize_compressed_blocks()
    logits_p = outputs_patched.logits[0, -1, :].cpu().float()
    
    # Compare first logits
    diff = (logits_p - logits_b).abs()
    print(f"\n--- Prefill/Step 0 comparison: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f} ---")
    
    print("\n============================================================")
    print("  Autoregressive Generation (independent generation) - 50 Steps")
    print("============================================================")
    
    cur_pos = prefill_len
    
    generated_b = []
    generated_p = []
    
    for step in range(50):
        next_id_b = logits_b.argmax().item()
        next_id_p = logits_p.argmax().item()
        
        generated_b.append(next_id_b)
        generated_p.append(next_id_p)
        
        token_str_b = tokenizer.decode([next_id_b])
        token_str_p = tokenizer.decode([next_id_p])
        
        diff = (logits_p - logits_b).abs()
        
        print(f"Step {step+1:02d} (pos={cur_pos}):")
        print(f"  Baseline: {next_id_b:<6} ({token_str_b!r}) -> {tokenizer.decode(generated_b)!r}")
        print(f"  Patched : {next_id_p:<6} ({token_str_p!r}) -> {tokenizer.decode(generated_p)!r}")
        print(f"  Max Diff: {diff.max().item():.4f}")
        
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
