import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Initializing standard baseline model {MODEL}...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    
    base_text = "The quick brown fox jumps over the lazy dog. " * 100  # ~900 tokens
    long_prompt = (base_text + "\n\n") * 8 + "\n\nBased on the text above, answer this question: What does the fox jump over?"
    
    print("\nRunning baseline prefill...")
    inputs = tokenizer(long_prompt, return_tensors='pt').to(device)
    prompt_ids = inputs.input_ids[0].tolist()
    print(f"Prompt length: {len(prompt_ids)} tokens")
    
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, use_cache=True)
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"Baseline Prefill time: {t_prefill:.4f} seconds")
    
    past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    
    generated = prompt_ids.copy()
    cur_pos = len(prompt_ids)
    
    decode_times = []
    for step in range(10):
        next_id = torch.argmax(logits, dim=-1)
        generated.append(next_id.item())
        
        pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        input_ids = next_id.unsqueeze(0)
        
        t_step_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=pos_tensor,
                past_key_values=past_kv,
                use_cache=True,
            )
        t_step = time.perf_counter() - t_step_start
        decode_times.append(t_step)
        print(f"Baseline Decode step {step:2d}: {t_step*1000:.2f} ms")
        
        logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values
        cur_pos += 1
        
    print(f"\nAverage baseline decode step time: {sum(decode_times)/len(decode_times)*1000:.2f} ms")

if __name__ == "__main__":
    main()
