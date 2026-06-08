import os
import sys
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_baseline(model_id, prompt, max_tokens, device):
    print("\n" + "="*80)
    print("RUNNING BASELINE HF GENERATION (DENSE)")
    print("="*80)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy for exact comparison
            pad_token_id=tokenizer.eos_token_id
        )
    dur = time.perf_counter() - t0
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Baseline generated in {dur:.2f}s")
    print(f"Output:\n{generated_text.strip()}")
    return generated_text.strip()

def run_diffkv(model_id, prompt, max_tokens, rank, device):
    print("\n" + "="*80)
    print(f"RUNNING DIFFKV GENERATION WITH rank={rank}")
    print("="*80)
    
    os.environ["DIFFKV_TELEMETRY"] = "0"
    os.environ["DIFFKV_SRL_THRESHOLD"] = "99999"  # disable SRL to isolate SVD compression quality
    
    wrapper = DiffKVHFWrapper(
        model_id,
        config={"rank": rank, "micro_block_size": 256, "serving_mode": "balanced"},
        device=device
    )
    
    t0 = time.perf_counter()
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,  # greedy
    )
    dur = time.perf_counter() - t0
    
    wrapper.stop()
    
    # Strip prompt from response
    ans = response
    if response.startswith(prompt):
        ans = response[len(prompt):]
        
    ans_clean = ans.strip()
    print(f"DiffKV (rank={rank}) generated in {dur:.2f}s")
    print(f"Output:\n{ans_clean}")
    return ans_clean

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Explain the theory of general relativity in detail, including its mathematical foundations, historical context, key experiments that validated it, and its modern implications for astrophysics. Write a long, comprehensive essay.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    # We will generate 300 tokens to see where loop begins or if higher ranks prevent it.
    max_tokens = 300
    
    baseline = run_baseline(model_id, prompt, max_tokens, device)
    
    ranks = [16, 32, 64, 128]
    outputs = {}
    
    for r in ranks:
        try:
            outputs[r] = run_diffkv(model_id, prompt, max_tokens, r, device)
        except Exception as e:
            print(f"Error for rank {r}: {e}")

if __name__ == "__main__":
    main()
