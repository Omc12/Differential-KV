import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import os
from validation.reset_environment import reset_environment

def run_baseline(model_id, prompts, max_new_tokens=100):
    reset_environment()
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    results = []
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Warmup
        if i == 0:
            model.generate(**inputs, max_new_tokens=5)
            torch.cuda.synchronize()
            
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )
        torch.cuda.synchronize()
        end_time = time.time()
        
        latency = end_time - start_time
        num_tokens = output.shape[1] - inputs.input_ids.shape[1]
        tps = num_tokens / latency
        
        vram_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        results.append({
            "prompt_idx": i,
            "latency": latency,
            "tokens_per_sec": tps,
            "vram_peak_mb": vram_peak,
            "output_text": tokenizer.decode(output[0], skip_special_tokens=True)
        })
        
        print(f"Prompt {i}: {tps:.2f} tok/s, {vram_peak:.2f} MB peak VRAM")

    return results

if __name__ == "__main__":
    # Example usage for testing the runner itself
    test_prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to calculate the Fibonacci sequence."
    ]
    # Defaulting to a small model for test runs if not specified
    model_id = "microsoft/phi-3-mini-4k-instruct"
    
    results = run_baseline(model_id, test_prompts)
    
    output_path = "results/reality_reset/baseline_phi3_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")
