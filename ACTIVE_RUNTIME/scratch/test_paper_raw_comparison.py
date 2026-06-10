import os
import sys
import torch
import asyncio

# Ensure ACTIVE_RUNTIME is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

async def run_diffkv(prompt, preset="low", rank=16):
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    os.environ["DIFFKV_TELEMETRY"] = "1"
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
    
    print(f"\n--- Running DiffKV (Preset={preset}, Rank={rank}) ---")
    wrapper = DiffKVHFWrapper(MODEL, config={"preset": preset, "rank": rank}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    q = await engine.submit("sess_diffkv", {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.0,
    })
    
    full_output = []
    while True:
        chunk = await q.get()
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            full_output.append(text)
        if chunk.get("is_final"):
            break
            
    ans = "".join(full_output).strip()
    await engine.stop()
    wrapper.close()
    
    import gc
    del wrapper, engine
    gc.collect()
    torch.mps.empty_cache()
    return ans

async def run_dense(prompt):
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("\n--- Running Standard HF (Dense) ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=device
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.0, do_sample=False)
        
    generated = out[0][inputs.input_ids.shape[1]:]
    ans = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    import gc
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return ans

async def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load research paper text
    paper_path = "/Users/omchimurkar1/.gemini/antigravity/brain/3b15502c-2913-4cf2-82e1-d764b49db8c4/scratch/research_paper.txt"
    with open(paper_path, "r") as f:
        paper_text = f.read()
    
    # Recreate the 8332-token prompt exactly
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # We want system/user wrapper around the paper
    header = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    footer = "<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize header + footer
    header_ids = tokenizer.encode(header)
    footer_ids = tokenizer.encode(footer)
    
    # Tokenize paper and repeat it to fill up 8332 total tokens
    paper_ids = tokenizer.encode(paper_text)
    
    # How many tokens do we need from the repeated paper?
    target_paper_tokens = 8332 - len(header_ids) - len(footer_ids)
    
    # Repeat paper_ids
    repeated_paper_ids = []
    while len(repeated_paper_ids) < target_paper_tokens:
        repeated_paper_ids.extend(paper_ids)
    repeated_paper_ids = repeated_paper_ids[:target_paper_tokens]
    
    # Combine everything
    full_ids = header_ids + repeated_paper_ids + footer_ids
    assert len(full_ids) == 8332, f"Expected 8332 tokens, got {len(full_ids)}"
    
    prompt = tokenizer.decode(full_ids)
    
    dense_ans = await run_dense(prompt)
    print(f"\n======================================")
    print(f"DENSE OUTPUT:")
    print(f"======================================")
    print(dense_ans)
    
    diffkv_ans = await run_diffkv(prompt, preset="low", rank=16)
    print(f"\n======================================")
    print(f"DIFFKV OUTPUT:")
    print(f"======================================")
    print(diffkv_ans)

if __name__ == "__main__":
    asyncio.run(main())
