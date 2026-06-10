import os
import sys
import torch
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine
from scratch.test_sustainable_ai_prompt import PROMPT

async def main():
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + PROMPT + "<|im_end|>\n"
        "Question: What is the proposed definition of Sustainable AI?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    # 1. Test HF standard model
    print("\n==================================================")
    print("1. Standard Hugging Face Model (No DiffKV)")
    print("==================================================")
    os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "99999"  # Disable chunked prefill
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=device
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model.generate(**inputs, max_new_tokens=50, temperature=0.0, do_sample=False)
    print(f"HF Output: {repr(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())}")
    
    del hf_model
    import gc
    gc.collect()
    torch.mps.empty_cache()

    # 2. Test DiffKV with exact attn and rank=16
    print("\n==================================================")
    print("2. DiffKV Model (Exact Attn, Rank=16)")
    print("==================================================")
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "0"
    os.environ["DIFFKV_SRL_THRESHOLD"] = "99999"  # disable SRL to isolate compression
    
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    q = await engine.submit("sess_diffkv_16", {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.0,
    })
    
    full_output = []
    while True:
        chunk = await q.get()
        text = chunk.get("text", "")
        if text:
            full_output.append(text)
        if chunk.get("is_final"):
            break
    print(f"DiffKV Output (Rank=16): {repr(''.join(full_output).strip())}")
    await engine.stop()
    wrapper.close()
    del wrapper, engine
    torch.mps.empty_cache()

    # 3. Test DiffKV with exact attn and rank=32
    print("\n==================================================")
    print("3. DiffKV Model (Exact Attn, Rank=32)")
    print("==================================================")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 32}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    q = await engine.submit("sess_diffkv_32", {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.0,
    })
    
    full_output = []
    while True:
        chunk = await q.get()
        text = chunk.get("text", "")
        if text:
            full_output.append(text)
        if chunk.get("is_final"):
            break
    print(f"DiffKV Output (Rank=32): {repr(''.join(full_output).strip())}")
    await engine.stop()
    wrapper.close()
    del wrapper, engine
    torch.mps.empty_cache()

    # 4. Test DiffKV with exact attn and rank=64
    print("\n==================================================")
    print("4. DiffKV Model (Exact Attn, Rank=64)")
    print("==================================================")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 64}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    q = await engine.submit("sess_diffkv_64", {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.0,
    })
    
    full_output = []
    while True:
        chunk = await q.get()
        text = chunk.get("text", "")
        if text:
            full_output.append(text)
        if chunk.get("is_final"):
            break
    print(f"DiffKV Output (Rank=64): {repr(''.join(full_output).strip())}")
    
    await engine.stop()
    wrapper.close()

if __name__ == "__main__":
    asyncio.run(main())
