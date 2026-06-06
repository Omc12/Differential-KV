import os
import sys
import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_chunked_prefill():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + (filler * 30) + "\n\n"
        "What is the capital of France? Answer in one word.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print(f"Prompt length: {len(prompt)} characters (~{len(prompt)//4} tokens)")
    
    # 1. Test HF standard model
    print("\n==================================================")
    print("1. Standard Hugging Face Model (No DiffKV)")
    print("==================================================")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=device
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model.generate(**inputs, max_new_tokens=10, temperature=0.0, do_sample=False)
    print(f"HF Output: {repr(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())}")
    
    del hf_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps"):
        torch.mps.empty_cache()
        
    # 2. Test DiffKV
    print("\n==================================================")
    print("2. DiffKV Model")
    print("==================================================")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    q = await engine.submit("sess_chunked_test", {
        "prompt": prompt,
        "max_tokens": 10,
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
    print(f"DiffKV Output: {repr(''.join(full_output).strip())}")
    
    await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_chunked_prefill())
