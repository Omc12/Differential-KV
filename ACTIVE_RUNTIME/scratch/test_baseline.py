import os
import sys
import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_baseline():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"
    
    secret_info = "The secret code word is: ALBATROSS. Remember this secret word.\n\n"
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. "
    )
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + secret_info
        + (filler * 40) + "\n\n"
        "Question: What is the secret code word? Answer in exactly one word.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
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
        out = hf_model.generate(**inputs, max_new_tokens=16, temperature=0.0, do_sample=False)
    print(f"HF Output: {repr(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())}")
    
    del hf_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps"):
        torch.mps.empty_cache()
        
    # 2. Test DiffKV with attention cache disabled (threshold = 2.0)
    print("\n==================================================")
    print("2. DiffKV with SVD Compression only (No Attention Cache)")
    print("==================================================")
    os.environ["DIFFKV_SRL_THRESHOLD"] = "99999"  # disable SRL routing
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    # Set threshold to 2.0 just to be sure
    wrapper.manager.attention_score_cache.threshold = 2.0
    
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    q = await engine.submit("sess_diffkv_full", {
        "prompt": prompt,
        "max_tokens": 16,
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
    print(f"DiffKV Full Output: {repr(''.join(full_output).strip())}")
    
    # 3. Test follow-up turn on DiffKV
    print("\nFollow-up turn (user says 'hi'):")
    prompt2 = prompt + "".join(full_output) + "\n<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    q2 = await engine.submit("sess_diffkv_full", {
        "prompt": prompt2,
        "max_tokens": 16,
        "temperature": 0.0,
    })
    full_output2 = []
    while True:
        chunk = await q2.get()
        text = chunk.get("text", "")
        if text:
            full_output2.append(text)
        if chunk.get("is_final"):
            break
    print(f"DiffKV Follow-up Output: {repr(''.join(full_output2).strip())}")
    
    await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_baseline())
