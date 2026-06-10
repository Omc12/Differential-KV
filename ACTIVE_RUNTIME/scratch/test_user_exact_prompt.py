import os
import sys
import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

USER_PROMPT = """### Random Features for Large-Scale Kernel Machines

#### Abstract
We propose a method to accelerate the training of kernel machines, specifically focusing on learning linearly approximating kernels. Our randomized features are designed so that their inner products approximately equal those in the feature space of user specified shift-invariant kernels. We explore two sets of random features and provide convergence bounds for their ability to approximate various radial basis functions (RBFs). In large-scale classification tasks, our approach outperforms linear machine learning algorithms using existing state-of-the-art techniques.

#### Introduction
Kernel machines such as Support Vector Machines are popular because they can accurately approximate any function or decision boundary with enough training data. However, kernel methods operate on the kernel matrix of the input data and scale poorly with the size of the dataset. For example, a dataset containing half a million examples might take days to train using modern workstations.

On the other hand, specialized algorithms for linear Support Vector Machines (SVMs) and regularized regression run much more quickly when the dimensionality is small. However, finding fast methods for large-scale kernel machines remains challenging because the training of these models scales poorly with the dataset size. This paper proposes a solution by combining existing randomized techniques for approximating kernels.

#### Random Fourier Features
We introduce random Fourier features (RFFs), which consist of randomly sampled complex exponentials from the Fourier transform of the input data. These mappings project the points onto"""

async def run_diffkv(preset="mid", rank=32):
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    os.environ["DIFFKV_TELEMETRY"] = "1"
    os.environ["DIFFKV_SRL_THRESHOLD"] = "99999"  # disable SRL to isolate compression
    
    print(f"\n--- Running DiffKV (Preset={preset}, Rank={rank}) ---")
    wrapper = DiffKVHFWrapper(MODEL, config={"preset": preset, "rank": rank}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    # Format the prompt exactly as it is sent via chat completions API
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + USER_PROMPT + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    q = await engine.submit("sess_user", {
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
            
    print(f"DiffKV Output: {repr(''.join(full_output).strip())}")
    
    await engine.stop()
    wrapper.close()
    
    import gc
    del wrapper, engine
    gc.collect()
    torch.mps.empty_cache()

async def run_dense():
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("\n--- Running Standard HF (Dense) ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=device
    )
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + USER_PROMPT + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.0, do_sample=False)
        
    generated = out[0][inputs.input_ids.shape[1]:]
    print(f"Dense Output: {repr(tokenizer.decode(generated, skip_special_tokens=True).strip())}")
    
    import gc
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()

async def main():
    await run_dense()
    await run_diffkv(preset="low", rank=16)
    await run_diffkv(preset="mid", rank=32)

if __name__ == "__main__":
    asyncio.run(main())
