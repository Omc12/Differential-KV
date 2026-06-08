import os
import sys
import psutil
try:
    _total_mem = psutil.virtual_memory().total
    if _total_mem >= 16 * 1024 ** 3:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.3"
    else:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
except Exception:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Set environment variables BEFORE any other imports so they are registered at module load time
os.environ["DIFFKV_TELEMETRY"] = "1"
os.environ["DIFFKV_SRL_VERBOSE"] = "1"
os.environ["DIFFKV_SRL_THRESHOLD"] = "15" # Enable SRL routing
os.environ["DIFFKV_VALIDATE_SRL"] = "1"    # Enable validation
os.environ["DIFFKV_VALIDATE_EVERY"] = "5"  # Validate every 5 steps
os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"

import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_repro():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from scratch.test_sustainable_ai_prompt import PROMPT
    
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + PROMPT + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("Loading model...")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    print("\nSubmitting user prompt (SRL enabled + validation)...")
    q = await engine.submit("sess_user_prompt", {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
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
            print(text, end="", flush=True)
        if chunk.get("is_final"):
            break
    print("\n\nDiffKV Output:")
    print("".join(full_output))
    
    await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_repro())
