import os
import sys
import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

async def test_low_ram_incremental_prefill():
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "500"
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
    
    print("Initializing DiffKV wrapper...")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    # Send a prompt of 3K tokens (first turn)
    prompt1 = "Hello " * 600
    print("Submitting Turn 1 (3K tokens)...")
    q1 = await engine.submit("test_session_low_ram", {
        "prompt": prompt1,
        "max_tokens": 10,
        "temperature": 0.0,
    })
    
    # Read the response
    res1 = []
    while True:
        chunk = await q1.get()
        text = chunk.get("text", "")
        if text:
            res1.append(text)
        if chunk.get("is_final"):
            break
    print(f"Turn 1 response: {repr(''.join(res1))}")
    
    # Send a prompt of 50 tokens (second turn, continuation)
    prompt2 = prompt1 + " What is 2+2? "
    print("Submitting Turn 2 (continuation)...")
    q2 = await engine.submit("test_session_low_ram", {
        "prompt": prompt2,
        "max_tokens": 10,
        "temperature": 0.0,
    })
    
    res2 = []
    while True:
        chunk = await q2.get()
        text = chunk.get("text", "")
        if text:
            res2.append(text)
        if chunk.get("is_final"):
            break
    print(f"Turn 2 response: {repr(''.join(res2))}")
    
    await engine.stop()
    print("Test passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_low_ram_incremental_prefill())
