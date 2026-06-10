import os
import sys
import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

# Let's construct a prompt of ~800 tokens to exceed the 512 recency window
# We use a repeating sentence to make it easy to reach 800 tokens.
BASE_SENTENCE = "Differential KV is a zero-copy cache compression runtime designed to optimize context scaling in large language models. It uses randomized SVD projection to compress key-value cache blocks while keeping outlier keys dense. "
LONG_PROMPT = BASE_SENTENCE * 8 + "Question: Explain how Differential KV optimizes memory usage."

async def test_repro():
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Enable srl and validation to catch errors
    os.environ["DIFFKV_TELEMETRY"] = "1"
    os.environ["DIFFKV_SRL_THRESHOLD"] = "25"  # default threshold
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + LONG_PROMPT + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("\n--- Running DiffKV (Rank=32) ---")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 32}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    q = await engine.submit("sess_test", {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.0,
    })
    
    full_output = []
    token_idx = 0
    while True:
        chunk = await q.get()
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            full_output.append(text)
            token_idx += 1
            print(f"Token {token_idx:2d}: {repr(text)}")
        if chunk.get("is_final"):
            break
            
    print(f"\nFinal DiffKV Output: {''.join(full_output).strip()}")
    await engine.stop()
    wrapper.close()

if __name__ == "__main__":
    asyncio.run(test_repro())
