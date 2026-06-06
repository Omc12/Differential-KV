import os
import sys
import time
import asyncio
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_prefix_matching():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\n=== INITIALIZING PREFIX VERIFICATION WITH MODEL: {MODEL} ===")
    
    wrapper = DiffKVHFWrapper(
        MODEL, 
        config={"rank": 32, "micro_block_size": 256, "serving_mode": "balanced"}, 
        device=device
    )
    
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()

    session_id = "test_prefix_session"

    # --- Step 1: First turn (gravity) ---
    print("\n--- TURN 1: Explain gravity (Fresh prefill) ---")
    prompt_1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain gravity in one sentence.<|im_end|>\n<|im_start|>assistant\n"
    
    q1 = await engine.submit(session_id, {"prompt": prompt_1, "max_tokens": 50, "temperature": 0.0})
    resp_1_tokens = []
    while True:
        chunk = await q1.get()
        text = chunk.get("text", "")
        if text:
            resp_1_tokens.append(text)
        if chunk.get("is_final"):
            break
    assistant_resp_1 = "".join(resp_1_tokens).strip()
    print(f"Assistant 1: {repr(assistant_resp_1)}")

    # --- Step 2: Second turn (completely different prompt: France capital) ---
    print("\n--- TURN 2: France Capital (Prefix mismatch - should clear cache) ---")
    prompt_2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    
    q2 = await engine.submit(session_id, {"prompt": prompt_2, "max_tokens": 50, "temperature": 0.0})
    resp_2_tokens = []
    while True:
        chunk = await q2.get()
        text = chunk.get("text", "")
        if text:
            resp_2_tokens.append(text)
        if chunk.get("is_final"):
            break
    assistant_resp_2 = "".join(resp_2_tokens).strip()
    print(f"Assistant 2: {repr(assistant_resp_2)}")

    # --- Step 3: Third turn (continuation of prompt 2 - prefix matching) ---
    print("\n--- TURN 3: Continuation (Prefix matching - should reuse cache) ---")
    prompt_3 = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_resp_2}<|im_end|>\n"
        f"<|im_start|>user\nParis is known for what? Answer in one sentence.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    q3 = await engine.submit(session_id, {"prompt": prompt_3, "max_tokens": 50, "temperature": 0.0})
    resp_3_tokens = []
    while True:
        chunk = await q3.get()
        text = chunk.get("text", "")
        if text:
            resp_3_tokens.append(text)
        if chunk.get("is_final"):
            break
    assistant_resp_3 = "".join(resp_3_tokens).strip()
    print(f"Assistant 3: {repr(assistant_resp_3)}")

    print("\nShutting down engine...")
    await engine.stop()
    print("\n=== PREFIX VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_prefix_matching())
