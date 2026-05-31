import os
import sys
import time
import asyncio
import torch

sys.path.insert(0, "d:\\Codes\\Projects\\Differential KV\\ACTIVE_RUNTIME")

async def test_prefix_matching():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine

    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    device = "cuda"

    print(f"\n=== INITIALIZING PREFIX VERIFICATION WITH MODEL: {MODEL} ===")
    
    wrapper = DiffKVHFWrapper(
        MODEL, 
        config={"rank": 16, "micro_block_size": 16, "serving_mode": "performance"}, 
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
    
    # Check that it generated something about gravity
    assert "gravit" in assistant_resp_1.lower() or "attract" in assistant_resp_1.lower() or "force" in assistant_resp_1.lower(), f"Unexpected response: {assistant_resp_1}"

    # --- Step 2: Second turn (completely different prompt: France capital) ---
    print("\n--- TURN 2: France Capital (Prefix mismatch - should clear cache) ---")
    prompt_2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    
    q2 = await engine.submit(session_id, {"prompt": prompt_2, "max_tokens": 50, "temperature": 0.0})
    resp_2_tokens = []
    while True:
        chunk = await q2.get()
        print(f"[DEBUG Turn 2] chunk received: {chunk}")
        text = chunk.get("text", "")
        if text:
            resp_2_tokens.append(text)
        if chunk.get("is_final"):
            break
    assistant_resp_2 = "".join(resp_2_tokens).strip()
    print(f"Assistant 2: {repr(assistant_resp_2)}")
    
    # Assert that there is NO talk of gravity or orbits in this response!
    assert "paris" in assistant_resp_2.lower(), f"Expected Paris in response, got: {assistant_resp_2}"
    assert "gravit" not in assistant_resp_2.lower() and "orbit" not in assistant_resp_2.lower(), f"Leaked history found: {assistant_resp_2}"

    # --- Step 3: Third turn (continuation of prompt 2 - prefix matching) ---
    print("\n--- TURN 3: Continuation (Prefix matching - should reuse cache) ---")
    prompt_3 = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_resp_2}<|im_end|>\n"
        f"<|im_start|>user\nParis is known for what? Answer in one sentence.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    # We expect this to print "Found cached history... Reusing KV cache!" in the console output!
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

    # Assert correct continuation
    assert len(assistant_resp_3) > 0, "Response is empty!"
    
    print("\nShutting down engine...")
    await engine.stop()
    print("\n=== PREFIX VERIFICATION PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(test_prefix_matching())
