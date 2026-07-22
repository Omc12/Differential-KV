import asyncio
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_cloning_decode_isolation():
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine

    MODEL = os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Initializing model {MODEL} for isolation test...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nTell me a story about a little bird named Bluey.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print("\nSubmitting request on Session A...")
    q_a = await engine.submit("session_a", {
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.0,
    })

    out_a1 = []
    while True:
        chunk = await q_a.get()
        text = chunk.get("text", "")
        if text:
            out_a1.append(text)
        if chunk.get("is_final"):
            break
    resp_a1 = "".join(out_a1)
    print(f"Session A generated first 10 tokens: {repr(resp_a1)}")

    # Clone Session A to Session B
    print("\nCloning session_a to session_b...")
    engine.wrapper.manager.clone_session("session_a", "session_b")
    # Register the tokens in engine registry
    engine.session_token_ids["session_b"] = list(engine.session_token_ids["session_a"])

    # Now prompt Session A to continue with a user message "Why?"
    prompt_a2 = prompt + resp_a1 + "\n<|im_end|>\n<|im_start|>user\nWhy?<|im_end|>\n<|im_start|>assistant\n"
    print("\nSubmitting follow-up request on Session A...")
    q_a2 = await engine.submit("session_a", {
        "prompt": prompt_a2,
        "max_tokens": 10,
        "temperature": 0.0,
    })

    # Simultaneously, prompt Session B to continue with a user message "Where?"
    prompt_b2 = prompt + resp_a1 + "\n<|im_end|>\n<|im_start|>user\nWhere?<|im_end|>\n<|im_start|>assistant\n"
    print("\nSubmitting follow-up request on Session B...")
    q_b2 = await engine.submit("session_b", {
        "prompt": prompt_b2,
        "max_tokens": 10,
        "temperature": 0.0,
    })

    out_a2 = []
    while True:
        chunk = await q_a2.get()
        text = chunk.get("text", "")
        if text:
            out_a2.append(text)
        if chunk.get("is_final"):
            break
    resp_a2 = "".join(out_a2)
    print(f"Session A follow-up response: {repr(resp_a2)}")

    out_b2 = []
    while True:
        chunk = await q_b2.get()
        text = chunk.get("text", "")
        if text:
            out_b2.append(text)
        if chunk.get("is_final"):
            break
    resp_b2 = "".join(out_b2)
    print(f"Session B follow-up response: {repr(resp_b2)}")

    await engine.stop()

    # Verify that the two outputs are different and not corrupted
    assert resp_a2 != resp_b2, "Outputs leaked between isolated cloned sessions!"
    assert len(resp_a2) > 0 and len(resp_b2) > 0, "Outputs are empty!"
    print("\n[PASS] test_cloning_decode_isolation successfully verified!")

if __name__ == "__main__":
    asyncio.run(test_cloning_decode_isolation())
