import os
import sys
import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_large_partial_rollback():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    os.environ["DIFFKV_TELEMETRY"] = "1"
    os.environ["DIFFKV_SRL_THRESHOLD"] = "15" # Enable SRL routing
    os.environ["DIFFKV_SRL_VERBOSE"] = "1"
    
    # Let's construct a very long prompt of about 7000 tokens
    base_text = "This paper proposes the concept of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products. "
    # Repeating this ~45 times to get ~7000 tokens (approx 150 words per block)
    long_doc = base_text * 120
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + long_doc + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("Loading model...")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 32, "micro_block_size": 256, "serving_mode": "balanced"}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    # ──── TURN 1 ────
    print("\nSubmitting Turn 1 (long prompt)...")
    q1 = await engine.submit("sess_large_test", {
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.0,
    })
    
    full_output_1 = []
    while True:
        chunk = await q1.get()
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            full_output_1.append(text)
            print(text, end="", flush=True)
        if chunk.get("is_final"):
            break
    
    result_text_1 = "".join(full_output_1)
    print("\n\nTurn 1 Output Complete. Waiting 8 seconds for background compression...")
    await asyncio.sleep(8)
    
    # Get the exact stored prefix IDs
    stored_ids = list(engine.session_token_ids["sess_large_test"])
    print(f"Original stored token IDs length: {len(stored_ids)}")
    
    # ── Simulating mismatch by partially rolling back the session ──
    # We will simulate a mismatch at index 1500 (which is well within compressed blocks)
    mismatch_idx = 1500
    print(f"Simulating mismatch at index {mismatch_idx}")
    
    # Create a new prompt where the tokens match up to mismatch_idx, and then differ.
    # It must be longer than stored_ids (3030 tokens) to satisfy cached_len < len(req.prompt_ids)
    # So we append 1000 extra tokens.
    mismatched_prompt_ids = stored_ids[:mismatch_idx] + [9999] * 10 + stored_ids[mismatch_idx+10:] + stored_ids[:1000]
    mismatched_prompt = wrapper.tokenizer.decode(mismatched_prompt_ids)
    
    # Update stored IDs to have a mismatch at mismatch_idx
    stored_ids_tampered = list(stored_ids)
    stored_ids_tampered[mismatch_idx] = 9999
    engine.session_token_ids["sess_large_test"] = stored_ids_tampered
    
    # ──── TURN 2 ────
    print(f"\nSubmitting Turn 2 (with mismatch at {mismatch_idx})...")
    q2 = await engine.submit("sess_large_test", {
        "prompt": mismatched_prompt,
        "max_tokens": 5,
        "temperature": 0.0,
    })
    
    full_output_2 = []
    while True:
        chunk = await q2.get()
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            full_output_2.append(text)
            print(text, end="", flush=True)
        if chunk.get("is_final"):
            break
    print(f"\n\nTurn 2 Output: {''.join(full_output_2)}")
    
    await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_large_partial_rollback())
