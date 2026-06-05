import os
import sys
import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_partial_rollback():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    os.environ["DIFFKV_TELEMETRY"] = "1"
    os.environ["DIFFKV_SRL_THRESHOLD"] = "15" # Enable SRL routing
    os.environ["DIFFKV_SRL_VERBOSE"] = "1"
    
    user_document = """This paper proposes the concept of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance) towards greater ecological integrity and social justice."""
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + user_document + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("Loading model...")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 32, "micro_block_size": 256, "serving_mode": "balanced"}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    # ──── TURN 1 ────
    print("\nSubmitting Turn 1 prompt...")
    q1 = await engine.submit("sess_rollback_test", {
        "prompt": prompt,
        "max_tokens": 30,
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
    print("\n\nTurn 1 Output Complete.")
    
    # Get the exact stored prefix IDs
    stored_ids = list(engine.session_token_ids["sess_rollback_test"])
    print(f"Original stored token IDs length: {len(stored_ids)}")
    
    # ── Simulating mismatch by tampering the last few tokens of the stored prefix ──
    # We alter token at len(stored_ids) - 5 to some other token (e.g. 9999) to force mismatch
    tamper_idx = len(stored_ids) - 5
    original_val = stored_ids[tamper_idx]
    stored_ids[tamper_idx] = 9999
    engine.session_token_ids["sess_rollback_test"] = stored_ids
    print(f"Tampered token at index {tamper_idx} (was {original_val}, now 9999) to simulate template difference.")
    
    # Now build prompt for Turn 2
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_document},
        {"role": "assistant", "content": result_text_1},
        {"role": "user", "content": "hi"}
    ]
    turn2_prompt = wrapper.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # ──── TURN 2 ────
    print("\nSubmitting Turn 2 (with prefix mismatch)...")
    q2 = await engine.submit("sess_rollback_test", {
        "prompt": turn2_prompt,
        "max_tokens": 15,
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
    asyncio.run(test_partial_rollback())
