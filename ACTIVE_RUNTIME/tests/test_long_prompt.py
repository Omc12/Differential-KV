import asyncio
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_long_prompt():
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine

    MODEL = os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Initializing model {MODEL}...")
    
    # Enable telemetry to see state transitions
    os.environ["DKV_TELEMETRY"] = "1"
    os.environ["DKV_SRL_THRESHOLD"] = "10"
    os.environ["DKV_SRL_VERBOSE"] = "1"
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()

    # Place the crucial information at the very beginning of the prompt!
    secret_info = "The secret code word is: ALBATROSS. Remember this secret word.\n\n"
    
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + secret_info
        + (filler * 40) + "\n\n"
        "Question: What is the secret code word? Answer in exactly one word.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print(f"\nSubmitting long prompt request (~{len(prompt)//4} tokens)...")
    q = await engine.submit("session_long_prompt", {
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0,  # greedy decoding
        "top_p": 0.9,
        "repetition_penalty": 1.15,
    })

    full_output = []
    while True:
        chunk = await asyncio.wait_for(q.get(), timeout=90.0)
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            full_output.append(text)
            print(text, end="", flush=True)
        if chunk.get("is_final"):
            break

    generated_text = "".join(full_output).strip()
    print("\n\n" + "=" * 50)
    print("GENERATED ANSWER:")
    print("=" * 50)
    print(f"REPR: {repr(generated_text)}")
    print("=" * 50)

    # ── Second Turn (Follow-up) ──────────────────────────────────────────────
    print("\nSubmitting second turn follow-up prompt (user says 'hi')...")
    # Set threshold lower for the test if needed, but the session has 912 compressed blocks,
    # which is well above the default threshold (50). So it will definitely route!
    prompt2 = (
        prompt
        + generated_text + "\n<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    # Enable verbose logging for routing
    os.environ["DKV_SRL_VERBOSE"] = "1"
    
    q2 = await engine.submit("session_long_prompt", {
        "prompt": prompt2,
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
    })
    
    full_output2 = []
    while True:
        chunk = await asyncio.wait_for(q2.get(), timeout=90.0)
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            full_output2.append(text)
            print(text, end="", flush=True)
        if chunk.get("is_final"):
            break

    generated_text2 = "".join(full_output2).strip()
    print("\n\n" + "=" * 50)
    print("GENERATED ANSWER 2:")
    print("=" * 50)
    print(f"REPR: {repr(generated_text2)}")
    print("=" * 50)

    await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_long_prompt())
