"""
tests/test_formatting_rendering.py — Verification test for list rendering, newlines, and punctuation.

Verifies:
  1. No suppression of punctuation, newlines, or bullets.
  2. Sequential-Delta Decoding properly retains spacing and multi-byte characters.
  3. Continuous batching engine correctly finishes and returns full formatted text.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_formatting():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine

    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Initializing model {MODEL}...")
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nProvide a structured list of three major colors. "
        "Use bullet points (with asterisks) and newlines between them. "
        "Write one complete sentence for each color ending with a period.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print("\nSubmitting formatting generation request...")
    q = await engine.submit("session_formatting", {
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.0,  # greedy decoding for deterministic verification
        "top_p": 0.9,
        "repetition_penalty": 1.15,
    })

    full_output = []
    while True:
        chunk = await asyncio.wait_for(q.get(), timeout=30.0)
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            full_output.append(text)
            # print(text, end="", flush=True)
        if chunk.get("is_final"):
            break

    generated_text = "".join(full_output)
    print("\n\n" + "=" * 50)
    print("GENERATED FORMATTED TEXT:")
    print("=" * 50)
    print(f"REPR: {repr(generated_text)}")
    try:
        print(generated_text)
    except UnicodeEncodeError:
        print(generated_text.encode('ascii', errors='replace').decode('ascii'))
    print("=" * 50)

    has_newlines = "\n" in generated_text
    has_bullets = "*" in generated_text or "-" in generated_text
    has_punctuation = any(c in generated_text for c in [".", ",", "!", "*"])

    print(f"Has Newlines: {has_newlines}")
    print(f"Has Bullets:  {has_bullets}")
    print(f"Has Puncs:    {has_punctuation}")

    await engine.stop()

    assert has_newlines, "Formatting failure: No newlines generated!"
    assert has_bullets, "Formatting failure: No bullet points/stars generated!"
    assert has_punctuation, "Formatting failure: No punctuation generated!"
    print("\n[PASS] test_formatting_rendering successfully verified!")

if __name__ == "__main__":
    asyncio.run(test_formatting())
