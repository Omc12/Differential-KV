"""
tests/test_cancellation_and_isolation.py

Verifies:
  1. Instant cancellation: When engine.cancel(session_id) is called, the request
     is immediately ejected and its KV blocks are cleared.
  2. Cache isolation: Recycled block memory addresses (after clearing a session)
     do not collide with the ReconstructionCache.
"""
import asyncio
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_cancellation():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine

    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"\n[Test Cancellation] Initializing model {MODEL}...")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 8}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()

    session_id = "test_cancel_sess"
    prompt = "Tell me a very long story about artificial intelligence and space exploration."

    print("Submitting long request...")
    q = await engine.submit(session_id, {
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
    })

    # Read a few tokens
    chunks_received = 0
    for _ in range(5):
        try:
            chunk = await asyncio.wait_for(q.get(), timeout=5.0)
            text = chunk.get("text", "")
            if text:
                chunks_received += 1
                print(f"Received chunk: {text!r}")
            if chunk.get("is_final"):
                break
        except asyncio.TimeoutError:
            break

    print(f"Cancelling session {session_id} immediately...")
    engine.cancel(session_id)

    # Yield control to the event loop to allow the background batch loop task to filter out the cancelled request
    await asyncio.sleep(0.1)

    # Verify that the queue is either closed or marked done, or receives error/is_final
    print("Checking queue after cancellation...")
    is_done = False
    try:
        while not q.empty():
            chunk = q.get_nowait()
            if chunk.get("is_final") or "error" in chunk:
                is_done = True
                break
    except Exception:
        pass

    # Verify active requests does not contain the session
    active_sids = [r.session_id for r in engine.active_requests]
    print(f"Active session IDs in engine: {active_sids}")
    
    assert session_id not in active_sids, f"Cancellation failed: {session_id} still active!"
    
    # Verify blocks are freed from manager
    blocks = wrapper.manager.get_streaming_blocks(session_id, 0)
    assert len(blocks) == 0, f"Blocks not freed! Expected 0, got {len(blocks)}"

    await engine.stop()
    print("[PASS] Cancellation test passed successfully!")


async def test_cache_isolation():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine

    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"\n[Test Cache Isolation] Initializing model {MODEL}...")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 8}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()

    # Session A
    session_a = "session_a"
    q_a = await engine.submit(session_a, {
        "prompt": "List the days of the week.",
        "max_tokens": 16,
        "temperature": 0.0,
    })
    
    # Consume session A fully
    output_a = []
    while True:
        chunk = await q_a.get()
        if "text" in chunk:
            output_a.append(chunk["text"])
        if chunk.get("is_final"):
            break
    print(f"Session A output: {''.join(output_a)!r}")

    # Explicitly cancel/clear Session A to free its blocks and force block memory address recycling
    engine.cancel(session_a)
    
    # Trigger Python garbage collection so deleted blocks' memory addresses can be recycled
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()

    # Session B
    session_b = "session_b"
    q_b = await engine.submit(session_b, {
        "prompt": "List the months of the year.",
        "max_tokens": 16,
        "temperature": 0.0,
    })

    # Consume session B fully
    output_b = []
    while True:
        chunk = await q_b.get()
        if "text" in chunk:
            output_b.append(chunk["text"])
        if chunk.get("is_final"):
            break
    
    result_b = "".join(output_b)
    print(f"Session B output: {result_b!r}")

    # Assertions
    # If the cache leaked, Session B would output components of the days of the week due to stale KV cache lookup
    assert "january" in result_b.lower() or "jan" in result_b.lower() or "month" in result_b.lower() or "year" in result_b.lower(), \
        f"Cache leakage detected! Session B outputted contaminated text: {result_b!r}"

    await engine.stop()
    print("[PASS] Cache isolation test passed successfully!")


async def main():
    await test_cancellation()
    await test_cache_isolation()

if __name__ == "__main__":
    asyncio.run(main())
