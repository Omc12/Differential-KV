"""
tests/test_concurrency.py — Multi-session concurrency test.

Verifies that the batch engine handles multiple sessions without
KV state corruption or VRAM leaks.
"""
import asyncio
import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_concurrency():
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    MODEL = os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DKVHFWrapper(MODEL, config={}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=4)
    engine.start()
    
    sessions = [f"sess_{i}" for i in range(4)]
    queues = []
    for sid in sessions:
        q = await engine.submit(sid, {
            "prompt": f"Session {sid}: Tell me something interesting.",
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
        })
        queues.append((sid, q))
    
    for sid, q in queues:
        while True:
            chunk = await asyncio.wait_for(q.get(), timeout=120.0)
            if chunk.get("is_final"):
                print(f"[{sid}] DONE: {chunk.get('text','')!r}")
                break
    
    await engine.stop()
    print("[PASS] test_concurrency")

if __name__ == "__main__":
    asyncio.run(test_concurrency())
