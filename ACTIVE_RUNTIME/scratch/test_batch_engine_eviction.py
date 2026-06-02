#!/usr/bin/env python3
"""
scratch/test_batch_engine_eviction.py

Tests the ProductionSessionManager LRU eviction and restoration pipeline
specifically using the ContinuousBatchEngine, replicating the exact environment
of the FastAPI API gateway.
"""

import os
import sys
import asyncio
import time

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFKV_USE_TORCH_COMPILE", "0")
os.environ.setdefault("DIFFKV_TELEMETRY", "1")

import torch
from native_core.mac_utils import get_best_device
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine
from serving.production_session_manager import ProductionSessionManager

DEVICE = get_best_device()

async def run_completion(engine, session_manager, session_id, messages, max_tokens=32):
    # Ensure session is resident
    session = session_manager.get_session(session_id)
    
    # Render prompt
    prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.15,
    }
    
    queue = await engine.submit(session_id, payload)
    
    full_text = []
    while True:
        chunk = await queue.get()
        if "error" in chunk:
            print(f"Error during generation: {chunk['error']}")
            return ""
        if chunk.get("text"):
            full_text.append(chunk["text"])
        if chunk.get("is_final"):
            break
            
    result = "".join(full_text)
    
    # Update history in manager
    session_manager.clear_history(session_id)
    for msg in messages:
        session_manager.append_message(session_id, msg["role"], msg["content"])
    session_manager.append_message(session_id, "assistant", result)
    
    return result

async def main_async():
    print("=" * 60)
    print("  Continuous Batch Engine Session Eviction Coherence Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    config = {
        "rank": 8,
        "micro_block_size": 32,
        "serving_mode": "lightweight",
    }

    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config=config,
        device=DEVICE,
    )

    print("\nStarting Continuous Batching Engine...")
    engine = ContinuousBatchEngine(wrapper, max_batch_size=4)
    engine.start()

    print("Starting Session Manager...")
    session_manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=2,  # easy eviction
    )

    # 1. Turn 1 for Alice
    print("\n1. Running Session A (Alice)...")
    sid_a = session_manager.create_session()
    messages_a = [{"role": "user", "content": "Hello! I am Alice. Remember my name."}]
    res_a = await run_completion(engine, session_manager, sid_a, messages_a)
    print(f"Session A response: {res_a.strip()!r}")

    # 2. Turn 1 for Bob
    print("\n2. Running Session B (Bob)...")
    sid_b = session_manager.create_session()
    messages_b = [{"role": "user", "content": "Hello! I am Bob. Remember my name."}]
    res_b = await run_completion(engine, session_manager, sid_b, messages_b)
    print(f"Session B response: {res_b.strip()!r}")

    # 3. Turn 1 for Charlie (forces eviction of Session A)
    print("\n3. Running Session C (Charlie) -> Evicts A...")
    sid_c = session_manager.create_session()
    messages_c = [{"role": "user", "content": "Hello! I am Charlie. Remember my name."}]
    res_c = await run_completion(engine, session_manager, sid_c, messages_c)
    print(f"Session C response: {res_c.strip()!r}")

    print(f"Resident sessions: {session_manager.resident_sessions}")
    assert sid_a not in session_manager.resident_sessions, "Session A was not evicted!"
    print("Eviction Check: PASS ✓")

    # 4. Access Session A (restores from VRAM snapshot)
    print("\n4. Restoring Session A...")
    history_a = session_manager.get_history(sid_a)
    follow_up = {"role": "user", "content": "What is my name?"}
    messages_follow_up = history_a + [follow_up]
    
    res_follow_up = await run_completion(engine, session_manager, sid_a, messages_follow_up)
    print(f"Session A follow-up response: {res_follow_up.strip()!r}")

    assert "Alice" in res_follow_up, f"Model forgot Alice's name or returned garbage! Response: {res_follow_up}"
    print("Coherence Verification: PASS ✓ (Restored batch engine session remembered Alice's name!)")

    await engine.stop()
    print("\nAll batch engine eviction tests PASSED ✓")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
