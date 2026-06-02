#!/usr/bin/env python3
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

# Disable MLX SVD completely to see if it fixes the gibberish!
import native_core.mac_utils
native_core.mac_utils.mlx_available = lambda: False
print("Forced mlx_available = False for diagnostic purposes.")

import torch
from native_core.mac_utils import get_best_device
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine
from serving.production_session_manager import ProductionSessionManager
from native_core.streaming_sparse_ingest import StreamingKVBlock, StreamingSparseIngestManager

# Force compression on all blocks by patching the guards
def patched_is_compression_eligible(self) -> bool:
    return (
        self.state == "ACCUMULATING"
        and self.active_k is not None
        and self.active_k.shape[2] >= self.micro_block_size
    )

StreamingKVBlock.is_compression_eligible = patched_is_compression_eligible

DEVICE = get_best_device()

async def run_completion(engine, session_manager, session_id, messages, max_tokens=32):
    session = session_manager.get_session(session_id)
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
            print(f"Error: {chunk['error']}")
            return ""
        if chunk.get("text"):
            full_text.append(chunk["text"])
        if chunk.get("is_final"):
            break
            
    result = "".join(full_text)
    
    session_manager.clear_history(session_id)
    for msg in messages:
        session_manager.append_message(session_id, msg["role"], msg["content"])
    session_manager.append_message(session_id, "assistant", result)
    
    return result

async def main_async():
    print("=" * 60)
    print("  Continuous Batch Engine Compressed Eviction Coherence Test (NO MLX)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    large_prompt_part = "Alice is a software engineer. " * 650

    config = {
        "rank": 8,
        "micro_block_size": 32,
        "serving_mode": "balanced",
    }

    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config=config,
        device=DEVICE,
    )

    engine = ContinuousBatchEngine(wrapper, max_batch_size=4)
    engine.start()

    session_manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=2,
    )

    # 1. Turn 1 for Alice
    print("\n1. Running Session A (Alice with LARGE prompt)...")
    sid_a = session_manager.create_session()
    messages_a = [{"role": "user", "content": f"Here is some background info:\n{large_prompt_part}\nRemember that my name is Alice. I repeat, my name is Alice. Hello!\nSummarize the background info in one sentence."}]
    res_a = await run_completion(engine, session_manager, sid_a, messages_a, max_tokens=64)
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

    # 4. Access Session A (restores from VRAM snapshot)
    print("\n4. Restoring Session A...")
    history_a = session_manager.get_history(sid_a)
    follow_up = {"role": "user", "content": "What is my name?"}
    messages_follow_up = history_a + [follow_up]
    
    res_follow_up = await run_completion(engine, session_manager, sid_a, messages_follow_up)
    print(f"Session A follow-up response: {res_follow_up.strip()!r}")

    assert "Alice" in res_follow_up, f"Model forgot Alice's name or returned garbage! Response: {res_follow_up}"
    print("Coherence Verification: PASS ✓ (Restored session remembered Alice's name!)")

    await engine.stop()
    print("\nAll batch engine compressed eviction tests PASSED ✓")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
