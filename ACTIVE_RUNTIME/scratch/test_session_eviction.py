#!/usr/bin/env python3
"""
scratch/test_session_eviction.py

Tests the multi-session residency management (ProductionSessionManager)
LRU eviction and restoration pipeline, verifying coherence after swap-out and swap-in.
"""

import os
import sys
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
from serving.production_session_manager import ProductionSessionManager

DEVICE = get_best_device()

def main():
    print("=" * 60)
    print("  Multi-Session Eviction & Restoration Coherence Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    config = {
        "rank": 8,
        "micro_block_size": 32,
        "serving_mode": "lightweight",  # lightweight for fast tests
    }

    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config=config,
        device=DEVICE,
    )

    # Initialize ProductionSessionManager with max_resident_sessions = 2
    # to easily force eviction!
    manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=2,
    )

    print("\n1. Creating Session A...")
    sid_a = manager.create_session()
    wrapper.switch_session(sid_a)
    prompt_a = "<|im_start|>user\nHello! I am Alice. Remember my name.<|im_end|>\n<|im_start|>assistant\n"
    response_a = wrapper.generate(prompt=prompt_a, max_new_tokens=32, temperature=0.0)
    print(f"Session A response: {response_a.strip()!r}")
    manager.append_message(sid_a, "user", "Hello! I am Alice. Remember my name.")
    manager.append_message(sid_a, "assistant", response_a)

    print("\n2. Creating Session B...")
    sid_b = manager.create_session()
    wrapper.switch_session(sid_b)
    prompt_b = "<|im_start|>user\nHello! I am Bob. Remember my name.<|im_end|>\n<|im_start|>assistant\n"
    response_b = wrapper.generate(prompt=prompt_b, max_new_tokens=32, temperature=0.0)
    print(f"Session B response: {response_b.strip()!r}")
    manager.append_message(sid_b, "user", "Hello! I am Bob. Remember my name.")
    manager.append_message(sid_b, "assistant", response_b)

    # Resident sessions should be [sid_a, sid_b]
    print(f"Resident sessions: {manager.resident_sessions}")

    print("\n3. Creating Session C (forces eviction of Session A!)...")
    sid_c = manager.create_session()
    wrapper.switch_session(sid_c)
    prompt_c = "<|im_start|>user\nHello! I am Charlie. Remember my name.<|im_end|>\n<|im_start|>assistant\n"
    response_c = wrapper.generate(prompt=prompt_c, max_new_tokens=32, temperature=0.0)
    print(f"Session C response: {response_c.strip()!r}")
    manager.append_message(sid_c, "user", "Hello! I am Charlie. Remember my name.")
    manager.append_message(sid_c, "assistant", response_c)

    # Resident sessions should be [sid_b, sid_c] (sid_a is evicted to snapshot!)
    print(f"Resident sessions: {manager.resident_sessions}")
    assert sid_a not in manager.resident_sessions, "Session A was not evicted!"
    print("Eviction Check: PASS ✓")

    print("\n4. Accessing Session A (restores from VRAM snapshot!)...")
    # This call to get_session ensures residency and loads it back from the VRAM checkpoint
    manager.get_session(sid_a)
    print(f"Resident sessions: {manager.resident_sessions}")
    assert sid_a in manager.resident_sessions, "Session A was not loaded back!"
    print("Restoration Check: PASS ✓")

    # Ask the follow-up question using the restored session
    print("\n5. Asking follow-up question to Session A...")
    wrapper.switch_session(sid_a)
    history_a = manager.get_history(sid_a)
    follow_up_msg = {"role": "user", "content": "What is my name?"}
    full_messages = history_a + [follow_up_msg]
    prompt_follow_up = wrapper.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
    
    response_follow_up = wrapper.generate(
        prompt=prompt_follow_up,
        max_new_tokens=32,
        temperature=0.0,
        repetition_penalty=1.0,
    )
    print(f"Session A follow-up response: {response_follow_up.strip()!r}")
    
    # Verify that the model remembers Alice's name after session restoration!
    assert "Alice" in response_follow_up, f"Model forgot name after restoration! Response: {response_follow_up}"
    print("Coherence Verification: PASS ✓ (Restored session remembered Alice's name!)")

    wrapper.stop()
    print("\nAll session eviction and restoration tests PASSED ✓")

if __name__ == "__main__":
    main()
