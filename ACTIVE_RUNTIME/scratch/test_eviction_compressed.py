#!/usr/bin/env python3
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
from native_core.streaming_sparse_ingest import StreamingKVBlock, StreamingSparseIngestManager

# ── Force compression on all blocks by patching the guards ──
def patched_is_compression_eligible(self) -> bool:
    return (
        self.state == "ACCUMULATING"
        and self.active_k is not None
        and self.active_k.shape[2] >= self.micro_block_size
    )

StreamingKVBlock.is_compression_eligible = patched_is_compression_eligible

DEVICE = get_best_device()

def main():
    print("=" * 60)
    print("  Compressed Session Eviction & Restoration Coherence Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    large_prompt_part = "Alice is a software engineer. " * 650  # 650 * 6 = ~3900 tokens
    
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

    manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=2,
    )

    print("\n1. Creating Session A...")
    sid_a = manager.create_session()
    wrapper.switch_session(sid_a)
    
    prompt_a = f"<|im_start|>user\nHere is some background information:\n{large_prompt_part}\nRemember that my name is Alice. I repeat, my name is Alice. Hello!\nSummarize the background info in one short sentence.<|im_end|>\n<|im_start|>assistant\n"
    
    encoded = wrapper.tokenizer(prompt_a, return_tensors="pt")
    print(f"Session A prompt length: {encoded.input_ids.shape[1]} tokens")
    
    response_a = wrapper.generate(prompt=prompt_a, max_new_tokens=32, temperature=0.0)
    print(f"Session A response: {response_a.strip()!r}")
    
    manager.append_message(sid_a, "user", f"Here is some background information:\n{large_prompt_part}\nRemember that my name is Alice. I repeat, my name is Alice. Hello!\nSummarize the background info in one short sentence.")
    manager.append_message(sid_a, "assistant", response_a)

    print("\n2. Creating Session B...")
    sid_b = manager.create_session()
    wrapper.switch_session(sid_b)
    prompt_b = "<|im_start|>user\nHello! I am Bob. Remember my name.<|im_end|>\n<|im_start|>assistant\n"
    response_b = wrapper.generate(prompt=prompt_b, max_new_tokens=32, temperature=0.0)
    manager.append_message(sid_b, "user", "Hello! I am Bob. Remember my name.")
    manager.append_message(sid_b, "assistant", response_b)

    print("\n3. Creating Session C (forces eviction of Session A!)...")
    sid_c = manager.create_session()
    wrapper.switch_session(sid_c)
    prompt_c = "<|im_start|>user\nHello! I am Charlie. Remember my name.<|im_end|>\n<|im_start|>assistant\n"
    response_c = wrapper.generate(prompt=prompt_c, max_new_tokens=32, temperature=0.0)
    manager.append_message(sid_c, "user", "Hello! I am Charlie. Remember my name.")
    manager.append_message(sid_c, "assistant", response_c)

    print("\n4. Accessing Session A (restores from VRAM snapshot!)...")
    manager.get_session(sid_a)

    # Ask the follow-up question
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
    assert "Alice" in response_follow_up, f"Model forgot name or returned gibberish after restoration! Response: {response_follow_up}"
    print("Coherence Verification: PASS ✓ (Restored compressed session remembered Alice's name!)")

    wrapper.stop()

if __name__ == "__main__":
    main()
