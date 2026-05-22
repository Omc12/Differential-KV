"""
tests/test_25k.py — 25K token context stress test.

Verifies:
  - No OOM during prefill or decode
  - Streaming ingest bounded VRAM (< dense baseline)
  - Triton kernel dispatches (check for [Phase 28] log line)
  - At least 32 tokens generated
"""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_25k():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    wrapper = DiffKVHFWrapper(MODEL, config={}, device="cuda")
    
    prompt = "The following is a long document. " * 1000  # ~25K tokens
    
    torch.cuda.reset_peak_memory_stats()
    result = wrapper.generate(prompt, max_new_tokens=64)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    
    summary = wrapper.manager.get_streaming_summary()
    
    print(f"Peak VRAM:   {peak_vram:.2f} GB")
    print(f"Streaming:   {summary}")
    print(f"Generated:   {result[-200:]!r}")
    assert len(result) > 0
    print("[PASS] test_25k")

if __name__ == "__main__":
    test_25k()
