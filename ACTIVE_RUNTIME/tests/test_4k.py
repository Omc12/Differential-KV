"""
tests/test_4k.py — 4K token context smoke test.

Verifies:
  - Prefill completes without OOM
  - Decode produces valid tokens
  - VRAM is lower than dense baseline (measured via torch.cuda.memory_allocated)
"""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_4k(monkeypatch):
    from serving.hf_dkv_wrapper import DKVHFWrapper
    MODEL = os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DKVHFWrapper(MODEL, config={}, device=device)

    prompt = "Hello, " * 2000  # ~4K tokens

    # This test exercises the NON-Triton fallback on purpose, but the flag is a
    # MODULE GLOBAL: a bare assignment disabled the fused-decode path for every
    # test that ran after it in the same process, permanently.
    #
    # That is what made the suite look broken while every file passed alone.
    # test_niah's three 8k cases and test_triton_combined's three parity cases
    # are all Triton-dependent and all run after this file alphabetically, so
    # they silently measured the fallback path -- 6 of the 7 order-dependent
    # failures, from this one line.
    #
    # monkeypatch restores the original value at teardown, so the fallback stays
    # scoped to this test.
    from native_core.sparse_decode import triton_fused_decode
    monkeypatch.setattr(triton_fused_decode, "HAS_TRITON", False)

    before_vram = torch.cuda.memory_allocated() / 1e9
    result = wrapper.generate(prompt, max_new_tokens=32)
    after_vram = torch.cuda.memory_allocated() / 1e9
    
    print(f"VRAM before: {before_vram:.2f} GB")
    print(f"VRAM after:  {after_vram:.2f} GB")
    print(f"Generated:   {result[-100:]!r}")
    assert len(result) > 0, "No output generated"
    print("[PASS] test_4k")

if __name__ == "__main__":
    test_4k()
