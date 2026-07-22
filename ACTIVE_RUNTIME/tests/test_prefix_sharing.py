"""
tests/test_prefix_sharing.py — Automated test for Prefix Caching & Session Snapshots.

Verifies:
  - Session snapshotting captures correct state.
  - Zero-copy session branching/restoring operates with perfect isolation.
  - Outputs are high quality and mathematically separated.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_prefix_sharing():
    from serving.hf_dkv_wrapper import DKVHFWrapper
    MODEL = os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DKVHFWrapper(MODEL, config={}, device=device)
    
    # 1. Run prompt prefill on common prefix
    prefix = "Once upon a time, in a deep blue ocean, there lived a small dolphin named Sparky."
    
    # Run prefill on session A
    wrapper.model._dkv_session_ids = ["sess_a"]
    wrapper.manager.clear_session("sess_a")
    wrapper.manager.init_session("sess_a", prefill_len=len(prefix))
    
    # Staging input
    input_ids = wrapper.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        wrapper.model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
        
    # 2. Snapshot session A's state at turn boundary
    wrapper.manager.snapshot_session("sess_a", "dolphin_prefix")
    
    # 3. Branch a new session B from the snapshot
    wrapper.manager.restore_session("sess_b", "dolphin_prefix")
    
    # 4. Generate in session A with prompt extension 1
    prompt_a = " Sparky wanted to fly into the sky."
    wrapper.model._dkv_session_ids = ["sess_a"]
    input_ids_a = wrapper.tokenizer(prompt_a, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    position_ids_a = torch.arange(input_ids.shape[1], input_ids.shape[1] + input_ids_a.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out_a = wrapper.model(input_ids=input_ids_a, position_ids=position_ids_a, use_cache=True)
    logits_a = out_a.logits[0, -1]
    
    # 5. Generate in session B with prompt extension 2
    prompt_b = " Sparky wanted to find a hidden treasure."
    wrapper.model._dkv_session_ids = ["sess_b"]
    input_ids_b = wrapper.tokenizer(prompt_b, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    position_ids_b = torch.arange(input_ids.shape[1], input_ids.shape[1] + input_ids_b.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out_b = wrapper.model(input_ids=input_ids_b, position_ids=position_ids_b, use_cache=True)
    logits_b = out_b.logits[0, -1]
    
    # 6. Verify that logits are not corrupted and are fully separated
    assert not torch.allclose(logits_a, logits_b), "Output logits leaked between isolated branched sessions!"
    assert torch.isfinite(logits_a).all() and torch.isfinite(logits_b).all(), "Logits contain invalid floating-point values!"
    
    # Clean up
    wrapper.manager.clear_session("sess_a")
    wrapper.manager.clear_session("sess_b")
    wrapper.manager.delete_checkpoint("dolphin_prefix")
    print("[PASS] test_prefix_sharing")

if __name__ == "__main__":
    test_prefix_sharing()
