"""
tests/test_session_deletion.py — Verification test for the complete session deletion and resource reclamation.

Verifies:
  1. Creating a session, running a request using ContinuousBatchEngine, and allocating blocks.
  2. LRU eviction successfully creates VRAM/host checkpoints.
  3. Calling delete_session() fully clears active/resident metadata, history, checkpoints, and KV blocks.
  4. Block pool remains completely leak-free and balanced.
"""
import os
import sys
import torch
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_session_deletion():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.production_session_manager import ProductionSessionManager
    from serving.batch_engine import ContinuousBatchEngine
    
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"\n[Test Session Deletion] Initializing model {MODEL}...")
    try:
        from native_core.mac_utils import get_best_device
        device = get_best_device()
    except ImportError:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 8}, device=device)
    kv_mgr = wrapper.manager
    pool = kv_mgr.native_pool
    
    storage_path = "./test_session_checkpoints_del"
    if os.path.exists(storage_path):
        import shutil
        shutil.rmtree(storage_path)
        
    psm = ProductionSessionManager(
        storage_path=storage_path,
        max_resident_sessions=2,
        kv_manager=kv_mgr
    )
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()
    
    initial_free_blocks = len(pool._free_indices)
    print(f"Initial free block slots: {initial_free_blocks}")
    
    # 1. Create a session and submit a generation request to allocate blocks
    sid = psm.create_session()
    psm.append_message(sid, "user", "Write a single sentence about coding.")
    
    history = psm.get_history(sid)
    formatted_prompt = wrapper.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    
    q = await engine.submit(sid, {"prompt": formatted_prompt, "max_tokens": 32, "temperature": 0.0})
    
    # Consume output
    output = []
    while True:
        chunk = await q.get()
        if chunk.get("text"):
            output.append(chunk["text"])
        if chunk.get("is_final"):
            break
            
    response = "".join(output).strip()
    print(f"Response: {response}")
    psm.append_message(sid, "assistant", response)
    
    # Wait for SVD background task (in case any occurred)
    await asyncio.sleep(0.5)
    
    # Verify blocks/cache exist in manager
    blocks = kv_mgr.get_streaming_blocks(sid, 0)
    assert len(blocks) > 0, "No blocks were allocated for session!"
    
    # 2. Force eviction to create checkpoint in kv_mgr
    psm._evict_from_vram(sid)
    psm.resident_sessions.remove(sid)  # Simulate LRU residency list update
    assert sid not in psm.resident_sessions, "Session should have been evicted!"
    assert f"persisted_{sid}" in kv_mgr._session_checkpoints, "Checkpoint should exist in KV manager!"
    
    # Check that blocks in active residency are cleared
    assert len(kv_mgr.get_raw_blocks(sid, 0)) == 0, "Raw resident blocks should be cleared after eviction!"
    
    # 3. Call delete_session and verify complete resource reclamation
    print(f"Deleting session {sid}...")
    psm.delete_session(sid)
    
    # Verification checks
    assert sid not in psm.active_sessions, "Session still in active_sessions!"
    assert sid not in psm.resident_sessions, "Session still in resident_sessions!"
    assert len(psm.get_history(sid)) == 0, "History was not cleared!"
    
    # Check checkpoints in KV manager are deleted
    assert f"persisted_{sid}" not in kv_mgr._session_checkpoints, "Checkpoint still in KV manager checkpoints!"
    
    # Check block pool indices are fully reclaimed
    final_blocks = kv_mgr.get_streaming_blocks(sid, 0)
    assert len(final_blocks) == 0, "KV blocks were not cleared from KV manager!"
    
    # Flush GPU cache and collect garbage
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()
    
    final_free_blocks = len(pool._free_indices)
    print(f"Final free block slots: {final_free_blocks}")
    assert final_free_blocks == initial_free_blocks, f"Leak detected! Expected {initial_free_blocks} free slots, got {final_free_blocks}"
    
    # Clean up disk directory if created
    if os.path.exists(storage_path):
        import shutil
        shutil.rmtree(storage_path)
        
    await engine.stop()
    print("\n[PASS] test_session_deletion verified complete reclamation successfully!")

if __name__ == "__main__":
    asyncio.run(test_session_deletion())
