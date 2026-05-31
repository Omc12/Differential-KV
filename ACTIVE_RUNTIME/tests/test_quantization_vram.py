"""
ACTIVE_RUNTIME/tests/test_quantization_vram.py

Unit tests for dynamic growable block pool and VRAM reclamation.
Verifies:
1. Upfront pool allocation is lightweight (initial_blocks).
2. Pool dynamically grows up to the max_blocks cap.
3. Reset shrinks the pool back to initial_blocks, releasing VRAM.
4. Out-of-memory guards work perfectly.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from runtime.native_block_pool import NativeBlockPool

def test_growable_block_pool():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize with lightweight initial_blocks
    pool = NativeBlockPool(
        max_blocks=64,
        num_kv_heads=2,
        head_dim=128,
        rank=8,
        max_seq_len=256,
        device=device,
        dtype=torch.float16,
        initial_blocks=16
    )
    
    assert pool.initial_blocks == 16
    assert pool.current_blocks == 16
    assert len(pool._free_indices) == 16
    assert len(pool._ref_counts) == 16
    assert pool.U.shape[0] == 16
    
    # 2. Allocate 16 blocks (completely filling the initial pool)
    blocks_1 = pool.allocate_blocks(16)
    assert len(blocks_1) == 16
    assert len(pool._free_indices) == 0
    
    # 3. Allocate 1 more block (should trigger dynamic growth)
    idx = pool.allocate_block()
    
    # Check that pool grew up to the max_blocks cap of 64
    assert pool.current_blocks == 64
    assert pool.U.shape[0] == 64
    assert len(pool._ref_counts) == 64
    assert len(pool._free_indices) == 64 - 17 # 64 total - 17 active = 47 free indices remaining
    
    # 4. Fill the remaining free indices
    blocks_2 = pool.allocate_blocks(47)
    assert len(pool._free_indices) == 0
    
    # 5. Attempting to allocate beyond max_blocks should raise RuntimeError guard
    with pytest.raises(RuntimeError) as excinfo:
        pool.allocate_block()
    assert "absolute maximum limit" in str(excinfo.value)
    
    # 6. Reset the pool and verify VRAM shrinkage
    pool.reset()
    assert pool.current_blocks == 16
    assert pool.U.shape[0] == 16
    assert len(pool._free_indices) == 16
    assert len(pool._ref_counts) == 16

if __name__ == "__main__":
    test_growable_block_pool()
    print("[PASS] test_growable_block_pool")
