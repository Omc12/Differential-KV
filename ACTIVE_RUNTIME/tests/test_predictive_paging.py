import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time
from native_core.paging.paged_kv_store import PagedKVStore, BlockResidency

class MockTensor:
    def __init__(self, shape, is_cuda=True):
        self.shape = shape
        self.is_cuda = is_cuda
    def numel(self):
        import numpy as np
        return int(np.prod(self.shape))
    def element_size(self):
        return 2
    def to(self, device, non_blocking=False):
        self.is_cuda = (device == "cuda")
        return self

class MockBlock:
    def __init__(self, pool_idx):
        self.pool_idx = pool_idx
        self.anchor_kv = MockTensor((1, 2, 8, 128), is_cuda=True)
        self.U = MockTensor((255, 16), is_cuda=True)
        self.V = MockTensor((16, 8, 128), is_cuda=True)
        self.active_k = None
        self.active_v = None

def test_predictive_paging(monkeypatch):
    # TWO flags are needed, and only one was set. DKV_PREDICTIVE_PAGING gates
    # prefetch() ENQUEUEING the request; DKV_PAGER_BG_PREFETCH starts the thread
    # that DRAINS the queue (paged_kv_store.py:135) and is opt-in, default off,
    # because it mutates residency on its own thread. With only the first set the
    # request was queued and nobody ever read it, so the block stayed on CPU
    # forever -- not a slow prefetch, no prefetch at all.
    #
    # monkeypatch.setenv rather than a bare os.environ write: the old form leaked
    # DKV_PREDICTIVE_PAGING=1 into every test that ran afterwards in the process,
    # the same class of cross-test contamination as test_4k's HAS_TRITON.
    monkeypatch.setenv("DKV_PREDICTIVE_PAGING", "1")
    monkeypatch.setenv("DKV_PAGER_BG_PREFETCH", "1")

    # Init store with small budget (roughly 2 blocks worth of memory)
    store = PagedKVStore(gpu_budget_gb=0.0001, device="cuda")
    
    blocks = [MockBlock(i) for i in range(5)]
    
    # Register blocks (this will exceed budget and trigger eviction eventually)
    for i, b in enumerate(blocks):
        store.register_block("sess-0", 0, i, b)
        
    store.maybe_evict()
    summary = store.summary()
    print("Post-eviction summary:", summary)
    
    # Find which blocks were evicted (should be 0, 1, 2)
    evicted_keys = [k for k, entry in store._entries.items() if entry.residency == BlockResidency.CPU]
    assert len(evicted_keys) > 0, "No blocks were evicted under memory pressure!"
    print("Evicted keys:", evicted_keys)
    
    # Prefetch one evicted block
    target_key = evicted_keys[0]
    _, _, block_idx = target_key
    print(f"Issuing prefetch for block {block_idx}...")
    store.prefetch("sess-0", 0, block_idx)
    
    # POLL, don't sleep a fixed 0.1s and hope. This waits on a background thread,
    # so a fixed sleep is a race that fails on a loaded machine and says nothing
    # about the code. Polling to a generous deadline is fast when it works and
    # only slow when it is genuinely about to fail.
    entry = store._entries[target_key]
    deadline = time.time() + 5.0
    while time.time() < deadline and entry.residency != BlockResidency.GPU:
        time.sleep(0.01)
    assert entry.residency == BlockResidency.GPU, "Prefetch did not reload the block to GPU!"
    assert entry.prefetched is True, "entry.prefetched flag not set!"
    
    # Access it (touch)
    store.touch("sess-0", 0, block_idx)
    assert entry.prefetched is False, "prefetched flag should be reset after touch!"
    
    stats = store.summary()
    print("Final stats:", stats)
    assert stats["prefetch_hits"] == 1, f"Expected 1 prefetch hit, got {stats['prefetch_hits']}"
    print("SUCCESS: Predictive prefetching test passed!")
    store.stop()

if __name__ == "__main__":
    test_predictive_paging()
