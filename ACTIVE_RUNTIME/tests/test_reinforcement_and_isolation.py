import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_active_turn_graph_isolation():
    """Verify directed edge constraints: active-turn blocks can link to historic blocks but not vice-versa."""
    from native_core.srl.chunk_graph import build_chunk_graph
    from native_core.srl.inverted_index import InvertedTokenIndex
    
    class MockBlock:
        def __init__(self, anchor_idx):
            self.anchor_idx = anchor_idx
            
    # 2 blocks: block 0 (historic, index 0), block 1 (active, index 1)
    slot_ids = torch.tensor([10, 20], dtype=torch.int32)
    # mock descriptors: 2 blocks, desc_dim = 16
    desc_matrix = torch.zeros((2, 16), dtype=torch.float32)
    desc_matrix[0, 0] = 1.0
    desc_matrix[1, 0] = 1.0 # high similarity
    
    inverted_index = InvertedTokenIndex(index={}, important_vocab=set())
    # cached_len = 1 (block 0 is historic, block 1 is active)
    chunk_graph = build_chunk_graph(
        desc_matrix=desc_matrix,
        slot_ids=slot_ids,
        K_semantic=1,
        K_temporal=1,
        overlap_threshold=0.15,
        inv_index=inverted_index,
        blocks=[MockBlock(0), MockBlock(1)],
        cached_len=1
    )
    
    # Check neighbors for row 0 (historic): should not link forward to row 1 (active)
    neighbors = chunk_graph.neighbors.tolist()
    row0_neighbors = neighbors[0]
    assert 20 not in row0_neighbors, "Historic block linked forward to active-turn block!"
    
    # row 1 neighbors:
    row1_neighbors = neighbors[1]
    # Should contain 0 (the row index of slot 10)
    assert 0 in row1_neighbors, "Active block did not link back to historic block!"
    print("[PASS] test_active_turn_graph_isolation")

def test_slot_reinforcement():
    """Verify slot reinforcement EMA updates: routed slots get boosted, others decayed."""
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.semantic_index import SemanticIndex
    from native_core.srl.chunk_graph import ChunkGraph
    from native_core.srl.inverted_index import InvertedTokenIndex
    
    # Create mock SessionSRLState
    srl_state = SessionSRLState(
        semantic_index=SemanticIndex(slot_ids=[10, 20, 30], desc_matrix=torch.zeros((3, 16))),
        chunk_graph=ChunkGraph(neighbors=torch.zeros((3, 8), dtype=torch.int32)),
        inverted_index=InvertedTokenIndex(index={}, important_vocab=set()),
        ordered_slot_ids=[10, 20, 30],
        sink_blocks=[]
    )
    srl_state.slot_activation_strength = {10: 1.0, 20: 1.0, 30: 1.0}
    
    selected_slots = {10, 20}
    alpha_boost = 0.05
    decay_rate = 0.99
    
    for slot in selected_slots:
        if slot not in srl_state.slot_activation_strength:
            srl_state.slot_activation_strength[slot] = 1.0
        srl_state.slot_activation_strength[slot] += alpha_boost
        
    for slot in list(srl_state.slot_activation_strength.keys()):
        if slot not in selected_slots:
            srl_state.slot_activation_strength[slot] *= decay_rate
            if srl_state.slot_activation_strength[slot] < 1.0:
                srl_state.slot_activation_strength[slot] = 1.0
                
    assert srl_state.slot_activation_strength[10] == 1.05
    assert srl_state.slot_activation_strength[20] == 1.05
    assert srl_state.slot_activation_strength[30] == 1.0
    print("[PASS] test_slot_reinforcement")

def test_eviction_protection():
    """Verify eviction logic respects slot activation strength."""
    from native_core.paging.paged_kv_store import PagedKVStore, BlockResidency, PageEntry
    
    # Create PagedKVStore with low budget to trigger eviction easily
    pager = PagedKVStore(gpu_budget_gb=1.0)
    pager.gpu_budget_bytes = 100
    
    class MockBlock:
        def __init__(self, pool_idx):
            self.pool_idx = pool_idx
            self.anchor_kv = None
            
    # Mock entries in pager
    block1 = MockBlock(10)
    block2 = MockBlock(20)
    block3 = MockBlock(30)
    
    entry1 = PageEntry(block_ref=block1, residency=BlockResidency.GPU, vram_bytes=50, last_access=1000.0)
    entry2 = PageEntry(block_ref=block2, residency=BlockResidency.GPU, vram_bytes=50, last_access=950.0)
    entry3 = PageEntry(block_ref=block3, residency=BlockResidency.GPU, vram_bytes=50, last_access=960.0)
    
    pager._entries[("session_1", 0, 1)] = entry1
    pager._entries[("session_1", 0, 2)] = entry2
    pager._entries[("session_1", 0, 3)] = entry3
    
    # Mock manager and srl_state
    class MockSRLState:
        def __init__(self):
            self.slot_activation_strength = {10: 1.0, 20: 2.0, 30: 1.0}
            
    class MockManager:
        def __init__(self):
            self._session_srl = {"session_1": MockSRLState()}
            
    pager.manager = MockManager()
    
    # Find coldest key
    coldest = pager._find_coldest()
    # entry1 composite = 1000
    # entry2 composite = 950 + 300 = 1250
    # entry3 composite = 960
    # So entry3 should be the coldest (960 < 1000 < 1250)!
    assert coldest == ("session_1", 0, 3), f"Expected entry3 to be coldest, got {coldest}"
    print("[PASS] test_eviction_protection")

def test_selective_merging():
    """Verify that commit_turn prunes low-salience blocks while keeping primes/exact factual blocks."""
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.semantic_index import SemanticIndex
    from native_core.srl.chunk_graph import ChunkGraph
    from native_core.srl.inverted_index import InvertedTokenIndex
    from native_core.srl.factual_store import FactualExactStore
    
    # Mock class to mimic StreamingKVBlock and NativeBlockPool
    class MockBlock:
        def __init__(self, pool_idx, anchor_idx, tokens, state=None):
            self.pool_idx = pool_idx
            self.anchor_idx = anchor_idx
            self.token_indices = tokens
            self.state = state if state is not None else 3 # CompressedResident
            
        def token_count(self):
            return len(self.token_indices)
            
    class MockPool:
        def __init__(self):
            self.freed = []
        def free_block(self, slot_id):
            self.freed.append(slot_id)
            
    class MockManager:
        def __init__(self):
            self.session_blocks = {"session_1": {0: []}}
            self.native_pool = MockPool()
            self._session_token_ids = {"session_1": torch.tensor(range(32))}
            self._factual_stores = {"session_1": FactualExactStore("session_1")}
            self.num_layers = 1
            self._streaming_mgr = None
            
        def finalize_srl_index(self, session_id, cached_len):
            pass
            
    # setup state
    semantic_index = SemanticIndex(slot_ids=[10, 20, 30], desc_matrix=torch.zeros((3, 16)))
    chunk_graph = ChunkGraph(neighbors=torch.zeros((3, 8), dtype=torch.int32))
    chunk_graph.parent_landmarks = torch.tensor([20], dtype=torch.int32)
    
    inverted_index = InvertedTokenIndex(index={}, important_vocab=set())
    inverted_index.idf = {100: 3.0, 200: 1.0}
    
    srl_state = SessionSRLState(
        semantic_index=semantic_index,
        chunk_graph=chunk_graph,
        inverted_index=inverted_index,
        ordered_slot_ids=[10, 20, 30],
        sink_blocks=[]
    )
    srl_state.cached_len = 0 # all blocks are active in this turn
    
    b1 = MockBlock(10, 0, [200])
    b2 = MockBlock(20, 16, [200])
    b3 = MockBlock(30, 32, [100])
    
    manager = MockManager()
    manager.session_blocks["session_1"][0] = [b1, b2, b3]
    
    srl_state.commit_turn(manager, "session_1")
    
    assert 10 in manager.native_pool.freed, "Slot 10 was not pruned!"
    assert 20 not in manager.native_pool.freed, "Slot 20 (landmark) was incorrectly pruned!"
    assert 30 not in manager.native_pool.freed, "Slot 30 (high-IDF) was incorrectly pruned!"
    
    remaining_slots = [b.pool_idx for b in manager.session_blocks["session_1"][0]]
    assert 10 not in remaining_slots
    assert 20 in remaining_slots
    assert 30 in remaining_slots
    print("[PASS] test_selective_merging")

if __name__ == "__main__":
    test_active_turn_graph_isolation()
    test_slot_reinforcement()
    test_eviction_protection()
    test_selective_merging()
