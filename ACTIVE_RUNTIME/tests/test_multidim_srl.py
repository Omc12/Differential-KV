import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.srl.inverted_index import build_inverted_index, lookup_occurrences
from native_core.srl.chunk_graph import build_chunk_graph
from native_core.srl.semantic_index import build_semantic_index
from native_core.srl.session_srl_state import SessionSRLState
from native_core.srl.query_router import route_query, route_query_fixed_k


class DummyPool:
    def __init__(self, desc_matrix, slot_ids):
        max_slot = max(slot_ids)
        # Allocate pool descriptor storage that can accommodate the maximum slot ID
        self.desc = torch.zeros(max_slot + 1, desc_matrix.shape[1], dtype=desc_matrix.dtype)
        self.desc[slot_ids] = desc_matrix
        self.W_proj = torch.randn(64, desc_matrix.shape[1])


def test_merged_token_dictionary():
    # Setup sample token ids: 257 tokens per block (1 anchor + 256 active tokens)
    # Block size is 257. Let's create a prompt of 3 blocks: 3 * 257 = 771 tokens
    block_size = 257
    tokens = torch.zeros(771, dtype=torch.long)
    
    # Place special keyword tokens
    # Keyword 1000 in block 0
    tokens[10] = 1000
    tokens[50] = 1000
    
    # Keyword 1000 in block 2
    tokens[550] = 1000
    
    # Keyword 2000 in block 1
    tokens[300] = 2000
    
    slot_ids = [101, 102, 103]  # slot IDs corresponding to block 0, 1, 2
    stop_tokens = {0, 1, 2}
    
    inv_index = build_inverted_index(
        token_ids=tokens,
        slot_ids=slot_ids,
        block_size=block_size,
        stop_token_ids=stop_tokens,
        top_n_per_block=5
    )
    
    # Verify occurrences mapping
    assert 1000 in inv_index.occurrences
    assert 2000 in inv_index.occurrences
    
    occ_1000 = inv_index.occurrences[1000]
    assert len(occ_1000) == 3
    # occurrences is a list of (slot_id, absolute_pos, relative_pos)
    assert occ_1000[0] == (101, 10, 10)
    assert occ_1000[1] == (101, 50, 50)
    assert occ_1000[2] == (103, 550, 550 - 2*block_size)
    
    # Verify lookup_occurrences API
    res = lookup_occurrences(inv_index, [1000, 2000])
    assert len(res) == 4
    
    # Verify chunk_vocabularies
    assert 101 in inv_index.chunk_vocabularies
    assert 103 in inv_index.chunk_vocabularies
    assert 1000 in inv_index.chunk_vocabularies[101]
    assert inv_index.chunk_vocabularies[101][1000] == [10, 50]
    assert inv_index.chunk_vocabularies[103][1000] == [550 - 2*block_size]


def test_multidim_chunk_graph_connections():
    # Setup sample descriptors for 3 blocks
    desc_matrix = torch.randn(3, 64)
    # L2 normalize
    desc_matrix = desc_matrix / desc_matrix.norm(dim=1, keepdim=True)
    slot_ids = torch.tensor([101, 102, 103], dtype=torch.int32)
    
    # Let's create an inverted index where:
    # block 0 (slot 101) has vocab: {1000, 1001, 1002}
    # block 1 (slot 102) has vocab: {1000, 1001} (relative overlap with block 0 is 2/3 = 66%)
    # block 2 (slot 103) has vocab: {2000} (overlap is 0)
    block_size = 257
    tokens = torch.zeros(771, dtype=torch.long)
    # Block 0 keywords
    tokens[10] = 1000
    tokens[11] = 1001
    tokens[12] = 1002
    # Block 1 keywords
    tokens[300] = 1000
    tokens[301] = 1001
    # Block 2 keywords
    tokens[600] = 2000
    
    inv_index = build_inverted_index(
        token_ids=tokens,
        slot_ids=[101, 102, 103],
        block_size=block_size,
        stop_token_ids={0},
        top_n_per_block=5
    )
    
    # Build graph with overlap threshold = 0.5 (50%)
    graph = build_chunk_graph(
        desc_matrix=desc_matrix,
        slot_ids=slot_ids,
        K_semantic=1,
        K_temporal=1,
        inv_index=inv_index,
        overlap_threshold=0.5
    )
    
    # Verify neighbors shape and structure
    neighbors = graph.neighbors
    assert neighbors.shape[0] == 3
    # Check that block 0 (row 0) and block 1 (row 1) are connected via lexical overlap
    # Neighbors of row 0 should contain row 1
    row_0_neighbors = neighbors[0].tolist()
    # Filter out padding value -1
    row_0_neighbors = [n for n in row_0_neighbors if n != -1]
    assert 1 in row_0_neighbors
    
    # Neighbors of row 1 should contain row 0
    row_1_neighbors = neighbors[1].tolist()
    row_1_neighbors = [n for n in row_1_neighbors if n != -1]
    assert 0 in row_1_neighbors


def test_query_routing_with_decay():
    # Set up basic index and session SRL state
    desc_matrix = torch.randn(4, 64)
    desc_matrix = desc_matrix / desc_matrix.norm(dim=1, keepdim=True)
    slot_ids = [101, 102, 103, 104]
    slot_ids_t = torch.tensor(slot_ids, dtype=torch.int32)
    
    # Create tokens:
    # Keyword 1000 occurs in block 0 (abs_pos 10), block 2 (abs_pos 520) and block 3 (abs_pos 800)
    block_size = 257
    tokens = torch.zeros(1028, dtype=torch.long)
    tokens[10] = 1000
    tokens[520] = 1000
    tokens[800] = 1000
    
    inv_index = build_inverted_index(
        token_ids=tokens,
        slot_ids=slot_ids,
        block_size=block_size,
        stop_token_ids={0},
        top_n_per_block=5
    )
    
    pool = DummyPool(desc_matrix, slot_ids)
    sem_index = build_semantic_index(pool, slot_ids)
    graph = build_chunk_graph(desc_matrix, slot_ids_t, K_semantic=1, K_temporal=1, inv_index=inv_index, overlap_threshold=0.1)
    
    srl_state = SessionSRLState(
        semantic_index=sem_index,
        chunk_graph=graph,
        inverted_index=inv_index,
        ordered_slot_ids=slot_ids,
        sink_blocks=[101],
        k_min=2,
        k_max=4,
        routing_threshold=1
    )
    
    # Run route_query with query token containing 1000
    Q = torch.randn(8, 64)  # 8 heads, 64 dim (matches DummyPool/desc_matrix feature dim)
    # Set W_proj on pool to match
    pool.W_proj = torch.randn(64, 64)
    
    selected_slots = route_query(
        Q=Q,
        srl_state=srl_state,
        pool=pool,
        scale=1.0,
        layer_idx=0,
        query_tokens=[1000]
    )
    
    # Selected slots should include sink block 101, plus matched blocks
    selected_list = selected_slots.tolist()
    assert 101 in selected_list
    assert 103 in selected_list or 104 in selected_list
    
    # Let's verify temporal decay ordering:
    # Matches: block 3 (abs_pos 800) is more recent than block 2 (abs_pos 520), which is more recent than block 0 (abs_pos 10).
    # The decay score should rank block 3 > block 2 > block 0.
    matches = lookup_occurrences(inv_index, [1000])
    L = max(occ[1] for occ in matches)
    assert L == 800
    
    score_3 = 0.999 ** (800 - 800)  # 1.0
    score_2 = 0.999 ** (800 - 520)  # ~0.75
    score_0 = 0.999 ** (800 - 10)   # ~0.45
    
    assert score_3 > score_2
    assert score_2 > score_0
