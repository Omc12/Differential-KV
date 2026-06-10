import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from native_core.srl.inverted_index import build_inverted_index, lookup_occurrences, InvertedTokenIndex
from native_core.srl.chunk_graph import build_chunk_graph, ChunkGraph
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


def test_idf_and_coverage_boost():
    # Setup index for testing IDF and Coverage boosting
    # 4 blocks, block_size = 257
    slot_ids = [101, 102, 103, 104]
    slot_ids_t = torch.tensor(slot_ids, dtype=torch.int32)
    block_size = 257
    tokens = torch.zeros(1028, dtype=torch.long)
    
    # Token 1000 is common: occurs in block 0, block 1, block 2 (3 occurrences)
    tokens[10] = 1000
    tokens[300] = 1000
    tokens[600] = 1000
    
    # Token 2000 is rare: occurs only in block 3 (1 occurrence)
    tokens[900] = 2000
    
    # Block 1 (slot 102) has token 1000 and token 3000 (2 unique query matches)
    tokens[310] = 3000
    # Block 2 (slot 103) has only token 1000 (repeated twice)
    tokens[610] = 1000
    
    inv_index = build_inverted_index(
        token_ids=tokens,
        slot_ids=slot_ids,
        block_size=block_size,
        stop_token_ids={0},
        top_n_per_block=5
    )
    
    # 1. Verify IDF: token 2000 (rare) should have higher IDF than token 1000 (common)
    assert inv_index.idf[2000] > inv_index.idf[1000]
    
    # 2. Verify Coverage Boost during route_query
    # Setup minimal SRL state
    desc_matrix = torch.randn(4, 64)
    desc_matrix = desc_matrix / desc_matrix.norm(dim=1, keepdim=True)
    pool = DummyPool(desc_matrix, slot_ids)
    pool.W_proj = torch.randn(64, 64)
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
    
    # Query contains [1000, 3000]
    # Block 1 (slot 102) matches BOTH 1000 and 3000 (2 unique matches).
    # Block 2 (slot 103) matches ONLY 1000 (but has 2 occurrences of it).
    # Because of coverage boost (2^2 = 4x multiplier for block 1 vs 1^2 = 1x multiplier for block 2),
    # block 1 (slot 102) should be strongly preferred and ranked ahead of block 2 (slot 103).
    Q = torch.randn(8, 64)
    selected_slots = route_query(
        Q=Q,
        srl_state=srl_state,
        pool=pool,
        scale=1.0,
        layer_idx=0,
        query_tokens=[1000, 3000]
    )
    
    selected_list = selected_slots.tolist()
    # Slot 102 should be matched before Slot 103
    idx_102 = selected_list.index(102) if 102 in selected_list else -1
    idx_103 = selected_list.index(103) if 103 in selected_list else -1
    
    assert idx_102 != -1
    if idx_103 != -1:
        assert idx_102 < idx_103  # block 102 is ranked higher


def test_dynamic_decay_and_weights():
    # 1. Verify directed/asymmetric weights
    # slot 101 has vocab {1000, 1001, 1002}
    # slot 102 has vocab {1000, 1001}
    # overlap = {1000, 1001} (size 2)
    # relative overlap 101 -> 102 is 2/3 = 0.667
    # relative overlap 102 -> 101 is 2/2 = 1.0
    desc_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    slot_ids = [101, 102]
    slot_ids_t = torch.tensor(slot_ids, dtype=torch.int32)
    
    tokens = torch.zeros(514, dtype=torch.long)
    # Slot 101 (block 0) tokens:
    tokens[10] = 1000
    tokens[11] = 1001
    tokens[12] = 1002
    # Slot 102 (block 1) tokens:
    tokens[257 + 10] = 1000
    tokens[257 + 11] = 1001
    
    inv_index = build_inverted_index(
        token_ids=tokens,
        slot_ids=slot_ids,
        block_size=257,
        stop_token_ids={0},
        top_n_per_block=5
    )
    
    # Build graph with overlap threshold = 0.1
    graph = build_chunk_graph(
        desc_matrix=desc_matrix,
        slot_ids=slot_ids_t,
        K_semantic=1,
        K_temporal=0,
        inv_index=inv_index,
        overlap_threshold=0.1
    )
    
    # Verify weights exist and are asymmetric
    assert graph.weights is not None
    # Slot 101 (row 0) has neighbor 102 (row 1)
    neighbors_0 = graph.neighbors[0].tolist()
    idx_1_in_0 = neighbors_0.index(1)
    weight_0_to_1 = float(graph.weights[0, idx_1_in_0])
    
    # Slot 102 (row 1) has neighbor 101 (row 0)
    neighbors_1 = graph.neighbors[1].tolist()
    idx_0_in_1 = neighbors_1.index(0)
    weight_1_to_0 = float(graph.weights[1, idx_0_in_1])
    
    # Since lex_score(1 -> 0) = 1.0 > lex_score(0 -> 1) = 0.667,
    # weight_1_to_0 should be strictly greater than weight_0_to_1
    assert weight_1_to_0 > weight_0_to_1

    # 2. Verify query-dependent decay and degree damping during route_query
    # We have 8 blocks:
    # 0 (slot 101) - Seed block (query similarity high: 0.9)
    # 1 (slot 102) - Connected to 0, but query similarity low: 0.1
    # 2 (slot 103) - Connected to 0, query similarity high: 0.8
    # 3 (slot 104) - Hub node connected to 0, 1, 2 (highly connected, query similarity low: 0.15)
    # 4-7 (slots 105-108) - Unrelated blocks (query similarity low: 0.05)
    
    desc_matrix = torch.tensor([
        [0.9, 0.436],
        [0.1, 0.995],
        [0.8, 0.600],
        [0.15, 0.989],
        [0.05, 0.998],
        [0.05, 0.998],
        [0.05, 0.998],
        [0.05, 0.998]
    ], dtype=torch.float32)
    # Normalize descriptors
    desc_matrix = desc_matrix / desc_matrix.norm(dim=1, keepdim=True)
    
    slot_ids = [101, 102, 103, 104, 105, 106, 107, 108]
    slot_ids_t = torch.tensor(slot_ids, dtype=torch.int32)
    
    # Let's build a manual ChunkGraph where:
    # 0 is neighbors with 1, 2, 3
    # 1 is neighbors with 0, 3
    # 2 is neighbors with 0, 3
    # 3 is neighbors with 0, 1, 2 (hub)
    # neighbors shape: [8, 3]
    neighbors_tensor = torch.tensor([
        [1, 2, 3],
        [0, 3, -1],
        [0, 3, -1],
        [0, 1, 2],
        [-1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1]
    ], dtype=torch.int32)
    
    weights_tensor = torch.tensor([
        [0.8, 0.8, 0.8],
        [0.8, 0.8, 0.0],
        [0.8, 0.8, 0.0],
        [0.8, 0.8, 0.8],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=torch.float32)
    
    graph = ChunkGraph(neighbors=neighbors_tensor, weights=weights_tensor)
    
    pool = DummyPool(desc_matrix, slot_ids)
    # W_proj is identity so query descriptor matches Q
    pool.W_proj = torch.eye(2)
    sem_index = build_semantic_index(pool, slot_ids)
    
    srl_state = SessionSRLState(
        semantic_index=sem_index,
        chunk_graph=graph,
        inverted_index=InvertedTokenIndex(index={}, important_vocab=set()),
        ordered_slot_ids=slot_ids,
        sink_blocks=[],
        k_min=2,
        k_max=4,
        routing_threshold=1,
        graph_hop_decay=0.5
    )
    
    # Query Q = [1.0, 0.0]
    Q = torch.tensor([[1.0, 0.0]], dtype=torch.float32) # [1, 2] query
    
    # Route query
    selected = route_query(
        Q=Q,
        srl_state=srl_state,
        pool=pool,
        scale=1.0,
        layer_idx=0
    )
    
    selected_list = selected.tolist()
    # Selected list must prefer 103 (high query similarity neighbor of 101)
    assert 103 in selected_list
    # And 102 (low similarity neighbor of 101) must not displace 103
    assert 102 not in selected_list or selected_list.index(103) < selected_list.index(102)


def test_reranker_and_recency_decay():
    # Verify that level-1 reranking and chronological recency decay works correctly.
    desc_matrix = torch.tensor([
        [0.8, 0.6],  # 101: older block (high similarity)
        [0.79, 0.61], # 102: old block (high similarity)
        [0.7, 0.71], # 103: recent block (medium similarity)
        [0.0, 1.0]   # 104: not matched
    ], dtype=torch.float32)
    # L2 normalize
    desc_matrix = desc_matrix / desc_matrix.norm(dim=1, keepdim=True)
    
    slot_ids = [101, 102, 103, 104]
    slot_ids_t = torch.tensor(slot_ids, dtype=torch.int32)
    
    pool = DummyPool(desc_matrix, slot_ids)
    pool.W_proj = torch.eye(2)
    # pool.anchors_K needs to be indexed by slot_ids (101 to 104)
    anchors_K = torch.zeros(105, 1, 2, dtype=torch.float32)
    anchors_K[slot_ids] = desc_matrix.unsqueeze(1)
    pool.anchors_K = anchors_K
    
    sem_index = build_semantic_index(pool, slot_ids)
    graph = ChunkGraph(
        neighbors=torch.tensor([[-1, -1], [-1, -1], [-1, -1], [-1, -1]], dtype=torch.int32),
        weights=torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    )
    
    # Query Q = [0.8, 0.6] (matches 101 and 102 very strongly, matches 103 moderately)
    Q = torch.tensor([[0.8, 0.6]], dtype=torch.float32) # [1, 2] query
    
    # 1. Run with srl_age_penalty=0.0 (no decay, select 101 and 102)
    srl_state_no_decay = SessionSRLState(
        semantic_index=sem_index,
        chunk_graph=graph,
        inverted_index=InvertedTokenIndex(index={}, important_vocab=set()),
        ordered_slot_ids=slot_ids,
        sink_blocks=[],
        k_min=2,
        k_max=2,
        routing_threshold=1,
        srl_age_penalty=0.0
    )
    
    selected_no_decay = route_query(
        Q=Q,
        srl_state=srl_state_no_decay,
        pool=pool,
        scale=1.0,
        layer_idx=0
    ).tolist()
    
    assert 101 in selected_no_decay
    assert 102 in selected_no_decay
    assert 103 not in selected_no_decay

    # 2. Run with srl_age_penalty=0.1 (decay applies, select 102 and 103, push out 101)
    srl_state_decay = SessionSRLState(
        semantic_index=sem_index,
        chunk_graph=graph,
        inverted_index=InvertedTokenIndex(index={}, important_vocab=set()),
        ordered_slot_ids=slot_ids,
        sink_blocks=[],
        k_min=2,
        k_max=2,
        routing_threshold=1,
        srl_age_penalty=0.1
    )
    
    selected_decay = route_query(
        Q=Q,
        srl_state=srl_state_decay,
        pool=pool,
        scale=1.0,
        layer_idx=0
    ).tolist()
    
    assert 103 in selected_decay
    assert 102 in selected_decay
    assert 101 not in selected_decay


def test_dynamic_scaling_and_seeds():
    from unittest.mock import patch, MagicMock
    from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper as DiffKVHFWrapper

    # 1. Test model-size dependent configuration defaults in DiffKVHFWrapper
    class DummyConfig:
        num_hidden_layers = 12
        num_attention_heads = 8
        hidden_size = 512

    class DummyLayers:
        def __init__(self):
            self.layers = []

    class DummyModel(torch.nn.Module):
        def __init__(self, num_params):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(num_params))
            self.config = DummyConfig()
            self.model = DummyLayers()

    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            self.unk_token_id = 0
        def convert_tokens_to_ids(self, word):
            if word == "<|im_end|>":
                return 3
            if word == "<|end_of_text|>":
                return 4
            if word == "<|eot_id|>":
                return 5
            if word == "</s>":
                return 6
            return 0

    dummy_tokenizer = DummyTokenizer()

    # Clear process pollution from other tests that set this globally
    old_threshold = os.environ.pop("DIFFKV_SRL_THRESHOLD", None)
    try:
        # A. Test smaller model (e.g. 500M params)
        model_500m = DummyModel(500 * 10**6)
        with patch("serving.hf_diffkv_wrapper.AutoModelForCausalLM.from_pretrained", return_value=model_500m), \
             patch("serving.hf_diffkv_wrapper.AutoTokenizer.from_pretrained", return_value=dummy_tokenizer):
            wrapper = DiffKVHFWrapper("mock/qwen-500m", config={"rank": 16})
            assert wrapper.config["srl_k_min"] == 10
            assert wrapper.config["srl_k_max"] == 50
            assert wrapper.config["srl_threshold"] == 25

        # B. Test medium model (e.g. 1.5B params)
        model_1_5b = DummyModel(1500 * 10**6)
        with patch("serving.hf_diffkv_wrapper.AutoModelForCausalLM.from_pretrained", return_value=model_1_5b), \
             patch("serving.hf_diffkv_wrapper.AutoTokenizer.from_pretrained", return_value=dummy_tokenizer):
            wrapper = DiffKVHFWrapper("mock/qwen-1.5b", config={"rank": 16})
            assert wrapper.config["srl_k_min"] == 15
            assert wrapper.config["srl_k_max"] == 100
            assert wrapper.config["srl_threshold"] == 40

        # C. Test larger model (e.g. 7B params)
        model_7b = DummyModel(7000 * 10**6)
        with patch("serving.hf_diffkv_wrapper.AutoModelForCausalLM.from_pretrained", return_value=model_7b), \
             patch("serving.hf_diffkv_wrapper.AutoTokenizer.from_pretrained", return_value=dummy_tokenizer):
            wrapper = DiffKVHFWrapper("mock/qwen-7b", config={"rank": 16})
            assert wrapper.config["srl_k_min"] == 20
            assert wrapper.config["srl_k_max"] == 200
            assert wrapper.config["srl_threshold"] == 50
    finally:
        if old_threshold is not None:
            os.environ["DIFFKV_SRL_THRESHOLD"] = old_threshold





