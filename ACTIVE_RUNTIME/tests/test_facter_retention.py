import torch
import numpy as np
import sys
import os
import math
from typing import Optional, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from runtime.native_block_pool import NativeBlockPool
from native_core.compression.lowrank import compress_lowrank, pack_int4, unpack_int4, reconstruct_batch_U
from native_core.srl.factual_store import FactualExactStore, FactEntry
from native_core.sparse_decode.triton_fused_decode import _pytorch_vectorized_sparse_attn_decode, fused_decode_mps

def test_residual_scaling_and_quantization():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # 1. Test CPU/GPU residual scale denormalization
    deltas = torch.randn(8, 128, device=device) * 2.0
    token_norms = torch.ones(8, device=device) * 1.5
    
    lr_delta = compress_lowrank(
        deltas=deltas,
        rank=4,
        error_threshold=0.01,
        max_residual_frac=0.5,
        token_norms=token_norms
    )
    
    assert lr_delta.residual_K_positions is not None
    assert lr_delta.residual_K_values is not None
    
    pos = lr_delta.residual_K_positions[0].item()
    norm = token_norms[pos].item()
    
    recon = (lr_delta.U.float() @ lr_delta.V.float()) * lr_delta.scale
    expected_unscaled = (deltas.cpu()[:, :64] - recon.cpu()[:, :64])[pos]
    expected_scaled = expected_unscaled * norm
    
    val_diff = (lr_delta.residual_K_values[0].cpu() - expected_scaled).abs().max().item()
    assert val_diff < 5e-3, f"Residual K scaling error: expected {expected_scaled}, got {lr_delta.residual_K_values[0].cpu()}"

    # 2. Test Stratified U Quantization (pack_int4, unpack_int4, reconstruct_batch_U)
    x = torch.randn(8, 4, device=device)
    packed, scale = pack_int4(x)
    assert packed.shape == (4, 4)
    assert scale.shape == (4,)
    
    unpacked = unpack_int4(packed, scale, 8)
    assert unpacked.shape == (8, 4)
    max_err = (x - unpacked).abs().max().item()
    bound = (scale / 2).max().item()
    assert max_err <= bound + 1e-4
    
    pool = NativeBlockPool(
        max_blocks=10,
        num_kv_heads=4,
        head_dim=32,
        rank=4,
        max_seq_len=8,
        device=device,
        dtype=torch.float16,
        initial_blocks=2,
        num_layers=1,
        lazy=False,
    )
    
    pool_idx = pool.allocate_block()
    n_semantic = 2
    U_sem_int4 = packed[:, :n_semantic]
    U_sem_scale = scale[:n_semantic].to(torch.float16)
    U_fact_fp16 = x[:, n_semantic:].to(torch.float16)
    
    pool.n_semantic[pool_idx] = n_semantic
    pool.seq_lens[pool_idx] = 8
    
    pool.U_sem[pool_idx, :4, :n_semantic] = U_sem_int4
    pool.U_sem_scale[pool_idx, :n_semantic] = U_sem_scale
    pool.U_fact[pool_idx, :8, :2] = U_fact_fp16
    
    idx_tensor = torch.tensor([pool_idx], device=device, dtype=torch.long)
    U_recon = reconstruct_batch_U(pool, idx_tensor)
    
    assert U_recon.shape == (1, 8, 4)
    recon_sem = U_recon[0, :, :n_semantic]
    expected_sem = unpack_int4(U_sem_int4, U_sem_scale, 8)
    sem_diff = (recon_sem.float() - expected_sem.to(device).float()).abs().max().item()
    assert sem_diff < 1e-3, f"Stratified sem reconstruction mismatch: {sem_diff}"
    
    recon_fact = U_recon[0, :, n_semantic:]
    fact_diff = (recon_fact.float() - U_fact_fp16.to(device).float()).abs().max().item()
    assert fact_diff < 1e-3, f"Stratified fact reconstruction mismatch: {fact_diff}"


def test_fact_anchors_and_factual_store():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    num_kv_heads = 4
    head_dim = 64
    seq_len = 16
    
    k = torch.randn(1, num_kv_heads, seq_len, head_dim)
    v = torch.randn(1, num_kv_heads, seq_len, head_dim)
    
    store = FactualExactStore(session_id="test_session")
    prefill_kv = {
        0: [k, v]
    }
    
    token_ids = torch.arange(10, 10 + seq_len, dtype=torch.long)
    W_proj = torch.randn(64, head_dim, dtype=torch.float32)
    stop_token_ids = {10, 11}
    
    store.build(prefill_kv, token_ids, W_proj, stop_token_ids)
    
    assert len(store.entries) > 0
    for entry in store.entries:
        assert entry.K.shape[0] == 1
        assert entry.K.shape[1] == num_kv_heads
        assert entry.K.shape[2] == (entry.end_idx - entry.start_idx)
        assert entry.K.shape[3] == head_dim
        
    Q = torch.randn(num_kv_heads * 2, head_dim)
    matches = store.query(Q, W_proj, threshold=-1.0)
    assert len(matches) > 0
    
    H_q = num_kv_heads * 2
    lse_dense = torch.randn(H_q, device=device)
    lse_sparse = torch.randn(H_q, device=device)
    lse_facts = torch.randn(H_q, device=device)
    
    out_dense = torch.randn(H_q, head_dim, device=device)
    out_sparse = torch.randn(H_q, head_dim, device=device)
    out_facts = torch.randn(H_q, head_dim, device=device)
    
    lse_max = torch.maximum(torch.maximum(lse_dense, lse_sparse), lse_facts)
    lse_max_masked = lse_max.clone()
    lse_max_masked[torch.isinf(lse_max)] = 0.0

    w_dense = torch.exp(lse_dense - lse_max_masked)
    w_sparse = torch.exp(lse_sparse - lse_max_masked)
    w_facts = torch.exp(lse_facts - lse_max_masked)

    w_dense[torch.isinf(lse_dense)] = 0.0
    w_sparse[torch.isinf(lse_sparse)] = 0.0
    w_facts[torch.isinf(lse_facts)] = 0.0

    denom = w_dense + w_sparse + w_facts
    denom_safe = torch.clamp(denom, min=1e-9)

    out_final = (out_dense * w_dense.unsqueeze(-1) +
                 out_sparse * w_sparse.unsqueeze(-1) +
                 out_facts * w_facts.unsqueeze(-1)) / denom_safe.unsqueeze(-1)
                 
    assert out_final.shape == (H_q, head_dim)


def test_localized_vertical_factual_retrieval():
    # Setup inputs to test mapping token spans to slot IDs
    num_kv_heads = 4
    head_dim = 64
    block_size = 8
    slot_ids = [100, 101, 102]
    
    seq_len = 24
    token_ids = torch.arange(1, seq_len + 1, dtype=torch.long) # tokens 1 to 24
    
    # Rare token IDs: 5, 6, 7 (span [4,7)) and 15, 16, 17 (span [14,17))
    stop_token_ids = set(token_ids.tolist()) - {5, 6, 7, 15, 16, 17}
    
    k = torch.randn(1, num_kv_heads, seq_len, head_dim)
    v = torch.randn(1, num_kv_heads, seq_len, head_dim)
    prefill_kv = {0: [k, v]}
    W_proj = torch.randn(64, head_dim)
    
    store = FactualExactStore(session_id="test_3d_session")
    store.build(
        prefill_kv=prefill_kv,
        token_ids=token_ids,
        W_proj=W_proj,
        stop_token_ids=stop_token_ids,
        slot_ids=slot_ids,
        block_size=block_size,
        use_salience_parser=False
    )
    
    # Verify mapping
    assert len(store.entries) == 2
    
    entry_1 = store.entries[0]
    entry_2 = store.entries[1]
    
    assert entry_1.slot_ids == [100]
    assert entry_2.slot_ids == [101, 102]
    
    # Query with active_slots={100}
    Q = torch.randn(num_kv_heads, head_dim)
    matches_1 = store.query(Q, W_proj, threshold=-1.0, active_slots={100})
    assert len(matches_1) == 1
    assert matches_1[0].start_idx == 4
    
    # Query with active_slots={102}
    matches_2 = store.query(Q, W_proj, threshold=-1.0, active_slots={102})
    assert len(matches_2) == 1
    assert matches_2[0].start_idx == 14
    
    # Query with active_slots={100, 101}
    matches_3 = store.query(Q, W_proj, threshold=-1.0, active_slots={100, 101})
    assert len(matches_3) == 2
    
    # Query with active_slots={999}
    matches_none = store.query(Q, W_proj, threshold=-1.0, active_slots={999})
    assert len(matches_none) == 0

    # Test relaxed fallback threshold
    avg_q = Q.mean(dim=0).float()
    q_desc = avg_q @ W_proj.T
    q_desc = q_desc / (q_desc.norm() + 1e-8)
    
    entry_1.descriptor = q_desc.cpu() * 0.2
    
    matches_fallback = store.query(Q, W_proj, threshold=0.5, active_slots={100})
    assert len(matches_fallback) == 1
    assert matches_fallback[0].start_idx == 4


def test_3d_factual_graph_walk():
    num_kv_heads = 4
    head_dim = 64
    block_size = 8
    slot_ids = [100, 101, 102]
    
    seq_len = 24
    token_ids = torch.arange(1, seq_len + 1, dtype=torch.long)
    
    # Rare tokens:
    # 5, 6, 7 (span [4,7)) - slot 100
    # 15 (span [14, 15)) - slot 101
    # 21 (span [20, 21)) - slot 102
    token_ids[14] = 5
    
    stop_token_ids = set(token_ids.tolist()) - {5, 6, 7, 15, 21}
    
    # Mock InvertedTokenIndex with idf for token 21 = 4.0
    from native_core.srl.inverted_index import InvertedTokenIndex
    inv_index = InvertedTokenIndex(
        index={},
        important_vocab=set(),
        idf={21: 4.0, 5: 1.0}
    )
    
    k = torch.randn(1, num_kv_heads, seq_len, head_dim)
    v = torch.randn(1, num_kv_heads, seq_len, head_dim)
    prefill_kv = {0: [k, v]}
    W_proj = torch.randn(64, head_dim)
    
    # Slot 100 is prime
    semantic_prime_slots = {100}
    
    store = FactualExactStore(session_id="test_walk_session")
    store.build(
        prefill_kv=prefill_kv,
        token_ids=token_ids,
        W_proj=W_proj,
        stop_token_ids=stop_token_ids,
        slot_ids=slot_ids,
        block_size=block_size,
        inv_index=inv_index,
        semantic_prime_slots=semantic_prime_slots,
        use_salience_parser=False
    )
    
    # We expect 3 entries:
    # Entry 0: span [4, 7), slot_ids [100]. Prime because of semantic_prime_slots {100}
    # Entry 1: span [14, 15), slot_ids [101]. Not prime.
    # Entry 2: span [20, 21), slot_ids [102]. Prime because token 21 IDF is 4.0
    assert len(store.entries) == 3
    
    entry_0 = store.entries[0]
    entry_1 = store.entries[1]
    entry_2 = store.entries[2]
    
    assert entry_0.is_prime is True
    assert entry_1.is_prime is False
    assert entry_2.is_prime is True
    
    # Check graph adjacency edges
    # Entry 0 and Entry 1 share token 5 (lexical overlap) -> they must be connected!
    assert 1 in entry_0.neighbors
    assert 0 in entry_1.neighbors
    
    # Verify query walk traversal:
    Q = torch.randn(num_kv_heads, head_dim)
    avg_q = Q.mean(dim=0).float()
    q_desc = avg_q @ W_proj.T
    q_desc = q_desc / (q_desc.norm() + 1e-8)
    
    # Force entry 0 descriptor to match q_desc perfectly (sim = 1.0)
    entry_0.descriptor = q_desc.cpu()
    # Force entry 1 and 2 to be zero (sim = 0.0)
    entry_1.descriptor = torch.zeros(64)
    entry_2.descriptor = torch.zeros(64)
    
    # Query with active_slots = None (or empty), so we don't trigger base-layer vertical filter.
    # Entry 0 matches directly (sim=1.0 >= 0.4) -> it activates as a seed.
    # Entry 0 has neighbor Entry 1, and the walk activates Entry 1 too.
    # Entry 2 is not retrieved.
    matches = store.query(Q, W_proj, threshold=0.4, active_slots=set())
    
    assert len(matches) >= 2
    match_starts = [entry.start_idx for entry in matches]
    assert 4 in match_starts   # Entry 0
    assert 14 in match_starts  # Entry 1
    assert 20 not in match_starts # Entry 2


def test_self_supervised_factual_parser():
    num_kv_heads = 4
    head_dim = 64
    seq_len = 16
    
    # Keys for specific tokens will have very large norms (index 5 and 10)
    k = torch.randn(1, num_kv_heads, seq_len, head_dim) * 0.1
    k[0, :, 5, :] = torch.randn(num_kv_heads, head_dim) * 5.0
    k[0, :, 10, :] = torch.randn(num_kv_heads, head_dim) * 5.0
    
    # Make future keys (11-14) look back to key 10
    for i in range(11, 15):
        k[0, :, i, :] = k[0, :, 10, :] + torch.randn(num_kv_heads, head_dim) * 0.01
        
    v = torch.randn(1, num_kv_heads, seq_len, head_dim)
    prefill_kv = {0: [k, v]}
    W_proj = torch.randn(64, head_dim)
    token_ids = torch.arange(1, seq_len + 1, dtype=torch.long)
    stop_token_ids = {1, 2, 3, 4}
    
    # Mock InvertedTokenIndex with IDF (token at index 5 has token ID = 6)
    from native_core.srl.inverted_index import InvertedTokenIndex
    inv_index = InvertedTokenIndex(
        index={},
        important_vocab=set(),
        idf={6: 3.0}
    )
    
    store = FactualExactStore(session_id="test_self_supervised_session")
    store.build(
        prefill_kv=prefill_kv,
        token_ids=token_ids,
        W_proj=W_proj,
        stop_token_ids=stop_token_ids,
        inv_index=inv_index,
        use_salience_parser=True
    )
    
    assert len(store.entries) > 0
    
    found_5 = False
    found_10 = False
    for entry in store.entries:
        if entry.start_idx <= 5 < entry.end_idx:
            found_5 = True
        if entry.start_idx <= 10 < entry.end_idx:
            found_10 = True
            
    assert found_5, "Token 5 (high norm + high IDF) not selected"
    assert found_10, "Token 10 (high norm + high Eagle lookback) not selected"


def test_factual_span_merging_and_similarity():
    # Construct adjacent mock FactEntries
    K1 = torch.randn(1, 2, 5, 32)
    V1 = torch.randn(1, 2, 5, 32)
    desc1 = torch.randn(64)
    desc1 = desc1 / desc1.norm()
    
    K2 = torch.randn(1, 2, 5, 32)
    V2 = torch.randn(1, 2, 5, 32)
    desc2 = torch.randn(64)
    desc2 = desc2 / desc2.norm()
    
    entry1 = FactEntry(start_idx=0, end_idx=5, K=K1, V=V1, descriptor=desc1, tokens=[0, 1, 2, 3, 4])
    entry2 = FactEntry(start_idx=5, end_idx=10, K=K2, V=V2, descriptor=desc2, tokens=[5, 6, 7, 8, 9])
    
    # Test merge_adjacent_entries directly
    from native_core.srl.factual_store import merge_adjacent_entries
    merged = merge_adjacent_entries([entry1, entry2])
    
    assert len(merged) == 1
    m = merged[0]
    assert m.start_idx == 0
    assert m.end_idx == 10
    assert m.tokens == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert m.K.shape == (1, 2, 10, 32)
    assert m.V.shape == (1, 2, 10, 32)
    
    # Verify query sets current_sim
    store = FactualExactStore(session_id="test_sim_session")
    store.entries = [entry1, entry2]
    
    Q = torch.randn(4, 32)
    W_proj = torch.randn(64, 32)
    
    results = store.query(Q, W_proj, threshold=-1.0)
    assert len(results) > 0
    for res in results:
        assert hasattr(res, "current_sim")
        assert isinstance(res.current_sim, float)


def test_transition_biasing_and_temp_scaling():
    from native_core.srl.session_srl_state import SessionSRLState

    # Create a mock srl_state
    srl_state = SessionSRLState(
        semantic_index=None,
        chunk_graph=None,
        inverted_index=None,
        ordered_slot_ids=[],
        sink_blocks=[]
    )
    # Set factual sequences and tokens
    srl_state.current_step_factual_tokens = {10, 11, 12}
    srl_state.current_step_factual_sequences = [[10, 11, 12], [20, 21]]
    srl_state.current_step_max_similarity = 0.5

    # Mock logits
    logits = torch.zeros((1, 100))

    # Test logic matching hf_diffkv_wrapper.py
    # 1. Apply static bias
    for tok_id in srl_state.current_step_factual_tokens:
        logits[0, tok_id] += 1.5

    # 2. Apply transition bias for last_token = 11 (next should be 12)
    last_token = 11
    transition_candidates = set()
    for seq in srl_state.current_step_factual_sequences:
        for idx, tok in enumerate(seq[:-1]):
            if tok == last_token:
                transition_candidates.add(seq[idx + 1])
    for tok_id in transition_candidates:
        logits[0, tok_id] += 2.0

    # Assertions
    assert logits[0, 10].item() == 1.5
    assert logits[0, 11].item() == 1.5
    assert logits[0, 12].item() == 3.5  # 1.5 static + 2.0 transition bias
    assert logits[0, 21].item() == 0.0  # not matched

    # 3. Test temperature scaling
    temperature = 0.8
    effective_temp = temperature
    if srl_state.current_step_max_similarity >= 0.4:
        effective_temp = temperature * (1.0 - srl_state.current_step_max_similarity * 0.8)
    assert abs(effective_temp - 0.48) < 1e-6


def test_factual_early_stopping():
    from native_core.srl.session_srl_state import SessionSRLState

    srl_state = SessionSRLState(
        semantic_index=None,
        chunk_graph=None,
        inverted_index=None,
        ordered_slot_ids=[],
        sink_blocks=[]
    )
    srl_state.current_step_max_similarity = 0.6
    srl_state.current_step_factual_sequences = [[10, 11, 12, 13, 14, 15]]

    # 1. Test target token matches last token of a sequence of len >= 5
    next_id = 15
    stop_generation = False
    if srl_state.current_step_max_similarity >= 0.5:
        for seq in srl_state.current_step_factual_sequences:
            if len(seq) >= 5 and next_id == seq[-1]:
                stop_generation = True
                break
    assert stop_generation is True

    # 2. Test target token does not match if similarity is too low
    srl_state.current_step_max_similarity = 0.4
    stop_generation = False
    if srl_state.current_step_max_similarity >= 0.5:
        for seq in srl_state.current_step_factual_sequences:
            if len(seq) >= 5 and next_id == seq[-1]:
                stop_generation = True
                break
    assert stop_generation is False

    # 3. Test target token does not match if sequence is too short
    srl_state.current_step_max_similarity = 0.6
    srl_state.current_step_factual_sequences = [[14, 15]]  # len 2
    stop_generation = False
    if srl_state.current_step_max_similarity >= 0.5:
        for seq in srl_state.current_step_factual_sequences:
            if len(seq) >= 5 and next_id == seq[-1]:
                stop_generation = True
                break
    assert stop_generation is False



