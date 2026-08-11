import torch
import pytest
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
    
    # What this test is actually about is the token_norms DENORMALIZATION, so it
    # has to expect whichever residual FORM is in effect rather than hard-coding
    # one. It hard-coded the correction form (delta - recon); CUDA has since
    # defaulted to the MLX-parity EXACT form (delta), which stores a genuinely
    # different value per element -- so this failed while the scaling it checks
    # was perfectly correct.
    from native_core.compression.lowrank import _exact_keys_enabled
    recon = (lr_delta.U.float() @ lr_delta.V.float()) * lr_delta.scale
    if _exact_keys_enabled(device):
        expected_unscaled = deltas.cpu()[:, :64][pos]
    else:
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


@pytest.fixture
def _deterministic_numerics():
    """Make this test independent of WHAT RAN BEFORE IT. Two separate leaks:

    1. UNSEEDED RNG -- the real root cause. The test builds k, v and W_proj with
       bare torch.randn, and its final assertion turns on a cosine similarity
       crossing an explicit 0.5 threshold. Run alone the global RNG is at its
       default state and the draw happens to clear the threshold; run after ANY
       other test, those draws are consumed and different values land on the
       other side. Nothing about the store changed -- the inputs did.

    2. TF32 -- an independent second trigger. Importing DKV sets
       torch.set_float32_matmul_precision("high") GLOBALLY (a documented speed
       choice, DKV_TF32=0 disables it), so once any file pulls DKV in every later
       test runs at ~10-bit mantissa. Verified separately: running this test by
       itself, with the default RNG state but precision forced to 'high',
       reproduces the failure on its own.

    Both are pinned rather than the assertion being loosened, because the test is
    about the store's THRESHOLD LOGIC, not about which random matrix it got or
    how many mantissa bits the matmul used. Both are restored on teardown so this
    fixture cannot become the next leak.
    """
    torch.manual_seed(0)

    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def test_localized_vertical_factual_retrieval(_deterministic_numerics):
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
    # Mutating a descriptor in place breaks an invariant the store documents and
    # relies on: _ensure_desc_matrix caches the stacked [E, DESC] descriptors and
    # rebuilds ONLY when the entry COUNT changes ("entries are fixed after
    # prefill"). Four queries above have already populated that cache, so without
    # this the query below scores against entry_1's ORIGINAL descriptor and the
    # assignment has no effect at all.
    #
    # Production never hits this -- descriptors are written during build() and
    # never touched again, which is exactly why the cache is safe there. This is
    # a white-box test reaching past the public API, so it has to maintain the
    # invariant it just broke.
    store._desc_matrix = None

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

    # Test logic matching hf_dkv_wrapper.py
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


def test_strict_factual_alignment():
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.factual_alignment import is_token_id_allowed

    # Mock tokenizer
    class MockTokenizer:
        def decode(self, token_ids):
            mapping = {
                10: "codimension",
                11: "diabolical",
                12: "points",
                100: "the",
                101: "is",
                200: "quantum",
                201: "eigenvalues"
            }
            return mapping.get(token_ids[0], "unknown")

    tokenizer = MockTokenizer()
    
    srl_state = SessionSRLState(
        semantic_index=None,
        chunk_graph=None,
        inverted_index=None,
        ordered_slot_ids=[],
        sink_blocks=[]
    )
    srl_state.current_step_factual_tokens = {10, 11, 12}
    srl_state.current_step_factual_sequences = [[10, 11, 12]]
    srl_state.current_step_max_similarity = 0.5

    # 1. Factual tokens: only the start of the sequence is allowed initially
    assert is_token_id_allowed(10, srl_state, None, tokenizer) is True
    assert is_token_id_allowed(11, srl_state, None, tokenizer) is False

    # 2. If we simulate locking by manually setting active candidates
    srl_state.vsl_active_candidates = [[11, 12]]
    assert is_token_id_allowed(11, srl_state, 10, tokenizer) is True
    assert is_token_id_allowed(10, srl_state, 10, tokenizer) is False
    srl_state.vsl_active_candidates = []  # reset
    
    # 3. Allowed grammatical helpers are allowed
    assert is_token_id_allowed(100, srl_state, None, tokenizer) is True
    assert is_token_id_allowed(101, srl_state, None, tokenizer) is True

    # 4. Ungrounded content/technical tokens are NOT allowed
    assert is_token_id_allowed(200, srl_state, None, tokenizer) is False
    assert is_token_id_allowed(201, srl_state, None, tokenizer) is False


def test_sticky_sfa_early_stopping():
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.factual_alignment import is_token_id_allowed

    # Mock tokenizer
    class MockTokenizer:
        def decode(self, token_ids):
            mapping = {
                10: "codimension",
                11: "diabolical",
                12: "points",
                100: "the",
                101: "is",
                200: "quantum",
                201: "eigenvalues"
            }
            return mapping.get(token_ids[0], "unknown")

    tokenizer = MockTokenizer()
    
    srl_state = SessionSRLState(
        semantic_index=None,
        chunk_graph=None,
        inverted_index=None,
        ordered_slot_ids=[],
        sink_blocks=[]
    )
    srl_state.current_step_factual_tokens = {10, 11, 12}
    srl_state.current_step_factual_sequences = [[10, 11, 12]]

    # Simulation of generation loop with Sticky SFA
    sfa_active = False
    
    # Step 1: High similarity (SFA triggers)
    srl_state.current_step_max_similarity = 0.45
    if srl_state.current_step_max_similarity >= 0.4:
        sfa_active = True
    
    # Check that a factual or helper token is allowed
    assert sfa_active is True
    assert is_token_id_allowed(10, srl_state, None, tokenizer) is True
    assert is_token_id_allowed(100, srl_state, None, tokenizer) is True
    # Check that an ungrounded token is blocked
    assert is_token_id_allowed(200, srl_state, None, tokenizer) is False

    # Step 2: Similarity drops to 0.1 (SFA should REMAIN active because it is sticky)
    srl_state.current_step_max_similarity = 0.1
    if srl_state.current_step_max_similarity >= 0.4:
        sfa_active = True
    
    # Assert sfa_active is still True (sticky)
    assert sfa_active is True
    # Verify that it still blocks ungrounded content
    assert is_token_id_allowed(200, srl_state, None, tokenizer) is False


def test_lm_vsl_masking():
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.factual_alignment import get_helper_token_ids, get_allowed_tokens_vsl, update_vsl_state

    # Mock tokenizer
    class MockTokenizer:
        @property
        def vocab_size(self):
            return 300
        def decode(self, token_ids):
            mapping = {
                10: "codimension",
                11: "diabolical",
                12: "singularities",
                100: "the",
                101: "is",
                200: "quantum",
                201: "eigenvalues"
            }
            return mapping.get(token_ids[0], "unknown")
        def encode(self, text, add_special_tokens=False):
            mapping = {
                "the": [100],
                "is": [101],
                " codimension": [10],
                " diabolical": [11],
                " singularities": [12]
            }
            return mapping.get(text, [299])

    tokenizer = MockTokenizer()
    
    # 1. Test helper token pre-caching
    helper_ids = get_helper_token_ids(tokenizer)
    assert 100 in helper_ids  # "the" is a helper word
    assert 101 in helper_ids  # "is" is a helper word
    assert 10 not in helper_ids  # "codimension" is a content word
    
    # Create session SRL state
    srl_state = SessionSRLState(
        semantic_index=None,
        chunk_graph=None,
        inverted_index=None,
        ordered_slot_ids=[],
        sink_blocks=[]
    )
    srl_state.current_step_factual_tokens = {10, 11, 12}
    srl_state.current_step_factual_sequences = [[10, 11, 12]]
    srl_state.vsl_active_candidates = []
    srl_state.vsl_consecutive_helpers = 0

    # Step 1: Unlocked state. Only factual sequence starts / helpers are allowed.
    allowed = get_allowed_tokens_vsl(srl_state, helper_ids)
    assert 10 in allowed  # codimension (sequence start)
    assert 11 not in allowed  # diabolical (non-start content)
    assert 12 not in allowed  # points (non-start content)
    assert 100 in allowed  # "the" (helper)
    assert 200 not in allowed  # "quantum" (ungrounded)

    # Model generates token 10 ("codimension")
    update_vsl_state(10, srl_state, helper_ids)
    
    # Verify state is locked onto suffix of sequence 0: [[11, 12]]
    assert len(srl_state.vsl_active_candidates) == 1
    assert srl_state.vsl_active_candidates[0] == [11, 12]
    assert srl_state.vsl_consecutive_helpers == 0

    # Step 2: Locked state. Next allowed content word must be 11.
    allowed = get_allowed_tokens_vsl(srl_state, helper_ids)
    assert 11 in allowed  # diabolical
    assert 12 not in allowed  # "points" is blocked because sequence order is codimension -> diabolical
    assert 100 in allowed  # helper still allowed
    assert 200 not in allowed  # ungrounded still blocked

    # Model generates token 11 ("diabolical")
    update_vsl_state(11, srl_state, helper_ids)
    
    # Suffix advanced: [[12]]
    assert srl_state.vsl_active_candidates[0] == [12]

    # Model generates helper token 100 ("the")
    update_vsl_state(100, srl_state, helper_ids)
    
    # Helpers do not change candidate suffix, but increment helper count
    assert srl_state.vsl_active_candidates[0] == [12]
    assert srl_state.vsl_consecutive_helpers == 1

    # Step 3: Locked state. Next allowed content word must be 12.
    allowed = get_allowed_tokens_vsl(srl_state, helper_ids)
    assert 12 in allowed
    assert 11 not in allowed

    # Model generates 12 ("points")
    update_vsl_state(12, srl_state, helper_ids)
    
    # Suffix completed: [[]]
    assert srl_state.vsl_active_candidates[0] == []

    # Check lock release on sentence ending helper words (12 consecutive helpers)
    for _ in range(12):
        update_vsl_state(100, srl_state, helper_ids)
        
    # Lock should be completely empty/released now
    assert srl_state.vsl_active_candidates == []


def _mk_entry(tokens, start, prime=False, dist_tok=None):
    e = FactEntry(
        start_idx=start, end_idx=start + len(tokens),
        K=torch.zeros(1, 1, 1, 1), V=torch.zeros(1, 1, 1, 1),
        descriptor=torch.zeros(64), tokens=list(tokens),
    )
    e.is_prime = prime
    if prime:
        e.entity_id = start
        e.distinguishing_token = dist_tok
    return e


def test_rc4_interleaved_entity_binding():
    """RC4: interleaved comparison properties must bind to the correct entity.

    Source: "EP2 is codimension 2 while EP3 is codimension 3".
    The EP2 property span "codimension 2 while" shares the filler token "while"
    with the EP3 prime span and names neither entity — plain token overlap
    (the old logic) mis-binds it to EP3.  Reading-order ownership binds it to EP2.
    """
    EP2, EP3, codim, two, three, is_, while_ = 1001, 1002, 2002, 301, 302, 50, 60

    class Inv:
        idf = {EP2: 4.0, EP3: 4.0, codim: 2.0, two: 2.5, three: 2.5, is_: 0.2, while_: 0.3}

    store = FactualExactStore(session_id="rc4")
    store.entries = [
        _mk_entry([EP2, is_, codim], 0, prime=True, dist_tok=EP2),   # "EP2 is codimension"
        _mk_entry([codim, two, while_], 2),                          # "codimension 2 while"  (EP2's property)
        _mk_entry([while_, EP3, is_, codim], 4, prime=True, dist_tok=EP3),  # "while EP3 is codimension"
        _mk_entry([codim, three], 7),                                # "codimension 3"  (EP3's property)
    ]

    store._assign_entities(Inv())

    # Property spans bind to the entity that owns them, not the one they share filler with.
    assert store.entries[1].entity_id == 0, "codimension-2 span should bind to EP2 (start 0)"
    assert store.entries[3].entity_id == 4, "codimension-3 span should bind to EP3 (start 4)"


def test_rc4_distinguishing_token_override():
    """A span that explicitly names an entity binds to it regardless of position."""
    EP2, EP3, codim, two = 1001, 1002, 2002, 301

    class Inv:
        idf = {EP2: 4.0, EP3: 4.0, codim: 2.0, two: 2.5}

    store = FactualExactStore(session_id="rc4b")
    store.entries = [
        _mk_entry([EP2], 0, prime=True, dist_tok=EP2),
        _mk_entry([EP3], 10, prime=True, dist_tok=EP3),
        # Property names EP2 but sits closer to the EP3 prime in absolute position.
        _mk_entry([EP2, codim, two], 8),
    ]
    store._assign_entities(Inv())
    assert store.entries[2].entity_id == 0, "span naming EP2 must bind to EP2 even when nearer EP3"


def test_rc2_quote_grounded_connectives():
    """RC2: under SFA, relational connectives are admitted only where the source
    grounds them — as a captured-sequence start or source-adjacent (in a span's
    prefix) — never freely.
    """
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.factual_alignment import get_allowed_tokens_vsl

    # Token ids: content {codim=10, two=11}; connectives is=101 (copula), while=102
    # (contrastive), because=103 (causal). All three are "relational binders".
    helper_ids = {100, 101, 102, 103}            # the, is, while, because
    structural_helper_ids = {100}                # only "the" is purely grammatical
    # relational_ids = helper_ids - structural = {101, 102, 103}

    srl_state = SessionSRLState(
        semantic_index=None, chunk_graph=None, inverted_index=None,
        ordered_slot_ids=[], sink_blocks=[],
    )
    # One enterable property span [10, 11] whose source prefix was "... while codim".
    # -> "while" (102) is source-adjacent and should be admitted; "because" (103)
    #    never appears in the source and must stay blocked.
    srl_state.current_step_factual_sequences = [[10, 11]]
    srl_state.current_step_sequence_entity_ids = [-1]
    srl_state.current_step_sequence_is_prime = [False]
    srl_state.current_step_sequence_prefixes = [[102, 10]]   # "while codim" preceded the span
    srl_state.vsl_active_candidates = []

    allowed = get_allowed_tokens_vsl(
        srl_state, helper_ids,
        structural_helper_ids=structural_helper_ids, sfa_active=True,
    )

    assert 100 in allowed, "purely grammatical helper 'the' must stay free"
    assert 10 in allowed, "enterable span start must be allowed"
    assert 102 in allowed, "'while' is source-adjacent (in prefix) -> grounded, admitted"
    assert 103 not in allowed, "'because' never appears in source -> blocked"
    assert 101 not in allowed, "'is' is neither a sequence start nor source-adjacent -> blocked"


def test_rc2_triple_bridge_connective_allowed():
    """A connective that begins a captured triple sequence (its bridge) is grounded."""
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.factual_alignment import get_allowed_tokens_vsl

    helper_ids = {100, 101, 102}                  # the, is, while
    structural_helper_ids = {100}

    srl_state = SessionSRLState(
        semantic_index=None, chunk_graph=None, inverted_index=None,
        ordered_slot_ids=[], sink_blocks=[],
    )
    # Triple sequence starts with its bridge connective "is" (101) -> grounded.
    srl_state.current_step_factual_sequences = [[101, 10, 11]]   # "is codimension 2"
    srl_state.current_step_sequence_entity_ids = [5]
    srl_state.current_step_sequence_is_prime = [False]
    srl_state.current_step_sequence_prefixes = [[]]              # triples carry no external prefix
    srl_state.vsl_active_candidates = []

    allowed = get_allowed_tokens_vsl(
        srl_state, helper_ids,
        structural_helper_ids=structural_helper_ids, sfa_active=True,
    )
    assert 101 in allowed, "'is' begins a captured triple bridge -> grounded, admitted"
    assert 102 not in allowed, "'while' is ungrounded here -> blocked"


def test_rc3_entity_bias_reranking():
    """RC3: when two property spans have identical descriptors but belong to
    different entities, an entity-biased query ranks the queried entity's span
    higher.  Without a bias, ranking is unchanged (null-safe)."""
    EP2, EP3, codimA, codimB = 1001, 1002, 2001, 2002

    class Inv:
        idf = {EP2: 4.0, EP3: 4.0, codimA: 2.0, codimB: 2.0}

    d = torch.tensor([1.0, 0.0, 0.0, 0.0])          # shared property descriptor
    orth = torch.tensor([0.0, 1.0, 0.0, 0.0])       # primes: below threshold, not seeds

    store = FactualExactStore(session_id="rc3")
    p2 = _mk_entry([EP2], 0, prime=True, dist_tok=EP2); p2.descriptor = orth.clone()
    p3 = _mk_entry([EP3], 10, prime=True, dist_tok=EP3); p3.descriptor = orth.clone()
    propA = _mk_entry([EP2, codimA], 1); propA.descriptor = d.clone()   # names EP2 -> binds EP2
    propB = _mk_entry([EP3, codimB], 11); propB.descriptor = d.clone()  # names EP3 -> binds EP3
    store.entries = [p2, p3, propA, propB]
    store._assign_entities(Inv())

    W_proj = torch.eye(4)
    Q = d.unsqueeze(0)                               # q_desc == d -> sim(propA)=sim(propB)=1

    def sim_of(start, results):
        for e in results:
            if e.start_idx == start:
                return e.current_sim
        return None

    # Unbiased: both property spans score equally.
    res = store.query(Q, W_proj, threshold=0.5, active_slots=None)
    assert abs(sim_of(1, res) - sim_of(11, res)) < 1e-6, "unbiased ranking must be identical"

    # Biased toward EP2: EP2's property outranks EP3's despite identical descriptors.
    res_b = store.query(Q, W_proj, threshold=0.5, active_slots=None, query_entity_bias={0})
    assert sim_of(1, res_b) > sim_of(11, res_b), "EP2-biased query must rank EP2's span higher"


def test_rc5_advance_comparison_entity():
    """RC5: a comparison block advances to the next entity only after the active
    entity's prime AND a property have appeared in recent output."""
    from native_core.srl.factual_alignment import advance_comparison_entity

    comparison = [0, 10]                              # entity A=0, B=10
    prime_toks = {0: {1001}, 10: {1002}}
    prop_toks  = {0: {2001}, 10: {2002}}

    # Only A's prime seen so far -> stay on A (block not yet substantive).
    idx, cov = advance_comparison_entity(comparison, 0, set(), [1001], prime_toks, prop_toks)
    assert idx == 0 and cov == set()

    # A's prime + property seen -> A covered, advance to B.
    idx, cov = advance_comparison_entity(comparison, 0, set(), [1001, 2001], prime_toks, prop_toks)
    assert idx == 1 and cov == {0}

    # On B's block with everything seen -> B covered, but stay (no further blocks).
    idx, cov = advance_comparison_entity(comparison, 1, {0}, [1002, 2002], prime_toks, prop_toks)
    assert idx == 1 and cov == {0, 10}


def test_rc8_entity_token_license():
    """RC8: tokens exclusive to other entities are 'foreign' and get penalised;
    shared, prime, and entity-agnostic tokens stay licensed."""
    from native_core.srl.factual_alignment import compute_entity_token_license

    sequences  = [[1001, 50],   [2001, 2002], [3001, 3002], [50, 99]]
    entity_ids = [0,            0,            10,           -1]
    is_prime   = [True,         False,        False,        False]

    licensed, foreign = compute_entity_token_license(sequences, entity_ids, is_prime, current_entity=0)
    # Entity 0's prime + property + the entity-agnostic span are licensed.
    assert {1001, 50, 2001, 2002, 99} <= licensed
    # Entity 10's exclusive property tokens are foreign.
    assert foreign == {3001, 3002}
    # 50 is shared (also in agnostic span) so it must NOT be penalised.
    assert 50 not in foreign


def test_structured_attention_segmenting():
    import torch
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.query_router import route_query_fixed_k
    from native_core.srl.inverted_index import InvertedTokenIndex
    from native_core.srl.chunk_graph import ChunkGraph

    # Mock components
    class MockSemanticIndex:
        def __init__(self, slot_ids):
            self.slot_ids = torch.tensor(slot_ids, dtype=torch.int32)
            # Create a mock descriptor matrix
            self.desc_matrix = torch.eye(len(slot_ids), 64).half()

        def slot_to_row_vec(self, slots):
            slot_to_row = {int(s): i for i, s in enumerate(self.slot_ids.tolist())}
            rows = []
            for s in slots.tolist():
                rows.append(slot_to_row.get(s, -1))
            return torch.tensor(rows, dtype=torch.long)

        def search(self, q_desc, k):
            return self.slot_ids[:k]

        def slot_to_idx(self, slot_id):
            slot_list = self.slot_ids.tolist()
            if slot_id in slot_list:
                return slot_list.index(slot_id)
            return -1

    class MockBlockPool:
        def __init__(self, slot_ids):
            self.slot_ids = torch.tensor(slot_ids, dtype=torch.int32)
            self.seq_lens = torch.ones(max(slot_ids) + 1, dtype=torch.int32) * 256
            self.scales = torch.ones(max(slot_ids) + 1, dtype=torch.float32)
            self.W_proj = torch.ones(64, 64, dtype=torch.float32)
            self.anchors_K = torch.ones(max(slot_ids) + 1, 2, 64, dtype=torch.float32)
            
    slot_ids = [10, 20, 30, 40]
    
    # Inverted index with concept occurrences
    token_list = [0] * 1024
    token_list[1] = 500  # in slot 10
    token_list[257] = 600  # in slot 20
    token_ids_index = torch.tensor(token_list, dtype=torch.int32)
    
    from native_core.srl.inverted_index import build_inverted_index
    inv_index = build_inverted_index(
        token_ids=token_ids_index,
        slot_ids=slot_ids,
        block_size=256,
        stop_token_ids=set()
    )

    # ChunkGraph (with neighbors)
    neighbors = torch.full((4, 8), -1, dtype=torch.int32)
    neighbors[0, 0] = 2  # slot 10 -> row 2 (slot 30)
    neighbors[1, 0] = 3  # slot 20 -> row 3 (slot 40)
    chunk_graph = ChunkGraph(neighbors=neighbors)

    semantic_index = MockSemanticIndex(slot_ids)
    
    srl_state = SessionSRLState(
        semantic_index=semantic_index,
        chunk_graph=chunk_graph,
        inverted_index=inv_index,
        ordered_slot_ids=slot_ids,
        sink_blocks=[10]
    )

    # Mock prompt Eagle scores and setup segmenting
    srl_state.prompt_eagle_scores = torch.ones(10, dtype=torch.float32)
    token_ids = torch.tensor([100, 500, 200, 600, 300], dtype=torch.int32)
    srl_state.setup_sas_and_eqa(token_ids, {100, 200, 300})

    # Concept tokens should be concept_tok_1=500, concept_tok_2=600
    assert srl_state.concept_tok_1 == 500
    assert srl_state.concept_tok_2 == 600

    # Segments should be:
    # Slot 10, 30 -> Segment 1
    # Slot 20, 40 -> Segment 2
    assert srl_state.segment_ids[10] == 1
    assert srl_state.segment_ids[30] == 1
    assert srl_state.segment_ids[20] == 2
    assert srl_state.segment_ids[40] == 2

    # Step 1: Active query segment is 1
    srl_state.current_query_segment_id = 1

    # Run query router and verify that segment 2 blocks are completely filtered out!
    pool = MockBlockPool(slot_ids)
    Q = torch.ones(2, 64, dtype=torch.float32)
    
    srl_state.k_min = 2
    srl_state.k_max = 2
    
    selected = route_query_fixed_k(
        Q=Q,
        srl_state=srl_state,
        pool=pool,
        scale=1.0,
        layer_idx=0
    )
    
    selected_list = selected.tolist()
    assert 20 not in selected_list
    assert 40 not in selected_list


def test_dynamic_routing_anchor():
    import torch
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.query_router import route_query_fixed_k
    from native_core.srl.inverted_index import InvertedTokenIndex
    from native_core.srl.chunk_graph import ChunkGraph

    class MockSemanticIndex:
        def __init__(self, slot_ids):
            self.slot_ids = torch.tensor(slot_ids, dtype=torch.int32)
            self.desc_matrix = torch.eye(len(slot_ids), 64).half()
        def slot_to_row_vec(self, slots):
            slot_to_row = {int(s): i for i, s in enumerate(self.slot_ids.tolist())}
            rows = [slot_to_row.get(int(s), -1) for s in slots]
            return torch.tensor(rows, dtype=torch.long)

        def search(self, q_desc, k):
            return self.slot_ids[:k]

        def slot_to_idx(self, slot_id):
            slot_list = self.slot_ids.tolist()
            if slot_id in slot_list:
                return slot_list.index(slot_id)
            return -1

    class MockBlockPool:
        def __init__(self, slot_ids):
            self.slot_ids = torch.tensor(slot_ids, dtype=torch.int32)
            self.seq_lens = torch.ones(max(slot_ids) + 1, dtype=torch.int32) * 256
            self.scales = torch.ones(max(slot_ids) + 1, dtype=torch.float32)
            self.W_proj = torch.ones(64, 64, dtype=torch.float32)
            self.anchors_K = torch.ones(max(slot_ids) + 1, 2, 64, dtype=torch.float32)

    slot_ids = [10, 20, 30]
    semantic_index = MockSemanticIndex(slot_ids)
    
    neighbors = torch.full((3, 8), -1, dtype=torch.int32)
    neighbors[1, 0] = 2  # slot 20 (row 1) has neighbor 30 (row 2)
    chunk_graph = ChunkGraph(neighbors=neighbors)

    srl_state = SessionSRLState(
        semantic_index=semantic_index,
        chunk_graph=chunk_graph,
        inverted_index=InvertedTokenIndex(index={}, important_vocab=set()),
        ordered_slot_ids=slot_ids,
        sink_blocks=[10]
    )

    k0 = torch.zeros(64)
    k0[0] = 1.0
    k1 = torch.zeros(64)
    k1[1] = 1.0
    k2 = torch.zeros(64)
    k2[0] = 1.0
    k3 = torch.zeros(64)
    k3[2] = 1.0
    
    srl_state.recent_decode_keys = [k0, k1, k2, k3]
    srl_state.recent_generated_tokens = [50, 51, 52, 53]
    srl_state.generated_token_slots = [20, 20, 20, 20]

    srl_state.update_dynamic_anchors(set())

    assert len(srl_state.dynamic_anchors) > 0
    assert 20 in srl_state.dynamic_anchors

    pool = MockBlockPool(slot_ids)
    Q = torch.ones(2, 64, dtype=torch.float32)
    srl_state.k_min = 3
    srl_state.k_max = 3

    selected = route_query_fixed_k(
        Q=Q,
        srl_state=srl_state,
        pool=pool,
        scale=1.0,
        layer_idx=0
    )
    selected_list = selected.tolist()
    assert 30 in selected_list


def test_update_vsl_state_with_helpers():
    """Verify that helper tokens in sequences can start or advance locks correctly,
    and unmatched helpers pass through, while unmatched non-helpers break the lock.
    """
    from native_core.srl.session_srl_state import SessionSRLState
    from native_core.srl.factual_alignment import update_vsl_state

    # 10 is content, 101 is a helper ("is")
    helper_ids = {101}
    srl_state = SessionSRLState(
        semantic_index=None, chunk_graph=None, inverted_index=None,
        ordered_slot_ids=[], sink_blocks=[],
    )
    srl_state.current_step_factual_sequences = [[10, 101, 12]]
    srl_state.current_step_sequence_entity_ids = [-1]
    srl_state.vsl_active_candidates = []
    srl_state.vsl_consecutive_helpers = 0

    # Start lock with content token 10
    update_vsl_state(10, srl_state, helper_ids)
    assert srl_state.vsl_active_candidates == [[101, 12]]

    # Advance lock with helper token 101 (which is the expected next token)
    update_vsl_state(101, srl_state, helper_ids)
    assert srl_state.vsl_active_candidates == [[12]]

    # Unmatched helper (e.g. 101 again, when 12 is expected) passes through
    update_vsl_state(101, srl_state, helper_ids)
    assert srl_state.vsl_active_candidates == [[12]]
    assert srl_state.vsl_consecutive_helpers == 1

    # Unmatched non-helper breaks the lock
    update_vsl_state(99, srl_state, helper_ids)
    assert srl_state.vsl_active_candidates == []


def test_factual_store_build_relational_boost():
    """Verify that low-IDF tokens are only boosted if their decoded text matches RELATIONAL_KEYWORDS."""
    from native_core.srl.inverted_index import InvertedTokenIndex
    from native_core.srl.factual_store import FactualExactStore

    class MockTokenizer:
        def decode(self, token_ids):
            mapping = {
                100: "the",
                101: "is",
                200: "quantum"
            }
            return mapping.get(token_ids[0], "unknown")

    tokenizer = MockTokenizer()
    inv_index = InvertedTokenIndex(
        index={},
        important_vocab=set(),
        idf={100: 0.5, 101: 0.4, 200: 4.5}
    )
    inv_index._tokenizer_ref = tokenizer

    store = FactualExactStore(session_id="test_boost")
    token_ids = torch.tensor([100, 101, 200], dtype=torch.long)
    prefill_kv = {0: [torch.zeros(1, 1, 3, 64), torch.zeros(1, 1, 3, 64)]}
    W_proj = torch.zeros(64, 64)

    store.build(
        prefill_kv=prefill_kv,
        token_ids=token_ids,
        W_proj=W_proj,
        stop_token_ids={100},
        inv_index=inv_index,
        use_salience_parser=True
    )

    # Check that inv_index._relational_token_ids contains 101 ("is") but NOT 100 ("the")
    assert 101 in inv_index._relational_token_ids
    assert 100 not in inv_index._relational_token_ids







