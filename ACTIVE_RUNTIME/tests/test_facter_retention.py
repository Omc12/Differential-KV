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
