import torch
import numpy as np
import sys
import os
import math
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from runtime.native_block_pool import NativeBlockPool
from native_core.compression.lowrank import compress_lowrank
from native_core.sparse_decode.triton_fused_decode import _pytorch_vectorized_sparse_attn_decode, fused_decode_mps

def test_sparse_residual_correctness():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Define dimensions
    num_kv_heads = 4
    head_dim = 64
    feat_dim = 2 * num_kv_heads * head_dim
    seq_len = 8
    rank = 4
    
    # We create a random base key/value delta
    torch.manual_seed(42)
    deltas = torch.randn(seq_len, feat_dim, device=device, dtype=torch.float32) * 0.1
    # The first token is the anchor token, so its delta must be exactly zero.
    deltas[0] = 0.0
    
    # Inject 4 orthogonal fact tokens.
    fact_indices = [1, 3, 5, 7]
    for idx in fact_indices:
        pattern = torch.zeros(feat_dim, device=device)
        start_dim = (idx * 32) % feat_dim
        pattern[start_dim : start_dim + 64] = 5.0
        deltas[idx] = pattern
    
    # 2. Compress using lowrank SVD with threshold=0.0 and frac=1.0 to capture ALL tokens as residuals
    lr_delta = compress_lowrank(deltas, rank=rank, error_threshold=0.0, max_residual_frac=1.0)
    
    k_positions = lr_delta.residual_K_positions.cpu().tolist()
    v_positions = lr_delta.residual_V_positions.cpu().tolist()
    print(f"Residual K positions: {k_positions}")
    print(f"Residual V positions: {v_positions}")
    
    # 3. Setup NativeBlockPool and write the block
    pool = NativeBlockPool(
        max_blocks=10,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rank=rank,
        max_seq_len=8,
        device=device,
        dtype=torch.float16,
        initial_blocks=2,
        num_layers=1,
        lazy=False,
    )
    
    pool_idx = pool.allocate_block()
    
    anchor_K = deltas[0, :num_kv_heads * head_dim].view(num_kv_heads, head_dim).to(torch.float16)
    anchor_V = deltas[0, num_kv_heads * head_dim:].view(num_kv_heads, head_dim).to(torch.float16)
    
    # Write the block with residuals
    pool.write_block(
        pool_idx=pool_idx,
        U=lr_delta.U,
        V=lr_delta.V,
        anchor_K=anchor_K,
        anchor_V=anchor_V,
        scale=lr_delta.scale,
        seq_len=seq_len,
        residual_K_positions=lr_delta.residual_K_positions,
        residual_K_values=lr_delta.residual_K_values,
        residual_V_positions=lr_delta.residual_V_positions,
        residual_V_values=lr_delta.residual_V_values
    )
    
    # 4. Setup Query and perform decode attention
    num_heads = num_kv_heads * 2 # 2 query heads per KV head (GQA factor = 2)
    Q = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=torch.float16)
    
    # Let's run _pytorch_vectorized_sparse_attn_decode
    block_indices = torch.tensor([pool_idx], device=device, dtype=torch.long)
    blk_sizes = torch.tensor([seq_len], device=device, dtype=torch.int32)
    anchor_indices = torch.tensor([0], device=device, dtype=torch.long)
    
    out_with_res = _pytorch_vectorized_sparse_attn_decode(
        q=Q,
        block_indices=block_indices,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=2,
        R=rank,
        S_MAX=16,
        anchor_indices=anchor_indices,
        cos=None,
        sin=None,
        total_seq_len=seq_len,
    )
    
    # Let's temporarily disable residual in pool and run again to see the benefit
    orig_k_pos = pool.residual_K_positions.clone()
    orig_v_pos = pool.residual_V_positions.clone()
    
    pool.residual_K_positions[pool_idx] = -1
    pool.residual_V_positions[pool_idx] = -1
    
    out_no_res = _pytorch_vectorized_sparse_attn_decode(
        q=Q,
        block_indices=block_indices,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=2,
        R=rank,
        S_MAX=16,
        anchor_indices=anchor_indices,
        cos=None,
        sin=None,
        total_seq_len=seq_len,
    )
    
    # Restore pool positions
    pool.residual_K_positions.copy_(orig_k_pos)
    pool.residual_V_positions.copy_(orig_v_pos)
    
    # Check intermediate values
    print(f"out_with_res sum: {out_with_res.sum().item():.6f}")
    print(f"out_no_res sum: {out_no_res.sum().item():.6f}")
    print(f"Difference: {(out_with_res - out_no_res).abs().max().item():.6f}")
    
    # Compute the true dense attention output for comparison
    K_dense = torch.zeros(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    V_dense = torch.zeros(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    
    K_dense[0] = anchor_K.float()
    V_dense[0] = anchor_V.float()
    
    K_dense[1:] = anchor_K.float().unsqueeze(0) + deltas[1:, :num_kv_heads*head_dim].view(seq_len-1, num_kv_heads, head_dim)
    V_dense[1:] = anchor_V.float().unsqueeze(0) + deltas[1:, num_kv_heads*head_dim:].view(seq_len-1, num_kv_heads, head_dim)
    
    # Repeat KV for GQA
    K_dense_rep = K_dense.repeat_interleave(2, dim=1).permute(1, 0, 2) # [num_heads, seq_len, head_dim]
    V_dense_rep = V_dense.repeat_interleave(2, dim=1).permute(1, 0, 2) # [num_heads, seq_len, head_dim]
    
    Q_sq = Q.view(num_heads, head_dim).float()
    scores_dense = torch.bmm(Q_sq.unsqueeze(1), K_dense_rep.transpose(1, 2)).squeeze(1) / math.sqrt(head_dim) # [num_heads, seq_len]
    probs_dense = torch.softmax(scores_dense, dim=-1) # [num_heads, seq_len]
    
    # Now let's do the SAME duplicate anchor logic for a perfect mathematical comparison:
    # duplicate the anchor token in dense scores
    scores_anchor_dense = scores_dense[:, 0:1] # [num_heads, 1]
    scores_all_dense = torch.cat([scores_anchor_dense, scores_dense], dim=-1) # [num_heads, 1 + seq_len]
    probs_all_dense = torch.softmax(scores_all_dense, dim=-1) # [num_heads, 1 + seq_len]
    
    P_anchor_dense = probs_all_dense[:, 0]
    P_comp_dense = probs_all_dense[:, 1:]
    out_dense_dup = (P_anchor_dense.unsqueeze(-1) + P_comp_dense.sum(dim=1).unsqueeze(-1)) * anchor_V.float().repeat_interleave(2, dim=0)
    # Add delta contributions
    delta_V_dense = V_dense[1:] - anchor_V.float().unsqueeze(0) # [seq_len-1, num_kv_heads, head_dim]
    delta_V_dense_rep = delta_V_dense.repeat_interleave(2, dim=1).permute(1, 0, 2) # [num_heads, seq_len-1, head_dim]
    out_dense_dup = out_dense_dup + torch.bmm(P_comp_dense[:, 1:].unsqueeze(1), delta_V_dense_rep).squeeze(1)
    
    diff_with_res = (out_with_res.squeeze(0).squeeze(1).float() - out_dense_dup).abs().mean().item()
    diff_no_res = (out_no_res.squeeze(0).squeeze(1).float() - out_dense_dup).abs().mean().item()
    
    print(f"Mean error with sparse residual vs duplicate dense: {diff_with_res:.6f}")
    print(f"Mean error without sparse residual vs duplicate dense: {diff_no_res:.6f}")
    
    print(f"out_with_res[0, 0, 0, :10]: {out_with_res[0, 0, 0, :10].float().tolist()}")
    print(f"out_no_res[0, 0, 0, :10]: {out_no_res[0, 0, 0, :10].float().tolist()}")
    print(f"out_dense_dup[0, :10]: {out_dense_dup[0, :10].tolist()}")
    assert diff_with_res < 0.01, f"Attention output with residual correction error {diff_with_res} is too high"
    assert diff_with_res < diff_no_res, "Residual correction should improve accuracy compared to compression-only"


def test_triton_matches_reference_on_gpu():
    """AUTHORITATIVE CUDA cert for the F1 residual-alignment fix (CUDA_TRITON_AUDIT.md).

    Runs the REAL Triton kernel (native_triton_sparse_attn_decode) and asserts it
    matches BOTH the PyTorch reference (_pytorch_vectorized_sparse_attn_decode) and
    exact dense attention on a real compressed block with residuals. Skips on non-CUDA
    (Triton is CUDA-only) — this is the check to run on the GPU box.
    """
    import pytest
    from native_core.sparse_decode.triton_fused_decode import (
        native_triton_sparse_attn_decode, HAS_TRITON,
    )
    if not (torch.cuda.is_available() and HAS_TRITON):
        pytest.skip("Triton kernel requires CUDA; run this on the GPU box.")

    device = "cuda"
    num_kv_heads, head_dim, seq_len, rank = 4, 64, 8, 4
    feat_dim = 2 * num_kv_heads * head_dim
    num_kv_groups = 2
    num_heads = num_kv_heads * num_kv_groups

    torch.manual_seed(42)
    deltas = torch.randn(seq_len, feat_dim, device=device, dtype=torch.float32) * 0.1
    deltas[0] = 0.0
    for idx in [1, 3, 5, 7]:                       # inject distinctive tokens
        pattern = torch.zeros(feat_dim, device=device)
        start = (idx * 32) % feat_dim
        pattern[start:start + 64] = 5.0
        deltas[idx] = pattern

    lr = compress_lowrank(deltas, rank=rank, error_threshold=0.0, max_residual_frac=1.0)

    pool = NativeBlockPool(max_blocks=10, num_kv_heads=num_kv_heads, head_dim=head_dim,
                           rank=rank, max_seq_len=8, device=device, dtype=torch.float16,
                           initial_blocks=2, num_layers=1, lazy=False)
    pool_idx = pool.allocate_block()
    aK = deltas[0, :num_kv_heads * head_dim].view(num_kv_heads, head_dim).to(torch.float16)
    aV = deltas[0, num_kv_heads * head_dim:].view(num_kv_heads, head_dim).to(torch.float16)
    pool.write_block(pool_idx=pool_idx, U=lr.U, V=lr.V, anchor_K=aK, anchor_V=aV,
                     scale=lr.scale, seq_len=seq_len,
                     residual_K_positions=lr.residual_K_positions,
                     residual_K_values=lr.residual_K_values,
                     residual_V_positions=lr.residual_V_positions,
                     residual_V_values=lr.residual_V_values)

    Q = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=torch.float16)
    block_indices = torch.tensor([pool_idx], device=device, dtype=torch.long)
    anchor_indices = torch.tensor([0], device=device, dtype=torch.long)
    kw = dict(q=Q, block_indices=block_indices, pool=pool, dense_blocks=[], active_k=None,
              active_v=None, num_key_value_groups=num_kv_groups, R=rank, S_MAX=16,
              anchor_indices=anchor_indices, cos=None, sin=None, total_seq_len=seq_len)

    out_triton = native_triton_sparse_attn_decode(**kw)          # real @triton.jit kernel
    out_ref = _pytorch_vectorized_sparse_attn_decode(**kw)       # reference of record

    # Exact dense attention (duplicate-anchor formulation, matching the decoders)
    Kd = torch.zeros(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    Vd = torch.zeros(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)
    Kd[0], Vd[0] = aK.float(), aV.float()
    Kd[1:] = aK.float().unsqueeze(0) + deltas[1:, :num_kv_heads*head_dim].view(seq_len-1, num_kv_heads, head_dim)
    Vd[1:] = aV.float().unsqueeze(0) + deltas[1:, num_kv_heads*head_dim:].view(seq_len-1, num_kv_heads, head_dim)
    Kr = Kd.repeat_interleave(num_kv_groups, 1).permute(1, 0, 2)
    Vr = Vd.repeat_interleave(num_kv_groups, 1).permute(1, 0, 2)
    Qs = Q.view(num_heads, head_dim).float()
    sd = torch.bmm(Qs.unsqueeze(1), Kr.transpose(1, 2)).squeeze(1) / math.sqrt(head_dim)
    sd = torch.cat([sd[:, 0:1], sd], dim=-1)                     # duplicate anchor
    pd = torch.softmax(sd, dim=-1)
    O_dense = (pd[:, 0:1] + pd[:, 1:].sum(-1, keepdim=True)) * aV.float().repeat_interleave(num_kv_groups, 0)
    dVr = (Vd[1:] - aV.float().unsqueeze(0)).repeat_interleave(num_kv_groups, 1).permute(1, 0, 2)
    O_dense = O_dense + torch.bmm(pd[:, 2:].unsqueeze(1), dVr).squeeze(1)

    t = out_triton.squeeze(0).squeeze(1).float()
    r = out_ref.squeeze(0).squeeze(1).float()
    d_tr = (t - r).abs().max().item()
    d_td = (t - O_dense).abs().mean().item()
    print(f"Triton vs reference max-diff = {d_tr:.6f}")
    print(f"Triton vs dense    mean-diff = {d_td:.6f}")
    assert d_tr < 1e-2, f"Triton kernel diverges from reference: {d_tr}"
    assert d_td < 1e-2, f"Triton kernel diverges from exact dense: {d_td}"
    print("Success! Triton fused decode aligned to Mac reference + dense truth.")


def test_metal_residual_and_fact_parity():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        print("Metal not available, skipping Metal parity test.")
        return
        
    from runtime.dkv_attention import _decode_attention_metal, _DKV_HAS_METAL_ATTN
    if not _DKV_HAS_METAL_ATTN:
        print("Metal extension bindings not compiled, skipping.")
        return

    print("\n─── Running test_metal_residual_and_fact_parity ───")
    # 1. Define dimensions
    num_kv_heads = 4
    num_heads = 8  # GQA factor = 2
    head_dim = 64
    feat_dim = 2 * num_kv_heads * head_dim
    seq_len = 8
    rank = 4
    
    torch.manual_seed(42)
    deltas = torch.randn(seq_len, feat_dim, device=device, dtype=torch.float16) * 0.1
    deltas[0] = 0.0
    
    # Inject residual fact overrides
    fact_indices = [1, 3, 5, 7]
    for idx in fact_indices:
        pattern = torch.zeros(feat_dim, device=device, dtype=torch.float16)
        start_dim = (idx * 32) % feat_dim
        pattern[start_dim : start_dim + 64] = 5.0
        deltas[idx] = pattern
        
    lr_delta = compress_lowrank(deltas.float(), rank=rank, error_threshold=0.0, max_residual_frac=1.0)
    
    # Setup NativeBlockPool and write the block
    pool = NativeBlockPool(
        max_blocks=10,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rank=rank,
        max_seq_len=8,
        device=device,
        dtype=torch.float16,
        initial_blocks=2,
        num_layers=1,
        lazy=False,
    )
    
    pool_idx = pool.allocate_block()
    anchor_K = deltas[0, :num_kv_heads * head_dim].view(num_kv_heads, head_dim)
    anchor_V = deltas[0, num_kv_heads * head_dim:].view(num_kv_heads, head_dim)
    
    pool.write_block(
        pool_idx=pool_idx,
        U=lr_delta.U,
        V=lr_delta.V,
        anchor_K=anchor_K,
        anchor_V=anchor_V,
        scale=lr_delta.scale,
        seq_len=seq_len,
        residual_K_positions=lr_delta.residual_K_positions,
        residual_K_values=lr_delta.residual_K_values,
        residual_V_positions=lr_delta.residual_V_positions,
        residual_V_values=lr_delta.residual_V_values
    )
    
    Q = torch.randn(num_heads, head_dim, device=device, dtype=torch.float16)
    # int32, not long: this test drives dkv_core.decode_attention_metal DIRECTLY,
    # and the shader reads slot_indices through a typed pointer. The binding grew
    # an explicit dtype guard with the fp16-RoPE-table fix (39a4a9d1) and this
    # test was not updated with it, so it has failed on every Mac since -- and
    # CUDA never runs it, which is why it stayed unnoticed. The sibling
    # test_decode_attention_metal_isolated.py already passes int32 here.
    block_indices = torch.tensor([pool_idx], device=device, dtype=torch.int32)
    blk_sizes = torch.tensor([seq_len], device=device, dtype=torch.int32)
    anchor_indices = torch.tensor([0], device=device, dtype=torch.long)
    
    _ca = torch.empty(0, device=device, dtype=torch.float32)
    _sa = torch.empty(0, device=device, dtype=torch.float32)
    _scale = 1.0 / math.sqrt(head_dim)
    
    _res_pos_K = pool.residual_K_positions if pool.residual_K_positions is not None else torch.empty(0, device=device, dtype=torch.int16)
    _res_val_K = pool.residual_K_values if pool.residual_K_values is not None else torch.empty(0, device=device, dtype=torch.float16)
    _res_pos_V = pool.residual_V_positions if pool.residual_V_positions is not None else torch.empty(0, device=device, dtype=torch.int16)
    _res_val_V = pool.residual_V_values if pool.residual_V_values is not None else torch.empty(0, device=device, dtype=torch.float16)
    _fact_pos = pool.fact_anchor_positions if pool.fact_anchor_positions is not None else torch.empty(0, device=device, dtype=torch.int16)
    _fact_val_K = pool.fact_anchors_K if pool.fact_anchors_K is not None else torch.empty(0, device=device, dtype=torch.float16)
    _fact_val_V = pool.fact_anchors_V if pool.fact_anchors_V is not None else torch.empty(0, device=device, dtype=torch.float16)

    # 1. Launch the custom Metal shader path
    out_metal, lse_metal = _decode_attention_metal(
        Q.contiguous(),
        pool.U.contiguous(),
        pool.U_scale.contiguous(),
        pool.V_K.contiguous(),
        pool.V_V.contiguous(),
        pool.anchors_K.contiguous(),
        pool.anchors_V.contiguous(),
        pool.seq_lens.contiguous(),
        pool.scales.contiguous(),
        _ca,
        _sa,
        block_indices.contiguous(),
        _scale,
        num_heads,
        num_kv_heads,
        rank,
        _res_pos_K.contiguous(),
        _res_val_K.contiguous(),
        _res_pos_V.contiguous(),
        _res_val_V.contiguous(),
        _fact_pos.contiguous(),
        _fact_val_K.contiguous(),
        _fact_val_V.contiguous(),
        # Trailing dense-window args (dense_K/V + cos/sin_dense) were added to the
        # binding after this test was written; empties → the impl skips the dense loop.
        torch.empty(0, device=device, dtype=torch.float16),
        torch.empty(0, device=device, dtype=torch.float16),
        torch.empty(0, device=device, dtype=torch.float16),
        torch.empty(0, device=device, dtype=torch.float16),
    )

    # 2. Run Python/MPS reference fallback (fused_decode_mps)
    out_ref, lse_ref = fused_decode_mps(
        Q=Q,
        pool=pool,
        block_indices=block_indices,
        blk_sizes=blk_sizes,
        num_key_value_groups=2,
        anchor_indices=anchor_indices,
        cos=None,
        sin=None,
    )
    
    # 3. Assert outputs match
    diff_out = (out_metal - out_ref).abs().max().item()
    diff_lse = (lse_metal - lse_ref).abs().max().item()
    print(f"Output Max Difference: {diff_out:.6f}")
    print(f"LSE Max Difference: {diff_lse:.6f}")
    
    assert diff_out < 1e-2, f"Metal output mismatch: max diff {diff_out}"
    assert diff_lse < 1e-2, f"Metal LSE mismatch: max diff {diff_lse}"
    print("Success! Metal shader parity validated.")

