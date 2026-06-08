import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from transformers import AutoTokenizer
from native_core.sparse_decode.triton_fused_decode import _pytorch_vectorized_sparse_attn_decode, fused_decode_mps

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Long prompt to ensure we have both compressed and dense blocks
    prompt = "This is a test prompt to verify the correct application of post-RoPE anchors and log-sum-exp combinations. " * 80
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 16},
        device=device,
    )
    
    session_id = "default"
    wrapper.manager.clear_session(session_id)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = encoded.input_ids[0].tolist()
    prefill_len = len(prompt_ids)
    
    wrapper.manager.init_session(session_id, prefill_len=prefill_len)
    wrapper.model._diffkv_session_ids = [session_id]
    
    with torch.no_grad():
        outputs = wrapper.model(input_ids=encoded.input_ids, use_cache=True)
        wrapper.manager.compress_deferred_prefill_blocks(session_id)
        wrapper.manager.finalize_compressed_blocks()

    # Get the state for Layer 5
    layer_idx = 5
    blocks = wrapper.manager.get_streaming_blocks(session_id, layer_idx)
    comp_blocks = [b for b in blocks if getattr(b, "state", None) == "COMPRESSED"]
    dense_blocks = [b for b in blocks if getattr(b, "state", None) != "COMPRESSED"]
    print(f"Number of compressed blocks: {len(comp_blocks)}")
    print(f"Number of dense blocks: {len(dense_blocks)}")
    
    pool = wrapper.manager.native_pool
    pool_indices = [b.pool_idx for b in comp_blocks]
    pool_indices_t = torch.tensor(pool_indices, device=device, dtype=torch.long)
    blk_sizes = pool.seq_lens[pool_indices_t]
    anchor_indices = torch.tensor([b.anchor_idx for b in comp_blocks], device=device, dtype=torch.long)
    
    # Create a dummy query for Layer 5
    num_heads = wrapper.model.config.num_attention_heads
    num_kv_heads = wrapper.model.config.num_key_value_heads
    gpk = num_heads // num_kv_heads
    head_dim = wrapper.model.config.hidden_size // num_heads
    
    torch.manual_seed(42)
    Q = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=torch.float16)
    
    # Get cos/sin from the rotary embedding
    hist_pos = torch.arange(prefill_len + 1, device=device, dtype=torch.long).unsqueeze(0)
    cos_all, sin_all = wrapper.model.model.rotary_emb(torch.randn(1, num_kv_heads, 1, head_dim, device=device, dtype=torch.float16), hist_pos)
    
    # 1. Assemble dense window
    dense_k, dense_v = wrapper.manager.assemble_dense_window_kv(session_id, layer_idx, dense_blocks, torch.float16)
    
    # 2. Run reference combined decoder (which handles both dense and compressed via unified softmax)
    out_unified = _pytorch_vectorized_sparse_attn_decode(
        q=Q,
        block_indices=pool_indices_t,
        pool=pool,
        dense_blocks=dense_blocks,
        active_k=dense_k,
        active_v=dense_v,
        num_key_value_groups=gpk,
        R=32,
        S_MAX=16,
        anchor_indices=anchor_indices,
        cos=cos_all,
        sin=sin_all,
        total_seq_len=prefill_len,
    )
    out_unified = out_unified.squeeze(0).squeeze(1) # [H_q, D]
    
    # 3. Run LSE-combined fallback path manually
    # 3.1 Dense path
    # RoPE on dense
    L_dense = dense_k.shape[2]
    dense_positions_list = []
    for blk in dense_blocks:
        dense_positions_list.extend(blk.token_indices)
    dense_positions = torch.tensor(dense_positions_list, dtype=torch.long, device=device)
    cos_flat = cos_all.squeeze(0) if cos_all.dim() == 3 else cos_all
    sin_flat = sin_all.squeeze(0) if sin_all.dim() == 3 else sin_all
    cos_sliced = cos_flat[dense_positions].unsqueeze(0)
    sin_sliced = sin_flat[dense_positions].unsqueeze(0)
    cos_dense = cos_sliced.unsqueeze(1)
    sin_dense = sin_sliced.unsqueeze(1)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    dense_k_rot = (dense_k * cos_dense) + rotate_half(dense_k) * sin_dense
    
    # repeat_kv
    def repeat_kv(hidden_states, n_rep):
        bs, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(bs, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(bs, num_key_value_heads * n_rep, slen, head_dim)
        
    k_rep = repeat_kv(dense_k_rot, gpk)
    v_rep = repeat_kv(dense_v, gpk)
    
    # Dense attention output
    import torch.nn.functional as F
    out_dense = F.scaled_dot_product_attention(Q, k_rep, v_rep, is_causal=False)
    # Wait, F.scaled_dot_product_attention doesn't return LSE directly unless we compute it:
    
    # Dense LSE
    _q = Q[0, :, 0, :]
    _kd = k_rep[0]
    _scale = (head_dim ** -0.5)
    scores_dense = torch.matmul(_kd, _q.unsqueeze(-1)).squeeze(-1) * _scale
    print("scores_dense[:, 0]:", scores_dense[:, 0].tolist())
    lse_dense = torch.logsumexp(scores_dense.float(), dim=-1)
    
    # 3.2 Sparse path
    out_sparse, lse_sparse = fused_decode_mps(
        Q=_q,
        pool=pool,
        block_indices=pool_indices_t,
        blk_sizes=blk_sizes,
        num_key_value_groups=gpk,
        anchor_indices=anchor_indices,
        cos=cos_all,
        sin=sin_all,
    )
    
    # 3.3 Combine
    out_dense_hd = out_dense[0, :, 0, :].float()
    out_sparse_fp32 = out_sparse.float()
    lse_dense = lse_dense.to(torch.float32)
    lse_sparse = lse_sparse.to(torch.float32)
    
    lse_max = torch.maximum(lse_dense, lse_sparse)
    w_dense = torch.exp(lse_dense - lse_max)
    w_sparse = torch.exp(lse_sparse - lse_max)
    denom = w_dense + w_sparse
    
    out_combined = (out_dense_hd * w_dense.unsqueeze(-1) + out_sparse_fp32 * w_sparse.unsqueeze(-1)) / denom.unsqueeze(-1)
    out_combined = out_combined.to(torch.float16)
    
    # Define variables for unified LSE calculation
    idx = pool_indices_t.long()
    N = idx.shape[0]
    AncK_a = pool.anchors_K[idx].float()
    VK_a = pool.V_K[idx].float()
    U_a = pool.U[idx].float() * pool.U_scale[idx].view(N, 1, 1).float()
    
    cos_flat = cos_all.squeeze(0) if cos_all.dim() == 3 else cos_all
    sin_flat = sin_all.squeeze(0) if sin_all.dim() == 3 else sin_all
    cos_anc = cos_flat[anchor_indices].unsqueeze(1).unsqueeze(2)
    sin_anc = sin_flat[anchor_indices].unsqueeze(1).unsqueeze(2)
    cos_anc_2d = cos_flat[anchor_indices].unsqueeze(1)
    sin_anc_2d = sin_flat[anchor_indices].unsqueeze(1)
    
    VK_a_rot = VK_a * cos_anc + rotate_half(VK_a) * sin_anc
    AncK_a_rot = AncK_a * cos_anc_2d + rotate_half(AncK_a) * sin_anc_2d
    
    AncK_e = AncK_a_rot.repeat_interleave(gpk, dim=1)
    VK_e = VK_a_rot.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
    
    block_capacity = U_a.shape[1]
    scores_anchor_unified = torch.einsum('hd,nhd->hn', Q[0, :, 0, :].float(), AncK_e) * _scale
    q_proj_u = torch.einsum('hd,nhrd->nhr', Q[0, :, 0, :].float(), VK_e) * _scale
    scores_block_u = torch.einsum('nhr,nsr->hns', q_proj_u, U_a) * pool.scales[idx].float().view(1, N, 1)
    scores_block_u = scores_block_u + scores_anchor_unified.unsqueeze(-1)
    scores_comp_u = scores_block_u.reshape(num_heads, N * block_capacity)
    s_range = torch.arange(block_capacity, device=device).view(1, 1, -1)
    valid_mask = s_range < blk_sizes.view(1, N, 1)
    scores_comp_u = scores_comp_u.reshape(num_heads, N, block_capacity)
    scores_comp_u = scores_comp_u.masked_fill(~valid_mask, float('-inf'))
    scores_comp_u = scores_comp_u.reshape(num_heads, N * block_capacity)
    # Extract sparse and dense parts of probabilities from unified
    scores_all_unified = torch.cat([scores_anchor_unified, scores_comp_u, scores_dense], dim=-1)
    probs_all_unified = torch.nn.functional.softmax(scores_all_unified, dim=-1)
    
    # split
    P_anchor, P_comp, P_dense = torch.split(probs_all_unified, [N, N * block_capacity, L_dense], dim=-1)
    
    # Compute dense output from unified probs
    w_dense_unified = P_dense.sum(dim=-1) # [H_q]
    # normalize dense part to get unified out_dense
    out_dense_unified = torch.sum(P_dense.unsqueeze(-1) * v_rep[0].float(), dim=1) / w_dense_unified.unsqueeze(-1)
    
    # Compute sparse output from unified probs
    w_sparse_unified = P_anchor.sum(dim=-1) + P_comp.sum(dim=-1) # [H_q]
    
    P_comp_reshaped = P_comp.view(num_heads, N, block_capacity).permute(1, 0, 2)
    P_U = torch.bmm(P_comp_reshaped.float(), U_a.float())
    p_total_anchor = P_anchor.transpose(0, 1) + P_comp_reshaped.sum(dim=-1)
    AncV_e = pool.anchors_V[idx].float().repeat_interleave(gpk, dim=1)
    O_anchor_fused = torch.sum(p_total_anchor.unsqueeze(-1) * AncV_e, dim=0)
    
    P_U_flat = P_U.reshape(N * num_heads, 1, 32)
    VV_e = pool.V_V[idx].float().repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous().reshape(N * num_heads, 32, head_dim)
    O_delta = torch.bmm(P_U_flat, VV_e).reshape(N, num_heads, head_dim) * pool.scales[idx].float().view(N, 1, 1)
    out_sparse_unified = (O_anchor_fused + O_delta.sum(0)) / w_sparse_unified.unsqueeze(-1)
    
    # Compute LSE of dense scores in unified path
    lse_dense_unified = torch.logsumexp(scores_dense.float(), dim=-1)
    # Compute LSE of sparse scores in unified path
    lse_sparse_unified = torch.logsumexp(torch.cat([scores_anchor_unified, scores_comp_u], dim=-1).float(), dim=-1)
    
    print(f"lse_dense manual: {lse_dense.tolist()[:3]}")
    print(f"lse_dense unified: {lse_dense_unified.tolist()[:3]}")
    print(f"lse_sparse manual: {lse_sparse.tolist()[:3]}")
    print(f"lse_sparse unified: {lse_sparse_unified.tolist()[:3]}")
    
    # Print detailed index diagnostic
    start_pos = dense_blocks[0].anchor_idx
    dense_positions_list = []
    for blk in dense_blocks:
        dense_positions_list.extend(blk.token_indices)
    dense_positions = torch.tensor(dense_positions_list, dtype=torch.long, device=device)
    manual_positions = torch.arange(start_pos, start_pos + L_dense, device=device)
    print("Positions identical:", torch.equal(dense_positions, manual_positions))
    if not torch.equal(dense_positions, manual_positions):
        print("dense_positions[:10]:", dense_positions[:10].tolist())
        print("manual_positions[:10]:", manual_positions[:10].tolist())
        print("dense_positions[500:510]:", dense_positions[500:510].tolist())
        print("manual_positions[500:510]:", manual_positions[500:510].tolist())

    print("scores_dense manual[0, 500:505]:", scores_dense[0, 500:505].tolist())
    print("scores_dense unified[0, 500:505]:", scores_all_unified[0, N + N * block_capacity + 500 : N + N * block_capacity + 505].tolist())

    diff_lse_dense = (lse_dense - lse_dense_unified).abs()
    print(f"Dense LSE Max Diff: {diff_lse_dense.max().item():.6f}")
    
    diff_lse_sparse = (lse_sparse - lse_sparse_unified).abs()
    print(f"Sparse LSE Max Diff: {diff_lse_sparse.max().item():.6f}")
    
    # Compare weights (normalized)
    w_dense_norm = w_dense / denom
    w_sparse_norm = w_sparse / denom
    
    diff_w_dense = (w_dense_norm - w_dense_unified).abs()
    print(f"w_dense Max Diff (normalized): {diff_w_dense.max().item():.6f}")
    
    diff_w_sparse = (w_sparse_norm - w_sparse_unified).abs()
    print(f"w_sparse Max Diff (normalized): {diff_w_sparse.max().item():.6f}")
    
    # Compare dense outputs
    diff_out_dense = (out_dense_hd - out_dense_unified).abs()
    print(f"out_dense Max Diff: {diff_out_dense.max().item():.6f}")
    
    # Compare sparse outputs
    diff_out_sparse = (out_sparse_fp32 - out_sparse_unified).abs()
    print(f"out_sparse Max Diff: {diff_out_sparse.max().item():.6f}")
    
    # Compare output
    diff = (out_combined - out_unified).abs()
    print(f"Combined Output Max Diff: {diff.max().item():.6f}")
    print(f"Combined Output Mean Diff: {diff.mean().item():.6f}")
    
    # Check sum of components for unified output
    print("P_dense[0, 500:505]:", P_dense[0, 500:505].tolist())
    print("v_rep sum:", v_rep.sum().item())
    print("v_rep[0, 0, 0, :5]:", v_rep[0, 0, 0, :5].tolist())
    O_dense_total = torch.sum(P_dense.unsqueeze(-1) * v_rep[0].float(), dim=1)
    print("O_dense_total[:, 0]:", O_dense_total[:, 0].tolist())
    O_sparse_total = O_anchor_fused + O_delta.sum(0)
    print("O_dense_total[0][0]:", O_dense_total[0][0].item())
    print("O_sparse_total[0][0]:", O_sparse_total[0][0].item())
    print("O_dense_total + O_sparse_total:", (O_dense_total[0][0] + O_sparse_total[0][0]).item())
    print("out_unified[0][0]:", out_unified[0][0].item())
    
    wrapper.stop()

if __name__ == "__main__":
    main()
