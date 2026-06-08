import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from transformers import AutoTokenizer

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prompt = "This is a test prompt to verify the correct application of post-RoPE anchors and log-sum-exp combinations. " * 70
    
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
    print(f"Number of compressed blocks: {len(comp_blocks)}")
    
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
    Q = torch.randn(num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Get cos/sin from the rotary embedding
    hist_pos = torch.arange(prefill_len + 1, device=device, dtype=torch.long).unsqueeze(0)
    cos_all, sin_all = wrapper.model.model.rotary_emb(torch.randn(1, num_kv_heads, 1, head_dim, device=device, dtype=torch.float16), hist_pos)
    
    # Setup inputs
    idx = pool_indices_t.long()
    N = idx.shape[0]
    
    # 1. Load data
    U_a = pool.U[idx].float() * pool.U_scale[idx].view(N, 1, 1).float()
    AncK_a = pool.anchors_K[idx].float()
    AncV_a = pool.anchors_V[idx].float()
    VK_a = pool.V_K[idx].float()
    VV_a = pool.V_V[idx].float()
    
    # helper functions
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def repeat_kv_at_dim(t, n_rep, dim):
        if n_rep == 1:
            return t
        shape = list(t.shape)
        val = shape[dim]
        t = t.unsqueeze(dim + 1)
        expand_shape = list(t.shape)
        expand_shape[dim + 1] = n_rep
        t = t.expand(*expand_shape)
        new_shape = shape[:dim] + [val * n_rep] + shape[dim + 1:]
        return t.reshape(*new_shape)

    # cos/sin sliced for exact path
    cos_flat = cos_all.squeeze(0) if cos_all.dim() == 3 else cos_all
    sin_flat = sin_all.squeeze(0) if sin_all.dim() == 3 else sin_all
    
    block_capacity = U_a.shape[1]
    positions = anchor_indices.view(N, 1) + torch.arange(1 + block_capacity, device=device).view(1, 1 + block_capacity)
    positions_flat = positions.view(-1)
    cos_sliced = cos_flat[positions_flat].view(N, 1 + block_capacity, 1, head_dim).float()
    sin_sliced = sin_flat[positions_flat].view(N, 1 + block_capacity, 1, head_dim).float()
    
    # --- Exact Path Steps ---
    print("\n--- Running PyTorch Vectorized Fallback Decoder manually ---")
    U_exact = pool.U[idx].float() * pool.U_scale[idx].view(-1, 1, 1).float()
    V_K_exact = repeat_kv_at_dim(pool.V_K[idx].float(), gpk, dim=2)
    anchors_K_exact = repeat_kv_at_dim(pool.anchors_K[idx].float(), gpk, dim=1)
    scales_exact = pool.scales[idx].view(N, 1, 1).float()
    
    deltas_k_flat = torch.bmm(U_exact, V_K_exact.reshape(N, V_K_exact.shape[1], -1))
    deltas_k_exact = deltas_k_flat.reshape(N, block_capacity, num_heads, head_dim) * scales_exact.unsqueeze(-1)
    zeros_pad = torch.zeros((N, 1, num_heads, head_dim), dtype=torch.float32, device=device)
    deltas_k_full = torch.cat([zeros_pad, deltas_k_exact], dim=1)
    K_unrot_full = anchors_K_exact.unsqueeze(1) + deltas_k_full
    
    # RoPE on full reconstructed keys
    K_unrot_rotated = rotate_half(K_unrot_full)
    K_rot_full = K_unrot_full * cos_sliced + K_unrot_rotated * sin_sliced
    
    # Score
    q_sq = Q.float()
    q_expanded = q_sq.view(1, 1, num_heads, head_dim)
    scores_exact = torch.sum(q_expanded * K_rot_full, dim=-1) * (head_dim ** -0.5)
    
    # --- Fused Path Steps ---
    print("\n--- Running Fused Decoder manually ---")
    # Apply RoPE at query-time
    cos_anc = cos_flat[anchor_indices].to(device=device, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    sin_anc = sin_flat[anchor_indices].to(device=device, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    
    cos_anc_2d = cos_flat[anchor_indices].to(device=device, dtype=torch.float32).unsqueeze(1)
    sin_anc_2d = sin_flat[anchor_indices].to(device=device, dtype=torch.float32).unsqueeze(1)
    
    VK_a_rot = VK_a * cos_anc + rotate_half(VK_a) * sin_anc
    AncK_a_rot = AncK_a * cos_anc_2d + rotate_half(AncK_a) * sin_anc_2d
    
    AncK_e = AncK_a_rot.repeat_interleave(gpk, dim=1)
    VK_e = VK_a_rot.repeat_interleave(gpk, dim=2).permute(0, 2, 1, 3).contiguous()
    
    # Anchor score
    s_anc = torch.einsum('hd,nhd->hn', q_sq, AncK_e) * (head_dim ** -0.5)
    
    # Delta score
    q_proj_n = torch.einsum('hd,nhrd->nhr', q_sq, VK_e) * (head_dim ** -0.5)
    delta_s = torch.einsum('nhr,nsr->hns', q_proj_n, U_a)
    delta_s = delta_s * pool.scales[idx].float().view(1, N, 1)
    # Add anchor score contribution to the body token scores
    delta_s = delta_s + s_anc.unsqueeze(-1)
    
    # Compare Anchor keys rotation
    diff_anchor_keys = (K_rot_full[:, 0] - AncK_e).abs()
    print(f"Anchor Keys Max Diff: {diff_anchor_keys.max().item():.6f}")
    
    # Compare Anchor scores
    diff_anchor_scores = (scores_exact[:, 0].transpose(0, 1) - s_anc).abs()
    print(f"Anchor Scores Max Diff: {diff_anchor_scores.max().item():.6f}")
    
    # Compare Delta scores
    scores_exact_delta = scores_exact[:, 1:].permute(2, 0, 1)
    diff_delta_scores = (scores_exact_delta - delta_s).abs()
    print(f"Delta Scores (Approx vs Exact) Max Diff: {diff_delta_scores.max().item():.6f}")
    print(f"Delta Scores (Approx vs Exact) Mean Diff: {diff_delta_scores.mean().item():.6f}")
    
    # Compare final outputs from both python calls
    from native_core.sparse_decode.triton_fused_decode import _pytorch_vectorized_sparse_attn_decode, fused_decode_mps
    out_exact = _pytorch_vectorized_sparse_attn_decode(
        q=q_sq.unsqueeze(0).unsqueeze(2).to(pool.anchors_K.dtype),
        block_indices=pool_indices_t,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=gpk,
        R=32,
        S_MAX=16,
        anchor_indices=anchor_indices,
        cos=cos_all,
        sin=sin_all,
        total_seq_len=prefill_len,
    ).squeeze(0).squeeze(1)
    
    out_fused, _ = fused_decode_mps(
        Q=q_sq.to(pool.anchors_K.dtype),
        pool=pool,
        block_indices=pool_indices_t,
        blk_sizes=blk_sizes,
        num_key_value_groups=gpk,
        anchor_indices=anchor_indices,
        cos=cos_all,
        sin=sin_all,
    )


    
    diff_out = (out_fused - out_exact).abs()
    print(f"\nOverall Output Max Diff: {diff_out.max().item():.6f}")
    
    wrapper.stop()

if __name__ == "__main__":
    main()
