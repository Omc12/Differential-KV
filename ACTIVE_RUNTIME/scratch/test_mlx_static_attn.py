import mlx.core as mx
import numpy as np

@mx.compile
def compute_decode_attention_static(
    q: mx.array,              # [H_q, D]
    comp_U: mx.array,         # [max_blocks, S_comp, rank]
    comp_VK: mx.array,        # [max_blocks, kv_heads, rank, head_dim]
    comp_VV: mx.array,        # [max_blocks, kv_heads, rank, head_dim]
    comp_anc_k: mx.array,     # [max_blocks, kv_heads, head_dim]
    comp_anc_v: mx.array,     # [max_blocks, kv_heads, head_dim]
    comp_scale: mx.array,     # [max_blocks]
    comp_seq_len: mx.array,   # [max_blocks]
    num_blocks: mx.array,     # scalar
    dense_k: mx.array,        # [kv_heads, max_dense_len, head_dim]
    dense_v: mx.array,        # [kv_heads, max_dense_len, head_dim]
    dense_len: mx.array,      # scalar
    scale: float,
    gpk: int,
    kv_heads: int,
    block_size: int,
    rank: int,
    max_blocks: int,
    max_dense_len: int
):
    H_q, D = q.shape
    S_comp = block_size - 1
    
    # 1. Sparse / Compressed Attention
    block_idx = mx.arange(max_blocks)
    block_mask = block_idx < num_blocks
    block_mask_expanded = mx.expand_dims(block_mask, 0) # [1, max_blocks]
    
    AncK_e = comp_anc_k
    AncV_e = comp_anc_v
    VK_e = comp_VK
    VV_e = comp_VV
    
    if gpk > 1:
        AncK_e = mx.repeat(AncK_e, gpk, axis=1)
        AncV_e = mx.repeat(AncV_e, gpk, axis=1)
        VK_e = mx.repeat(VK_e, gpk, axis=1)
        VV_e = mx.repeat(VV_e, gpk, axis=1)
        
    AncK_e_perm = AncK_e.transpose(1, 0, 2)
    AncV_e_perm = AncV_e.transpose(1, 0, 2)
    
    s_anc = mx.sum(mx.expand_dims(q, 1) * AncK_e_perm, axis=-1) * scale
    
    VK_e_perm = VK_e.transpose(1, 0, 2, 3)
    q_expanded = mx.expand_dims(mx.expand_dims(q, 1), 2)
    q_proj_n = mx.sum(q_expanded * VK_e_perm, axis=-1) * scale
    
    # Fix broadcasting for delta_s
    q_proj_n_perm = q_proj_n.transpose(1, 0, 2) # [max_blocks, H_q, rank]
    comp_U_transposed = comp_U.transpose(0, 2, 1) # [max_blocks, rank, S_comp]
    comp_U_transposed_exp = mx.expand_dims(comp_U_transposed, 1) # [max_blocks, 1, rank, S_comp]
    q_proj_n_exp = mx.expand_dims(q_proj_n_perm, 2) # [max_blocks, H_q, 1, rank]
    
    delta_s = mx.matmul(q_proj_n_exp, comp_U_transposed_exp).squeeze(2) # [max_blocks, H_q, S_comp]
    delta_s = delta_s.transpose(1, 0, 2) # [H_q, max_blocks, S_comp]
    
    delta_s = delta_s * comp_scale.reshape(1, -1, 1)
    delta_s = delta_s + mx.expand_dims(s_anc, -1)
    
    s_range = mx.arange(S_comp).reshape(1, 1, -1)
    valid_msk = s_range < comp_seq_len.reshape(1, -1, 1)
    delta_s = mx.where(valid_msk, delta_s, -float('inf'))
    
    scores_blocks = mx.concatenate([mx.expand_dims(s_anc, -1), delta_s], axis=-1)
    scores_sparse = scores_blocks.reshape(H_q, -1)
    
    block_mask_sparse = mx.repeat(block_mask_expanded, block_size, axis=1)
    scores_sparse = mx.where(block_mask_sparse, scores_sparse, -float('inf'))
    
    lse_sparse = mx.logsumexp(scores_sparse, axis=-1)
    w = mx.softmax(scores_sparse, axis=-1)
    
    W_comp = w.reshape(H_q, max_blocks, block_size)
    w_anc = W_comp[:, :, 0]
    w_d = W_comp[:, :, 1:]
    
    w_block_sum = w_anc + mx.sum(w_d, axis=-1)
    O_anc = mx.sum(mx.expand_dims(w_block_sum, -1) * AncK_e_perm, axis=1)
    
    # Fix broadcasting for w_proj
    w_d_perm = w_d.transpose(1, 0, 2) # [max_blocks, H_q, S_comp]
    comp_U_exp = mx.expand_dims(comp_U, 1) # [max_blocks, 1, S_comp, rank]
    w_proj = mx.matmul(mx.expand_dims(w_d_perm, 2), comp_U_exp).squeeze(2) # [max_blocks, H_q, rank]
    w_proj = w_proj * comp_scale.reshape(-1, 1, 1)
    
    # Batch matrix multiplication: VV_e has shape [max_blocks, H_q, rank, head_dim]
    O_delta_block = mx.matmul(mx.expand_dims(w_proj, 2), VV_e).squeeze(2) # [max_blocks, H_q, head_dim]
    O_delta = mx.sum(O_delta_block, axis=0) # [H_q, head_dim]
    
    out_sparse = O_anc + O_delta
    
    # 2. Dense Attention
    dense_idx = mx.arange(max_dense_len)
    dense_mask = dense_idx < dense_len
    dense_mask_expanded = mx.expand_dims(dense_mask, 0)
    
    if gpk > 1:
        dense_k_rot_perm = mx.repeat(dense_k, gpk, axis=0)
        dense_v_perm = mx.repeat(dense_v, gpk, axis=0)
    else:
        dense_k_rot_perm = dense_k
        dense_v_perm = dense_v
        
    scores_dense = mx.sum(mx.expand_dims(q, 1) * dense_k_rot_perm, axis=-1) * scale
    scores_dense = mx.where(dense_mask_expanded, scores_dense, -float('inf'))
    
    lse_dense = mx.logsumexp(scores_dense, axis=-1)
    weights_dense = mx.softmax(scores_dense, axis=-1)
    out_dense = mx.sum(mx.expand_dims(weights_dense, -1) * dense_v_perm, axis=1)
    
    # 3. Combined
    lses = mx.stack([lse_sparse, lse_dense], axis=0)
    lse_max = mx.max(lses, axis=0)
    w_sparse = mx.exp(lse_sparse - lse_max)
    w_dense = mx.exp(lse_dense - lse_max)
    denom = w_sparse + w_dense
    
    out_combined = (out_sparse * mx.expand_dims(w_sparse, -1) + out_dense * mx.expand_dims(w_dense, -1)) / mx.expand_dims(denom, -1)
    return out_combined

def main():
    print("Testing compiled static attention...")
    max_blocks = 32
    max_dense_len = 768
    block_size = 256
    rank = 16
    kv_heads = 2
    gpk = 6
    H_q = 12
    head_dim = 64
    
    # Inputs
    q = mx.random.normal((H_q, head_dim))
    comp_U = mx.random.normal((max_blocks, block_size - 1, rank))
    comp_VK = mx.random.normal((max_blocks, kv_heads, rank, head_dim))
    comp_VV = mx.random.normal((max_blocks, kv_heads, rank, head_dim))
    comp_anc_k = mx.random.normal((max_blocks, kv_heads, head_dim))
    comp_anc_v = mx.random.normal((max_blocks, kv_heads, head_dim))
    comp_scale = mx.ones((max_blocks,))
    comp_seq_len = mx.array([block_size] * max_blocks, dtype=mx.int32)
    
    dense_k = mx.random.normal((kv_heads, max_dense_len, head_dim))
    dense_v = mx.random.normal((kv_heads, max_dense_len, head_dim))
    
    # Run once to compile
    print("Compiling...")
    out = compute_decode_attention_static(
        q, comp_U, comp_VK, comp_VV, comp_anc_k, comp_anc_v,
        comp_scale, comp_seq_len, mx.array(10),
        dense_k, dense_v, mx.array(512),
        1.0, gpk, kv_heads, block_size, rank, max_blocks, max_dense_len
    )
    mx.eval(out)
    print("Compiled successfully. Shape:", out.shape)
    
    # Benchmark 10 runs
    import time
    for i in range(10):
        t0 = time.perf_counter()
        out = compute_decode_attention_static(
            q, comp_U, comp_VK, comp_VV, comp_anc_k, comp_anc_v,
            comp_scale, comp_seq_len, mx.array(10),
            dense_k, dense_v, mx.array(512),
            1.0, gpk, kv_heads, block_size, rank, max_blocks, max_dense_len
        )
        mx.eval(out)
        print(f"Step {i+1} done in {(time.perf_counter() - t0)*1000:.3f}ms")

if __name__ == "__main__":
    main()
