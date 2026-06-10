import mlx.core as mx
import time

num_layers = 28
kv_heads = 2
head_dim = 64
max_dense_len = 768
rank = 16
block_size = 256
max_blocks = 32

@mx.compile
def compute_decode_attention_static(
    q: mx.array,
    comp_U: mx.array,
    comp_VK: mx.array,
    comp_VV: mx.array,
    comp_anc_k: mx.array,
    comp_anc_v: mx.array,
    comp_scale: mx.array,
    comp_seq_len: mx.array,
    num_blocks: mx.array,
    dense_k: mx.array,
    dense_v: mx.array,
    dense_len: mx.array,
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
    block_mask_expanded = mx.expand_dims(block_mask, 0)
    
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
    
    q_proj_n_perm = q_proj_n.transpose(1, 0, 2)
    comp_U_transposed = comp_U.transpose(0, 2, 1)
    comp_U_transposed_exp = mx.expand_dims(comp_U_transposed, 1)
    q_proj_n_exp = mx.expand_dims(q_proj_n_perm, 2)
    
    delta_s = mx.matmul(q_proj_n_exp, comp_U_transposed_exp).squeeze(2)
    delta_s = delta_s.transpose(1, 0, 2)
    
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
    
    w_d_perm = w_d.transpose(1, 0, 2)
    comp_U_exp = mx.expand_dims(comp_U, 1)
    w_proj = mx.matmul(mx.expand_dims(w_d_perm, 2), comp_U_exp).squeeze(2)
    w_proj = w_proj * comp_scale.reshape(-1, 1, 1)
    
    O_delta_block = mx.matmul(mx.expand_dims(w_proj, 2), VV_e).squeeze(2)
    O_delta = mx.sum(O_delta_block, axis=0)
    
    out_sparse = O_anc + O_delta
    out_sparse = mx.where(mx.isnan(out_sparse), 0.0, out_sparse)
    
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
    out_dense = mx.where(mx.isnan(out_dense), 0.0, out_dense)
    
    # 3. Combined
    lses = mx.stack([lse_sparse, lse_dense], axis=0)
    lse_max = mx.max(lses, axis=0)
    w_sparse = mx.exp(lse_sparse - lse_max)
    w_dense = mx.exp(lse_dense - lse_max)
    denom = w_sparse + w_dense
    
    out_combined = (out_sparse * mx.expand_dims(w_sparse, -1) + out_dense * mx.expand_dims(w_dense, -1)) / mx.expand_dims(denom, -1)
    return out_combined

def simulate_step(session, q_list, k_list, v_list, dense_lens):
    # Simulate ingest for all layers
    for layer_idx in range(num_layers):
        dense_len = dense_lens[layer_idx]
        session["dense_keys"][layer_idx, 0, :, dense_len:dense_len + 1] = k_list[layer_idx].squeeze(0)
        session["dense_values"][layer_idx, 0, :, dense_len:dense_len + 1] = v_list[layer_idx].squeeze(0)
        dense_lens[layer_idx] += 1
        
        # We evaluate the slice assignment
        mx.eval(session["dense_keys"][layer_idx], session["dense_values"][layer_idx])
        
    # Simulate execute_decode_attention for all layers
    outputs = []
    for layer_idx in range(num_layers):
        q = q_list[layer_idx].squeeze(2).squeeze(0)
        comp_U = session["comp_U"][layer_idx]
        comp_VK = session["comp_VK"][layer_idx]
        comp_VV = session["comp_VV"][layer_idx]
        comp_anc_k = session["comp_anc_k"][layer_idx]
        comp_anc_v = session["comp_anc_v"][layer_idx]
        comp_scale = session["comp_scale"][layer_idx]
        comp_seq_len = session["comp_seq_len"][layer_idx]
        
        num_blocks = mx.array(session["num_blocks"][layer_idx])
        dense_k = session["dense_keys"][layer_idx, 0]
        dense_v = session["dense_values"][layer_idx, 0]
        dense_len = mx.array(dense_lens[layer_idx])
        
        out = compute_decode_attention_static(
            q, comp_U, comp_VK, comp_VV, comp_anc_k, comp_anc_v,
            comp_scale, comp_seq_len, num_blocks,
            dense_k, dense_v, dense_len,
            1.0, 6, kv_heads, block_size, rank, max_blocks, max_dense_len
        )
        outputs.append(out)
        
    # Final evaluation of logits/outputs
    mx.eval(*outputs)

def main():
    print("Initializing state...")
    session = {
        "dense_keys": mx.zeros((num_layers, 1, kv_heads, max_dense_len, head_dim)),
        "dense_values": mx.zeros((num_layers, 1, kv_heads, max_dense_len, head_dim)),
        "num_blocks": [0 for _ in range(num_layers)],
        "comp_U": mx.zeros((num_layers, max_blocks, block_size - 1, rank)),
        "comp_VK": mx.zeros((num_layers, max_blocks, kv_heads, rank, head_dim)),
        "comp_VV": mx.zeros((num_layers, max_blocks, kv_heads, rank, head_dim)),
        "comp_anc_k": mx.zeros((num_layers, max_blocks, kv_heads, head_dim)),
        "comp_anc_v": mx.zeros((num_layers, max_blocks, kv_heads, head_dim)),
        "comp_scale": mx.zeros((num_layers, max_blocks)),
        "comp_seq_len": mx.zeros((num_layers, max_blocks), dtype=mx.int32),
    }
    
    dense_lens = [512 for _ in range(num_layers)]
    
    q_list = [mx.random.normal((1, 12, 1, head_dim)) for _ in range(num_layers)]
    k_list = [mx.random.normal((1, kv_heads, 1, head_dim)) for _ in range(num_layers)]
    v_list = [mx.random.normal((1, kv_heads, 1, head_dim)) for _ in range(num_layers)]
    
    print("Warming up / compiling...")
    simulate_step(session, q_list, k_list, v_list, dense_lens)
    print("Warmup done.")
    
    print("Benchmarking 10 decode steps...")
    for step in range(10):
        t0 = time.perf_counter()
        simulate_step(session, q_list, k_list, v_list, dense_lens)
        print(f"Step {step+1} took {(time.perf_counter() - t0)*1000:.2f}ms")

if __name__ == "__main__":
    main()
