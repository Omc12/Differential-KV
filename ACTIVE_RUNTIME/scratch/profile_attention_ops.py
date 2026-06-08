import torch
import time
import math

DEVICE = torch.device("mps")
D = 64
H_q = 16
H_kv = 2
g = H_q // H_kv  # 8
L_dense = 256
rank = 32
K_slots = 3
S_max = 256

# Mock inputs
Q = torch.randn(1, H_q, 1, D, device=DEVICE, dtype=torch.float16)
_Q_sq = Q[0, :, 0, :]  # [H_q, D]

dense_k = torch.randn(1, H_kv, L_dense, D, device=DEVICE, dtype=torch.float16)
dense_v = torch.randn(1, H_kv, L_dense, D, device=DEVICE, dtype=torch.float16)
cos_dense = torch.randn(1, 1, L_dense, D, device=DEVICE, dtype=torch.float16)
sin_dense = torch.randn(1, 1, L_dense, D, device=DEVICE, dtype=torch.float16)

# Pool data
U = torch.randn(10, S_max, rank, device=DEVICE, dtype=torch.float16)
U_scale = torch.randn(10, device=DEVICE, dtype=torch.float16)
V_K = torch.randn(10, rank, H_kv, D, device=DEVICE, dtype=torch.float16)
V_V = torch.randn(10, rank, H_kv, D, device=DEVICE, dtype=torch.float16)
anchors_K = torch.randn(10, H_kv, D, device=DEVICE, dtype=torch.float16)
anchors_V = torch.randn(10, H_kv, D, device=DEVICE, dtype=torch.float16)
seq_lens = torch.randint(10, 256, (10,), device=DEVICE, dtype=torch.int32)
block_indices = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int32)

# Warmup
torch.mps.synchronize()

def profile_op(name, fn, iterations=500):
    # Warmup
    for _ in range(20):
        fn()
    torch.mps.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.mps.synchronize()
    t_avg = (time.perf_counter() - t0) * 1000 / iterations
    print(f"{name:<50} : {t_avg:.4f} ms")
    return t_avg

# 1. RoPE computation in-place
workspace_k_rot = torch.zeros((1, H_kv, L_dense, D), device=DEVICE, dtype=torch.float16)
workspace_k_half = torch.zeros((1, H_kv, L_dense, D), device=DEVICE, dtype=torch.float16)

def test_rope():
    half_d = D // 2
    workspace_k_half[..., :half_d] = -dense_k[..., half_d:]
    workspace_k_half[..., half_d:] = dense_k[..., :half_d]
    torch.mul(dense_k, cos_dense, out=workspace_k_rot)
    workspace_k_rot.addcmul_(workspace_k_half, sin_dense)

profile_op("1. In-place RoPE computation", test_rope)

# 2. repeat_kv
def test_repeat_kv():
    hidden_states = workspace_k_rot
    bs, num_kv, slen, hdim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bs, num_kv, g, slen, hdim)
    out = hidden_states.reshape(bs, num_kv * g, slen, hdim)

profile_op("2. repeat_kv (reshape/copy overhead)", test_repeat_kv)

# 3. Dense SDPA
k_rep = workspace_k_rot.repeat_interleave(g, dim=1)
v_rep = dense_v.repeat_interleave(g, dim=1)
def test_dense_sdpa():
    F_out = torch.nn.functional.scaled_dot_product_attention(
        Q, k_rep, v_rep, is_causal=False
    )

profile_op("3. Dense SDPA", test_dense_sdpa)

# 4. Dense LSE
_scale = D ** -0.5
def test_dense_lse():
    _kd = k_rep[0]
    scores_dense = torch.matmul(_kd, _Q_sq.unsqueeze(-1)).squeeze(-1) * _scale
    lse_dense = torch.logsumexp(scores_dense.float(), dim=-1)

profile_op("4. Dense LSE (matmul + logsumexp)", test_dense_lse)

# 5. Metal shader decode_attention_metal
try:
    import sys
    sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/native_core/diffkv_core")
    import diffkv_core
    _scale = 1.0 / math.sqrt(D)
    def test_metal():
        out, lse = diffkv_core.decode_attention_metal(
            _Q_sq.contiguous(),
            U.contiguous(),
            U_scale.contiguous(),
            V_K.contiguous(),
            V_V.contiguous(),
            anchors_K.contiguous(),
            anchors_V.contiguous(),
            seq_lens.contiguous(),
            block_indices.contiguous(),
            _scale,
            H_q,
            H_kv,
            rank,
        )
    profile_op("5. Metal shader (decode_attention_metal)", test_metal)
except Exception as e:
    print("Failed to test Metal shader:", e)

# 6. Combining logic
lse_dense = torch.randn(H_q, device=DEVICE, dtype=torch.float32)
lse_sparse = torch.randn(H_q, device=DEVICE, dtype=torch.float32)
out_dense = torch.randn(1, H_q, 1, D, device=DEVICE, dtype=torch.float16)
out_sparse = torch.randn(H_q, D, device=DEVICE, dtype=torch.float16)

def test_combine():
    out_dense_hd = out_dense[0, :, 0, :].float()
    out_sparse_fp32 = out_sparse.float()
    lse_dense_32 = lse_dense.to(torch.float32)
    lse_sparse_32 = lse_sparse.to(torch.float32)

    lse_max = torch.maximum(lse_dense_32, lse_sparse_32)
    w_dense = torch.exp(lse_dense_32 - lse_max)
    w_sparse = torch.exp(lse_sparse_32 - lse_max)
    denom = w_dense + w_sparse

    out_final = (out_dense_hd * w_dense.unsqueeze(-1) +
                 out_sparse_fp32 * w_sparse.unsqueeze(-1)) / denom.unsqueeze(-1)
    attn_out_b = out_final.to(torch.float16).unsqueeze(0).unsqueeze(2)

profile_op("6. Combining logic (FP32 LSE math)", test_combine)

# 7. Total python loop (sum of all ops per layer)
def test_all():
    # 1. RoPE
    half_d = D // 2
    workspace_k_half[..., :half_d] = -dense_k[..., half_d:]
    workspace_k_half[..., half_d:] = dense_k[..., :half_d]
    torch.mul(dense_k, cos_dense, out=workspace_k_rot)
    workspace_k_rot.addcmul_(workspace_k_half, sin_dense)
    
    # 2. repeat_kv
    k_rep = workspace_k_rot.repeat_interleave(g, dim=1)
    v_rep = dense_v.repeat_interleave(g, dim=1)
    
    # 3. Dense SDPA
    out_dense = torch.nn.functional.scaled_dot_product_attention(
        Q, k_rep, v_rep, is_causal=False
    )
    
    # 4. Dense LSE
    _kd = k_rep[0]
    scores_dense = torch.matmul(_kd, _Q_sq.unsqueeze(-1)).squeeze(-1) * _scale
    lse_dense = torch.logsumexp(scores_dense.float(), dim=-1)
    
    # 5. Metal
    out_sparse, lse_sparse = diffkv_core.decode_attention_metal(
        _Q_sq.contiguous(),
        U.contiguous(),
        U_scale.contiguous(),
        V_K.contiguous(),
        V_V.contiguous(),
        anchors_K.contiguous(),
        anchors_V.contiguous(),
        seq_lens.contiguous(),
        block_indices.contiguous(),
        _scale,
        H_q,
        H_kv,
        rank,
    )
    
    # 6. Combine
    out_dense_hd = out_dense[0, :, 0, :].float()
    out_sparse_fp32 = out_sparse.float()
    lse_dense_32 = lse_dense.to(torch.float32)
    lse_sparse_32 = lse_sparse.to(torch.float32)

    lse_max = torch.maximum(lse_dense_32, lse_sparse_32)
    w_dense = torch.exp(lse_dense_32 - lse_max)
    w_sparse = torch.exp(lse_sparse_32 - lse_max)
    denom = w_dense + w_sparse

    out_final = (out_dense_hd * w_dense.unsqueeze(-1) +
                 out_sparse_fp32 * w_sparse.unsqueeze(-1)) / denom.unsqueeze(-1)
    attn_out_b = out_final.to(torch.float16).unsqueeze(0).unsqueeze(2)

profile_op("7. COMPLETE decode attention (all ops sequentially)", test_all)
