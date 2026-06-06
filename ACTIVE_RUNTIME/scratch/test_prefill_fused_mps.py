import torch
import math
from native_core.sparse_decode.triton_sparse_attn import _prefill_fused_history_attend

def test():
    device = "mps"
    dtype = torch.float16
    
    N = 4
    S = 16
    R = 8
    H = 8
    Q = 4
    D = 64
    
    torch.manual_seed(42)
    
    U = torch.randn(N, S, R, dtype=dtype)
    V_K = torch.randn(N, R, H, D, dtype=dtype)
    V_V = torch.randn(N, R, H, D, dtype=dtype)
    anchors_K = torch.randn(N, H, D, dtype=dtype)
    anchors_V = torch.randn(N, H, D, dtype=dtype)
    scales = torch.randn(N, dtype=dtype)
    cos_sliced = torch.randn(N, 1+S, 1, D, dtype=dtype)
    sin_sliced = torch.randn(N, 1+S, 1, D, dtype=dtype)
    q = torch.randn(1, H, Q, D, dtype=dtype)
    seq_lens = torch.randint(1, S, (N,), dtype=torch.int32)
    inv_scale = 1.0 / math.sqrt(D)
    
    # Run CPU
    res_cpu = _prefill_fused_history_attend(
        U.to("cpu"),
        V_K.to("cpu"),
        V_V.to("cpu"),
        anchors_K.to("cpu"),
        anchors_V.to("cpu"),
        scales.to("cpu"),
        cos_sliced.to("cpu"),
        sin_sliced.to("cpu"),
        q.to("cpu"),
        seq_lens.to("cpu"),
        inv_scale
    )
    
    # Run MPS
    res_mps = _prefill_fused_history_attend(
        U.to(device),
        V_K.to(device),
        V_V.to(device),
        anchors_K.to(device),
        anchors_V.to(device),
        scales.to(device),
        cos_sliced.to(device),
        sin_sliced.to(device),
        q.to(device),
        seq_lens.to(device),
        inv_scale
    )
    
    diff = (res_mps.to("cpu") - res_cpu).abs()
    print(f"Fused History Attend MPS vs CPU Max Diff: {diff.max().item():.6f}")
    print(f"Fused History Attend MPS vs CPU Mean Diff: {diff.mean().item():.6f}")

if __name__ == "__main__":
    test()
