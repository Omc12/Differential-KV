import torch
import torch.nn.functional as F

def test_sdpa():
    device = "mps"
    torch.manual_seed(42)
    B, H, T, D = 1, 8, 128, 64
    
    q = torch.randn(B, H, T, D, dtype=torch.float16)
    k = torch.randn(B, H, T, D, dtype=torch.float16)
    v = torch.randn(B, H, T, D, dtype=torch.float16)
    
    # CPU reference
    q_cpu, k_cpu, v_cpu = q.to("cpu"), k.to("cpu"), v.to("cpu")
    out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=True)
    
    # MPS execution
    q_mps, k_mps, v_mps = q.to(device), k.to(device), v.to(device)
    out_mps = F.scaled_dot_product_attention(q_mps, k_mps, v_mps, is_causal=True)
    
    diff = (out_mps.to("cpu") - out_cpu).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"SDPA Causal MPS vs CPU Max Diff: {max_diff:.6f}")
    print(f"SDPA Causal MPS vs CPU Mean Diff: {mean_diff:.6f}")

if __name__ == "__main__":
    test_sdpa()
