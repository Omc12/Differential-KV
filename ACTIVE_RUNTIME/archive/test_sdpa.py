import torch
import torch.nn.functional as F

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
DEVICE = 'cuda'

bsz = 1
heads = 14
q_len = 25000
k_len = 25000
head_dim = 64

q = torch.randn(bsz, heads, q_len, head_dim, dtype=torch.float16, device=DEVICE)
k = torch.randn(bsz, heads, k_len, head_dim, dtype=torch.float16, device=DEVICE)
v = torch.randn(bsz, heads, k_len, head_dim, dtype=torch.float16, device=DEVICE)

print("Before SDPA:", torch.cuda.memory_allocated(DEVICE)/1e6, "MB")

with torch.no_grad():
    try:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print("After SDPA:", torch.cuda.memory_allocated(DEVICE)/1e6, "MB")
        print("Peak SDPA:", torch.cuda.max_memory_allocated(DEVICE)/1e6, "MB")
    except Exception as e:
        print("Error:", e)
