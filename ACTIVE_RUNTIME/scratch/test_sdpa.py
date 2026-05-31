import torch
import torch.nn.functional as F

# Test SDPA causal mask with unequal q_len and kv_len
q_len = 2
kv_len = 4

q = torch.randn(1, 1, q_len, 8, device="cuda")
k = torch.randn(1, 1, kv_len, 8, device="cuda")
v = torch.randn(1, 1, kv_len, 8, device="cuda")

# 1. Compute with is_causal=True
out_causal = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

# 2. Compute with manual mask (True means attend, False means mask out)
# For F.scaled_dot_product_attention, we can pass a boolean mask of shape [q_len, kv_len]
mask = torch.ones(q_len, kv_len, dtype=torch.bool, device="cuda")
mask = torch.tril(mask, diagonal=kv_len - q_len)

# Pass the boolean mask directly
out_manual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

# Compare outputs
diff = (out_causal - out_manual).abs().max().item()
print("Max absolute difference between is_causal=True and manual mask:", diff)
