import torch

def repeat_kv_at_dim(t, n_rep, dim):
    if n_rep == 1:
        return t
    if dim < 0:
        dim = t.dim() + dim
    shape = list(t.shape)
    val = shape[dim]
    t = t.unsqueeze(dim + 1)
    expand_shape = list(t.shape)
    expand_shape[dim + 1] = n_rep
    t = t.expand(*expand_shape)
    new_shape = shape[:dim] + [val * n_rep] + shape[dim + 1:]
    return t.reshape(*new_shape)

def standard_hf_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bs, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bs, num_key_value_heads * n_rep, slen, head_dim)

# Test input
x = torch.tensor([[[[10, 11], [12, 13]], [[20, 21], [22, 23]]]]) # [1, 2, 2, 2]
# heads = 2, seq_len = 2, head_dim = 2

# HF repeat_kv with n_rep = 2 (dim 1 is heads)
hf_res = standard_hf_repeat_kv(x, 2)
print("Standard HF repeat_kv:")
print(hf_res)

# Let's test repeat_kv_at_dim on x (dim=1 is heads)
my_res = repeat_kv_at_dim(x, 2, dim=1)
print("\nOur repeat_kv_at_dim:")
print(my_res)

# Compare
if torch.equal(hf_res, my_res):
    print("\nSUCCESS: The two implementations are IDENTICAL.")
else:
    print("\nFAILED: The two implementations are DIFFERENT!")
