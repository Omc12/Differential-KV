import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Testing on device: {device}")

# pool
V_K = torch.zeros((5, 8, 2, 4), device=device, dtype=torch.float16)

# V (rank=8, d=16, which is 2 * 2 * 4)
V = torch.arange(8 * 16, device=device, dtype=torch.float16).view(8, 16)

# vk slice and view
num_kv = 2
h_dim = 4
vk = V[:, :num_kv * h_dim].view(8, num_kv, h_dim)

# Write to pool
V_K[2, :8] = vk

# Read back
readback = V_K[2, :8]
expected = vk

print(f"Is contiguous (vk): {vk.is_contiguous()}")
print(f"Is contiguous (V_K slice): {V_K[2, :8].is_contiguous()}")

# Compare
if torch.equal(expected, readback):
    print("SUCCESS: Non-contiguous slice assignment matches perfectly.")
else:
    print("FAILED: Non-contiguous slice assignment corrupted the data!")
    print("Expected:")
    print(expected)
    print("Readback:")
    print(readback)
