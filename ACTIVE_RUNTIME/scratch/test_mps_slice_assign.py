import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Testing on device: {device}")

# Create pool
pool = torch.zeros((5, 10, 4), device=device, dtype=torch.float16)

# Create block data
U = torch.ones((8, 4), device=device, dtype=torch.float16) * 3.5

# Write to slice
pool[2, :8, :4] = U

# Read back
readback = pool[2, :8, :4]
print(f"Original U sum: {U.sum().item():.4f}")
print(f"Readback U sum: {readback.sum().item():.4f}")
print(f"Pool sum: {pool.sum().item():.4f}")

if torch.equal(U, readback):
    print("SUCCESS: Slice assignment matches perfectly.")
else:
    print("FAILED: Slice assignment corrupted the data!")
