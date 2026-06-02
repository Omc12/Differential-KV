import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Testing on device: {device}")

# Create a tensor with some -inf values
x = torch.tensor([1.0, 2.0, float('-inf'), 3.0], device=device)
probs = torch.softmax(x, dim=-1)
print(f"Input: {x}")
print(f"Softmax: {probs}")

# Check for NaN
if torch.isnan(probs).any():
    print("FAILED: Softmax produced NaN!")
else:
    print("SUCCESS: Softmax is fine.")
