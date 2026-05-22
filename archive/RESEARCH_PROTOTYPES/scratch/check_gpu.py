import torch
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
else:
    print("CUDA not available")
