import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from native_core.compression.lowrank import compress_lowrank

def test():
    print("Testing compress_lowrank on CPU...")
    # shape [16, 2048]
    deltas = torch.randn(16, 2048, dtype=torch.float32)
    t0 = time.perf_counter()
    lr = compress_lowrank(deltas, rank=16)
    t1 = time.perf_counter()
    print(f"SVD completed in {(t1-t0)*1000:.2f}ms. U shape: {lr.U.shape}, V shape: {lr.V.shape}")

if __name__ == "__main__":
    test()
