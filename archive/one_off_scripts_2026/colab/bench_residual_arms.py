"""Evaluation matrix for Residual Capacity & Quantization:
Hypothesis: More residual slots recover quality -> INT4 lets us afford those slots.

Matrix:
1. Baseline:      R=40,  FP16
2. More capacity: R=80,  FP16
3. Quantized:     R=80,  INT4
4. Aggressive:    R=128, INT4
"""
import os
import sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, _ROOT)

from runtime.native_block_pool import NativeBlockPool
from native_core.config import DKVConfig


def measure_pool_allocator_memory(num_blocks=1024, num_kv_heads=4, head_dim=128, rank=64):
    """Measures actual physical memory allocated by the CUDA allocator for residual buffers."""
    arms = [
        ("Baseline (R=40, FP16)", 40, "none"),
        ("More capacity (R=80, FP16)", 80, "none"),
        ("Quantized (R=80, INT4)", 80, "int4"),
        ("Aggressive (R=128, INT4)", 128, "int4"),
    ]

    print("\n" + "=" * 80)
    print("PHYSICAL CUDA ALLOCATOR MEMORY COMPARISON (1024 BLOCKS, H_kv=4, D=128)")
    print("=" * 80)
    print(f"{'Configuration':<30} | {'R':<5} | {'Prec':<6} | {'Residual VRAM':<15} | {'Total Pool':<12}")
    print("-" * 80)

    results = []
    for name, r, quant in arms:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        os.environ["DKV_RESIDUAL_QUANT"] = quant
        os.environ["DKV_MAX_RESIDUAL_TOKENS"] = str(r)

        pool = NativeBlockPool(
            max_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rank=rank,
            max_seq_len=64,
            device="cuda",
            dtype=torch.float16,
            initial_blocks=num_blocks,
            max_residual_tokens=r,
        )

        after = torch.cuda.memory_allocated()
        total_alloc_mb = (after - before) / (1024 ** 2)

        # Residual buffer memory specifically
        if quant == "int4":
            res_tensors = [pool.comp_res_k_q, pool.comp_res_v_q,
                           pool.comp_res_k_s, pool.comp_res_k_b,
                           pool.comp_res_v_s, pool.comp_res_v_b]
        else:
            res_tensors = [pool.residual_K_values, pool.residual_V_values]

        res_bytes = sum(t.numel() * t.element_size() for t in res_tensors if t is not None)
        res_mb = res_bytes / (1024 ** 2)

        print(f"{name:<30} | {r:<5} | {('INT4' if quant == 'int4' else 'FP16'):<6} | {res_mb:>10.2f} MB    | {total_alloc_mb:>8.2f} MB")
        results.append((name, r, quant, res_mb, total_alloc_mb))

        del pool
        torch.cuda.empty_cache()

    print("-" * 80)
    # Compare R=80 FP16 vs R=80 INT4
    res_fp16_80 = results[1][3]
    res_int4_80 = results[2][3]
    ratio_80 = res_fp16_80 / res_int4_80
    print(f"Memory reduction at R=80: {ratio_80:.2f}x smaller residual buffer in INT4")

    # Compare R=40 FP16 vs R=128 INT4
    res_fp16_40 = results[0][3]
    res_int4_128 = results[3][3]
    print(f"Aggressive R=128 INT4 vs Baseline R=40 FP16: {res_int4_128:.2f} MB vs {res_fp16_40:.2f} MB")
    print(f"-> 3.2x more residual capacity at {res_int4_128 / res_fp16_40:.2f}x the residual memory cost!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    measure_pool_allocator_memory()
