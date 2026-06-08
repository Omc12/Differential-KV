import os
import sys
import math
import torch

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    import diffkv_core
    print("[DiffKV Test] diffkv_core successfully imported.")
    print("  HAS_DECODE_ATTN:", getattr(diffkv_core, "HAS_DECODE_ATTN", False))
    print("  HAS_METAL_ATTN :", getattr(diffkv_core, "HAS_METAL_ATTN", False))
except ImportError as e:
    print("[DiffKV Test] ERROR: Failed to import diffkv_core:", e)
    sys.exit(1)

if not getattr(diffkv_core, "HAS_METAL_ATTN", False):
    print("[DiffKV Test] ERROR: diffkv_core compiled without Metal support!")
    sys.exit(1)

# Ensure device is MPS (Apple Silicon GPU)
if not torch.backends.mps.is_available():
    print("[DiffKV Test] ERROR: MPS device not available on this system!")
    sys.exit(1)

DEVICE = torch.device("mps")
print(f"[DiffKV Test] Running correctness test on device: {DEVICE}")

def run_test():
    torch.manual_seed(42)

    # Dimensions
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    rank = 16
    S_max = 256
    num_blocks = 20
    K_active = 8
    scale = 1.0 / math.sqrt(head_dim)

    # 1. Initialize random input tensors on MPS
    Q = torch.randn((num_heads, head_dim), dtype=torch.float16, device=DEVICE)
    
    # Pool variables (must be large enough to hold index_select)
    U_pool = torch.randint(-128, 127, (num_blocks, S_max, rank), dtype=torch.int8, device=DEVICE)
    U_scale_pool = torch.rand((num_blocks,), dtype=torch.float16, device=DEVICE) * 0.05
    VK_pool = torch.randn((num_blocks, rank, num_kv_heads, head_dim), dtype=torch.float16, device=DEVICE) * 0.1
    VV_pool = torch.randn((num_blocks, rank, num_kv_heads, head_dim), dtype=torch.float16, device=DEVICE) * 0.1
    anchors_K = torch.randn((num_blocks, num_kv_heads, head_dim), dtype=torch.float16, device=DEVICE) * 0.5
    anchors_V = torch.randn((num_blocks, num_kv_heads, head_dim), dtype=torch.float16, device=DEVICE) * 0.5
    
    # Sequence lengths per block (between 64 and S_max)
    seq_lens = torch.randint(64, S_max, (num_blocks,), dtype=torch.int32, device=DEVICE)
    
    # Active slot indices selection
    slot_indices = torch.tensor([1, 3, 5, 7, 11, 13, 17, 19], dtype=torch.int32, device=DEVICE)

    print(f"\n[Test Case 1] Standard execution with K={K_active} blocks...")
    # ── Call C++ ATen Reference ──
    out_cpp, lse_cpp = diffkv_core.decode_attention_aten_lse(
        Q.contiguous(),
        U_pool.contiguous(),
        U_scale_pool.contiguous(),
        VK_pool.contiguous(),
        VV_pool.contiguous(),
        anchors_K.contiguous(),
        anchors_V.contiguous(),
        seq_lens.contiguous(),
        slot_indices.contiguous(),
        scale,
        num_heads,
        num_kv_heads,
        rank
    )

    # ── Call Metal Shader ──
    out_metal, lse_metal = diffkv_core.decode_attention_metal(
        Q.contiguous(),
        U_pool.contiguous(),
        U_scale_pool.contiguous(),
        VK_pool.contiguous(),
        VV_pool.contiguous(),
        anchors_K.contiguous(),
        anchors_V.contiguous(),
        seq_lens.contiguous(),
        slot_indices.contiguous(),
        scale,
        num_heads,
        num_kv_heads,
        rank
    )

    # Synchronize MPS to ensure execution finishes
    torch.mps.synchronize()

    # Compare outputs
    max_diff_out = torch.max(torch.abs(out_cpp - out_metal)).item()
    mean_diff_out = torch.mean(torch.abs(out_cpp - out_metal)).item()
    print(f"  Output Vector Difference -> Max: {max_diff_out:.6f} | Mean: {mean_diff_out:.6f}")

    # LSE comparison
    max_diff_lse = torch.max(torch.abs(lse_cpp - lse_metal)).item()
    mean_diff_lse = torch.mean(torch.abs(lse_cpp - lse_metal)).item()
    print(f"  LSE Vector Difference    -> Max: {max_diff_lse:.6f} | Mean: {mean_diff_lse:.6f}")

    # Parity assertions (float16 math allows small epsilon difference due to accumulator ordering)
    success = max_diff_out < 0.02 and max_diff_lse < 0.05
    print("  Result Case 1:", "PASS" if success else "FAIL")
    if not success:
        return False

    # ── Test Case 2: Zero/Empty active slots ──
    print("\n[Test Case 2] Empty slot list execution...")
    empty_slots = torch.zeros((0,), dtype=torch.int32, device=DEVICE)
    
    out_cpp_empty, lse_cpp_empty = diffkv_core.decode_attention_aten_lse(
        Q.contiguous(), U_pool, U_scale_pool, VK_pool, VV_pool, anchors_K, anchors_V, seq_lens, empty_slots,
        scale, num_heads, num_kv_heads, rank
    )
    out_metal_empty, lse_metal_empty = diffkv_core.decode_attention_metal(
        Q.contiguous(), U_pool, U_scale_pool, VK_pool, VV_pool, anchors_K, anchors_V, seq_lens, empty_slots,
        scale, num_heads, num_kv_heads, rank
    )
    torch.mps.synchronize()

    max_diff_empty = torch.max(torch.abs(out_cpp_empty - out_metal_empty)).item()
    print(f"  Empty Slot Output Difference: {max_diff_empty:.6f}")
    assert max_diff_empty == 0.0, "Empty slot output mismatch!"
    assert (lse_metal_empty == float('-inf')).all(), "Empty LSE should be -inf!"
    print("  Result Case 2: PASS")

    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
