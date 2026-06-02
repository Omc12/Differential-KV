import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.hf_diffkv_wrapper import DiffKVHFWrapper

def analyze_kv_scales():
    print("=" * 60)
    print("  Analyzing Key vs Value Scale Mismatch and SVD Reconstruction Error")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 16, "micro_block_size": 16},
        device=device,
    )

    prompt = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. "
    ) * 10  # ~500 tokens
    
    encoded = wrapper.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    print(f"Prompt length: {input_ids.shape[1]} tokens")

    # We do a forward pass and extract captured prefill keys and values from layer 5
    with torch.no_grad():
        outputs = wrapper.model(input_ids=input_ids, use_cache=True)
    
    # Let's inspect the captured prefill KV in wrapper.manager
    # In hf_diffkv_wrapper.py, capture_prefill_kv captures the KV in wrapper.manager._prefill_kv_capture[sid][layer_idx]
    sid = "default"
    cap_dict = getattr(wrapper.manager, "_prefill_kv_capture", {})
    session_cap = cap_dict.get(sid, {})
    
    if not session_cap:
        print("Error: No prefill KV captured!")
        return

    # Let's analyze layer 5
    layer_idx = 5
    if layer_idx not in session_cap:
        layer_idx = list(session_cap.keys())[0]
    
    k_captured, v_captured = session_cap[layer_idx]
    # shapes: [1, num_kv_heads, seq_len, head_dim]
    print(f"Layer {layer_idx} captured KV shape: k={k_captured.shape}, v={v_captured.shape}")

    k_seq = k_captured[0].permute(1, 0, 2).float() # [seq_len, num_kv_heads, head_dim]
    v_seq = v_captured[0].permute(1, 0, 2).float() # [seq_len, num_kv_heads, head_dim]
    seq_len = k_seq.shape[0]
    num_heads = k_seq.shape[1]
    head_dim = k_seq.shape[2]

    # Compute stats for keys and values separately
    k_norm = k_seq.norm().item()
    v_norm = v_seq.norm().item()
    k_max = k_seq.abs().max().item()
    v_max = v_seq.abs().max().item()
    k_mean = k_seq.abs().mean().item()
    v_mean = v_seq.abs().mean().item()

    print(f"\nStats for original K and V:")
    print(f"  K - Norm: {k_norm:.4f}, Max: {k_max:.4f}, Mean: {k_mean:.4f}")
    print(f"  V - Norm: {v_norm:.4f}, Max: {v_max:.4f}, Mean: {v_mean:.4f}")
    print(f"  Scale Ratio (Norm V / Norm K): {v_norm / k_norm:.4f}")
    print(f"  Scale Ratio (Max V / Max K)  : {v_max / k_max:.4f}")

    # Let's segment into blocks of size 16
    block_size = 16
    rank = 8
    
    # We will test reconstruction error on a single block (tokens 1 to 16)
    k_block = k_captured[:, :, 1:block_size] # delta tokens
    v_block = v_captured[:, :, 1:block_size]
    anchor_k = k_captured[:, :, 0]
    anchor_v = v_captured[:, :, 0]
    
    # Stacked SVD (Current approach)
    # shape: [seq_len - 1, 2, num_heads, head_dim]
    stacked = torch.stack([k_block[0].transpose(0, 1), v_block[0].transpose(0, 1)], dim=1)
    flat_tokens = stacked.reshape(block_size - 1, 2 * num_heads * head_dim).float()
    anchor_flat = torch.stack([anchor_k[0], anchor_v[0]], dim=0).reshape(-1).float()
    deltas = flat_tokens - anchor_flat.unsqueeze(0)
    
    # Run SVD
    from native_core.compression.lowrank import compress_lowrank
    lr_stacked = compress_lowrank(deltas, rank)
    recon_stacked = (lr_stacked.U.float() @ lr_stacked.V.float()) * lr_stacked.scale
    
    # Separate reconstruction back to K and V
    recon_stacked_tokens = recon_stacked + anchor_flat.unsqueeze(0)
    recon_stacked_kv = recon_stacked_tokens.reshape(block_size - 1, 2, num_heads, head_dim)
    recon_k_stacked = recon_stacked_kv[:, 0]
    recon_v_stacked = recon_stacked_kv[:, 1]
    
    err_k_stacked = (recon_k_stacked - k_block[0].transpose(0, 1)).norm() / k_block[0].transpose(0, 1).norm()
    err_v_stacked = (recon_v_stacked - v_block[0].transpose(0, 1)).norm() / v_block[0].transpose(0, 1).norm()
    print(f"\nStacked SVD (Current implementation) - Rank {rank}:")
    print(f"  Key reconstruction error  : {err_k_stacked.item():.6f}")
    print(f"  Value reconstruction error: {err_v_stacked.item():.6f}")
    
    # Decoupled SVD (Separate K and V)
    # deltas for K
    k_flat = k_block[0].transpose(0, 1).reshape(block_size - 1, num_heads * head_dim).float()
    k_anchor_flat = anchor_k[0].reshape(-1).float()
    k_deltas = k_flat - k_anchor_flat.unsqueeze(0)
    
    # deltas for V
    v_flat = v_block[0].transpose(0, 1).reshape(block_size - 1, num_heads * head_dim).float()
    v_anchor_flat = anchor_v[0].reshape(-1).float()
    v_deltas = v_flat - v_anchor_flat.unsqueeze(0)
    
    lr_k = compress_lowrank(k_deltas, rank // 2)
    lr_v = compress_lowrank(v_deltas, rank // 2)
    
    recon_k_decoupled = (lr_k.U.float() @ lr_k.V.float()) * lr_k.scale + k_anchor_flat.unsqueeze(0)
    recon_v_decoupled = (lr_v.U.float() @ lr_v.V.float()) * lr_v.scale + v_anchor_flat.unsqueeze(0)
    
    err_k_decoupled = (recon_k_decoupled - k_flat).norm() / k_flat.norm()
    err_v_decoupled = (recon_v_decoupled - v_flat).norm() / v_flat.norm()
    print(f"\nDecoupled SVD (Separate K and V, Rank {rank//2} each):")
    print(f"  Key reconstruction error  : {err_k_decoupled.item():.6f}")
    print(f"  Value reconstruction error: {err_v_decoupled.item():.6f}")
    
    # Decoupled SVD (Separate K and V, Rank 8 each)
    lr_k_full = compress_lowrank(k_deltas, rank)
    lr_v_full = compress_lowrank(v_deltas, rank)
    
    recon_k_decoupled_full = (lr_k_full.U.float() @ lr_k_full.V.float()) * lr_k_full.scale + k_anchor_flat.unsqueeze(0)
    recon_v_decoupled_full = (lr_v_full.U.float() @ lr_v_full.V.float()) * lr_v_full.scale + v_anchor_flat.unsqueeze(0)
    
    err_k_decoupled_full = (recon_k_decoupled_full - k_flat).norm() / k_flat.norm()
    err_v_decoupled_full = (recon_v_decoupled_full - v_flat).norm() / v_flat.norm()
    print(f"\nDecoupled SVD (Separate K and V, Rank {rank} each):")
    print(f"  Key reconstruction error  : {err_k_decoupled_full.item():.6f}")
    print(f"  Value reconstruction error: {err_v_decoupled_full.item():.6f}")

    wrapper.stop()

if __name__ == "__main__":
    analyze_kv_scales()
