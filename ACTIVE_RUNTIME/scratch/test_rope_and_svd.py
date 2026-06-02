import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.hf_diffkv_wrapper import DiffKVHFWrapper

def diagnose_rope_and_svd():
    print("=" * 60)
    print("  Rope and SVD Mathematics Diagnostics")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32},
        device=device,
    )

    prompt = "Quantum computing is a multidisciplinary field. " * 10
    encoded = wrapper.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)

    with torch.no_grad():
        outputs = wrapper.model(input_ids=input_ids, use_cache=True)
    
    sid = "default"
    cap_dict = getattr(wrapper.manager, "_prefill_kv_capture", {})
    session_cap = cap_dict.get(sid, {})
    
    layer_idx = 5
    k_captured, v_captured = session_cap[layer_idx]
    
    # Let's inspect a single block (tokens 0 to 32)
    block_size = 32
    k_block = k_captured[0, :, :block_size].float() # [heads, block_size, head_dim]
    v_block = v_captured[0, :, :block_size].float() # [heads, block_size, head_dim]
    
    # Print original unrotated K and V
    print(f"Original unrotated K block: shape={k_block.shape}, min={k_block.min().item():.4f}, max={k_block.max().item():.4f}")
    
    # ── Let's perform SVD compression on this block ──
    # Anchor is token 0
    anchor_k = k_block[:, 0] # [heads, head_dim]
    anchor_v = v_block[:, 0]
    
    delta_k = k_block[:, 1:] - anchor_k.unsqueeze(1) # [heads, block_size-1, head_dim]
    delta_v = v_block[:, 1:] - anchor_v.unsqueeze(1)
    
    # Flatten deltas
    seq_len = block_size - 1
    num_heads = k_block.shape[0]
    head_dim = k_block.shape[2]
    feat_dim = 2 * num_heads * head_dim
    
    # stack deltas
    stacked = torch.stack([k_block[:, 1:].transpose(0, 1), v_block[:, 1:].transpose(0, 1)], dim=1) # [seq_len, 2, heads, head_dim]
    flat_tokens = stacked.reshape(seq_len, feat_dim)
    anchor_flat = torch.stack([anchor_k, anchor_v], dim=0).reshape(-1)
    
    deltas = flat_tokens - anchor_flat.unsqueeze(0)
    
    # Run SVD
    from native_core.compression.lowrank import compress_lowrank
    rank = 16
    lr = compress_lowrank(deltas, rank)
    
    print(f"SVD Scale: {lr.scale:.4f}")
    print(f"U shape: {lr.U.shape}, min={lr.U.min().item():.4f}, max={lr.U.max().item():.4f}")
    print(f"V shape: {lr.V.shape}, min={lr.V.min().item():.4f}, max={lr.V.max().item():.4f}")
    
    # Reconstruct unrotated keys
    recon_deltas = (lr.U @ lr.V) * lr.scale
    recon_flat = recon_deltas + anchor_flat.unsqueeze(0)
    recon_stacked = recon_flat.reshape(seq_len, 2, num_heads, head_dim)
    
    recon_k_unrot = torch.zeros(block_size, num_heads, head_dim, device=device)
    recon_k_unrot[0] = anchor_k
    recon_k_unrot[1:] = recon_stacked[:, 0]
    
    # Let's apply RoPE to original unrotated keys and reconstructed unrotated keys
    # Position IDs are 0 to block_size-1
    pos_ids = torch.arange(block_size, device=device).unsqueeze(0)
    cos, sin = wrapper.model.model.rotary_emb(v_block.unsqueeze(0).transpose(1, 2), pos_ids)
    
    # original rotated keys
    cos_sliced = cos.squeeze(0).unsqueeze(1) # [block_size, 1, head_dim]
    sin_sliced = sin.squeeze(0).unsqueeze(1)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    k_block_perm = k_block.transpose(0, 1) # [block_size, heads, head_dim]
    k_rot_orig = k_block_perm * cos_sliced + rotate_half(k_block_perm) * sin_sliced
    
    # reconstructed rotated keys
    k_rot_recon = recon_k_unrot * cos_sliced + rotate_half(recon_k_unrot) * sin_sliced
    
    # Let's compare them!
    err_unrot = (recon_k_unrot - k_block_perm).norm() / k_block_perm.norm()
    err_rot = (k_rot_recon - k_rot_orig).norm() / k_rot_orig.norm()
    
    print(f"\nReconstruction Analysis:")
    print(f"  Unrotated key reconstruction rel error: {err_unrot.item():.6f}")
    print(f"  Rotated key reconstruction rel error  : {err_rot.item():.6f}")
    
    # Let's compute attention scores with a random query
    q = torch.randn(num_heads, head_dim, device=device) * 0.5
    
    scores_orig = torch.sum(q.unsqueeze(0) * k_rot_orig, dim=-1) / math.sqrt(head_dim) # [block_size, heads]
    scores_recon = torch.sum(q.unsqueeze(0) * k_rot_recon, dim=-1) / math.sqrt(head_dim) # [block_size, heads]
    
    print(f"\nAttention Score Analysis:")
    print(f"  Original scores : min={scores_orig.min().item():.4f}, max={scores_orig.max().item():.4f}, mean={scores_orig.mean().item():.4f}")
    print(f"  Recon scores    : min={scores_recon.min().item():.4f}, max={scores_recon.max().item():.4f}, mean={scores_recon.mean().item():.4f}")
    
    diff_scores = (scores_recon - scores_orig).abs()
    print(f"  Absolute score difference: max={diff_scores.max().item():.4f}, mean={diff_scores.mean().item():.4f}")
    
    wrapper.stop()

if __name__ == "__main__":
    diagnose_rope_and_svd()
