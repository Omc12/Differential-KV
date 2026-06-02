#!/usr/bin/env python3
import os
import sys
import math
import torch

# Simple RoPE implementation
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, pos, theta=10000.0):
    # x: [seq_len, dim]
    seq_len, dim = x.shape
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(pos, pos + seq_len).float()
    freqs = torch.outer(t, inv_freq) # [seq_len, dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, dim]
    cos = emb.cos()
    sin = emb.sin()
    return x * cos + rotate_half(x) * sin

def main():
    print("=" * 60)
    print("  RoPE-SVD Mismatch Diagnosis Test")
    print("=" * 60)
    
    seq_len = 32
    dim = 64
    rank = 8
    
    # 1. Generate unrotated keys (smooth, highly low-rank semantic signals)
    torch.manual_seed(42)
    t = torch.linspace(0, 2 * math.pi, seq_len).unsqueeze(1)
    freqs = torch.arange(1, 5).float()
    # Unrotated keys are a combination of low frequency sines/cosines
    k_unrot = torch.zeros(seq_len, dim)
    for f in freqs:
        k_unrot += torch.sin(f * t) * torch.randn(dim)
        k_unrot += torch.cos(f * t) * torch.randn(dim)
    
    # Standard query vector
    q_unrot = torch.randn(dim)
    q_rot = apply_rotary(q_unrot.unsqueeze(0), pos=seq_len)[0]
    
    # Apply RoPE to get rotated keys
    k_rot = apply_rotary(k_unrot, pos=0)
    
    print(f"Unrotated Keys: shape={k_unrot.shape}")
    print(f"Rotated Keys  : shape={k_rot.shape}")
    
    # ----------------------------------------------------
    # Method A: SVD on Rotated Keys (Post-RoPE SVD - Current)
    # ----------------------------------------------------
    anchor_a = k_rot[0]
    delta_a = k_rot - anchor_a.unsqueeze(0)
    
    U_a, S_a, Vh_a = torch.linalg.svd(delta_a, full_matrices=False)
    # Project to rank
    U_a_proj = U_a[:, :rank] * S_a[:rank].unsqueeze(0)
    Vh_a_proj = Vh_a[:rank, :]
    
    recon_delta_a = U_a_proj @ Vh_a_proj
    recon_k_rot_a = anchor_a.unsqueeze(0) + recon_delta_a
    
    # Attention scores
    true_scores = (k_rot @ q_rot) / math.sqrt(dim)
    scores_a = (recon_k_rot_a @ q_rot) / math.sqrt(dim)
    
    # ----------------------------------------------------
    # Method B: SVD on Unrotated Keys (Pre-RoPE SVD - Proposed)
    # ----------------------------------------------------
    anchor_b = k_unrot[0]
    delta_b = k_unrot - anchor_b.unsqueeze(0)
    
    U_b, S_b, Vh_b = torch.linalg.svd(delta_b, full_matrices=False)
    # Project to rank
    U_b_proj = U_b[:, :rank] * S_b[:rank].unsqueeze(0)
    Vh_b_proj = Vh_b[:rank, :]
    
    recon_delta_b = U_b_proj @ Vh_b_proj
    recon_k_unrot_b = anchor_b.unsqueeze(0) + recon_delta_b
    
    # Apply RoPE to the reconstructed unrotated keys
    recon_k_rot_b = apply_rotary(recon_k_unrot_b, pos=0)
    scores_b = (recon_k_rot_b @ q_rot) / math.sqrt(dim)
    
    # ── Comparison ──
    err_a_k = (recon_k_rot_a - k_rot).norm() / k_rot.norm()
    err_b_k = (recon_k_rot_b - k_rot).norm() / k_rot.norm()
    
    err_a_scores = (scores_a - true_scores).norm() / true_scores.norm()
    err_b_scores = (scores_b - true_scores).norm() / true_scores.norm()
    
    print("\nResults Summary:")
    print("-" * 50)
    print(f"Method A (Post-RoPE SVD - Current):")
    print(f"  Key Reconstruction Rel Error : {err_a_k.item():.6f}")
    print(f"  Attention Score Rel Error    : {err_a_scores.item():.6f}")
    print(f"  Singular Values              : {S_a[:rank].tolist()}")
    print(f"  SVD Energy Captured          : {((S_a[:rank]**2).sum() / (S_a**2).sum()).item():.6f}")
    
    print("-" * 50)
    print(f"Method B (Pre-RoPE SVD - Proposed):")
    print(f"  Key Reconstruction Rel Error : {err_b_k.item():.6f}")
    print(f"  Attention Score Rel Error    : {err_b_scores.item():.6f}")
    print(f"  Singular Values              : {S_b[:rank].tolist()}")
    print(f"  SVD Energy Captured          : {((S_b[:rank]**2).sum() / (S_b**2).sum()).item():.6f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
