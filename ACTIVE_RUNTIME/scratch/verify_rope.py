import torch
import math

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(k, cos, sin):
    return (k * cos) + (rotate_half(k) * sin)

def test_rope_correctness():
    # Setup dimensions
    N = 1          # 1 block
    S = 16         # 16 tokens in block
    R = 8          # rank
    H = 4          # heads
    D = 64         # head_dim
    
    # Random tensors representing unrotated variables
    U = torch.randn(N, S, R)
    V_K = torch.randn(N, R, H, D)
    anchors_K = torch.randn(N, H, D)
    scale = 0.5
    
    # Query tensor (rotated, at some decode position)
    q = torch.randn(H, D)
    
    # Cos and Sin at the block's anchor position
    # Let's say the anchor is at position 100
    cos_val = torch.randn(N, D) # [N, D]
    sin_val = torch.randn(N, D) # [N, D]
    
    # Expand cos/sin for rotation
    cos_anc = cos_val.unsqueeze(1).unsqueeze(2) # [N, 1, 1, D]
    sin_anc = sin_val.unsqueeze(1).unsqueeze(2)
    
    cos_anc_2d = cos_val.unsqueeze(1) # [N, 1, D]
    sin_anc_2d = sin_val.unsqueeze(1)
    
    # ------------------------------------------------------------------------
    # 1. Exact Reference Path:
    # ------------------------------------------------------------------------
    # Reconstruct unrotated K
    # deltas: [N, S, H, D]
    deltas = (U.float() @ V_K.float().reshape(N, R, H * D)).reshape(N, S, H, D) * scale
    K_unrot = anchors_K.unsqueeze(1) + deltas
    
    # Rotate K using the block-level RoPE approximation (same angle for all tokens in block)
    K_rot_exact = apply_rotary_pos_emb(K_unrot, cos_val.unsqueeze(1).unsqueeze(2), sin_val.unsqueeze(1).unsqueeze(2))
    
    # Compute dot product with rotated query
    # q is [H, D], K_rot_exact is [N, S, H, D]
    scores_exact = torch.einsum('hd,nshd->hns', q, K_rot_exact)
    
    # ------------------------------------------------------------------------
    # 2. Current Implementation Path (No RoPE on VK / anchors_K):
    # ------------------------------------------------------------------------
    scores_anchor_current = torch.einsum('hd,nhd->hn', q, anchors_K)
    q_proj_current = torch.einsum('hd,nrhd->nhr', q, V_K)
    scores_block_current = torch.einsum('nhr,nsr->hns', q_proj_current, U) * scale
    scores_approx_current = scores_block_current + scores_anchor_current.unsqueeze(-1)
    
    # ------------------------------------------------------------------------
    # 3. Proposed Fix Path (Apply RoPE to V_K and anchors_K before projection):
    # ------------------------------------------------------------------------
    # Rotate V_K and anchors_K by anchor position RoPE
    V_K_rot = V_K * cos_anc + rotate_half(V_K) * sin_anc
    anchors_K_rot = anchors_K * cos_anc_2d + rotate_half(anchors_K) * sin_anc_2d
    
    scores_anchor_fixed = torch.einsum('hd,nhd->hn', q, anchors_K_rot)
    q_proj_fixed = torch.einsum('hd,nrhd->nhr', q, V_K_rot)
    scores_block_fixed = torch.einsum('nhr,nsr->hns', q_proj_fixed, U) * scale
    scores_approx_fixed = scores_block_fixed + scores_anchor_fixed.unsqueeze(-1)
    
    # Compare errors
    err_current = (scores_exact - scores_approx_current).norm() / (scores_exact.norm() + 1e-12)
    err_fixed = (scores_exact - scores_approx_fixed).norm() / (scores_exact.norm() + 1e-12)
    
    print(f"Error of current implementation vs Exact: {err_current.item():.6f}")
    print(f"Error of proposed fix vs Exact:           {err_fixed.item():.6f}")

if __name__ == "__main__":
    test_rope_correctness()
