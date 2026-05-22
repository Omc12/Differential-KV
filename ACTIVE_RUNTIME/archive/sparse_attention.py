"""
runtime/sparse_attention.py

Phase 6: Fused Sparse Attention Decode

Computes causal attention for a single decode step (q_len=1) directly over
Low-Rank compressed KV blocks, WITHOUT materialising full dense KV tensors.

Algorithm — Online FlashAttention-style softmax:
  For each block b:
    if dense:   standard q @ k^T, then q @ v update
    if sparse:  q @ (anchor + scale*U@V_K)^T without forming (anchor + scale*U@V_K)
  Running max/sum state (m_i, l_i, O_i) exactly matches the FlashAttention
  recurrence, so the output is numerically identical to the dense baseline
  (up to float16 rounding).
"""

import torch
import math
from typing import Optional


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA kv heads → query heads. Identical to HF repeat_kv."""
    if n_rep == 1:
        return hidden_states
    batch, kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, kv_heads * n_rep, slen, head_dim)


def fused_sparse_attention_decode(
    q:                    torch.Tensor,   # [1, H_q, 1, D]
    blocks:               list,           # list[KVBlock] — history only, NOT the active block
    active_k:             torch.Tensor,   # [1, H_kv, T, D]
    active_v:             torch.Tensor,   # [1, H_kv, T, D]
    num_key_value_groups: int,
    recon_cache:          Optional[object] = None,  # ReconstructionCache instance (optional)
) -> torch.Tensor:
    """
    Returns attention output [1, H_q, 1, D] without forming any dense KV seq.
    """
    bsz, H_q, q_len, D = q.shape
    assert bsz == 1 and q_len == 1, \
        "fused_sparse_attention_decode is decode-only (bsz=1, q_len=1)"

    inv_scale = 1.0 / math.sqrt(D)
    q_fp32 = (q * inv_scale).float()   # [1, H_q, 1, D]

    # FlashAttention running state (fp32 for numerical stability)
    m_i = torch.full((1, H_q, 1, 1), float("-inf"), device=q.device, dtype=torch.float32)
    l_i = torch.zeros ((1, H_q, 1, 1), device=q.device, dtype=torch.float32)
    O_i = torch.zeros ((1, H_q, 1, D), device=q.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Helper: update running state with a dense KV chunk [1, H_kv, S, D]
    # ------------------------------------------------------------------
    def _update_dense(k_kv: torch.Tensor, v_kv: torch.Tensor):
        nonlocal m_i, l_i, O_i
        if k_kv is None or k_kv.shape[2] == 0:
            return
        k = repeat_kv(k_kv, num_key_value_groups).float()  # [1, H_q, S, D]
        v = repeat_kv(v_kv, num_key_value_groups).float()

        S   = torch.matmul(q_fp32, k.transpose(-1, -2))  # [1, H_q, 1, S]
        m_b = S.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_i, m_b)

        a = torch.exp(m_i - m_new)
        P = torch.exp(S - m_new)                          # [1, H_q, 1, S]

        l_i[:] = a * l_i + P.sum(-1, keepdim=True)
        O_i[:] = a * O_i + torch.matmul(P, v)            # [1, H_q, 1, D]
        m_i[:] = m_new

    # ------------------------------------------------------------------
    # Iterate over history blocks
    # ------------------------------------------------------------------
    for block in blocks:
        if block.U is None:
            # ---- Dense block (recent uncompressed) ---------------------
            _update_dense(block.anchor_kv[:, 0].unsqueeze(2),
                          block.anchor_kv[:, 1].unsqueeze(2))
            if block.active_k is not None:
                _update_dense(block.active_k, block.active_v)

        else:
            # ---- Fused Sparse Block ------------------------------------
            # Shapes:
            #   anchor_kv: [1, 2, H_kv, D]
            #   U:         [S, R]
            #   V:         [R, F]   where F = 2 * H_kv * D
            #   scale:     scalar
            #
            # Dense reconstruction (never formed):
            #   token_i_K = anchor_K  +  scale * U[i] @ V_K
            #   token_i_V = anchor_V  +  scale * U[i] @ V_V
            #
            # Score for token i:
            #   s_i = q @ token_i_K^T
            #       = q @ anchor_K^T  +  scale * (q @ V_K^T) @ U[i]^T
            #       = s_anchor  +  scale * q_proj @ U[i]^T
            # -------------------------------------------------------

            U        = block.U.float()      # [S, R]
            V        = block.V.float()      # [R, F]
            S_blk    = U.shape[0]
            R        = U.shape[1]
            H_kv     = block.anchor_kv.shape[2]

            # Split V into K and V halves
            F        = 2 * H_kv * D
            V_split  = V.view(R, 2, H_kv, D)
            V_K      = V_split[:, 0]            # [R, H_kv, D]
            V_V      = V_split[:, 1]            # [R, H_kv, D]

            # Expand KV heads → query heads (GQA)
            G = num_key_value_groups
            if G > 1:
                V_K_exp = V_K.unsqueeze(2).expand(R, H_kv, G, D).reshape(R, H_q, D)
                V_V_exp = V_V.unsqueeze(2).expand(R, H_kv, G, D).reshape(R, H_q, D)
            else:
                V_K_exp = V_K.view(R, H_q, D)
                V_V_exp = V_V.view(R, H_q, D)

            # Anchor key/value, GQA-expanded
            a_k = block.anchor_kv[:, 0].unsqueeze(2)           # [1, H_kv, 1, D]
            a_v = block.anchor_kv[:, 1].unsqueeze(2)
            a_k_e = repeat_kv(a_k, G).float()                  # [1, H_q, 1, D]
            a_v_e = repeat_kv(a_v, G).float()

            # q_fp32: [1, H_q, 1, D]  →  q_hd: [H_q, D]  (single token, squeeze)
            q_hd = q_fp32[0, :, 0, :]                          # [H_q, D]

            # s_anchor = q @ anchor_K^T  →  [H_q]
            s_anchor = (q_hd * a_k_e[0, :, 0, :]).sum(-1)      # [H_q]

            # q_proj = q @ V_K^T  →  [H_q, R]
            # V_K_exp: [R, H_q, D]  →  [H_q, R, D]
            V_K_hq = V_K_exp.transpose(0, 1)                   # [H_q, R, D]
            q_proj = (q_hd.unsqueeze(1) * V_K_hq).sum(-1)      # [H_q, R]

            # delta_scores = scale * q_proj @ U^T  →  [H_q, S]
            delta_scores = block.scale * torch.matmul(q_proj, U.t())  # [H_q, S]

            # Full score for each of the S delta-tokens  →  [H_q, S]
            S_full = s_anchor.unsqueeze(-1) + delta_scores      # [H_q, S]

            # Also include the anchor token itself
            S_anchor_tok = s_anchor.unsqueeze(-1)               # [H_q, 1]
            S_all  = torch.cat([S_anchor_tok, S_full], dim=-1)  # [H_q, 1+S]
            S_all  = S_all.unsqueeze(0).unsqueeze(2)            # [1, H_q, 1, 1+S]

            # FlashAttention update for this block
            m_b    = S_all.max(-1, keepdim=True).values
            m_new  = torch.maximum(m_i, m_b)
            a_prev = torch.exp(m_i - m_new)
            P_all  = torch.exp(S_all - m_new)                  # [1, H_q, 1, 1+S]

            l_i[:] = a_prev * l_i + P_all.sum(-1, keepdim=True)

            # Value output for this block
            # O_block = P_anchor * a_v  +  sum_i{ P_i * (a_v + scale * U[i] @ V_V) }
            # = (P_anchor + sum_i P_i) * a_v  +  scale * (P_delta @ U) @ V_V
            P_anchor_tok = P_all[:, :, :, 0:1]                 # [1, H_q, 1, 1]
            P_delta      = P_all[:, :, :, 1:]                  # [1, H_q, 1, S]

            # Anchor contribution
            O_anchor = (P_anchor_tok + P_delta.sum(-1, keepdim=True)) * a_v_e  # [1,H_q,1,D]

            # Delta contribution via low-rank: (P_delta @ U) @ V_V
            # P_delta: [H_q, S]  →  P_U: [H_q, R]
            P_delta_hq = P_delta[0, :, 0, :]                   # [H_q, S]
            P_U = torch.matmul(P_delta_hq, U)                  # [H_q, R]

            # V_V_exp: [R, H_q, D]  →  [H_q, R, D]
            V_V_hq = V_V_exp.transpose(0, 1)                   # [H_q, R, D]
            O_delta = block.scale * torch.bmm(
                P_U.unsqueeze(1),                              # [H_q, 1, R]
                V_V_hq,                                        # [H_q, R, D]
            ).squeeze(1)                                       # [H_q, D]
            O_delta = O_delta.unsqueeze(0).unsqueeze(2)        # [1, H_q, 1, D]

            O_block = O_anchor + O_delta
            O_i[:] = a_prev * O_i + O_block
            m_i[:] = m_new

    # ---- Active dense window (last block's recent tokens) ---------------
    _update_dense(active_k, active_v)

    # Normalise and return
    O_out = O_i / l_i
    return O_out.to(q.dtype)
