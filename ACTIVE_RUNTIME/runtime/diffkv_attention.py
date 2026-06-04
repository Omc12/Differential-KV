import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threading
from typing import Optional, Tuple
from native_core.sparse_decode.triton_diffkv import TritonDiffKV
from native_core.sparse_decode.triton_sparse_attn import (
    native_triton_sparse_attn_decode,
    _prefill_fused_history_attend,
    HAS_TRITON,
)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Universal repeat_kv implementation to support GQA.
    """
    if n_rep == 1:
        return hidden_states
    bs, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bs, num_key_value_heads, n_rep, slen, head_dim)
    out = hidden_states.reshape(bs, num_key_value_heads * n_rep, slen, head_dim)
    if hidden_states.device.type == "mps":
        # MPS scaled_dot_product_attention supports non-contiguous views natively
        return out
    return out.contiguous()

# ---------------------------------------------------------------------------
# PHASE 6: Fused Sparse Attention Integration
#
# Execution paths:
#   PREFILL (q_len > 1):  Dense path — required for causal masking over new tokens.
#   DECODE  (q_len == 1): FUSED SPARSE path — directly reads U/V/anchor without
#                          materializing a dense KV sequence via torch.cat().
#
# Performance notes (Phase 29):
#   - All module imports hoisted to top level — eliminates 84+ sys.modules lookups/token
#   - Ring buffer decode ingest — eliminates torch.cat allocations in hot path
#   - Metadata on CPU — eliminates 112+ CUDA sync points per token
#   - micro_block_size cached — eliminates O(N·L) block scan per token
# ---------------------------------------------------------------------------

def apply_diffkv_attention_patch(model, kv_manager):
    """
    Monkey-patches the HF model's attention layers to route KV operations
    through our KVRuntimeManager.

    Phase 6 change: decode step no longer calls kv_manager.get_kv() (which
    issues aten::cat over reconstructed blocks). Instead it calls
    kv_manager.get_raw_blocks() and passes them directly to
    fused_sparse_attention_decode().
    """
    num_heads             = model.config.num_attention_heads
    num_key_value_heads   = getattr(model.config, "num_key_value_heads", num_heads)
    hidden_size           = model.config.hidden_size
    head_dim              = hidden_size // num_heads
    num_key_value_groups  = num_heads // num_key_value_heads

    for i, layer in enumerate(model.model.layers):

        def make_diffkv_forward(captured_layer_idx):
            def diffkv_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()

                # Helpers for Decomposed Prefill Attention (Part 1)
                def _flash_local_attention(q, k, v):
                    # Local causal self-attention over the new chunk
                    k_rep = repeat_kv(k, num_key_value_groups)
                    v_rep = repeat_kv(v, num_key_value_groups)
                    
                    out = F.scaled_dot_product_attention(
                        q.contiguous(), k_rep.contiguous(), v_rep.contiguous(),
                        attn_mask=None, dropout_p=0.0, is_causal=True
                    )
                    
                    # Compute Log-Sum-Exp manually for local causal attention
                    scale = 1.0 / math.sqrt(head_dim)
                    scores = torch.matmul(q * scale, k_rep.transpose(-2, -1))
                    
                    # Mask future tokens — specify q.dtype to avoid float32 upcast
                    mask = torch.triu(torch.full((q_len, q_len), float('-inf'), device=q.device, dtype=q.dtype), diagonal=1)
                    scores = scores + mask
                    lse = torch.logsumexp(scores, dim=-1)
                    return out, lse

                def _project_then_attend_history(q, comp_blocks, pool, sid, cos_all=None, sin_all=None):
                    # O5b: Full history attention path routed through the JIT-fused kernel.
                    # All 8 inner ops (K-recon, cat, RoPE, score, mask, logsumexp, softmax,
                    # value-reduction) execute in a single TorchScript compilation unit.
                    _, H, q_len_inner, head_dim = q.shape
                    pool_indices = [b.pool_idx for b in comp_blocks]
                    pool_indices_t = torch.tensor(pool_indices, device=q.device, dtype=torch.long)

                    N_blocks  = len(comp_blocks)
                    max_seq_len = pool.U.shape[1]

                    # Gather block data from the pool
                    U_stack    = pool.U[pool_indices_t].to(q.dtype) * pool.U_scale[pool_indices_t].view(-1, 1, 1)
                    V_K_stack  = pool.V_K[pool_indices_t]        # [N, R, num_kv_heads, D]
                    V_V_stack  = pool.V_V[pool_indices_t]        # [N, R, num_kv_heads, D]
                    anc_K      = pool.anchors_K[pool_indices_t]  # [N, num_kv_heads, D]
                    anc_V      = pool.anchors_V[pool_indices_t]  # [N, num_kv_heads, D]
                    scales_1d  = pool.scales[pool_indices_t]     # [N]
                    seq_lens_t = pool.seq_lens[pool_indices_t]   # [N] int32

                    # GQA repeat-expansion (zero-copy expand → contiguous only if needed)
                    v_k_rep   = repeat_kv(V_K_stack.permute(0, 2, 1, 3), num_key_value_groups)   # [N, H, R, D]
                    v_v_rep   = repeat_kv(V_V_stack.permute(0, 2, 1, 3), num_key_value_groups)   # [N, H, R, D]
                    anc_k_rep = repeat_kv(anc_K.unsqueeze(2), num_key_value_groups).squeeze(2)   # [N, H, D]
                    anc_v_rep = repeat_kv(anc_V.unsqueeze(2), num_key_value_groups).squeeze(2)   # [N, H, D]

                    # Cache-aware RoPE slicing to eliminate redundant GPU gathers per layer
                    anchors_tuple = tuple(b.anchor_idx for b in comp_blocks)
                    cache_cos_key = (sid, "prefill_cos_sliced", anchors_tuple)
                    cache_sin_key = (sid, "prefill_sin_sliced", anchors_tuple)
                    cos_sliced = kv_manager.decode_workspace.get(cache_cos_key)
                    sin_sliced = kv_manager.decode_workspace.get(cache_sin_key)

                    if cos_sliced is None or sin_sliced is None:
                        block_anchors   = torch.tensor(anchors_tuple, device=q.device, dtype=torch.long)
                        positions       = block_anchors.view(N_blocks, 1) + torch.arange(1 + max_seq_len, device=q.device).view(1, 1 + max_seq_len)
                        positions_flat  = positions.view(-1)
                        cos_ref = cos_all if cos_all is not None else cos
                        sin_ref = sin_all if sin_all is not None else sin
                        cos_sliced = cos_ref[0, positions_flat].view(N_blocks, 1 + max_seq_len, 1, head_dim)
                        sin_sliced = sin_ref[0, positions_flat].view(N_blocks, 1 + max_seq_len, 1, head_dim)
                        kv_manager.decode_workspace[cache_cos_key] = cos_sliced
                        kv_manager.decode_workspace[cache_sin_key] = sin_sliced

                        # Evict stale prefill sliced RoPE cache entries
                        stale_keys = [
                            k for k in kv_manager.decode_workspace.keys()
                            if isinstance(k, tuple) and len(k) == 3 and k[0] == sid
                            and k[1] in ("prefill_cos_sliced", "prefill_sin_sliced")
                            and k[2] != anchors_tuple
                        ]
                        for k in stale_keys:
                            del kv_manager.decode_workspace[k]

                    inv_scale_val = 1.0 / math.sqrt(head_dim)

                    # ── O5b: Single JIT dispatch covering all inner math ─────────────
                    # result[0] = out_hist  [1, H, q_len, D]
                    # result[1, 0, :, :, 0] = lse_hist [H, q_len]  (last dim replicated)
                    result = _prefill_fused_history_attend(
                        U          = U_stack,
                        V_K        = v_k_rep.permute(0, 2, 1, 3),   # [N, R, H, D]
                        V_V        = v_v_rep.permute(0, 2, 1, 3),   # [N, R, H, D]
                        anchors_K  = anc_k_rep,
                        anchors_V  = anc_v_rep,
                        scales     = scales_1d,
                        cos_sliced = cos_sliced,
                        sin_sliced = sin_sliced,
                        q          = q,
                        seq_lens   = seq_lens_t,
                        inv_scale  = inv_scale_val,
                    )
                    out_hist  = result[0]                     # [1, H, q_len, D]
                    lse_hist  = result[1, 0, :, :, 0]        # [H, q_len]
                    lse_hist  = lse_hist.unsqueeze(0)        # [1, H, q_len]  — matches _combine_outputs API
                    return out_hist, lse_hist

                def _combine_outputs(out_lse_list):
                    # Filter out empty paths
                    valid_list = [x for x in out_lse_list if x[0] is not None and x[1] is not None]
                    if not valid_list:
                        return None
                    if len(valid_list) == 1:
                        return valid_list[0][0]
                    
                    # Log-sum-exp stable online softmax combination
                    lses = torch.stack([x[1] for x in valid_list], dim=0)  # [M, 1, H, q_len]
                    lse_max, _ = torch.max(lses, dim=0)                  # [1, H, q_len]
                    
                    weights_list = []
                    denom = torch.zeros_like(lse_max)
                    for out, lse in valid_list:
                        w = torch.exp(lse - lse_max)
                        weights_list.append(w)
                        denom = denom + w
                    
                    # Prevent divide-by-zero
                    denom = torch.clamp(denom, min=1e-9)
                    
                    out_combined = torch.zeros_like(valid_list[0][0])
                    for idx, (out, lse) in enumerate(valid_list):
                        w = weights_list[idx] / denom
                        out_combined = out_combined + out * w.unsqueeze(-1)
                    
                    # Ensure dtype is cast back to avoid precision mismatch during linear projection
                    return out_combined.to(valid_list[0][0].dtype)

                # --- Projection ---
                query_states = self.q_proj(hidden_states)
                key_states   = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                key_states   = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

                # --- RoPE ---
                if position_embeddings is None:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = position_embeddings
                unrot_key_states = key_states.clone()
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                session_ids = getattr(model, "_diffkv_session_ids", ["default"] * bsz)

                # ==================================================================
                # PHASE 6 BRANCHING
                # ==================================================================
                is_decode = (use_cache and q_len == 1)

                if is_decode:
                    # ----------------------------------------------------------
                    # DECODE PATH — sparse execution via Triton or PyTorch fallback
                    # ----------------------------------------------------------

                    # 1. Ingest new tokens for all active batch elements
                    for b_idx in range(bsz):
                        sid = session_ids[b_idx]
                        if sid == "dummy_session":
                            continue
                        curr_k = unrot_key_states[b_idx:b_idx+1]
                        curr_v = value_states[b_idx:b_idx+1]
                        kv_manager.ingest_streaming(sid, captured_layer_idx, curr_k, curr_v)

                    # 2. Attention decode dispatch
                    # Always route through the unified native_triton_sparse_attn_decode path.
                    # This dispatches to either the GPU Triton kernel or the high-performance
                    # Project-Then-Attend PyTorch fallback!
                    attn_outputs = []
                    for b_idx in range(bsz):
                        sid = session_ids[b_idx]
                        if sid == "dummy_session":
                            attn_outputs.append(
                                torch.zeros(
                                    (1, num_heads, 1, head_dim),
                                    device=query_states.device, dtype=query_states.dtype
                                )
                            )
                            continue

                        block_indices, dense_blocks, anchor_indices, max_anchor_idx, max_valid_len = kv_manager.get_cached_decode_blocks(
                            sid, captured_layer_idx, query_states.device
                        )
                        pool = getattr(kv_manager, 'native_pool', None)
                        session_mbs = kv_manager.get_session_micro_block_size(sid)

                        # Assemble ONLY the dense window (non-compressed blocks) into a
                        # persistent workspace tensor. Uses lightweight slice copies, zero GEMM.
                        # Compressed blocks are handled by block_indices in the sparse kernel.
                        dense_k_assembled, dense_v_assembled = None, None
                        if dense_blocks:
                            dense_k_assembled, dense_v_assembled = kv_manager.assemble_dense_window_kv(
                                sid, captured_layer_idx, dense_blocks, query_states.dtype
                            )

                        total_seq_len = kv_manager.get_session_sequence_length(sid)
                        max_pos = total_seq_len
                        if max_anchor_idx is not None:
                            max_pos = max(max_pos, max_anchor_idx + session_mbs)
                        
                        rope_cos_key = (sid, "rope_cos")
                        rope_sin_key = (sid, "rope_sin")
                        cached_cos = kv_manager.decode_workspace.get(rope_cos_key)
                        cached_sin = kv_manager.decode_workspace.get(rope_sin_key)

                        if cached_cos is not None and cached_cos.shape[1] >= max_pos:
                            cos_all = cached_cos[:, :max_pos]
                            sin_all = cached_sin[:, :max_pos]
                        else:
                            hist_pos = torch.arange(max_pos, device=query_states.device, dtype=torch.long).unsqueeze(0)
                            cos_all, sin_all = model.model.rotary_emb(value_states[b_idx:b_idx+1], hist_pos)
                            kv_manager.decode_workspace[rope_cos_key] = cos_all
                            kv_manager.decode_workspace[rope_sin_key] = sin_all

                        cos_sliced_arg = None
                        sin_sliced_arg = None
                        if anchor_indices is not None and anchor_indices.numel() > 0 and pool is not None:
                            anchors_tuple = tuple(anchor_indices.cpu().tolist())
                            rope_cos_sliced_key = (sid, "decode_cos_sliced", anchors_tuple)
                            rope_sin_sliced_key = (sid, "decode_sin_sliced", anchors_tuple)
                            cos_sliced_cached = kv_manager.decode_workspace.get(rope_cos_sliced_key)
                            sin_sliced_cached = kv_manager.decode_workspace.get(rope_sin_sliced_key)

                            if cos_sliced_cached is None or sin_sliced_cached is None:
                                N_blocks = anchor_indices.shape[0]
                                max_seq_len = pool.U.shape[1]
                                positions = anchor_indices.view(N_blocks, 1) + torch.arange(1 + max_seq_len, device=query_states.device).view(1, 1 + max_seq_len)
                                positions_flat = positions.view(-1)
                                
                                cos_flat = cos_all.squeeze(0) if cos_all.dim() == 3 else cos_all
                                sin_flat = sin_all.squeeze(0) if sin_all.dim() == 3 else sin_all
                                
                                cos_sliced_cached = cos_flat[positions_flat].view(N_blocks, 1 + max_seq_len, 1, head_dim)
                                sin_sliced_cached = sin_flat[positions_flat].view(N_blocks, 1 + max_seq_len, 1, head_dim)
                                
                                kv_manager.decode_workspace[rope_cos_sliced_key] = cos_sliced_cached
                                kv_manager.decode_workspace[rope_sin_sliced_key] = sin_sliced_cached
                                
                                stale_keys = [
                                    k for k in kv_manager.decode_workspace.keys()
                                    if isinstance(k, tuple) and len(k) == 3 and k[0] == sid
                                    and k[1] in ("decode_cos_sliced", "decode_sin_sliced")
                                    and k[2] != anchors_tuple
                                ]
                                for k in stale_keys:
                                    del kv_manager.decode_workspace[k]

                            cos_sliced_arg = cos_sliced_cached
                            sin_sliced_arg = sin_sliced_cached

                        attn_out_b = native_triton_sparse_attn_decode(
                            q=query_states[b_idx:b_idx+1],
                            block_indices=block_indices,
                            pool=pool,
                            dense_blocks=dense_blocks,
                            active_k=dense_k_assembled,
                            active_v=dense_v_assembled,
                            num_key_value_groups=num_key_value_groups,
                            R=kv_manager.rank,
                            S_MAX=session_mbs,
                            anchor_indices=anchor_indices,
                            cos=cos_all,
                            sin=sin_all,
                            total_seq_len=total_seq_len,
                            max_valid_len=max_valid_len,
                            cos_sliced=cos_sliced_arg,
                            sin_sliced=sin_sliced_arg,
                        )
                        attn_outputs.append(attn_out_b)

                    attn_output = torch.cat(attn_outputs, dim=0)

                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
                    attn_output = self.o_proj(attn_output)

                    outputs = (attn_output,)
                    if output_attentions:
                        outputs += (None,)
                    if use_cache:
                        outputs += (None,)
                    return outputs

                # ==============================================================
                # PREFILL / MULTI-QUERY PATH
                # ==============================================================
                if use_cache:
                    # Check if ANY active session already has history blocks in the pool.
                    # This includes both uncompressed (accumulating) and compressed blocks,
                    # ensuring that short contexts (<2048 tokens) are correctly routed to the
                    # incremental prefill path instead of losing context.
                    has_compressed_history = False
                    for sid in session_ids:
                        if sid != "dummy_session":
                            blocks = kv_manager.get_streaming_blocks(sid, captured_layer_idx)
                            if blocks:
                                has_compressed_history = True
                                break

                    if has_compressed_history:
                        # ── INCREMENTAL PREFILL (2nd+ turn) ─────────────────────────────────
                        # Compressed history already exists in the pool from a prior turn.
                        # Use the decomposed A+B attention: Path A = local causal FA over the
                        # new chunk; Path B = Project-Then-Attend over compressed history.
                        # Capture KV for post-forward compression of the NEW chunk.
                        seq_lens = []
                        for b_idx, sid in enumerate(session_ids):
                            if sid == "dummy_session":
                                seq_lens.append(q_len)
                            else:
                                seq_lens.append(kv_manager.get_session_sequence_length(sid))

                        attn_outputs = []
                        for b_idx, sid in enumerate(session_ids):
                            if sid == "dummy_session":
                                attn_outputs.append(
                                    torch.zeros((1, num_heads, q_len, head_dim),
                                                device=query_states.device, dtype=query_states.dtype)
                                )
                                continue

                            # ── Path A: Causal Local Self-Attention over new chunk ──
                            curr_q = query_states[b_idx:b_idx+1]
                            curr_k = key_states[b_idx:b_idx+1]
                            curr_v = value_states[b_idx:b_idx+1]

                            out_local, lse_local = _flash_local_attention(curr_q, curr_k, curr_v)

                            # ── Path B: History Cross-Attention ──
                            # The history sequence length is exactly the sequence length of the session
                            # before the new chunk is appended. Subtracting q_len was a math bug that caused
                            # history to be partially or completely ignored (when q_len >= seq_lens[b_idx]
                            # during large prefills or long chat messages).
                            K_b = seq_lens[b_idx]
                            out_hist_dense, lse_hist_dense = None, None
                            out_hist_comp, lse_hist_comp   = None, None

                            if K_b > 0:
                                prev_len = 0

                                blocks = kv_manager.get_streaming_blocks(sid, captured_layer_idx)
                                history_blocks = [b for b in blocks if b.anchor_idx < K_b]

                                comp_blocks = []
                                dense_k = []
                                dense_v = []
                                dense_positions_list = []

                                for b in history_blocks:
                                    if getattr(b, "state", None) == "COMPRESSED" \
                                            and b.U is not None and b.V is not None:
                                        comp_blocks.append(b)
                                    else:
                                        ak = b.anchor_kv[0, 0]
                                        av = b.anchor_kv[0, 1]
                                        if b.active_k is not None:
                                            k_blk = b.active_k[0]
                                            v_blk = b.active_v[0]
                                            act_len = k_blk.shape[1]
                                        elif getattr(b, "active_k_cpu", None) is not None:
                                            k_blk = b.active_k_cpu[0].to(query_states.device, non_blocking=True)
                                            v_blk = b.active_v_cpu[0].to(query_states.device, non_blocking=True)
                                            act_len = k_blk.shape[1]
                                        else:
                                            k_blk, v_blk = None, None
                                            act_len = 0
                                        ak_u = ak.unsqueeze(1)
                                        av_u = av.unsqueeze(1)
                                        if k_blk is not None:
                                            dense_k.append(torch.cat([ak_u, k_blk], dim=1))
                                            dense_v.append(torch.cat([av_u, v_blk], dim=1))
                                        else:
                                            dense_k.append(ak_u)
                                            dense_v.append(av_u)
                                        dense_positions_list.extend(range(b.anchor_idx, b.anchor_idx + 1 + act_len))

                                k_dense, v_dense = None, None
                                if dense_k:
                                    k_dense = torch.cat(dense_k, dim=1).unsqueeze(0)
                                    v_dense = torch.cat(dense_v, dim=1).unsqueeze(0)

                                # Safe bounds calculation for rotary embeddings in prefill path
                                max_pos = K_b + prev_len + q_len
                                if comp_blocks:
                                    mbs = getattr(comp_blocks[0], "micro_block_size", kv_manager.get_session_micro_block_size(sid))
                                    max_pos = max(max_pos, max(b.anchor_idx for b in comp_blocks) + mbs)

                                hist_pos = torch.arange(max_pos, device=query_states.device, dtype=torch.long).unsqueeze(0)
                                cos_all, sin_all = model.model.rotary_emb(value_states[b_idx:b_idx+1], hist_pos)

                                # Phase 29+ Fix: Also attend to the previous chunks of the CURRENT prefill turn!
                                # This is critical for progressive prompt chunking correctness (e.g. long prompts / research papers).
                                if prev_len > 0:
                                    prev_k, prev_v = session_cap[captured_layer_idx]
                                    if k_dense is not None:
                                        k_dense = torch.cat([k_dense, prev_k], dim=2)
                                        v_dense = torch.cat([v_dense, prev_v], dim=2)
                                        dense_positions_list.extend(range(K_b, K_b + prev_len))
                                    else:
                                        k_dense = prev_k
                                        v_dense = prev_v
                                        dense_positions_list = list(range(K_b, K_b + prev_len))

                                if k_dense is not None:
                                    dense_positions_tensor = torch.tensor(dense_positions_list, dtype=torch.long, device=query_states.device)
                                    cos_dense = cos_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1) # [1, 1, len, D]
                                    sin_dense = sin_all[0, dense_positions_tensor].unsqueeze(0).unsqueeze(1) # [1, 1, len, D]
                                    k_dense_rot = (k_dense * cos_dense) + (rotate_half(k_dense) * sin_dense)
                                    
                                    k_dense_rep = repeat_kv(k_dense_rot, num_key_value_groups)
                                    v_dense_rep = repeat_kv(v_dense, num_key_value_groups)
                                    out_hist_dense = F.scaled_dot_product_attention(
                                        curr_q.contiguous(), k_dense_rep.contiguous(), v_dense_rep.contiguous(),
                                        attn_mask=None, dropout_p=0.0, is_causal=False
                                    )
                                    _scale = 1.0 / math.sqrt(head_dim)
                                    scores_dense = torch.matmul(curr_q * _scale, k_dense_rep.transpose(-2, -1))
                                    lse_hist_dense = torch.logsumexp(scores_dense, dim=-1)

                                if comp_blocks and getattr(kv_manager, "native_pool", None) is not None:
                                    out_hist_comp, lse_hist_comp = _project_then_attend_history(
                                        curr_q, comp_blocks, kv_manager.native_pool, sid, cos_all, sin_all
                                    )

                            out_b = _combine_outputs([
                                (out_local,     lse_local),
                                (out_hist_dense, lse_hist_dense),
                                (out_hist_comp,  lse_hist_comp),
                            ])
                            attn_outputs.append(out_b)

                        attn_output = torch.cat(attn_outputs, dim=0)

                        # Capture new-chunk KV for post-forward compression
                        for b_idx, sid in enumerate(session_ids):
                            if sid != "dummy_session":
                                kv_manager.capture_prefill_kv(
                                    sid, captured_layer_idx,
                                    unrot_key_states[b_idx:b_idx+1].detach(),
                                    value_states[b_idx:b_idx+1].detach(),
                                )

                    else:
                        # ── FRESH PREFILL (1st turn / new session) ───────────────────────────
                        # Attend to previous prefill chunks of the same prompt if they exist.
                        # This is critical for progressive prompt chunking correctness!
                        attn_outputs = []
                        for b_idx in range(bsz):
                            sid = session_ids[b_idx]
                            curr_q = query_states[b_idx:b_idx+1]
                            curr_k = key_states[b_idx:b_idx+1]
                            curr_v = value_states[b_idx:b_idx+1]
                            
                            full_k = curr_k
                            full_v = curr_v
                            is_causal_flag = True
                            attn_mask_flag = None
                                
                            k_rep = repeat_kv(full_k, num_key_value_groups)
                            v_rep = repeat_kv(full_v, num_key_value_groups)
                            
                            out_b = F.scaled_dot_product_attention(
                                curr_q.contiguous(), k_rep.contiguous(), v_rep.contiguous(),
                                attn_mask=attn_mask_flag, dropout_p=0.0, is_causal=is_causal_flag
                            )
                            attn_outputs.append(out_b)
                            
                            # Capture this chunk's KV (without prev concat)
                            if sid != "dummy_session":
                                kv_manager.capture_prefill_kv(
                                    sid, captured_layer_idx,
                                    unrot_key_states[b_idx:b_idx+1].detach(),
                                    curr_v.detach(),
                                )
                                
                        attn_output = torch.cat(attn_outputs, dim=0)

                    attn_weights = None

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, hidden_size)
                attn_output = self.o_proj(attn_output)

                outputs = (attn_output,)
                if output_attentions:
                    outputs += (attn_weights,)
                if use_cache:
                    outputs += (None,)
                return outputs

            return diffkv_forward

        layer.self_attn.forward = make_diffkv_forward(i).__get__(layer.self_attn, layer.self_attn.__class__)

    # Phase 25: Patch LM Head to only compute logits for the last token
    if hasattr(model, "lm_head"):
        original_lm_head_forward = model.lm_head.forward
        def last_token_lm_head_forward(hidden_states):
            if hidden_states.shape[1] > 1:
                return original_lm_head_forward(hidden_states[:, -1:, :])
            return original_lm_head_forward(hidden_states)
        model.lm_head.forward = last_token_lm_head_forward

    print("Differential KV Attention Interception Applied. [Phase 29: Zero-overhead decode active]")
