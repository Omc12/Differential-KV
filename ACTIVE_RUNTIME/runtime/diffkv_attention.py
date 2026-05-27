import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threading
from typing import Optional, Tuple
from native_core.sparse_decode.triton_diffkv import TritonDiffKV
from native_core.sparse_decode.triton_sparse_attn import native_triton_sparse_attn_decode, HAS_TRITON

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
        return hidden_states.contiguous()
    bs, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bs, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bs, num_key_value_heads * n_rep, slen, head_dim).contiguous()

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
                        curr_k = key_states[b_idx:b_idx+1]
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

                        block_indices, dense_blocks = kv_manager.get_cached_decode_blocks(
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

                        attn_out_b = native_triton_sparse_attn_decode(
                            q=query_states[b_idx:b_idx+1],
                            block_indices=block_indices,
                            pool=pool,
                            dense_blocks=[],          # dense window handled by active_k/v below
                            active_k=dense_k_assembled,
                            active_v=dense_v_assembled,
                            num_key_value_groups=num_key_value_groups,
                            R=kv_manager.rank,
                            S_MAX=session_mbs
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
                    # Check if ANY active session in this prefill batch already has resident history
                    # (i.e. this is an incremental prefill).
                    has_history = False
                    for sid in session_ids:
                        if sid != "dummy_session":
                            blocks = kv_manager.get_streaming_blocks(sid, captured_layer_idx)
                            if blocks and len(blocks) > 0:
                                has_history = True
                                break

                    # Step 1: Store K/V in streaming blocks for future decode
                    for b_idx, sid in enumerate(session_ids):
                        curr_k = key_states[b_idx:b_idx + 1]
                        curr_v = value_states[b_idx:b_idx + 1]
                        kv_manager.ingest_streaming(sid, captured_layer_idx, curr_k, curr_v)

                    if has_history:
                        # Incremental prefill! We must attend to the full historical KV cache.
                        # Reconstruct the entire sequence (history + new) for each batch element.
                        batch_k = []
                        batch_v = []
                        seq_lens = []
                        for b_idx, sid in enumerate(session_ids):
                            if sid == "dummy_session":
                                k_rep_b = repeat_kv(key_states[b_idx:b_idx+1], num_key_value_groups)
                                v_rep_b = repeat_kv(value_states[b_idx:b_idx+1], num_key_value_groups)
                                batch_k.append(k_rep_b)
                                batch_v.append(v_rep_b)
                                seq_lens.append(q_len)
                                continue
                            full_k, full_v = kv_manager.assemble_decode_kv(
                                sid, captured_layer_idx, query_states.dtype
                            )
                            if full_k is not None:
                                k_rep_b = repeat_kv(full_k, num_key_value_groups)
                                v_rep_b = repeat_kv(full_v, num_key_value_groups)
                                batch_k.append(k_rep_b)
                                batch_v.append(v_rep_b)
                                seq_lens.append(full_k.shape[2])
                            else:
                                k_rep_b = repeat_kv(key_states[b_idx:b_idx+1], num_key_value_groups)
                                v_rep_b = repeat_kv(value_states[b_idx:b_idx+1], num_key_value_groups)
                                batch_k.append(k_rep_b)
                                batch_v.append(v_rep_b)
                                seq_lens.append(q_len)
                        
                        S_max = max(seq_lens)
                        
                        # Pad keys/values along the sequence dimension if lengths differ
                        padded_k = []
                        padded_v = []
                        for b_idx, (k_b, v_b) in enumerate(zip(batch_k, batch_v)):
                            S_b = seq_lens[b_idx]
                            if S_b < S_max:
                                pad_len = S_max - S_b
                                k_pad = torch.zeros((1, num_heads, pad_len, head_dim), dtype=query_states.dtype, device=query_states.device)
                                v_pad = torch.zeros((1, num_heads, pad_len, head_dim), dtype=query_states.dtype, device=query_states.device)
                                padded_k.append(torch.cat([k_b, k_pad], dim=2))
                                padded_v.append(torch.cat([v_b, v_pad], dim=2))
                            else:
                                padded_k.append(k_b)
                                padded_v.append(v_b)
                                
                        k_rep = torch.cat(padded_k, dim=0)
                        v_rep = torch.cat(padded_v, dim=0)
                        
                        # Build correct custom causal attention mask for unequal sequence lengths
                        # attn_mask shape: [bsz, 1, q_len, S_max]
                        attn_mask = torch.zeros((bsz, 1, q_len, S_max), dtype=torch.bool, device=query_states.device)
                        for b_idx in range(bsz):
                            S_b = seq_lens[b_idx]
                            K_b = S_b - q_len
                            for i in range(q_len):
                                attn_mask[b_idx, 0, i, :K_b + i + 1] = True
                                
                        attn_output = F.scaled_dot_product_attention(
                            query_states.contiguous(), k_rep.contiguous(), v_rep.contiguous(),
                            attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                        )
                    else:
                        # Fresh prefill — just repeat the current chunk's K/V states
                        k_rep   = repeat_kv(key_states,   num_key_value_groups)
                        v_rep   = repeat_kv(value_states, num_key_value_groups)
                        
                        attn_output = F.scaled_dot_product_attention(
                            query_states.contiguous(), k_rep.contiguous(), v_rep.contiguous(),
                            attn_mask=None, dropout_p=0.0, is_causal=True
                        )
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
