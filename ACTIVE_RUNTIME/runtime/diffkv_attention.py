import torch
import torch.nn as nn
import math
import threading
from typing import Optional, Tuple
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb
from native_core.sparse_decode.triton_diffkv import TritonDiffKV

# ---------------------------------------------------------------------------
# PHASE 6: Fused Sparse Attention Integration
#
# Execution paths:
#   PREFILL (q_len > 1):  Dense path — required for causal masking over new tokens.
#   DECODE  (q_len == 1): FUSED SPARSE path — directly reads U/V/anchor without
#                          materializing a dense KV sequence via torch.cat().
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
                    # DECODE PATH — 100% GPU serving path via Triton Fused Attention Decode
                    # ----------------------------------------------------------
                    import transformers.models.qwen2.modeling_qwen2 as _mq
                    import torch.nn.functional as _F
                    from native_core.sparse_decode.triton_sparse_attn import native_triton_sparse_attn_decode

                    # 1. Ingest new tokens for all batch elements
                    for b_idx in range(bsz):
                        sid = session_ids[b_idx]
                        curr_k = key_states[b_idx:b_idx+1]
                        curr_v = value_states[b_idx:b_idx+1]
                        kv_manager.ingest_streaming(sid, captured_layer_idx, curr_k, curr_v)

                    # 2. Triton Fused Attention Decode Kernel Dispatch per batch element
                    attn_outputs = []
                    for b_idx in range(bsz):
                        sid = session_ids[b_idx]
                        blocks = kv_manager.get_streaming_blocks(sid, captured_layer_idx)
                        
                        compressed_pool_indices = []
                        dense_blocks = []
                        
                        for blk in blocks:
                            pool_idx = getattr(blk, 'pool_idx', None)
                            if blk.U is not None and blk.V is not None and pool_idx is not None:
                                compressed_pool_indices.append(pool_idx)
                            else:
                                dense_blocks.append(blk)
                        
                        pool = getattr(kv_manager, 'native_pool', None)
                        
                        session_mbs = kv_manager.get_session_micro_block_size(sid)
                        
                        if pool is not None and compressed_pool_indices:
                            block_indices = torch.tensor(
                                compressed_pool_indices, 
                                device=query_states.device, 
                                dtype=torch.int32
                            )
                            attn_out_b = native_triton_sparse_attn_decode(
                                q=query_states[b_idx:b_idx+1],
                                block_indices=block_indices,
                                pool=pool,
                                dense_blocks=dense_blocks,
                                active_k=None,
                                active_v=None,
                                num_key_value_groups=num_key_value_groups,
                                R=kv_manager.rank,
                                S_MAX=session_mbs
                            )
                        else:
                            attn_out_b = native_triton_sparse_attn_decode(
                                q=query_states[b_idx:b_idx+1],
                                block_indices=None,
                                pool=pool,
                                dense_blocks=dense_blocks,
                                active_k=None,
                                active_v=None,
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
                    # ── Step 1: Store K/V in streaming blocks for future decode ──
                    for b_idx, sid in enumerate(session_ids):
                        curr_k = key_states[b_idx:b_idx + 1]   # [1, kv_heads, q_len, head_dim]
                        curr_v = value_states[b_idx:b_idx + 1]
                        kv_manager.ingest_streaming(sid, captured_layer_idx, curr_k, curr_v)

                    # ── Step 2: Compute attention using the raw K/V from this
                    #    forward pass (NOT from the blocks). This is always correct
                    #    because key_states/value_states are the ground truth for
                    #    this prompt. We only read from blocks during decode, when
                    #    we need to attend back to compressed history.
                    import transformers.models.qwen2.modeling_qwen2 as mq
                    import torch.nn.functional as F
                    key_rep   = mq.repeat_kv(key_states,   num_key_value_groups)
                    value_rep = mq.repeat_kv(value_states, num_key_value_groups)
                    attn_output = F.scaled_dot_product_attention(
                        query_states, key_rep, value_rep,
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

        layer.self_attn.forward = make_diffkv_forward(i).__get__(layer.self_attn, Qwen2Attention)

    # Phase 25: Patch LM Head to only compute logits for the last token
    if hasattr(model, "lm_head"):
        original_lm_head_forward = model.lm_head.forward
        def last_token_lm_head_forward(hidden_states):
            # hidden_states is [B, S, D]
            if hidden_states.shape[1] > 1:
                # Only project the last token to save massive vocab memory (e.g. 7.6GB -> 300KB)
                return original_lm_head_forward(hidden_states[:, -1:, :])
            return original_lm_head_forward(hidden_states)
        model.lm_head.forward = last_token_lm_head_forward

    print("Differential KV Attention Interception Applied. [Phase 6: Fused Sparse Decode Active]")

