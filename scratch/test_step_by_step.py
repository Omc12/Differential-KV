import os
import sys
sys.modules["diffkv_core"] = None
import torch
import math
from transformers import AutoTokenizer

sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME")
from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper

def main():
    # Force DiffKV to engage on short prompt and run PyTorch path
    os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1" # Set to 1 to match C++ approximate path
    
    import native_core.sparse_decode.triton_fused_decode as triton_fused_decode
    orig_fused_decode_mps = triton_fused_decode.fused_decode_mps

    def patched_fused_decode_mps(Q, pool, block_indices, blk_sizes, num_key_value_groups, anchor_indices=None, cos=None, sin=None):
        out, lse = orig_fused_decode_mps(Q, pool, block_indices, blk_sizes, num_key_value_groups, anchor_indices, cos, sin)
        if getattr(patched_fused_decode_mps, "has_printed", False) == False:
            patched_fused_decode_mps.has_printed = True
            H_q, D = Q.shape
            scale = D ** -0.5
            q = Q.float()
            
            print("\n[Python CPU ATTN DEBUG - Layer 0 Step 0]")
            print(f"  D: {D}, rank: {pool.U.shape[2]}, K: {block_indices.shape[0]}, scale: {scale}")
            print("  Q[head 0] (first 10):", " ".join(f"{v:.6f}" for v in q[0, :10].tolist()))
            print("  slots (first 5):", block_indices[:5].tolist())
            
            print("\n[Python Pool Slots Metadata]")
            for s in range(pool.U.shape[0]):
                slen = pool.seq_lens[s].item()
                if slen > 0:
                    anc_pos = anchor_indices[s].item() if (anchor_indices is not None and s < len(anchor_indices)) else -1
                    print(f"  slot {s}: slen={slen} anchor_pos={anc_pos} scale_u={pool.U_scale[s].item():.6f} block_scale={pool.scales[s].item():.6f}")

            idx = block_indices[0].item()
            U_a = pool.U[idx].float() * pool.U_scale[idx].float()
            AncK_a = pool.anchors_K[idx].float()
            VK_a = pool.V_K[idx].float()
            
            if anchor_indices is not None and cos is not None and sin is not None:
                cos_flat = cos.squeeze(0) if cos.dim() == 3 else cos
                sin_flat = sin.squeeze(0) if sin.dim() == 3 else sin
                anchor_pos = anchor_indices[0].item()
                cos_anc_2d = cos_flat[anchor_pos].to(device=AncK_a.device, dtype=AncK_a.dtype).unsqueeze(0)
                sin_anc_2d = sin_flat[anchor_pos].to(device=AncK_a.device, dtype=AncK_a.dtype).unsqueeze(0)
                cos_anc = cos_flat[anchor_pos].to(device=VK_a.device, dtype=VK_a.dtype).unsqueeze(0).unsqueeze(1)
                sin_anc = sin_flat[anchor_pos].to(device=VK_a.device, dtype=VK_a.dtype).unsqueeze(0).unsqueeze(1)
                
                def rotate_half(x):
                    x1 = x[..., :x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2:]
                    return torch.cat((-x2, x1), dim=-1)
                    
                VK_a = VK_a * cos_anc + rotate_half(VK_a) * sin_anc
                AncK_a = AncK_a * cos_anc_2d + rotate_half(AncK_a) * sin_anc_2d
                
            print("  [Block 0, Head 0]")
            print(f"    slot_id: {idx}, slen: {pool.seq_lens[idx].item()}, scale_u: {pool.U_scale[idx].item()}, block_scale: {pool.scales[idx].item()}, anchor_pos: {anchor_indices[0].item()}")
            
            gpk = num_key_value_groups
            AncK_e = AncK_a.repeat_interleave(gpk, dim=0)
            VK_e = VK_a.repeat_interleave(gpk, dim=1).permute(1, 0, 2).contiguous()
            
            print("    raw_ak (first 10):", " ".join(f"{v:.6f}" for v in pool.anchors_K[idx, 0, :10].tolist()))
            
            s_anc = (q[0] * AncK_e[0]).sum() * scale
            print("    score_anc:", s_anc.item())
            
            q_proj_n = (q[0].unsqueeze(0) * VK_e[0]).sum(dim=-1) * scale
            print("    q_proj (first 10):", " ".join(f"{v:.6f}" for v in q_proj_n[:10].tolist()))
            
            delta_s = torch.einsum('r,sr->s', q_proj_n, U_a) * pool.scales[idx].item()
            token_scores = delta_s + s_anc
            print("    token_scores (first 10):", " ".join(f"{v:.6f}" for v in token_scores[:10].tolist()))
            print("    lse_sparse[0]:", lse[0].item())
            print("    out_sparse[0] (first 10):", " ".join(f"{v:.6f}" for v in out[0, :10].tolist()))
            
        return out, lse

    patched_fused_decode_mps.has_printed = False
    triton_fused_decode.fused_decode_mps = patched_fused_decode_mps
    import runtime.diffkv_attention as diffkv_attention
    diffkv_attention.fused_decode_mps = patched_fused_decode_mps
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    with open("/Users/omchimurkar1/Desktop/Differential-KV/scratch/long_prompt.txt", "r") as f:
        prompt_content = f.read()
        
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + prompt_content + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("Loading PyTorch DiffKV wrapper...")
    wrapper = PyTorchDiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 63}, # micro_block_size 63 to match C++
        device=device,
    )
    
    # Monkey patch to define L_dense in diffkv_attention module
    orig_assemble = wrapper.manager.assemble_dense_window_kv
    def patched_assemble(*args, **kwargs):
        k_assembled, v_assembled = orig_assemble(*args, **kwargs)
        import runtime.diffkv_attention as diffkv_attention
        if k_assembled is not None:
            diffkv_attention.L_dense = k_assembled.shape[2]
        else:
            diffkv_attention.L_dense = 0
        return k_assembled, v_assembled
    wrapper.manager.assemble_dense_window_kv = patched_assemble

    session_id = "default"
    inputs = wrapper.tokenizer(prompt, return_tensors='pt').to(wrapper.device)
    prompt_ids = inputs.input_ids[0].tolist()
    
    wrapper.manager.clear_session(session_id)
    wrapper._session_token_ids = {session_id: []}
    
    input_ids = torch.tensor([prompt_ids], device=wrapper.device)
    prefill_len = input_ids.shape[1]
    
    wrapper.manager.init_session(session_id, prefill_len=prefill_len)
    wrapper.manager.register_prefill_tokens(session_id, torch.tensor(prompt_ids, dtype=torch.long))
    wrapper.model._diffkv_session_ids = [session_id]
    
    print(f"Prefill length: {prefill_len}")
    
    # Run prefill chunk loop
    PREFILL_CHUNK = 512
    total_new = len(prompt_ids)
    outputs = None
    
    _prefill_buf = torch.zeros((1, PREFILL_CHUNK), dtype=torch.long)
    _pos_buf     = torch.zeros((1, PREFILL_CHUNK), dtype=torch.long)
    
    for chunk_idx, chunk_start in enumerate(range(0, total_new, PREFILL_CHUNK)):
        chunk_end = min(chunk_start + PREFILL_CHUNK, total_new)
        chunk = prompt_ids[chunk_start:chunk_end]
        clen = chunk_end - chunk_start
        
        _prefill_buf[0, :clen] = torch.as_tensor(chunk, dtype=torch.long)
        chunk_tensor = _prefill_buf[:, :clen].to(wrapper.device)
        _pos_buf[0, :clen] = torch.arange(chunk_start, chunk_start + clen, dtype=torch.long)
        pos_tensor = _pos_buf[:, :clen].to(wrapper.device)
        
        wrapper.manager.finalize_compressed_blocks()
        
        with torch.no_grad():
            outputs = wrapper.model(
                input_ids=chunk_tensor,
                position_ids=pos_tensor,
                use_cache=True,
            )
            
        wrapper.manager.compress_prefill_kv(session_id)
        
    del _prefill_buf, _pos_buf
    
    wrapper.manager.compress_deferred_prefill_blocks(session_id)
    
    # Wait for SVD
    import time
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        wrapper.manager.finalize_compressed_blocks()
        if getattr(wrapper.manager, "_pending_cpu_blocks", 0) <= 0:
            break
        time.sleep(0.01)
        
    wrapper.manager.finalize_srl_index(session_id, cached_len=0)
    
    logits = outputs.logits[:, -1, :] # [1, vocab]
    cur_pos = prefill_len
    
    # Print Prefill logits top 5
    val, idx = torch.topk(logits[0], 5)
    print("\n[Prefill Phase Top predictions]:")
    for k in range(5):
        tok_id = idx[k].item()
        tok_piece = wrapper.tokenizer.decode([tok_id])
        print(f"  {k}: \"{tok_piece}\" (id: {tok_id}, logit: {val[k].item():.4f})")
        
    generated = []
    # Greedy first token (no repetition penalty applied to select it, just like C++)
    first_decode_token = idx[0].item() 
    last_token = first_decode_token
    generated.append(last_token)
    
    max_new_tokens = 13
    repetition_penalty = 1.15
    
    # Pre-allocate position cache
    max_total_len = cur_pos + max_new_tokens + 10
    pos_cache = torch.arange(max_total_len, dtype=torch.long, device=wrapper.device)
    
    for step in range(max_new_tokens):
        pos_tensor = pos_cache[cur_pos].view(1, 1)
        input_ids = torch.tensor([[last_token]], device=wrapper.device)
        
        wrapper.manager.finalize_compressed_blocks()
        
        with torch.no_grad():
            outputs = wrapper.model(
                input_ids=input_ids,
                position_ids=pos_tensor,
                use_cache=True,
            )
            
        logits = outputs.logits[:, -1, :]
        cur_pos += 1
        
        # Apply repetition penalty to logits
        step_logits = logits.clone()
        for tok_id in set(generated):
            if tok_id < step_logits.shape[-1]:
                # Skip non-alphanumeric tokens like punctuation/newlines
                is_alnum = wrapper._alphanumeric_tokens.get(tok_id)
                if is_alnum is None:
                    tok_text = wrapper.tokenizer.decode([tok_id], skip_special_tokens=True)
                    is_alnum = any(c.isalnum() for c in tok_text)
                    wrapper._alphanumeric_tokens[tok_id] = is_alnum
                if not is_alnum:
                    continue
                if step_logits[0, tok_id] > 0:
                    step_logits[0, tok_id] /= repetition_penalty
                else:
                    step_logits[0, tok_id] *= repetition_penalty
                    
        # Print top 5 for current step
        val, idx = torch.topk(step_logits[0], 5)
        print(f"\n[Step {step} Top predictions]:")
        for k in range(5):
            tok_id = idx[k].item()
            tok_piece = wrapper.tokenizer.decode([tok_id])
            print(f"  {k}: \"{tok_piece}\" (id: {tok_id}, logit: {val[k].item():.4f})")
            
        next_id = idx[0].item()
        
        # In modern Qwen models, check if EOS
        if next_id in wrapper.stop_token_ids:
            print(" [EOS]")
            break
            
        generated.append(next_id)
        last_token = next_id
        
    wrapper.stop()

if __name__ == "__main__":
    main()
