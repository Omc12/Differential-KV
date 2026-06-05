import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    # x: [seq_len, dim] or [heads, seq_len, dim]
    # cos: [seq_len, dim]
    # sin: [seq_len, dim]
    if x.dim() == 3:
        # [heads, seq_len, dim]
        cos_sliced = cos.unsqueeze(0) # [1, seq_len, dim]
        sin_sliced = sin.unsqueeze(0)
    else:
        cos_sliced = cos
        sin_sliced = sin
    return x * cos_sliced + rotate_half(x) * sin_sliced

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    LARGE_PROMPT_PAPER = """
Abstract
While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) it is time to move beyond that and to address the sustainability of developing and using AI systems. In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI (sustainable development, training, and use of AI). The focus of this paper is on the latter.
In particular, I argue that the current trajectory of AI development and use (characterized by massive deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, greenhouse gas emissions from data centers during training and inference, and the social inequalities perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. These include energy-efficient hardware, green software engineering, open data and models, and robust governance frameworks that incorporate environmental impact assessments.
"""
    long_abstract = "\n".join([f"Section {i+1}:\n{LARGE_PROMPT_PAPER}" for i in range(10)])
    prompt = f"<|im_start|>user\nHere is a long research text:\n{long_abstract}\n\nBased on the text above, summarize the key points of Sustainable AI in 3 bullet points.<|im_end|>\n<|im_start|>assistant\n"
    
    print("1. Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model_baseline = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
    )
    model_baseline.eval()
    
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = encoded.input_ids[0].tolist()
    prefill_len = len(prompt_ids)
    
    print(f"Prefill length: {prefill_len} tokens")
    
    with torch.no_grad():
        outputs_baseline = model_baseline(input_ids=encoded.input_ids, use_cache=True)
    past_b = outputs_baseline.logits # dummy
    past_kv_b = outputs_baseline.past_key_values
    
    # Run 1 decode step on baseline
    pos_tensor = torch.tensor([[prefill_len]], dtype=torch.long, device=device)
    input_ids = torch.tensor([[16]], dtype=torch.long, device=device) # token '1'
    with torch.no_grad():
        outputs_b = model_baseline(input_ids=input_ids, position_ids=pos_tensor, past_key_values=past_kv_b, use_cache=True)
        past_kv_b = outputs_b.past_key_values
        
    print("\n2. Loading patched model with current active runtime configuration...")
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32},
        device=device,
    )
    
    session_id = "default"
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=prefill_len)
    wrapper.model._diffkv_session_ids = [session_id]
    
    with torch.no_grad():
        outputs_patched = wrapper.model(input_ids=encoded.input_ids, use_cache=True)
        wrapper.manager.compress_prefill_kv(session_id)
        
    # Wait for SVD background compression thread to finish
    import time
    for _ in range(100):
        with wrapper.manager._pending_lock:
            pending = wrapper.manager._pending_cpu_blocks
        # Also check if any block is in SUBMITTED state
        blocks = wrapper.manager.get_streaming_blocks(session_id, 0)
        submitted = sum(1 for b in blocks if getattr(b, "state", "") == "SUBMITTED")
        if pending == 0 and submitted == 0:
            break
        time.sleep(0.1)
        
    # Finalize (GPU upload)
    wrapper.manager.finalize_compressed_blocks()
        
    # Run 1 decode step on patched
    with torch.no_grad():
        outputs_p = wrapper.model(input_ids=input_ids, position_ids=pos_tensor, use_cache=True)
        
    print("\n3. Comparing Key Cache directly at Layer 5...")
    layer_idx = 5
    
    # Baseline rotated keys: shape [1, num_kv_heads, total_seq_len, head_dim]
    K_b = past_kv_b.layers[layer_idx].keys.float()
    print(f"Baseline keys shape: {K_b.shape}")
    
    # Reconstruct keys from Patched runtime
    # We will fetch the blocks for this layer from the manager
    blocks = wrapper.manager.get_streaming_blocks(session_id, layer_idx)
    print(f"Patched manager has {len(blocks)} blocks for Layer {layer_idx}")
    
    # We obtain the rotary embeddings for all sequence positions
    hist_pos = torch.arange(prefill_len + 1, device=device, dtype=torch.long).unsqueeze(0)
    cos_all, sin_all = wrapper.model.model.rotary_emb(K_b, hist_pos)
    cos_flat = cos_all.squeeze(0).float() # [seq_len, head_dim]
    sin_flat = sin_all.squeeze(0).float()
    
    reconstructed_k_parts = []
    pool = wrapper.manager.native_pool
    
    for idx, b in enumerate(blocks):
        # Determine anchor and active tokens
        anchor_idx = b.anchor_idx
        
        # 1. Reconstruct unrotated block keys
        if b.state == "COMPRESSED":
            pool_idx = b.pool_idx
            # seq_len is the number of active tokens in the block
            seq_len = b.token_count()
            block_len = 1 + seq_len
            
            U = pool.U[pool_idx, :seq_len, :wrapper.manager.rank].float() * pool.U_scale[pool_idx].item() # [seq_len, rank]
            V_K = pool.V_K[pool_idx, :wrapper.manager.rank].float() # [rank, heads, head_dim]
            anchor_k = pool.anchors_K[pool_idx].float() # [heads, head_dim]
            
            # deltas: [seq_len, heads, head_dim]
            deltas = torch.einsum('sr,rhd->shd', U, V_K) * pool.scales[pool_idx].item()
            
            # unrotated block keys: [1 + seq_len, heads, head_dim]
            zeros = torch.zeros((1, wrapper.manager.kv_heads, wrapper.manager.head_dim), device=device)
            deltas_full = torch.cat([zeros, deltas], dim=0)
            k_unrot = anchor_k.unsqueeze(0) + deltas_full
            
            # Apply RoPE
            # Positions are [anchor_idx, anchor_idx + 1, ...]
            block_len = k_unrot.shape[0]
            positions = list(range(anchor_idx, anchor_idx + block_len))
            cos_b = cos_flat[positions].unsqueeze(1) # [block_len, 1, D]
            sin_b = sin_flat[positions].unsqueeze(1)
            
            # Apply to keys permuted to [block_len, heads, head_dim]
            k_rot = k_unrot * cos_b + rotate_half(k_unrot) * sin_b
            
            # permute back to [1, heads, block_len, head_dim]
            reconstructed_k_parts.append(k_rot.permute(1, 0, 2).unsqueeze(0))
            
        else:
            # Dense block
            anchor_k = b.anchor_kv[0, 0].float()
            dense_parts = [anchor_k.unsqueeze(1)]
            if b.active_k is not None:
                dense_parts.append(b.active_k[0].float())
            elif getattr(b, "active_k_cpu", None) is not None:
                dense_parts.append(b.active_k_cpu[0].to(device).float())
            
            k_unrot = torch.cat(dense_parts, dim=1) # [heads, block_len, head_dim]
            
            # Positions
            block_len = k_unrot.shape[1]
            positions = list(range(anchor_idx, anchor_idx + block_len))
            cos_b = cos_flat[positions].unsqueeze(0).unsqueeze(1) # [1, 1, block_len, D]
            sin_b = sin_flat[positions].unsqueeze(0).unsqueeze(1)
            
            # Apply RoPE to [1, heads, block_len, head_dim]
            k_unrot_4d = k_unrot.unsqueeze(0)
            k_rot = k_unrot_4d * cos_b + rotate_half(k_unrot_4d) * sin_b
            
            reconstructed_k_parts.append(k_rot)

    K_patched_recon = torch.cat(reconstructed_k_parts, dim=2)
    print(f"Reconstructed patched keys shape: {K_patched_recon.shape}")
    
    # Check shape and compare
    min_len = min(K_b.shape[2], K_patched_recon.shape[2])
    print(f"Comparing first {min_len} positions...")
    
    diff = (K_b[:, :, :min_len] - K_patched_recon[:, :, :min_len]).abs()
    print(f"Key discrepancy:")
    print(f"  Max difference  : {diff.max().item():.6f}")
    print(f"  Mean difference : {diff.mean().item():.6f}")
    
    # Print states of all blocks
    print("\nBlock States:")
    for idx, b in enumerate(blocks):
        print(f"  Block {idx:02d}: anchor_idx={b.anchor_idx:<4} state={b.state:<12} is_outlier={getattr(b, 'is_outlier', False)}")

    # Find top 5 largest discrepancies and their positions
    diff_flat = diff[0].max(dim=0)[0].max(dim=-1)[0] # [min_len]
    val, idx = torch.topk(diff_flat, k=5)
    print("\nTop 5 Largest Key Discrepancies:")
    for i in range(5):
        pos = idx[i].item()
        error = val[i].item()
        # Find which block contains this position
        for blk_idx, b in enumerate(blocks):
            block_len = b.token_count()
            if b.anchor_idx <= pos < b.anchor_idx + block_len:
                print(f"  Pos {pos:04d}: error={error:.6f} in Block {blk_idx:02d} (state={b.state}, is_outlier={getattr(b, 'is_outlier', False)})")
                break

    # Focus on Pos 2199
    target_pos = 2199
    # Invert RoPE for baseline key at target_pos to get unrotated key
    # K_b shape is [1, num_kv_heads, total_seq_len, head_dim]
    K_rot_b_target = K_b[0, :, target_pos, :] # [heads, D]
    cos_t = cos_flat[target_pos].unsqueeze(0) # [1, D]
    sin_t = sin_flat[target_pos].unsqueeze(0)
    # Inverse RoPE: K_unrot = K_rot * cos - rotate_half(K_rot) * sin
    K_unrot_b_target = K_rot_b_target * cos_t - rotate_half(K_rot_b_target) * sin_t

    # Find the block containing target_pos in patched model
    target_blk_idx = -1
    for idx, b in enumerate(blocks):
        block_len = b.token_count()
        if b.anchor_idx <= target_pos < b.anchor_idx + block_len:
            target_blk_idx = idx
            target_blk = b
            break

    print(f"\nDetailed Analysis at Pos {target_pos} (Block {target_blk_idx}):")
    if target_blk.state == "COMPRESSED":
        local_pos = target_pos - target_blk.anchor_idx
        pool_idx = target_blk.pool_idx
        # Reconstruct unrotated key
        U_val = pool.U[pool_idx, local_pos - 1, :wrapper.manager.rank].float() * pool.U_scale[pool_idx].item() # [rank]
        V_K = pool.V_K[pool_idx, :wrapper.manager.rank].float() # [rank, heads, head_dim]
        anchor_k = pool.anchors_K[pool_idx].float() # [heads, head_dim]
        # delta = U @ V * scale
        delta = torch.einsum('r,rhd->hd', U_val, V_K) * pool.scales[pool_idx].item()
        K_unrot_p_target = anchor_k + delta
        
        # Rotated reconstructed key
        K_rot_p_target = K_patched_recon[0, :, target_pos, :]

        # SVD Reconstruction Error on unrotated keys
        unrot_diff = (K_unrot_b_target - K_unrot_p_target).abs()
        rot_diff = (K_rot_b_target - K_rot_p_target).abs()
        print(f"  Unrotated key absolute diff: max={unrot_diff.max().item():.6f}, mean={unrot_diff.mean().item():.6f}")
        print(f"  Rotated key absolute diff  : max={rot_diff.max().item():.6f}, mean={rot_diff.mean().item():.6f}")
        print(f"  Block scale: {target_blk.scale:.6f}, dynamic_rank: {target_blk.dynamic_rank}")
        print(f"  Pool U block shape: {pool.U[pool_idx].shape}")
        print(f"  Pool U block sum: {pool.U[pool_idx].sum().item():.6f}")
        print(f"  Pool U block max abs: {pool.U[pool_idx].abs().max().item():.6f}")
        print(f"  Pool U block non-zero count: {(pool.U[pool_idx].abs() > 1e-5).sum().item()}")
        print(f"  U_val row values: {U_val.tolist()}")
        print(f"  V_K[0] norm: {V_K[0].norm().item():.6f}")
        print(f"  anchor_k norm: {anchor_k.norm().item():.6f}")
        print(f"  delta norm: {delta.norm().item():.6f}")
        print(f"  Baseline unrotated key first 5 elements: {K_unrot_b_target[0, :5].tolist()}")
        print(f"  anchor_k first 5 elements              : {anchor_k[0, :5].tolist()}")
        print(f"  delta first 5 elements                 : {delta[0, :5].tolist()}")
        print(f"  Patched unrotated key first 5 elements : {K_unrot_p_target[0, :5].tolist()}")
        print(f"  Baseline unrotated key norm: {K_unrot_b_target.norm().item():.4f}")
        print(f"  Patched unrotated key norm : {K_unrot_p_target.norm().item():.4f}")
    else:
        print("  Target block is not compressed!")
 
    wrapper.stop()

if __name__ == "__main__":
    main()
