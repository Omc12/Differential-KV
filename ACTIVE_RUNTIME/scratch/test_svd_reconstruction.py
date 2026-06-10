import os
import sys
os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from native_core.compression.lowrank import compress_lowrank

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32},
        device=device,
    )

    LARGE_PROMPT_PAPER = """
Abstract
While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) it is time to move beyond that and to address the sustainability of developing and using AI systems. In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI (sustainable development, training, and use of AI). The focus of this paper is on the latter.
In particular, I argue that the current trajectory of AI development and use (characterized by massive deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, greenhouse gas emissions from data centers during training and inference, and the social inequalities perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. These include energy-efficient hardware, green software engineering, open data and models, and robust governance frameworks that incorporate environmental impact assessments.
"""
    long_abstract = "\n".join([f"Section {i+1}:\n{LARGE_PROMPT_PAPER}" for i in range(10)])
    prompt = f"<|im_start|>user\nHere is a long research text:\n{long_abstract}\n\nBased on the text above, summarize the key points of Sustainable AI in 3 bullet points.<|im_end|>\n<|im_start|>assistant\n"
    
    encoded = wrapper.tokenizer(prompt, return_tensors="pt").to(device)
    
    session_id = "default"
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=encoded.input_ids.shape[1])
    wrapper.model._diffkv_session_ids = [session_id]
    
    with torch.no_grad():
        wrapper.model(**encoded, use_cache=True)
        # We catch the captured KV before compression to run SVD manually
        cap_dict = getattr(wrapper.manager, "_prefill_kv_capture", {})
        K_raw, V_raw = cap_dict[session_id][5] # layer 5
        
        # Let's compress layer 5 prefill KV manually block-by-block
        # We will look at Block 45 (anchor_idx = 2182)
        # In ingest_chunk, the block size is 49 (1 anchor + 48 active)
        block_idx = 45
        anchor_idx = 2182
        block_size = 49 # 1 + 48
        
        k_block = K_raw[0, :, anchor_idx : anchor_idx + block_size].float() # [heads, 49, dim]
        v_block = V_raw[0, :, anchor_idx : anchor_idx + block_size].float() # [heads, 49, dim]
        
        # SVD delta extraction
        anchor_k = k_block[:, 0] # [heads, dim]
        anchor_v = v_block[:, 0]
        
        # deltas (active tokens 1 to 48)
        k_active = k_block[:, 1:] # [heads, 48, dim]
        v_active = v_block[:, 1:]
        
        heads = k_block.shape[0]
        head_dim = k_block.shape[2]
        feat_dim = 2 * heads * head_dim
        
        stacked = torch.stack([k_active.transpose(0, 1), v_active.transpose(0, 1)], dim=1) # [48, 2, heads, dim]
        flat_tokens = stacked.reshape(48, feat_dim)
        anchor_flat = torch.stack([anchor_k, anchor_v], dim=0).reshape(-1)
        
        deltas = flat_tokens - anchor_flat.unsqueeze(0)
        
        print(f"Original Deltas shape: {deltas.shape}")
        print(f"Original Deltas norm : {deltas.norm().item():.4f}")
        
        # Token-wise Norm-Normalization
        token_norms = deltas.norm(dim=1)
        token_norms = torch.clamp(token_norms, min=1e-5)
        normalized_deltas = deltas / token_norms.unsqueeze(1)
        
        # Run SVD
        rank = 32
        lr = compress_lowrank(normalized_deltas, rank)
        
        print(f"Dynamic rank: {lr.dynamic_rank}")
        print(f"Singular values scale: {lr.scale:.6f}")
        
        # Reconstruct normalized deltas
        recon_norm_deltas = (lr.U @ lr.V) * lr.scale
        
        # Denormalize
        recon_deltas = recon_norm_deltas * token_norms.unsqueeze(1)
        
        # Errors
        diff = (deltas - recon_deltas).abs()
        rel_error = (deltas - recon_deltas).norm() / deltas.norm()
        print(f"Reconstruction error:")
        print(f"  Max absolute difference: {diff.max().item():.6f}")
        print(f"  Mean absolute difference: {diff.mean().item():.6f}")
        print(f"  Relative error         : {rel_error.item():.6f}")
        
        # Let's inspect token at pos 2199 (which is active index 17)
        print("\nActive Token Index 17 (Pos 2199) detail:")
        print(f"  Original delta norm: {deltas[17].norm().item():.4f}")
        print(f"  Recon delta norm   : {recon_deltas[17].norm().item():.4f}")
        print(f"  Token norm         : {token_norms[17].item():.4f}")
        print(f"  U row slice        : {lr.U[17].tolist()[:8]}...")
        
    wrapper.stop()

if __name__ == "__main__":
    main()
