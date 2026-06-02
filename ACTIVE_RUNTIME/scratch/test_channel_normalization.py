import os
import sys
import torch
import math

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
    
    print("Ingesting prompt...")
    encoded = wrapper.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    
    with torch.no_grad():
        _ = wrapper.model(input_ids=input_ids, use_cache=True)
        
    sid = "default"
    cap_dict = getattr(wrapper.manager, "_prefill_kv_capture", {})
    k_captured, v_captured = cap_dict[sid][8] # Layer 8
    
    # We will test Block SVD reconstruction errors for different sizes starting at 520
    anchor_idx = 520
    k_captured_cpu = k_captured[0].cpu().float()
    v_captured_cpu = v_captured[0].cpu().float()
    
    num_heads = k_captured_cpu.shape[0]
    head_dim = k_captured_cpu.shape[2]
    feat_dim = 2 * num_heads * head_dim
    
    sizes = [16, 32, 64]
    
    for size in sizes:
        k_block = k_captured_cpu[:, anchor_idx+1 : anchor_idx+1+size]
        v_block = v_captured_cpu[:, anchor_idx+1 : anchor_idx+1+size]
        anchor_k = k_captured_cpu[:, anchor_idx]
        anchor_v = v_captured_cpu[:, anchor_idx]
        
        stacked = torch.stack([k_block.transpose(0, 1), v_block.transpose(0, 1)], dim=1) # [size, 2, heads, dim]
        flat_tokens = stacked.reshape(size, feat_dim)
        anchor_flat = torch.stack([anchor_k, anchor_v], dim=0).reshape(-1)
        deltas = flat_tokens - anchor_flat.unsqueeze(0)
        
        # Norm-Normalized SVD
        channel_norms = deltas.norm(dim=0)
        channel_norms = torch.clamp(channel_norms, min=1e-5)
        normalized_deltas = deltas / channel_norms.unsqueeze(0)
        
        # Use rank = 16 for size=16, rank = 32 for others
        rank = min(32, size)
        lr = compress_lowrank(normalized_deltas, rank)
        recon = (lr.U @ lr.V) * lr.scale * channel_norms.unsqueeze(0)
        
        err = (recon - deltas).norm() / deltas.norm()
        
        err_channels = []
        for c in range(feat_dim):
            norm_val = deltas[:, c].norm().item()
            if norm_val > 1e-4:
                err_c = (recon[:, c] - deltas[:, c]).norm().item() / norm_val
                err_channels.append(err_c)
                
        print(f"\nSize {size} (Rank {rank}):")
        print(f"  Overall Frobenius Norm Error: {err.item():.6f}")
        print(f"  Channel-wise Relative Error - Mean: {sum(err_channels)/len(err_channels):.6f}, Max: {max(err_channels):.6f}")
        
    wrapper.stop()

if __name__ == "__main__":
    main()
