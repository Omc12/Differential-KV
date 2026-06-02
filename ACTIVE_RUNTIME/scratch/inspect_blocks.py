import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

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
    
    print("Generating a few tokens with wrapper...")
    response = wrapper.generate(prompt, max_new_tokens=5, temperature=0.0)
    
    print("\nDeep Analysis of Layer 8 Block 8:")
    manager = wrapper.manager
    blocks = manager.get_streaming_blocks("default", 8)
    b = blocks[8]
    print(f"Block 8: anchor_idx={b.anchor_idx} state={b.state} scale={b.scale:.4f}")
    
    # Let's print anchor_kv min/max/mean
    anchor_k = b.anchor_kv[0, 0]
    anchor_v = b.anchor_kv[0, 1]
    print(f"  Anchor Key: min={anchor_k.min().item():.4f} max={anchor_k.max().item():.4f} mean={anchor_k.mean().item():.4f}")
    print(f"  Anchor Value: min={anchor_v.min().item():.4f} max={anchor_v.max().item():.4f} mean={anchor_v.mean().item():.4f}")
    
    # Since it is compressed, let's reconstruct the deltas
    recon_deltas = (b.U.float() @ b.V.float()) * b.scale
    print(f"  Reconstructed Deltas: min={recon_deltas.min().item():.4f} max={recon_deltas.max().item():.4f} mean={recon_deltas.mean().item():.4f}")
    
    # Print reconstruction error
    # Let's get the original uncompressed block from the capture dictionary if we can
    cap_dict = getattr(manager, "_prefill_kv_capture", {})
    # Wait, the capture dict was popped in compress_prefill_kv.
    # So we don't have it anymore. But we can see the range of reconstructed keys
    feat_dim = 2 * anchor_k.shape[0] * anchor_k.shape[1]
    recon_flat = recon_deltas + b.anchor_kv.cpu().reshape(-1).unsqueeze(0)
    recon_stacked = recon_flat.reshape(recon_flat.shape[0], 2, anchor_k.shape[0], anchor_k.shape[1])
    recon_k = recon_stacked[:, 0]
    print(f"  Reconstructed Keys: min={recon_k.min().item():.4f} max={recon_k.max().item():.4f} mean={recon_k.mean().item():.4f}")
            
    wrapper.stop()

if __name__ == "__main__":
    main()
